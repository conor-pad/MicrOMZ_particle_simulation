# run_suite.py
import os
import sys
import torch
torch.set_default_dtype(torch.float32)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import time as _time

import config as cfg
import bcs
from biopar import BioPar
from physics import setup_physics
from loop import run_simulation

try:
    from progress import progress
except ImportError:
    from tqdm import tqdm as progress

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
warnings.filterwarnings('ignore')
logging.getLogger('torch').setLevel(logging.ERROR)


# ═════════════════════════════════════════════════════════════════════════════
# ── SWEEP MODE SWITCH ─────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
#
#   'poc_o2'     — Initial POC (mmol C m⁻³)   ×  Ambient O₂ (mmol m⁻³)
#   'o2_radius'  — Ambient O₂ (mmol m⁻³)      ×  Particle Radius (mm)
#   'radius_poc' — Particle Radius (mm)       ×  Initial POC (mmol C m⁻³)
#
SWEEP_MODE = 'o2_radius'


# ═════════════════════════════════════════════════════════════════════════════
# ── SWEEP AXIS DEFINITIONS ────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

N_POC      = 14
N_O2       = 14
N_RADIUS   = 14
POC_LEVELS    = np.linspace(50_000, 800_000, N_POC).tolist()   # mmol C m⁻³
O2_LEVELS     = np.linspace(1.0, 25.0, N_O2).tolist()          # mmol O₂ m⁻³
RADIUS_LEVELS = np.linspace(0.5, 1.0, N_RADIUS).tolist()       # mm

# ═════════════════════════════════════════════════════════════════════════════
# ── FIXED PARAMETERS ──────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
RADIUS_FIXED = 1.0     # mm  — fixed for poc_o2
NO3_FIXED    = 10.0    # mmol NO₃ m⁻³
O2_FIXED     = 6.0     # mmol O₂ m⁻³   — fixed for radius_poc
POC_FIXED    = 850_000 # mmol C m⁻³    — fixed for o2_radius
BIO_ACCEL    = 1.0
VMAX_MULTIPLIER = 300.0
INITIAL_AEROBIC = 8.0

# All functional groups with vmax_oxi/vmax_red params — VMAX_MULTIPLIER scales
# every one of these. 'zoo' is excluded — it has no vmax_oxi/red (uses zoo_umax).
BUG_PREFIXES = ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox']


# ═════════════════════════════════════════════════════════════════════════════
# ── SWEEP METADATA ────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def get_sweep_meta():
    """
    Returns a dict describing the active sweep so that run_experiment() and
    generate_all_plots() can adapt without mode-specific if/else trees.

    Keys
    ----
    axis1_col   : DataFrame column for x-axis (outer loop)
    axis2_col   : DataFrame column for y-axis (inner loop)
    axis1_vals  : list of values for axis 1
    axis2_vals  : list of values for axis 2
    axis1_label : human-readable x-axis label
    axis2_label : human-readable y-axis label
    csv_name    : output CSV filename
    chunk_size  : experiments per batch
    """
    if SWEEP_MODE == 'poc_o2':
        return dict(
            axis1_col   = 'Initial_POC_Density',
            axis2_col   = 'Ext_O2',
            axis1_vals  = POC_LEVELS,
            axis2_vals  = O2_LEVELS,
            axis1_label = 'Initial POC Density (mmol C m⁻³)',
            axis2_label = 'Ambient O₂ (mmol O₂ m⁻³)',
            csv_name    = 'outputs/MicrOMZ_POC_O2_Sweep.csv',
            chunk_size  = len(O2_LEVELS),
        )
    elif SWEEP_MODE == 'o2_radius':
        return dict(
            axis1_col   = 'Ext_O2',
            axis2_col   = 'Radius_mm',
            axis1_vals  = O2_LEVELS,
            axis2_vals  = RADIUS_LEVELS,
            axis1_label = 'Ambient O₂ (mmol O₂ m⁻³)',
            axis2_label = 'Particle Radius (mm)',
            csv_name    = 'outputs/MicrOMZ_O2_Radius_Sweep.csv',
            chunk_size  = len(O2_LEVELS),
        )
    elif SWEEP_MODE == 'radius_poc':
        return dict(
            axis1_col   = 'Radius_mm',
            axis2_col   = 'Initial_POC_Density',
            axis1_vals  = RADIUS_LEVELS,
            axis2_vals  = POC_LEVELS,
            axis1_label = 'Particle Radius (mm)',
            axis2_label = 'Initial POC Density (mmol C m⁻³)',
            csv_name    = 'outputs/MicrOMZ_Radius_POC_Sweep.csv',
            chunk_size  = len(POC_LEVELS),
        )
    else:
        raise ValueError(f"Unknown SWEEP_MODE: '{SWEEP_MODE}'. "
                         f"Choose 'poc_o2', 'o2_radius', or 'radius_poc'.")


# ═════════════════════════════════════════════════════════════════════════════
# ── EXPERIMENT RUNNER ─────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(axis1_vals, axis2_vals):
    """
    Runs one BATCHED chunk — all len(axis1_vals) experiments simultaneously
    as one bs>1 simulation, not one bs=1 run per entry.

    No extrapolation/early-exit — every member still runs the full
    Total_Time, same as before. This only removes the serial bs=1 loop.
    """
    bs = len(axis1_vals)

    # ── Resolve physical variable arrays ────────────────────────────────────
    if SWEEP_MODE == 'poc_o2':
        poc_arr  = np.array(axis1_vals, dtype=np.float32)
        o2_arr   = np.array(axis2_vals, dtype=np.float32)
        no3_arr  = np.array([NO3_FIXED]    * bs, dtype=np.float32)
        mort_arr = np.array([1.0]          * bs, dtype=np.float32)
        radii    = np.array([RADIUS_FIXED] * bs, dtype=np.float32)
    elif SWEEP_MODE == 'o2_radius':
        o2_arr   = np.array(axis1_vals, dtype=np.float32)
        radii    = np.array(axis2_vals, dtype=np.float32)
        poc_arr  = np.array([POC_FIXED]  * bs, dtype=np.float32)
        no3_arr  = np.array([NO3_FIXED]  * bs, dtype=np.float32)
        mort_arr = np.array([1.0]        * bs, dtype=np.float32)
    elif SWEEP_MODE == 'radius_poc':
        radii    = np.array(axis1_vals, dtype=np.float32)
        poc_arr  = np.array(axis2_vals, dtype=np.float32)
        o2_arr   = np.array([O2_FIXED]  * bs, dtype=np.float32)
        no3_arr  = np.array([NO3_FIXED] * bs, dtype=np.float32)
        mort_arr = np.array([1.0]       * bs, dtype=np.float32)

    # ── Patch config (array-valued — same pattern as NitrOMZ's run_suite) ──
    cfg.batch_size             = bs
    cfg.is_suite               = True
    cfg.terminal_snapshot_only = True
    cfg.vmax_multiplier        = VMAX_MULTIPLIER
    cfg.initial_aerobic_biomass = INITIAL_AEROBIC
    cfg.radius                 = radii
    cfg.U_bg                   = 1.6 * (cfg.radius / 1.0) ** 0.56
    cfg.Lx                     = 10.0 * cfg.radius
    cfg.Ly                     = 10.0 * cfg.radius
    cfg.cx                     = 5.0  * cfg.radius
    cfg.cy                     = cfg.Ly / 2.0
    cfg.dx                     = cfg.Lx / (cfg.Nx - 1)
    cfg.dy                     = cfg.Ly / (cfg.Ny - 1)
    cfg.K                      = cfg.nu / cfg.Sc_target
    # Total_Time must be a single scalar shared by the whole batch — take the
    # worst case across members (identical to the old value when radius is fixed).
    cfg.Total_Time             = 650 # float(np.max(25.0 * cfg.Lx / cfg.U_bg))
    cfg.BIO_ACCEL               = BIO_ACCEL
    cfg.MORT_AMP                = mort_arr

    # ── Patch boundary conditions (per-member arrays) ───────────────────────
    bcs.inflow.o2  = o2_arr.reshape(bs, 1)
    bcs.inflow.no3 = no3_arr.reshape(bs, 1)
    bcs.inflow.aer = INITIAL_AEROBIC   # scalar — applies uniformly across the batch

    # ── Silence per-run stdout ───────────────────────────────────────────────
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    device_, state = setup_physics(cfg)

    # setup_physics() just instantiated a fresh BioPar() with unscaled defaults
    # (~2.3e-5 s⁻¹) — apply VMAX_MULTIPLIER to every functional group here so
    # growth is actually fast enough to matter within Total_Time.
    bgc = state['bgc']
    for prefix in BUG_PREFIXES:
        setattr(bgc, f'{prefix}_vmax_oxi', getattr(bgc, f'{prefix}_vmax_oxi') * VMAX_MULTIPLIER)
        setattr(bgc, f'{prefix}_vmax_red', getattr(bgc, f'{prefix}_vmax_red') * VMAX_MULTIPLIER)

    # ── Save Batch Physics Metrics to Text File ─────────────────────────────
    with open('outputs/batch_physics_log.txt', 'a') as f:
        f.write(f"Radius: {float(radii[0]):.2f} mm\n"
                f"  dx:         {float(np.atleast_1d(cfg.dx)[0]):.4e} m\n"
                f"  dy:         {float(np.atleast_1d(cfg.dy)[0]):.4e} m\n"
                f"  U_bg:       {float(np.atleast_1d(cfg.U_bg)[0]):.4f} m/s\n"
                f"  Total_Time: {cfg.Total_Time:.1f} s\n"
                f"  K:          {float(np.atleast_1d(cfg.K)[0]):.4e} m²/s\n"
                f"  VMAX_MULT:  {VMAX_MULTIPLIER:.1f}×  (aer_vmax_oxi eff="
                f"{bgc.aer_vmax_oxi:.4e}/s)\n"
                f"  Init_Aer:   {INITIAL_AEROBIC:.2f} mmol C m⁻³\n"
                f"{'-'*30}\n")

    # Override POC density (set after setup_physics so it can read k_hyd)
    state['poc_initial'] = torch.tensor(
        poc_arr.reshape(bs, 1, 1), dtype=torch.float32, device=device_)

    _t0 = _time.perf_counter()
    results = run_simulation(state, cfg, device_)
    _elapsed = _time.perf_counter() - _t0

    sys.stdout.close()
    sys.stdout = original_stdout

    # ── Extract terminal-state fields (now [bs, Nx, Ny], not squeezed) ──────
    final_o2  = torch.tensor(results[0][-1],  device=device_)
    final_n2o = torch.tensor(results[1][-1],  device=device_)
    final_no3 = torch.tensor(results[2][-1],  device=device_)
    final_no2 = torch.tensor(results[3][-1],  device=device_)
    final_n2  = torch.tensor(results[4][-1],  device=device_)
    final_doc = torch.tensor(results[5][-1],  device=device_)
    final_nh4 = torch.tensor(results[6][-1],  device=device_)
    bio_snap    = results[13]
    growth_snap = results[14]   # kept in suite mode — per-cell specific growth rate (day⁻¹)

    particle_mask = state['particle_mask']   # [bs, Nx, Ny]

    # Per-member cell volume (dx/dy vary with radius in radius_* sweeps)
    dV_m3_np = np.atleast_1d(cfg.dx * cfg.dy * 1.0 * 1e-9).astype(np.float32).reshape(-1)
    if dV_m3_np.size == 1 and bs > 1:
        dV_m3_np = np.repeat(dV_m3_np, bs)
    dV_m3_t = torch.tensor(dV_m3_np, dtype=torch.float32, device=device_)   # [bs]

    final_tracers = {
        'o2': final_o2, 'no3': final_no3, 'no2': final_no2,
        'n2o': final_n2o, 'n2': final_n2, 'doc': final_doc,
        'nh4': final_nh4,
        'po4': torch.zeros_like(final_o2),
        'n2o_ammox': torch.zeros_like(final_o2),
        'n2o_denit': torch.zeros_like(final_o2),
    }
    for bname in state['bio_names']:
        final_tracers[bname] = torch.tensor(bio_snap[bname][-1], device=device_)

    # Growth-rate fields are computed on the interior (Nx-2, Ny-2) grid in
    # loop.py — build a matching interior particle mask to integrate them.
    particle_mask_interior = particle_mask[..., 1:-1, 1:-1]
    final_growth = {
        n: torch.tensor(growth_snap[n][-1], device=device_) for n in state['bio_names']
    }

    bgc = state['bgc']
    # Temporarily apply per-member mort_amp ([bs,1,1] tensor) for the terminal
    # SMS evaluation — computed once, vectorized over the whole batch.
    orig_m_l,  orig_m_q  = bgc.m_l, bgc.m_q
    orig_zm_l, orig_zm_q = bgc.zoo_m_l, bgc.zoo_m_q
    mort_amp_t = state['mort_amp']   # [bs,1,1]
    bgc.m_l = orig_m_l * mort_amp_t;  bgc.m_q = orig_m_q * mort_amp_t
    bgc.zoo_m_l = orig_zm_l * mort_amp_t; bgc.zoo_m_q = orig_zm_q * mort_amp_t
    from sms import microbial_sms_omz
    ddt, _ = microbial_sms_omz(final_tracers, bgc)
    bgc.m_l, bgc.m_q = orig_m_l, orig_m_q
    bgc.zoo_m_l, bgc.zoo_m_q = orig_zm_l, orig_zm_q

    def vol_int_b(field, mask=None):
        """Per-member volume integral. Returns a [bs] tensor."""
        f = field * mask if mask is not None else field
        return f.sum(dim=(-2, -1)) * dV_m3_t

    # N₂O budget (per member)
    n2o_net_domain_b = vol_int_b(ddt['n2o'])
    n2o_net_core_b   = vol_int_b(ddt['n2o'], particle_mask)
    n2_net_domain_b  = vol_int_b(ddt['n2'])

    # Anoxic core fraction (per member)
    oxic_threshold    = 0.3   # mmol O₂ m⁻³
    anoxic_vox_b      = vol_int_b((final_o2 < oxic_threshold).float(), particle_mask)
    total_core_vol_b  = vol_int_b(particle_mask)
    frac_anoxic_b     = torch.where(total_core_vol_b > 0,
                                     anoxic_vox_b / total_core_vol_b,
                                     torch.zeros_like(total_core_vol_b))

    # Biomass (per member, per functional group)
    bio_core_amounts_b = {
        n: vol_int_b(final_tracers[n], particle_mask) for n in state['bio_names']
    }
    total_bio_core_b = sum(bio_core_amounts_b.values())

    # Relative biomass growth factor (terminal / initial core biomass).
    # Bio tracers are seeded uniformly at their bcs.inflow concentration
    # everywhere inside particle_mask (physics.py tracer-init), so initial
    # total core biomass = (sum of all group seed concentrations) * core volume.
    # bcs.inflow.aer already reflects INITIAL_AEROBIC (patched above); the rest
    # are their dataclass defaults — read live so this never goes stale.
    initial_bio_density_total = sum(getattr(bcs.inflow, n) for n in state['bio_names'])
    bio_growth_factor_b = torch.where(
        total_core_vol_b > 0,
        total_bio_core_b / (initial_bio_density_total * total_core_vol_b),
        torch.zeros_like(total_core_vol_b))

    bio_names_list = state['bio_names']
    bio_stack       = torch.stack([bio_core_amounts_b[n] for n in bio_names_list])  # [n_bugs, bs]
    dom_idx_b       = bio_stack.argmax(dim=0)   # [bs]

    # Core-mean specific growth rate per functional group (day⁻¹), per member.
    # (Core-mean rather than volume-integrated — a rate, not an amount.)
    core_cells_b = particle_mask_interior.sum(dim=(-2, -1))   # [bs]
    growth_core_mean_b = {
        n: torch.where(core_cells_b > 0,
                        (final_growth[n] * particle_mask_interior).sum(dim=(-2, -1)) / core_cells_b,
                        torch.zeros_like(core_cells_b))
        for n in bio_names_list
    }

    # ── Assemble per-member metric dicts ────────────────────────────────────
    batch_metrics = []
    for b in range(bs):
        poc_val, o2_val = float(poc_arr[b]), float(o2_arr[b])
        no3_val, mort_val, radius_val = float(no3_arr[b]), float(mort_arr[b]), float(radii[b])
        n2o_net_core   = n2o_net_core_b[b].item()
        frac_anoxic    = frac_anoxic_b[b].item()
        dominant_bug   = bio_names_list[dom_idx_b[b].item()]

        metrics = {
            # ── Identity columns (always present) ──
            'Initial_POC_Density': poc_val,
            'Ext_O2':              o2_val,
            'Ext_NO3':             no3_val,
            'Mort_Amp':            mort_val,
            'Radius_mm':           radius_val,
            # ── N₂O budget ──
            'N2O_Net_Domain_SMS_mmol_s': n2o_net_domain_b[b].item(),
            'N2O_Net_Core_SMS_mmol_s':   n2o_net_core,
            'N2_Net_Domain_SMS_mmol_s':  n2_net_domain_b[b].item(),
            # ── Anoxia ──
            'Frac_Anoxic_Core': frac_anoxic,
            # ── Biomass ──
            'Total_Bio_Core_mmol':  total_bio_core_b[b].item(),
            'Bio_Growth_Factor':    bio_growth_factor_b[b].item(),
            'Dominant_Bug':         dominant_bug,
            # ── Individual functional group core biomass ──
            **{f'Bio_Core_{n}_mmol': bio_core_amounts_b[n][b].item() for n in bio_names_list},
            # ── Terminal core-mean specific growth rate per group (day⁻¹) ──
            **{f'GrowthRate_{n}_perday': growth_core_mean_b[n][b].item() for n in bio_names_list},
            # ── Metadata (whole-chunk time, shared across this batch) ──
            'Elapsed_s': _elapsed,
        }
        batch_metrics.append(metrics)

        print(f"  POC={poc_val:.0f}  O2={o2_val:.2f}  MORT={mort_val:.2f}  "
              f"anoxic={frac_anoxic:.2f}  N2O_core={n2o_net_core:.2e}  "
              f"dom={dominant_bug}", flush=True)

    print(f"  [chunk of {bs} finished in {_elapsed:.1f}s]", flush=True)
    return batch_metrics


# ═════════════════════════════════════════════════════════════════════════════
# ── PLOTTING ──────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_plots(csv_filename):
    df   = pd.read_csv(csv_filename)
    meta = get_sweep_meta()

    ax1_col   = meta['axis1_col']
    ax2_col   = meta['axis2_col']
    ax1_label = meta['axis1_label']
    ax2_label = meta['axis2_label']

    dead_zone = df['Frac_Anoxic_Core'] == 0.0
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    print(f'\n📊 Generating {SWEEP_MODE} sweep plots…')

    def plot_contour(values_col, title, cbar_label, filename, cmap='viridis'):
        plt.figure(figsize=(9, 6))
        pivot = df.pivot_table(index=ax2_col, columns=ax1_col,
                               values=values_col, dropna=False)
        X, Y = np.meshgrid(pivot.columns, pivot.index)
        Z    = pivot.values
        min_z, max_z = np.nanmin(Z), np.nanmax(Z)
        if min_z <= 0.0 < max_z:
            eps    = max_z * 1e-5
            levels = [0.0, eps] + list(np.linspace(eps, max_z, 20))[1:]
            cf     = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
            cl     = plt.contour(X, Y, Z, levels=[eps],
                                 colors='cyan', linewidths=2, linestyles='dashed')
            plt.clabel(cl, inline=True, fontsize=10, fmt='Zero')
        else:
            cf = plt.contourf(X, Y, Z, levels=20, cmap=cmap)
        cbar = plt.colorbar(cf); cbar.set_label(cbar_label)
        plt.scatter(X, Y, color='white', edgecolor='black', s=20, alpha=0.8, zorder=5)
        plt.xlabel(ax1_label); plt.ylabel(ax2_label)
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_3d_biomass_fractions(df, ax1_col, ax2_col, ax1_label, ax2_label):
        from mpl_toolkits.mplot3d import Axes3D
        
        bio_cols = [c for c in df.columns if c.startswith('Bio_Core_')]
        total_bio = df[bio_cols].sum(axis=1).replace(0, np.nan)
        
        frac_cols = []
        for col in bio_cols:
            fname = f'Frac_{col.split("_")[2]}'
            df[fname] = df[col] / total_bio
            frac_cols.append(fname)
            
        # Get the top 4 bugs by mean fraction across the entire sweep
        top_4 = df[frac_cols].mean().nlargest(4).index
        
        fig = plt.figure(figsize=(14, 10))
        for i, bug in enumerate(top_4, 1):
            ax = fig.add_subplot(2, 2, i, projection='3d')
            pivot = df.pivot_table(index=ax2_col, columns=ax1_col, values=bug)
            X, Y = np.meshgrid(pivot.columns, pivot.index)
            
            surf = ax.plot_surface(X, Y, pivot.values, cmap='viridis', alpha=0.9, edgecolor='k', linewidth=0.2)
            ax.set_xlabel(ax1_label, labelpad=10)
            ax.set_ylabel(ax2_label, labelpad=10)
            ax.set_zlabel('Biomass Fraction', labelpad=10)
            ax.set_title(f'{bug.replace("Frac_", "")} Dominance Surface', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/Plot_3D_Fractions.png', dpi=300)
        plt.close()

    unique_ax1 = np.sort(df[ax1_col].dropna().unique())
    unique_ax2 = np.sort(df[ax2_col].dropna().unique())
    median_ax1 = unique_ax1[len(unique_ax1) // 2]
    median_ax2 = unique_ax2[len(unique_ax2) // 2]

    bio_group_cols = [f'Bio_Core_{n}_mmol' for n in
                      ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox', 'zoo']]
    bio_labels     = ['Aer', 'NaR', 'NaI', 'NaO', 'NiR', 'NiO', 'NoS', 'AOA', 'NOB', 'AOX', 'Zoo']

    # ── Plot 1: N₂O net production rate, core ────────────────────────────────
    plot_contour(
        'N2O_Net_Core_SMS_mmol_s',
        f'Plot 1: Terminal Net N₂O SMS Rate Inside Particle Core\n'
        f'(mmol N₂O m⁻³ s⁻¹ integrated over core; + = source, − = sink)',
        'mmol N₂O s⁻¹', 'Plot1_N2O_Core_Rate.png', 'magma')

    # ── Plot 2: N₂O net production rate, full domain ──────────────────────────
    plot_contour(
        'N2O_Net_Domain_SMS_mmol_s',
        f'Plot 2: Terminal Net N₂O SMS Rate — Full Domain\n'
        f'(core + plume; mmol N₂O s⁻¹)',
        'mmol N₂O s⁻¹', 'Plot2_N2O_Domain_Rate.png', 'viridis')

    # ── Plot 3: Anoxic core fraction ──────────────────────────────────────────
    plot_contour(
        'Frac_Anoxic_Core',
        f'Plot 3: Anoxic Core Fraction at Terminal State\n'
        f'(fraction of core cells with O₂ < 0.3 mmol m⁻³)',
        'Fraction', 'Plot3_Anoxic_Core.png', 'inferno')

    # ── Plot 4: Biomass growth factor (relative to initial seed) ─────────────
    plot_contour(
        'Bio_Growth_Factor',
        f'Plot 4: Core Biomass Growth Factor (Terminal ÷ Initial)\n'
        f'(all functional groups combined; 1.0 = no net growth)',
        'Growth factor (×)', 'Plot4_Bio_Growth_Factor.png', 'cividis')

    # ── Plot 5: Biomass fractions vs axis 2 (at median axis 1) ───────────────
    df_p5 = df[df[ax1_col] == median_ax1].copy()
    total_bio = df_p5[bio_group_cols].sum(axis=1).replace(0, np.nan)
    frac_cols = []
    for col, lbl in zip(bio_group_cols, bio_labels):
        fcol = f'Frac_{lbl}'
        df_p5[fcol] = df_p5[col] / total_bio
        frac_cols.append(fcol)

    melted5 = df_p5.melt(id_vars=[ax2_col], value_vars=frac_cols,
                          var_name='Group', value_name='Fraction')
    melted5['Group'] = melted5['Group'].str.replace('Frac_', '')
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=melted5, x=ax2_col, y='Fraction', hue='Group',
                 marker='o', linewidth=2.5)
    plt.title(f'Plot 5: Core Biomass Fractions vs. {ax2_label.split("(")[0].strip()}\n'
              f'(fixed {ax1_label.split("(")[0].strip()} = {median_ax1:.4g})',
              fontweight='bold')
    plt.xlabel(ax2_label); plt.ylabel('Fraction of Total Core Biomass')
    plt.tight_layout()
    plt.savefig('outputs/Plot5_Bio_Fractions_vs_Axis2.png', dpi=300)
    plt.close()

    # ── Plot 6: Biomass fractions vs axis 1 (at median axis 2) ───────────────
    df_p6 = df[df[ax2_col] == median_ax2].copy()
    total_bio6 = df_p6[bio_group_cols].sum(axis=1).replace(0, np.nan)
    frac_cols6 = []
    for col, lbl in zip(bio_group_cols, bio_labels):
        fcol = f'Frac_{lbl}'
        df_p6[fcol] = df_p6[col] / total_bio6
        frac_cols6.append(fcol)

    melted6 = df_p6.melt(id_vars=[ax1_col], value_vars=frac_cols6,
                          var_name='Group', value_name='Fraction')
    melted6['Group'] = melted6['Group'].str.replace('Frac_', '')
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=melted6, x=ax1_col, y='Fraction', hue='Group',
                 marker='s', linewidth=2.5)
    plt.title(f'Plot 6: Core Biomass Fractions vs. {ax1_label.split("(")[0].strip()}\n'
              f'(fixed {ax2_label.split("(")[0].strip()} = {median_ax2:.4g})',
              fontweight='bold')
    plt.xlabel(ax1_label); plt.ylabel('Fraction of Total Core Biomass')
    plt.tight_layout()
    plt.savefig('outputs/Plot6_Bio_Fractions_vs_Axis1.png', dpi=300)
    plt.close()

    # ── Plot 7: Dominant functional group regime map ──────────────────────────
    from matplotlib.colors import ListedColormap
    bug_order = ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox', 'zoo']
    bug_labels_map = dict(zip(bug_order, bio_labels))
    df['Dom_Idx'] = df[bio_group_cols].values.argmax(axis=1)
    pivot7 = df.pivot_table(index=ax2_col, columns=ax1_col,
                             values='Dom_Idx', dropna=False)
    X7, Y7 = np.meshgrid(pivot7.columns, pivot7.index)
    n_bugs = len(bug_order)
    cmap7 = plt.cm.get_cmap('tab20', n_bugs)
    plt.figure(figsize=(11, 7))
    cf7 = plt.contourf(X7, Y7, pivot7.values,
                        levels=np.arange(-0.5, n_bugs + 0.5, 1.0), cmap=cmap7)
    cbar7 = plt.colorbar(cf7, ticks=range(n_bugs))
    cbar7.ax.set_yticklabels(bio_labels)
    cbar7.set_label('Dominant Functional Group (by core biomass)')
    plt.xlabel(ax1_label); plt.ylabel(ax2_label)
    plt.title('Plot 7: Dominant Core Functional Group Regime Map\n'
              '(argmax of terminal-state biomass per functional group, particle mask only)',
              fontweight='bold')

    # ── Overlay: anoxia-onset boundary (Frac_Anoxic_Core = 0 crossing) ────────
    # Same X7/Y7 grid, so this lines up cell-for-cell with the regime map above.
    pivot7_anoxic = df.pivot_table(index=ax2_col, columns=ax1_col,
                                    values='Frac_Anoxic_Core', dropna=False)
    Z7_anoxic = pivot7_anoxic.values
    min_z7, max_z7 = np.nanmin(Z7_anoxic), np.nanmax(Z7_anoxic)
    if min_z7 <= 0.0 < max_z7:
        eps7 = max_z7 * 1e-5
        cl7 = plt.contour(X7, Y7, Z7_anoxic, levels=[eps7],
                          colors='cyan', linewidths=2.5, linestyles='dashed')
        plt.clabel(cl7, inline=True, fontsize=10, fmt='Anoxia onset')

    plt.tight_layout()
    plt.savefig('outputs/Plot7_Dominant_Bug_Regime.png', dpi=300)
    plt.close()


    # ── Plot 8: N₂ net production rate ────────────────────────────────────────
    plot_contour(
        'N2_Net_Domain_SMS_mmol_s',
        f'Plot 8: Terminal Net N₂ SMS Rate — Full Domain\n'
        f'(mmol N₂ s⁻¹; proxy for completed denitrification)',
        'mmol N₂ s⁻¹', 'Plot8_N2_Domain_Rate.png', 'plasma')


    # ── Plot 9: 3D Biomass Fractions ──────────────────────────────────────────
    plot_3d_biomass_fractions(df, ax1_col, ax2_col, ax1_label, ax2_label)

    # ── Plot 10: Aerobic specific growth rate ──────────────────────────────────
    # Core-mean μ_aer (day⁻¹). Flat/saturated across the O2 axis means growth
    # isn't O2-limited over your swept range (aer_k_oxi = 0.2 is small next to
    # typical O2 sweep values) — a real gradient here means O2 is the limiter.
    plot_contour(
        'GrowthRate_aer_perday',
        f'Plot 10: Aerobic Heterotroph Specific Growth Rate\n'
        f'(core-mean μ, day⁻¹; flat = O2-saturated over this range)',
        'μ (day⁻¹)', 'Plot10_Aerobic_Growth_Rate.png', 'plasma')

    print('✅ All plots generated successfully!')


# ═════════════════════════════════════════════════════════════════════════════
# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def main():
    meta         = get_sweep_meta()
    ax1_col      = meta['axis1_col']
    ax2_col      = meta['axis2_col']
    ax1_vals     = meta['axis1_vals']
    ax2_vals     = meta['axis2_vals']
    csv_filename = meta['csv_name']
    chunk_size   = meta['chunk_size']

    n_total  = len(ax1_vals) * len(ax2_vals)
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    W = 62
    print(f"\n{'═'*W}")
    print(f"  MicrOMZ Suite  │  {_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*W}")
    print(f"  Mode         : {SWEEP_MODE}")
    print(f"  Grid         : {len(ax1_vals)} × {len(ax2_vals)} = {n_total} experiments")
    print(f"  Chunks       : {n_chunks}  (batch size {chunk_size}, SERIAL)")
    print(f"  Output       : {csv_filename}")
    print(f"{'─'*W}")
    print(f"  Axis 1 ({ax1_col}):")
    for v in ax1_vals:
        print(f"    {v:.4g}")
    print(f"  Axis 2 ({ax2_col}):")
    for v in ax2_vals:
        print(f"    {v:.4g}")
    print(f"{'═'*W}\n")

    os.makedirs('outputs', exist_ok=True)
    results_data = []
    run_configs  = [(a1, a2) for a2 in ax2_vals for a1 in ax1_vals]
    _suite_t0    = _time.perf_counter()

    with progress(total=n_total, desc='Overall sweep', unit='run') as pbar:
        for i in range(0, n_total, chunk_size):
            chunk_num = i // chunk_size + 1
            batch     = run_configs[i: i + chunk_size]
            ax1_batch = [b[0] for b in batch]
            ax2_batch = [b[1] for b in batch]

            print(f"\n{'─'*W}", flush=True)
            print(f"  Chunk {chunk_num}/{n_chunks}  │  runs {i+1}–{min(i+len(batch), n_total)} "
                  f"of {n_total}  │  {_time.strftime('%H:%M:%S')}", flush=True)
            print(f"{'─'*W}", flush=True)

            batched_results = run_experiment(ax1_batch, ax2_batch)
            results_data.extend(batched_results)

            # Save incrementally
            temp_df = pd.DataFrame(results_data)
            temp_df.to_csv(csv_filename, index=False)

            _elapsed = _time.perf_counter() - _suite_t0
            _done    = i + len(batch)
            _rate    = _done / _elapsed if _elapsed > 0 else 0
            _eta     = (n_total - _done) / _rate if _rate > 0 else None
            _eta_str = f"{int(_eta//3600)}h{int(_eta%3600//60):02d}m" if _eta else 'unknown'
            print(f"  → {_done}/{n_total} done │ elapsed {int(_elapsed//60)}m{int(_elapsed%60):02d}s"
                  f" │ eta {_eta_str} │ CSV saved", flush=True)

            pbar.update(len(batch))

    final_df = pd.DataFrame(results_data)
    final_df = final_df.sort_values(by=[ax1_col, ax2_col])
    final_df.to_csv(csv_filename, index=False)

    total_elapsed = _time.perf_counter() - _suite_t0
    print(f"\n{'═'*W}")
    print(f"  ✅ Suite complete!  │  {_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time   : "
          f"{int(total_elapsed//3600)}h{int(total_elapsed%3600//60):02d}m{int(total_elapsed%60):02d}s")
    print(f"  Results      : {csv_filename}")
    print(f"{'═'*W}\n")

    generate_all_plots(csv_filename)


if __name__ == '__main__':
    main()