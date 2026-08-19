# run_mortality_hydrolysis_sweep.py
"""
Sequential (non-batched) 2D sweep: Mortality Multiplier x Hydrolysis Rate
(k_hyd_max in biopar.py, default 1e-2).

Utilizes asynchronous operator splitting (bio-skipping) for accelerated execution.
"""
import os
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import config as cfg
from physics import setup_physics, get_psi_pert, get_rhs_batched, apply_implicit_visc, get_rhs_bio_only
from bcs import apply_bcs, enforce_symmetry, inflow

# ── 1. Sweep axes ────────────────────────────────────────────────────────────
N_MORT = 6
N_HYD  = 6

mort_vals = np.linspace(0.01, 0.15, N_MORT)
hyd_vals  = np.logspace(np.log10(1e-4), np.log10(1.0), N_HYD)

cfg.poc_initial_core = 2.5e9
cfg.batch_size = 1
cfg.BIO_ACCEL = 1.0
cfg.use_klawonn_density = True

MAX_SIM_TIME = 86400 * 20
MACRO_CYCLE_TIME = 50.0
FLUSH_TIME = 20.0

BIO_TRACER_NAMES = ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox']

total_runs = N_MORT * N_HYD
print(f"\n{'#'*70}\n  2D sweep: {N_MORT} mortality x {N_HYD} hydrolysis = {total_runs} "
      f"INDEPENDENT full simulations.\n  MAX_SIM_TIME per run = {MAX_SIM_TIME/86400:.1f} days.\n{'#'*70}\n")

_run_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = (
    f"MortalityHydrolysis_Sweep"
    f"_mort{mort_vals[0]:.3f}-{mort_vals[-1]:.3f}"
    f"_hyd{hyd_vals[0]:.0e}-{hyd_vals[-1]:.0e}"
    f"_POCunlimited"
    f"_maxT{int(MAX_SIM_TIME)}s"
    f"_{_run_stamp}"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving all outputs to: {OUTPUT_DIR}/")

csv_data = []

results_min_o2            = np.zeros((N_HYD, N_MORT))
results_anoxic_frac       = np.zeros((N_HYD, N_MORT))
results_realized_growth   = np.zeros((N_HYD, N_MORT))
results_avg_growth        = np.zeros((N_HYD, N_MORT))
results_bulk_growth       = np.zeros((N_HYD, N_MORT))
results_peak_biomass      = np.zeros((N_HYD, N_MORT))
results_min_o2_monod      = np.zeros((N_HYD, N_MORT))
results_min_doc_monod     = np.zeros((N_HYD, N_MORT))
results_o2_lim_frac       = np.zeros((N_HYD, N_MORT))
results_terminal_min_o2   = np.zeros((N_HYD, N_MORT))
results_mu_eq             = np.zeros((N_HYD, N_MORT))
results_b_eq_sim          = np.zeros((N_HYD, N_MORT))
results_b_eq_theory       = np.full((N_HYD, N_MORT), np.nan)


O2_ATOL, O2_RTOL = 1e-8, 1e-7

def pointwise_ok(field, dot, atol, rtol, mask):
    tol = atol + rtol * field.abs()
    violation = dot.abs() - tol
    violation = torch.where(mask, torch.full_like(violation, -float('inf')), violation)
    worst = violation.max()
    return bool((worst <= 0.0).item()), float(worst.item())


# ── 2. Sequential execution ──────────────────────────────────────────────────
for i, hyd_val in enumerate(hyd_vals):
    for b, mort_val in enumerate(mort_vals):
        run_label = f"mort={mort_val:.4f}, k_hyd_max={hyd_val:.2e}"
        print(f"\n{'='*70}\nRun {i*N_MORT + b + 1}/{total_runs}: {run_label}\n{'='*70}")

        device, state = setup_physics(cfg)

        def _scalar(x):
            return float(np.asarray(x.detach().cpu() if torch.is_tensor(x) else x).reshape(-1)[0])
        base_m_l = _scalar(state['bgc'].m_l)
        base_m_q = _scalar(state['bgc'].m_q)

        state['bgc'].m_l        = state['bgc'].m_l        * mort_val
        state['bgc'].m_q        = state['bgc'].m_q        * mort_val
        state['bgc'].zoo_m_l    = state['bgc'].zoo_m_l    * mort_val
        state['bgc'].zoo_m_q    = state['bgc'].zoo_m_q    * mort_val
        state['bgc'].k_hyd_max  = float(hyd_val)

        dt = state['dt']
        impl_drag  = {s: state[f'impl_drag_{s}']  for s in ('s1', 's2', 's3')}
        helm_denom = {s: state[f'helm_denom_{s}'] for s in ('s1', 's2', 's3')}

        inv_particle_mask = (state['particle_mask'] == 0)
        core_mask  = ~inv_particle_mask
        core_cells = core_mask.sum().float()

        k_o2  = getattr(state['bgc'], 'aer_k_oxi', 1.0)
        k_doc = getattr(state['bgc'], 'aer_k_red', 1.0)
        base_vmax_s = 2.3148e-05

        min_o2_path            = float('inf')
        max_anoxic_frac_path   = 0.0
        max_realized_growth_path = 0.0
        max_avg_growth_path    = 0.0
        max_bulk_growth_path   = 0.0
        max_peak_biomass_path  = 0.0
        min_o2_monod_path      = 1.0
        min_doc_monod_path     = 1.0
        max_o2_lim_frac_path   = 0.0

        current_time = 0.0
        psi_bg = state['psi_bg']
        
        with tqdm(total=int(MAX_SIM_TIME), desc=run_label) as pbar:
            while current_time < MAX_SIM_TIME:
                
                # ── Step A: Asynchronous Biology Fast-Forward ──
                t_bio_elapsed = 0.0
                while t_bio_elapsed < MACRO_CYCLE_TIME and current_time < MAX_SIM_TIME:
                    rhs_bio, _ = get_rhs_bio_only(state['tracers'], state, cfg)
                    
                    dt_max = min(MACRO_CYCLE_TIME - t_bio_elapsed, MAX_SIM_TIME - current_time)
                    for name, rhs in rhs_bio.items():
                        interior = state['tracers'][name][..., 1:-1, 1:-1]
                        neg_mask = rhs < -1e-10
                        if neg_mask.any():
                            safe_dt = (interior[neg_mask] / -rhs[neg_mask]).min().item() * 0.9 
                            dt_max = min(dt_max, safe_dt)
                    
                    dt_bio = max(dt_max, 1e-3) 
                    
                    for name in state['tracer_names']:
                        state['tracers'][name][..., 1:-1, 1:-1] += rhs_bio[name] * dt_bio
                        state['tracers'][name] = torch.clamp(state['tracers'][name], min=0.0)
                        
                    t_bio_elapsed += dt_bio
                    current_time += dt_bio
                    pbar.update(dt_bio)

                # ── Step B: Synchronous Physics Flush ──
                flush_steps = max(1, int(FLUSH_TIME / dt))
                for _ in range(flush_steps):
                    if current_time >= MAX_SIM_TIME:
                        break
                    
                    w = state['w']
                    tracers = state['tracers']
                    
                    psi_tot = get_psi_pert(w, state) + psi_bg
                    rhs_w, rhs_t = get_rhs_batched(w, tracers, psi_tot, state, cfg)
                    w1_temp = apply_implicit_visc((w + dt * rhs_w) * impl_drag['s1'], helm_denom['s1'], state)
                    t1_temp = {k: v + dt * rhs_t[k] for k, v in tracers.items()}
                    w1, t1 = apply_bcs(w1_temp, t1_temp)

                    psi_tot = get_psi_pert(w1, state) + psi_bg
                    rhs_w, rhs_t = get_rhs_batched(w1, t1, psi_tot, state, cfg)
                    w2_temp = apply_implicit_visc((0.75 * w + 0.25 * (w1 + dt * rhs_w)) * impl_drag['s2'], helm_denom['s2'], state)
                    t2_temp = {k: 0.75 * tracers[k] + 0.25 * (t1[k] + dt * rhs_t[k]) for k in tracers.keys()}
                    w2, t2 = apply_bcs(w2_temp, t2_temp)

                    psi_tot = get_psi_pert(w2, state) + psi_bg
                    rhs_w, rhs_t = get_rhs_batched(w2, t2, psi_tot, state, cfg)
                    w_temp = apply_implicit_visc(((1 / 3) * w + (2 / 3) * (w2 + dt * rhs_w)) * impl_drag['s3'], helm_denom['s3'], state)
                    t_temp = {k: (1 / 3) * tracers[k] + (2 / 3) * (t2[k] + dt * rhs_t[k]) for k in tracers.keys()}
                    w, tracers = apply_bcs(w_temp, t_temp)

                    if getattr(cfg, 'use_symmetry', True):
                        w, tracers = enforce_symmetry(w, tracers, state['tracer_names'])

                    state['w'] = w
                    state['tracers'] = tracers
                    current_time += dt
                    pbar.update(dt)

                # ── Convergence Check (End of Macro-Cycle) ──
                psi_tot_chk = get_psi_pert(w, state) + psi_bg
                _, rhs_t_chk = get_rhs_batched(w, tracers, psi_tot_chk, state, cfg)

                o2_ok, o2_worst = pointwise_ok(state['tracers']['o2'], rhs_t_chk['o2'], O2_ATOL, O2_RTOL, inv_particle_mask)

                if o2_ok:
                    print(f"  Converged at t={current_time:.0f}s -- O2 flat at every in-particle cell.")
                    break

                # ── Track metrics at end of cycle ──
                o2 = state['tracers']['o2']; doc = state['tracers']['doc']
                total_biomass = sum(state['tracers'][bug] for bug in BIO_TRACER_NAMES)

                o2_core = torch.where(inv_particle_mask, torch.full_like(o2, float('inf')), o2)
                min_o2_path = min(min_o2_path, float(o2_core.min().item()))

                anoxic_frac = float((((o2 <= 1.0) & core_mask).sum().float() / core_cells * 100.0).item())
                max_anoxic_frac_path = max(max_anoxic_frac_path, anoxic_frac)

                o2_monod  = o2 / (k_o2 + o2)
                doc_monod = doc / (k_doc + doc)
                o2_monod_core  = torch.where(inv_particle_mask, torch.ones_like(o2_monod), o2_monod)
                doc_monod_core = torch.where(inv_particle_mask, torch.ones_like(doc_monod), doc_monod)
                min_o2_monod_path  = min(min_o2_monod_path,  float(o2_monod_core.min().item()))
                min_doc_monod_path = min(min_doc_monod_path, float(doc_monod_core.min().item()))

                o2_lim_frac = float((((o2_monod < doc_monod) & core_mask).sum().float() / core_cells * 100.0).item())
                max_o2_lim_frac_path = max(max_o2_lim_frac_path, o2_lim_frac)

                monod_min = torch.minimum(o2_monod, doc_monod)
                realized_growth = base_vmax_s * monod_min
                realized_growth_core = torch.where(inv_particle_mask, torch.zeros_like(realized_growth), realized_growth)
                max_realized_growth_path = max(max_realized_growth_path, float(realized_growth_core.max().item()))
                max_avg_growth_path = max(max_avg_growth_path, float((realized_growth_core.sum() / core_cells).item()))

                total_biomass_core = torch.where(inv_particle_mask, torch.zeros_like(total_biomass), total_biomass)
                max_peak_biomass_path = max(max_peak_biomass_path, float(total_biomass_core.max().item()))

                bulk_growth_core = realized_growth_core * total_biomass_core
                max_bulk_growth_path = max(max_bulk_growth_path, float(bulk_growth_core.max().item()))

        # ── Terminal snapshot ──
        o2 = state['tracers']['o2']; doc = state['tracers']['doc']
        total_biomass = sum(state['tracers'][bug] for bug in BIO_TRACER_NAMES)

        terminal_min_o2 = float(torch.where(inv_particle_mask, torch.full_like(o2, float('inf')), o2).min().item())
        total_biomass_core = torch.where(inv_particle_mask, torch.zeros_like(total_biomass), total_biomass)
        terminal_peak_biomass = float(total_biomass_core.max().item())

        o2_monod = o2 / (k_o2 + o2); doc_monod = doc / (k_doc + doc)
        monod_min = torch.minimum(o2_monod, doc_monod)
        realized_growth_core = torch.where(inv_particle_mask, torch.zeros_like(monod_min), base_vmax_s * monod_min)
        terminal_avg_growth = float((realized_growth_core.sum() / core_cells).item())

        m_l_eff_day = base_m_l * mort_val * 86400.0
        m_q_eff_day = base_m_q * mort_val * 86400.0
        mu_eq_day   = terminal_avg_growth * 86400.0
        numerator   = mu_eq_day - m_l_eff_day
        b_eq_theory = np.nan
        if m_q_eff_day > 0 and numerator > 0:
            b_eq_theory = numerator / m_q_eff_day

        results_min_o2[i, b]          = min_o2_path
        results_anoxic_frac[i, b]     = max_anoxic_frac_path
        results_realized_growth[i, b] = max_realized_growth_path * 86400.0
        results_avg_growth[i, b]      = max_avg_growth_path * 86400.0
        results_bulk_growth[i, b]     = max_bulk_growth_path * 86400.0
        results_peak_biomass[i, b]    = max_peak_biomass_path
        results_min_o2_monod[i, b]    = min_o2_monod_path
        results_min_doc_monod[i, b]   = min_doc_monod_path
        results_o2_lim_frac[i, b]     = max_o2_lim_frac_path
        results_terminal_min_o2[i, b] = terminal_min_o2
        results_mu_eq[i, b]           = mu_eq_day
        results_b_eq_sim[i, b]        = terminal_peak_biomass
        results_b_eq_theory[i, b]     = b_eq_theory

        csv_data.append({
            'Mortality_Multiplier': float(mort_val),
            'Hydrolysis_Rate_k_hyd_max': float(hyd_val),
            'POC_Initial_Core': float(cfg.poc_initial_core),
            'Vmax_Multiplier_Untouched': 1.0,
            'Initial_Aerobic_Biomass_Untouched': float(inflow.aer),
            'Min_O2_Path': min_o2_path,
            'Max_Anoxic_Frac_Path': max_anoxic_frac_path,
            'Peak_Specific_Growth_day_Path': max_realized_growth_path * 86400.0,
            'Avg_Specific_Growth_day_Path': max_avg_growth_path * 86400.0,
            'Peak_Bulk_Growth_day_Path': max_bulk_growth_path * 86400.0,
            'Peak_Total_Biomass_Path': max_peak_biomass_path,
            'Lowest_O2_Monod_Factor_Path': min_o2_monod_path,
            'Lowest_DOC_Monod_Factor_Path': min_doc_monod_path,
            'Max_O2_Limitation_Frac_Path': max_o2_lim_frac_path,
            'Terminal_Min_O2': terminal_min_o2,
            'B_eq_Simulated': terminal_peak_biomass,
            'Mu_Eq_day': mu_eq_day,
            'M_L_Eff_day': m_l_eff_day,
            'M_Q_Eff_day': m_q_eff_day,
            'B_eq_Theory': b_eq_theory,
        })
        pd.DataFrame(csv_data).to_csv(os.path.join(OUTPUT_DIR, 'mortality_hydrolysis_sweep_results.csv'), index=False)

        print(f"  Done | Terminal min O2={terminal_min_o2:.3f} | B_eq={terminal_peak_biomass:.2f}")

# ── 3. Plotting: 2D contour maps ─────────────────────────────────────────────
X, Y = np.meshgrid(mort_vals, hyd_vals)
x_label, y_label = 'Mortality Multiplier', 'Hydrolysis Rate $k_{hyd,max}$ (s$^{-1}$)'

def contour_plot(Z, cbar_label, title, fname, cmap='viridis', levels=30):
    plt.figure(figsize=(9, 7))
    cp = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
    cbar = plt.colorbar(cp); cbar.set_label(cbar_label, fontsize=12)
    plt.yscale('log')
    plt.xlabel(x_label, fontsize=12); plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    plt.close()

contour_plot(results_min_o2, 'Lowest $O_2$ recorded (mmol/m³)',
             'Lowest $O_2$ Reached (path minimum)', 'sweep_min_o2.png', cmap='magma_r')
contour_plot(results_anoxic_frac, 'Max Core Anoxic Fraction (%)',
             'Maximum Core Anoxic Fraction Reached', 'sweep_anoxic_frac.png', cmap='inferno')
contour_plot(results_realized_growth, 'Peak Specific Growth Rate (day$^{-1}$)',
             'Peak Local Specific Growth Rate', 'sweep_realized_growth.png')
contour_plot(results_avg_growth, 'Peak Average Core Growth Rate (day$^{-1}$)',
             'Peak Average Core Specific Growth Rate', 'sweep_avg_growth.png', cmap='plasma')
contour_plot(results_bulk_growth, 'Peak Bulk Growth Rate (mmol C m$^{-3}$ day$^{-1}$)',
             'Peak Bulk Growth Rate (Specific Rate × Biomass)', 'sweep_bulk_growth.png', cmap='plasma')
contour_plot(results_peak_biomass, 'Peak Total Community Biomass (mmol C/m³)',
             'Peak Local Total Biomass Reached', 'sweep_peak_biomass.png', cmap='cividis')
contour_plot(results_min_o2_monod, 'Lowest $O_2$ Monod Factor', 'Maximum $O_2$ Limitation Reached',
             'sweep_min_o2_monod.png', cmap='RdYlGn', levels=np.linspace(0, 1, 21))
contour_plot(results_min_doc_monod, 'Lowest DOC Monod Factor', 'Maximum DOC Limitation Reached',
             'sweep_min_doc_monod.png', cmap='RdYlGn', levels=np.linspace(0, 1, 21))
contour_plot(results_o2_lim_frac, 'Max Core % Limited by $O_2$', '$O_2$ vs. DOC Limitation Dominance',
             'sweep_limiting_factor.png', cmap='coolwarm', levels=np.linspace(0, 100, 21))
contour_plot(results_terminal_min_o2, 'Terminal (Steady-State) Min $O_2$ (mmol/m³)',
             'Terminal $O_2$ at Stopping Point', 'sweep_terminal_o2.png', cmap='magma_r')
contour_plot(results_mu_eq, 'Terminal Specific Growth Rate (day$^{-1}$)',
             'Terminal (Steady-State) Specific Growth Rate', 'sweep_mu_eq.png', cmap='plasma')
contour_plot(results_b_eq_sim, 'Terminal Total Biomass, $B_{eq}$ (mmol C/m³)',
             'Terminal Total Biomass at Lock', 'sweep_b_eq.png', cmap='cividis')

print(f"\nSweep complete. Outputs saved to: {OUTPUT_DIR}/")