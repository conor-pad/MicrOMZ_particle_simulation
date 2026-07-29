# run_amplification_experiments.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import config as cfg
from physics import setup_physics, get_psi_pert, get_rhs_batched, apply_implicit_visc
from bcs import apply_bcs, enforce_symmetry, inflow
from biopar import BioPar

# ── 1. Sweep Configuration ──────────────────────────────────────────────────
BATCH_SIZE = 7

# Force settings to prevent DOC limitation and set temporal/batch configs
cfg.batch_size = BATCH_SIZE                  
cfg.BIO_ACCEL = 1.0                 
cfg.poc_initial_core = 1e6          
cfg.use_klawonn_density = True              

# ── SWEEP MODE SELECTOR ──
# Options: 'vmax_aer' (Original), 'mort_aer' (New 1), 'mort_vmax' (New 2)
SWEEP_MODE = 'vmax_aer'

if SWEEP_MODE == 'vmax_aer':
    y_vals = np.linspace(0.1, 15.0, BATCH_SIZE) # biomass
    x_vals = np.linspace(1.0, 400.0, BATCH_SIZE) # mvax
    y_name, x_name = 'aer', 'vmax'
    y_label, x_label = 'Initial Aerobic Biomass (mmol C/m³)', '$V_{max}$ Multiplier'
elif SWEEP_MODE == 'mort_aer':
    y_vals = np.linspace(0.1, 32.0, BATCH_SIZE) # biomass
    x_vals = np.linspace(0.1, 2, BATCH_SIZE) # mortlaity
    y_name, x_name = 'aer', 'mort'
    y_label, x_label = 'Initial Aerobic Biomass (mmol C/m³)', 'Mortality Multiplier'
elif SWEEP_MODE == 'mort_vmax':
    y_vals = np.linspace(1.0, 500.0, BATCH_SIZE)
    x_vals = np.linspace(0.01, 1.0, BATCH_SIZE)
    y_name, x_name = 'vmax', 'mort'
    y_label, x_label = '$V_{max}$ Multiplier', 'Mortality Multiplier'

FIXED_AER = 20.0
FIXED_VMAX = 300.0
FIXED_MORT = 1.0

# ── Tag output filenames with the configured total simulation time ──
TIME_TAG = f"_T{int(cfg.Total_Time)}"

# Store results
results_min_o2 = np.zeros((BATCH_SIZE, BATCH_SIZE))
results_anoxic_frac = np.zeros((BATCH_SIZE, BATCH_SIZE))
results_realized_growth = np.zeros((BATCH_SIZE, BATCH_SIZE))
results_avg_growth = np.zeros((BATCH_SIZE, BATCH_SIZE))
results_bulk_growth = np.zeros((BATCH_SIZE, BATCH_SIZE))
results_peak_biomass = np.zeros((BATCH_SIZE, BATCH_SIZE))
results_min_monod = np.zeros((BATCH_SIZE, BATCH_SIZE))
results_min_o2_monod = np.zeros((BATCH_SIZE, BATCH_SIZE))
results_min_doc_monod = np.zeros((BATCH_SIZE, BATCH_SIZE))
results_o2_lim_frac = np.zeros((BATCH_SIZE, BATCH_SIZE))
csv_data = []

# ── 2. Execution Loop ────────────────────────────────────────────────────────
for i, y_val in enumerate(y_vals):
    
    # ── Set loop variables BEFORE setup_physics for bcs.inflow ──
    if y_name == 'aer':
        inflow.aer = y_val
        current_aer = y_val
        current_vmax_scalar = FIXED_VMAX
    elif y_name == 'vmax':
        inflow.aer = FIXED_AER
        current_aer = FIXED_AER
        current_vmax_scalar = y_val
    
    # Initialise physics with the current batch size
    device, state = setup_physics(cfg)
    
    # ── Assign Tensor variables ──
    if x_name == 'vmax':
        vmax_tensor = torch.tensor(x_vals, dtype=torch.float32, device=device).view(BATCH_SIZE, 1, 1)
        mort_tensor = torch.tensor([FIXED_MORT]*BATCH_SIZE, dtype=torch.float32, device=device).view(BATCH_SIZE, 1, 1)
    elif x_name == 'mort':
        vmax_tensor = torch.tensor([current_vmax_scalar]*BATCH_SIZE, dtype=torch.float32, device=device).view(BATCH_SIZE, 1, 1)
        mort_tensor = torch.tensor(x_vals, dtype=torch.float32, device=device).view(BATCH_SIZE, 1, 1)
    
    # ── Wire in Biological Modifiers ──
    # Apply Vmax multiplier to all functional groups
    for prefix in ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox']:
        base_oxi = getattr(state['bgc'], f'{prefix}_vmax_oxi')
        base_red = getattr(state['bgc'], f'{prefix}_vmax_red')
        setattr(state['bgc'], f'{prefix}_vmax_oxi', base_oxi * vmax_tensor)
        setattr(state['bgc'], f'{prefix}_vmax_red', base_red * vmax_tensor)
        
    # Apply Mortality multiplier directly to bgc parameters
    state['bgc'].m_l = state['bgc'].m_l * mort_tensor
    state['bgc'].m_q = state['bgc'].m_q * mort_tensor
    state['bgc'].zoo_m_l = state['bgc'].zoo_m_l * mort_tensor
    state['bgc'].zoo_m_q = state['bgc'].zoo_m_q * mort_tensor
    
    dt = state['dt']
    n_steps = int(cfg.Total_Time / dt)
    
    # Pre-extract variables for the loop to minimize dict lookups
    impl_drag  = {s: state[f'impl_drag_{s}']  for s in ('s1', 's2', 's3')}
    helm_denom = {s: state[f'helm_denom_{s}'] for s in ('s1', 's2', 's3')}
    
    # Track metrics per batch
    min_o2_batch = torch.full((BATCH_SIZE,), float('inf'), device=device)
    max_anoxic_frac = torch.zeros(BATCH_SIZE, device=device)
    max_realized_growth = torch.zeros(BATCH_SIZE, device=device)
    max_avg_growth = torch.zeros(BATCH_SIZE, device=device)
    max_bulk_growth = torch.zeros(BATCH_SIZE, device=device)
    max_peak_biomass = torch.zeros(BATCH_SIZE, device=device)
    min_monod_batch = torch.ones(BATCH_SIZE, device=device)
    min_o2_monod_batch = torch.ones(BATCH_SIZE, device=device)
    min_doc_monod_batch = torch.ones(BATCH_SIZE, device=device)
    max_frac_o2_lim = torch.zeros(BATCH_SIZE, device=device)
    
    # Precompute masks OUTSIDE the loop to save memory
    inv_particle_mask = (state['particle_mask'] == 0)
    core_mask = ~inv_particle_mask
    core_cells = core_mask[0].sum().float()
    
    # ── Constants for Realized Growth Rate ──
    k_o2 = getattr(state['bgc'], 'aer_k_oxi', 1.0) 
    k_doc = getattr(state['bgc'], 'aer_k_red', 1.0)
    base_vmax_s = 2.3148e-05
    applied_vmax = base_vmax_s * vmax_tensor
    
    # Condensed SSP-RK3 Loop
    for n in tqdm(range(n_steps), desc=f"Sweep Row {i+1}/{BATCH_SIZE} ({y_name}={y_val:.1f})"):
        current_time = n * dt
        
        w = state['w']
        tracers = state['tracers']
        psi_bg = state['psi_bg']
        
        # Stage 1
        psi_tot = get_psi_pert(w, state) + psi_bg
        rhs_w, rhs_t = get_rhs_batched(w, tracers, psi_tot, state, cfg)
        w1_temp = apply_implicit_visc((w + dt * rhs_w) * impl_drag['s1'], helm_denom['s1'], state)
        t1_temp = {k: v + dt * rhs_t[k] for k, v in tracers.items()}
        w1, t1 = apply_bcs(w1_temp, t1_temp)
        
        # Stage 2
        psi_tot = get_psi_pert(w1, state) + psi_bg
        rhs_w, rhs_t = get_rhs_batched(w1, t1, psi_tot, state, cfg)
        w2_temp = apply_implicit_visc((0.75 * w + 0.25 * (w1 + dt * rhs_w)) * impl_drag['s2'], helm_denom['s2'], state)
        t2_temp = {k: 0.75 * tracers[k] + 0.25 * (t1[k] + dt * rhs_t[k]) for k in tracers.keys()}
        w2, t2 = apply_bcs(w2_temp, t2_temp)
        
        # Stage 3
        psi_tot = get_psi_pert(w2, state) + psi_bg
        rhs_w, rhs_t = get_rhs_batched(w2, t2, psi_tot, state, cfg)
        w_temp = apply_implicit_visc(((1/3) * w + (2/3) * (w2 + dt * rhs_w)) * impl_drag['s3'], helm_denom['s3'], state)
        t_temp = {k: (1/3) * tracers[k] + (2/3) * (t2[k] + dt * rhs_t[k]) for k in tracers.keys()}
        w, tracers = apply_bcs(w_temp, t_temp)
        
        if getattr(cfg, 'use_symmetry', True):
            w, tracers = enforce_symmetry(w, tracers, state['tracer_names'])
            
        state['w'] = w
        state['tracers'] = tracers
        
        # Calculate stats WITHOUT cloning or editing the original tracer
        o2 = tracers['o2']
        doc = tracers['doc']
        
        # Sum biomass across all amplified functional groups for total community mass
        total_biomass = sum(tracers[bug] for bug in ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox'])
        
        # Find minimum O2
        o2_core = torch.where(inv_particle_mask, float('inf'), o2)
        current_min = o2_core.view(BATCH_SIZE, -1).min(dim=1)[0]
        min_o2_batch = torch.minimum(min_o2_batch, current_min)
        
        # Calculate percentage of core cells <= 1.0 mmol/m³
        anoxic_cells = ((o2 <= 1.0) & core_mask).view(BATCH_SIZE, -1).sum(dim=1).float()
        current_frac = (anoxic_cells / core_cells) * 100.0
        max_anoxic_frac = torch.maximum(max_anoxic_frac, current_frac)
        
        # Calculate Individual Monod Factors
        o2_monod = o2 / (k_o2 + o2)
        doc_monod = doc / (k_doc + doc)
        monod_min = torch.minimum(o2_monod, doc_monod)
        
        # Track minimum Monod factor strictly inside the core
        monod_core = torch.where(inv_particle_mask, 1.0, monod_min)
        current_min_monod = monod_core.view(BATCH_SIZE, -1).min(dim=1)[0]
        min_monod_batch = torch.minimum(min_monod_batch, current_min_monod)

        o2_monod_core = torch.where(inv_particle_mask, 1.0, o2_monod)
        doc_monod_core = torch.where(inv_particle_mask, 1.0, doc_monod)
        if n > 10:
            current_min_o2_monod = o2_monod_core.view(BATCH_SIZE, -1).min(dim=1)[0]
            current_min_doc_monod = doc_monod_core.view(BATCH_SIZE, -1).min(dim=1)[0]
            min_o2_monod_batch = torch.minimum(min_o2_monod_batch, current_min_o2_monod)
            min_doc_monod_batch = torch.minimum(min_doc_monod_batch, current_min_doc_monod)

        # Track percentage of core cells strictly limited by O2 (O2 factor < DOC factor)
        o2_limiting_cells = ((o2_monod < doc_monod) & core_mask).view(BATCH_SIZE, -1).sum(dim=1).float()
        current_o2_lim_frac = (o2_limiting_cells / core_cells) * 100.0
        max_frac_o2_lim = torch.maximum(max_frac_o2_lim, current_o2_lim_frac)
        
        # Fixed specific growth rate (independent of biomass population size)
        realized_growth = applied_vmax * monod_min
        realized_growth_core = torch.where(inv_particle_mask, 0.0, realized_growth)
        
        # Peak local specific growth
        current_max_growth = realized_growth_core.view(BATCH_SIZE, -1).max(dim=1)[0]
        max_realized_growth = torch.maximum(max_realized_growth, current_max_growth)

        # Peak average specific growth across the entire core
        current_avg_growth = realized_growth_core.view(BATCH_SIZE, -1).sum(dim=1) / core_cells
        max_avg_growth = torch.maximum(max_avg_growth, current_avg_growth)

        # Track peak total biomass reached locally within the core
        total_biomass_core = torch.where(inv_particle_mask, 0.0, total_biomass)
        current_peak_biomass = total_biomass_core.view(BATCH_SIZE, -1).max(dim=1)[0]
        max_peak_biomass = torch.maximum(max_peak_biomass, current_peak_biomass)

        # Bulk growth rate: specific growth rate scaled by the local biomass actually present
        bulk_growth_core = realized_growth_core * total_biomass_core
        current_max_bulk_growth = bulk_growth_core.view(BATCH_SIZE, -1).max(dim=1)[0]
        max_bulk_growth = torch.maximum(max_bulk_growth, current_max_bulk_growth)
        
    results_min_o2[i, :] = min_o2_batch.cpu().numpy()
    results_anoxic_frac[i, :] = max_anoxic_frac.cpu().numpy()
    results_min_monod[i, :] = min_monod_batch.cpu().numpy()
    results_min_o2_monod[i, :] = min_o2_monod_batch.cpu().numpy()
    results_min_doc_monod[i, :] = min_doc_monod_batch.cpu().numpy()
    results_o2_lim_frac[i, :] = max_frac_o2_lim.cpu().numpy()
    
    # Convert growth from s^-1 to day^-1 for final results
    results_realized_growth[i, :] = max_realized_growth.cpu().numpy() * 86400.0
    results_avg_growth[i, :] = max_avg_growth.cpu().numpy() * 86400.0
    results_bulk_growth[i, :] = max_bulk_growth.cpu().numpy() * 86400.0
    results_peak_biomass[i, :] = max_peak_biomass.cpu().numpy()

    # Append the results of this batch to our list
    for b in range(BATCH_SIZE):
        csv_data.append({
            'Initial_Aerobic_Biomass': float(current_aer),
            'Vmax_Multiplier': float(vmax_tensor[b].item()),
            'Mortality_Multiplier': float(mort_tensor[b].item()),
            'Min_O2': float(results_min_o2[i, b]),
            'Max_Anoxic_Frac': float(results_anoxic_frac[i, b]),
            'Peak_Specific_Growth_day': float(results_realized_growth[i, b]),
            'Avg_Specific_Growth_day': float(results_avg_growth[i, b]),
            'Peak_Bulk_Growth_day': float(results_bulk_growth[i, b]),
            'Peak_Total_Biomass': float(results_peak_biomass[i, b]),
            'Lowest_Monod_Factor': float(results_min_monod[i, b]),
            'Lowest_O2_Monod_Factor': float(results_min_o2_monod[i, b]),
            'Lowest_DOC_Monod_Factor': float(results_min_doc_monod[i, b]),
            'Max_O2_Limitation_Frac': float(results_o2_lim_frac[i, b])
        })
    
    # Save to CSV as it goes (overwrites file with updated data)
    pd.DataFrame(csv_data).to_csv(f'run_amplification_results{TIME_TAG}.csv', index=False)


# ── 3. Plotting: Lowest O2 ───────────────────────────────────────────────────
plt.figure(figsize=(9, 7))
X, Y = np.meshgrid(x_vals, y_vals)

cp = plt.contourf(X, Y, results_min_o2, levels=30, cmap='magma_r')
cbar = plt.colorbar(cp)
cbar.set_label('Lowest Particle $O_2$ recorded (mmol/m³)', fontsize=12)

contours = plt.contour(X, Y, results_min_o2, levels=15, colors='black', linewidths=0.5, alpha=0.7)
plt.clabel(contours, inline=True, fontsize=8)

denit_threshold = 1.0
thresh = plt.contour(X, Y, results_min_o2, levels=[denit_threshold], colors='red', linewidths=2.5)
plt.clabel(thresh, inline=True, fmt='Denit. Threshold ($O_2$ = 1.0)', fontsize=10, colors='red')

plt.title('Parameter Sweep: Conditions for Denitrification Onset', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_experiments{TIME_TAG}.png', dpi=300)

# ── 4. Plotting: Anoxic Percentage ───────────────────────────────────────────
plt.figure(figsize=(9, 7))

cp2 = plt.contourf(X, Y, results_anoxic_frac, levels=np.linspace(0, 100, 21), cmap='inferno')
cbar2 = plt.colorbar(cp2)
cbar2.set_label('Max Core Anoxic Percentage (%)', fontsize=12)

contours_frac = plt.contour(X, Y, results_anoxic_frac, levels=[1.0, 50.0, 99.0], 
                            colors='white', linewidths=1.5, linestyles='dashed', alpha=0.8)
plt.clabel(contours_frac, inline=True, fmt='%1.0f%%', fontsize=10, colors='white')

plt.title('Maximum Percentage of Core Reaching Anoxia (O2 ≤ 1.0)', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_anoxic_frac{TIME_TAG}.png', dpi=300)

# ── 5. Plotting: Realized Specific Growth Rates (day⁻¹) ───────────────────────
plt.figure(figsize=(9, 7))

cp3 = plt.contourf(X, Y, results_realized_growth, levels=30, cmap='viridis')
cbar3 = plt.colorbar(cp3)
cbar3.set_label('Peak Specific Growth Rate (day$^{-1}$)', fontsize=12)

contours_rates = plt.contour(X, Y, results_realized_growth, levels=10, colors='white', linewidths=0.5, alpha=0.7)
plt.clabel(contours_rates, inline=True, fontsize=8, fmt='%.0f')

plt.title('Parameter Sweep: Peak Local Specific Growth Rate', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_realized_rates{TIME_TAG}.png', dpi=300)

# ── 6. Plotting: Average Specific Growth Rate (day⁻¹) ────────────────────────
plt.figure(figsize=(9, 7))

cp4 = plt.contourf(X, Y, results_avg_growth, levels=30, cmap='plasma')
cbar4 = plt.colorbar(cp4)
cbar4.set_label('Peak Average Core Specific Growth Rate (day$^{-1}$)', fontsize=12)
contours_avg = plt.contour(X, Y, results_avg_growth, levels=10, colors='white', linewidths=0.5, alpha=0.7)
plt.clabel(contours_avg, inline=True, fontsize=8, fmt='%.0f')

plt.title('Parameter Sweep: Peak Average Core Specific Growth Rate', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_avg_growth{TIME_TAG}.png', dpi=300)

# ── 6b. Plotting: Bulk Growth Rate (Specific Rate x Biomass) ─────────────────
plt.figure(figsize=(9, 7))

cp4b = plt.contourf(X, Y, results_bulk_growth, levels=30, cmap='plasma')
cbar4b = plt.colorbar(cp4b)
cbar4b.set_label('Peak Bulk Growth Rate (mmol C m$^{-3}$ day$^{-1}$)', fontsize=12)
contours_bulk = plt.contour(X, Y, results_bulk_growth, levels=10, colors='white', linewidths=0.5, alpha=0.7)
plt.clabel(contours_bulk, inline=True, fontsize=8, fmt='%.0f')

plt.title('Parameter Sweep: Peak Bulk Growth Rate (Specific Rate × Biomass)', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_bulk_growth{TIME_TAG}.png', dpi=300)

# ── 7. Plotting: Peak Total Biomass ──────────────────────────────────────────
plt.figure(figsize=(9, 7))

cp5 = plt.contourf(X, Y, results_peak_biomass, levels=30, cmap='cividis')
cbar5 = plt.colorbar(cp5)
cbar5.set_label('Peak Total Community Biomass (mmol C/m³)', fontsize=12)

contours_bio = plt.contour(X, Y, results_peak_biomass, levels=10, colors='white', linewidths=0.5, alpha=0.7)
plt.clabel(contours_bio, inline=True, fontsize=8, fmt='%.0f')

plt.title('Parameter Sweep: Peak Local Total Biomass', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_peak_biomass{TIME_TAG}.png', dpi=300)

# ── 8a. Plotting: Lowest O2 Monod Factor ─────────────────────────────────────
plt.figure(figsize=(9, 7))

cp6a = plt.contourf(X, Y, results_min_o2_monod, levels=np.linspace(0, 1, 21), cmap='RdYlGn')
cbar6a = plt.colorbar(cp6a)
cbar6a.set_label('Lowest $O_2$ Monod Factor (0 = $O_2$-Starved, 1 = Unlimited)', fontsize=12)

contours_o2_monod = plt.contour(X, Y, results_min_o2_monod, levels=[0.1, 0.5, 0.9], colors='black', linewidths=1.0, linestyles='dashed', alpha=0.8)
plt.clabel(contours_o2_monod, inline=True, fontsize=10, colors='black')

plt.title('Parameter Sweep: Maximum $O_2$ Limitation (Lowest $O_2$ Monod Factor)', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_min_o2_monod{TIME_TAG}.png', dpi=300)

# ── 8b. Plotting: Lowest DOC Monod Factor ────────────────────────────────────
plt.figure(figsize=(9, 7))

cp6b = plt.contourf(X, Y, results_min_doc_monod, levels=np.linspace(0, 1, 21), cmap='RdYlGn')
cbar6b = plt.colorbar(cp6b)
cbar6b.set_label('Lowest DOC Monod Factor (0 = DOC-Starved, 1 = Unlimited)', fontsize=12)

contours_doc_monod = plt.contour(X, Y, results_min_doc_monod, levels=[0.1, 0.5, 0.9], colors='black', linewidths=1.0, linestyles='dashed', alpha=0.8)
plt.clabel(contours_doc_monod, inline=True, fontsize=10, colors='black')

plt.title('Parameter Sweep: Maximum DOC Limitation (Lowest DOC Monod Factor)', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_min_doc_monod{TIME_TAG}.png', dpi=300)

# ── 8c. Plotting: Which Substrate Actually Drives the Minimum ────────────────
monod_dominance = results_min_doc_monod - results_min_o2_monod
dominance_extent = np.max(np.abs(monod_dominance))
dominance_extent = dominance_extent if dominance_extent > 0 else 1.0

plt.figure(figsize=(9, 7))

cp6c = plt.contourf(X, Y, monod_dominance, levels=21, cmap='coolwarm', vmin=-dominance_extent, vmax=dominance_extent)
cbar6c = plt.colorbar(cp6c)
cbar6c.set_label('DOC-limited  $\\leftarrow$  (Doc$_{min}$ - O2$_{min}$)  $\\rightarrow$  $O_2$-limited', fontsize=12)

zero_line = plt.contour(X, Y, monod_dominance, levels=[0.0], colors='black', linewidths=2.0)
plt.clabel(zero_line, inline=True, fmt='Crossover', fontsize=10, colors='black')

plt.title('Parameter Sweep: Which Substrate Drives the Growth Limitation', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_monod_dominance{TIME_TAG}.png', dpi=300)

# ── 9. Plotting: Dominant Limiting Substrate ─────────────────────────────────
plt.figure(figsize=(9, 7))

cp7 = plt.contourf(X, Y, results_o2_lim_frac, levels=np.linspace(0, 100, 21), cmap='coolwarm')
cbar7 = plt.colorbar(cp7)
cbar7.set_label('Max Core Percentage Limited by $O_2$ (%)', fontsize=12)

contours_o2_lim = plt.contour(X, Y, results_o2_lim_frac, levels=[1.0, 50.0, 99.0], colors='black', linewidths=1.0, linestyles='dashed', alpha=0.8)
plt.clabel(contours_o2_lim, inline=True, fmt='%1.0f%%', fontsize=10, colors='black')

plt.title('Parameter Sweep: $O_2$ vs DOC Limitation ($O_2$ Dominance)', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_limiting_factor{TIME_TAG}.png', dpi=300)

print("Sweep complete. Outputs saved.")