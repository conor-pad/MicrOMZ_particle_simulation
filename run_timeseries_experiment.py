# run_timeseries_experiment.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as cfg
from physics import setup_physics, get_psi_pert, get_rhs_batched, apply_implicit_visc
from bcs import apply_bcs, enforce_symmetry, inflow
from biopar import BioPar

# ── 1. Configuration ─────────────────────────────────────────────────────────
BATCH_SIZE = 10
TOTAL_TIME = 5000.0
FIXED_VMAX = 100.0

cfg.batch_size = BATCH_SIZE
cfg.Total_Time = TOTAL_TIME
cfg.BIO_ACCEL = 1.0
cfg.poc_initial_core = 1e6
cfg.use_klawonn_density = True

# Sweep over initial biomass (targeting the anoxia crash threshold)
initial_biomasses = np.linspace(1.0, 25.0, BATCH_SIZE)

BIO_TRACER_NAMES = ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox']

# ── 2. Setup Physics & Overrides ──────────────────────────────────────────────
device, state = setup_physics(cfg)
dt = state['dt']
n_steps = int(TOTAL_TIME / dt)

# Override the initial aerobic biomass per batch member manually
for b in range(BATCH_SIZE):
    mask = state['particle_mask'][b] > 0
    state['tracers']['aer'][b, mask] = initial_biomasses[b]

# Apply fixed Vmax multiplier to all batch members
vmax_tensor = torch.tensor([FIXED_VMAX]*BATCH_SIZE, dtype=torch.float32, device=device).view(BATCH_SIZE, 1, 1)
for prefix in BIO_TRACER_NAMES:
    base_oxi = getattr(state['bgc'], f'{prefix}_vmax_oxi')
    base_red = getattr(state['bgc'], f'{prefix}_vmax_red')
    setattr(state['bgc'], f'{prefix}_vmax_oxi', base_oxi * vmax_tensor)
    setattr(state['bgc'], f'{prefix}_vmax_red', base_red * vmax_tensor)

impl_drag  = {s: state[f'impl_drag_{s}']  for s in ('s1', 's2', 's3')}
helm_denom = {s: state[f'helm_denom_{s}'] for s in ('s1', 's2', 's3')}

core_mask = (state['particle_mask'] > 0)
core_cells = core_mask[0].sum().float()

# ── 3. Storage Arrays ────────────────────────────────────────────────────────
history_time = []
history_biomass = [] # Shape will be (num_records, BATCH_SIZE)

# ── 4. Execution Loop ────────────────────────────────────────────────────────
for n in tqdm(range(n_steps), desc=f"Simulating to t={TOTAL_TIME}s"):
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

    # Record data every 20 steps to save memory
    if n % 20 == 0:
        total_biomass = sum(tracers[bug] for bug in BIO_TRACER_NAMES)
        biomass_core_mean = torch.where(core_mask, total_biomass, 0.0).view(BATCH_SIZE, -1).sum(dim=1) / core_cells
        
        history_time.append(current_time)
        history_biomass.append(biomass_core_mean.cpu().numpy())

# ── 5. Plotting ──────────────────────────────────────────────────────────────
history_time = np.array(history_time)
history_biomass = np.array(history_biomass)

plt.figure(figsize=(10, 6))
colormap = plt.cm.viridis

for b in range(BATCH_SIZE):
    color = colormap(b / (BATCH_SIZE - 1))
    plt.plot(history_time, history_biomass[:, b], 
             label=f'{initial_biomasses[b]:.1f}', 
             color=color, linewidth=2.5)

plt.title(f'Microbial Population Growth Over Time ($V_{{max}}$ Multiplier = {FIXED_VMAX})', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Mean Core Total Biomass (mmol C/m³)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Initial Aerobic Biomass\n(mmol C/m³)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('biomass_timeseries_vmax100.png', dpi=300)
print("Simulation complete. Timeseries plot saved as 'biomass_timeseries_vmax100.png'.")