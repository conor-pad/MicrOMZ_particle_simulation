# run_amplification_experiments.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import pandas as pd

import config as cfg
from physics import setup_physics, get_psi_pert, get_rhs_batched, apply_implicit_visc
from bcs import apply_bcs, enforce_symmetry, inflow
from biopar import BioPar

# ── 1. Sweep Configuration ──────────────────────────────────────────────────
BATCH_SIZE = 5

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

# Whether this sweep configuration has a mortality axis at all. The B_eq vs.
# mortality comparison plot (added below) only makes sense when we're
# actually sweeping m_l/m_q via the Mortality_Multiplier, so it's skipped
# entirely for 'vmax_aer' (no mort axis present).
HAS_MORT_AXIS = (x_name == 'mort') or (y_name == 'mort')

# ── Tag output filenames with the configured safety-cap time ────────────────
# NOTE: now that stopping is steady-state/POC-driven (see below), cfg.Total_Time
# is no longer "how long the sim ran" — it's just the base unit for the hard
# step cap (MAX_STEPS_MULTIPLIER × this). Kept in the filename purely for
# provenance (which cap config produced this run), not as the actual duration.
CAP_TAG = f"_capT{int(cfg.Total_Time)}"

# ── Biomass functional groups summed for total community mass / checked
# for the growth=loss convergence gate. Kept as one explicit list (rather
# than pulling from state['bio_names']) so it's guaranteed to match exactly
# what "total_biomass" below has always summed over.
BIO_TRACER_NAMES = ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox']

# ── Steady-State Convergence Configuration ───────────────────────────────────
# TWO-STAGE gate, checked in this specific order for each batch member:
#
#   STAGE 1 — Biomass growth ≈ loss, PER TRACER.
#     rhs_t[bug] computed by get_rhs_batched() IS dB/dt for that functional
#     group exactly as the solver sees it (growth, mortality/loss, and
#     transport all folded in) — so "growth ≈ loss" is just |dB/dt|/B small,
#     checked per functional group and combined with max() so EVERY group has
#     to have settled, not just whichever one dominates total biomass. This
#     replaces the earlier lumped "has the aggregate growth-rate proxy
#     stopped changing" check, which only really reflected the aerobic
#     Monod formula and could look flat while total biomass was still moving.
#
#   STAGE 2 — Oxygen stabilization, ONLY evaluated once Stage 1 has passed.
#     Before biomass settles, O2 is *expected* to keep moving (it's tracking
#     a still-changing consumption demand), so checking O2 flatness earlier
#     would trigger on a coincidental plateau rather than a real one. Once
#     biomass has stopped changing, |rhs_t['o2']|/O2 is the same kind of
#     direct residual check as Stage 1 — no history/checkpointing needed.
#     DOC is intentionally NOT part of this gate (per your instruction) —
#     it's produced entirely by biomass-driven POC hydrolysis, so once
#     biomass (Stage 1) is flat, DOC's forcing is flat too; it doesn't carry
#     independent information the way O2 (which has its own external
#     ambient/inflow forcing) does.
#
# POC exhaustion is still checked independently, in parallel with both
# stages — running out of fuel pre-empts waiting on either one.
CONVERGENCE_CHECK_EVERY = 10000    # steps between convergence checks
CONVERGENCE_REL_TOL     = 1e-3     # relative residual (|dX/dt| / X) to call a quantity "flat"
POC_DEPLETION_FRAC      = 0.05     # core-mean POC / initial POC below this counts as "meaningfully depleted"
MAX_STEPS_MULTIPLIER    = 4        # hard cap = this × (cfg.Total_Time / dt); gives slow (low-Vmax) members
                                    # a real chance to converge before we give up and flag them inconclusive

# Stop-reason codes (used both in the CSV and in the regime-map plot below)
STOP_STEADY_STATE = 0   # converged: biomass AND O2 both flat, in that order
STOP_POC_EXHAUSTED = 1  # POC ran out before both stages were satisfied
STOP_INCONCLUSIVE  = 2  # hit the hard step cap without doing either — treat with caution

# Store results (existing "peak/transient" trajectory diagnostics — these
# describe the MOST EXTREME condition reached anywhere along the run, which
# remains useful even once we're steady-state-aware; see explanation at bottom)
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

# New: terminal/equilibrium diagnostics — the INSTANTANEOUS state at the
# moment a member was locked, not a running max/min over the trajectory.
results_stop_reason   = np.zeros((BATCH_SIZE, BATCH_SIZE))            # 0/1/2, see codes above
results_time_to_stop  = np.zeros((BATCH_SIZE, BATCH_SIZE))            # simulated seconds at final lock (Stage 2 / exhaustion)
results_biomass_time  = np.full((BATCH_SIZE, BATCH_SIZE), np.nan)     # simulated seconds when Stage 1 (biomass) passed
results_o2_lag        = np.full((BATCH_SIZE, BATCH_SIZE), np.nan)     # Stage-2 stop time minus Stage-1 time — how long O2 took to catch up
results_poc_remaining = np.zeros((BATCH_SIZE, BATCH_SIZE))            # core-mean POC fraction remaining at lock
results_b_eq_sim      = np.zeros((BATCH_SIZE, BATCH_SIZE))            # terminal core-max total biomass ("B_eq")
results_mu_eq         = np.zeros((BATCH_SIZE, BATCH_SIZE))            # terminal specific growth rate, day⁻¹
results_m_l_eff       = np.zeros((BATCH_SIZE, BATCH_SIZE))            # effective m_l (day⁻¹) at this sweep point
results_m_q_eff       = np.zeros((BATCH_SIZE, BATCH_SIZE))            # effective m_q (day⁻¹ per unit biomass)
results_b_eq_theory   = np.full((BATCH_SIZE, BATCH_SIZE), np.nan)     # analytic (mu_eq - m_l)/m_q estimate

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

    # ── Capture PRE-multiplier mortality base rates ──
    # We need these (in their original, un-swept form) to reconstruct the
    # effective m_l/m_q at each swept Mortality_Multiplier value later, for
    # the analytic B_eq = (mu_eq - m_l) / m_q comparison. Must be captured
    # here, before the next block multiplies them in place.
    # NOTE: assuming these are stored in per-second units, consistent with
    # base_vmax_s below — if biopar.py uses different units this conversion
    # will need adjusting; treat B_eq_theory as an order-of-magnitude sanity
    # check against the simulated equilibrium, not an exact prediction (the
    # real community has multiple functional groups combining nonlinearly in
    # sms.py, whereas this is a single lumped linear+quadratic loss term).
    def _scalar(x):
        # bgc params may be plain floats or tensors; normalize to a python float
        return float(np.asarray(x.detach().cpu() if torch.is_tensor(x) else x).reshape(-1)[0])
    base_m_l_val = _scalar(state['bgc'].m_l)
    base_m_q_val = _scalar(state['bgc'].m_q)

    # Apply Mortality multiplier directly to bgc parameters
    state['bgc'].m_l = state['bgc'].m_l * mort_tensor
    state['bgc'].m_q = state['bgc'].m_q * mort_tensor
    state['bgc'].zoo_m_l = state['bgc'].zoo_m_l * mort_tensor
    state['bgc'].zoo_m_q = state['bgc'].zoo_m_q * mort_tensor
    
    dt = state['dt']

    # POC initial density (uniform across the batch in this script — cfg is
    # forced globally above). Used below to compute the fraction of POC
    # remaining, so we can tell "converged because of true balance" apart
    # from "converged because it ran out of fuel and everything crashed".
    POC_INITIAL = float(cfg.poc_initial_core)

    # Hard safety cap on steps. cfg.Total_Time is no longer "the" duration —
    # it's just the base unit for this cap. See CONVERGENCE config comments above.
    n_steps_max = int(MAX_STEPS_MULTIPLIER * cfg.Total_Time / dt)
    
    # Pre-extract variables for the loop to minimize dict lookups
    impl_drag  = {s: state[f'impl_drag_{s}']  for s in ('s1', 's2', 's3')}
    helm_denom = {s: state[f'helm_denom_{s}'] for s in ('s1', 's2', 's3')}
    
    # ── Track metrics per batch (existing PATH-MAX/MIN trackers) ──
    # These continue to answer "what's the single most extreme value reached
    # anywhere along the whole trajectory" — still useful (e.g. "how anoxic
    # did it ever get"), independent of whether steady state was reached.
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

    # ── Per-member steady-state tracking tensors ──
    # `locked[b]` becomes True the instant member b hits either FINAL stopping
    # condition (Stage 2 passed, or POC exhausted); once True, its terminal_*
    # values below are never touched again for the rest of this row's loop.
    locked          = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=device)
    stop_reason     = torch.full((BATCH_SIZE,), -1, dtype=torch.long, device=device)  # -1 = not yet decided
    convergence_step = torch.full((BATCH_SIZE,), -1, dtype=torch.long, device=device)  # step of FINAL lock

    # Stage-1-specific tracking: biomass_converged[b] flips True the first
    # time growth≈loss holds for EVERY functional group; biomass_time records
    # WHEN that happened, separately from the final lock step, so we can
    # measure the O2 "lag" between the two stages afterward.
    biomass_converged = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=device)
    biomass_step       = torch.full((BATCH_SIZE,), -1, dtype=torch.long, device=device)

    # Terminal (instantaneous-at-lock) diagnostics — these are the actual
    # "steady state" numbers, as opposed to the path-max trackers above.
    terminal_avg_growth   = torch.zeros(BATCH_SIZE, device=device)  # s⁻¹, converted to day⁻¹ after the loop
    terminal_peak_biomass = torch.zeros(BATCH_SIZE, device=device)  # B_eq candidate
    terminal_bulk_growth  = torch.zeros(BATCH_SIZE, device=device)  # s⁻¹, converted after
    terminal_min_o2       = torch.zeros(BATCH_SIZE, device=device)
    terminal_poc_frac     = torch.zeros(BATCH_SIZE, device=device)
    
    # Precompute masks OUTSIDE the loop to save memory
    inv_particle_mask = (state['particle_mask'] == 0)
    core_mask = ~inv_particle_mask
    core_cells = core_mask[0].sum().float()

    def core_mean(x):
        # Mean of a field over core (in-particle) cells only — same masking
        # convention as every other core diagnostic in this script.
        return torch.where(inv_particle_mask, 0.0, x).view(BATCH_SIZE, -1).sum(dim=1) / core_cells
    
    # ── Constants for Realized Growth Rate ──
    k_o2 = getattr(state['bgc'], 'aer_k_oxi', 1.0) 
    k_doc = getattr(state['bgc'], 'aer_k_red', 1.0)
    base_vmax_s = 2.3148e-05
    applied_vmax = base_vmax_s * vmax_tensor
    
    # Condensed SSP-RK3 Loop
    for n in tqdm(range(n_steps_max), desc=f"Sweep Row {i+1}/{BATCH_SIZE} ({y_name}={y_val:.1f})"):
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
        total_biomass = sum(tracers[bug] for bug in BIO_TRACER_NAMES)
        
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

        # Peak average specific growth across the entire core (also our
        # recorded terminal diagnostic below — this is already a per-step,
        # non-running value at this point in the code)
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

        # ── Two-stage convergence check ────────────────────────────────────
        # Checked periodically (not every step) — the underlying residuals
        # are smooth relative to the timestep, so this adds negligible cost.
        if (n % CONVERGENCE_CHECK_EVERY == 0) and (n > 0):

            # Re-evaluate the RHS at the actual post-step (w, tracers) state.
            # The stage-3 `rhs_t` still in scope above was evaluated at the
            # stage-2 intermediate state (w2, t2), not the final state this
            # step produced — slightly stale. Checkpoints are far apart
            # (every CONVERGENCE_CHECK_EVERY steps), so one extra RHS
            # evaluation here is negligible and keeps the residual exact.
            psi_tot_chk = get_psi_pert(w, state) + psi_bg
            _, rhs_t_chk = get_rhs_batched(w, tracers, psi_tot_chk, state, cfg)

            # ── STAGE 1: per-tracer biomass growth ≈ loss ──
            # |dB/dt| / B, per functional group, combined with max() so every
            # group (not just the dominant one) has to have settled.
            biomass_residual = torch.stack([
                core_mean(rhs_t_chk[bug].abs()) / (core_mean(tracers[bug]) + 1e-15)
                for bug in BIO_TRACER_NAMES
            ], dim=0).max(dim=0)[0]

            newly_biomass_converged = (~biomass_converged) & (biomass_residual < CONVERGENCE_REL_TOL)
            if newly_biomass_converged.any():
                biomass_converged = biomass_converged | newly_biomass_converged
                biomass_step = torch.where(newly_biomass_converged,
                                            torch.full_like(biomass_step, n),
                                            biomass_step)

            # ── STAGE 2: oxygen stabilization — gated on Stage 1 ──
            # Only members that have ALREADY passed Stage 1 are eligible to
            # pass Stage 2 this checkpoint. Same direct-residual form as
            # Stage 1: |dO2/dt| / O2, using the same rhs_t_chk.
            o2_residual = core_mean(rhs_t_chk['o2'].abs()) / (core_mean(o2) + 1e-15)
            newly_converged = biomass_converged & (o2_residual < CONVERGENCE_REL_TOL) & (~locked)

            # POC exhaustion is independent of both stages — running out of
            # fuel pre-empts waiting further, regardless of stage progress.
            poc_remaining_frac_now = core_mean(tracers['poc']) / POC_INITIAL
            newly_exhausted = (~newly_converged) & (poc_remaining_frac_now < POC_DEPLETION_FRAC) & (~locked)

            newly_locked = newly_converged | newly_exhausted

            if newly_locked.any():
                # Freeze diagnostics using THIS STEP'S INSTANTANEOUS values,
                # not the running max/min trackers above — we want the value
                # AT the equilibrium point, not the peak seen earlier in the
                # transient.
                terminal_avg_growth   = torch.where(newly_locked, current_avg_growth,     terminal_avg_growth)
                terminal_peak_biomass = torch.where(newly_locked, current_peak_biomass,   terminal_peak_biomass)
                terminal_bulk_growth  = torch.where(newly_locked, current_max_bulk_growth, terminal_bulk_growth)
                terminal_min_o2       = torch.where(newly_locked, current_min,            terminal_min_o2)
                terminal_poc_frac     = torch.where(newly_locked, poc_remaining_frac_now, terminal_poc_frac)
                convergence_step = torch.where(newly_locked,
                                                torch.full_like(convergence_step, n),
                                                convergence_step)
                stop_reason = torch.where(newly_converged & newly_locked,
                                           torch.full_like(stop_reason, STOP_STEADY_STATE),
                                           stop_reason)
                stop_reason = torch.where(newly_exhausted & newly_locked,
                                           torch.full_like(stop_reason, STOP_POC_EXHAUSTED),
                                           stop_reason)
                locked = locked | newly_locked

            # IMPORTANT: this does NOT stop individual members early — the
            # physics step above is one batched tensor op across all members
            # at once, so every member keeps physically evolving regardless
            # of its lock state. Locking only freezes that member's RECORDED
            # numbers. The row only stops early once EVERY member is locked:
            if locked.all():
                print(f"  All {BATCH_SIZE} members locked by step {n} (t={current_time:.1f}s) — stopping row early.")
                break

    # ── Finalize any members that never locked (hit the hard step cap) ──
    still_running = ~locked
    if still_running.any():
        final_poc_frac = core_mean(tracers['poc']) / POC_INITIAL
        terminal_avg_growth   = torch.where(still_running, current_avg_growth,     terminal_avg_growth)
        terminal_peak_biomass = torch.where(still_running, current_peak_biomass,   terminal_peak_biomass)
        terminal_bulk_growth  = torch.where(still_running, current_max_bulk_growth, terminal_bulk_growth)
        terminal_min_o2       = torch.where(still_running, current_min,            terminal_min_o2)
        terminal_poc_frac     = torch.where(still_running, final_poc_frac,         terminal_poc_frac)
        convergence_step = torch.where(still_running,
                                        torch.full_like(convergence_step, n),
                                        convergence_step)
        stop_reason = torch.where(still_running,
                                   torch.full_like(stop_reason, STOP_INCONCLUSIVE),
                                   stop_reason)
        locked = locked | still_running

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

    # ── Terminal/equilibrium results (converted s⁻¹ → day⁻¹ where relevant) ──
    results_stop_reason[i, :]   = stop_reason.cpu().numpy()
    results_time_to_stop[i, :]  = (convergence_step.cpu().numpy().astype(np.float64)) * dt
    results_poc_remaining[i, :] = terminal_poc_frac.cpu().numpy()
    results_b_eq_sim[i, :]      = terminal_peak_biomass.cpu().numpy()
    results_mu_eq[i, :]         = terminal_avg_growth.cpu().numpy() * 86400.0

    # Stage-1 (biomass) timing and the O2 "lag" behind it — direct evidence
    # for/against the fast-gas-vs-slow-growth assumption discussed earlier.
    # NaN where biomass never converged within this row's step budget.
    biomass_step_np = biomass_step.cpu().numpy().astype(np.float64)
    biomass_time_row = np.where(biomass_step_np >= 0, biomass_step_np * dt, np.nan)
    results_biomass_time[i, :] = biomass_time_row
    # Lag only meaningful where the run actually reached true steady state
    # (Stage 2 passed) — otherwise "time to stop" reflects exhaustion/cap,
    # not O2 catching up to biomass.
    lag_row = np.where(results_stop_reason[i, :] == STOP_STEADY_STATE,
                        results_time_to_stop[i, :] - biomass_time_row,
                        np.nan)
    results_o2_lag[i, :] = lag_row

    # Effective (day⁻¹) mortality rates at each swept Mortality_Multiplier
    # value in this row, for the analytic B_eq comparison.
    mort_np = mort_tensor.view(-1).cpu().numpy()
    m_l_eff_day = base_m_l_val * mort_np * 86400.0
    m_q_eff_day = base_m_q_val * mort_np * 86400.0
    results_m_l_eff[i, :] = m_l_eff_day
    results_m_q_eff[i, :] = m_q_eff_day

    # Analytic steady-state biomass: at dB/dt = 0,  mu_eq = m_l + m_q * B_eq
    #   =>  B_eq_theory = (mu_eq - m_l) / m_q
    # Only meaningful where a genuine steady state was actually reached
    # (stop_reason == STOP_STEADY_STATE) AND m_q > 0 AND the numerator is
    # positive (otherwise there's no valid positive equilibrium under a
    # linear+quadratic loss model — mask these out as NaN rather than
    # plotting a nonsense negative "biomass").
    numerator = results_mu_eq[i, :] - m_l_eff_day
    valid = (results_stop_reason[i, :] == STOP_STEADY_STATE) & (m_q_eff_day > 0) & (numerator > 0)
    b_eq_theory_row = np.full(BATCH_SIZE, np.nan)
    b_eq_theory_row[valid] = numerator[valid] / m_q_eff_day[valid]
    results_b_eq_theory[i, :] = b_eq_theory_row

    # Print status report for the row
    steady_count = (results_stop_reason[i, :] == 0).sum()
    poc_count = (results_stop_reason[i, :] == 1).sum()
    inc_count = (results_stop_reason[i, :] == 2).sum()
    print(f"\n  Row {i+1} finished: {steady_count} Steady State | {poc_count} POC Exhausted | {inc_count} Inconclusive")

    # Map the number codes to actual text strings for the CSV
    reason_map = {0: 'Steady State', 1: 'POC Exhausted', 2: 'Inconclusive'}

    # Append the results of this batch to our list
    for b in range(BATCH_SIZE):
        stop_code = int(results_stop_reason[i, b])
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
            'Max_O2_Limitation_Frac': float(results_o2_lim_frac[i, b]),
            # Updated string mapping for Stop Reason
            'Stop_Reason': reason_map.get(stop_code, 'Unknown'),
            'Time_To_Stop_s': float(results_time_to_stop[i, b]),
            'Biomass_Converged_Time_s': float(results_biomass_time[i, b]) if not np.isnan(results_biomass_time[i, b]) else None,
            'O2_Lag_After_Biomass_s': float(results_o2_lag[i, b]) if not np.isnan(results_o2_lag[i, b]) else None,
            'POC_Remaining_Frac': float(results_poc_remaining[i, b]),
            'B_eq_Simulated': float(results_b_eq_sim[i, b]),
            'Mu_Eq_day': float(results_mu_eq[i, b]),
            'M_L_Eff_day': float(results_m_l_eff[i, b]),
            'M_Q_Eff_day': float(results_m_q_eff[i, b]),
            'B_eq_Theory': float(results_b_eq_theory[i, b]),
        })
    
    # Save to CSV as it goes (overwrites file with updated data)
    pd.DataFrame(csv_data).to_csv(f'run_amplification_results{CAP_TAG}.csv', index=False)


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
plt.savefig(f'run_amplification_experiments{CAP_TAG}.png', dpi=300)

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
plt.savefig(f'run_amplification_anoxic_frac{CAP_TAG}.png', dpi=300)

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
plt.savefig(f'run_amplification_realized_rates{CAP_TAG}.png', dpi=300)

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
plt.savefig(f'run_amplification_avg_growth{CAP_TAG}.png', dpi=300)

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
plt.savefig(f'run_amplification_bulk_growth{CAP_TAG}.png', dpi=300)

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
plt.savefig(f'run_amplification_peak_biomass{CAP_TAG}.png', dpi=300)

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
plt.savefig(f'run_amplification_min_o2_monod{CAP_TAG}.png', dpi=300)

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
plt.savefig(f'run_amplification_min_doc_monod{CAP_TAG}.png', dpi=300)

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
plt.savefig(f'run_amplification_monod_dominance{CAP_TAG}.png', dpi=300)

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
plt.savefig(f'run_amplification_limiting_factor{CAP_TAG}.png', dpi=300)

# ── 10. Plotting: Terminal (Equilibrium) Specific Growth Rate, Mu_Eq ────────
plt.figure(figsize=(9, 7))
cp10 = plt.contourf(X, Y, results_mu_eq, levels=30, cmap='plasma')
cbar10 = plt.colorbar(cp10)
cbar10.set_label('Terminal (Steady-State) Specific Growth Rate (day$^{-1}$)', fontsize=12)
contours_10 = plt.contour(X, Y, results_mu_eq, levels=10, colors='white', linewidths=0.5, alpha=0.7)
plt.clabel(contours_10, inline=True, fontsize=8, fmt='%.1f')
plt.title('Parameter Sweep: Terminal Specific Growth Rate at Lock', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_mu_eq{CAP_TAG}.png', dpi=300)

# ── 11. Plotting: Terminal (Equilibrium) Total Biomass, B_eq ────────────────
plt.figure(figsize=(9, 7))
cp11 = plt.contourf(X, Y, results_b_eq_sim, levels=30, cmap='cividis')
cbar11 = plt.colorbar(cp11)
cbar11.set_label('Terminal (Steady-State) Total Biomass, $B_{eq}$ (mmol C/m³)', fontsize=12)
contours_11 = plt.contour(X, Y, results_b_eq_sim, levels=10, colors='white', linewidths=0.5, alpha=0.7)
plt.clabel(contours_11, inline=True, fontsize=8, fmt='%.0f')
plt.title('Parameter Sweep: Terminal Total Biomass ($B_{eq}$) at Lock', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_b_eq{CAP_TAG}.png', dpi=300)

# ── 12. Plotting: Stop Reason / Regime Map ──────────────────────────────────
plt.figure(figsize=(9, 7))
regime_colors = ['#2ca02c', '#ff7f0e', '#7f7f7f']  # green / orange / gray
regime_cmap = ListedColormap(regime_colors)
regime_bounds = [-0.5, 0.5, 1.5, 2.5]
regime_norm = BoundaryNorm(regime_bounds, regime_cmap.N)

cp12 = plt.pcolormesh(X, Y, results_stop_reason, cmap=regime_cmap, norm=regime_norm, shading='nearest')
cbar12 = plt.colorbar(cp12, ticks=[0, 1, 2])
cbar12.ax.set_yticklabels(['Steady State', 'POC Exhausted', 'Inconclusive (hit cap)'])
cbar12.set_label('Stopping Regime', fontsize=12)

plt.title('Parameter Sweep: How Each Run Terminated', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_stop_reason{CAP_TAG}.png', dpi=300)

# ── 12b. Plotting: POC Remaining Fraction at Stop ───────────────────────────
plt.figure(figsize=(9, 7))
cp12b = plt.contourf(X, Y, results_poc_remaining * 100.0, levels=np.linspace(0, 100, 21), cmap='YlGnBu')
cbar12b = plt.colorbar(cp12b)
cbar12b.set_label('POC Remaining at Stop (% of initial)', fontsize=12)
contours_12b = plt.contour(X, Y, results_poc_remaining * 100.0, levels=[POC_DEPLETION_FRAC * 100.0],
                            colors='red', linewidths=2.0)
plt.clabel(contours_12b, inline=True, fmt='Depletion threshold', fontsize=9, colors='red')
plt.title('Parameter Sweep: POC Remaining at Time of Stop', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_poc_remaining{CAP_TAG}.png', dpi=300)

# ── 12c. Plotting: Time to Stop ──────────────────────────────────────────────
plt.figure(figsize=(9, 7))
cp12c = plt.contourf(X, Y, results_time_to_stop, levels=30, cmap='magma')
cbar12c = plt.colorbar(cp12c)
cbar12c.set_label('Time to Stop (s)', fontsize=12)
plt.title('Parameter Sweep: Simulated Time Until Lock', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_time_to_stop{CAP_TAG}.png', dpi=300)

# ── 14. NEW Plotting: O2 Lag Behind Biomass Convergence ─────────────────────
# Direct, empirical answer to "does gas equilibrium happen fast compared to
# growth equilibrium": this is Time_To_Stop minus Biomass_Converged_Time,
# i.e. how many extra seconds O2 needed to flatten AFTER biomass already had.
# Small/near-zero everywhere = gas genuinely is the fast variable here, as
# expected at realistic Vmax. Large or growing toward the high-Vmax corner =
# the fast/slow separation is breaking down there (consistent with growth
# timescale approaching the internal diffusive timescale at extreme Vmax).
# NaN (blank) cells are members that were POC-exhausted or inconclusive —
# this lag is only meaningful for genuine Steady State points.
plt.figure(figsize=(9, 7))
cp14 = plt.contourf(X, Y, results_o2_lag, levels=30, cmap='cividis')
cbar14 = plt.colorbar(cp14)
cbar14.set_label('$O_2$ Lag Behind Biomass Convergence (s)', fontsize=12)
plt.title('Parameter Sweep: How Long $O_2$ Took to Stabilize After Biomass Did', fontsize=14, fontweight='bold')
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.tight_layout()
plt.savefig(f'run_amplification_o2_lag{CAP_TAG}.png', dpi=300)

# ── 13. Plotting: B_eq vs Mortality Multiplier (only if mort axis swept) ────
if HAS_MORT_AXIS:
    mort_vals = x_vals if x_name == 'mort' else y_vals
    other_vals = y_vals if x_name == 'mort' else x_vals
    other_label = y_label if x_name == 'mort' else x_label

    b_eq_sim_masked = np.where(results_stop_reason == STOP_STEADY_STATE, results_b_eq_sim, np.nan)

    plt.figure(figsize=(9, 7))
    cmap_lines = plt.get_cmap('viridis', BATCH_SIZE)
    for k in range(BATCH_SIZE):
        if x_name == 'mort':
            sim_line   = b_eq_sim_masked[k, :]
            theory_line = results_b_eq_theory[k, :]
        else:
            sim_line   = b_eq_sim_masked[:, k]
            theory_line = results_b_eq_theory[:, k]
        color = cmap_lines(k)
        plt.plot(mort_vals, sim_line, marker='o', color=color,
                 label=f'{other_label}={other_vals[k]:.2g} (sim)')
        plt.plot(mort_vals, theory_line, linestyle='--', color=color, alpha=0.6,
                 label=f'{other_label}={other_vals[k]:.2g} (theory)')

    plt.title('$B_{eq}$ vs. Mortality Multiplier (simulated vs. analytic balance)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Mortality Multiplier', fontsize=12)
    plt.ylabel('$B_{eq}$ (mmol C/m³)', fontsize=12)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(f'run_amplification_b_eq_vs_mortality{CAP_TAG}.png', dpi=300)
else:
    print("SWEEP_MODE has no mortality axis — skipping B_eq vs Mortality Multiplier plot.")

print("Sweep complete. Outputs saved.")