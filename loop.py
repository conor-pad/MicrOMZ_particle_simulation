# loop.py
import torch
import numpy as np
import time as _time
from tqdm import tqdm
from alive_progress import alive_bar

from bcs import apply_bcs, enforce_symmetry
from physics import get_psi_pert, get_rhs_batched

def run_simulation(state, cfg, device):
    dt = state['dt']
    w = state['w']
    tracers = state['tracers'] 
    psi_bg = state['psi_bg']
    u_full = state['u_full']
    v_full = state['v_full']
    tracer_names = state['tracer_names']

    # We only save O2 and N2O to keep memory usage low and because plotting.py expects them
    w_snapshots, c_snapshots, n2o_snapshots, no3_snapshots, no2_snapshots, n2_snapshots, nh4_snapshots, doc_snapshots = [], [], [], [], [], [], [], []
    u_snapshots, v_snapshots = [], []
    snapshot_times = []
    
    # ADDED: Initialize the empty lists for your new trackers
    n2o_ammox_snapshots, n2o_denit_snapshots = [], []

    n_steps = int(cfg.Total_Time / dt)
    snapshot_interval = max(1, int(0.03 / dt))

    print(f"Total steps: {n_steps}  |  Snapshot every {snapshot_interval} steps")
    print("Starting 8-Tracer SSP-RK3 Loop on M2 GPU...")
    _loop_start = _time.perf_counter()

    # ── Main Loop ────────────────────────────────────────────────────────────
    for n in tqdm(range(n_steps),
                  desc="Simulating..",
                  ascii="⡀⡄⡆⡇▞▚░▒▓",
                  unit="steps",
                  bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):

        # with alive_bar(n_steps, title="Simulating..", bar="smooth", spinner="waves", length=40) as bar:
    
        # for n in range(n_steps): # remove if using tqdm, indent everything back 1.

        current_time = n * dt

        # ── STAGE 1 ──────────────────────────────────────────────────────────
        psi_pert = get_psi_pert(w, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w, tracers, psi_tot, state, cfg)
        
        # Dictionary comprehension to apply Euler step to all 8 tracers instantly
        w1_temp = w + dt * rhs_w
        t1_temp = {name: tracers[name] + dt * rhs_tracers[name] for name in tracer_names}
        w1, t1  = apply_bcs(w1_temp, t1_temp)

        # ── STAGE 2 ──────────────────────────────────────────────────────────
        psi_pert = get_psi_pert(w1, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w1, t1, psi_tot, state, cfg)
        
        w2_temp = 0.75 * w + 0.25 * (w1 + dt * rhs_w)
        t2_temp = {name: 0.75 * tracers[name] + 0.25 * (t1[name] + dt * rhs_tracers[name]) for name in tracer_names}
        w2, t2  = apply_bcs(w2_temp, t2_temp)

        # ── STAGE 3 ──────────────────────────────────────────────────────────
        psi_pert = get_psi_pert(w2, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w2, t2, psi_tot, state, cfg)
        
        w_temp = (1/3) * w + (2/3) * (w2 + dt * rhs_w)
        t_temp = {name: (1/3) * tracers[name] + (2/3) * (t2[name] + dt * rhs_tracers[name]) for name in tracer_names}
        w, tracers = apply_bcs(w_temp, t_temp)

        w, tracers = enforce_symmetry(w, tracers, tracer_names)

        u_full.zero_()
        v_full.zero_()
        u_full[:, 1:-1] = (psi_tot[:, 2:] - psi_tot[:, :-2]) * state['inv_2dy']
        v_full[1:-1, :] = -(psi_tot[2:, :] - psi_tot[:-2, :]) * state['inv_2dx']

        # ── Snapshots ────────────────────────────────────────────────────────
        # 1. RAM Fix: Only save the very last frame if running the suite
        if getattr(cfg, 'is_suite', False):
            take_snapshot = (n == n_steps - 1)
        else:
            take_snapshot = (n % snapshot_interval == 0)

        if take_snapshot:
            # We extract the tensors to pass to plotting.py
            c_snapshots.append(tracers['o2'].cpu().numpy().astype(np.float32))
            n2o_snapshots.append(tracers['n2o'].cpu().numpy().astype(np.float32))
            n2o_ammox_snapshots.append(tracers['n2o_ammox'].cpu().numpy().astype(np.float32))
            n2o_denit_snapshots.append(tracers['n2o_denit'].cpu().numpy().astype(np.float32))

            no3_snapshots.append(tracers['no3'].cpu().numpy().astype(np.float32))
            no2_snapshots.append(tracers['no2'].cpu().numpy().astype(np.float32))
            n2_snapshots.append(tracers['n2'].cpu().numpy().astype(np.float32))
            nh4_snapshots.append(tracers['nh4'].cpu().numpy().astype(np.float32))
            doc_snapshots.append(tracers['doc'].cpu().numpy().astype(np.float32))

            w_snapshots.append(w.cpu().numpy().astype(np.float32))
            u_snapshots.append(u_full.cpu().numpy().astype(np.float32))
            v_snapshots.append(v_full.cpu().numpy().astype(np.float32))
            snapshot_times.append(current_time)

        # 2. Garbage Collection Fix: Run this independently of the snapshots!
        if n % 1000 == 0:
            torch.mps.empty_cache()

            # bar()

        # ── 3. SAFETY KILL SWITCH ──
        # Check for NaNs/Infs every 100 steps (prevents GPU sync bottleneck)
        if n % 1000 == 0:
            # 1. Check fluid momentum
            if not torch.isfinite(w).all().item():
                print(f"\n🚨 FATAL ERROR: NaN or Inf detected in fluid vorticity (w) at step {n} (time = {current_time:.3f}s)!")
                break # Kills the loop
            
            # 2. Check all biological tracers
            tracer_crashed = False
            for name, tensor in tracers.items():
                if not torch.isfinite(tensor).all().item():
                    print(f"\n🚨 FATAL ERROR: NaN or Inf detected in tracer '{name}' at step {n} (time = {current_time:.3f}s)!")
                    tracer_crashed = True
                    break
            
            if tracer_crashed:
                break # Kills the outer loop

    total_elapsed = _time.perf_counter() - _loop_start
    print(f"\nSimulation complete in {total_elapsed:.1f}s. Generating animations...")

    return c_snapshots, n2o_snapshots, no3_snapshots, no2_snapshots, n2_snapshots, doc_snapshots, nh4_snapshots, w_snapshots, u_snapshots, v_snapshots, snapshot_times, n2o_ammox_snapshots, n2o_denit_snapshots