# loop.py
import torch
import numpy as np
import time as _time
from tqdm import tqdm

from bcs import apply_bcs, enforce_symmetry
from physics import get_psi_pert, get_rhs_batched

def run_simulation(state, cfg, device):

    # ── torch.no_grad() ───────────────────────────────────────────────────────
    # The simulation never backpropagates. Disabling gradient tracking removes
    # all autograd overhead from every tensor operation in the loop.
    with torch.no_grad():

        dt           = state['dt']
        w            = state['w']
        tracers      = state['tracers']
        psi_bg       = state['psi_bg']
        u_full       = state['u_full']
        v_full       = state['v_full']
        tracer_names = state['tracer_names']

        w_snapshots, c_snapshots, n2o_snapshots = [], [], []
        no3_snapshots, no2_snapshots, n2_snapshots = [], [], []
        nh4_snapshots, doc_snapshots = [], []
        u_snapshots, v_snapshots = [], []
        snapshot_times = []
        n2o_ammox_snapshots, n2o_denit_snapshots = [], []

        bio_tracers   = ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox', 'zoo']
        bio_snapshots = {name: [] for name in bio_tracers}

        n_steps           = int(cfg.Total_Time / dt)
        snapshot_interval = max(1, int(0.03 / dt))

        # How often to run safety checks and cache flushes.
        # These cause a GPU→CPU sync so keep them infrequent.
        CHECK_INTERVAL = max(1000, n_steps // 200)   # ~0.5% of total steps

        print(f"Total steps: {n_steps}  |  Snapshot every {snapshot_interval} steps")
        print(f"Safety check / cache flush every {CHECK_INTERVAL} steps")
        print("Starting 8-Tracer SSP-RK3 Loop on M2 GPU...")
        _loop_start = _time.perf_counter()

        # ── Main Loop ─────────────────────────────────────────────────────────
        for n in tqdm(range(n_steps),
                      desc="Simulating..",
                      ascii="⡀⡄⡆⡇▞▚░▒▓",
                      unit="steps",
                      bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):

            current_time = n * dt

            # ── STAGE 1 ───────────────────────────────────────────────────────
            psi_pert = get_psi_pert(w, state)
            psi_tot  = psi_pert + psi_bg
            rhs_w, rhs_tracers = get_rhs_batched(w, tracers, psi_tot, state, cfg)

            w1_temp = w + dt * rhs_w
            t1_temp = {name: tracers[name] + dt * rhs_tracers[name] for name in tracer_names}
            w1, t1  = apply_bcs(w1_temp, t1_temp)

            # ── STAGE 2 ───────────────────────────────────────────────────────
            psi_pert = get_psi_pert(w1, state)
            psi_tot  = psi_pert + psi_bg
            rhs_w, rhs_tracers = get_rhs_batched(w1, t1, psi_tot, state, cfg)

            w2_temp = 0.75 * w + 0.25 * (w1 + dt * rhs_w)
            t2_temp = {name: 0.75 * tracers[name] + 0.25 * (t1[name] + dt * rhs_tracers[name])
                       for name in tracer_names}
            w2, t2  = apply_bcs(w2_temp, t2_temp)

            # ── STAGE 3 ───────────────────────────────────────────────────────
            psi_pert = get_psi_pert(w2, state)
            psi_tot  = psi_pert + psi_bg
            rhs_w, rhs_tracers = get_rhs_batched(w2, t2, psi_tot, state, cfg)

            w_temp = (1.0/3.0) * w + (2.0/3.0) * (w2 + dt * rhs_w)
            t_temp = {name: (1.0/3.0) * tracers[name] + (2.0/3.0) * (t2[name] + dt * rhs_tracers[name])
                      for name in tracer_names}
            w, tracers = apply_bcs(w_temp, t_temp)

            if getattr(cfg, 'use_symmetry', True):
                w, tracers = enforce_symmetry(w, tracers, tracer_names)

            u_full.zero_()
            v_full.zero_()
            u_full[:, 1:-1] = (psi_tot[:, 2:] - psi_tot[:, :-2]) * state['inv_2dy']
            v_full[1:-1, :] = -(psi_tot[2:, :] - psi_tot[:-2, :]) * state['inv_2dx']

            # ── Snapshots ─────────────────────────────────────────────────────
            if getattr(cfg, 'is_suite', False):
                take_snapshot = (n == n_steps - 1)
            else:
                take_snapshot = (n % snapshot_interval == 0)

            if take_snapshot:
                c_snapshots.append(tracers['o2'].cpu().numpy())
                n2o_snapshots.append(tracers['n2o'].cpu().numpy())
                n2o_ammox_snapshots.append(tracers['n2o_ammox'].cpu().numpy())
                n2o_denit_snapshots.append(tracers['n2o_denit'].cpu().numpy())
                no3_snapshots.append(tracers['no3'].cpu().numpy())
                no2_snapshots.append(tracers['no2'].cpu().numpy())
                n2_snapshots.append(tracers['n2'].cpu().numpy())
                nh4_snapshots.append(tracers['nh4'].cpu().numpy())
                doc_snapshots.append(tracers['doc'].cpu().numpy())
                w_snapshots.append(w.cpu().numpy())
                u_snapshots.append(u_full.cpu().numpy())
                v_snapshots.append(v_full.cpu().numpy())
                snapshot_times.append(current_time)

                for name in bio_tracers:
                    bio_snapshots[name].append(tracers[name].cpu().numpy())

            # ── Periodic maintenance (infrequent to avoid GPU→CPU stalls) ────
            if n % CHECK_INTERVAL == 0:

                # Flush the MPS allocator cache to keep memory tidy.
                # Kept intentionally infrequent — every call stalls the pipeline.
                torch.mps.empty_cache()

                # NaN / Inf safety kill-switch
                if not torch.isfinite(w).all().item():
                    print(f"\n🚨 FATAL: NaN/Inf in vorticity at step {n} (t={current_time:.3f}s)")
                    break

                for name, tensor in tracers.items():
                    if not torch.isfinite(tensor).all().item():
                        print(f"\n🚨 FATAL: NaN/Inf in tracer '{name}' at step {n} (t={current_time:.3f}s)")
                        break
                else:
                    continue   # inner for-loop completed without break → keep going
                break           # inner loop hit break → exit outer loop too

        # ── End of loop ───────────────────────────────────────────────────────
        total_elapsed = _time.perf_counter() - _loop_start
        print(f"\nSimulation complete in {total_elapsed:.1f}s. Generating animations...")

        return (c_snapshots, n2o_snapshots, no3_snapshots, no2_snapshots,
                n2_snapshots, doc_snapshots, nh4_snapshots, w_snapshots,
                u_snapshots, v_snapshots, snapshot_times,
                n2o_ammox_snapshots, n2o_denit_snapshots, bio_snapshots)