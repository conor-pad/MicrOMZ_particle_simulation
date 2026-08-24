# loop.py
import torch
import numpy as np
import time as _time
import os
from bcs import apply_bcs, enforce_symmetry
from physics import get_psi_pert, get_rhs_batched, apply_implicit_visc, get_rhs_bio_only
from sms import microbial_sms_omz

try:
    from progress import progress
except ImportError:
    from tqdm import tqdm as progress

class MemmapSnapshotWriter:
    """
    Disk-backed memmap writer to prevent RAM Out-Of-Memory crashes.
    """
    def __init__(self, name, n_max, cache_dir):
        self.name  = name
        self.n_max = int(n_max)
        self.path  = os.path.join(cache_dir, f"{name}.dat")
        self.mmap  = None
        self.count = 0
        self.shape = None

    def append(self, frame):
        if self.count >= self.n_max:
            return  # Safety limit to prevent index crashes
            
        frame = np.asarray(frame, dtype=np.float32)
        if self.mmap is None:
            self.shape = frame.shape
            self.mmap = np.memmap(self.path, dtype=np.float32, mode='w+',
                                   shape=(self.n_max, *self.shape))
        self.mmap[self.count] = frame
        self.count += 1
        if self.count % 20 == 0:   
            self.mmap.flush()

    def finalize(self):
        if self.mmap is None:
            return []
        self.mmap.flush()
        del self.mmap
        return np.memmap(self.path, dtype=np.float32, mode='r',
                          shape=(self.count, *self.shape))


@torch.no_grad() # <-- This GUARANTEES PyTorch never builds a memory-leaking autograd graph
def run_simulation(state, cfg, device):
    dt           = state['dt']
    w            = state['w']
    tracers      = state['tracers']
    psi_bg       = state['psi_bg']
    u_full       = state['u_full']
    v_full       = state['v_full']
    tracer_names = state['tracer_names']
    bio_tracers  = state['bio_names']

    impl_drag  = {s: state[f'impl_drag_{s}']  for s in ('s1', 's2', 's3')}
    helm_denom = {s: state[f'helm_denom_{s}'] for s in ('s1', 's2', 's3')}

    is_suite = getattr(cfg, 'is_suite', False)
    bio_skipping = getattr(cfg, 'bio_skipping', True)

    # ── Flow-freeze config ────────────────────────────────────────────────────
    FLOW_FREEZE_ENABLED     = getattr(cfg, 'flow_freeze_enabled', True)
    FLOW_FREEZE_TOL         = getattr(cfg, 'flow_freeze_tol', 3e-5)
    FLOW_FREEZE_CHECK_EVERY = max(1, int(getattr(cfg, 'flow_freeze_check_every', 200)))
    FLOW_FREEZE_MIN_CONSEC  = max(1, int(getattr(cfg, 'flow_freeze_min_consecutive', 2)))
    FLOW_FREEZE_CKPT_PATH   = getattr(cfg, 'flow_freeze_ckpt_path', 'flow_freeze_checkpoint.pt')

    flow_frozen        = False
    frozen_psi_tot     = None
    w_prev_check       = w.clone()
    w_started          = False   
    consec_below_tol   = 0

    # ── Snapshot writers ──────────────────────────────────────────────────────
    SNAPSHOT_CACHE_DIR = getattr(cfg, 'snapshot_cache_dir', 'snapshot_cache')
    os.makedirs(SNAPSHOT_CACHE_DIR, exist_ok=True)
    
    snapshot_interval_seconds = getattr(cfg, 'snapshot_time', 1.0)
    n_snap_max = (int(cfg.Total_Time / snapshot_interval_seconds)) + 50

    c_snapshots         = MemmapSnapshotWriter('o2',        n_snap_max, SNAPSHOT_CACHE_DIR)
    n2o_snapshots       = MemmapSnapshotWriter('n2o',       n_snap_max, SNAPSHOT_CACHE_DIR)
    no3_snapshots       = MemmapSnapshotWriter('no3',       n_snap_max, SNAPSHOT_CACHE_DIR)
    no2_snapshots       = MemmapSnapshotWriter('no2',       n_snap_max, SNAPSHOT_CACHE_DIR)
    n2_snapshots        = MemmapSnapshotWriter('n2',        n_snap_max, SNAPSHOT_CACHE_DIR)
    nh4_snapshots       = MemmapSnapshotWriter('nh4',       n_snap_max, SNAPSHOT_CACHE_DIR)
    doc_snapshots       = MemmapSnapshotWriter('doc',       n_snap_max, SNAPSHOT_CACHE_DIR)
    w_snapshots         = MemmapSnapshotWriter('w',         n_snap_max, SNAPSHOT_CACHE_DIR)
    u_snapshots         = MemmapSnapshotWriter('u',         n_snap_max, SNAPSHOT_CACHE_DIR)
    v_snapshots         = MemmapSnapshotWriter('v',         n_snap_max, SNAPSHOT_CACHE_DIR)
    n2o_ammox_snapshots = MemmapSnapshotWriter('n2o_ammox', n_snap_max, SNAPSHOT_CACHE_DIR)
    n2o_denit_snapshots = MemmapSnapshotWriter('n2o_denit', n_snap_max, SNAPSHOT_CACHE_DIR)
    snapshot_times      = []

    bio_snapshots = {name: MemmapSnapshotWriter(f'bio_{name}', n_snap_max, SNAPSHOT_CACHE_DIR) for name in bio_tracers}
    growth_rate_snapshots = {name: MemmapSnapshotWriter(f'mu_{name}', n_snap_max, SNAPSHOT_CACHE_DIR) for name in bio_tracers}

    last_snapshot_time = -snapshot_interval_seconds
    current_time = 0.0
    
    MACRO_CYCLE_TIME = getattr(cfg, 'macro_cycle_time', 10.0)
    FLUSH_TIME       = getattr(cfg, 'intermittent_physics_flush_time', 10.0)       

    # ── Helper to avoid massive duplicate code blocks ─────────────────────────
    def save_snapshots(curr_t):
        def snap(t): return (t if is_suite else t[0]).cpu().numpy().astype(np.float32)
        interior = {n: tracers[n][..., 1:-1, 1:-1] for n in tracer_names}
        _, gross_growth = microbial_sms_omz(interior, state['bgc'])

        c_snapshots.append(snap(tracers['o2']))
        n2o_snapshots.append(snap(tracers['n2o']))
        n2o_ammox_snapshots.append(snap(tracers['n2o_ammox']))
        n2o_denit_snapshots.append(snap(tracers['n2o_denit']))
        no3_snapshots.append(snap(tracers['no3']))
        no2_snapshots.append(snap(tracers['no2']))
        n2_snapshots.append(snap(tracers['n2']))
        nh4_snapshots.append(snap(tracers['nh4']))
        doc_snapshots.append(snap(tracers['doc']))
        snapshot_times.append(curr_t)
        
        if not is_suite:
            w_snapshots.append(snap(w))
            u_snapshots.append(snap(u_full))
            v_snapshots.append(snap(v_full))
            
        for name_ in bio_tracers:
            bio_snapshots[name_].append(snap(tracers[name_]))
            mu_s = gross_growth[name_] / (interior[name_] + 1e-15)
            mu_d = mu_s * 86400.0 * state['bio_accel']
            growth_rate_snapshots[name_].append(snap(mu_d))

    print("\nStarting Phase 1: Synchronous Spin-Up Loop...")
    _loop_start = _time.perf_counter()

    n = 0
    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: SYNCHRONOUS SPIN-UP
    # ══════════════════════════════════════════════════════════════════════════
    while current_time < cfg.Total_Time and not flow_frozen:
        
        psi_pert = get_psi_pert(w, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w, tracers, psi_tot, state, cfg, compute_bio=True)

        w1_temp = (w + dt * rhs_w) * impl_drag['s1']
        w1_temp = apply_implicit_visc(w1_temp, helm_denom['s1'], state)
        t1_temp = {n_: tracers[n_] + dt * rhs_tracers[n_] for n_ in tracer_names}
        w1, t1  = apply_bcs(w1_temp, t1_temp)

        psi_pert = get_psi_pert(w1, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w1, t1, psi_tot, state, cfg, compute_bio=True)

        w2_temp = (0.75 * w + 0.25 * (w1 + dt * rhs_w)) * impl_drag['s2']
        w2_temp = apply_implicit_visc(w2_temp, helm_denom['s2'], state)
        t2_temp = {n_: 0.75 * tracers[n_] + 0.25 * (t1[n_] + dt * rhs_tracers[n_]) for n_ in tracer_names}
        w2, t2  = apply_bcs(w2_temp, t2_temp)

        psi_pert = get_psi_pert(w2, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w2, t2, psi_tot, state, cfg, compute_bio=True)

        w_temp = ((1.0 / 3.0) * w + (2.0 / 3.0) * (w2 + dt * rhs_w)) * impl_drag['s3']
        w_temp = apply_implicit_visc(w_temp, helm_denom['s3'], state)
        t_temp = {n_: (1.0 / 3.0) * tracers[n_] + (2.0 / 3.0) * (t2[n_] + dt * rhs_tracers[n_]) for n_ in tracer_names}
        w, tracers = apply_bcs(w_temp, t_temp)

        if getattr(cfg, 'use_symmetry', True):
            w, tracers = enforce_symmetry(w, tracers, tracer_names)

        u_full.zero_()
        v_full.zero_()
        u_full[..., :, 1:-1] = (psi_tot[..., :, 2:] - psi_tot[..., :, :-2]) * state['inv_2dy']
        v_full[..., 1:-1, :] = -(psi_tot[..., 2:, :] - psi_tot[..., :-2, :]) * state['inv_2dx']

        current_time += dt
        n += 1

        # Periodic GPU Cache Flush
        if n % 100 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if FLOW_FREEZE_ENABLED and not flow_frozen and n > 0 and n % FLOW_FREEZE_CHECK_EVERY == 0:
            w_max = torch.abs(w).max().item()
            if w_max < 0.01:
                w_prev_check = w.clone()
            elif not w_started:
                w_started = True
                w_prev_check = w.clone()
            else:
                rel_change = torch.abs(w - w_prev_check).max().item() / (w_max + 1e-12)
                print(f"  [spin-up flow-freeze check] step {n} (t={current_time:.2f}s): "
                      f"Δw_rel={rel_change:.2e} (need <{FLOW_FREEZE_TOL:.1e} × {FLOW_FREEZE_MIN_CONSEC} consecutive)")
                if rel_change < FLOW_FREEZE_TOL:
                    consec_below_tol += 1
                else:
                    consec_below_tol = 0
                w_prev_check = w.clone()

                if consec_below_tol >= FLOW_FREEZE_MIN_CONSEC:
                    psi_pert       = get_psi_pert(w, state)
                    frozen_psi_tot = (psi_pert + psi_bg).clone()
                    torch.save({'psi_tot': frozen_psi_tot.cpu(), 'w': w.cpu(), 'step': n, 'time': current_time}, FLOW_FREEZE_CKPT_PATH)
                    flow_frozen = True
                    print("\nflow has frozen. doing bio skipping now\n" if bio_skipping else "\nflow has frozen. continuing fully-coupled...\n")

        if (current_time - last_snapshot_time) >= snapshot_interval_seconds:
            save_snapshots(current_time)
            last_snapshot_time = current_time

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: POST-FREEZE LOOP
    # ══════════════════════════════════════════════════════════════════════════
    do_flush = getattr(cfg, 'extended_physics_flush_at_end', False)
    final_flush_duration = cfg.final_flush_duration if do_flush else 0.0
    bio_skip_end_time = cfg.Total_Time - final_flush_duration

    if bio_skipping and current_time < bio_skip_end_time:
        print(f"Starting Phase 2a: Asynchronous Bio-Skipping Loop (until t={bio_skip_end_time}s)...")

        while current_time < bio_skip_end_time:
            
            # ── Step A: Asynchronous Biology Fast-Forward ──
            t_bio_elapsed = 0.0
            while t_bio_elapsed < MACRO_CYCLE_TIME and current_time < bio_skip_end_time:
                rhs_bio, gross_growth = get_rhs_bio_only(tracers, state, cfg)
                
                dt_max = min(MACRO_CYCLE_TIME - t_bio_elapsed, bio_skip_end_time - current_time)
                for name, rhs in rhs_bio.items():
                    interior = tracers[name][..., 1:-1, 1:-1]
                    neg_mask = rhs < -1e-10
                    if neg_mask.any():
                        safe_dt = (interior[neg_mask] / -rhs[neg_mask]).min().item() * 0.69
                        dt_max = min(dt_max, safe_dt)
                
                dt_bio = max(dt_max, 1e-3) 
                
                for name in tracer_names:
                    tracers[name][..., 1:-1, 1:-1] += rhs_bio[name] * dt_bio
                    tracers[name] = torch.clamp(tracers[name], min=0.0)
                    
                t_bio_elapsed += dt_bio
                current_time += dt_bio

                # Snapshot correctly handles Bio-phases now
                if (current_time - last_snapshot_time) >= snapshot_interval_seconds:
                    save_snapshots(current_time)
                    last_snapshot_time = current_time

                remaining = cfg.Total_Time - current_time
                print(f"  [Bio Fast-Forward] Took dt = {dt_bio:.2f} s | Bio cycle elapsed: {t_bio_elapsed:.2f} / {MACRO_CYCLE_TIME} s | Remaining: {remaining:.1f} s")
                
            # ── Step B: Synchronous Physics Flush ──
            flush_steps = max(1, int(FLUSH_TIME / dt))
            for f_step in range(flush_steps):
                psi_tot = frozen_psi_tot
                    
                rhs_w, rhs_tracers = get_rhs_batched(w, tracers, psi_tot, state, cfg, compute_bio=True)

                w1_temp = (w + dt * rhs_w) * impl_drag['s1']
                w1_temp = apply_implicit_visc(w1_temp, helm_denom['s1'], state)
                t1_temp = {n_: tracers[n_] + dt * rhs_tracers[n_] for n_ in tracer_names}
                w1, t1  = apply_bcs(w1_temp, t1_temp)

                rhs_w, rhs_tracers = get_rhs_batched(w1, t1, psi_tot, state, cfg, compute_bio=True)

                w2_temp = (0.75 * w + 0.25 * (w1 + dt * rhs_w)) * impl_drag['s2']
                w2_temp = apply_implicit_visc(w2_temp, helm_denom['s2'], state)
                t2_temp = {n_: 0.75 * tracers[n_] + 0.25 * (t1[n_] + dt * rhs_tracers[n_]) for n_ in tracer_names}
                w2, t2  = apply_bcs(w2_temp, t2_temp)

                rhs_w, rhs_tracers = get_rhs_batched(w2, t2, psi_tot, state, cfg, compute_bio=True)

                w_temp = ((1.0 / 3.0) * w + (2.0 / 3.0) * (w2 + dt * rhs_w)) * impl_drag['s3']
                w_temp = apply_implicit_visc(w_temp, helm_denom['s3'], state)
                t_temp = {n_: (1.0 / 3.0) * tracers[n_] + (2.0 / 3.0) * (t2[n_] + dt * rhs_tracers[n_]) for n_ in tracer_names}
                w, tracers = apply_bcs(w_temp, t_temp)

                if getattr(cfg, 'use_symmetry', True):
                    w, tracers = enforce_symmetry(w, tracers, tracer_names)

                if not is_suite:
                    u_full.zero_()
                    v_full.zero_()
                    u_full[..., :, 1:-1] = (psi_tot[..., :, 2:] - psi_tot[..., :, :-2]) * state['inv_2dy']
                    v_full[..., 1:-1, :] = -(psi_tot[..., 2:, :] - psi_tot[..., :-2, :]) * state['inv_2dx']

                current_time += dt

                # Keep RAM flat during flush
                if f_step % 1000 == 0 and torch.backends.mps.is_available():
                    
                    torch.mps.empty_cache()
                    print(f"  [Sync Loop] step {f_step} | t = {current_time:.2f}s / {cfg.Total_Time}s | Remaining: {cfg.Total_Time - current_time:.2f}s")
                # Snapshots actually save during the washout now
                if (current_time - last_snapshot_time) >= snapshot_interval_seconds:
                    save_snapshots(current_time)
                    last_snapshot_time = current_time

    # ── Phase 2b: The Synchronous Finish ──
    if current_time < cfg.Total_Time:
        if bio_skipping and do_flush:
            print(f"\nStarting Phase 2b: Extended Physics Flush (Final {final_flush_duration}s Fully-Coupled)...")
        else:
            print("\nStarting Phase 2: Fully-Coupled Synchronous Loop (Frozen Flow)...")

        f_step = 0
        while current_time < cfg.Total_Time:
            psi_tot = frozen_psi_tot

            rhs_w, rhs_tracers = get_rhs_batched(w, tracers, psi_tot, state, cfg, compute_bio=True)

            w1_temp = (w + dt * rhs_w) * impl_drag['s1']
            w1_temp = apply_implicit_visc(w1_temp, helm_denom['s1'], state)
            t1_temp = {n_: tracers[n_] + dt * rhs_tracers[n_] for n_ in tracer_names}
            w1, t1  = apply_bcs(w1_temp, t1_temp)

            rhs_w, rhs_tracers = get_rhs_batched(w1, t1, psi_tot, state, cfg, compute_bio=True)

            w2_temp = (0.75 * w + 0.25 * (w1 + dt * rhs_w)) * impl_drag['s2']
            w2_temp = apply_implicit_visc(w2_temp, helm_denom['s2'], state)
            t2_temp = {n_: 0.75 * tracers[n_] + 0.25 * (t1[n_] + dt * rhs_tracers[n_]) for n_ in tracer_names}
            w2, t2  = apply_bcs(w2_temp, t2_temp)

            rhs_w, rhs_tracers = get_rhs_batched(w2, t2, psi_tot, state, cfg, compute_bio=True)

            w_temp = ((1.0 / 3.0) * w + (2.0 / 3.0) * (w2 + dt * rhs_w)) * impl_drag['s3']
            w_temp = apply_implicit_visc(w_temp, helm_denom['s3'], state)
            t_temp = {n_: (1.0 / 3.0) * tracers[n_] + (2.0 / 3.0) * (t2[n_] + dt * rhs_tracers[n_]) for n_ in tracer_names}
            w, tracers = apply_bcs(w_temp, t_temp)

            if getattr(cfg, 'use_symmetry', True):
                w, tracers = enforce_symmetry(w, tracers, tracer_names)

            if not is_suite:
                u_full.zero_()
                v_full.zero_()
                u_full[..., :, 1:-1] = (psi_tot[..., :, 2:] - psi_tot[..., :, :-2]) * state['inv_2dy']
                v_full[..., 1:-1, :] = -(psi_tot[..., 2:, :] - psi_tot[..., :-2, :]) * state['inv_2dx']

            current_time += dt
            f_step += 1

            # Cache clearing
            if f_step % 1000 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print(f"  [Sync Loop] step {f_step} | t = {current_time:.2f}s / {cfg.Total_Time}s | Remaining: {cfg.Total_Time - current_time:.2f}s")

            if (current_time - last_snapshot_time) >= snapshot_interval_seconds:
                save_snapshots(current_time)
                last_snapshot_time = current_time

    total_elapsed = _time.perf_counter() - _loop_start
    print(f'\nSimulation complete in {total_elapsed/60:.1f} mins.')

    c_snapshots         = c_snapshots.finalize()
    n2o_snapshots       = n2o_snapshots.finalize()
    no3_snapshots       = no3_snapshots.finalize()
    no2_snapshots       = no2_snapshots.finalize()
    n2_snapshots        = n2_snapshots.finalize()
    nh4_snapshots       = nh4_snapshots.finalize()
    doc_snapshots       = doc_snapshots.finalize()
    w_snapshots         = w_snapshots.finalize()
    u_snapshots         = u_snapshots.finalize()
    v_snapshots         = v_snapshots.finalize()
    n2o_ammox_snapshots = n2o_ammox_snapshots.finalize()
    n2o_denit_snapshots = n2o_denit_snapshots.finalize()
    bio_snapshots         = {k: v.finalize() for k, v in bio_snapshots.items()}
    growth_rate_snapshots = {k: v.finalize() for k, v in growth_rate_snapshots.items()}

    return (c_snapshots, n2o_snapshots, no3_snapshots, no2_snapshots,
            n2_snapshots, doc_snapshots, nh4_snapshots,
            w_snapshots, u_snapshots, v_snapshots,
            snapshot_times,
            n2o_ammox_snapshots, n2o_denit_snapshots,
            bio_snapshots, growth_rate_snapshots)