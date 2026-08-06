# loop.py
import torch
import numpy as np
import time as _time
import os
from bcs import apply_bcs, enforce_symmetry
from physics import get_psi_pert, get_rhs_batched, apply_implicit_visc




try:
    from progress import progress
except ImportError:
    from tqdm import tqdm as progress

class MemmapSnapshotWriter:
    """
    Drop-in replacement for a plain list of 2D snapshot arrays. Instead of
    accumulating frames in RAM forever (the actual cause of your OOM on
    long runs), each frame is written straight into a disk-backed memmap.
    Supports .append(frame) just like a list, so nothing else in the loop
    needs to change. After the run, .finalize() returns a read-only memmap
    that plotting.py can index/slice exactly like the old lists — data is
    read from disk lazily, not held fully in memory.
    """
    def __init__(self, name, n_max, cache_dir):
        self.name  = name
        self.n_max = n_max
        self.path  = os.path.join(cache_dir, f"{name}.dat")
        self.mmap  = None
        self.count = 0
        self.shape = None

    def append(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if self.mmap is None:
            self.shape = frame.shape
            self.mmap = np.memmap(self.path, dtype=np.float32, mode='w+',
                                   shape=(self.n_max, *self.shape))
        self.mmap[self.count] = frame
        self.count += 1
        if self.count % 20 == 0:   # flush every 20 frames — tune if still climbing
            self.mmap.flush()

    def finalize(self):
        if self.mmap is None:
            return []
        self.mmap.flush()
        del self.mmap
        # Reopen read-only, trimmed to the frames actually written
        # (n_max was just a generous upper bound, not the real count).
        return np.memmap(self.path, dtype=np.float32, mode='r',
                          shape=(self.count, *self.shape))

def run_simulation(state, cfg, device):
    """
    Main SSP-RK3 loop with IMEX drag + viscosity.

    POC → DOC hydrolysis is biomass-driven and saturating:
        doc_flux = k_hyd_max * total_heterotroph_biomass * POC / (K_POC + POC)
    This is now computed internally inside get_rhs_batched() every stage,
    using the current POC tracer field and heterotroph biomass.

    ── Flow-freeze optimization ──────────────────────────────────────────────
    The vorticity/streamfunction RHS depends only on w and static fields
    (drag_mask, psi_bg) — never on the tracers/biology. So once w stops
    changing, psi_tot is fixed forever and it's exact (not an approximation)
    to stop re-solving the Poisson/Helmholtz problems every stage and just
    reuse the cached psi_tot for the remaining tracer integration. This does
    NOT change is_suite logic, timestepping, or the RK3 tracer update itself.
    """
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

    # ── Flow-freeze config (all additive, safe defaults) ─────────────────────
    FLOW_FREEZE_ENABLED     = getattr(cfg, 'flow_freeze_enabled', True)
    FLOW_FREEZE_TOL         = getattr(cfg, 'flow_freeze_tol', 3e-6)
    FLOW_FREEZE_CHECK_EVERY = max(1, int(getattr(cfg, 'flow_freeze_check_every', 200)))
    FLOW_FREEZE_MIN_CONSEC  = max(1, int(getattr(cfg, 'flow_freeze_min_consecutive', 2)))
    FLOW_FREEZE_CKPT_PATH   = getattr(cfg, 'flow_freeze_ckpt_path', 'flow_freeze_checkpoint.pt')

    flow_frozen        = False
    frozen_psi_tot      = None
    w_prev_check        = w.clone()
    w_started           = False   # guards against freezing while w is still at ~0 (pre-startup)
    consec_below_tol    = 0

    # ── Snapshot writers (disk-backed memmaps, not RAM lists) ────────────────
    SNAPSHOT_CACHE_DIR = getattr(cfg, 'snapshot_cache_dir', 'snapshot_cache')
    os.makedirs(SNAPSHOT_CACHE_DIR, exist_ok=True)
    n_snap_max = (int(cfg.Total_Time / dt) // max(1, int(getattr(cfg, 'snapshot_time', 1.0) / dt))) + 2

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

    bio_snapshots = {
        name: MemmapSnapshotWriter(f'bio_{name}', n_snap_max, SNAPSHOT_CACHE_DIR)
        for name in bio_tracers
    }
    growth_rate_snapshots = {
        name: MemmapSnapshotWriter(f'mu_{name}', n_snap_max, SNAPSHOT_CACHE_DIR)
        for name in bio_tracers
    }

    n_steps           = int(cfg.Total_Time / dt)
    snapshot_interval = max(1, int(getattr(cfg, 'snapshot_time', 1.0) / dt))
    CHECK_INTERVAL    = max(1000, n_steps // 200)

    print(f"Total steps: {n_steps}  |  Snapshot every {snapshot_interval} steps")
    print(f"Safety check every {CHECK_INTERVAL} steps")
    if FLOW_FREEZE_ENABLED:
        print(f"Flow-freeze: checking every {FLOW_FREEZE_CHECK_EVERY} steps, "
              f"tol={FLOW_FREEZE_TOL:.1e}, needs {FLOW_FREEZE_MIN_CONSEC} consecutive passes")
    print("Starting SSP-RK3 loop (IMEX drag + viscosity, POC-driven DOC)...")
    _loop_start = _time.perf_counter()

    for n in progress(range(n_steps),
                      desc='Simulating..',
                      ascii='⡀⡄⡆⡇▞▚░▒▓',
                      unit='steps',
                      bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} '
                                 '[{elapsed}<{remaining}, {rate_fmt}]'):

        current_time = n * dt

        # ── SSP-RK3 Stage 1 ───────────────────────────────────────────────────
        if flow_frozen:
            psi_tot = frozen_psi_tot
        else:
            psi_pert = get_psi_pert(w, state)
            psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w, tracers, psi_tot, state, cfg)

        if flow_frozen:
            w1_temp = w
        else:
            w1_temp = (w + dt * rhs_w) * impl_drag['s1']
            w1_temp = apply_implicit_visc(w1_temp, helm_denom['s1'], state)
        t1_temp = {n_: tracers[n_] + dt * rhs_tracers[n_] for n_ in tracer_names}
        w1, t1  = apply_bcs(w1_temp, t1_temp)

        # ── SSP-RK3 Stage 2 ───────────────────────────────────────────────────
        if flow_frozen:
            psi_tot = frozen_psi_tot
        else:
            psi_pert = get_psi_pert(w1, state)
            psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w1, t1, psi_tot, state, cfg)

        if flow_frozen:
            w2_temp = w
        else:
            w2_temp = (0.75 * w + 0.25 * (w1 + dt * rhs_w)) * impl_drag['s2']
            w2_temp = apply_implicit_visc(w2_temp, helm_denom['s2'], state)
        t2_temp = {n_: 0.75 * tracers[n_] + 0.25 * (t1[n_] + dt * rhs_tracers[n_])
                   for n_ in tracer_names}
        w2, t2  = apply_bcs(w2_temp, t2_temp)

        # ── SSP-RK3 Stage 3 ───────────────────────────────────────────────────
        if flow_frozen:
            psi_tot = frozen_psi_tot
        else:
            psi_pert = get_psi_pert(w2, state)
            psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w2, t2, psi_tot, state, cfg)

        if flow_frozen:
            w_temp = w
        else:
            w_temp = ((1.0 / 3.0) * w + (2.0 / 3.0) * (w2 + dt * rhs_w)) * impl_drag['s3']
            w_temp = apply_implicit_visc(w_temp, helm_denom['s3'], state)
        t_temp = {n_: (1.0 / 3.0) * tracers[n_] + (2.0 / 3.0) * (t2[n_] + dt * rhs_tracers[n_])
                  for n_ in tracer_names}
        w, tracers = apply_bcs(w_temp, t_temp)

        if getattr(cfg, 'use_symmetry', True):
            w, tracers = enforce_symmetry(w, tracers, tracer_names)

        # ── Velocity field (only needed for non-suite snapshots/animation) ──────
        if not is_suite:
            u_full.zero_()
            v_full.zero_()
            u_full[..., :, 1:-1] = (psi_tot[..., :, 2:] - psi_tot[..., :, :-2]) * state['inv_2dy']
            v_full[..., 1:-1, :] = -(psi_tot[..., 2:, :] - psi_tot[..., :-2, :]) * state['inv_2dx']

        # ── Flow-freeze detection ────────────────────────────────────────────
        if FLOW_FREEZE_ENABLED and not flow_frozen and n > 0 and n % FLOW_FREEZE_CHECK_EVERY == 0:
            w_max = torch.abs(w).max().item()
            if not w_started:
                # Flow hasn't developed yet (still ~0 from rest) — nothing to freeze.
                if w_max > 1e-6:
                    w_started = True
                w_prev_check = w.clone()
            else:
                rel_change = torch.abs(w - w_prev_check).max().item() / (w_max + 1e-12)
                print(f"  [flow-freeze check] step {n} (t={current_time:.2f}s): "
                      f"Δw_rel={rel_change:.2e} (need <{FLOW_FREEZE_TOL:.1e} "
                      f"× {FLOW_FREEZE_MIN_CONSEC} consecutive)")
                if rel_change < FLOW_FREEZE_TOL:
                    consec_below_tol += 1
                else:
                    consec_below_tol = 0
                w_prev_check = w.clone()

                if consec_below_tol >= FLOW_FREEZE_MIN_CONSEC:
                    psi_pert       = get_psi_pert(w, state)
                    psi_tot_frozen = (psi_pert + psi_bg).clone()

                    print(f"  💾 Saving flow checkpoint to '{FLOW_FREEZE_CKPT_PATH}'...")
                    torch.save({'psi_tot': psi_tot_frozen.cpu(),
                                'w': w.cpu(),
                                'step': n,
                                'time': current_time},
                               FLOW_FREEZE_CKPT_PATH)
                    print(f"  📂 Loading flow checkpoint back from '{FLOW_FREEZE_CKPT_PATH}'...")
                    ckpt = torch.load(FLOW_FREEZE_CKPT_PATH, map_location=device, weights_only=False)
                    frozen_psi_tot = ckpt['psi_tot'].to(device)
                    print(f"  ✅ Checkpoint loaded (saved at step {ckpt['step']}, t={ckpt['time']:.2f}s)")
                    flow_frozen = True

                    print(f"\n🧊 Flow frozen at step {n} (t={current_time:.2f}s). "
                          f"Checkpoint saved to '{FLOW_FREEZE_CKPT_PATH}'. "
                          f"Skipping Poisson/Helmholtz solves for the remaining "
                          f"{n_steps - n} steps.\n")

        # ── Snapshots ─────────────────────────────────────────────────────────
        if getattr(cfg, 'terminal_snapshot_only', False):
            take_snapshot = (n == n_steps - 1)
        else:
            take_snapshot = (n % snapshot_interval == 0)

        if take_snapshot:
            def snap(t): return (t if is_suite else t[0]).cpu().numpy().astype(np.float32)

            from sms import microbial_sms_omz
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
            snapshot_times.append(current_time)
            if not is_suite:
                w_snapshots.append(snap(w))
                u_snapshots.append(snap(u_full))
                v_snapshots.append(snap(v_full))
            for name in bio_tracers:
                bio_snapshots[name].append(snap(tracers[name]))
                mu_s = gross_growth[name] / (interior[name] + 1e-15)
                mu_d = mu_s * 86400.0 * state['bio_accel']
                growth_rate_snapshots[name].append(snap(mu_d))

        # ── Periodic maintenance ───────────────────────────────────────────────
        if n % CHECK_INTERVAL == 0:
            dev_type = getattr(device, 'type', str(device))
            if dev_type == 'mps' and hasattr(torch, 'mps'):
                torch.mps.empty_cache()
            elif dev_type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not torch.isfinite(w).all().item():
                print(f'\n🚨 FATAL: NaN/Inf in vorticity at step {n} (t={current_time:.3f}s)')
                break
            if flow_frozen:
                print(f"  ❄️  [frozen mode] step {n}/{n_steps} (t={current_time:.1f}s) — still running fast")

            crashed = False
            for name, tensor in tracers.items():
                if not torch.isfinite(tensor).all().item():
                    print(f'\n🚨 FATAL: NaN/Inf in tracer \'{name}\' at step {n}'
                          f' (t={current_time:.3f}s)')
                    crashed = True
                    break

                t_max = tensor.abs().max().item()
                if t_max > 1e6:   # tune this — your tracers are all mmol/m³, none should ever get near here
                    print(f'\n🚨 FATAL: Tracer \'{name}\' blew up to {t_max:.2e} at step {n}'
                          f' (t={current_time:.3f}s) — numerical instability, not physical')
                    crashed = True
                    break
            if crashed:
                break

    total_elapsed = _time.perf_counter() - _loop_start
    print(f'\nSimulation complete in {total_elapsed:.1f}s.'
          + (f' (Flow frozen at t={snapshot_times[0] if False else ""})' if False else ''))

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