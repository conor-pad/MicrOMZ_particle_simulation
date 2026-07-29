# loop.py
import torch
import numpy as np
import time as _time

from bcs import apply_bcs, enforce_symmetry
from physics import get_psi_pert, get_rhs_batched, apply_implicit_visc

try:
    from progress import progress
except ImportError:
    from tqdm import tqdm as progress


def run_simulation(state, cfg, device):
    """
    Main SSP-RK3 loop with IMEX drag + viscosity.

    POC → DOC hydrolysis is biomass-driven and saturating:
        doc_flux = k_hyd_max * total_heterotroph_biomass * POC / (K_POC + POC)
    This is now computed internally inside get_rhs_batched() every stage,
    using the current POC tracer field and heterotroph biomass.
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

    # ── Snapshot lists ────────────────────────────────────────────────────────
    c_snapshots         = []
    n2o_snapshots       = []
    no3_snapshots       = []
    no2_snapshots       = []
    n2_snapshots        = []
    nh4_snapshots       = []
    doc_snapshots       = []
    w_snapshots         = []
    u_snapshots         = []
    v_snapshots         = []
    snapshot_times      = []
    n2o_ammox_snapshots = []
    n2o_denit_snapshots = []
    bio_snapshots       = {name: [] for name in bio_tracers}
    growth_rate_snapshots = {name: [] for name in bio_tracers}

    n_steps           = int(cfg.Total_Time / dt)
    snapshot_interval = max(1, int(getattr(cfg, 'snapshot_time', 1.0) / dt))
    CHECK_INTERVAL    = max(1000, n_steps // 200)

    print(f"Total steps: {n_steps}  |  Snapshot every {snapshot_interval} steps")
    print(f"Safety check every {CHECK_INTERVAL} steps")
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
        psi_pert = get_psi_pert(w, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w, tracers, psi_tot, state, cfg)

        w1_temp = (w + dt * rhs_w) * impl_drag['s1']
        w1_temp = apply_implicit_visc(w1_temp, helm_denom['s1'], state)
        t1_temp = {n_: tracers[n_] + dt * rhs_tracers[n_] for n_ in tracer_names}
        w1, t1  = apply_bcs(w1_temp, t1_temp)

        # ── SSP-RK3 Stage 2 ───────────────────────────────────────────────────
        psi_pert = get_psi_pert(w1, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w1, t1, psi_tot, state, cfg)

        w2_temp = (0.75 * w + 0.25 * (w1 + dt * rhs_w)) * impl_drag['s2']
        w2_temp = apply_implicit_visc(w2_temp, helm_denom['s2'], state)
        t2_temp = {n_: 0.75 * tracers[n_] + 0.25 * (t1[n_] + dt * rhs_tracers[n_])
                   for n_ in tracer_names}
        w2, t2  = apply_bcs(w2_temp, t2_temp)

        # ── SSP-RK3 Stage 3 ───────────────────────────────────────────────────
        psi_pert = get_psi_pert(w2, state)
        psi_tot  = psi_pert + psi_bg
        rhs_w, rhs_tracers = get_rhs_batched(w2, t2, psi_tot, state, cfg)

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

        # ── Snapshots ─────────────────────────────────────────────────────────
        if getattr(cfg, 'terminal_snapshot_only', False):
            take_snapshot = (n == n_steps - 1)
        else:
            take_snapshot = (n % snapshot_interval == 0)

        if take_snapshot:
            # Non-suite (main.py, bs=1): squeeze to a plain 2D array, as before.
            # Suite mode (bs>1 sweeps): keep the full batch dim — squeezing to
            # [0] here was the actual bug silently discarding every batch member
            # except the first, which is why bs>1 never worked in run_suite.py.
            def snap(t): return (t if is_suite else t[0]).cpu().numpy().astype(np.float32)

            # Get instantaneous exact growth rates (kept in suite mode too)
            from sms import microbial_sms_omz
            interior = {n: tracers[n][..., 1:-1, 1:-1] for n in tracer_names}
            _, gross_growth = microbial_sms_omz(interior, state['bgc'])
            # -------------------------------------------------

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
                # u/v aren't rebuilt in suite mode (see above) — nothing to snapshot.
                w_snapshots.append(snap(w))
                u_snapshots.append(snap(u_full))
                v_snapshots.append(snap(v_full))
            for name in bio_tracers:
                bio_snapshots[name].append(snap(tracers[name]))

                # Calculate specific growth rate (1/s) and convert to day⁻¹
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

            crashed = False
            for name, tensor in tracers.items():
                if not torch.isfinite(tensor).all().item():
                    print(f'\n🚨 FATAL: NaN/Inf in tracer \'{name}\' at step {n}'
                          f' (t={current_time:.3f}s)')
                    crashed = True
                    break
            if crashed:
                break

    total_elapsed = _time.perf_counter() - _loop_start
    print(f'\nSimulation complete in {total_elapsed:.1f}s.')

    return (c_snapshots, n2o_snapshots, no3_snapshots, no2_snapshots,
            n2_snapshots, doc_snapshots, nh4_snapshots,
            w_snapshots, u_snapshots, v_snapshots,
            snapshot_times,
            n2o_ammox_snapshots, n2o_denit_snapshots,
            bio_snapshots, growth_rate_snapshots)