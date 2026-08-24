# validate_micromz_physics_2d.py
"""
MicrOMZ physics-only validation script -- SINGLE FILE, no imports between
your own scripts (that was the source of the earlier ImportError).

Runs ONLY the Phase-1 fully-coupled synchronous physics (flow + tracer
transport, no bio-skipping) until the flow-freeze convergence criterion
trips, then produces Kiorboe, Ploug & Thygesen (2001, Mar Ecol Prog Ser
211:1-13) style validation figures on your 2D Cartesian grid:

  Fig 1 analog: streamlines + vorticity across a Re sweep (3 panels)
  Fig 2 analog: velocity magnitude vs radial distance, up/downstream and
                off-equator, for each Re in the sweep (model-only curves --
                panels C-H in the paper are their lab data vs model, which
                doesn't apply since you don't have physical experiments)
  Fig 3 analog: concentration field across a Pe sweep (4 panels)

Fig 4 (modeled vs. observed lab data) and Fig 5 (bulk Sherwood number) are
skipped -- Fig 4 needs real experimental data you don't have, and Fig 5's
domain-size and non-steady-bio caveats make it a separate task.

NOTE ON GEOMETRY: Kiorboe et al. solved a 3D axisymmetric sphere. This is a
2D Cartesian analog -- qualitative patterns (streamline asymmetry, wake
elongation, plume shape) are the right things to check, not exact
quantitative matches to their Re/Pe power-law fits.

Re/Pe sweep mechanics: cfg.U_bg is normally DERIVED from radius via an
empirical scaling law (U_bg = 1.6 * radius**0.56), so Re isn't an
independent knob in a normal run. This script temporarily overrides
cfg.U_bg directly to hit target Re/Pe values, recomputing the dependent
cfg.Re_actual / cfg.Pe_calc before each run. radius, domain size (Lx, Ly),
and everything else are left untouched.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from bcs import apply_bcs, enforce_symmetry
from physics import get_psi_pert, get_rhs_batched, apply_implicit_visc, setup_physics
import config as cfg

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────────
# CORE PHYSICS RUNNER
# ─────────────────────────────────────────────────────────────────────────
def run_physics_to_steady_state(state, cfg, device, tol=None, check_every=None, min_consecutive=None):
    dt = state['dt']
    w = state['w']
    tracers = state['tracers']
    psi_bg = state['psi_bg']
    tracer_names = state['tracer_names']

    impl_drag = {s: state[f'impl_drag_{s}'] for s in ('s1', 's2', 's3')}
    helm_denom = {s: state[f'helm_denom_{s}'] for s in ('s1', 's2', 's3')}

    tol = tol if tol is not None else getattr(cfg, 'flow_freeze_tol', 7e-4)
    check_every = check_every if check_every is not None else max(1, int(getattr(cfg, 'flow_freeze_check_every', 200)))
    min_consecutive = min_consecutive if min_consecutive is not None else max(1, int(getattr(cfg, 'flow_freeze_min_consecutive', 2)))

    w_prev_check = w.clone()
    consec_below_tol = 0
    n = 0
    current_time = 0.0
    
    flow_frozen = False
    frozen_psi_tot = None
    end_time = float('inf')  # Runs indefinitely until flow freezes

    print("Phase 1: Flow convergence...")
    with torch.no_grad():
        while current_time < end_time:
            psi_tot_0 = frozen_psi_tot if flow_frozen else get_psi_pert(w, state) + psi_bg
            rhs_w, rhs_tracers = get_rhs_batched(w, tracers, psi_tot_0, state, cfg, compute_bio=True)
            w1_temp = apply_implicit_visc((w + dt * rhs_w) * impl_drag['s1'], helm_denom['s1'], state)
            t1_temp = {n_: tracers[n_] + dt * rhs_tracers[n_] for n_ in tracer_names}
            w1, t1 = apply_bcs(w1_temp, t1_temp)

            psi_tot_1 = frozen_psi_tot if flow_frozen else get_psi_pert(w1, state) + psi_bg
            rhs_w, rhs_tracers = get_rhs_batched(w1, t1, psi_tot_1, state, cfg, compute_bio=True)
            w2_temp = apply_implicit_visc((0.75 * w + 0.25 * (w1 + dt * rhs_w)) * impl_drag['s2'], helm_denom['s2'], state)
            t2_temp = {n_: 0.75 * tracers[n_] + 0.25 * (t1[n_] + dt * rhs_tracers[n_]) for n_ in tracer_names}
            w2, t2 = apply_bcs(w2_temp, t2_temp)

            psi_tot_2 = frozen_psi_tot if flow_frozen else get_psi_pert(w2, state) + psi_bg
            rhs_w, rhs_tracers = get_rhs_batched(w2, t2, psi_tot_2, state, cfg, compute_bio=True)
            w_temp = apply_implicit_visc(((1.0 / 3.0) * w + (2.0 / 3.0) * (w2 + dt * rhs_w)) * impl_drag['s3'], helm_denom['s3'], state)
            t_temp = {n_: (1.0 / 3.0) * tracers[n_] + (2.0 / 3.0) * (t2[n_] + dt * rhs_tracers[n_]) for n_ in tracer_names}
            w, tracers = apply_bcs(w_temp, t_temp)

            if getattr(cfg, 'use_symmetry', True):
                w, tracers = enforce_symmetry(w, tracers, tracer_names)

            n += 1
            current_time += dt

            if not flow_frozen and n % check_every == 0:
                delta = (w - w_prev_check).abs().max().item()
                w_prev_check = w.clone()
                if delta < tol:
                    consec_below_tol += 1
                    if consec_below_tol >= min_consecutive:
                        print(f"  Flow converged after {n} steps (t={current_time:.1f}s).")
                        print("Phase 2: Tracer advection for 100 seconds...")
                        flow_frozen = True
                        frozen_psi_tot = get_psi_pert(w, state) + psi_bg
                        end_time = current_time + 100.0  # Set stop clock
                else:
                    consec_below_tol = 0
                    print(f"  step {n}: max|dw| = {delta:.3e} (waiting for < {tol})")

            if flow_frozen and n % 2000 == 0:
                print(f"  [Phase 2] Sim time: {current_time:.1f} / {end_time:.1f} s")

    psi_tot = frozen_psi_tot if flow_frozen else get_psi_pert(w, state) + psi_bg
    
    def to_np(t):
        t = t.detach()
        return (t[0] if t.dim() > 2 else t).cpu().numpy()

    return to_np(w), to_np(psi_tot), {k: to_np(v) for k, v in tracers.items()}


# ─────────────────────────────────────────────────────────────────────────
# SWEEP RUNNER
# ─────────────────────────────────────────────────────────────────────────
def run_case(cfg, device, target_Re=None, target_Pe=None):
    """
    Overrides cfg.U_bg to hit a target Re or Pe (exactly one of the two),
    recomputes the dependent cfg.Re_actual / cfg.Pe_calc, rebuilds state via
    setup_physics, and runs physics to steady state. Returns a dict with
    w, psi, u, v, tracers (numpy, axis0=x axis1=y), plus dx, dy, cx, cy,
    radius, Re_actual, Pe_calc.
    """
    assert (target_Re is None) != (target_Pe is None), \
        "Specify exactly one of target_Re or target_Pe"

    if target_Re is not None:
        cfg.U_bg = target_Re * cfg.nu / (2.0 * cfg.radius)
    else:
        # Pe = Re * Sc  =>  Re = Pe / Sc  =>  U_bg = Re * nu / (2*radius)
        target_Re_from_Pe = target_Pe / cfg.Sc_target
        cfg.U_bg = target_Re_from_Pe * cfg.nu / (2.0 * cfg.radius)

    cfg.Re_actual = (cfg.U_bg * 2.0 * cfg.radius) / cfg.nu
    cfg.Pe_calc   = cfg.Re_actual * cfg.Sc_target

    device, state = setup_physics(cfg)
    w_np, psi_np, tracers_np = run_physics_to_steady_state(state, cfg, device)

    dx, dy = cfg.dx, cfg.dy

    # Velocity from the converged streamfunction (u = d(psi)/dy, v = -d(psi)/dx),
    # same axis0=x, axis1=y layout as w_np/psi_np -- transpose happens only
    # inside the plotting functions.
    u_np = np.zeros_like(psi_np)
    v_np = np.zeros_like(psi_np)
    u_np[:, 1:-1] = (psi_np[:, 2:] - psi_np[:, :-2]) / (2 * dy)
    v_np[1:-1, :] = -(psi_np[2:, :] - psi_np[:-2, :]) / (2 * dx)

    return {
        'w': w_np, 'psi': psi_np, 'u': u_np, 'v': v_np,
        'tracers': tracers_np,
        'dx': dx, 'dy': dy, 'cx': cfg.cx, 'cy': cfg.cy, 'radius': cfg.radius,
        'Re_actual': cfg.Re_actual, 'Pe_calc': cfg.Pe_calc,
    }


def _radii_axes(shape, dx, dy, cx, cy, radius):
    nx, ny = shape
    x = (np.arange(nx) * dx - cx) / radius
    y = (np.arange(ny) * dy - cy) / radius
    return x, y


# ─────────────────────────────────────────────────────────────────────────
# FIG 1 analog -- streamlines + vorticity across a Re sweep
# ─────────────────────────────────────────────────────────────────────────
def plot_fig1_re_sweep(cases, labels, savepath=None):
    fig, axes = plt.subplots(3, len(cases), figsize=(4.5 * len(cases), 12), squeeze=False)

    for col, (case, label) in enumerate(zip(cases, labels)):
        x, y = _radii_axes(case['w'].shape, case['dx'], case['dy'],
                           case['cx'], case['cy'], case['radius'])
        X, Y = np.meshgrid(x, y)
        w_plot, psi_plot = case['w'].T, case['psi'].T
        u, v = case['u'].T, case['v'].T
        speed = np.sqrt(u**2 + v**2)

        # Row 0: Vorticity
        ax = axes[0, col]
        vmax = np.percentile(np.abs(w_plot), 99) or 1.0
        cf_w = ax.contourf(X, Y, w_plot, levels=21, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=True, facecolor='0.15', edgecolor='k'))
        ax.set_aspect('equal')
        ax.set_title(f"{label}\n(Re={case['Re_actual']:.2f})")
        if col == 0: ax.set_ylabel('y (radii)\n[Vorticity]')

        # Row 1: Velocity Magnitude (with contour lines)
        ax = axes[1, col]
        cf_v = ax.contourf(X, Y, speed, levels=21, cmap='viridis')
        levels_v = np.linspace(speed.min(), speed.max(), 15)
        ax.contour(X, Y, speed, levels=levels_v, colors='w', alpha=0.5, linewidths=0.5)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=True, facecolor='0.15', edgecolor='k'))
        ax.set_aspect('equal')
        if col == 0: ax.set_ylabel('y (radii)\n[Speed]')

        # Row 2: Streamlines
        ax = axes[2, col]
        levels_psi = np.linspace(psi_plot.min(), psi_plot.max(), 25)
        ax.contour(X, Y, psi_plot, levels=levels_psi, colors='k', linewidths=0.5)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=True, facecolor='0.15', edgecolor='k'))
        ax.set_aspect('equal')
        ax.set_xlabel('x (particle radii)')
        if col == 0: ax.set_ylabel('y (radii)\n[Streamlines]')

    # Add shared colorbars for the contourf rows
    fig.colorbar(cf_w, ax=axes[0, :], label='Vorticity', shrink=0.8)
    fig.colorbar(cf_v, ax=axes[1, :], label='Speed (mm/s)', shrink=0.8)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print(f"Saved {savepath}")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────
# FIG 2 analog -- velocity magnitude vs radial distance, up/downstream and
# off-equator, one curve per Re
# ─────────────────────────────────────────────────────────────────────────
def plot_fig2_velocity_transects(cases, labels, savepath=None):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5))

    for case, label in zip(cases, labels):
        dx, dy, cx, cy, radius = case['dx'], case['dy'], case['cx'], case['cy'], case['radius']
        u, v = case['u'], case['v']
        speed = np.sqrt(u**2 + v**2)
        nx, ny = speed.shape

        j_cy = int(round(cy / dy))
        i_cx = int(round(cx / dx))

        # up/downstream: vary x at fixed y = cy
        x_line = (np.arange(nx) * dx - cx) / radius
        speed_line_x = speed[:, j_cy]
        inside_x = np.abs(x_line) < 1.0
        axA.plot(x_line[~inside_x], speed_line_x[~inside_x], label=f"{label} (Re={case['Re_actual']:.2f})")

        # equator: vary y at fixed x = cx (cut in half)
        y_line = (np.arange(ny) * dy - cy) / radius
        speed_line_y = speed[i_cx, :]
        
        # Only plot the positive half (from the surface outwards)
        mask_y = (y_line >= 0) & (np.abs(y_line) >= 1.0)
        axB.plot(y_line[mask_y], speed_line_y[mask_y], label=f"{label} (Re={case['Re_actual']:.2f})")

    axA.set_title('Up- and downstream')
    axA.set_xlabel('Radial distance from center (a)')
    axA.set_ylabel('Speed (mm/s)')
    axA.legend(fontsize=8)

    axB.set_title('Equator')
    axB.set_xlabel('Radial distance from center (a)')
    axB.legend(fontsize=8)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print(f"Saved {savepath}")
    return fig, (axA, axB)


# ─────────────────────────────────────────────────────────────────────────
# FIG 3 analog -- concentration field across a Pe sweep (4 panels)
# ─────────────────────────────────────────────────────────────────────────
def plot_fig3_pe_sweep(cases, labels, tracer_name='o2', savepath=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    for ax, case, label in zip(axes, cases, labels):
        x, y = _radii_axes(case['tracers'][tracer_name].shape, case['dx'], case['dy'],
                           case['cx'], case['cy'], case['radius'])
        X, Y = np.meshgrid(x, y)
        field = case['tracers'][tracer_name].T

        # Fix for Matplotlib rendering perfectly uniform fields as transparent/white
        vmin, vmax = np.nanmin(field), np.nanmax(field)
        if np.isnan(vmin) or (vmax - vmin) < 1e-6:
            vmin, vmax = vmin - 0.1, vmax + 0.1

        cf = ax.pcolormesh(X, Y, field, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=True, facecolor='k', edgecolor='k'))
        ax.set_aspect('equal')
        
        # Title redundancy removed
        ax.set_title(label)
        fig.colorbar(cf, ax=ax, label=f'{tracer_name.upper()} conc.', shrink=0.8)

    for ax in axes[2:]:
        ax.set_xlabel('Distance (a)')
    for ax in axes[::2]:
        ax.set_ylabel('Distance (a)')

    fig.suptitle(f'{tracer_name.upper()} field across Pe sweep\n'
                  '(NOTE: includes real biological consumption, not a pure passive scalar '
                  'like the reference figure -- transport-vs-Pe shape is the qualitative check)')
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print(f"Saved {savepath}")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR = 'validation_figs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = None  # unused positional arg, kept for signature symmetry with run_case

    # ── Re sweep (Fig 1 + Fig 2) ──────────────────────────────────────────
    re_targets = [0.01, 1.0, 10.0]
    re_labels  = ["Stokes' flow", "Re=1", "Re=10"]
    print("Running Re sweep for Fig 1 / Fig 2 ...")
    re_cases = [run_case(cfg, device, target_Re=r) for r in re_targets]

    plot_fig1_re_sweep(re_cases, re_labels,
                        savepath=os.path.join(OUTPUT_DIR, 'fig1_re_sweep.png'))
    plot_fig2_velocity_transects(re_cases, re_labels,
                                  savepath=os.path.join(OUTPUT_DIR, 'fig2_velocity_transects.png'))

    # ── Pe sweep (Fig 3) ──────────────────────────────────────────────────
    pe_targets = [1e-3, 100, 1000, 10000]
    pe_labels  = ["Pe~0", "Pe=100", "Pe=1000", "Pe=10000"]
    print("Running Pe sweep for Fig 3 ...")
    pe_cases = [run_case(cfg, device, target_Pe=p) for p in pe_targets]

    plot_fig3_pe_sweep(pe_cases, pe_labels, tracer_name='o2',
                        savepath=os.path.join(OUTPUT_DIR, 'fig3_pe_sweep.png'))

    print(f"\nAll figures saved to ./{OUTPUT_DIR}/")
    plt.show()