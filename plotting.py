# plotting.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import re
import biopar
import importlib
importlib.reload(biopar)


BIO_NAMES   = ['aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox', 'zoo']
BIO_LABELS  = ['Aer', 'NaR', 'NaI', 'NaO', 'NiR', 'NiO', 'NoS', 'AOA', 'NOB', 'AOX', 'Zoo']
BIO_COLORS  = [
    '#4daf4a',  # aer  — green
    '#377eb8',  # nar  — blue
    '#a65628',  # nai  — brown
    '#984ea3',  # nao  — purple
    '#ff7f00',  # nir  — orange
    '#e41a1c',  # nio  — red
    '#f781bf',  # nos  — pink
    '#999999',  # aoa  — grey
    '#ffff33',  # nob  — yellow
    '#1f78b4',  # aox  — dark blue
    '#b2df8a',  # zoo  — light green
]


def generate_plots(c_snapshots, n2o_snapshots, no3_snapshots, no2_snapshots,
                   n2_snapshots, doc_snapshots, nh4_snapshots,
                   w_snapshots, u_snapshots, v_snapshots,
                   snapshot_times, n2o_ammox_snapshots, n2o_denit_snapshots,
                   bio_snapshots, growth_rate_snapshots, cfg):
    

    t_tot = getattr(cfg, 'Total_Time', 'NA')
    t_mac = getattr(cfg, 'macro_cycle_time', 'NA')
    m_amp = getattr(biopar, 'loss_multiplier', 'NA')
    bio_skip = getattr(cfg, 'bio_skipping', getattr(cfg, 'bio_stepping', 'NA'))
    

    from biopar import BioPar

    try:
        with open("loop.py", "r") as f:
            match = re.search(r'\.min\(\)\.item\(\)\s*\*\s*([\d\.]+)', f.read())
            safe_dt_lim = float(match.group(1)) if match else "Unknown"
    except FileNotFoundError:
        safe_dt_lim = "Unknown"

    loss_mult = BioPar.loss_multiplier
    int_flush = getattr(cfg, 'intermittent_physics_flush_time', 750.0)

    # Generate the strictly accurate directory name
    out_dir = f"Plots_T{t_tot/86400:.1f}days_Mac{t_mac}sec_R{cfg.radius}mm_MortAmp{loss_mult}_BioOnlyTime_{bio_skip}s_Lim{safe_dt_lim}_IntFlush{int_flush}s"
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)

    x = np.linspace(0, cfg.Lx, cfg.Nx)
    y = np.linspace(0, cfg.Ly, cfg.Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    dt_val  = getattr(cfg, 'dt', 0.0)
    dt_str  = f"{dt_val:.6f}" if dt_val > 0 else "Unknown"
    drag_coeff = getattr(cfg, 'drag_max', 240.0)

    def get_bounds(data_list):
        val_min = np.min(data_list)
        val_max = np.max(data_list)
        if val_max - val_min < 1e-5:
            val_max = val_min + 0.01
        return val_min, val_max

    def add_perfect_colorbar(im, ax, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label=label)

    o2_min,  o2_max  = get_bounds(c_snapshots)
    no3_min, no3_max = get_bounds(no3_snapshots)
    no2_min, no2_max = get_bounds(no2_snapshots)
    n2o_min, n2o_max = get_bounds(n2o_snapshots)
    n2_min,  n2_max  = get_bounds(n2_snapshots)
    doc_min, doc_max = get_bounds(doc_snapshots)
    nh4_min, nh4_max = get_bounds(nh4_snapshots)

    n2o_ammox_max = max(np.max(n2o_ammox_snapshots), 1e-5)
    n2o_denit_max = max(np.max(n2o_denit_snapshots), 1e-5)

    # ── Shared param string ───────────────────────────────────────────────────
    Re_val = getattr(cfg, 'Re_actual', getattr(cfg, 'Re', 0.0))
    Sc_val = getattr(cfg, 'Sc_target', getattr(cfg, 'Sc', 0.0))
    Pe_val = getattr(cfg, 'Pe_calc',   getattr(cfg, 'Pe', 0.0))
    param_str = (f"Re: {Re_val:.2f} | Sc: {Sc_val:.1f} | Pe: {Pe_val:.2f}\n"
                 f"U: {cfg.U_bg} | Radius: {cfg.radius} | $\\nu$: {cfg.nu:.2f} | $K$: {cfg.K:.5f}\n"
                 f"$dx$: {cfg.dx:.3f} | $dy$: {cfg.dy:.3f} | $dt$: {dt_str}")

    base_filename = (f"U{cfg.U_bg}_R{cfg.radius}_nu{cfg.nu}_K{cfg.K:.5f}"
                     f"_dt{dt_str}_dx{cfg.dx:.3f}_dy{cfg.dy:.3f}_drag{drag_coeff}")

    zoom_y_min = max(0, cfg.cy - 2.5 * cfg.radius)
    zoom_y_max = min(cfg.Ly, cfg.cy + 2.5 * cfg.radius)

    # ═════════════════════════════════════════════════════════════════════════
    # ── 2D CHEMICAL ANIMATION (5 × 2) ────────────────────────────────────────
    # ═════════════════════════════════════════════════════════════════════════
    plt.rcParams['animation.embed_limit'] = 250
    fig, axes = plt.subplots(5, 2, figsize=(15, 10.9375), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    ax_vel, ax_o2, ax_no3, ax_no2, ax_n2o, ax_n2, ax_doc, ax_nh4, ax_n2o_ammox, ax_n2o_denit = axes_flat

    skip = 8
    for ax in axes_flat:
        ax.add_patch(Circle((cfg.cx, cfg.cy), cfg.radius, color='white', fill=False, linewidth=2))
        ax.axhline(cfg.cy, color='black', linestyle='--', alpha=0.6)
        ax.axvline(cfg.cx, color='black', linestyle='--', alpha=0.6)

    speed0 = np.sqrt(u_snapshots[0] ** 2 + v_snapshots[0] ** 2)
    im_vel = ax_vel.imshow(speed0.T, origin='lower', extent=[0, cfg.Lx, 0, cfg.Ly],
                           cmap='jet', vmin=0, vmax=cfg.U_bg * 1.5)
    add_perfect_colorbar(im_vel, ax_vel, 'Speed (mm/s)')
    Q = ax_vel.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                      u_snapshots[0][::skip, ::skip], v_snapshots[0][::skip, ::skip],
                      color='white', scale=cfg.U_bg * 40, alpha=0.8)
    ax_vel.set_title("Velocity", fontweight='bold')

    im_o2  = ax_o2.imshow(c_snapshots[0].T,   origin='lower', extent=[0,cfg.Lx,0,cfg.Ly], cmap='magma',  vmin=o2_min,  vmax=o2_max);  add_perfect_colorbar(im_o2,  ax_o2,  'O2 Concentration');  ax_o2.set_title("O2",    fontweight='bold')
    im_no3 = ax_no3.imshow(no3_snapshots[0].T, origin='lower', extent=[0,cfg.Lx,0,cfg.Ly], cmap='magma',  vmin=no3_min, vmax=no3_max); add_perfect_colorbar(im_no3, ax_no3, 'NO3 Concentration'); ax_no3.set_title("NO3",   fontweight='bold')
    im_no2 = ax_no2.imshow(no2_snapshots[0].T, origin='lower', extent=[0,cfg.Lx,0,cfg.Ly], cmap='plasma', vmin=no2_min, vmax=no2_max); add_perfect_colorbar(im_no2, ax_no2, 'NO2 Concentration'); ax_no2.set_title("NO2",   fontweight='bold')
    im_n2o = ax_n2o.imshow(n2o_snapshots[0].T, origin='lower', extent=[0,cfg.Lx,0,cfg.Ly], cmap='magma',  vmin=n2o_min, vmax=n2o_max); add_perfect_colorbar(im_n2o, ax_n2o, 'N2O Concentration'); ax_n2o.set_title("Total N2O", fontweight='bold')
    im_n2  = ax_n2.imshow(n2_snapshots[0].T,   origin='lower', extent=[0,cfg.Lx,0,cfg.Ly], cmap='magma',  vmin=n2_min,  vmax=n2_max);  add_perfect_colorbar(im_n2,  ax_n2,  'N2 Concentration');  ax_n2.set_title("N2",    fontweight='bold')
    im_doc = ax_doc.imshow(doc_snapshots[0].T,  origin='lower', extent=[0,cfg.Lx,0,cfg.Ly], cmap='YlGn',   vmin=doc_min, vmax=doc_max); add_perfect_colorbar(im_doc, ax_doc, 'DOC Concentration'); ax_doc.set_title("DOC",   fontweight='bold')
    im_nh4 = ax_nh4.imshow(nh4_snapshots[0].T,  origin='lower', extent=[0,cfg.Lx,0,cfg.Ly], cmap='magma',  vmin=nh4_min, vmax=nh4_max); add_perfect_colorbar(im_nh4, ax_nh4, 'NH4 Concentration'); ax_nh4.set_title("NH4",   fontweight='bold')
    im_n2o_ammox = ax_n2o_ammox.imshow(n2o_ammox_snapshots[0].T, origin='lower', extent=[0,cfg.Lx,0,cfg.Ly], cmap='magma', vmin=0, vmax=n2o_ammox_max); add_perfect_colorbar(im_n2o_ammox, ax_n2o_ammox, 'N2O (Ammox)'); ax_n2o_ammox.set_title("N2O from Ammonia Oxidation", fontweight='bold')
    im_n2o_denit = ax_n2o_denit.imshow(n2o_denit_snapshots[0].T, origin='lower', extent=[0,cfg.Lx,0,cfg.Ly], cmap='magma', vmin=0, vmax=n2o_denit_max); add_perfect_colorbar(im_n2o_denit, ax_n2o_denit, 'N2O (Denit)'); ax_n2o_denit.set_title("N2O from Denitrification", fontweight='bold')

    global_title = fig.suptitle("Time: 0.00", fontsize=18, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, param_str, ha='center', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    for ax in axes_flat:
        ax.set_xlim(0, cfg.Lx)
        ax.set_ylim(zoom_y_min, zoom_y_max)
    plt.subplots_adjust(top=0.84, bottom=0.05, left=0.05, right=0.95, hspace=0.15, wspace=0.1)

    def update(frame_idx):
        speed = np.sqrt(u_snapshots[frame_idx] ** 2 + v_snapshots[frame_idx] ** 2)
        im_vel.set_data(speed.T)
        Q.set_UVC(u_snapshots[frame_idx][::skip, ::skip], v_snapshots[frame_idx][::skip, ::skip])
        im_o2.set_data(c_snapshots[frame_idx].T)
        im_no3.set_data(no3_snapshots[frame_idx].T)
        im_no2.set_data(no2_snapshots[frame_idx].T)
        im_n2o.set_data(n2o_snapshots[frame_idx].T)
        im_n2.set_data(n2_snapshots[frame_idx].T)
        im_doc.set_data(doc_snapshots[frame_idx].T)
        im_nh4.set_data(nh4_snapshots[frame_idx].T)
        im_n2o_ammox.set_data(n2o_ammox_snapshots[frame_idx].T)
        im_n2o_denit.set_data(n2o_denit_snapshots[frame_idx].T)
        global_title.set_text(f"Time: {snapshot_times[frame_idx]:.2f}")
        return [im_vel, Q, im_o2, im_no3, im_no2, im_n2o, im_n2, im_doc, im_nh4, im_n2o_ammox, im_n2o_denit]

    anim = animation.FuncAnimation(fig, update, frames=len(c_snapshots), interval=15, blit=False)
    filename_2d = f"2D_Zoomed_{base_filename}.mp4"
    print(f"\nSaving 2D Animation to {filename_2d}...")
    pbar = tqdm(total=len(c_snapshots), desc="Rendering 2D", ascii="⡀⡄⡆⡇▞▚░▒▓", unit="frames",
                bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    anim.save(filename_2d, writer='ffmpeg', fps=60, progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    print(f"Saved {filename_2d}!")
    


    # ═════════════════════════════════════════════════════════════════════════
    # ── 1D CROSS-SECTION ANIMATION ────────────────────────────────────────────
    # ═════════════════════════════════════════════════════════════════════════
    y_slice = cfg.cy
    idx_y   = int(y_slice / cfg.dy)

    fig_1d, axes_1d = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    axes_1d_flat = axes_1d.flatten()

    titles_1d = ["Velocity Speed", "O2", "NO3", "NO2", "Total N2O",
                 "N2", "DOC", "NH4", "N2O (Ammox)", "N2O (Denit)"]
    colors_1d = ['black', 'blue', 'green', 'orange', 'purple',
                 'red', 'darkgreen', 'magenta', 'teal', 'brown']

    speed_0 = np.sqrt(u_snapshots[0] ** 2 + v_snapshots[0] ** 2)
    line_vel,       = axes_1d_flat[0].plot(x, speed_0[:, idx_y],               color=colors_1d[0], lw=2)
    line_o2,        = axes_1d_flat[1].plot(x, c_snapshots[0][:, idx_y],        color=colors_1d[1], lw=2)
    line_no3,       = axes_1d_flat[2].plot(x, no3_snapshots[0][:, idx_y],      color=colors_1d[2], lw=2)
    line_no2,       = axes_1d_flat[3].plot(x, no2_snapshots[0][:, idx_y],      color=colors_1d[3], lw=2)
    line_n2o,       = axes_1d_flat[4].plot(x, n2o_snapshots[0][:, idx_y],      color=colors_1d[4], lw=2)
    line_n2,        = axes_1d_flat[5].plot(x, n2_snapshots[0][:, idx_y],       color=colors_1d[5], lw=2)
    line_doc,       = axes_1d_flat[6].plot(x, doc_snapshots[0][:, idx_y],      color=colors_1d[6], lw=2)
    line_nh4,       = axes_1d_flat[7].plot(x, nh4_snapshots[0][:, idx_y],      color=colors_1d[7], lw=2)
    line_n2o_ammox, = axes_1d_flat[8].plot(x, n2o_ammox_snapshots[0][:, idx_y],color=colors_1d[8], lw=2)
    line_n2o_denit, = axes_1d_flat[9].plot(x, n2o_denit_snapshots[0][:, idx_y],color=colors_1d[9], lw=2)
    lines_1d = [line_vel, line_o2, line_no3, line_no2, line_n2o,
                line_n2, line_doc, line_nh4, line_n2o_ammox, line_n2o_denit]

    y_limits_1d = [
        (0, cfg.U_bg * 3),
        (max(0, o2_min - 0.05), o2_max + 0.05),
        (max(0, no3_min - 1.0), no3_max + 1.0),
        (max(0, no2_min - 0.001), no2_max + 0.001),
        (max(0, n2o_min - 0.001), n2o_max + 0.001),
        (max(0, n2_min  - 0.001), n2_max  + 0.001),
        (max(0, doc_min - 1.0),   doc_max + 1.0),
        (max(0, nh4_min - 0.001), nh4_max + 0.001),
        (0, n2o_ammox_max + 0.001),
        (0, n2o_denit_max + 0.001),
    ]
    for i, ax in enumerate(axes_1d_flat):
        ax.axvline(cfg.cx - cfg.radius, color='k', linestyle='--', alpha=0.5)
        ax.axvline(cfg.cx + cfg.radius, color='k', linestyle='--', alpha=0.5)
        ax.set_title(titles_1d[i], fontweight='bold')
        ax.set_xlim(0, cfg.Lx)
        ax.set_ylim(y_limits_1d[i])
        ax.set_ylabel("Concentration")
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0: ax.set_ylabel("Speed (mm/s)")
        if i >= 8: ax.set_xlabel("X coordinate")

    title_1d = fig_1d.suptitle("Horizontal Cross-Section — Time: 0.00", fontsize=16, fontweight='bold', y=0.98)
    fig_1d.text(0.5, 0.94, param_str, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    def update_1d(frame_idx):
        speed = np.sqrt(u_snapshots[frame_idx] ** 2 + v_snapshots[frame_idx] ** 2)
        lines_1d[0].set_ydata(speed[:, idx_y])
        lines_1d[1].set_ydata(c_snapshots[frame_idx][:, idx_y])
        lines_1d[2].set_ydata(no3_snapshots[frame_idx][:, idx_y])
        lines_1d[3].set_ydata(no2_snapshots[frame_idx][:, idx_y])
        lines_1d[4].set_ydata(n2o_snapshots[frame_idx][:, idx_y])
        lines_1d[5].set_ydata(n2_snapshots[frame_idx][:, idx_y])
        lines_1d[6].set_ydata(doc_snapshots[frame_idx][:, idx_y])
        lines_1d[7].set_ydata(nh4_snapshots[frame_idx][:, idx_y])
        lines_1d[8].set_ydata(n2o_ammox_snapshots[frame_idx][:, idx_y])
        lines_1d[9].set_ydata(n2o_denit_snapshots[frame_idx][:, idx_y])
        title_1d.set_text(f"Horizontal Cross-Section — Time: {snapshot_times[frame_idx]:.2f}")
        return lines_1d

    anim_1d = animation.FuncAnimation(fig_1d, update_1d, frames=len(c_snapshots), interval=15, blit=False)
    filename_1d = f"1D_{base_filename}.mp4"
    print(f"\nSaving 1D Animation to {filename_1d}...")
    pbar_1d = tqdm(total=len(c_snapshots), desc="Rendering 1D", ascii="⡀⡄⡆⡇▞▚░▒▓", unit="frames",
                   bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    anim_1d.save(filename_1d, writer='ffmpeg', fps=60, progress_callback=lambda i, n: pbar_1d.update(1))
    pbar_1d.close()
    print(f"Saved {filename_1d}!")


    # ═════════════════════════════════════════════════════════════════════════
    # ── 1D CROSS-SECTION ANIMATION FOR FUNCTIONAL TYPES ──────────────────────
    # ═════════════════════════════════════════════════════════════════════════
    fig_1d_bio, axes_1d_bio = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes_1d_bio_flat = axes_1d_bio.flatten()
    axes_1d_bio_flat[-1].set_visible(False)  # 11 bugs, 12th slot hidden

    lines_1d_bio = []
    
    for bi, (bname, blabel, bcolor) in enumerate(zip(BIO_NAMES, BIO_LABELS, BIO_COLORS)):
        ax = axes_1d_bio_flat[bi]
        
        # Calculate max along midline across all frames for scaling
        max_val = float(np.max([np.max(bio_snapshots[bname][fi][:, idx_y]) for fi in range(len(snapshot_times))]))
        y_max = max(1e-4, max_val * 1.05)
        
        line, = ax.plot(x, bio_snapshots[bname][0][:, idx_y], color=bcolor, lw=2)
        lines_1d_bio.append(line)
        
        ax.axvline(cfg.cx - cfg.radius, color='k', linestyle='--', alpha=0.5)
        ax.axvline(cfg.cx + cfg.radius, color='k', linestyle='--', alpha=0.5)
        ax.set_title(blabel, fontweight='bold')
        ax.set_xlim(0, cfg.Lx)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("Density (mmol C m⁻³)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if bi >= 8: 
            ax.set_xlabel("X coordinate")

    title_1d_bio = fig_1d_bio.suptitle("Horizontal Cross-Section: Functional Types — Time: 0.00", fontsize=16, fontweight='bold', y=0.98)
    fig_1d_bio.text(0.5, 0.94, param_str, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    def update_1d_bio(frame_idx):
        for bi, bname in enumerate(BIO_NAMES):
            lines_1d_bio[bi].set_ydata(bio_snapshots[bname][frame_idx][:, idx_y])
        title_1d_bio.set_text(f"Horizontal Cross-Section: Functional Types — Time: {snapshot_times[frame_idx]:.2f}")
        return lines_1d_bio

    anim_1d_bio = animation.FuncAnimation(fig_1d_bio, update_1d_bio, frames=len(snapshot_times), interval=15, blit=False)
    filename_1d_bio = f"1D_Bio_{base_filename}.mp4"
    print(f"\nSaving 1D Bio Animation to {filename_1d_bio}...")
    pbar_1d_bio = tqdm(total=len(snapshot_times), desc="Rendering 1D Bio", ascii="⡀⡄⡆⡇▞▚░▒▓", unit="frames",
                   bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    anim_1d_bio.save(filename_1d_bio, writer='ffmpeg', fps=60, progress_callback=lambda i, n: pbar_1d_bio.update(1))
    pbar_1d_bio.close()
    print(f"Saved {filename_1d_bio}!")



    # ═════════════════════════════════════════════════════════════════════════
    # ── POPULATION DENSITY ANIMATION ─────────────────────────────────────────
    #
    # Layout: 4 columns
    #   Left       — stacked-area chart: total biomass along horizontal midline.
    #                Shows WHERE on the particle each guild dominates.
    #   Mid-Left   — 100% stacked area: relative abundance over time.
    #   Mid-Right  — heatmap grid (11 rows × time): each row is one functional
    #                group; colour encodes mean biomass inside the particle mask
    #                as it evolves through the snapshot times.
    #                Shows WHEN each guild rises and falls.
    #   Right      — Shannon diversity index H(t) and species richness S(t)
    #                inspired by Stephens et al. 2024 (ISME J).
    #
    # Both panels update every frame.
    # ═════════════════════════════════════════════════════════════════════════

    # Pre-extract midline transect biomass for all frames and all bugs
    # Shape: [n_frames, n_bugs, Nx]
    n_frames = len(snapshot_times)
    n_bugs   = len(BIO_NAMES)

    mid_bio = np.zeros((n_frames, n_bugs, cfg.Nx), dtype=np.float32)
    for fi in range(n_frames):
        for bi, bname in enumerate(BIO_NAMES):
            mid_bio[fi, bi, :] = bio_snapshots[bname][fi][:, idx_y]

    # Pre-compute per-frame mean inside particle for the heatmap
    # We approximate "inside particle" as x within [cx-r, cx+r] on the midline
    ix_lo = np.searchsorted(x, cfg.cx - cfg.radius)
    ix_hi = np.searchsorted(x, cfg.cx + cfg.radius)
    if ix_hi <= ix_lo: ix_hi = ix_lo + 1

    # mean_core[frame, bug]
    mean_core = np.zeros((n_frames, n_bugs), dtype=np.float32)
    for fi in range(n_frames):
        for bi in range(n_bugs):
            mean_core[fi, bi] = mid_bio[fi, bi, ix_lo:ix_hi].mean()

    # Pre-compute Relative Abundance (0 to 1) for the new panel
    rel_abund = np.zeros_like(mean_core)
    for fi in range(n_frames):
        tot = mean_core[fi, :].sum()
        if tot > 1e-12:
            rel_abund[fi, :] = mean_core[fi, :] / tot

    # Global max for normalising stacked area (per-frame sum for y-limit)
    stack_max = float(mid_bio.sum(axis=1).max()) * 1.05
    if stack_max < 1e-10: stack_max = 1.0

    # Heatmap colour limits: 0 → 99th percentile of mean_core
    heat_vmax = float(np.percentile(mean_core, 99)) if mean_core.max() > 0 else 1.0
    if heat_vmax < 1e-10: heat_vmax = 1.0

    # ── Build figure — 4 columns ──────────────────────────────────────────────
    fig_pop, (ax_stack, ax_rel, ax_heat, ax_div) = plt.subplots(
        1, 4, figsize=(28, 6),
        gridspec_kw={'width_ratios': [1.5, 1.2, 1, 0.9]})

    fig_pop.patch.set_facecolor('#1a1a2e')
    for ax in (ax_stack, ax_rel, ax_heat, ax_div):
        ax.set_facecolor('#16213e')

    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    # ── LEFT: stacked area (PolyCollection — no clear/redraw needed) ──────────
    # Pre-compute polygon vertices for every frame and every bug group.
    # Each polygon is a closed loop:  x left-to-right along top edge,
    # then x right-to-left along bottom edge.
    # Shape per bug per frame: [2*Nx, 2] array of (x, y) vertices.
    def _stack_paths(frame_idx):
        """Return list of 11 closed polygon vertex arrays for the given frame."""
        bottoms = np.zeros(cfg.Nx, dtype=np.float64)
        paths   = []
        for bi in range(n_bugs):
            tops = bottoms + mid_bio[frame_idx, bi, :].astype(np.float64)
            # Closed polygon: go right along top, back left along bottom
            verts = np.empty((2 * cfg.Nx, 2), dtype=np.float64)
            verts[:cfg.Nx, 0] = x
            verts[:cfg.Nx, 1] = tops
            verts[cfg.Nx:, 0] = x[::-1]
            verts[cfg.Nx:, 1] = bottoms[::-1]
            paths.append(verts)
            bottoms = tops
        return paths

    # Build legend proxy patches before adding PolyCollections
    legend_handles = [Patch(facecolor=BIO_COLORS[bi], alpha=0.85, label=BIO_LABELS[bi])
                      for bi in range(n_bugs)]

    poly_cols = []
    init_paths = _stack_paths(0)
    for bi in range(n_bugs):
        pc = PolyCollection([init_paths[bi]], facecolors=BIO_COLORS[bi],
                            edgecolors='none', alpha=0.85)
        ax_stack.add_collection(pc)
        poly_cols.append(pc)

    ax_stack.set_xlim(0, cfg.Lx)
    ax_stack.set_ylim(0, stack_max)
    ax_stack.axvline(cfg.cx - cfg.radius, color='white', linestyle='--', lw=1, alpha=0.7)
    ax_stack.axvline(cfg.cx + cfg.radius, color='white', linestyle='--', lw=1, alpha=0.7)
    ax_stack.set_xlabel("X (mm)", color='white', fontsize=11)
    ax_stack.set_ylabel("Biomass (mmol C m⁻³)", color='white', fontsize=11)
    ax_stack.tick_params(colors='white')
    for spine in ax_stack.spines.values(): spine.set_edgecolor('#444466')
    ax_stack.legend(handles=legend_handles, loc='upper right', fontsize=8,
                    facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white',
                    ncol=2, framealpha=0.9)
    title_stack = ax_stack.set_title("Population Density — Midline Transect\nTime: 0.00 s",
                                     color='white', fontweight='bold', fontsize=12)

    # ── MID-LEFT: Relative Abundance over Time ────────────────────────────────
    def _rel_stack_paths(frame_idx):
        if frame_idx == 0:
            # Need at least two points to draw a polygon to prevent render glitch on frame 0
            t_vals = np.array([snapshot_times[0], snapshot_times[0] + 1e-5])
            r_vals = np.vstack([rel_abund[0], rel_abund[0]])
        else:
            t_vals = np.array(snapshot_times[:frame_idx + 1])
            r_vals = rel_abund[:frame_idx + 1, :]
            
        bottoms = np.zeros(len(t_vals), dtype=np.float64)
        paths = []
        for bi in range(n_bugs):
            tops = bottoms + r_vals[:, bi].astype(np.float64)
            verts = np.empty((2 * len(t_vals), 2), dtype=np.float64)
            verts[:len(t_vals), 0] = t_vals
            verts[:len(t_vals), 1] = tops
            verts[len(t_vals):, 0] = t_vals[::-1]
            verts[len(t_vals):, 1] = bottoms[::-1]
            paths.append(verts)
            bottoms = tops
        return paths

    poly_cols_rel = []
    init_paths_rel = _rel_stack_paths(0)
    for bi in range(n_bugs):
        pc = PolyCollection([init_paths_rel[bi]], facecolors=BIO_COLORS[bi],
                            edgecolors='none', alpha=0.9)
        ax_rel.add_collection(pc)
        poly_cols_rel.append(pc)

    ax_rel.set_xlim(snapshot_times[0], snapshot_times[-1])
    ax_rel.set_ylim(0, 1.0)
    ax_rel.set_xlabel("Time (s)", color='white', fontsize=11)
    ax_rel.set_ylabel("Relative Abundance", color='white', fontsize=11)
    ax_rel.tick_params(colors='white')
    for spine in ax_rel.spines.values(): spine.set_edgecolor('#444466')
    ax_rel.set_title("Core Relative Abundance vs. Time", color='white', fontweight='bold', fontsize=12)
    vline_rel = ax_rel.axvline(snapshot_times[0], color='cyan', lw=1.5, alpha=0.9)

    # ── MID-RIGHT: heatmap (11 bugs × current frame index rolling window) ─────
    # We show the full time series as a static image that un-masks as time passes.
    # Trick: initialise with NaN everywhere, reveal columns up to current frame.
    heat_data_full = mean_core.T.copy()
    heat_init      = np.full_like(heat_data_full, np.nan)
    heat_init[:, 0] = heat_data_full[:, 0]

    im_heat = ax_heat.imshow(
        heat_init,
        origin='upper', aspect='auto', cmap='inferno',
        vmin=0, vmax=heat_vmax, interpolation='nearest',
        extent=[snapshot_times[0], snapshot_times[-1], n_bugs - 0.5, -0.5])

    ax_heat.set_yticks(range(n_bugs))
    ax_heat.set_yticklabels(BIO_LABELS, color='white', fontsize=9)
    ax_heat.set_xlabel("Time (s)", color='white', fontsize=11)
    ax_heat.tick_params(colors='white', axis='x')
    for spine in ax_heat.spines.values(): spine.set_edgecolor('#444466')

    vline_heat = ax_heat.axvline(snapshot_times[0], color='cyan', lw=1.5, alpha=0.9)

    divider_h = make_axes_locatable(ax_heat)
    cax_h = divider_h.append_axes("right", size="5%", pad=0.08)
    cbar_h = plt.colorbar(im_heat, cax=cax_h)
    cbar_h.set_label("Mean core biomass\n(mmol C m⁻³)", color='white', fontsize=9)
    cbar_h.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar_h.ax.yaxis.get_ticklabels(), color='white')

    ax_heat.set_title("Core Mean Biomass vs. Time", color='white', fontweight='bold', fontsize=12)

    # ── RIGHT: Shannon diversity H(t) and richness S(t) over time ────────────
    # Computed from core mean biomass each frame (mean_core shape: [n_frames, n_bugs]).
    # Richness S: number of functional groups with mean_core > 1% of max that frame.
    # Shannon H: -Σ p_i ln(p_i), p_i = B_i / ΣB, only over groups with B_i > 0.
    # Both metrics mirror Stephens et al. 2024 (ISME J), who used richness as
    # the primary diversity metric and found Shannon patterns were nearly identical.
    RICH_THRESHOLD = 0.01   # fraction of total biomass below which a group is "absent"

    shannon_t  = np.zeros(n_frames, dtype=np.float64)
    richness_t = np.zeros(n_frames, dtype=np.float64)
    for fi in range(n_frames):
        row  = mean_core[fi, :]
        tot  = row.sum()
        if tot > 1e-12:
            p = row / tot
            # richness: groups above threshold
            richness_t[fi] = float(np.sum(p > RICH_THRESHOLD))
            # Shannon: only over positive proportions
            pos = p[p > 0]
            shannon_t[fi]  = float(-np.sum(pos * np.log(pos)))
        else:
            richness_t[fi] = 0.0
            shannon_t[fi]  = 0.0

    h_max = max(float(shannon_t.max()),  0.1)
    s_max = max(float(richness_t.max()), 1.0)

    # Twin axes: left = Shannon H (line), right = Richness S (dashed)
    ax_div2 = ax_div.twinx()

    line_H,  = ax_div.plot([], [], color='#00e5ff', lw=2.5, label='Shannon H')
    line_S,  = ax_div2.plot([], [], color='#ff9f43', lw=2, linestyle='--', label='Richness S')
    dot_H    = ax_div.scatter([], [], color='#00e5ff', s=50, zorder=5)
    dot_S    = ax_div2.scatter([], [], color='#ff9f43', s=50, zorder=5)
    vline_div = ax_div.axvline(snapshot_times[0], color='white', lw=1, alpha=0.5)

    ax_div.set_xlim(snapshot_times[0], snapshot_times[-1])
    ax_div.set_ylim(0, h_max * 1.15)
    ax_div2.set_ylim(0, s_max * 1.15)

    ax_div.set_xlabel("Time (s)",          color='white', fontsize=10)
    ax_div.set_ylabel("Shannon H",         color='#00e5ff', fontsize=10)
    ax_div2.set_ylabel("Richness (groups)", color='#ff9f43', fontsize=10)
    ax_div.tick_params(colors='white', axis='x')
    ax_div.tick_params(colors='#00e5ff', axis='y')
    ax_div2.tick_params(colors='#ff9f43', axis='y')
    ax_div2.set_facecolor('#16213e')
    for spine in ax_div.spines.values(): spine.set_edgecolor('#444466')
    for spine in ax_div2.spines.values(): spine.set_edgecolor('#444466')

    # Combined legend
    handles_div = [line_H, line_S]
    labels_div  = ['Shannon H (core)', f'Richness S (>{int(RICH_THRESHOLD*100)}% biomass)']
    ax_div.legend(handles_div, labels_div, loc='upper right', fontsize=8,
                  facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white',
                  framealpha=0.9)
    ax_div.set_title("Core Functional Diversity\n(Stephens et al. 2024 metric)",
                     color='white', fontweight='bold', fontsize=11)

    fig_pop.suptitle("Microbial Population Dynamics", color='white',
                     fontsize=15, fontweight='bold', y=1.01)
    fig_pop.text(0.5, -0.02, param_str, ha='center', va='top', fontsize=9,
                 color='white',
                 bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8, edgecolor='#444466'))
    plt.tight_layout()

    def update_pop(frame_idx):
        # ── Stacked area: update polygon vertices via set_paths ──────────────
        new_paths = _stack_paths(frame_idx)
        for bi, pc in enumerate(poly_cols):
            pc.set_paths([new_paths[bi]])
        title_stack.set_text(f"Population Density — Midline Transect\nTime: {snapshot_times[frame_idx]:.2f} s")

        # ── Relative Abundance ────────────────────────────────────────────────
        new_rel_paths = _rel_stack_paths(frame_idx)
        for bi, pc in enumerate(poly_cols_rel):
            pc.set_paths([new_rel_paths[bi]])
        vline_rel.set_xdata([snapshot_times[frame_idx]])

        # ── Heatmap: reveal columns up to current frame ───────────────────────
        heat_reveal = np.full_like(heat_data_full, np.nan)
        heat_reveal[:, :frame_idx + 1] = heat_data_full[:, :frame_idx + 1]
        im_heat.set_data(heat_reveal)
        vline_heat.set_xdata([snapshot_times[frame_idx]])

        # ── Diversity panel: reveal trace up to current frame ─────────────────
        t_so_far = snapshot_times[:frame_idx + 1]
        line_H.set_data(t_so_far, shannon_t[:frame_idx + 1])
        line_S.set_data(t_so_far, richness_t[:frame_idx + 1])
        if frame_idx > 0:
            dot_H.set_offsets([[snapshot_times[frame_idx], shannon_t[frame_idx]]])
            dot_S.set_offsets([[snapshot_times[frame_idx], richness_t[frame_idx]]])
        vline_div.set_xdata([snapshot_times[frame_idx]])

        return []

    anim_pop = animation.FuncAnimation(
        fig_pop, update_pop, frames=n_frames, interval=15, blit=False)

    filename_pop = f"PopDensity_{base_filename}.mp4"
    print(f"\nSaving Population Density Animation to {filename_pop}...")
    pbar_pop = tqdm(total=n_frames, desc="Rendering PopDensity",
                    ascii="⡀⡄⡆⡇▞▚░▒▓", unit="frames",
                    bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    anim_pop.save(filename_pop, writer='ffmpeg', fps=60,
                  progress_callback=lambda i, n: pbar_pop.update(1))
    pbar_pop.close()
    print(f"Saved {filename_pop}!")

    # # ═════════════════════════════════════════════════════════════════════════
    # # ── 2D SPATIAL MICROBIAL DENSITY ANIMATION ────────────────────────────────
    # #
    # # One imshow panel per functional group (11 groups + zoo = 11 panels total),
    # # arranged in a 3×4 grid (last cell empty).  Each panel shows the 2D spatial
    # # biomass field [mmol C m⁻³] at the current snapshot, colour-coded with
    # # each group's own colour fading from black.  Axes zoomed around the particle.
    # # The particle boundary circle is overlaid in white on each panel.
    # #
    # # This lets you see WHERE on and around the particle each functional group
    # # concentrates — the spatial succession picture that complements the
    # # time-series panels in PopDensity.
    # # ═════════════════════════════════════════════════════════════════════════

    # # Pre-compute global vmax for each bug across all frames (99th pct for robustness)
    # bio_vmaxes = []
    # for bname in BIO_NAMES:
    #     all_vals = np.concatenate([f.ravel() for f in bio_snapshots[bname]])
    #     vmax_b   = float(np.percentile(all_vals, 99)) if all_vals.max() > 0 else 1.0
    #     bio_vmaxes.append(max(vmax_b, 1e-10))

    # # Build per-bug colormaps: black → group colour
    # from matplotlib.colors import LinearSegmentedColormap
    # bug_cmaps = []
    # for col in BIO_COLORS:
    #     cmap_b = LinearSegmentedColormap.from_list(col, ['#000000', col], N=256)
    #     bug_cmaps.append(cmap_b)

    # n_cols_bio = 4
    # n_rows_bio = 3   # 3×4 = 12 slots for 11 groups
    # fig_bio, axes_bio = plt.subplots(n_rows_bio, n_cols_bio,
    #                                  figsize=(n_cols_bio * 4.5, n_rows_bio * 3.2))
    # axes_bio_flat = axes_bio.flatten()

    # # Hide the 12th (unused) panel
    # axes_bio_flat[-1].set_visible(False)

    # fig_bio.patch.set_facecolor('#0d0d1a')

    # im_bios = []
    # for bi, (bname, blabel, bcmap) in enumerate(zip(BIO_NAMES, BIO_LABELS, bug_cmaps)):
    #     ax_b = axes_bio_flat[bi]
    #     ax_b.set_facecolor('#0d0d1a')

    #     data0 = bio_snapshots[bname][0]
    #     im_b  = ax_b.imshow(data0.T, origin='lower',
    #                          extent=[0, cfg.Lx, 0, cfg.Ly],
    #                          cmap=bcmap, vmin=0, vmax=bio_vmaxes[bi],
    #                          interpolation='bilinear')
    #     im_bios.append(im_b)

    #     ax_b.add_patch(Circle((cfg.cx, cfg.cy), cfg.radius,
    #                            color='white', fill=False, linewidth=1.5))
    #     ax_b.set_xlim(0, cfg.Lx)
    #     ax_b.set_ylim(zoom_y_min, zoom_y_max)
    #     ax_b.set_title(blabel, color='white', fontsize=11, fontweight='bold', pad=4)
    #     ax_b.tick_params(colors='#555577', labelsize=7)
    #     for spine in ax_b.spines.values():
    #         spine.set_edgecolor('#222244')

    #     divider_b = make_axes_locatable(ax_b)
    #     cax_b     = divider_b.append_axes("right", size="5%", pad=0.05)
    #     cb_b      = plt.colorbar(im_b, cax=cax_b)
    #     cb_b.ax.tick_params(colors='#aaaacc', labelsize=7)
    #     cb_b.set_label("mmol C m⁻³", color='#aaaacc', fontsize=7)

    # title_bio = fig_bio.suptitle("Microbial Spatial Density — Time: 0.00 s",
    #                               color='white', fontsize=14, fontweight='bold', y=1.01)
    # fig_bio.text(0.5, -0.01, param_str, ha='center', va='top', fontsize=8,
    #              color='#aaaacc',
    #              bbox=dict(boxstyle='round', facecolor='#0d0d1a',
    #                        alpha=0.8, edgecolor='#333355'))
    # plt.tight_layout(rect=[0, 0.02, 1, 0.99])

    # def update_bio(frame_idx):
    #     for bi, bname in enumerate(BIO_NAMES):
    #         im_bios[bi].set_data(bio_snapshots[bname][frame_idx].T)
    #     title_bio.set_text(
    #         f"Microbial Spatial Density — Time: {snapshot_times[frame_idx]:.2f} s")
    #     return im_bios

    # anim_bio = animation.FuncAnimation(
    #     fig_bio, update_bio, frames=n_frames, interval=15, blit=False)

    # filename_bio = f"BioSpatial_{base_filename}.mp4"
    # print(f"\nSaving 2D Microbial Density Animation to {filename_bio}...")
    # pbar_bio = tqdm(total=n_frames, desc="Rendering BioSpatial",
    #                 ascii="⡀⡄⡆⡇▞▚░▒▓", unit="frames",
    #                 bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    # anim_bio.save(filename_bio, writer='ffmpeg', fps=60,
    #               progress_callback=lambda i, n: pbar_bio.update(1))
    # pbar_bio.close()
    # print(f"Saved {filename_bio}!")

    # # plt.show()

    # ═════════════════════════════════════════════════════════════════════════
    # ── EXACT SPECIFIC GROWTH RATES ──────────────────────────────────────────
    # ═════════════════════════════════════════════════════════════════════════

    mean_mu = np.zeros((n_frames, n_bugs), dtype=np.float32)
    
    # growth_rate_snapshots is saved from the interior array (Nx-2, Ny-2)
    # We must shift our original grid indices by 1 to match this smaller shape
    idx_y_interior = idx_y - 1
    ix_lo_interior = max(0, ix_lo - 1)
    ix_hi_interior = max(0, ix_hi - 1)

    for fi in range(n_frames):
        for bi, bname in enumerate(BIO_NAMES):
            # 1. Grab the 1D horizontal midline (all X, at the center Y)
            mid_mu_1d = growth_rate_snapshots[bname][fi][:, idx_y_interior]
            
            # 2. Take the mean only inside the particle bounds (X-axis slice)
            mean_mu[fi, bi] = mid_mu_1d[ix_lo_interior:ix_hi_interior].mean()

    # Scale back down to true biological rates
    bio_accel = getattr(cfg, 'BIO_ACCEL', 1.0)
    mean_mu_real = mean_mu / bio_accel

    fig_gr, ax_gr = plt.subplots(figsize=(10, 6))
    fig_gr.patch.set_facecolor('#1a1a2e')
    ax_gr.set_facecolor('#16213e')

    for bi, bname in enumerate(BIO_NAMES):
        ax_gr.plot(snapshot_times, mean_mu_real[:, bi], color=BIO_COLORS[bi], lw=2, label=BIO_LABELS[bi])

    ax_gr.set_title(f"Exact Specific Gross Growth Rates (Core)\nScaled to real time (BIO_ACCEL = {bio_accel})", color='white', fontweight='bold')
    ax_gr.set_xlabel("Time (s)", color='white')
    ax_gr.set_ylabel("Growth Rate (day⁻¹)", color='white')
    ax_gr.tick_params(colors='white')
    for spine in ax_gr.spines.values(): spine.set_edgecolor('#444466')
    ax_gr.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white', ncol=2)

    filename_gr = f"ExactGrowthRates_{base_filename}.png"
    plt.tight_layout()
    plt.savefig(filename_gr, dpi=300)
    plt.close(fig_gr)
    print(f"Saved {filename_gr}!")


    # ── EXACT BIOMASS OVER TIME (STATIC PLOT) ────────────────────────────
    fig_bio_ts, ax_bio_ts = plt.subplots(figsize=(10, 6))
    fig_bio_ts.patch.set_facecolor('#1a1a2e')
    ax_bio_ts.set_facecolor('#16213e')

    for bi, bname in enumerate(BIO_NAMES):
        ax_bio_ts.plot(snapshot_times, mean_core[:, bi], color=BIO_COLORS[bi], lw=2, label=BIO_LABELS[bi])

    ax_bio_ts.set_title("Mean Core Biomass Over Time", color='white', fontweight='bold')
    ax_bio_ts.set_xlabel("Time (s)", color='white')
    ax_bio_ts.set_ylabel("Biomass (mmol C m⁻³)", color='white')
    ax_bio_ts.tick_params(colors='white')
    for spine in ax_bio_ts.spines.values(): spine.set_edgecolor('#444466')
    ax_bio_ts.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white', ncol=2)

    filename_bio_ts = f"BiomassTimeSeries_{base_filename}.png"
    plt.tight_layout()
    plt.savefig(filename_bio_ts, dpi=300)
    plt.close(fig_bio_ts)
    print(f"Saved {filename_bio_ts}!")


    # ── TRACER TIME SERIES AT MULTIPLE POINTS (STEADY-STATE CHECK) ───────────
    # Three sample points along the centerline, same y (cfg.cy):
    #   - Core:       particle center
    #   - Upstream:   halfway between the core and the particle's LEFT edge
    #                 (still inside the particle, ambient-facing side)
    #   - Downstream: halfway between the core and the particle's RIGHT edge
    #                 (still inside the particle, wake-facing side)
    iy_c = int(round(cfg.cy / cfg.dy))
    ix_core       = int(round(cfg.cx / cfg.dx))
    ix_upstream   = int(round((cfg.cx - cfg.radius / 2.0) / cfg.dx))
    ix_downstream = int(round((cfg.cx + cfg.radius / 2.0) / cfg.dx))

    sample_points = {
        'Core (particle center)': ix_core,
        'Upstream (halfway to left edge)': ix_upstream,
        'Downstream (halfway to right edge)': ix_downstream,
    }
    sample_colors = {
        'Core (particle center)': '#1f78b4',
        'Upstream (halfway to left edge)': '#33a02c',
        'Downstream (halfway to right edge)': '#e31a1c',
    }

    tracer_snapshot_map = {
        'O2':  c_snapshots,
        'NO3': no3_snapshots,
        'NO2': no2_snapshots,
        'N2O': n2o_snapshots,
        'N2':  n2_snapshots,
        'DOC': doc_snapshots,
        'NH4': nh4_snapshots,
    }

    fig_ss, axes_ss = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    axes_ss_flat = axes_ss.flatten()
    axes_ss_flat[-1].set_visible(False)  # 7 tracers, 8th slot unused

    for ax_s, (tname, snaps) in zip(axes_ss_flat, tracer_snapshot_map.items()):
        for label, ix in sample_points.items():
            series = [snaps[fi][ix, iy_c] for fi in range(n_frames)]
            ax_s.plot(snapshot_times, series, lw=2, color=sample_colors[label], label=label)
        ax_s.set_title(tname, fontweight='bold')
        ax_s.set_ylabel('Concentration (mmol/m³)')
        ax_s.grid(alpha=0.3)

    axes_ss_flat[0].legend(loc='best', fontsize=8)
    for ax_s in axes_ss_flat[-3:-1]: ax_s.set_xlabel('Time (s)')

    fig_ss.suptitle(
        f"Tracer Concentrations at 3 Points — Steady-State Check\n"
        f"(y = {cfg.cy:.2f}; core x={cfg.cx:.2f}, "
        f"upstream x={cfg.cx - cfg.radius/2.0:.2f}, "
        f"downstream x={cfg.cx + cfg.radius/2.0:.2f})",
        fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename_ss = f"SteadyStateCheck_{base_filename}.png"
    plt.savefig(filename_ss, dpi=300)
    plt.close(fig_ss)
    print(f"Saved {filename_ss}!")

    # ── FUNCTIONAL TYPE TIME SERIES AT 3 POINTS (STEADY-STATE CHECK FOR BUGS) ──
    # Mirroring the layout above for bugs
    n_rows_b_ss = 4
    n_cols_b_ss = 3   # 4x3 = 12 slots for 11 bugs
    fig_ss_bugs, axes_ss_bugs = plt.subplots(n_rows_b_ss, n_cols_b_ss,
                                               figsize=(n_cols_b_ss * 4.67, n_rows_b_ss * 3.5),
                                               sharex=True)
    axes_ss_bugs_flat = axes_ss_bugs.flatten()
    axes_ss_bugs_flat[-1].set_visible(False)  # 11 bugs, 12th slot unused

    # Dictionary mapping each bug name to its time-series of 2D memmap snapshots
    bio_ts_snapshots = {
        bname: bio_snapshots[bname]
        for bname in BIO_NAMES
    }

    for ax_s, (bname, blabel) in zip(axes_ss_bugs_flat[:11], zip(BIO_NAMES, BIO_LABELS)):
        bug_snaps = bio_ts_snapshots[bname]
        for label, ix in sample_points.items():
            # series shape is (n_frames,)
            series = [bug_snaps[fi][ix, iy_c] for fi in range(n_frames)]
            ax_s.plot(snapshot_times, series, lw=2, color=sample_colors[label], label=label)
        ax_s.set_title(blabel, fontweight='bold')
        ax_s.set_ylabel('Density (mmol C m⁻³)')
        ax_s.grid(alpha=0.3)

    # First plot legend clarification
    axes_ss_bugs_flat[0].legend(loc='best', fontsize=8)

    # shared xlabel on last two plots only (12 slots, flat, hidden is -1, last row except -1 is -3,-2)
    # Correcting indices for a 4x3 grid with last hidden. Row 3 indices are 9,10,11. Valid are 9,10.
    # Flat indices are correct.
    for ax_s in axes_ss_bugs_flat[9:11]: ax_s.set_xlabel('Time (s)')

    fig_ss_bugs.suptitle(
        f"Steady-State Check for each Functional Type\n"
        f"(y = {cfg.cy:.2f}; core x={cfg.cx:.2f}, "
        f"upstream x={cfg.cx - cfg.radius/2.0:.2f}, "
        f"downstream x={cfg.cx + cfg.radius/2.0:.2f})",
        fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename_ss_bugs = f"SteadyStateCheck_Bugs_{base_filename}.png"
    plt.savefig(filename_ss_bugs, dpi=300)
    plt.close(fig_ss_bugs)
    print(f"Saved {filename_ss_bugs}!")


    # os.chdir(out_dir)