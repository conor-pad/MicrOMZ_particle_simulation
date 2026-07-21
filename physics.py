# physics.py
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

from bcs import inflow
from biopar import BioPar
from sms import microbial_sms_omz


def setup_physics(cfg):
    """
    Initialises the grid, calculates dt, and pre-allocates all GPU tensors.

    Carbon source design:
      POC is the ONLY carbon source. It hydrolyses via first-order decay:
          doc_flux(t) = k_hyd * poc(t)   where  poc(t) = poc_initial * exp(-k_hyd * t)
      doc_flux is computed every timestep in loop.py and passed here as doc_flux_t.
      There is NO doc_flux_rate config parameter and NO doc_initial_core.

    The same BIO_ACCEL factor is applied to k_hyd so the POC→DOC supply keeps
    pace with the accelerated microbial consumption rates.
    """
    # ── Device ────────────────────────────────────────────────────────────────
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Apple Silicon GPU (MPS) detected! Running on Metal.")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ CUDA GPU detected!")
    else:
        device = torch.device('cpu')
        print("⚠️  No GPU found, falling back to CPU.")

    bs = getattr(cfg, 'batch_size', 1)

    # Helper: broadcast scalar config values to [bs, 1, 1] numpy arrays
    def b_arr(val):
        arr = np.atleast_1d(val).astype(np.float64)
        if len(arr) == 1 and bs > 1:
            arr = np.repeat(arr, bs)
        return arr.reshape(bs, 1, 1)

    Lx_b, Ly_b     = b_arr(cfg.Lx), b_arr(cfg.Ly)
    cx_b, cy_b     = b_arr(cfg.cx), b_arr(cfg.cy)
    radius_b        = b_arr(cfg.radius)
    U_bg_b          = b_arr(cfg.U_bg)
    dx_b, dy_b     = b_arr(cfg.dx), b_arr(cfg.dy)

    # ── Grid ──────────────────────────────────────────────────────────────────
    X_norm, Y_norm = np.meshgrid(
        np.linspace(0, 1, cfg.Nx), np.linspace(0, 1, cfg.Ny), indexing='ij')
    X = X_norm[None, ...] * Lx_b
    Y = Y_norm[None, ...] * Ly_b

    psi_bg_np = U_bg_b * Y

    # ── Drag mask & particle mask ─────────────────────────────────────────────
    t_adv_cell = np.minimum(dx_b, dy_b) / U_bg_b
    max_alpha   = 30.0 / t_adv_cell
    cfg.drag_max = max_alpha

    particle_idx     = (X - cx_b) ** 2 + (Y - cy_b) ** 2 <= radius_b ** 2
    drag_mask_np     = np.where(particle_idx, max_alpha, 0.0)
    particle_mask_np = np.where(particle_idx, 1.0, 0.0)

    # Mirror → smooth → mirror (kills Gaussian asymmetry)
    mid_y = cfg.Ny // 2
    drag_mask_np[..., :mid_y] = drag_mask_np[..., cfg.Ny - 1 - np.arange(mid_y)]
    drag_mask_np = gaussian_filter(drag_mask_np, sigma=(0, 1.0, 1.0))
    drag_mask_np[..., :mid_y] = drag_mask_np[..., -1:-mid_y - 1:-1]

    da_dx_np = np.gradient(drag_mask_np, axis=1) / dx_b
    da_dy_np = np.gradient(drag_mask_np, axis=2) / dy_b

    # ── Timestep ──────────────────────────────────────────────────────────────
    # Drag and viscosity are IMPLICIT → only advection and scalar diffusion limit dt.
    dt_adv  = np.min(cfg.target_CFL * np.minimum(dx_b, dy_b) / U_bg_b)
    dt_diff = np.min(0.25 * np.minimum(dx_b, dy_b) ** 2 / cfg.K)
    dt_drag = np.min(0.5 / max_alpha)
    dt_visc = np.min(0.25 * np.minimum(dx_b, dy_b) ** 2 / cfg.nu)

    dt, min_name = min((dt_adv, 'dt_adv'), (dt_diff, 'dt_diff'))
    print(f"Time step: {dt:.6f} s  (Source: {min_name})")
    print(f"  [implicit — no longer limiting]  dt_drag={dt_drag:.6f},  dt_visc={dt_visc:.6f}")
    if min(dt_drag, dt_visc) < dt:
        print(f"  Effective speedup vs old explicit: ~{dt / min(dt_drag, dt_visc):.1f}×")
    cfg.dt = float(dt)

    # ── Tracer name lists ──────────────────────────────────────────────────────
    chem_names = [
        'o2', 'no3', 'doc', 'po4', 'n2o', 'n2o_ammox', 'n2o_denit',
        'nh4', 'no2', 'n2'
    ]
    bio_names = [
        'aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos', 'aoa', 'nob', 'aox', 'zoo'
    ]
    tracer_names = chem_names + bio_names

    # ── State dict ────────────────────────────────────────────────────────────
    state = {}
    state['dt']           = dt
    state['nu']           = torch.tensor(cfg.nu,  dtype=torch.float32, device=device)
    state['K']            = torch.tensor(cfg.K,   dtype=torch.float32, device=device)
    state['inv_dx']       = torch.tensor(1.0 / dx_b,           dtype=torch.float32, device=device)
    state['inv_dy']       = torch.tensor(1.0 / dy_b,           dtype=torch.float32, device=device)
    state['inv_dx2']      = torch.tensor(1.0 / dx_b ** 2,      dtype=torch.float32, device=device)
    state['inv_dy2']      = torch.tensor(1.0 / dy_b ** 2,      dtype=torch.float32, device=device)
    state['inv_2dx']      = torch.tensor(1.0 / (2.0 * dx_b),   dtype=torch.float32, device=device)
    state['inv_2dy']      = torch.tensor(1.0 / (2.0 * dy_b),   dtype=torch.float32, device=device)
    state['inv_4dxdy']    = torch.tensor(1.0 / (4.0 * dx_b * dy_b), dtype=torch.float32, device=device)
    state['bio_accel']    = float(getattr(cfg, 'BIO_ACCEL', 100_000.0))
    # Per-batch-member mortality amplifier — [bs,1,1], not a scalar, so
    # poc_mort/radius_mort sweeps apply each member's own MORT_AMP value.
    state['mort_amp']     = torch.tensor(b_arr(getattr(cfg, 'MORT_AMP', 1.0)),
                                          dtype=torch.float32, device=device)

    state['bgc']          = BioPar()
    state['chem_names']   = chem_names
    state['bio_names']    = bio_names
    state['tracer_names'] = tracer_names

    # ── POC hydrolysis ────────────────────────────────────────────────────────
    # Alldredge (1998) fractal scaling, or Klawonn density override.
    V_mm3    = np.pi * (radius_b ** 2) * 1.0          # cylinder: πr²×1mm depth
    poc_ug   = 0.99 * (V_mm3 ** 0.52)                 # Alldredge fractal mass (µg C)
    poc_mmol_alldredge = poc_ug / 12_010               # µg → mmol C (12010 µg/mmol)
    poc_density_alldredge = poc_mmol_alldredge / (V_mm3 * 1e-9)  # mmol C m⁻³

    if getattr(cfg, 'use_klawonn_density', False) and hasattr(cfg, 'poc_initial_core'):
        poc_initial_density = (np.ones((bs, 1, 1), dtype=np.float32)
                               * float(cfg.poc_initial_core))
    else:
        poc_initial_density = poc_density_alldredge.astype(np.float32)


    # THIS IS BEING ACCELERATED WITH BIO AMP BTW!!!!!!!!!
    # k_hyd is scaled by BIO_ACCEL so DOC supply keeps pace with accelerated biology.
    # POC decays as:  poc(t) = poc_initial * exp(-k_hyd_eff * t)
    # DOC flux:       doc_flux(t) = k_hyd_eff * poc(t)
    # This is computed every step in loop.py using state['k_hyd'] and state['poc_initial'].
    k_hyd_raw = float(state['bgc'].k_hyd)              # s⁻¹ from biopar
    k_hyd_eff = k_hyd_raw * state['bio_accel']         # accelerated

    state['k_hyd']       = torch.tensor(k_hyd_eff,           dtype=torch.float32, device=device)
    state['poc_initial'] = torch.tensor(poc_initial_density,  dtype=torch.float32, device=device)
    print(f"POC | density={poc_initial_density.flat[0]:.2e} mmol/m³  "
          f"k_hyd_raw={k_hyd_raw:.2e}/s  k_hyd_eff={k_hyd_eff:.2e}/s")

    # ── Static GPU arrays ─────────────────────────────────────────────────────
    state['psi_bg']        = torch.tensor(psi_bg_np,        dtype=torch.float32, device=device)
    state['drag_mask']     = torch.tensor(drag_mask_np,     dtype=torch.float32, device=device)
    state['da_dx']         = torch.tensor(da_dx_np,         dtype=torch.float32, device=device)
    state['da_dy']         = torch.tensor(da_dy_np,         dtype=torch.float32, device=device)
    state['particle_mask'] = torch.tensor(particle_mask_np, dtype=torch.float32, device=device)

    # ── Vorticity ─────────────────────────────────────────────────────────────
    state['w']          = torch.zeros((bs, cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)
    state['_rhs_buf_w'] = torch.zeros((bs, cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)

    # ── Tracers ───────────────────────────────────────────────────────────────
    state['tracers']      = {}
    state['_rhs_tracers'] = {}

    for name in tracer_names:
        init_val = getattr(inflow, name)

        if name == 'doc':
            # DOC starts at zero everywhere — it is produced solely by POC hydrolysis.
            state['tracers'][name] = torch.zeros(
                (bs, cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)

        elif name in set(bio_names):
            # Bugs initialised inside the particle using the value from bcs.inflow.
            arr = np.zeros((bs, cfg.Nx, cfg.Ny), dtype=np.float32)
            arr[particle_mask_np > 0] = float(init_val)
            state['tracers'][name] = torch.tensor(arr, device=device)

        else:
            # Chemical tracers: uniform inflow value everywhere.
            init_arr = b_arr(init_val)
            init_t   = torch.tensor(init_arr, dtype=torch.float32, device=device)
            state['tracers'][name] = (
                torch.ones((bs, cfg.Nx, cfg.Ny), dtype=torch.float32, device=device) * init_t)

        state['_rhs_tracers'][name] = torch.zeros(
            (bs, cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)

    # ── Velocity scratch buffers ───────────────────────────────────────────────
    state['u_full'] = torch.zeros((bs, cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)
    state['v_full'] = torch.zeros((bs, cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)

    # ── Poisson eigenvalues ────────────────────────────────────────────────────
    Ni, Nj = cfg.Nx - 2, cfg.Ny - 2
    state['Ni'], state['Nj'] = Ni, Nj

    ii    = torch.arange(1, Ni + 1, dtype=torch.float32, device=device)
    jj    = torch.arange(1, Nj + 1, dtype=torch.float32, device=device)
    dx_t  = torch.tensor(dx_b, dtype=torch.float32, device=device)
    dy_t  = torch.tensor(dy_b, dtype=torch.float32, device=device)
    lam_x = (2.0 / dx_t ** 2) * (torch.cos(np.pi * ii / (Ni + 1)) - 1.0).unsqueeze(1)
    lam_y = (2.0 / dy_t ** 2) * (torch.cos(np.pi * jj / (Nj + 1)) - 1.0).unsqueeze(0)
    state['Lambda'] = lam_x + lam_y   # shape [bs, Ni, Nj] or [Ni, Nj], all ≤ 0

    state['_dst_buf_axis0'] = torch.zeros((bs, 2 * (Ni + 1), Nj),  dtype=torch.float32, device=device)
    state['_dst_buf_axis1'] = torch.zeros((bs, Ni, 2 * (Nj + 1)), dtype=torch.float32, device=device)

    # ── IMEX denominators (one set per SSP-RK3 stage) ─────────────────────────
    Lambda = state['Lambda']
    for tag, c in [('s1', 1.0), ('s2', 0.25), ('s3', 2.0 / 3.0)]:
        state[f'impl_drag_{tag}'] = 1.0 / (1.0 + c * dt * state['drag_mask'])
        state[f'helm_denom_{tag}'] = 1.0 - c * dt * cfg.nu * Lambda
    print("✅ IMEX implicit drag + viscosity denominators precomputed.")

    return device, state


# ── DST-I Poisson solver ──────────────────────────────────────────────────────

def dstn1(x_in, state):
    n0, n1 = x_in.shape[-2], x_in.shape[-1]
    buf0   = state['_dst_buf_axis0']
    buf1   = state['_dst_buf_axis1']

    buf0.zero_()
    buf0[..., 1:n0 + 1, :] = x_in
    buf0[..., n0 + 2:,  :] = -torch.flip(x_in, dims=[-2])

    y0  = torch.fft.fft(buf0, dim=-2)
    mid = -y0[..., 1:n0 + 1, :].imag / float(np.sqrt(2 * (n0 + 1)))

    buf1.zero_()
    buf1[..., :, 1:n1 + 1] = mid
    buf1[..., :, n1 + 2:]  = -torch.flip(mid, dims=[-1])

    y1 = torch.fft.fft(buf1, dim=-1)
    return -y1[..., :, 1:n1 + 1].imag / float(np.sqrt(2 * (n1 + 1)))


def get_psi_pert(w_field, state):
    rhs       = -w_field[..., 1:-1, 1:-1]
    rhs_hat   = dstn1(rhs, state)
    psi_inner = dstn1(rhs_hat / state['Lambda'], state)
    psi       = torch.zeros_like(w_field)
    psi[..., 1:-1, 1:-1] = psi_inner
    return psi


def apply_implicit_visc(w_in, helm_denom, state):
    """
    Solve (I − c·dt·ν·∇²) w_out = w_in on interior points via DST-I.
    Boundary values are left as-is; apply_bcs corrects them immediately after.
    """
    rhs_hat = dstn1(w_in[..., 1:-1, 1:-1], state)
    w_in[..., 1:-1, 1:-1] = dstn1(rhs_hat / helm_denom, state)
    return w_in


# ── RHS ───────────────────────────────────────────────────────────────────────

def get_rhs_batched(w_f, tracers_dict, streamfunction, state, cfg, doc_flux_t):
    """
    Batched RHS for vorticity (Arakawa) and tracers.

    doc_flux_t : tensor [bs, 1, 1] — k_hyd_eff * poc(t), computed each step in loop.py.
                 Added to DOC inside the particle mask only. Units: mmol m⁻³ s⁻¹.
                 NOT multiplied by bio_accel here (k_hyd is already pre-scaled).

    Biology (SMS) is scaled by bio_accel.
    Mortality is additionally scaled by mort_amp before the SMS call.
    Bio tracers get SMS only — no advection, no diffusion.
    """
    p = streamfunction

    p_e  = p[..., 2:,   1:-1];  p_w  = p[..., :-2,  1:-1]
    p_n  = p[..., 1:-1, 2:];    p_s  = p[..., 1:-1, :-2]
    p_ne = p[..., 2:,   2:];    p_sw = p[..., :-2,  :-2]
    p_se = p[..., 2:,   :-2];   p_nw = p[..., :-2,  2:]

    dp_ew    = p_e  - p_w
    dp_ns    = p_n  - p_s
    dp_ne_se = p_ne - p_se
    dp_nw_sw = p_nw - p_sw
    dp_ne_nw = p_ne - p_nw
    dp_se_sw = p_se - p_sw

    chem_names = state['chem_names']
    bio_names  = state['bio_names']
    bio_accel  = state['bio_accel']
    mort_amp   = state['mort_amp']

    # ── Stack: index 0 = vorticity, 1..N = chem tracers ──────────────────────
    f_c_list = [w_f[..., 1:-1, 1:-1]] + [tracers_dict[n][..., 1:-1, 1:-1] for n in chem_names]
    f_e_list = [w_f[..., 2:,   1:-1]] + [tracers_dict[n][..., 2:,   1:-1] for n in chem_names]
    f_w_list = [w_f[..., :-2,  1:-1]] + [tracers_dict[n][..., :-2,  1:-1] for n in chem_names]
    f_n_list = [w_f[..., 1:-1, 2:  ]] + [tracers_dict[n][..., 1:-1, 2:  ] for n in chem_names]
    f_s_list = [w_f[..., 1:-1, :-2 ]] + [tracers_dict[n][..., 1:-1, :-2 ] for n in chem_names]

    w_ne = w_f[..., 2:,  2:  ]; w_sw = w_f[..., :-2, :-2]
    w_se = w_f[..., 2:,  :-2 ]; w_nw = w_f[..., :-2, 2:  ]

    f_c = torch.stack(f_c_list)
    f_e = torch.stack(f_e_list)
    f_w = torch.stack(f_w_list)
    f_n = torch.stack(f_n_list)
    f_s = torch.stack(f_s_list)

    w_c, w_e, w_w, w_n, w_s = f_c[0], f_e[0], f_w[0], f_n[0], f_s[0]

    # ── 1. Arakawa Jacobian ───────────────────────────────────────────────────
    J_std   = (dp_ew * (w_n - w_s) - dp_ns * (w_e - w_w)) * state['inv_4dxdy']
    J_hat   = (p_e  * (w_ne - w_se) - p_w  * (w_nw - w_sw)
             - p_n  * (w_ne - w_nw) + p_s  * (w_se - w_sw)) * state['inv_4dxdy']
    J_tilde = (w_n  * dp_ne_nw - w_s  * dp_se_sw
             - w_e  * dp_ne_se + w_w  * dp_nw_sw) * state['inv_4dxdy']
    J_avg   = (J_std + J_hat + J_tilde) / 3.0

    lap = ((f_e + f_w - 2.0 * f_c) * state['inv_dx2'] +
           (f_n + f_s - 2.0 * f_c) * state['inv_dy2'])

    # ── 2. Local velocities ───────────────────────────────────────────────────
    u_c =  dp_ns * state['inv_2dy']
    v_c = -dp_ew * state['inv_2dx']

    # ── 3. Vorticity explicit RHS ─────────────────────────────────────────────
    dax = state['da_dx'][..., 1:-1, 1:-1]
    day = state['da_dy'][..., 1:-1, 1:-1]
    rhs_w = J_avg + (day * u_c - dax * v_c)
    state['_rhs_buf_w'][..., 1:-1, 1:-1] = rhs_w

    # ── 4. SMS ────────────────────────────────────────────────────────────────
    interior_tracers = {n: tracers_dict[n][..., 1:-1, 1:-1] for n in state['tracer_names']}

    bgc = state['bgc']
    orig_m_l,  orig_m_q  = bgc.m_l,     bgc.m_q
    orig_zm_l, orig_zm_q = bgc.zoo_m_l, bgc.zoo_m_q
    bgc.m_l     = orig_m_l  * mort_amp
    bgc.m_q     = orig_m_q  * mort_amp
    bgc.zoo_m_l = orig_zm_l * mort_amp
    bgc.zoo_m_q = orig_zm_q * mort_amp
    ddt, _ = microbial_sms_omz(interior_tracers, bgc)
    bgc.m_l,     bgc.m_q     = orig_m_l,  orig_m_q
    bgc.zoo_m_l, bgc.zoo_m_q = orig_zm_l, orig_zm_q

    # ── 5. Chem tracer RHS  (advection + diffusion + bio) ────────────────────
    # 2nd-order upwind
    t_full = torch.stack([tracers_dict[n] for n in chem_names])
    t_pad  = torch.nn.functional.pad(t_full, (1, 1, 1, 1), mode='replicate')
    t_c2   = t_pad[..., 2:-2, 2:-2]
    t_e2   = t_pad[..., 3:-1, 2:-2]; t_ee = t_pad[..., 4:,   2:-2]
    t_w2   = t_pad[..., 1:-3, 2:-2]; t_ww = t_pad[..., :-4,  2:-2]
    t_n2   = t_pad[..., 2:-2, 3:-1]; t_nn = t_pad[..., 2:-2, 4:]
    t_s2   = t_pad[..., 2:-2, 1:-3]; t_ss = t_pad[..., 2:-2, :-4]

    u_vel = u_c.unsqueeze(0)
    v_vel = v_c.unsqueeze(0)

    adv_x = torch.where(u_vel > 0,
                        u_vel * (3.0 * t_c2 - 4.0 * t_w2 + t_ww) * state['inv_2dx'],
                        u_vel * (-t_ee + 4.0 * t_e2 - 3.0 * t_c2) * state['inv_2dx'])
    adv_y = torch.where(v_vel > 0,
                        v_vel * (3.0 * t_c2 - 4.0 * t_s2 + t_ss) * state['inv_2dy'],
                        v_vel * (-t_nn + 4.0 * t_n2 - 3.0 * t_c2) * state['inv_2dy'])
    tracer_advection = -(adv_x + adv_y)

    for i, name in enumerate(chem_names):
        rhs_t = tracer_advection[i] + state['K'] * lap[i + 1] + ddt[name] * bio_accel
        if name == 'doc':
            # POC → DOC hydrolysis: doc_flux_t = k_hyd_eff * poc(t), [bs,1,1]
            # k_hyd is already pre-scaled by bio_accel in setup_physics.
            # Deposit into particle mask only.
            rhs_t = rhs_t + doc_flux_t * state['particle_mask'][..., 1:-1, 1:-1]
        state['_rhs_tracers'][name][..., 1:-1, 1:-1] = rhs_t

    # ── 6. Bio tracer RHS  (SMS only — no advection, no diffusion) ───────────
    for name in bio_names:
        state['_rhs_tracers'][name][..., 1:-1, 1:-1] = ddt[name] * bio_accel

    return state['_rhs_buf_w'], state['_rhs_tracers']


# ── Optional torch.compile (CUDA only) ───────────────────────────────────────
if hasattr(torch, 'compile'):
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("⚠️  Skipping torch.compile (Apple Silicon MPS — no C++ JIT)")
    elif torch.cuda.is_available():
        try:
            get_rhs_batched = torch.compile(get_rhs_batched, mode='reduce-overhead')
            print("✅ torch.compile enabled for RHS physics")
        except Exception as e:
            print(f"⚠️  torch.compile skipped: {e}")
    else:
        print("⚠️  Skipping torch.compile (CPU-only)")