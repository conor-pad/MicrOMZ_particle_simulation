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
      POC is the ONLY carbon source, and is now a formal tracer (in its own
      no-transport category alongside bio_names) rather than an analytical
      decay — being solid particulate matter bound to the sinking aggregate,
      it does NOT advect or diffuse. It hydrolyses via a biomass-driven,
      saturating flux computed every timestep inside get_rhs_batched():
          doc_flux = k_hyd_max * total_heterotroph_biomass * POC / (K_POC + POC)
      This flux is added to DOC (which does advect/diffuse normally) and
      subtracted from POC in place. There is NO doc_flux_rate config
      parameter and NO doc_initial_core.

    The same BIO_ACCEL factor is applied to k_hyd_max so the POC→DOC supply
    keeps pace with the accelerated microbial consumption rates.
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
    tracer_names = chem_names + bio_names + ['poc']

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

    # ── Isotropic 9-point Laplacian coefficient ────────────────────────────────
    # ∇²f ≈ [4(f_E+f_W+f_N+f_S) + (f_NE+f_NW+f_SE+f_SW) - 20 f_C] / (6h²)
    # This weighting only cancels the leading grid-orientation error term when
    # cells are square (dx == dy) — true here since Nx==Ny and Lx==Ly, but
    # asserted explicitly so this breaks loudly rather than silently if that
    # ever changes.
    assert np.all(np.abs(dx_b - dy_b) < 1e-12 * np.maximum(dx_b, dy_b)), \
        "Isotropic 9-point Laplacian assumes dx == dy (square cells); got dx=%s, dy=%s" % (dx_b.ravel(), dy_b.ravel())
    state['inv_h2_iso'] = torch.tensor(1.0 / (6.0 * dx_b ** 2), dtype=torch.float32, device=device)
    state['bio_accel']    = float(getattr(cfg, 'BIO_ACCEL', 100_000.0))
    state['mort_amp']     = torch.tensor(b_arr(getattr(cfg, 'MORT_AMP', 1.0)),
                                          dtype=torch.float32, device=device)

    state['bgc']          = BioPar()
    state['chem_names']   = chem_names
    state['bio_names']    = bio_names
    state['tracer_names'] = tracer_names

    # ── POC hydrolysis ────────────────────────────────────────────────────────
    V_mm3    = np.pi * (radius_b ** 2) * 1.0          
    poc_ug   = 0.99 * (V_mm3 ** 0.52)                 
    poc_mmol_alldredge = poc_ug / 12_010               
    poc_density_alldredge = poc_mmol_alldredge / (V_mm3 * 1e-9)  

    if getattr(cfg, 'use_klawonn_density', False) and hasattr(cfg, 'poc_initial_core'):
        poc_initial_density = (np.ones((bs, 1, 1), dtype=np.float32)
                               * float(cfg.poc_initial_core))
    else:
        poc_initial_density = poc_density_alldredge.astype(np.float32)

    k_hyd_max_raw = float(state['bgc'].k_hyd_max)        
    k_hyd_max_eff = k_hyd_max_raw * state['bio_accel']   

    state['k_hyd_max'] = torch.tensor(k_hyd_max_eff,          dtype=torch.float32, device=device)
    state['K_POC']     = torch.tensor(state['bgc'].K_POC,     dtype=torch.float32, device=device)
    print(f"POC | density={poc_initial_density.flat[0]:.2e} mmol/m³  "
          f"k_hyd_max_raw={k_hyd_max_raw:.2e}/s  k_hyd_max_eff={k_hyd_max_eff:.2e}/s  "
          f"K_POC={state['bgc'].K_POC:.2e}")

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
            state['tracers'][name] = torch.zeros(
                (bs, cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)

        elif name == 'poc':
            arr = poc_initial_density * particle_mask_np
            state['tracers'][name] = torch.tensor(
                arr.astype(np.float32), device=device)

        elif name in set(bio_names):
            arr = np.zeros((bs, cfg.Nx, cfg.Ny), dtype=np.float32)
            arr[particle_mask_np > 0] = float(init_val)
            state['tracers'][name] = torch.tensor(arr, device=device)

        else:
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
    state['Lambda'] = lam_x + lam_y   

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
def get_rhs_bio_only(tracers_dict, state, cfg):
    """Standalone biological RHS for fast-forwarding without physics."""
    interior = {n: tracers_dict[n][..., 1:-1, 1:-1] for n in state['tracer_names']}
    total_het = sum(interior[b] for b in ('aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos'))
    
    poc = interior['poc']
    doc_flux = state['k_hyd_max'] * total_het * (poc / (state['K_POC'] + poc))
    
    bgc = state['bgc']
    orig_m_l, orig_m_q = bgc.m_l, bgc.m_q
    orig_zm_l, orig_zm_q = bgc.zoo_m_l, bgc.zoo_m_q
    bgc.m_l = orig_m_l * state['mort_amp']
    bgc.m_q = orig_m_q * state['mort_amp']
    bgc.zoo_m_l = orig_zm_l * state['mort_amp']
    bgc.zoo_m_q = orig_zm_q * state['mort_amp']
    
    ddt, gross_growth = microbial_sms_omz(interior, bgc)
    
    bgc.m_l, bgc.m_q = orig_m_l, orig_m_q
    bgc.zoo_m_l, bgc.zoo_m_q = orig_zm_l, orig_zm_q
    
    bio_accel = state['bio_accel']
    rhs = {}
    for n in state['tracer_names']:
        if n in state['chem_names']:
            rhs[n] = ddt[n] * bio_accel
            if n == 'doc':
                rhs[n] = rhs[n] + doc_flux
        elif n in state['bio_names']:
            rhs[n] = ddt[n] * bio_accel
        elif n == 'poc':
            rhs[n] = -doc_flux
            
    return rhs, gross_growth


def get_rhs_batched(w_f, tracers_dict, streamfunction, state, cfg, compute_bio=True):    
    """
    Batched RHS for vorticity (Arakawa) and tracers.

    POC → DOC hydrolysis is biomass-driven and saturating, computed internally
    here from the current heterotroph biomass and POC concentration:
        doc_flux = k_hyd_max * total_heterotroph_biomass * POC / (K_POC + POC)
    k_hyd_max is already pre-scaled by bio_accel in setup_physics. doc_flux is
    added to DOC's RHS (DOC advects/diffuses normally) and subtracted from
    POC's RHS. POC itself gets NO advection or diffusion — like the bio
    tracers, it's solid material bound to the particle; hydrolysis is its
    only source/sink term.

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

    # Diagonal neighbors — needed for the isotropic 9-point Laplacian below.
    # Vorticity's diagonals (w_ne/w_nw/w_se/w_sw) already get computed a few
    # lines down for the Arakawa Jacobian; stacking them here alongside the
    # tracer diagonals means that slicing is shared/reused, not duplicated.
    f_ne_list = [w_f[..., 2:,  2:  ]] + [tracers_dict[n][..., 2:,  2:  ] for n in chem_names]
    f_nw_list = [w_f[..., :-2, 2:  ]] + [tracers_dict[n][..., :-2, 2:  ] for n in chem_names]
    f_se_list = [w_f[..., 2:,  :-2 ]] + [tracers_dict[n][..., 2:,  :-2 ] for n in chem_names]
    f_sw_list = [w_f[..., :-2, :-2 ]] + [tracers_dict[n][..., :-2, :-2 ] for n in chem_names]

    w_ne = w_f[..., 2:,  2:  ]; w_sw = w_f[..., :-2, :-2]
    w_se = w_f[..., 2:,  :-2 ]; w_nw = w_f[..., :-2, 2:  ]

    f_c = torch.stack(f_c_list)
    f_e = torch.stack(f_e_list)
    f_w = torch.stack(f_w_list)
    f_n = torch.stack(f_n_list)
    f_s = torch.stack(f_s_list)
    f_ne = torch.stack(f_ne_list)
    f_nw = torch.stack(f_nw_list)
    f_se = torch.stack(f_se_list)
    f_sw = torch.stack(f_sw_list)

    w_c, w_e, w_w, w_n, w_s = f_c[0], f_e[0], f_w[0], f_n[0], f_s[0]

    # ── 1. Arakawa Jacobian ───────────────────────────────────────────────────
    J_std   = (dp_ew * (w_n - w_s) - dp_ns * (w_e - w_w)) * state['inv_4dxdy']
    J_hat   = (p_e  * (w_ne - w_se) - p_w  * (w_nw - w_sw)
             - p_n  * (w_ne - w_nw) + p_s  * (w_se - w_sw)) * state['inv_4dxdy']
    J_tilde = (w_n  * dp_ne_nw - w_s  * dp_se_sw
             - w_e  * dp_ne_se + w_w  * dp_nw_sw) * state['inv_4dxdy']
    J_avg   = (J_std + J_hat + J_tilde) / 3.0

    # ── Isotropic 9-point Laplacian ───────────────────────────────────────────
    # ∇²f ≈ [4(N+S+E+W) + (NE+NW+SE+SW) - 20 f_C] / (6h²)
    # Cancels the leading grid-orientation error term the plain 5-point
    # stencil has, which is what was producing the boxy/diamond-shaped
    # depletion pattern at the particle's diagonal "shoulders" (advection
    # is separately still dimensionally split, so this only fixes the
    # diffusion term's contribution to that anisotropy, not all of it).
    lap = (4.0 * (f_e + f_w + f_n + f_s) + (f_ne + f_nw + f_se + f_sw)
           - 20.0 * f_c) * state['inv_h2_iso']

    # ── 2. Local velocities ───────────────────────────────────────────────────
    u_c =  dp_ns * state['inv_2dy']
    v_c = -dp_ew * state['inv_2dx']

    # ── 3. Vorticity explicit RHS ─────────────────────────────────────────────
    dax = state['da_dx'][..., 1:-1, 1:-1]
    day = state['da_dy'][..., 1:-1, 1:-1]
    rhs_w = J_avg + (day * u_c - dax * v_c)
    state['_rhs_buf_w'][..., 1:-1, 1:-1] = rhs_w

    # ── 4. SMS ────────────────────────────────────────────────────────────────
    if compute_bio:
        interior_tracers = {n: tracers_dict[n][..., 1:-1, 1:-1] for n in state['tracer_names']}
        total_het_biomass = sum(interior_tracers[b] for b in ('aer', 'nar', 'nai', 'nao', 'nir', 'nio', 'nos'))
        poc_interior = interior_tracers['poc']
        doc_flux = state['k_hyd_max'] * total_het_biomass * (poc_interior / (state['K_POC'] + poc_interior))

        bgc = state['bgc']
        orig_m_l,  orig_m_q  = bgc.m_l,     bgc.m_q
        orig_zm_l, orig_zm_q = bgc.zoo_m_l, bgc.zoo_m_q
        bgc.m_l     = orig_m_l  * state['mort_amp']
        bgc.m_q     = orig_m_q  * state['mort_amp']
        bgc.zoo_m_l = orig_zm_l * state['mort_amp']
        bgc.zoo_m_q = orig_zm_q * state['mort_amp']
        ddt, _ = microbial_sms_omz(interior_tracers, bgc)
        bgc.m_l,     bgc.m_q     = orig_m_l,  orig_m_q
        bgc.zoo_m_l, bgc.zoo_m_q = orig_zm_l, orig_zm_q

    # ── 5. Chem tracer RHS  (advection + diffusion + bio) ────────────────────
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
    
    for i, name in enumerate(state['chem_names']):
        rhs_t = tracer_advection[i] + state['K'] * lap[i + 1]
        if compute_bio:
            rhs_t = rhs_t + (ddt[name] * state['bio_accel'])
            if name == 'doc':
                rhs_t = rhs_t + doc_flux
        state['_rhs_tracers'][name][..., 1:-1, 1:-1] = rhs_t

    # ── 6 & 7. Bio & POC tracer RHS ──────────────────────────────────────────
    if compute_bio:
        for name in state['bio_names']:
            state['_rhs_tracers'][name][..., 1:-1, 1:-1] = ddt[name] * state['bio_accel']
        state['_rhs_tracers']['poc'][..., 1:-1, 1:-1] = -doc_flux
    else:
        for name in state['bio_names']:
            state['_rhs_tracers'][name][..., 1:-1, 1:-1] = 0.0
        state['_rhs_tracers']['poc'][..., 1:-1, 1:-1] = 0.0

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