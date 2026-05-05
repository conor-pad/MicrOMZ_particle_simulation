# physics.py
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

# Import our new modular components
from bcs import inflow
from biopar import BioPar
from sms import nit_sms_omz

def setup_physics(cfg):
    """Initializes the grid, calculates dt, and pre-allocates all GPU tensors."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon GPU (MPS) detected! Running on Metal.")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS not found, falling back to CPU.")

    x = np.linspace(0, cfg.Lx, cfg.Nx)
    y = np.linspace(0, cfg.Ly, cfg.Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    psi_bg_np = cfg.U_bg * Y

    # Calculate the advective timescale across one grid cell
    t_adv_cell = min(cfg.dx, cfg.dy) / cfg.U_bg
    
    # Analytically scale alpha to be exactly 30x stronger than advection
    max_alpha = 30.0 / t_adv_cell
    cfg.drag_max = max_alpha

    # Circular Particle (Drag Mask)
    drag_mask_np = np.zeros_like(X)
    particle_mask_np = np.zeros_like(X)
    particle_idx = (X - cfg.cx)**2 + (Y - cfg.cy)**2 <= cfg.radius**2
    
    drag_mask_np[particle_idx] = max_alpha
    particle_mask_np[particle_idx] = 1.0

    # 1. Advective limit (How fast the fluid moves)
    dt_adv  = cfg.target_CFL * min(cfg.dx, cfg.dy) / cfg.U_bg
    
    # 2. Drag limit (How fast the particle stops the fluid)
    dt_drag = 0.5 / max_alpha if max_alpha > 0 else float('inf')
    
    # 3. Scalar diffusion limit (How fast the chemicals spread)
    dt_diff = 0.25 * min(cfg.dx, cfg.dy)**2 / cfg.K
    
    # 4. Momentum diffusion limit (How fast the fluid shears/sticks)
    dt_visc = 0.25 * min(cfg.dx, cfg.dy)**2 / cfg.nu

    # Take the absolute smallest required time step!
    # Notice: The 0.0004 band-aid is gone! The analytical drag limit protects us now.
    dt, min_name = min((dt_adv, "dt_adv"), (dt_drag, "dt_drag"), (dt_diff, "dt_diff"), (dt_visc, "dt_visc"))

    print(f"Time step: {dt:.6f} (Source: {min_name})")
    # store so it can be put onto the plot
    cfg.dt = dt

    # Mirror the raw drag mask about y = Ny//2 before smoothing
    mid_y = cfg.Ny // 2
    drag_mask_np[:, :mid_y] = drag_mask_np[:, cfg.Ny - 1 - np.arange(mid_y)]  # force symmetry
    drag_mask_np = gaussian_filter(drag_mask_np, sigma=1.0)
    # Mirror again after smoothing to kill any Gaussian discretization asymmetry
    drag_mask_np[:, :mid_y] = drag_mask_np[:, -1:-mid_y-1:-1]


    # drag_mask_np = gaussian_filter(drag_mask_np, sigma=1.5)
    da_dx_np, da_dy_np = np.gradient(drag_mask_np, cfg.dx, cfg.dy)

    # Dictionary to hold all dynamic tensor states
    state = {}
    state['dt'] = dt
    state['nu'] = torch.tensor(cfg.nu, dtype=torch.float32, device=device)
    state['K']  = torch.tensor(cfg.K, dtype=torch.float32, device=device)
    state['inv_dx'] = torch.tensor(1.0 / cfg.dx, dtype=torch.float32, device=device)
    state['inv_dy'] = torch.tensor(1.0 / cfg.dy, dtype=torch.float32, device=device)
    state['doc_flux'] = torch.tensor(getattr(cfg, 'doc_flux_rate', 0.0), dtype=torch.float32, device=device)
    
    # Instantiate Biological Parameters
    state['bgc'] = BioPar()
    
    # Tracer names for easy looping
    # tracer_names = ['o2', 'no3', 'doc', 'po4', 'n2o', 'nh4', 'no2', 'n2']
    tracer_names = ['o2', 'no3', 'doc', 'po4', 'n2o', 'n2o_ammox', 'n2o_denit', 'nh4', 'no2', 'n2']
    
    state['tracer_names'] = tracer_names
    
    # Push Static Arrays to GPU
    state['psi_bg']         = torch.tensor(psi_bg_np, dtype=torch.float32, device=device)
    state['drag_mask']      = torch.tensor(drag_mask_np, dtype=torch.float32, device=device)
    state['da_dx']          = torch.tensor(da_dx_np, dtype=torch.float32, device=device)
    state['da_dy']          = torch.tensor(da_dy_np, dtype=torch.float32, device=device)
    state['particle_mask']  = torch.tensor(particle_mask_np, dtype=torch.float32, device=device)

    # Allocate Vorticity
    state['w']          = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)
    state['_rhs_buf_w'] = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)

    # ── Allocate Tracers ───────────────────────────────────────────────────────
    state['tracers'] = {}
    state['_rhs_tracers'] = {}
    
    for name in tracer_names:
        init_val = getattr(inflow, name)
        
        if name == 'doc':
            # Ambient water has 0 DOC
            starting_doc = getattr(cfg, 'doc_initial_core', 30.0)
            doc_initial = np.zeros((cfg.Nx, cfg.Ny), dtype=np.float32)
            doc_initial[particle_mask_np > 0] = starting_doc
            state['tracers'][name] = torch.tensor(doc_initial, device=device)
        else:
            # All other tracers (O2, NO3, etc.) fill the domain normally
            state['tracers'][name] = torch.full((cfg.Nx, cfg.Ny), init_val, dtype=torch.float32, device=device)
            
        state['_rhs_tracers'][name] = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)
    # ───────────────────────────────────────────────────────────────────────────

    # Pre-allocate full velocity tensors
    state['u_full'] = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)
    state['v_full'] = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.float32, device=device)

    # Precompute reciprocal constants
    state['inv_4dxdy'] = torch.tensor(1.0 / (4.0 * cfg.dx * cfg.dy), dtype=torch.float32, device=device)
    state['inv_dx2']   = torch.tensor(1.0 / cfg.dx**2,                dtype=torch.float32, device=device)
    state['inv_dy2']   = torch.tensor(1.0 / cfg.dy**2,                dtype=torch.float32, device=device)
    state['inv_2dy']   = torch.tensor(1.0 / (2.0 * cfg.dy),           dtype=torch.float32, device=device)
    state['inv_2dx']   = torch.tensor(1.0 / (2.0 * cfg.dx),           dtype=torch.float32, device=device)

    # Precompute Poisson eigenvalues
    Ni, Nj = cfg.Nx - 2, cfg.Ny - 2
    state['Ni'], state['Nj'] = Ni, Nj
    
    ii = torch.arange(1, Ni + 1, dtype=torch.float32, device=device)
    jj = torch.arange(1, Nj + 1, dtype=torch.float32, device=device)
    lam_x = (2 / cfg.dx**2) * (torch.cos(np.pi * ii / (Ni + 1)) - 1)
    lam_y = (2 / cfg.dy**2) * (torch.cos(np.pi * jj / (Nj + 1)) - 1)
    state['Lambda'] = lam_x[:, None] + lam_y[None, :]

    # Pre-allocated DST-I buffers 
    state['_dst_buf_axis0'] = torch.zeros((2 * (Ni + 1), Nj), dtype=torch.float32, device=device)
    state['_dst_buf_axis1'] = torch.zeros((Ni, 2 * (Nj + 1)), dtype=torch.float32, device=device)

    return device, state


def dstn1(x_in, state):
    n0, n1 = x_in.shape
    buf0 = state['_dst_buf_axis0']
    buf1 = state['_dst_buf_axis1']

    buf0.zero_()
    buf0[1:n0 + 1, :] = x_in
    buf0[n0 + 2:,  :] = -torch.flip(x_in, dims=[0])

    y0  = torch.fft.fft(buf0, dim=0)
    mid = -y0[1:n0 + 1].imag / float(np.sqrt(2 * (n0 + 1)))

    buf1.zero_()
    buf1[:, 1:n1 + 1] = mid
    buf1[:, n1 + 2:]  = -torch.flip(mid, dims=[1])

    y1 = torch.fft.fft(buf1, dim=1)
    return -y1[:, 1:n1 + 1].imag / float(np.sqrt(2 * (n1 + 1)))


def get_psi_pert(w_field, state):
    rhs       = -w_field[1:-1, 1:-1]
    rhs_hat   = dstn1(rhs, state)
    psi_inner = dstn1(rhs_hat / state['Lambda'], state)
    psi       = torch.zeros_like(w_field)
    psi[1:-1, 1:-1] = psi_inner
    return psi


def get_rhs_batched(w_f, tracers_dict, streamfunction, state, cfg):
    """Batched RHS for vorticity (Arakawa) and 8 tracers (Upwind) simultaneously."""
    p = streamfunction

    p_e  = p[2:,   1:-1];  p_w  = p[:-2,  1:-1]
    p_n  = p[1:-1, 2:];    p_s  = p[1:-1, :-2]
    p_ne = p[2:,   2:];    p_sw = p[:-2,  :-2]
    p_se = p[2:,   :-2];   p_nw = p[:-2,  2:]

    dp_ew    = p_e  - p_w
    dp_ns    = p_n  - p_s
    dp_ne_se = p_ne - p_se
    dp_nw_sw = p_nw - p_sw
    dp_ne_nw = p_ne - p_nw
    dp_se_sw = p_se - p_sw

    tracer_names = state['tracer_names']

    # Stack 9 total fields (Index 0 = Vorticity, Index 1-8 = Tracers)
    f_c_list  = [w_f[1:-1, 1:-1]] + [tracers_dict[name][1:-1, 1:-1] for name in tracer_names]
    f_e_list  = [w_f[2:,   1:-1]] + [tracers_dict[name][2:,   1:-1] for name in tracer_names]
    f_w_list  = [w_f[:-2,  1:-1]] + [tracers_dict[name][:-2,  1:-1] for name in tracer_names]
    f_n_list  = [w_f[1:-1, 2:  ]] + [tracers_dict[name][1:-1, 2:  ] for name in tracer_names]
    f_s_list  = [w_f[1:-1, :-2 ]] + [tracers_dict[name][1:-1, :-2 ] for name in tracer_names]
    
    # We only need the corners for the Arakawa vorticity scheme, not the tracers
    f_ne_list = [w_f[2:,   2:  ]] 
    f_sw_list = [w_f[:-2,  :-2 ]] 
    f_se_list = [w_f[2:,   :-2 ]] 
    f_nw_list = [w_f[:-2,  2:  ]] 

    f_c  = torch.stack(f_c_list)
    f_e  = torch.stack(f_e_list)
    f_w  = torch.stack(f_w_list)
    f_n  = torch.stack(f_n_list)
    f_s  = torch.stack(f_s_list)
    
    # Isolate just the vorticity field (index 0) for Arakawa
    w_c, w_e, w_w, w_n, w_s = f_c[0], f_e[0], f_w[0], f_n[0], f_s[0]
    w_ne, w_sw = f_ne_list[0], f_sw_list[0]
    w_se, w_nw = f_se_list[0], f_nw_list[0]

    # ── 1. Fluid Momentum: Arakawa Jacobian ──
    J_std   = (dp_ew * (w_n - w_s) - dp_ns * (w_e - w_w)) * state['inv_4dxdy']
    J_hat   = (p_e  * (w_ne - w_se) - p_w  * (w_nw - w_sw)
             - p_n  * (w_ne - w_nw) + p_s  * (w_se - w_sw)) * state['inv_4dxdy']
    J_tilde = (w_n  * dp_ne_nw - w_s  * dp_se_sw
             - w_e  * dp_ne_se + w_w  * dp_nw_sw) * state['inv_4dxdy']
    
    J_avg = (J_std + J_hat + J_tilde) / 3.0 
    
    # Standard Central Diffusion (Laplacian) applies to EVERYTHING
    lap = (f_e + f_w - 2.0 * f_c) * state['inv_dx2'] + (f_n + f_s - 2.0 * f_c) * state['inv_dy2']

    # ── 2. Calculate Local Velocities ──
    u_c   = dp_ns * state['inv_2dy']
    v_c   = -dp_ew * state['inv_2dx']

    # ── Vorticity Drag & Final RHS ──
    dm  = state['drag_mask'][1:-1, 1:-1]
    dax = state['da_dx'][1:-1, 1:-1]
    day = state['da_dy'][1:-1, 1:-1]
    
    drag  = dm * w_c - (day * u_c - dax * v_c)
    rhs_w = J_avg + state['nu'] * lap[0] - drag
    state['_rhs_buf_w'][1:-1, 1:-1] = rhs_w

    # ── 3. Tracer Advection: 1st-Order Upwind Scheme ──
    # Isolate just the stacked tracers (indices 1 through 8)
    t_c, t_e, t_w, t_n, t_s = f_c[1:], f_e[1:], f_w[1:], f_n[1:], f_s[1:]
    
    # Broadcast velocities so they can multiply the entire stack of 8 tracers at once
    u_vel = u_c.unsqueeze(0)
    v_vel = v_c.unsqueeze(0)
    
    # # Pre-cast the floats to 32-bit so the compiler doesn't panic
    # inv_dx = float(1.0 / cfg.dx)
    # inv_dy = float(1.0 / cfg.dy)
    
    # Use > 0.0 instead of > 0 to prevent integer-to-float64 promotion
    adv_x = torch.where(u_vel > 0, u_vel * (t_c - t_w) * state['inv_dx'], u_vel * (t_e - t_c) * state['inv_dx'])
    adv_y = torch.where(v_vel > 0, v_vel * (t_c - t_s) * state['inv_dy'], v_vel * (t_n - t_c) * state['inv_dy'])
    
    # The advective flux leaving the cell
    tracer_advection = -(adv_x + adv_y)

    # ── 4. Biogeochemistry SMS Terms ──
    interior_tracers = {name: tracers_dict[name][1:-1, 1:-1] for name in tracer_names}
    ddt, _ = nit_sms_omz(interior_tracers, state['bgc'])

    # ── Tracers Final RHS ──
    for i, name in enumerate(tracer_names):
        
        # Upwind Advection + Central Diffusion + Microbial Consumption
        rhs_t = tracer_advection[i] + state['K'] * lap[i+1] + ddt[name]
        
        if name == 'doc':
            rhs_t += state['doc_flux'] * state['particle_mask'][1:-1, 1:-1]
            
        state['_rhs_tracers'][name][1:-1, 1:-1] = rhs_t

    return state['_rhs_buf_w'], state['_rhs_tracers']


def bilinear_interp_gpu(field, px_t, py_t, cfg):
    ix = torch.clamp((px_t / cfg.dx).long(), 0, cfg.Nx - 2)
    iy = torch.clamp((py_t / cfg.dy).long(), 0, cfg.Ny - 2)
    fx = px_t / cfg.dx - ix.float()
    fy = py_t / cfg.dy - iy.float()
    return (field[ix,     iy    ] * (1 - fx) * (1 - fy)
          + field[ix + 1, iy    ] * fx        * (1 - fy)
          + field[ix,     iy + 1] * (1 - fx)  * fy
          + field[ix + 1, iy + 1] * fx         * fy)

# # Apply Torch compile if available
# if hasattr(torch, 'compile'):
#     try:
#         get_rhs_batched = torch.compile(get_rhs_batched) 
#         print("✅ torch.compile enabled for RHS physics (inductor backend)")
#     except Exception as e:
#         print(f"⚠️  torch.compile skipped: {e}")

if hasattr(torch, 'compile'):
    try:
        get_rhs_batched = torch.compile(
            get_rhs_batched,
            fullgraph=True,       # Fails loudly if there are graph breaks (better than silent slowdowns)
            mode="reduce-overhead" # Reduces Python dispatch overhead per call — ideal for a loop
        )
        print("✅ torch.compile enabled for RHS physics")
    except Exception as e:
        print(f"⚠️  torch.compile skipped: {e}")

        