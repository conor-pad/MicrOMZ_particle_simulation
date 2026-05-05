# bcs.py
import torch
from dataclasses import dataclass

@dataclass
class InflowBCs:
    """
    Boundary and initial conditions for biological variables.
    Converted from hab_initialize_omz.m
    Values represent concentrations in mmol/m3
    """
    o2:  float = 6
    no3: float = 10.0
    doc: float = 0 # 17.2   # 10 * 1.72 (approx conversion to OM)

    # DOC: treat as it's STATIC inside the particle, not flowing in.

    po4: float = 0.0
    n2o: float = 0.0
    n2o_ammox: float = 0.0
    n2o_denit: float = 0.0
    nh4: float = 0.0
    no2: float = 0.0
    n2:  float = 0.0

    poc: float = 0.0               # Ambient water has no solid POC
    poc_core: float = 1000000.0    # Solid core density (mmol C / m^3). Used to calculate 2D Area.

# Instantiate the inflow values so other files can import and use them
inflow = InflowBCs()

def apply_bcs(w_f, tracers_dict):
    """
    Applies boundary conditions to the vorticity field and all tracers.
    
    w_f: 2D PyTorch tensor for vorticity
    tracers_dict: Dictionary of 2D PyTorch tensors {'o2': tensor, 'no3': tensor, ...}
    """
    # ── Vorticity Boundary Conditions ──
    w_f[:, 0]  = 0.0
    w_f[:, -1] = 0.0
    w_f[0, :]  = 0.0
    w_f[-1, :] = w_f[-2, :]
    
    # ── Tracer Boundary Conditions ──
    for name, tensor in tracers_dict.items():
        # Grab the specific inflow value for this tracer from the dataclass
        inflow_value = getattr(inflow, name)
        
        # Left boundary: Constant inflow
        tensor[0, :] = inflow_value
        
        # Right boundary: Zero gradient (outflow)
        tensor[-1, :] = tensor[-2, :]
        
        # Top and Bottom boundaries: Free slip / zero gradient
        tensor[:, 0]  = tensor[:, 1]
        tensor[:, -1] = tensor[:, -2]
        
    return w_f, tracers_dict


def enforce_symmetry(w, tracers, tracer_names):
    """
    Projects solution onto symmetric subspace each step.
    - Vorticity: odd (antisymmetric) about horizontal midline
    - Tracers:   even (symmetric)  about horizontal midline
    Ny=287 is odd, so index 143 is exactly the midline.
    """
    Ny  = w.shape[1]
    mid = Ny // 2  # = 143 for Ny=287

    # ── Vorticity (antisymmetric: w_top = -w_bot_mirrored) ──
    w_top    = w[:, mid + 1:]          # shape (Nx, 143)
    w_bot    = w[:, :mid].flip(dims=[1])  # flipped to align with top

    w_sym = (w_top - w_bot) * 0.5     # antisymmetric average
    w[:, mid + 1:] =  w_sym
    w[:, :mid]     = -w_sym.flip(dims=[1])
    w[:, mid]      =  0.0             # midline must be zero for odd field

    # ── Tracers (symmetric: c_top = c_bot_mirrored) ──
    for name in tracer_names:
        t     = tracers[name]
        t_top = t[:, mid + 1:]
        t_bot = t[:, :mid].flip(dims=[1])

        t_sym = (t_top + t_bot) * 0.5
        t[:, mid + 1:] = t_sym
        t[:, :mid]     = t_sym.flip(dims=[1])
        # midline is self-consistent for even fields; no change needed

    return w, tracers
