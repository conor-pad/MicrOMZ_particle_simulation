# bcs.py
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class InflowBCs:
    """
    Boundary and initial conditions for biological variables.
    Values represent concentrations in mmol/m3
    """
    o2:  float = 10#11
    no3: float = 10.0
    doc: float = 0 # 40

    po4: float = 0.0
    n2o: float = 0.0
    n2o_ammox: float = 0.0
    n2o_denit: float = 0.0
    nh4: float = 0.5
    no2: float = 0.0
    n2:  float = 0.0

    poc: float = 0.0               # Ambient water has no solid POC

    # Microbial Functional Types (mmol C m⁻³)
    # These are initial conditions only
    aer: float = 4
    nar: float = 4
    nai: float = 4
    nao: float = 4
    nir: float = 4
    nio: float = 4
    nos: float = 4
    aoa: float = 2 # late arrival chemotroph
    nob: float = 2 # late arrival chemotroph
    aox: float = 2 # late arrival chemotroph
    zoo: float = 0

inflow = InflowBCs()

# Only these get a constant inflow value at the left boundary.
# Bio tracers are excluded — they live on the particle, not the open ocean.
_CHEM_INFLOW_NAMES = frozenset([
    'o2', 'no3', 'doc', 'po4', 'n2o', 'n2o_ammox', 'n2o_denit', 'nh4', 'no2', 'n2'
])

def apply_bcs(w_f, tracers_dict):
    """
    Applies boundary conditions to the vorticity field and all tracers.
    """
    # ── Vorticity Boundary Conditions ──
    w_f[..., :, 0]  = 0.0
    w_f[..., :, -1] = 0.0
    w_f[..., 0, :]  = 0.0
    w_f[..., -1, :] = w_f[..., -2, :]
    
    # ── Tracer Boundary Conditions ──
    for name, tensor in tracers_dict.items():
        if name in _CHEM_INFLOW_NAMES:
            inflow_value = getattr(inflow, name)
            # Safely convert numpy array to a PyTorch tensor
            if isinstance(inflow_value, np.ndarray):
                inflow_value = torch.tensor(inflow_value, dtype=tensor.dtype, device=tensor.device)
            tensor[..., 0, :] = inflow_value   # Left: constant inflow
        else:
            tensor[..., 0, :] = tensor[..., 1, :]  # Left: zero-gradient (bio tracers)

        tensor[..., -1, :] = tensor[..., -2, :]    # Right:  zero-gradient (outflow)
        tensor[..., :, 0]  = tensor[..., :, 1]     # Bottom: zero-gradient
        tensor[..., :, -1] = tensor[..., :, -2]    # Top:    zero-gradient

        # don't let it go negative (2nd order overshoots sometimes)
        tracers_dict[name] = torch.clamp(tensor, min=0.0)

    return w_f, tracers_dict


def enforce_symmetry(w, tracers, tracer_names):
    """
    Projects solution onto symmetric subspace each step.
    """
    Ny  = w.shape[-1]
    mid = Ny // 2  

    # ── Vorticity ──
    w_top    = w[..., mid + 1:]          
    w_bot    = w[..., :mid].flip(dims=[-1])  

    w_sym = (w_top - w_bot) * 0.5     
    w[..., mid + 1:] =  w_sym
    w[..., :mid]     = -w_sym.flip(dims=[-1])
    w[..., mid]      =  0.0             

    # ── Tracers ──
    for name in tracer_names:
        t     = tracers[name]
        t_top = t[..., mid + 1:]
        t_bot = t[..., :mid].flip(dims=[-1])

        t_sym = (t_top + t_bot) * 0.5
        t[..., mid + 1:] = t_sym
        t[..., :mid]     = t_sym.flip(dims=[-1])

    return w, tracers