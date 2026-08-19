# config.py
use_symmetry = True   # Artificial symmetry enforcement to keep the model stable.
batch_size   = 1      # Default for main.py (overridden by run_suite.py)
snapshot_time = 1800.0   # Seconds between saved snapshots

# ── Suite / loop control flags ────────────────────────────────────────────────
is_suite                 = False
terminal_snapshot_only   = False
extrapolate_steady_state = False
bio_skipping             = True   # True = Asynchronous bio-skipping; False = Fully-coupled synchronous mode after flow freezes
# macro_cycle_time = 1905.0
macro_cycle_time = 5000
intermittent_physics_flush_time = 750.0


extended_physics_flush_at_end = True
if not bio_skipping:
    extended_physics_flush_at_end = False   # No need to run extra physics if bio is fully coupled
final_flush_duration = 1000.0

# ── Particle Parameters ───────────────────────────────────────────────────────
radius = 0.97   # mm

# ── POC initialisation ────────────────────────────────────────────────────────
# POC is the ONLY carbon source. It hydrolyses at rate k_hyd (in biopar.py)
# via:  doc_flux(t) = k_hyd * poc_initial * exp(-k_hyd * t)
# Use Klawonn density override if True; otherwise uses Alldredge fractal scaling.
use_klawonn_density = True
poc_initial_core    = 1500000000.0   # mmol C m⁻³ (only used when use_klawonn_density=True)

# ── Biology acceleration (dimensionless multiplier on all SMS rates) ──────────
# Speeds up slow microbial dynamics to reach quasi-steady state in short runs.
# The same factor is applied to POC hydrolysis so the DOC supply keeps pace.
BIO_ACCEL = 1

# ── Domain (Scaled by radius) ─────────────────────────────────────────────────
# Lx = 20.0 * radius
Lx = 9 * radius
Ly = 9 * radius
# Nx, Ny = int(351), int(307)
# Nx, Ny = int(151), int(151)  # Reduced grid for faster testing
# Nx, Ny = int(161), int(161)  
Nx, Ny = int(101), int(101)  
# Nx, Ny = int(71), int(71)  # Reduced grid for faster testing

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
# cx = 5.0 * radius
cx = 3.0 * radius
cy = Ly / 2.0

# ── Target Dimensionless Numbers ──────────────────────────────────────────────
Sc_target = 660

# ── Derived Physics Parameters ────────────────────────────────────────────────
nu   = 1.04                               # Kinematic viscosity of seawater (mm² s⁻¹)
# U_bg = 0.0
U_bg = 1.6 * (radius / 1.0) ** 0.56      # Sinking speed (Omand 2020 fractal scaling)


Re_actual = (U_bg * 2.0 * radius) / nu
Pe_calc   = Re_actual * Sc_target
K         = nu / Sc_target                # Scalar diffusivity (mm² s⁻¹)
Sh        = 1 + 0.619 * Re_actual ** 0.412 * Sc_target ** (1.0 / 3.0)


# Total_Time = 30000.0
# Total_Time = 86400*14
Total_Time = 9 * 86400
# Total_Time = 1800
# Total_Time = 25000.0


print(f"\n── Simulation Physics ──")
print(f"Radius   | R: {radius}")
print(f"Targets  | Re: {Re_actual:.2f}  |  Sc: {Sc_target}  |  Pe: {Pe_calc:.2f}")
print(f"Derived  | U_bg: {U_bg:.3f} mm/s | nu: {nu:.2f} |  K: {K:.5f}")
print(f"Time     | Total_Time: {Total_Time:.2f} s")
print(f"────────────────────────\n")

# ── Time Stepping ─────────────────────────────────────────────────────────────
# target_CFL = 0.2
target_CFL = 0.4
# target_CFL = 0.5