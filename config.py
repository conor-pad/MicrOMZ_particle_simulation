# config.py
# ── Particle Parameters ───────────────────────────────────────────────────────
radius = 1

# ── Domain (Scaled by radius) ─────────────────────────────────────────────────
Lx = 20.0 * radius  
Ly = 10 * radius  
Nx, Ny = int(331), int(287)
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
cx = 5.0 * radius   
cy = Ly / 2.0       

# ── DOC ───────────────────────────────────────────────────────────────────────
doc_flux_rate = 1.0       # Hydrolysis rate: solid POC to dissolved DOC (mmol/m^3/s)
doc_initial_core = 0.0    # Initial DOC concentration inside the particle

# ── Target Dimensionless Numbers ──────────────────────────────────────────────
Sc_target = 660   

# ── Derived Physics Parameters ────────────────────────────────────────────────
nu = 1.04  # Physically realistic kinematic viscosity for seawater (mm^2/s)

# Calculate required background velocity
U_bg = 2.2 * (radius / 1.0)**0.56

Re_actual = (U_bg * (2.0 * radius)) / nu
Pe_calc = Re_actual * Sc_target  

# Calculate total time based on 5x the domain length
Total_Time = 15#5 * Lx / U_bg

# 1. Calculate the true physical diffusivity
K = nu / Sc_target

# 2. Calculate the absolute minimum diffusivity required to stop the grid from exploding
# Formula derived from forcing Pe_grid = (U_bg * dx) / K <= 2.0
# K_stable = (U_bg * dx) / 2.0  

# 3. Use whichever is larger!
# K = max(K_ideal, K_stable)

Sh = 1 + 0.619 * Re_actual ** 0.412 * Sc_target**(1/3)

print(f"\n── Simulation Physics ──")
print(f"Radius   | R: {radius}")
print(f"Targets  | Re: {Re_actual:.2f}  |  Sc: {Sc_target}  |  Pe: {Pe_calc:.2f}")
print(f"Derived  | U_bg: {U_bg:.3f} mm/s | nu: {nu:.2f} |  K:  {K:.4f}")
print(f"Time     | Total_Time: {Total_Time:.2f} s")
print(f"────────────────────────\n")

# ── Time Stepping ─────────────────────────────────────────────────────────────
target_CFL = 0.2