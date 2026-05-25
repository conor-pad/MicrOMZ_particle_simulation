# main.py
import os
import warnings
import logging
import pickle

# Essential environment variables for M2 Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_LOGS"] = "recompiles"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore", message=".*resized since it had shape.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import config as cfg
from biopar import BioPar        
from physics import setup_physics
from loop import run_simulation
from plotting import generate_plots

def print_damkohler_diagnostics(cfg, bgc):
    print(f"\n{'─'*45}")
    print(f"── Internal Damköhler Diagnostics (Da_II) ──")
    print(f"   (Reaction vs. Diffusion: Da = k*R^2 / K)")
    print(f"{'─'*45}")
    
    # Quick helper function to keep the math clean
    def calc_da2(k):
        return (k * cfg.radius**2) / cfg.K

    # ── J-OXIC (Aerobic Reactions) ──
    da2_oxic_rem = calc_da2(bgc.krem)
    da2_ammox    = calc_da2(bgc.kAo)
    da2_nitrox   = calc_da2(bgc.kNo)
    
    # ── J-ANOXIC (Anaerobic Reactions) ──
    da2_den1    = calc_da2(bgc.kDen1)  # Nitrate reduction
    da2_den2    = calc_da2(bgc.kDen2)  # Nitrite reduction
    da2_den3    = calc_da2(bgc.kDen3)  # N2O reduction
    da2_anammox = calc_da2(bgc.kAx)    # Anammox
    
    print(f"Oxic Respiration   (O2 consumption) : {da2_oxic_rem:<7.2f}")
    print(f"Ammonium Oxidation (NH4 -> NO2)     : {da2_ammox:<7.2f}")
    print(f"Nitrite Oxidation  (NO2 -> NO3)     : {da2_nitrox:<7.2f}")
    print(f"{'-'*45}")
    print(f"Denitrification 1  (NO3 -> NO2)     : {da2_den1:<7.2f}")
    print(f"Denitrification 2  (NO2 -> N2O)     : {da2_den2:<7.2f}")
    print(f"Denitrification 3  (N2O -> N2)      : {da2_den3:<7.2f}")
    print(f"{'-'*45}")
    print(f"Anammox            (NH4+NO2 -> N2)  : {da2_anammox:<7.2f}")
    
    print(f"{'─'*45}\n")
# ───────────────────────────────────────────

def main():
    device, state = setup_physics(cfg)
    bgc_params = BioPar()
    print_damkohler_diagnostics(cfg, bgc_params)

    cache_file = "simulation_cache.pkl"
    
    # Toggle this to True if you changed the physics and NEED to overwrite the save!
    force_rerun = True 

    if os.path.exists(cache_file) and not force_rerun:
        print(f"\n🚀 Loading cached simulation data from {cache_file}...")
        print("Skipping physics. Going straight to plotting!")
        with open(cache_file, "rb") as f:
            results = pickle.load(f)
    else:
        # Run the heavy math
        results = run_simulation(state, cfg, device)
        
        # Save it for next time
        print(f"💾 Saving raw data to {cache_file} for instant plotting later...")
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)

    # Generate the MP4s and PNGs instantly
    generate_plots(*results, cfg)
    print("All processes complete!")


if __name__ == "__main__":
    main()