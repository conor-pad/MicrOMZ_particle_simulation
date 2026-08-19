# error_sweep_2d_highres.py
import importlib
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil
import os
import re

import config
from physics import setup_physics
import loop  
import biopar

def compute_relative_l2(test_field, true_field):
    return np.linalg.norm(test_field - true_field) / (np.linalg.norm(true_field) + 1e-12)

def patch_loop_py(limit_val):
    with open('loop.py', 'r') as f:
        content = f.read()
    content = re.sub(r'(\.min\(\)\.item\(\)\s*\*\s*)[\d\.]+', rf'\g<1>{limit_val}', content)
    with open('loop.py', 'w') as f:
        f.write(content)

def patch_biopar_py(rate_val):
    with open('biopar.py', 'r') as f:
        content = f.read()
    content = re.sub(r'(rate_amplifier:\s*float\s*=\s*)[\d\.]+', rf'\g<1>{rate_val}', content)
    with open('biopar.py', 'w') as f:
        f.write(content)

def main():
    csv_filename = "error_sweep_2d_highres.csv"
    plot_filename = "error_sweep_2d_highres.png"
    
    shutil.copyfile('loop.py', 'loop_backup.py')
    shutil.copyfile('biopar.py', 'biopar_backup.py')

    limits = np.linspace(0.1, 0.9, 20)
    macros = np.linspace(100.0, 5000.0, 20)
    
    error_matrix = np.zeros((len(macros), len(limits)))

    try:
        spin_keys = [k for k in dir(config) if 'spin' in k.lower() or 'freeze' in k.lower()]
        SPINUP_TIME = getattr(config, spin_keys[0]) if spin_keys else 0.0
        
        # Time required for advection/diffusion to re-equilibrate the domain
        FLUSH_TIME = 20.0 
        
        print(f"🔍 Detected Spin-Up Time: {SPINUP_TIME}s")

        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Macro_Cycle", "Depletion_Limit", "Relative_L2_Error"])

        patch_biopar_py(50.0)
        importlib.reload(biopar)

        print("\n🚀 Starting High-Res Dynamic 2D Sweep (20x20)...")
        for i, mac in enumerate(macros):
            
            TOTAL_TIME = SPINUP_TIME + mac + FLUSH_TIME
            print(f"\n--- Generating {TOTAL_TIME}s Baseline for Macro: {mac:.1f}s ---")
            
            importlib.reload(loop)
            importlib.reload(config)
            
            config.Total_Time = TOTAL_TIME
            config.bio_skipping = False
            config.extended_physics_flush_at_end = False 
            config.BIO_ACCEL = 1  
            
            with torch.no_grad():
                device, state = setup_physics(config)
                true_results = loop.run_simulation(state, config, device)
                true_final_aer = np.copy(true_results[13]['aer'][-1])

                del state, true_results
                if torch.backends.mps.is_available(): torch.mps.empty_cache()
                elif torch.cuda.is_available(): torch.cuda.empty_cache()

            for j, lim in enumerate(limits):
                print(f"   -> Testing Limit: {lim:.3f}")
                
                patch_loop_py(lim)
                importlib.reload(loop)
                
                config.Total_Time = TOTAL_TIME
                config.bio_skipping = True
                config.macro_cycle_time = mac
                
                config.extended_physics_flush_at_end = True 
                config.final_flush_duration = FLUSH_TIME 
                config.BIO_ACCEL = 1  
                
                with torch.no_grad():
                    device, state = setup_physics(config)
                    res = loop.run_simulation(state, config, device)
                    
                    err = compute_relative_l2(res[13]['aer'][-1], true_final_aer)
                    error_matrix[i, j] = err
                    
                    with open(csv_filename, mode="a", newline="") as f:
                        csv.writer(f).writerow([mac, lim, err])
                        
                    del state, res
                    if torch.backends.mps.is_available(): torch.mps.empty_cache()
                    elif torch.cuda.is_available(): torch.cuda.empty_cache()

        print("\n📊 Generating High-Res 2D Heatmap...")
        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.imshow(error_matrix, cmap='viridis_r', aspect='auto')
        
        ax.set_xticks(np.arange(len(limits)))
        ax.set_yticks(np.arange(len(macros)))
        ax.set_xticklabels([f"{val:.2f}" for val in limits], rotation=45)
        ax.set_yticklabels([f"{val:.0f}" for val in macros])
        
        ax.set_xlabel("Depletion Limit (Fraction)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Macro Cycle Time (s)", fontsize=12, fontweight='bold')
        ax.set_title("Relative L2 Error (Aerobe Biomass)\nSpin-up + Frozen Physics Phase + 1 Synchronous Phase", fontsize=15, fontweight='bold')

        fig.colorbar(cax, label='Error Fraction')
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        print(f"\n✅ High-res sweep complete! Saved to '{plot_filename}'.")

    finally:
        if os.path.exists('loop_backup.py'):
            shutil.copyfile('loop_backup.py', 'loop.py')
            os.remove('loop_backup.py')
        if os.path.exists('biopar_backup.py'):
            shutil.copyfile('biopar_backup.py', 'biopar.py')
            os.remove('biopar_backup.py')

if __name__ == "__main__":
    main()