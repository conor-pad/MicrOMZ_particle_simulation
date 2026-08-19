# mortality_sweep.py
import importlib
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil
import os
import re
import subprocess

import config
from physics import setup_physics
import loop  
import biopar

def patch_biopar_mortality(loss_val):
    with open('biopar.py', 'r') as f:
        content = f.read()
    content = re.sub(r'(loss_multiplier:\s*float\s*=\s*)[\d\.]+', rf'\g<1>{loss_val}', content)
    with open('biopar.py', 'w') as f:
        f.write(content)

def get_field(res, field_name):
    for item in res:
        if isinstance(item, dict) and field_name in item:
            return item[field_name]
    return None

def line_plot(x_data, y_data, title, ylabel, filename, color='indigo'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_data, y_data, marker='o', linestyle='-', linewidth=2, color=color)
    ax.set_xlabel("Mortality (loss_multiplier)", fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def main():
    shutil.copyfile('biopar.py', 'biopar_backup.py')

    mortalities = np.linspace(0.001, 1, 20)
    TOTAL_TIME_14_DAYS = 14 * 24 * 3600  
    ANOXIC_THRESHOLD = 1.0 

    # Create directory for plots and CSV
    sweep_dir = "mortality_sweep_data"
    os.makedirs(sweep_dir, exist_ok=True)
    csv_filename = os.path.join(sweep_dir, "mortality_sweep_results.csv")

    # Initialize CSV with headers
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Mortality", "Min_O2", "Anoxic_Fraction", "Peak_Specific_Growth", 
            "Avg_Core_Growth", "Bulk_Growth", "Peak_Biomass", "Min_O2_Monod", 
            "Min_DOC_Monod", "O2_Lim_Fraction", "Terminal_O2", "Terminal_Mu", "Terminal_Biomass"
        ])

    results_min_o2 = []
    results_anoxic_frac = []
    results_realized_growth = []
    results_avg_growth = []
    results_bulk_growth = []
    results_peak_biomass = []
    results_min_o2_monod = []
    results_min_doc_monod = []
    results_o2_lim_frac = []
    results_terminal_min_o2 = []
    results_mu_eq = []
    results_b_eq_sim = []

    try:
        print("🚀 Starting 14-Day Mortality Sweep...")
        
        for mort in mortalities:
            print(f"\n--- Testing loss_multiplier: {mort:.3f} ---")
            
            patch_biopar_mortality(mort)
            importlib.reload(biopar)
            importlib.reload(loop)
            importlib.reload(config)
            
            config.Total_Time = TOTAL_TIME_14_DAYS
            
            with torch.no_grad():
                device, state = setup_physics(config)
                res = loop.run_simulation(state, config, device)
                
                o2 = get_field(res, 'o2')
                doc = get_field(res, 'doc')
                aer = get_field(res, 'aer')

                if o2 is not None and aer is not None:
                    # Calculate metrics
                    min_o2_val = np.min(o2).item() if torch.is_tensor(o2) else np.min(o2)
                    anox_frac = (np.sum(o2 < ANOXIC_THRESHOLD) / o2.size) * 100
                    peak_bio = np.max(aer)
                    term_o2 = np.min(o2[-1])
                    
                    Ks_O2 = getattr(biopar, 'Ks_O2', 1.0)
                    Ks_DOC = getattr(biopar, 'Ks_DOC', 10.0)
                    o2_monod = o2 / (o2 + Ks_O2)
                    doc_monod = doc / (doc + Ks_DOC) if doc is not None else np.ones_like(o2)
                    
                    min_o2_mon = np.min(o2_monod)
                    min_doc_mon = np.min(doc_monod)
                    o2_lim = np.sum(o2_monod < doc_monod) / o2.size * 100
                    
                    mu_max = getattr(biopar, 'mu_max_aer', 2.0)
                    growth_rate = mu_max * o2_monod * doc_monod
                    
                    peak_grow = np.max(growth_rate)
                    avg_grow = np.mean(growth_rate)
                    bulk_grow = np.max(growth_rate * aer)
                    term_mu = np.mean(growth_rate[-1])
                    term_bio = np.mean(aer[-1])

                    # Append to lists for plotting
                    results_min_o2.append(min_o2_val)
                    results_anoxic_frac.append(anox_frac)
                    results_peak_biomass.append(peak_bio)
                    results_terminal_min_o2.append(term_o2)
                    results_min_o2_monod.append(min_o2_mon)
                    results_min_doc_monod.append(min_doc_mon)
                    results_o2_lim_frac.append(o2_lim)
                    results_realized_growth.append(peak_grow)
                    results_avg_growth.append(avg_grow)
                    results_bulk_growth.append(bulk_grow)
                    results_mu_eq.append(term_mu)
                    results_b_eq_sim.append(term_bio)

                    # Write to CSV immediately
                    with open(csv_filename, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            mort, min_o2_val, anox_frac, peak_grow, avg_grow, bulk_grow, 
                            peak_bio, min_o2_mon, min_doc_mon, o2_lim, term_o2, term_mu, term_bio
                        ])
                else:
                    print("⚠️ Warning: Could not extract O2 or Aerobe data.")

                del state, res, o2, doc, aer
                if torch.backends.mps.is_available(): torch.mps.empty_cache()
                elif torch.cuda.is_available(): torch.cuda.empty_cache()

        print("\n📊 Generating Plots...")
        line_plot(mortalities, results_min_o2, 'Lowest $O_2$ recorded (mmol/m³)', 'Lowest $O_2$ Reached (path minimum)', f'{sweep_dir}/sweep_min_o2.png', color='purple')
        line_plot(mortalities, results_anoxic_frac, 'Max Core Anoxic Fraction (%)', 'Maximum Core Anoxic Fraction Reached', f'{sweep_dir}/sweep_anoxic_frac.png', color='orangered')
        line_plot(mortalities, results_realized_growth, 'Peak Specific Growth Rate (day$^{-1}$)', 'Peak Local Specific Growth Rate', f'{sweep_dir}/sweep_realized_growth.png', color='steelblue')
        line_plot(mortalities, results_avg_growth, 'Peak Average Core Growth Rate (day$^{-1}$)', 'Peak Average Core Specific Growth Rate', f'{sweep_dir}/sweep_avg_growth.png', color='darkorange')
        line_plot(mortalities, results_bulk_growth, 'Peak Bulk Growth Rate (mmol C m$^{-3}$ day$^{-1}$)', 'Peak Bulk Growth Rate', f'{sweep_dir}/sweep_bulk_growth.png', color='darkorange')
        line_plot(mortalities, results_peak_biomass, 'Peak Total Community Biomass (mmol C/m³)', 'Peak Local Total Biomass Reached', f'{sweep_dir}/sweep_peak_biomass.png', color='gold')
        line_plot(mortalities, results_min_o2_monod, 'Lowest $O_2$ Monod Factor', 'Maximum $O_2$ Limitation Reached', f'{sweep_dir}/sweep_min_o2_monod.png', color='forestgreen')
        line_plot(mortalities, results_min_doc_monod, 'Lowest DOC Monod Factor', 'Maximum DOC Limitation Reached', f'{sweep_dir}/sweep_min_doc_monod.png', color='forestgreen')
        line_plot(mortalities, results_o2_lim_frac, 'Max Core % Limited by $O_2$', '$O_2$ vs. DOC Limitation Dominance', f'{sweep_dir}/sweep_limiting_factor.png', color='crimson')
        line_plot(mortalities, results_terminal_min_o2, 'Terminal (Steady-State) Min $O_2$ (mmol/m³)', 'Terminal $O_2$ at Stopping Point', f'{sweep_dir}/sweep_terminal_o2.png', color='purple')
        line_plot(mortalities, results_mu_eq, 'Terminal Specific Growth Rate (day$^{-1}$)', 'Terminal (Steady-State) Specific Growth Rate', f'{sweep_dir}/sweep_mu_eq.png', color='darkorange')
        line_plot(mortalities, results_b_eq_sim, 'Terminal Total Biomass, $B_{eq}$ (mmol C/m³)', 'Terminal Total Biomass at Lock', f'{sweep_dir}/sweep_b_eq.png', color='gold')

        anoxic_runs = [m for m, min_o2 in zip(mortalities, results_min_o2) if min_o2 <= ANOXIC_THRESHOLD]
        target_mortality = max(anoxic_runs) if anoxic_runs else mortalities[np.argmin(results_min_o2)]
        
        print(f"\n🎯 TIPPING POINT (or lowest O2): {target_mortality:.3f}")
        patch_biopar_mortality(target_mortality)
        
        print("🚀 Executing main.py for final folder generation...")
        subprocess.run(["python", "main.py"])

    finally:
        if os.path.exists('biopar_backup.py'):
            shutil.copyfile('biopar_backup.py', 'biopar.py')
            os.remove('biopar_backup.py')
            print("✅ Restored original biopar.py.")

if __name__ == "__main__":
    main()