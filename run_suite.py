# run_suite.py
import os
import sys
import torch
torch.set_default_dtype(torch.float32)
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import datetime

import config as cfg
import bcs
from biopar import BioPar
from sms import nit_sms_omz
from physics import setup_physics
from loop import run_simulation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

def run_experiment(radius, ext_o2, ext_no3):
    """Runs simulation and extracts true volume-integrated fluxes and C-yields."""
    
    # ── 1. Update Geometry & Speed ──
    cfg.radius = radius
    cfg.U_bg = 2.2 * (radius / 1.0)**0.56
    
    cfg.Lx = 20.0 * radius
    cfg.Ly = 10.0 * radius
    cfg.cx = 5.0 * radius
    cfg.cy = cfg.Ly / 2.0
    cfg.dx = cfg.Lx / (cfg.Nx - 1)
    cfg.dy = cfg.Ly / (cfg.Ny - 1)
    
    # ── 2. Time & Baseline Diffusion ──
    # Back to 5 flushes! Your dC/dt math makes this physically valid now.
    cfg.Total_Time = 5.0 * (cfg.Lx / cfg.U_bg) 
    cfg.K = cfg.nu / cfg.Sc_target
    
    # ── 3. NEW: Recalculate Dimensionless Physics Dynamically ──
    cfg.Re_actual = (cfg.U_bg * (2.0 * radius)) / cfg.nu
    cfg.Sh = 1 + 0.619 * (cfg.Re_actual ** 0.412) * (cfg.Sc_target ** (1/3))
    tqdm.write(f"▶ Simulating | R: {cfg.radius:.2f} mm | U: {cfg.U_bg:.2f} mm/s | Re: {cfg.Re_actual:.2f} | Time: {cfg.Total_Time:.1f}s")
    bcs.inflow.o2 = ext_o2
    bcs.inflow.no3 = ext_no3

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    device, state = setup_physics(cfg)
    results = run_simulation(state, cfg, device)
    sys.stdout.close()
    sys.stdout = original_stdout

    # ── 1. Reconstruct Final Biological State ──
    # Extract the final snapshot arrays and push them back to PyTorch
    final_tensors = {
        'o2':  torch.tensor(results[0][-1], device=device),
        'n2o': torch.tensor(results[1][-1], device=device),
        'no3': torch.tensor(results[2][-1], device=device),
        'no2': torch.tensor(results[3][-1], device=device),
        'n2':  torch.tensor(results[4][-1], device=device),
        'doc': torch.tensor(results[5][-1], device=device),
        'nh4': torch.tensor(results[6][-1], device=device),
        'po4': torch.zeros_like(torch.tensor(results[0][-1], device=device))
    }
    
    # NEW: Calculate true dC/dt for N2O using the last two saved frames
    frames_saved = len(results[1])
    time_between_frames = cfg.Total_Time / (frames_saved - 1) if frames_saved > 1 else cfg.Total_Time
    prev_n2o = torch.tensor(results[1][-2], device=device) if frames_saved > 1 else torch.zeros_like(final_tensors['n2o'])
    n2o_accumulation_rate = (final_tensors['n2o'] - prev_n2o) / time_between_frames

    bgc = BioPar()
    ddt, diags = nit_sms_omz(final_tensors, bgc)

    # ── 2. Gauss's Theorem: Volume Integrals for Flux ──
    # Volume of a single grid cell assuming 1mm slice depth (converted to m^3)
    dV_m3 = (cfg.dx * cfg.dy * 1.0) * 1e-9  
    
    # Total biological production rates (mmol / s)
    total_n2o_flux = torch.sum(ddt['n2o']).item() * dV_m3
    total_n2_flux = torch.sum(ddt['n2']).item() * dV_m3

    # Total DOC biologically consumed by all respiration pathways (mmol C / s)
    total_c_consumed = torch.sum(diags['RemOx_C'] + diags['RemDen1_C'] + 
                                 diags['RemDen2_C'] + diags['RemDen3_C']).item() * dV_m3

   # ── 3. Internal vs. Plume Spatial Masking ──
    particle_mask = state['particle_mask']
    plume_mask = 1.0 - particle_mask  # Everything outside the particle

    # Dead Core Tracker
    oxic_threshold = 1.0 # mmol/m3 (Below this, denitrification starts)
    anoxic_core_vol = torch.sum((final_tensors['o2'] < oxic_threshold) * particle_mask).item() * dV_m3
    total_core_vol = torch.sum(particle_mask).item() * dV_m3
    frac_anoxic_core = anoxic_core_vol / total_core_vol if total_core_vol > 0 else 0.0
    
    # Internal Production (Total + Separated by Pathway)
    n2o_production_internal = torch.sum(ddt['n2o'] * particle_mask).item() * dV_m3
    internal_n2o_ammox = torch.sum(ddt['n2o_ammox'] * particle_mask).item() * dV_m3
    internal_n2o_denit = torch.sum(ddt['n2o_denit'] * particle_mask).item() * dV_m3
    
    # NEW: Transient Gauss's Theorem for True Physical Leakage
    n2o_accumulation = torch.sum(n2o_accumulation_rate * particle_mask).item() * dV_m3
    true_n2o_leakage_out = n2o_production_internal - n2o_accumulation

    # Plume Production (Separated by Pathway)
    n2o_production_plume = torch.sum(ddt['n2o'] * plume_mask).item() * dV_m3
    plume_n2o_ammox = torch.sum(ddt['n2o_ammox'] * plume_mask).item() * dV_m3
    plume_n2o_denit = torch.sum(ddt['n2o_denit'] * plume_mask).item() * dV_m3

    # ── 4. Normalization (Yields and Fractions) ──
    # N2O Yield: How much N2O is leaked per unit of Carbon METABOLIZED
    n2o_yield_efficiency = total_n2o_flux / total_c_consumed if total_c_consumed > 0 else 0.0

    # Separate Total C consumed into Core vs Plume
    core_c_consumed = torch.sum((diags['RemOx_C'] + diags['RemDen1_C'] + 
                                 diags['RemDen2_C'] + diags['RemDen3_C']) * particle_mask).item() * dV_m3
    plume_c_consumed = torch.sum((diags['RemOx_C'] + diags['RemDen1_C'] + 
                                  diags['RemDen2_C'] + diags['RemDen3_C']) * plume_mask).item() * dV_m3

    # Core Metabolic Fractions
    frac_oxic_core = (torch.sum(diags['RemOx_C'] * particle_mask).item() * dV_m3) / core_c_consumed if core_c_consumed > 0 else 0
    frac_den1_core = (torch.sum(diags['RemDen1_C'] * particle_mask).item() * dV_m3) / core_c_consumed if core_c_consumed > 0 else 0
    frac_den2_core = (torch.sum(diags['RemDen2_C'] * particle_mask).item() * dV_m3) / core_c_consumed if core_c_consumed > 0 else 0
    frac_den3_core = (torch.sum(diags['RemDen3_C'] * particle_mask).item() * dV_m3) / core_c_consumed if core_c_consumed > 0 else 0

    # Plume Metabolic Fractions
    frac_oxic_plume = (torch.sum(diags['RemOx_C'] * plume_mask).item() * dV_m3) / plume_c_consumed if plume_c_consumed > 0 else 0
    frac_den1_plume = (torch.sum(diags['RemDen1_C'] * plume_mask).item() * dV_m3) / plume_c_consumed if plume_c_consumed > 0 else 0
    frac_den2_plume = (torch.sum(diags['RemDen2_C'] * plume_mask).item() * dV_m3) / plume_c_consumed if plume_c_consumed > 0 else 0
    frac_den3_plume = (torch.sum(diags['RemDen3_C'] * plume_mask).item() * dV_m3) / plume_c_consumed if plume_c_consumed > 0 else 0

    # ── 5. Internal Residence Times (Stock / Turnover Flux) ──
    residence_times = {}
    
    for tracer in ['o2', 'no3', 'n2o', 'n2', 'nh4']:
        stock = torch.sum(final_tensors[tracer] * particle_mask).item() * dV_m3
        bio_flux = torch.sum(torch.abs(ddt[tracer]) * particle_mask).item() * dV_m3
        tau = stock / bio_flux if bio_flux > 1e-12 else 0.0
        residence_times[f'Tau_{tracer}'] = tau


    # ── 6. O2 and NO3 Inward Fluxes ──
    o2_consumption = torch.where(ddt['o2'] < 0, ddt['o2'], torch.tensor(0.0, device=device))
    no3_consumption = torch.where(ddt['no3'] < 0, ddt['no3'], torch.tensor(0.0, device=device))
    
    o2_flux_in = torch.sum(torch.abs(o2_consumption) * particle_mask).item() * dV_m3
    no3_flux_in = torch.sum(torch.abs(no3_consumption) * particle_mask).item() * dV_m3

    # ── 7. DOC Outward Flux (Leakage) ──
    doc_produced_internal = torch.sum(cfg.doc_flux_rate * particle_mask).item() * dV_m3
    doc_consumed_internal = torch.sum(torch.abs(ddt['doc']) * particle_mask).item() * dV_m3
    doc_flux_out = max(0.0, doc_produced_internal - doc_consumed_internal)

    # ── 8. MERGE EVERYTHING INTO A CONSISTENT RETURN DICT ──
    metrics = {
        'Radius_mm': radius,
        'Speed_mms': cfg.U_bg,
        'Ext_O2': ext_o2,
        'Ext_NO3': ext_no3,
        'N2O_Flux_Total_mmol_s': total_n2o_flux,
        'N2_Flux_Total_mmol_s': total_n2_flux,
        'Frac_Anoxic_Core': frac_anoxic_core,
        'N2O_Flux_Internal': n2o_production_internal,
        'N2O_Flux_Plume': n2o_production_plume,
        'N2O_Leakage_Out_mmol_s': true_n2o_leakage_out, 
        'N2O_Yield_per_C': n2o_yield_efficiency,
        'Frac_Oxic_Core': frac_oxic_core,
        'Frac_Den1_Core': frac_den1_core,
        'Frac_Den2_Core': frac_den2_core,
        'Frac_Den3_Core': frac_den3_core,
        'Frac_Oxic_Plume': frac_oxic_plume,
        'Frac_Den1_Plume': frac_den1_plume,
        'Frac_Den2_Plume': frac_den2_plume,
        'Frac_Den3_Plume': frac_den3_plume,
        'Plume_N2O_Ammox': plume_n2o_ammox,
        'Plume_N2O_Denit': plume_n2o_denit,
        'Internal_N2O_Ammox': internal_n2o_ammox,  
        'Internal_N2O_Denit': internal_n2o_denit, 
        'O2_Flux_In_mmol_s': o2_flux_in,
        'NO3_Flux_In_mmol_s': no3_flux_in,
        'DOC_Flux_Out_mmol_s': doc_flux_out
    }
    
    metrics.update(residence_times)
    # torch.save(final_tensors, f"outputs/Tensors_R{radius}_O2{ext_o2}_NO3{ext_no3}.pt")
    return metrics

def generate_all_plots(csv_filename):
    df = pd.read_csv(csv_filename)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})

    # ── DYNAMIC BASELINE SELECTION ──
    available_no3 = df['Ext_NO3'].unique()
    available_o2 = df['Ext_O2'].unique()
    
    # If there's only 1 value, it forces it. If multiple, it prefers 10.0/6.0, or falls back to index 0.
    base_no3 = available_no3[0] if len(available_no3) == 1 else (10.0 if 10.0 in available_no3 else available_no3[0])
    base_o2 = available_o2[0] if len(available_o2) == 1 else (6.0 if 6.0 in available_o2 else available_o2[0])
    
    print(f"\n📊 Plotting Baselines Auto-Selected -> O2: {base_o2}, NO3: {base_no3}")

    # ── PLOT 1: Normalized N2O Yield (Whole Domain) ──
    g1 = sns.FacetGrid(df, col="Radius_mm", hue="Ext_NO3", palette="magma", height=4.5, aspect=1.2, sharey=False)
    g1.map(sns.lineplot, "Ext_O2", "N2O_Yield_per_C", marker="o", linewidth=2.5, markersize=8)
    g1.set_axis_labels("Ambient O2 (mmol/m3)", "Total Domain N2O Yield\n(mmol N2O / mmol Total C metabolized)")
    g1.set_titles(col_template="Particle Radius: {col_name} mm")
    g1.add_legend(title="Ambient NO3 (mmol/m3)", loc='upper center', bbox_to_anchor=(0.45, -0.15), ncol=3, frameon=True)
    plt.subplots_adjust(top=0.85, bottom=0.25)
    g1.fig.suptitle("N2O Production Efficiency vs. Ambient O2 & NO3 (Core + Plume)", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(f"outputs/Plot1_N2O_Yield.png", dpi=300, bbox_inches='tight')

    # ── PLOTS 2 & 3: Metabolic Architecture (Core vs Plume) ──
    # Filtering dynamically based on the available data!
    df_base = df[(df['Ext_O2'] == base_o2) & (df['Ext_NO3'] == base_no3)].copy()

    def get_melted_fractions(df_in, zone_suffix, x_var):
        cols = [f'Frac_Oxic_{zone_suffix}', f'Frac_Den1_{zone_suffix}', f'Frac_Den2_{zone_suffix}', f'Frac_Den3_{zone_suffix}']
        melted = df_in.melt(id_vars=[x_var], value_vars=cols, var_name='Pathway', value_name='Fraction')
        melted['Pathway'] = melted['Pathway'].str.replace('Frac_', '').str.replace(f'_{zone_suffix}', '')
        return melted

    # --- PLOT 2: vs Radius ---
    df_core_rad = get_melted_fractions(df_base, 'Core', 'Radius_mm')
    df_plume_rad = get_melted_fractions(df_base, 'Plume', 'Radius_mm')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.lineplot(data=df_core_rad, x='Radius_mm', y='Fraction', hue='Pathway', marker="s", linewidth=2.5, ax=axes[0])
    axes[0].set_title("Inside Particle Only (Core Metabolism)", fontweight='bold')
    axes[0].set_xlabel("Particle Radius (mm)")
    axes[0].set_ylabel("Fraction of DOC Consumed in Zone")

    sns.lineplot(data=df_plume_rad, x='Radius_mm', y='Fraction', hue='Pathway', marker="s", linewidth=2.5, ax=axes[1])
    axes[1].set_title("Outside Particle Only (Plume/Wake Metabolism)", fontweight='bold')
    axes[1].set_xlabel("Particle Radius (mm)")

    # Dynamic Title
    plt.suptitle(f"Spatial Metabolic Partitioning vs. Particle Size (O2={base_o2}, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot2_Metabolism_vs_Radius.png", dpi=300, bbox_inches='tight')

    # --- PLOT 3: vs Velocity ---
    df_core_vel = get_melted_fractions(df_base, 'Core', 'Speed_mms')
    df_plume_vel = get_melted_fractions(df_base, 'Plume', 'Speed_mms')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.lineplot(data=df_core_vel, x='Speed_mms', y='Fraction', hue='Pathway', marker="^", linewidth=2.5, ax=axes[0])
    axes[0].set_title("Inside Particle Only (Core Metabolism)", fontweight='bold')
    axes[0].set_xlabel("Sinking Velocity (mm/s)")
    axes[0].set_ylabel("Fraction of DOC Consumed in Zone")

    sns.lineplot(data=df_plume_vel, x='Speed_mms', y='Fraction', hue='Pathway', marker="^", linewidth=2.5, ax=axes[1])
    axes[1].set_title("Outside Particle Only (Plume/Wake Metabolism)", fontweight='bold')
    axes[1].set_xlabel("Sinking Velocity (mm/s)")

    # Dynamic Title
    plt.suptitle(f"Spatial Metabolic Partitioning vs. Sinking Velocity (O2={base_o2}, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot3_Metabolism_vs_Velocity.png", dpi=300, bbox_inches='tight')

    # ── PLOT 4: Residence Times (Inside Core Only) ──
    tau_cols = ['Tau_o2', 'Tau_no3', 'Tau_n2o', 'Tau_n2', 'Tau_nh4']
    df_tau = df.melt(id_vars=['Radius_mm', 'Ext_O2', 'Ext_NO3'],
                     value_vars=tau_cols,
                     var_name='Species', value_name='Residence_Time_s')

    df_tau['Species'] = df_tau['Species'].str.replace('Tau_', '').str.upper()

    g4 = sns.FacetGrid(df_tau, col="Radius_mm", row="Species", hue="Ext_NO3",
                       palette="viridis", height=3.0, aspect=1.5, sharey="row")
    
    g4.map(sns.lineplot, "Ext_O2", "Residence_Time_s", marker="o", linewidth=2.5)
    g4.add_legend(title="Ambient NO3 (mmol/m3)")
    g4.set_axis_labels("Ambient O2 (mmol/m3)", "Core Residence Time (s)")
    g4.set_titles(col_template="Radius: {col_name} mm", row_template="{row_name} Turnover")
    
    plt.subplots_adjust(top=0.92)
    g4.fig.suptitle("Internal Microenvironmental Residence Times vs. Ambient Forcing (Inside Particle Only)", fontsize=16, fontweight='bold')
    plt.savefig(f"outputs/Plot4_Residence_Times.png", dpi=300, bbox_inches='tight')

    # ── PLOT 5: N2O Source Apportionment (Inside vs Outside) ──
    df_plume = df_base.melt(id_vars=['Radius_mm', 'Speed_mms'], 
                            value_vars=['Plume_N2O_Ammox', 'Plume_N2O_Denit'],
                            var_name='Pathway', value_name='N2O Production (mmol/s)')
    df_plume['Pathway'] = df_plume['Pathway'].str.replace('Plume_N2O_', '')

    df_internal = df_base.melt(id_vars=['Radius_mm', 'Speed_mms'], 
                               value_vars=['Internal_N2O_Ammox', 'Internal_N2O_Denit'],
                               var_name='Pathway', value_name='N2O Production (mmol/s)')
    df_internal['Pathway'] = df_internal['Pathway'].str.replace('Internal_N2O_', '')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.lineplot(data=df_plume, x='Radius_mm', y='N2O Production (mmol/s)', 
                 hue='Pathway', marker="X", linewidth=2.5, palette=['teal', 'darkred'], ax=axes[0])
    axes[0].set_title("Outside Particle Only (Plume/Wake N2O Sources)", fontweight='bold')
    axes[0].set_xlabel("Particle Radius (mm)")
    axes[0].set_ylabel("N2O Production Rate (mmol/s)")
    axes[0].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df_internal, x='Radius_mm', y='N2O Production (mmol/s)', 
                 hue='Pathway', marker="o", linewidth=2.5, palette=['teal', 'darkred'], ax=axes[1])
    axes[1].set_title("Inside Particle Only (Core N2O Sources)", fontweight='bold')
    axes[1].set_xlabel("Particle Radius (mm)")
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Dynamic Title
    plt.suptitle(f"N2O Source Apportionment vs. Particle Size (O2={base_o2}, NO3={base_no3})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/Plot5_N2O_Sources_Dual.png", dpi=300, bbox_inches='tight')

    # ── PLOT 6: Boundary Layer Exchange Fluxes ──
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    sns.lineplot(data=df, x='Ext_O2', y='O2_Flux_In_mmol_s', hue='Radius_mm', 
                 palette='viridis', marker='o', linewidth=2.5, ax=axes[0])
    axes[0].set_title('O2 Flux Across Boundary (Into Particle)', fontweight='bold')
    axes[0].set_xlabel('Ambient O2 (mmol/m3)')
    axes[0].set_ylabel('O2 Flux (mmol/s)')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df, x='Ext_NO3', y='NO3_Flux_In_mmol_s', hue='Radius_mm', 
                 palette='flare', marker='s', linewidth=2.5, ax=axes[1])
    axes[1].set_title('NO3 Flux Across Boundary (Into Particle)', fontweight='bold')
    axes[1].set_xlabel('Ambient NO3 (mmol/m3)')
    axes[1].set_ylabel('NO3 Flux (mmol/s)')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df, x='Ext_O2', y='DOC_Flux_Out_mmol_s', hue='Radius_mm', 
                 palette='copper', marker='^', linewidth=2.5, ax=axes[2])
    axes[2].set_title('DOC Flux Across Boundary (Out of Particle)', fontweight='bold')
    axes[2].set_xlabel('Ambient O2 (mmol/m3)')
    axes[2].set_ylabel('DOC Leakage (mmol/s)')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=df, x='Ext_O2', y='N2O_Leakage_Out_mmol_s', hue='Radius_mm', 
                 palette='magma', marker='X', linewidth=2.5, ax=axes[3])
    axes[3].set_title('N2O Flux Across Boundary (Out of Particle)', fontweight='bold')
    axes[3].set_xlabel('Ambient O2 (mmol/m3)')
    axes[3].set_ylabel('N2O Leakage (mmol/s)')
    axes[3].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("outputs/Plot6_Boundary_Fluxes.png", dpi=300, bbox_inches='tight')

    # ── PLOT 7: Absolute N2O Flux (Inside Core Only) ──
    radii = sorted(df['Radius_mm'].unique())
    fig, axes = plt.subplots(1, len(radii), figsize=(5 * len(radii), 5), sharey=True)

    for i, r in enumerate(radii):
        ax = axes[i]
        subset = df[df['Radius_mm'] == r]

        sns.lineplot(data=subset, x='Ext_O2', y='N2O_Flux_Internal', hue='Ext_NO3', 
                     palette=['#5DADE2', '#2874A6', '#154360'], 
                     ax=ax, linewidth=3, marker='X', linestyle='--', errorbar=None)
        
        ax.axhline(0, color='black', linewidth=1.5, zorder=0)
        ax.set_title(f"Particle Radius: {r} mm", fontweight='bold')
        ax.set_xlabel('Ambient O2 (mmol/m3)')
        
        if i == 0:
            ax.set_ylabel('Internal Core N2O Net Rate (mmol/s)\n<-- Net consumption inside   |   Net production inside -->')
        else:
            ax.set_ylabel('')
            
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Ambient NO3 (mmol/m3)', loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, frameon=True)

    plt.subplots_adjust(top=0.85, bottom=0.25)
    fig.suptitle("Absolute N2O Net Flux (Inside Particle Core Only)", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig("outputs/Plot7_Internal_Flux_Only.png", dpi=300, bbox_inches='tight')

    # ── PLOT 8: Anoxic "Dead Core" Fraction (Inside Core Only) ──
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='Radius_mm', y='Frac_Anoxic_Core', hue='Ext_O2', 
                 marker="s", linewidth=2.5, palette="crest")
    
    plt.title("Anoxic Core Fraction (Inside Particle Only)", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Fraction of Core Volume (< 1.0 mmol/m3 O2)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Ambient O2 (mmol/m3)")
    plt.savefig("outputs/Plot8_Anoxic_Core.png", dpi=300, bbox_inches='tight')

    # ── PLOT 9: Terminal Ratio (Whole Domain) ──
    df['Terminal_Ratio'] = df['N2_Flux_Total_mmol_s'] / (df['N2O_Flux_Total_mmol_s'] + 1e-12)

    plt.figure(figsize=(8, 6))
    # Filtering dynamically based on available NO3
    df_ratio = df[df['Ext_NO3'] == base_no3]

    sns.lineplot(data=df_ratio, x='Radius_mm', y='Terminal_Ratio', hue='Ext_O2', 
                 marker="D", linewidth=2.5, palette="rocket")
    
    plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, zorder=0) 
    plt.yscale('log') 
    
    # Dynamic Title
    plt.title(f"Total Domain Terminal N2/N2O Ratio (Core + Plume, NO3={base_no3})", fontweight='bold')
    plt.xlabel("Particle Radius (mm)")
    plt.ylabel("Total System N2 / N2O Ratio (Log Scale)\n<-- System is N2O Source  |  System is N2 Sink -->")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Ambient O2 (mmol/m3)")
    plt.savefig("outputs/Plot9_Terminal_Ratio.png", dpi=300, bbox_inches='tight')

    plt.show()


def main():
    print("🌊 Starting NitrOMZ Parameter Suite...\n")
    cfg.is_suite = True 

    radii = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    o2_levels = [2.0, 6.0, 10.0]
    no3_levels = [10.0]
    
    # ── 1. Create a clean outputs directory ──
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # ── 2. Auto-generate a unique timestamped filename ──
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    # csv_filename = os.path.join(out_dir, f"NitrOMZ_Suite_{timestamp}.csv")
    csv_filename = "outputs/NitrOMZ_Suite_2026-05-04_1518.csv"

    if os.path.exists(csv_filename):
        results_df = pd.read_csv(csv_filename)
        results_df = results_df.round({'Radius_mm': 2, 'Ext_O2': 2, 'Ext_NO3': 2})
        results_df = results_df.drop_duplicates(subset=['Radius_mm', 'Ext_O2', 'Ext_NO3'], keep='last')
        results_df.to_csv(csv_filename, index=False)
        completed_runs = set(zip(results_df['Radius_mm'], results_df['Ext_O2'], results_df['Ext_NO3']))
        results_data = results_df.to_dict('records')
        
        # ── NEW: Resume Confirmation Print ──
        print(f"🔄 RESUMING RUN: Found existing data in '{csv_filename}'.")
        print(f"⏭️ Skipping {len(completed_runs)} previously completed runs.\n")

    else:
        completed_runs = set()
        results_data = []
        
        # ── NEW: Fresh Run Confirmation Print ──
        print(f"🆕 Starting a completely fresh suite. Saving to '{csv_filename}'...\n")

    total_runs = len(radii) * len(o2_levels) * len(no3_levels)
    
    with tqdm(total=total_runs, desc="Simulating Parameters", unit="run", bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        pbar.update(len(completed_runs))
        for r in radii:
            for o2 in o2_levels:
                for no3 in no3_levels:
                    run_id = (round(r, 2), round(o2, 2), round(no3, 2))
                    if run_id in completed_runs:
                        continue
                    
                    metrics = run_experiment(radius=r, ext_o2=o2, ext_no3=no3)
                    results_data.append(metrics)
                    
                    # Save intermediate progress
                    temp_df = pd.DataFrame(results_data)
                    temp_df.to_csv(csv_filename, index=False)
                    pbar.update(1)

    # ── 3. Final Data Organization ──
    final_df = pd.DataFrame(results_data)
    final_df = final_df.sort_values(by=['Radius_mm', 'Ext_O2', 'Ext_NO3'])
    final_df.to_csv(csv_filename, index=False)
    
    print(f"\n✅ Suite Complete! Neatly organized results safely saved to {csv_filename}")

    generate_all_plots(csv_filename)

if __name__ == "__main__":
    main()