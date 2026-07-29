# biopar.py
import dataclasses
from dataclasses import field

@dataclasses.dataclass
class BioPar:
    """
    Biogeochemical parameters based on nit_biopar_omz.m
    Includes dynamic stoichiometry from get_stoichiometry.m
    """
    
    # ── Organic Matter Composition (Anderson & Sarmiento 1994) ──
    stoch_a: float = 106.0  # C
    stoch_b: float = 175.0  # H
    stoch_c: float = 42.0   # O
    stoch_d: float = 16.0   # N
    
    # ── Derived Stoichiometric Coefficients ──
    # These are calculated automatically in __post_init__ below
    NCrem:  float = field(init=False)
    PCrem:  float = field(init=False)
    OCrem:  float = field(init=False)
    NCden1: float = field(init=False)
    NCden2: float = field(init=False)
    NCden3: float = field(init=False)

    def __post_init__(self):
        """Calculates stoichiometry of redox reactions based on C:H:O:N:P"""
        a, b, c, d = self.stoch_a, self.stoch_b, self.stoch_c, self.stoch_d
        
        # number of electrons required to oxidize organic matter
        Corg_e = 4*a + b - 2*c - 3*d + 5
        
        O2toH2O_e = 4
        HNO3toHNO2_e = 2
        HNO2toN2O_e = 2
        N2OtoN2_e = 2
        
        self.OCrem  = (Corg_e / O2toH2O_e) / a       # molO2 / molC
        self.NCrem  = d / a                          # molNH4 / molC
        self.PCrem  = 1.0 / a                        # molP / molC
        self.NCden1 = (Corg_e / HNO3toHNO2_e) / a    # Nitrate to Carbon ratio
        self.NCden2 = (Corg_e / HNO2toN2O_e) / a     # Nitrite to Carbon ratio
        self.NCden3 = (Corg_e / N2OtoN2_e) / a       # N2O to Carbon ratio

    # ── Ammonification ──
    krem:   float = 0.08          # Max remineralization rate (1/s)
    KO2Rem: float = 0.5           # Half sat. constant for respiration (mmolO2/m3)

    # ── Ammonium oxidation (Ammox: NH4 -> NO2) ──
    kAo:    float = 0.04556       # Max Ammonium oxidation rate (1/s)
    KNH4Ao: float = 0.0272        # Half sat. constant for NH4 (mmolN/m3)
    KO2Ao:  float = 0.333         # Half sat. constant for O2 (mmolO2/m3)

    # ── Nitrite oxidation (Nitrox: NO2 -> NO3) ──
    kNo:    float = 0.255         # Max Nitrite oxidation rate (1/s)
    KNO2No: float = 0.0272        # Half sat. constant for NO2 (mmolN/m3)
    KO2No:  float = 0.778         # Half sat. constant for O2 (mmolO2/m3)

    # ── Denitrification ──
    kDen1:    float = 0.08 / 2.0  # Max denitrif1 rate (1/s)
    KO2Den1:  float = 1.0         # O2 poisoning constant (mmolO2/m3)
    KNO3Den1: float = 0.5         # Half sat. constant for NO3 (mmolNO3/m3)

    kDen2:    float = 0.08 / 6.0  # Max denitrif2 rate (1/s)
    KO2Den2:  float = 0.3         # O2 poisoning constant (mmolO2/m3)
    KNO2Den2: float = 0.5         # Half sat. constant for NO2 (mmolNO2/m3)

    kDen3:    float = 0.08 / 3.0  # Max denitrif3 rate (1/s)
    KO2Den3:  float = 0.0292      # O2 poisoning constant (mmolO2/m3)
    KN2ODen3: float = 0.02        # Half sat. constant for N2O (mmolN2O/m3)

    # ── Anammox ──
    kAx:    float = 0.02          # Max Anaerobic Ammonium oxidation rate (1/s)
    KNH4Ax: float = 0.0274        # Half sat. constant for NH4 (mmolNH4/m3)
    KNO2Ax: float = 0.5           # Half sat. constant for NO2 (mmolNO2/m3)
    KO2Ax:  float = 0.886         # O2 inhibition constant (mmolO2/m3)

    # ── N2O prod via ammox (Ji et al 2018) ──
    n2o_yield: str = 'Ji'
    Ji_a: float = 0.2
    Ji_b: float = 0.08

    # ── POC Hydrolysis (Solid -> Dissolved, biomass-driven & saturating) ──
    # DOC_flux = k_hyd_max * total_heterotroph_biomass * POC / (K_POC + POC)
    # k_hyd_max must comfortably exceed the community's summed max carbon
    # demand (Σ Vmax_red / Y_red) so hydrolysis never starves the heterotrophs.
    k_hyd_max: float = 1 # e-3    # Max hydrolysis rate per unit heterotroph biomass (1/s per mmol C m⁻³)
    K_POC: float = 1e4         # Half-saturation constant for solid POC (mmol C m⁻³)


    # HIGHER K VALUES MEAN THE MICROBES ARE LESS EFFICIENT AT LOW FOOD CONCENTRATIONS.  LOWER K VALUES MEAN THEY ARE MORE EFFICIENT AT LOW FOOD CONCENTRATIONS.

    # ── MICROBIAL PARAMS  ──
    # aer (aerobic heterotroph)
    aer_vmax_oxi: float = 2.3148e-05
    aer_vmax_red: float = 2.4769e-05 
    aer_k_oxi: float = 0.2
    aer_k_red: float = 10 # 20.0? 10 is original
    aer_y_oxi: float = 4.00 # amount of oxidant (o2) needed to grow 1 unit of biomass ; number depends on free energy gain from each oxidant molecule
    aer_y_red: float = 4.28 # amount of DOC needed to grow 1 unit of biomass
    aer_e_nh4: float = 0.39

    # nar (no3 to no2 heterotroph)
    nar_vmax_oxi: float = 2.3148e-05 
    nar_vmax_red: float = 2.4769e-05  
    nar_k_oxi: float = 1.0
    nar_k_red: float = 20.0
    nar_y_oxi: float = 12.05 # amount of oxidant (no3) needed to grow 1 unit of biomass
    nar_y_red: float = 6.01 # amount of DOC needed to grow 1 unit of biomass # needs more DOC than aer so they'll be outcompeted by aer when O2 is present
    nar_e_no2: float = 12.05
    nar_e_nh4: float = 0.62

    # nai (no3 to n2o heterotroph)
    nai_vmax_oxi: float = 2.3148e-05  
    nai_vmax_red: float = 2.4769e-05  
    nai_k_oxi: float = 1.0
    nai_k_red: float = 20.0
    nai_y_oxi: float = 6.22 # this can be thought of as the "cost" in DOC it takes to grow 1 unit of biomass
    nai_y_red: float = 6.18
    nai_e_n2o: float = 3.11
    nai_e_nh4: float = 0.65

    # nao (no3 to n2 heterotroph)
    nao_vmax_oxi: float = 2.3148e-05  
    nao_vmax_red: float = 2.4769e-05  
    nao_k_oxi: float = 1.0
    nao_k_red: float = 20.0
    nao_y_oxi: float = 5.50
    nao_y_red: float = 6.72
    nao_e_n2: float = 2.75
    nao_e_nh4: float = 0.72

    # nir (no2 to n2o heterotroph)
    nir_vmax_oxi: float = 2.3148e-05  
    nir_vmax_red: float = 2.4769e-05  
    nir_k_oxi: float = 1.0
    nir_k_red: float = 20.0
    nir_y_oxi: float = 8.75
    nir_y_red: float = 4.60
    nir_e_n2o: float = 4.375
    nir_e_nh4: float = 0.43

    # nio (no2 to n2 heterotroph)
    nio_vmax_oxi: float = 2.3148e-05  
    nio_vmax_red: float = 2.4769e-05  
    nio_k_oxi: float = 1.0
    nio_k_red: float = 20.0
    nio_y_oxi: float = 6.07
    nio_y_red: float = 4.7531
    nio_e_n2: float = 3.035
    nio_e_nh4: float = 0.45

    # nos (n2o to n2 heterotroph)
    nos_vmax_oxi: float = 2.3148e-05  
    nos_vmax_red: float = 2.4769e-05  
    nos_k_oxi: float = 0.4
    nos_k_red: float = 20.0
    nos_y_oxi: float = 5.54
    nos_y_red: float = 3.22
    nos_e_n2: float = 5.54
    nos_e_nh4: float = 0.24

    # aoa (nh4 to no2 chemoautotroph)
    aoa_vmax_oxi: float = 1.2668e-04  
    aoa_vmax_red: float = 9.4482e-05  
    aoa_k_oxi: float = 0.333
    aoa_k_red: float = 0.134
    aoa_y_oxi: float = 8.16
    aoa_y_red: float = 10.95
    aoa_e_no2: float = 7.96

    # nob (no2 to no3 chemoautotroph)
    nob_vmax_oxi: float = 1.2043e-04  
    nob_vmax_red: float = 2.7557e-04  
    nob_k_oxi: float = 0.778
    nob_k_red: float = 0.254
    nob_y_oxi: float = 6.94
    nob_y_red: float = 15.87
    nob_e_no3: float = 15.87

    # aox (anammox)
    aox_vmax_oxi: float = 5.1244e-05  
    aox_vmax_red: float = 4.3287e-05  
    aox_k_oxi: float = 0.45
    aox_k_red: float = 0.45
    aox_y_oxi: float = 17.71
    aox_y_red: float = 14.96
    aox_e_no3: float = 2.57
    aox_e_n2: float = 14.95

    # ── loss parameters (all) ──
    m_l: float = 0.02 / 86400 # Linear loss (1/s) 
    m_q: float = 0.10 / 86400 # Quadratic loss (1 / uM C s)
    bmin: float = 1e-10        # Minimum biomass

    # ── zooplankton grazer ──
    zoo_umax: float = 1.00 / 86400  # Max grazing rate (1/s)
    zoo_m_l:  float = 0.00 / 86400  # Zoo linear loss (1/s)
    zoo_m_q:  float = 0.70 / 86400  # Zoo quadratic loss (1 / uM C s)
    zoo_k_b:  float = 1.0           # Half-sat for grazing (mmol C / m3)
    zoo_g_z:  float = 0.5           # Assimilation efficiency
    zoo_o2lim: float = 10.0         # O2 limit for grazing (mmol O2 / m3)

    # ── stoichiometry parameters ──
    CN_bio: float = 5.0         # biomass C:N (mol C / mol N)
    CN_det: float = 117.0/16.0  # detrital C:N (mol C / mol N)