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

    # ── POC Hydrolysis (Solid -> Dissolved) ──
    # Typical rates are ~0.1 to 1.0 per day. 
    # 1e-4 1/s is accelerated slightly so you can observe the particle shrink in your simulation timeframe!
    kHyd: float = 1e-4        # Hydrolysis rate of POC to DOC (1/s)