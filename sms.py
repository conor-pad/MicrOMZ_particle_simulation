# sms.py
import torch
from n2o_yield import get_n2o_yield

# ── Internal Helper Functions (from mm1.m and fexp.m) ──
def mm1(var, Kvar):
    Pvar = torch.clamp(var, min=0.0)
    return Pvar / (Pvar + Kvar)

def fexp(o2, ko2):
    o2pos = torch.clamp(o2, min=0.0)
    return torch.exp(-o2pos / ko2)


# ── Core SMS Function (from nit_sms_omz.m) ──
def nit_sms_omz(var_dict, bgc):
    """
    Compute biogeochemical source-minus-sink (SMS) terms.
    
    Args:
        var_dict: dictionary containing PyTorch tensors for 
                  {'o2', 'no3', 'doc', 'po4', 'n2o', 'nh4', 'no2', 'n2'}
        bgc: BioPar instance containing parameters.
        
    Returns:
        ddt: dict of SMS tendency tensors for each tracer
        diags: dict of diagnostic process rate tensors
    """
    
    # ── Preliminary processing (clip negatives to 0) ──
    o2  = torch.clamp(var_dict['o2'], min=0.0)
    no3 = torch.clamp(var_dict['no3'], min=0.0)
    doc = torch.clamp(var_dict['doc'], min=0.0)
    po4 = torch.clamp(var_dict['po4'], min=0.0)
    n2o = torch.clamp(var_dict['n2o'], min=0.0)
    nh4 = torch.clamp(var_dict['nh4'], min=0.0)
    no2 = torch.clamp(var_dict['no2'], min=0.0)
    n2  = torch.clamp(var_dict['n2'], min=0.0)

    # ── J-OXIC ──
    # (1) Oxic Respiration rate (C-units):
    RemOx = bgc.krem * mm1(o2, bgc.KO2Rem) * doc

    # (2) Ammonium oxidation (molN-units):
    Ammox = bgc.kAo * mm1(o2, bgc.KO2Ao) * mm1(nh4, bgc.KNH4Ao)

    # (3) Nitrite oxidation (molN-units):
    Nitrox = bgc.kNo * mm1(o2, bgc.KO2No) * mm1(no2, bgc.KNO2No)

    # (4) N2O and NO2 production by ammox and nitrifier-denitrif (molN-units):
    Y = get_n2o_yield(o2, bgc)
    
    # via NH2OH
    Jnn2o_hx  = Ammox * Y['nn2o_hx_nh4']
    Jno2_hx   = Ammox * Y['no2_hx_nh4']
    
    # via NH4->NO2->N2O
    Jnn2o_nden = Ammox * Y['nn2o_nden_nh4']
    Jno2_nden  = Ammox * Y['no2_nden_nh4']

    # ── J-ANOXIC ──
    # (5) Denitrification (C-units)
    RemDen1 = bgc.kDen1 * mm1(no3, bgc.KNO3Den1) * fexp(o2, bgc.KO2Den1) * doc
    RemDen2 = bgc.kDen2 * mm1(no2, bgc.KNO2Den2) * fexp(o2, bgc.KO2Den2) * doc
    RemDen3 = bgc.kDen3 * mm1(n2o, bgc.KN2ODen3) * fexp(o2, bgc.KO2Den3) * doc

    # (6) Anaerobic ammonium oxidation (molN-units):
    Anammox = bgc.kAx * mm1(nh4, bgc.KNH4Ax) * mm1(no2, bgc.KNO2Ax) * fexp(o2, bgc.KO2Ax)

    # ── Calculate SMS for each tracer ──
    ddt = {}
    ddt['o2']  = -bgc.OCrem * RemOx - 1.5 * Ammox - 0.5 * Nitrox
    ddt['no3'] = Nitrox - bgc.NCden1 * RemDen1
    ddt['doc'] = -RemOx - RemDen1 - RemDen2 - RemDen3
    ddt['po4'] = bgc.PCrem * (RemOx + RemDen1 + RemDen2 + RemDen3)
    ddt['nh4'] = bgc.NCrem * (RemOx + RemDen1 + RemDen2 + RemDen3) - (Jnn2o_hx + Jno2_hx) - Anammox
    ddt['no2'] = Jno2_hx + Jno2_nden + bgc.NCden1 * RemDen1 - bgc.NCden2 * RemDen2 - Anammox - Nitrox
    ddt['n2']  = bgc.NCden3 * RemDen3 + Anammox

    # N2O individual SMSs
    sms_n2o_ammox = 0.5 * Jnn2o_hx
    sms_n2o_nden  = 0.5 * Jnn2o_nden
    sms_n2o_den2  = 0.5 * bgc.NCden2 * RemDen2
    sms_n2o_den3  = -bgc.NCden3 * RemDen3
    
    # N2O total SMS
    ddt['n2o'] = sms_n2o_ammox + sms_n2o_nden + sms_n2o_den2 + sms_n2o_den3

    # Add the newly tracked separate source terms
    ddt['n2o_ammox'] = sms_n2o_ammox + sms_n2o_nden
    ddt['n2o_denit'] = sms_n2o_den2 + sms_n2o_den3

    # ── Wraps diagnostics ──
    diags = {}
    diags['NRemOx']   = bgc.NCrem * RemOx
    diags['NRemAnox'] = bgc.NCrem * (RemDen1 + RemDen2 + RemDen3)
    diags['Ammox']    = Ammox
    diags['Nitrox']   = Nitrox
    diags['RemDen1']  = bgc.NCden1 * RemDen1
    diags['RemDen2']  = bgc.NCden2 * RemDen2
    diags['RemDen3']  = 2.0 * bgc.NCden3 * RemDen3
    diags['Jnn2o_Ax'] = Jnn2o_hx + Jnn2o_nden
    diags['Jno2_Ax']  = Jno2_hx + Jno2_nden
    diags['Anammox']  = 2.0 * Anammox

    # same things as above, but this time kept in CARBON UNITS
    diags['RemOx_C']   = RemOx
    diags['RemDen1_C'] = RemDen1
    diags['RemDen2_C'] = RemDen2
    diags['RemDen3_C'] = RemDen3


    return ddt, diags