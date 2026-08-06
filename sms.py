# sms.py
import torch

def microbial_sms_omz(var_dict, bgc):
    """
    biological sources and sinks for MicrOMZ 
    """
    # prevent negative concentrations
    c = {k: torch.clamp(v, min=0.0) for k, v in var_dict.items()}

    def monod(S, K): return S / (S + K)
    def mort(B): return B * (bgc.m_l + B * bgc.m_q)

    # Growth Rates
    aer_bio = torch.minimum((bgc.aer_vmax_red * monod(c['doc'], bgc.aer_k_red)) / bgc.aer_y_red, (bgc.aer_vmax_oxi * monod(c['o2'], bgc.aer_k_oxi)) / bgc.aer_y_oxi) * c['aer']
    nar_bio = torch.minimum((bgc.nar_vmax_red * monod(c['doc'], bgc.nar_k_red)) / bgc.nar_y_red, (bgc.nar_vmax_oxi * monod(c['no3'], bgc.nar_k_oxi)) / bgc.nar_y_oxi) * c['nar']
    nai_bio = torch.minimum((bgc.nai_vmax_red * monod(c['doc'], bgc.nai_k_red)) / bgc.nai_y_red, (bgc.nai_vmax_oxi * monod(c['no3'], bgc.nai_k_oxi)) / bgc.nai_y_oxi) * c['nai']
    nao_bio = torch.minimum((bgc.nao_vmax_red * monod(c['doc'], bgc.nao_k_red)) / bgc.nao_y_red, (bgc.nao_vmax_oxi * monod(c['no3'], bgc.nao_k_oxi)) / bgc.nao_y_oxi) * c['nao']
    nir_bio = torch.minimum((bgc.nir_vmax_red * monod(c['doc'], bgc.nir_k_red)) / bgc.nir_y_red, (bgc.nir_vmax_oxi * monod(c['no2'], bgc.nir_k_oxi)) / bgc.nir_y_oxi) * c['nir']
    nio_bio = torch.minimum((bgc.nio_vmax_red * monod(c['doc'], bgc.nio_k_red)) / bgc.nio_y_red, (bgc.nio_vmax_oxi * monod(c['no2'], bgc.nio_k_oxi)) / bgc.nio_y_oxi) * c['nio']
    nos_bio = torch.minimum((bgc.nos_vmax_red * monod(c['doc'], bgc.nos_k_red)) / bgc.nos_y_red, (bgc.nos_vmax_oxi * monod(c['n2o'], bgc.nos_k_oxi)) / bgc.nos_y_oxi) * c['nos']
    aoa_bio = torch.minimum((bgc.aoa_vmax_red * monod(c['nh4'], bgc.aoa_k_red)) / bgc.aoa_y_red, (bgc.aoa_vmax_oxi * monod(c['o2'], bgc.aoa_k_oxi)) / bgc.aoa_y_oxi) * c['aoa']
    nob_bio = torch.minimum((bgc.nob_vmax_red * monod(c['no2'], bgc.nob_k_red)) / bgc.nob_y_red, (bgc.nob_vmax_oxi * monod(c['o2'], bgc.nob_k_oxi)) / bgc.nob_y_oxi) * c['nob']
    aox_bio = torch.minimum((bgc.aox_vmax_red * monod(c['nh4'], bgc.aox_k_red)) / bgc.aox_y_red, (bgc.aox_vmax_oxi * monod(c['no2'], bgc.aox_k_oxi)) / bgc.aox_y_oxi) * c['aox']

    # grazing
    # c refers to the line at the top to prevent negative concentrations
    total_prey = c['aer'] + c['nar'] + c['nai'] + c['nao'] + c['nir'] + c['nio'] + c['nos'] + c['aoa'] + c['nob'] + c['aox']
    z_o2lim = torch.exp(-c['o2'] / bgc.zoo_o2lim)
    grazing_rate = (bgc.zoo_umax * (1.0 - z_o2lim)) * c['zoo'] * (1.0 / (total_prey + bgc.zoo_k_b))
    
    graze = {
        'aer': grazing_rate * c['aer'], 'nar': grazing_rate * c['nar'], 'nai': grazing_rate * c['nai'],
        'nao': grazing_rate * c['nao'], 'nir': grazing_rate * c['nir'], 'nio': grazing_rate * c['nio'],
        'nos': grazing_rate * c['nos'], 'aoa': grazing_rate * c['aoa'], 'nob': grazing_rate * c['nob'],
        'aox': grazing_rate * c['aox']
    }
    
    grazeC_total = sum(graze.values())
    zoo_bio = grazeC_total * bgc.zoo_g_z
    zoo_excretion_C = grazeC_total * (1.0 - bgc.zoo_g_z)
    
    # mortality & nitrogen recycling
    lossC_total = (mort(c['aer']) + mort(c['nar']) + mort(c['nai']) + mort(c['nao']) + mort(c['nir']) + 
                   mort(c['nio']) + mort(c['nos']) + mort(c['aoa']) + mort(c['nob']) + mort(c['aox']) + 
                   c['zoo'] * (bgc.zoo_m_l + c['zoo'] * bgc.zoo_m_q))
    
    N_from_biomass = lossC_total / bgc.CN_bio
    N_needed_for_det = lossC_total / bgc.CN_det
    lossN_to_NH4 = torch.clamp(N_from_biomass - N_needed_for_det, min=0.0)

    grazeN_to_NH4 = torch.clamp((grazeC_total / bgc.CN_bio) - (zoo_bio / bgc.CN_det), min=0.0)

    # chemical sms
    ddt = {}

    ddt['doc'] = -(aer_bio * bgc.aer_y_red + nar_bio * bgc.nar_y_red + nai_bio * bgc.nai_y_red + 
                   nao_bio * bgc.nao_y_red + nir_bio * bgc.nir_y_red + nio_bio * bgc.nio_y_red + 
                   nos_bio * bgc.nos_y_red) + lossC_total + zoo_excretion_C

    ddt['o2']  = -(aer_bio * bgc.aer_y_oxi + aoa_bio * bgc.aoa_y_oxi + nob_bio * bgc.nob_y_oxi)

    ddt['no3'] = (nob_bio * bgc.nob_e_no3 + aox_bio * bgc.aox_e_no3) - \
                 (nar_bio * bgc.nar_y_oxi + nai_bio * bgc.nai_y_oxi + nao_bio * bgc.nao_y_oxi)

    ddt['no2'] = (nar_bio * bgc.nar_e_no2 + aoa_bio * bgc.aoa_e_no2) - \
                 (nir_bio * bgc.nir_y_oxi + nio_bio * bgc.nio_y_oxi + nob_bio * bgc.nob_y_red + aox_bio * bgc.aox_y_oxi)

    ddt['nh4'] = (aer_bio * bgc.aer_e_nh4 + nar_bio * bgc.nar_e_nh4 + nai_bio * bgc.nai_e_nh4 + 
                  nao_bio * bgc.nao_e_nh4 + nir_bio * bgc.nir_e_nh4 + nio_bio * bgc.nio_e_nh4 + 
                  nos_bio * bgc.nos_e_nh4) - (aoa_bio * bgc.aoa_y_red + aox_bio * bgc.aox_y_red) + lossN_to_NH4 + grazeN_to_NH4

    ddt['n2o'] = (nir_bio * bgc.nir_e_n2o + nai_bio * bgc.nai_e_n2o) - (nos_bio * bgc.nos_y_oxi) 

    ddt['n2']  = (nao_bio * bgc.nao_e_n2 + nio_bio * bgc.nio_e_n2 + aox_bio * bgc.aox_e_n2 + nos_bio * bgc.nos_e_n2)

    # Biological sms
    ddt['aer'] = aer_bio - mort(c['aer']) - graze['aer']
    ddt['nar'] = nar_bio - mort(c['nar']) - graze['nar']
    ddt['nai'] = nai_bio - mort(c['nai']) - graze['nai']
    ddt['nao'] = nao_bio - mort(c['nao']) - graze['nao']
    ddt['nir'] = nir_bio - mort(c['nir']) - graze['nir']
    ddt['nio'] = nio_bio - mort(c['nio']) - graze['nio']
    ddt['nos'] = nos_bio - mort(c['nos']) - graze['nos']
    ddt['aoa'] = aoa_bio - mort(c['aoa']) - graze['aoa']
    ddt['nob'] = nob_bio - mort(c['nob']) - graze['nob']
    ddt['aox'] = aox_bio - mort(c['aox']) - graze['aox']
    ddt['zoo'] = zoo_bio - c['zoo'] * (bgc.zoo_m_l + c['zoo'] * bgc.zoo_m_q)

    gross_growth = {
        'aer': aer_bio, 'nar': nar_bio, 'nai': nai_bio, 'nao': nao_bio,
        'nir': nir_bio, 'nio': nio_bio, 'nos': nos_bio,
        'aoa': aoa_bio, 'nob': nob_bio, 'aox': aox_bio, 'zoo': zoo_bio
    }

    # placeholders to make it more similar to nitromz
    ref = c['o2']
    ddt['po4']       = torch.zeros_like(ref)
    ddt['poc']       = torch.zeros_like(ref)  # hydrolysis sink applied in physics.py
    ddt['n2o_ammox'] = torch.zeros_like(ref)
    ddt['n2o_denit'] = (nir_bio * bgc.nir_e_n2o + nai_bio * bgc.nai_e_n2o) - (nos_bio * bgc.nos_y_oxi)

    return ddt, gross_growth