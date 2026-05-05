import torch


def get_n2o_yield(o2, bgc):
    """
    Converted from n2o_yield.m
    Calculates N2O and NO2 yields during Ammox and nitrifier-denitrification.
    """
    Y = {}
    
    if bgc.n2o_yield == 'Ji':
        # scale original params
        a1 = bgc.Ji_a / 100.0
        b1 = bgc.Ji_b / 100.0
        
        # total yields
        Y['nn2o_nh4'] = (a1 + b1 * o2) / (a1 + (b1 + 1.0) * o2)
        Y['no2_nh4']  = o2 / (a1 + (b1 + 1.0) * o2)
        
        # get yields for Hydroxylamine pathway
        Y['nn2o_hx_nh4'] = b1 / (b1 + 1.0)
        
        # get yields for nitrifier-denitrification pathway
        Y['nn2o_nden_nh4'] = 1.0 / ((b1 + 1.0) * (1.0 + ((b1 + 1.0) / a1) * o2))
        Y['no2_nden_nh4']  = -Y['nn2o_nden_nh4']  # nh4->no2->n2o
        
        # get yields for Hydroxylamine pathway by difference
        Y['no2_hx_nh4'] = 1.0 - Y['nn2o_hx_nh4'] - Y['nn2o_nden_nh4']
        
    return Y