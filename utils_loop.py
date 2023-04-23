import re

kernel_to_decomp_dict = {
    # matter
    '22_mm': ['plin nu=-0.3','plin nu=-0.3'],
    '13_mm': ['plin nu=-0.3','plin nu=-0.3'],
    # biased tracer (Gaussian)
    'I_d2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_d2_d2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_d2_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_G2_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'F_G2': ['plin nu=-1.6','plin nu=-1.6'],
    # local PNG
    'I_phi': ['p1phi nu=-1.6','plin nu=-0.3'],
    'I_phi-d': ['p1phi nu=-1.6','plin nu=-1.6'],
    'I_d2_phi': ['p1phi nu=-1.6','plin nu=-1.6'],
    'I_G2_phi': ['p1phi nu=-1.6','plin nu=-1.6'],
    'I_d2_phi-d': ['p1phi nu=-1.6','plin nu=-1.6'],
    'I_G2_phi-d': ['p1phi nu=-1.6','plin nu=-1.6'],
    'F_phi': ['p1phi nu=-1.6','plin nu=-1.6'],
    'F_G2_LPNG': ['plin nu=-1.6','p1phi nu=-1.6'],
    # 'I_phi_tilde': ['p1phi nu=-1.6','M nu=0.2'],
    # 'I_phi_tilde_d2': ['p1phi nu=-1.6','M nu=0.2'],
    # 'I_phi_tilde_G2': ['p1phi nu=-1.6','M nu=0.2'],
    'I_phi_tilde': ['p1phi nu=-2.1','p1phi nu=-2.1'],
    'I_phi_tilde_d2': ['p1phi nu=-2.1','p1phi nu=-2.1'],
    'I_phi_tilde_G2': ['p1phi nu=-2.1','p1phi nu=-2.1'],
    # biased tracer in redshift space (Gaussian)
    # '22_gg': ['plin nu=-1.6','plin nu=-1.6'],
    # '13_gg': ['plin nu=-1.6','plin nu=-1.6'],
    '22_gg': ['plin nu=-0.7','plin nu=-0.7'],
    '13_gg': ['plin nu=-0.7','plin nu=-0.7'],
    # biased tracer in redshift space (local PNG contribution)
    # '22_gg_lpng': ['plin nu=-1.6','p1phi nu=-1.6'],
    '22_gg_lpng': ['plin nu=-0.7','p1phi nu=-0.9'],
    '13_gg_lpng1': ['plin nu=-0.7','p1phi nu=-0.9'],
    '13_gg_lpng3': ['p1phi nu=-1.6','plin nu=-1.6'],
    '12_gg_lpng1': ['p1phi nu=-1.6','M nu=0.2'],
    '12_gg_lpng2': ['p1phi nu=-2.1','p1phi nu=-2.1']
}

kernel_name_dict = {
    # matter
    'linear': 'P_\mathrm{lin}',
    '22_mm': 'P_{22}',
    '13_mm': 'P_{13}',
    # biased tracer (Gaussian)
    'I_d2': '\mathcal{I}_{\\delta^2}',
    'I_G2': '\mathcal{I}_{\mathcal{G}_2}',
    'I_d2_d2': '\mathcal{I}_{\\delta^2\\delta^2}',
    'I_d2_G2': '\mathcal{I}_{\\delta^2\mathcal{G}_2}',
    'I_G2_G2': '\mathcal{I}_{\mathcal{G}_2\mathcal{G}_2}',
    'F_G2': '\mathcal{F}_{\mathcal{G}_2}',
    # local PNG
    'I_phi': '\mathcal{I}_{\\phi}',
    'I_phi-d': '\mathcal{I}_{\\phi\\delta}',
    'I_d2_phi': '\mathcal{I}_{\\delta^2,\\phi}',
    'I_G2_phi': '\mathcal{I}_{\mathcal{G}_2,\\phi}',
    'I_d2_phi-d': '\mathcal{I}_{\\delta^2,\\phi\\delta}',
    'I_G2_phi-d': '\mathcal{I}_{\mathcal{G}_2,\\phi\\delta}',
    'F_phi': '\mathcal{F}_{\\phi}',
    'F_G2_LPNG': '\mathcal{F}_{\mathcal{G}_2}^\mathrm{PNG}',
    'I_phi_tilde': '\\tilde{\mathcal{I}}_{\\phi}',
    'I_phi_tilde_d2': '\\tilde{\mathcal{I}}_{\\phi,\\delta^2}',
    'I_phi_tilde_G2': '\\tilde{\mathcal{I}}_{\\phi,\mathcal{G}_2}'
}

def get_deg_info(name):
    deg_name = re.split('=', name)[-1]
    str_list = re.split('_', deg_name)
    deg_dict = {}
    for s in str_list:
        string = re.split('-', s)
        deg_dict[string[0]] = int(string[1])
    nf = deg_dict['f']
    nmu = deg_dict['mu']
    _ = deg_dict.pop('f')
    _ = deg_dict.pop('mu')
    return nf, nmu, deg_dict
