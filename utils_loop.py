import re

kernel_to_decomp_dict = {
    # matter
    '22': ['plin nu=-0.3','plin nu=-0.3'],
    '13': ['plin nu=-0.3'],
    # Gaussian bias
    'I_d2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_d2_d2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_d2_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_G2_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'F_G2': ['plin nu=-1.6'],
    # local PNG bias
    'I_phi_tilde': ['p1phi nu=-1.6','M nu=0.2'],
    'I_phi': ['p1phi nu=-1.6','plin nu=-0.3'],
    'I_phi_d': ['p1phi nu=-1.6','plin nu=-0.3'],
    'I_d2_LPNG': ['p1phi nu=-1.6','plin nu=-0.3'],
    'I_G2_LPNG': ['p1phi nu=-1.6','plin nu=-0.3'],
    'I_d2_phi-d': ['p1phi nu=-1.6','plin nu=-1.6'],
    'I_G2_phi-d': ['p1phi nu=-1.6','plin nu=-0.3'],
    'F_G2_LPNG': ['plin nu=-1.6']
}

kernel_name_dict = {
    # matter
    'linear': 'P_\mathrm{lin}',
    '22': 'P_{22}',
    '13': 'P_{13}',
    # Gaussian bias
    'I_d2': '\mathcal{I}_{\\delta^2}',
    'I_G2': '\mathcal{I}_{\mathcal{G}_2}',
    'I_d2_d2': '\mathcal{I}_{\\delta^2\\delta^2}',
    'I_d2_G2': '\mathcal{I}_{\\delta^2\mathcal{G}_2}',
    'I_G2_G2': '\mathcal{I}_{\mathcal{G}_2\mathcal{G}_2}',
    'F_G2': '\mathcal{F}_{\mathcal{G}_2}',
    # local PNG bias
    'I_phi_tilde': '\\tilde{\mathcal{I}}_{\\phi}',
    'I_phi': '\mathcal{I}_{\\phi}',
    'I_phi_d': '\mathcal{I}_{\\phi\\delta}',
    'I_d2_LPNG': '\mathcal{I}_{\\delta^2}^\mathrm{PNG}',
    'I_G2_LPNG': '\mathcal{I}_{\mathcal{G}_2}^\mathrm{PNG}',
    'I_d2_phi-d': '\mathcal{I}_{\\delta^2,\\phi\\delta}',
    'I_G2_phi-d': '\mathcal{I}_{\mathcal{G}_2,\\phi\\delta}',
    'F_G2_LPNG': '\mathcal{F}_{\mathcal{G}_2}^\mathrm{PNG}'
}

def get_deg_info(name):
    str_list = re.split('_', name)[2:]
    deg_dict = {}
    for s in str_list:
        string = re.split('-', s)
        deg_dict[string[0]] = int(string[1])
    nf = deg_dict['f']
    nmu = deg_dict['mu']
    _ = deg_dict.pop('f')
    _ = deg_dict.pop('mu')
    return nf, nmu, deg_dict
