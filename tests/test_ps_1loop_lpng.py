import pytest
import os, sys
import numpy as np

if sys.version_info.minor >= 11:
    import tomllib
else:
    import toml

import ps_1loop

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, 'test_data')

config_name_list = ['planck18_fid_z0', 
                    'planck18_fid_z1', 
                    'random_input_z0', 
                    'random_input_z1',
                    ]

term_name_list = ['22_gg', '13_gg', '12_gg_lpng', '22_gg_lpng', '13_gg_lpng']

test_item_list = []
for config_name in config_name_list:
    test_item_list += [(config_name, term_name) for term_name in term_name_list]

model = ps_1loop.PowerSpectrum1LoopLPNG()

@pytest.mark.parametrize('config_name, term_name', test_item_list)
def test_PowerSpectrum1loopLPNG(config_name: str, term_name: str):

    config_file = os.path.join(test_data_dir, 'input/%s.toml' % (config_name))
    if sys.version_info.minor >= 11:
        with open(config_file, 'rb') as f:
            config = tomllib.load(f)
    else:
        config = toml.load(config_file)

    ## Specify the linear matter power spectrum
    d = np.loadtxt(os.path.join(test_data_dir, 'input/pk_lin_%s.txt' % (config_name)))
    model.set_pk_lin(d[:,0], d[:,1] / config['cosmology']['Dgrowth']**2)

    ## Specify M(k) = sqrt( P_lin(k) / P_phi(k) )
    d = np.loadtxt(os.path.join(test_data_dir, 'input/Mk_%s.txt' % (config_name)))
    model.set_Mk(d[:,0], d[:,1] / config['cosmology']['Dgrowth'])
    
    ## Preparation for computing 1-loop terms
    model.set_1loop(hubble=config['cosmology']['h'])

    ## Set the redshift
    model.set_Dgrowth(Dgrowth=config['cosmology']['Dgrowth'])
    model.set_fgrowth(fgrowth=config['cosmology']['fgrowth'])

    ## Set f_NL
    model.set_f_nl(f_nl=1.)

    ## Set galaxy bias parameters
    bias = {key: value for key, value in config['galaxy_bias'].items()}
    model.set_bias_params(bias)
    
    ## Set EFT counterterm parameters
    ctr = {'c0':0, 'c2':0, 'c4':0, 'cfog':0}
    model.set_ctr_params(ctr)
    
    ## Set stochasticity parameters
    stoch = {'P_shot':0, 'a0':0, 'a2':0} # means no stochasticity
    model.set_stoch_params(stoch=stoch, ndens=3e-4, k_nl=0.45)

    ## Load test data
    k = np.linspace(0.01, 0.4, 40)
    mu = np.linspace(0, 1, 11)
    data_pkmu = np.load(os.path.join(test_data_dir, 'output/%s_%s.npy' % (config_name, term_name)))
    
    ## Output P(k,mu) as numpy array of shape (len(k), len(mu))
    # model_pkmu = model.get_pkmu_gg(k, mu, irres=False)
    model_pkmu = model.get_pkmu_gg_1loop(k, mu, name=term_name)

    k_tile = np.tile(k, (len(mu), 1)).T # shape (len(k), len(mu))
    kmin = 0.1
    assert np.allclose(data_pkmu[k_tile >= kmin], model_pkmu[k_tile >= kmin], rtol=2e-2, atol=2e-2)
