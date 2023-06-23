# PowerSpectrumTheory

A Python code to compute the one-loop galaxy power spectrum, including Primordial non-Gaussianity.

This is a pure-Python implementation of the galaxy power spectrum following the Effective Field Theory of Large-Scale Structure, described in Chudaykin et al. 2020 (arXiv:2004.10607) for the Gaussian initial condition case, and in Cabass et al. 2022 (arXiv:2204.01781) for the local PNG case.
The implementation includes the galaxy bias expansion up to the third order, the redshift-space distortion, the ultraviolet counterterms, the infrared resummation, and the Alcock-Paczynski effect.

## Requirements

The following packages need to be installed (version information is to be determined). 

- numpy
- scipy
- sympy
- camb (https://camb.readthedocs.io/en/latest/)

