# PowerSpectrumTheory

A Python code to compute the one-loop galaxy power spectrum, including Primordial non-Gaussianity (currently local PNG only).

This is a pure-Python implementation of the galaxy power spectrum following the Effective Field Theory of Large-Scale Structure (EFTofLSS), described in [Chudaykin et al. (2020)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.063533) for the Gaussian initial condition case, and in [Cabass et al. (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.043506) for the local PNG case.

The implementation includes the galaxy bias expansion up to the third order, the redshift-space distortions, the ultraviolet counterterms, the infrared resummation, and the Alcock-Paczynski effect. 

### Installation

After cloning this repository, run 

```bash
cd PowerSpectrumTheory
python -m pip install -e .
```

### Basic Usage

You can find an example Jupyter notebook [here](example/ps_1loop.ipynb).

### Authors

Yosuke Kobayashi (yosukekobayashi@arizona.edu) 
