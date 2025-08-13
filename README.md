# ps_1loop_jax

A JAX code to compute the one-loop galaxy power spectrum.

This is a JAX implementation of the galaxy power spectrum following the Effective Field Theory of Large-Scale Structure (EFTofLSS) described in [Chudaykin et al. (2020)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.063533).

The implementation includes the galaxy bias expansion up to the third order, the redshift-space distortions, the ultraviolet counterterms, the infrared resummation, and the Alcock-Paczynski effect. 

### Installation

After cloning this repository, run 

```bash
cd ps_1loop_jax
python -m pip install -e .
```

### Basic Usage

You can find an example Jupyter notebook [here](example/ps_1loop_ept.ipynb) for Eulerian PT.

### Authors

- Yosuke Kobayashi (yosuke.kobayashi@cc.kyoto-su.ac.jp)

### Citations

- Yosuke Kobayashi & Kazuyuki Akitsu, TBD (in prep.)