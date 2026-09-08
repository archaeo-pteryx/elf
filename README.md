# ps_1loop_jax

A JAX code of the one-loop galaxy power spectrum in redshift space. 
The Eulerian and Lagrangian perturbation theory are provided and share the same parameter layout and public API:

| perturbation theory | class |  |
|---|---|---|
| Eulerian PT | `ps_1loop_jax.EPT` |  |
| Lagrangian PT | `ps_1loop_jax.LPT` |  |

Everything is written in JAX, so the models are `jit`-compiled and differentiable with respect to both the cosmological and the nuisance parameters. 

### Installation

After cloning this repository, run

```bash
cd ps_1loop_jax
python -m pip install -e .
```

### Basic usage

```python
import numpy as np, jax.numpy as jnp
import ps_1loop_jax
import ps_1loop_jax.params as ps_params

ept = ps_1loop_jax.EPT(do_irres=True)

d = np.loadtxt('example/data/pk_lin_planck18_fid_z0.txt')
pk_data = jnp.stack([d[:, 0], d[:, 1]], axis=0)      # (2, nk): k and P_lin(k)

params = ps_params.make_ept_params(
    f=0.578, h=0.6736,
    bias=jnp.array([2.0, -0.5, -0.2, 0.5]),          # b1, b2, bG2, bGamma3
    ctr=jnp.array([5.0, 10.0, -5.0, 0.0, 100.0]),    # c0, c2, c4, c6, c_nlo
    stoch=jnp.array([3333.0, 0.0, 0.0]),             # P_shot, a0, a2
)

k = jnp.linspace(1e-3, 0.3, 100)
pk_ells = ept.get_pk_ells(k, pk_data, params)      # (3, nk): P_0, P_2, P_4
```

Example notebooks: [Eulerian PT](example/ps_1loop_ept.ipynb) and [Lagrangian PT](example/ps_1loop_lpt.ipynb).

### Parameters

Both backends use the same three groups:

| group | entries |
|---|---|
| `bias` | `b1, b2, bG2, bGamma3` |
| `ctr` | `c0, c2, c4, c6, c_nlo` |
| `stoch` | `P_shot, a0, a2` |

The first four `ctr` entries multiply the leading `k^2` counterterm and `c_nlo` the next-to-leading `k^4` fingers-of-God term:

```
P_ctr = -2 k^2 (c0 + c2 f mu^2 + c4 f^2 mu^4 + c6 f^3 mu^6) P_base(k, mu)
        - c_nlo k^4 f^4 mu^4 P_tree(k, mu)
```

The stochastic contribution is `P_shot + (a0 + a2 mu^2) k^2`, i.e. `stoch` carries the shot noise in `(Mpc/h)^3` directly.

**`bias` is Eulerian in `EPT` and Lagrangian in `LPT`**.

`LPT` takes a `counterterm_base` argument selecting the spectrum the counterterms ride on: `'zeldovich'` (default, fused into the same Hankel transform), `'linear'`, or `'linear_ir_resum'` (which needs `h` in the parameters).

### Cross power spectra

Every public method takes an optional second parameter set. `params_b=None` (the default) gives the auto spectrum of `params_a`; passing both gives the cross spectrum, with the bias monomials symmetrised (`b1^2 -> b1_a b1_b`, `b1 b2 -> (b1_a b2_b + b1_b b2_a)/2`, ...) and the counterterm coefficients averaged.

Cross shot noise is an independent parameter rather than something derived from the two tracers, so it is passed explicitly:

```python
pk_ab = ept.get_pk_ells(k, pk_data, params_a, params_b, stoch=jnp.zeros(3))
```

### Authors

- Yosuke Kobayashi (yosuke.kobayashi@cc.kyoto-su.ac.jp)
- Kazuyuki Akitsu

### Citations

- Yosuke Kobayashi & Kazuyuki Akitsu, TBD (in prep.)
