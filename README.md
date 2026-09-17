# ps_1loop_jax

A JAX code of the one-loop galaxy power spectrum in redshift space. 

The Eulerian and Lagrangian perturbation theory are provided and share the same parameter layout and public API:

| perturbation theory | class |
|---|---|
| Eulerian PT | `ps_1loop_jax.EPT` |
| Lagrangian PT | `ps_1loop_jax.LPT` |

Everything is written in JAX, so the models are `jit`-compiled and differentiable with respect to both the cosmological and the nuisance parameters. 

### Installation

After cloning this repository, run the following commands:

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

params = ps_params.make_params(
    f=0.578, h=0.6736,
    bias=jnp.array([2.0, -0.5, -0.2, 0.5]),          # b1, b2, bG2, bGamma3
    ctr=jnp.array([5.0, 10.0, -5.0, 0.0, 100.0]),    # c0, c2, c4, c6, c_nlo
    stoch=jnp.array([3333.0, 0.0, 0.0]),             # P_shot, a0, a2
)

k = jnp.linspace(1e-3, 0.3, 100)
pk_ells = ept.get_pk_ells(k, pk_data, params)      # (3, nk): P_0, P_2, P_4
```

Example notebooks: [Eulerian PT](example/ept.ipynb) and [Lagrangian PT](example/lpt.ipynb).

### Parameters

Both EPT and LPT use the same three groups:

| group | entries |
|---|---|
| `bias` | `b1, b2, bG2, bGamma3` |
| `ctr` | `c0, c2, c4, c6, c_nlo` |
| `stoch` | `P_shot, a0, a2` |

**`bias` is Eulerian in `EPT` and Lagrangian in `LPT`**.

The first four `ctr` entries multiply the leading `k^2` counterterm and `c_nlo` the next-to-leading `k^4` term.

The stochastic contribution is `P_shot + (a0 + a2 mu^2) k^2`, i.e. `stoch` carries the shot noise in `(Mpc/h)^3` directly.

`LPT` takes a `counterterm_base` argument to specify the base spectrum for the counterterms: `'linear_ir_resum'` (default), `'linear'`, or `'zeldovich'`. 

### Alcock-Pacynski effect

The Alcock-Pacynski effect is invoked by passing `alpha_perp` and `alpha_para` parameters:

```python
alpha_perp, alpha_para = 1.01, 0.99
pk_AP = ept.get_pk_ells(k, pk_data, params, alpha_perp=alpha_perp, alpha_para=alpha_para)
```

### Cross power spectrum

`Params` carries an optional `bias2`. Leaving it `None` (the default) gives the auto spectrum; setting it gives the cross spectrum of the tracers with biases `bias` and `bias2`. The bias monomials are symmetrised internally (`b1^2 -> b1 b1'`, `b1 b2 -> (b1 b2' + b1' b2)/2`, ...), while `ctr` and `stoch` keep the same functional form as in the auto case and are read as the cross spectrum's own coefficients, so they are set directly rather than derived from the two auto spectra:

```python
params_ab = ps_params.make_params(
    f=0.578, h=0.6736,
    bias=jnp.array([2.0, -0.5, -0.2, 0.5]),
    bias2=jnp.array([1.4, -0.5, 0.05, -0.1]),        # second tracer
    ctr=jnp.array([4.0, 8.0, -3.0, 0.0, 70.0]),      # counterterms of the cross spectrum
    stoch=jnp.array([0.0, 0.0, 0.0]),                # cross shot noise (0 for disjoint samples)
)

pk_ab = ept.get_pk_ells(k, pk_data, params_ab)
```

### Authors

- Yosuke Kobayashi (yosuke.kobayashi@cc.kyoto-su.ac.jp)
- Kazuyuki Akitsu

### Citations

- Yosuke Kobayashi & Kazuyuki Akitsu, TBD (in prep.)
