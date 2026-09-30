# elf

A JAX code of the one-loop galaxy power spectrum in redshift space, in Eulerian (`elf.EPT`) and Lagrangian (`elf.LPT`) perturbation theory. Both classes have the same parameters and methods. Everything is `jit`-compiled and differentiable with respect to the linear power spectrum and all parameters.

### Installation

```bash
cd elf
python -m pip install -e .
```

### Usage

```python
import numpy as np, jax.numpy as jnp
import elf

model = elf.EPT()          # or elf.LPT()

d = np.loadtxt('example/data/pk_lin_planck18_fid_z0.txt')
pk_data = jnp.stack([d[:, 0], d[:, 1]], axis=0)      # (2, n): k [h/Mpc] and P_lin(k) [(Mpc/h)^3]

params = elf.make_params(
    f=0.578, h=0.6736,
    bias=jnp.array([2.0, -0.5, -0.2, 0.5]),                   # b1, b2, bG2, bGamma3
    ctr=jnp.array([5.0, 10.0, -5.0, 0.0, 100.0, 0.0, 0.0]),   # c0, c2, c4, c6, c44, c46, c48
    stoch=jnp.array([3333.0, 0.0, 0.0]),                      # P_shot, a0, a2
)

k = jnp.linspace(1e-3, 0.3, 100)
mu = jnp.linspace(0.0, 1.0, 11)
pkmu = model.get_pkmu(k, mu, pk_data, params)      # P(k, mu), shape (nk, nmu)
pk_ells = model.get_pk_ells(k, pk_data, params)    # P_0, P_2, P_4, shape (3, nk)
```

Both methods take the Alcock-Paczynski parameters as `alpha_perp=1.0, alpha_para=1.0`. Example notebooks: [EPT](example/ept.ipynb), [LPT](example/lpt.ipynb).

### Parameters

| group | entries |
|---|---|
| `bias` | `b1, b2, bG2, bGamma3` (Eulerian in `EPT`, Lagrangian in `LPT`) |
| `ctr` | `c0, c2, c4, c6, c44, c46, c48` |
| `stoch` | `P_shot, a0, a2` |

The counterterms are

```
P_ctr = [ -2 (c0 + c2 f mu^2 + c4 f^2 mu^4 + c6 f^3 mu^6) k^2
          -2 (c44 + c46 f mu^2 + c48 f^2 mu^4) (k mu f)^4 ] P_base(k, mu)
```

with `P_base` set by the constructor argument `counterterm_base` (`'linear_ir_resum'` by default, `'linear'`, or `'zeldovich'` for `LPT`), and the stochastic term is `P_shot + (a0 + a2 mu^2) k^2`.

### Changing the parameters

The constructor only fixes the numerical setup. For another spectrum, call the same object again with new `params` (or a new `pk_data`); nothing is recompiled:

```python
params2 = elf.make_params(f=0.55, h=0.70, bias=[1.8, -0.3, -0.1, 0.2], ctr=[0.0] * 7, stoch=[0.0, 0.0, 0.0])
pk_ells2 = model.get_pk_ells(k, pk_data, params2)
```

### Cross power spectrum

Give the second tracer's bias as `bias2` (without it the auto spectrum is returned). `ctr` and `stoch` are the coefficients of the cross spectrum itself. Counterterms built from each tracer's own coefficients, `[Z1^A c^B(mu) + Z1^B c^A(mu)] k^2 P` with `Z1 = b1 + f mu^2` and `c^X(mu) = c0^X + c2^X mu^2 + c4^X mu^4` (and likewise at NLO), are a polynomial in `mu^2`; pass them by matching its coefficients to the form above. The cross shot noise is a parameter of its own (zero for disjoint samples).

```python
params_ab = elf.make_params(
    f=0.578, h=0.6736,
    bias=jnp.array([2.0, -0.5, -0.2, 0.5]),        # tracer a
    bias2=jnp.array([1.4, -0.5, 0.05, -0.1]),      # tracer b
    ctr=jnp.array([4.0, 8.0, -3.0, 0.0, 70.0, 0.0, 0.0]),
    stoch=jnp.array([0.0, 0.0, 0.0]),              # cross shot noise (0 for disjoint samples)
)
pk_ab = model.get_pk_ells(k, pk_data, params_ab)
```

### Main options

| option | default | meaning |
|---|---|---|
| `ells` | `(0, 2, 4)` | multipoles returned by `get_pk_ells` |
| `ngauss` | `4` | Gauss-Legendre points on `mu` in `[0, 1]` for the multipoles (needs `max(ells) < 2 ngauss`) |
| `counterterm_base` | `'linear_ir_resum'` | base spectrum of the counterterms |
| `subtract_k0_limit` | `False` | drop the constant `b2^2` term of `P_22` |
| `method` (`EPT`) | `'hankel'` | `'hankel'` or `'matrix'` (the latter valid for `k` in `[3e-3, 3]` h/Mpc) |
| `do_irres` (`EPT`) | `True` | IR resummation of the BAO |
| `lmax` (`LPT`) | `5` | order of the angular expansion of the LPT integral; increase for accuracy at high `k` |

### Authors

- Yosuke Kobayashi (yosuke.kobayashi@cc.kyoto-su.ac.jp)
- Kazuyuki Akitsu (kakitsu@ias.edu)

### Citations

- Yosuke Kobayashi & Kazuyuki Akitsu, TBD (in prep.)
</content>
