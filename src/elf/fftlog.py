"""FFTLog / Mellin primitives shared by the EPT and LPT backends.

Conventions
-----------
Every logarithmic-grid transform in this package is built on one primitive: the
discrete Mellin coefficients of a function tabulated on a geometric grid
``x_i = x_0 e^{i dln}``,

    c_m = rfft( f(x) * x^{-nu} * W(x) )                    (``mellin_coefficients``)

with ``nu`` the FFTLog bias and ``W`` an optional end taper (``window``).  The
result is a *half* spectrum: only the non-negative rfft frequencies are stored,

    eta_half[m] = 2 pi rfftfreq(n, d=1) / dln,   m = 0 ... n//2   (``LogGrid.eta_half``)

Two different discrete models are built on that single primitive.  They are
deliberately kept apart: they share the primitive, not the coefficient arrays.

* **Power-law decomposition (PLD) / matrix backend.**  Works on the *unpadded*
  grid (``n_pad = 0``) with ``norm='forward'``, and re-expands the half spectrum
  into the full, symmetrically ordered set of complex frequencies

      eta_full[m] = 2 pi / (n dln) * (m - n//2),  m = 0 ... n-1   (``LogGrid.eta_full``)

  Because the rfft measures phases relative to the first grid node while the
  power-law expansion below uses ``x`` itself as the variable, the re-expansion
  also carries the phase origin ``x[0]``::

      c_m = x_0^{-i eta_m} * [ conj(c_half[::-1][:-1]), c_half[:-1] ]
      nu_m = nu + i eta_m                                   (``full_spectrum``)

  so that ``f(x) = Re sum_m c_m x^{nu_m}``.  ``power_law_xpow`` tabulates
  ``x^{nu_m}`` and ``power_law_terms`` forms ``c_m x^{nu_m}``; the analytic
  one-loop kernels in ``pt_matrix`` are then contracted with that expansion.

* **Hankel / FFTLog backend.**  Works on a *zero/power-law padded* grid
  (``n_pad > 0``), keeps the half spectrum as it comes out of the rfft, and
  multiplies it by the Mellin transform of the spherical Bessel function
  (``mellin_g``) evaluated on ``eta_half``.  The output grid is fixed by the
  low-ringing phase (``low_ringing_phase``, ``output_grid``) and the inverse
  transform is ``hankel_transform``.  ``hankel`` chains pad -> coefficients ->
  inverse for the common case.

The two models use different transform lengths (padded vs unpadded), different
frequency layouts (half vs full, symmetric) and different normalisations
(``None`` vs ``'forward'``), so their coefficients are *not* interchangeable.

Who uses what
-------------
* ``ept.EPT._get_pld_xpow`` / ``_get_decomp_pq_from_xpow`` / ``_set_matrix`` /
  ``get_pkmu_1loop_pld``: ``LogGrid.eta_full``, ``mellin_coefficients`` with
  ``norm='forward'``, ``full_spectrum``, ``power_law_xpow``, ``power_law_terms``.
* ``ept.EPT._set_hankel`` and ``lpt.LPT._set_hankel``: ``LogGrid.from_core``
  (via ``extend_log_grid``), ``window``, ``low_ringing_phase``, ``output_grid``,
  ``hankel_kernel``.
* EPT/LPT transform methods: ``hankel`` when one source feeds one kernel, and
  ``mellin_coefficients`` + ``hankel_transform`` when a single forward rfft is
  shared by several ``(l, n)`` kernels (the PLD forward path).

Grid arguments
--------------
The construction-time Hankel helpers (``low_ringing_phase``, ``output_grid``,
``hankel_kernel``) accept either a bare grid array or a :class:`LogGrid`.  A
backend's ``_set_hankel`` builds one ``LogGrid`` and passes it to all of them so
that ``dln`` and ``eta_half`` are computed once instead of once per kernel.  The
power-law-decomposition helpers (``full_spectrum``, ``power_law_xpow``,
``power_law_terms``) always take the bare unpadded grid, because they run inside
the traced per-call code rather than at construction time.

Padding and cropping
--------------------
``LogGrid.crop`` and ``hankel_transform(crop=True)`` both slice
``[..., n_pad : n - n_pad]`` rather than ``[..., n_pad : -n_pad]`` so that
``n_pad = 0`` is the identity.  ``hankel_transform(crop=False)`` returns the
whole padded output, which is the entry point for propagating the FFTLog guard
band between chained transforms.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.special import loggamma

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Logarithmic grids
# ---------------------------------------------------------------------------

def extend_log_grid(x, n_low, n_high):
    """Geometrically continue a log-spaced grid by ``n_low``/``n_high`` nodes.

    The low side is continued with the ratio ``x[1]/x[0]`` and the high side
    with ``x[-1]/x[-2]``; for a strictly geometric input both equal ``e^{dln}``.
    """
    ratio = x[1] / x[0]
    exp = jnp.arange(-n_low, 0)
    _x = x[0] * (ratio ** exp)
    ratio = x[-1] / x[-2]
    exp = jnp.arange(1, n_high + 1)
    x_ = x[-1] * (ratio ** exp)
    return jnp.concatenate([_x, x, x_], axis=0)


@dataclass(frozen=True, eq=False)
class LogGrid:
    """A geometric grid plus the size of its symmetric padding.

    ``x`` is the *full* (already padded) grid; the physical part is
    ``x[n_pad : n - n_pad]``.  ``n_pad = 0`` describes an unpadded grid, for
    which ``crop`` is the identity.
    """

    x: jnp.ndarray
    n_pad: int = 0

    # ``dln``/``eta_half``/``eta_full`` are cached per instance: they cost a
    # handful of JAX dispatches each and every kernel built in ``_set_hankel``
    # needs them.  ``cached_property`` writes straight into ``__dict__``, which
    # a frozen dataclass still has.

    @classmethod
    def from_core(cls, x_core, n_pad):
        """Build a grid by padding ``x_core`` symmetrically with ``n_pad`` nodes."""
        if n_pad == 0:
            return cls(x_core, 0)
        return cls(extend_log_grid(x_core, n_pad, n_pad), n_pad)

    @property
    def n(self):
        """Length of the full (padded) grid."""
        return self.x.shape[0]

    @property
    def n_core(self):
        """Length of the physical (unpadded) part."""
        return self.n - 2 * self.n_pad

    @cached_property
    def dln(self):
        """Logarithmic spacing ``log(x[1]/x[0])``."""
        return jnp.log(self.x[1] / self.x[0])

    @property
    def x_core(self):
        """The physical part of the grid."""
        return self.crop(self.x)

    @cached_property
    def eta_half(self):
        """Non-negative rfft frequencies, ``2 pi rfftfreq(n) / dln``."""
        return 2 * jnp.pi * jnp.fft.rfftfreq(self.n, d=1.0) / self.dln

    @cached_property
    def eta_full(self):
        """Full, symmetrically ordered frequencies used by the PLD/matrix model."""
        n = self.n
        return 2 * jnp.pi / (n * self.dln) * (jnp.arange(n) - n // 2)

    def crop(self, arr):
        """Drop the padding from the last axis of ``arr`` (identity if ``n_pad == 0``)."""
        return arr[..., self.n_pad:self.n - self.n_pad]


def _as_grid(x):
    """Accept either a bare log grid or a :class:`LogGrid`.

    ``dln``, ``eta_half`` and ``eta_full`` do not depend on ``n_pad``, so a
    padded ``LogGrid`` and ``LogGrid(x, 0)`` give identical arrays; passing the
    grid object simply reuses the cached ones.
    """
    return x if isinstance(x, LogGrid) else LogGrid(x, 0)


# ---------------------------------------------------------------------------
# Padding and windowing of the source function
# ---------------------------------------------------------------------------

def func_window(x: jnp.ndarray) -> jnp.ndarray:
    """Smooth 0 -> 1 taper on ``x`` in [0, 1] with vanishing first derivative."""
    return x - jnp.sin(2.0 * jnp.pi * x) / (2.0 * jnp.pi)


def window(n_window, array_x) -> jnp.ndarray:
    """Spatial end taper of width ``n_window`` nodes (ones when ``n_window <= 0``)."""
    if n_window <= 0:
        return jnp.ones_like(array_x)
    win = jnp.ones_like(array_x)
    i_left = jnp.arange(n_window, dtype=jnp.float64)
    x_left = (i_left + 1) / (n_window + 1)
    win = win.at[:n_window].set(func_window(x_left))

    i_right = jnp.arange(n_window, dtype=jnp.float64)
    x_right = (i_right + 1) / (n_window + 1)
    win = win.at[-n_window:].set(func_window(x_right[::-1]))
    return win


def pad(array_y, n_pad, mode='power-law'):
    """Extend the last axis of ``array_y`` by ``n_pad`` nodes on each side.

    ``'power-law'`` continues the endpoint log-slope, ``'zero-pad'`` pads with
    zeros, ``'smooth-zero-pad'`` tapers the endpoint value to zero with
    ``func_window``.
    """
    if mode == 'power-law':
        tiny = jnp.finfo(array_y.dtype).tiny
        ratio_low = jnp.where(jnp.abs(array_y[..., 0]) > tiny, array_y[..., 1] / array_y[..., 0], 1.0)
        ratio_high = jnp.where(jnp.abs(array_y[..., -2]) > tiny, array_y[..., -1] / array_y[..., -2], 1.0)

        exp_low = jnp.arange(-n_pad, 0)
        exp_high = jnp.arange(1, n_pad + 1)
        expand = (None,) * (array_y.ndim - 1)
        array_low = array_y[..., :1] * ratio_low[..., None] ** exp_low[expand]
        array_high = array_y[..., -1:] * ratio_high[..., None] ** exp_high[expand]
        return jnp.concatenate([array_low, array_y, array_high], axis=-1)
    elif mode == 'zero-pad':
        pad_width = [(0, 0)] * (array_y.ndim - 1) + [(n_pad, n_pad)]
        return jnp.pad(array_y, pad_width)
    elif mode == 'smooth-zero-pad':
        x = (jnp.arange(n_pad, dtype=array_y.dtype) + 1.0) / (n_pad + 1.0)
        taper = func_window(x)
        expand = (None,) * (array_y.ndim - 1)
        array_low = array_y[..., :1] * taper[expand]
        array_high = array_y[..., -1:] * taper[::-1][expand]
        return jnp.concatenate([array_low, array_y, array_high], axis=-1)
    else:
        raise ValueError(f"unknown pad mode: {mode!r}")


# ---------------------------------------------------------------------------
# The single rfft definition
# ---------------------------------------------------------------------------

def mellin_coefficients(fx, x, nu, window=None, norm=None):
    """Half-spectrum Mellin coefficients ``rfft(fx * x^{-nu} * window)``.

    This is the only place in the package where the forward rfft of a
    log-gridded function is taken.  ``fx`` may carry leading batch axes; ``nu``
    may be an array broadcastable against ``x`` (e.g. ``nu[:, None]`` for a
    batch of biases sharing one source).  ``norm`` is passed through to
    ``jnp.fft.rfft``: ``None`` for the Hankel model, ``'forward'`` for the
    PLD/matrix model.
    """
    weighted = fx * x ** (-nu)
    if window is not None:
        weighted = weighted * window
    return jnp.fft.rfft(weighted, norm=norm)


# ---------------------------------------------------------------------------
# Power-law decomposition (PLD / matrix model)
# ---------------------------------------------------------------------------

def full_spectrum(c_half, x, nu):
    """Re-expand a half spectrum into the full symmetric PLD spectrum.

    Returns ``(c_m, nu_m)`` with ``nu_m = nu + i eta_full`` and ``c_m`` carrying
    the ``x[0]`` phase origin, so that ``f(x) = Re sum_m c_m x^{nu_m}``.
    ``c_half`` must be one-dimensional and must come from
    ``mellin_coefficients(..., norm='forward')`` on the unpadded grid ``x``.
    """
    eta = LogGrid(x, 0).eta_full
    c_m = x[0] ** (-1j * eta) * jnp.concatenate([c_half[::-1][:-1].conj(), c_half[:-1]], axis=0)
    nu_m = nu + 1j * eta
    return c_m, nu_m


def power_law_xpow(nu_m, x):
    """Tabulate ``x^{nu_m}`` with shape ``(len(nu_m), len(x))``."""
    return x[None, :] ** nu_m[:, None]


def power_law_terms(c_m, xpow):
    """Form the per-mode power-law terms ``c_m x^{nu_m}``."""
    return c_m[:, None] * xpow


# ---------------------------------------------------------------------------
# Hankel / FFTLog model
# ---------------------------------------------------------------------------

def mellin_g(ell, z, eps=1e-15):
    """Mellin transform of the spherical Bessel function ``j_ell``.

    ``g_ell(z) = sqrt(pi) 2^{z-2} Gamma((ell+z)/2) / Gamma((3+ell-z)/2)``.
    Evaluated in numpy/scipy (``loggamma``); the imaginary part is nudged by
    ``eps`` to stay off the real axis where the ratio has poles.
    """
    z = np.asarray(z, dtype=np.complex128)
    z = np.where(np.abs(z.imag) < eps, z + 1j * eps, z)
    g_l = np.sqrt(np.pi) * np.exp(
        np.log(2.0) * (z - 2) + loggamma(0.5 * (ell + z)) - loggamma(0.5 * (3 + ell - z))
    )
    return g_l


def low_ringing_phase(ell, nu, x):
    """Low-ringing output-grid phase ``lnxy`` for kernel ``j_ell`` and bias ``nu``.

    ``x`` may be a bare grid array or a :class:`LogGrid`.
    """
    dln = _as_grid(x).dln
    return dln * jnp.angle(mellin_g(ell, nu + 1j * jnp.pi / dln)) / jnp.pi


def output_grid(x, lnxy):
    """Output grid conjugate to ``x`` for the low-ringing phase ``lnxy``.

    ``x`` may be a bare grid array or a :class:`LogGrid`.
    """
    grid = _as_grid(x)
    return jnp.exp(lnxy - grid.dln) / grid.x[::-1]


def hankel_kernel(ell, nu, x, lnxy):
    """Mellin kernel ``u_m = e^{-i eta lnxy} g_ell(nu + i eta)`` on ``eta_half``.

    ``x`` may be a bare grid array or a :class:`LogGrid`.

    The output phase is unified here as ``exp(-i eta ln(xy))``; the pre-refactor
    main ``_set_hankel`` path instead used the algebraically identical
    ``exp(lnxy) ** (-i eta)``, which differs from it by ~1e-15 in the last bits.
    That rounding difference is not uniformly harmless: the EPT P22 ``m = 0``
    blend is non-smooth and amplifies it to ~1e-7 in ``dP/dP_lin`` at
    ``k > 50`` h/Mpc -- a property of the blend, left for the Phase 3/4 work.
    """
    eta = _as_grid(x).eta_half
    return jnp.exp(lnxy * (-1j * eta)) * mellin_g(ell, nu + 1j * eta)


def hankel_transform(c_half, u_m, y, nu, n_pad, crop=True, y_pow=None):
    """Inverse leg of the FFTLog: ``irfft(conj(c_half * u_m)) * y^{-nu}``.

    ``y`` is the *padded* output grid.  ``y_pow`` may hold a precomputed
    ``y ** (-nu)``; the PLD forward path caches one such factor per ``(l, n)``
    kernel because each row has its own output grid and bias.  When ``y_pow``
    is given, ``y`` and ``nu`` are unused and may be ``None``.

    With ``crop=True`` the padding is removed from the last axis (``n_out`` is
    the padded output length, i.e. ``y.shape[-1]``); with ``crop=False`` the
    full padded result is returned, which is what a chained transform needs in
    order to keep the FFTLog guard band.
    """
    res = jnp.fft.irfft(jnp.conj(c_half * u_m))
    res = res * (y ** (-nu) if y_pow is None else y_pow)
    if not crop:
        return res
    n_out = res.shape[-1]
    return res[..., n_pad:n_out - n_pad]


def hankel(nu, fx_core, x, y, u_m, n_pad, window, pad_mode='zero-pad', crop=True):
    """Full FFTLog pass: pad ``fx_core``, take Mellin coefficients, transform back.

    ``fx_core`` is given on the physical grid and may carry leading batch axes;
    ``x`` is the padded input grid, ``y`` the padded output grid and ``u_m`` the
    kernel from :func:`hankel_kernel` (possibly batched along the leading axis).
    """
    fx_pad = pad(fx_core, n_pad, mode=pad_mode)
    c_half = mellin_coefficients(fx_pad, x, nu, window=window)
    return hankel_transform(c_half, u_m, y, nu, n_pad, crop=crop)
