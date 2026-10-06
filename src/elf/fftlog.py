"""FFTlog / Mellin primitives shared by the EPT and LPT backends.

Two discrete models share the Mellin coefficients ``c_m = rfft(f(x) x^{-nu})``
on a geometric grid: the power-law decomposition (EPT ``method='matrix'``;
unpadded grid, ``norm='forward'``, full symmetric frequencies with phase origin
``x[0]``, ``f = Re sum_m c_m x^{nu + i eta_m}``) and the Hankel transform
(:func:`hankel`; padded grid, half spectrum times the kernel of
:func:`hankel_setup`).  Their coefficients are not interchangeable.  Only the
input P(k) is padded; later transforms take the padded output of the previous
one, restricted to its support by :func:`trapezoid_weights`.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
from scipy.special import loggamma

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from .utils import get_log_extrap


@dataclass(frozen=True)
class FFTlogSettings:
    """Numerical constants of the FFTlog pipelines (for convergence studies, not tuning)."""

    nu: float = 1.1
    """Bias of the ordinary Hankel transforms of both backends.  Its low-ringing phase fixes
    the output grids ``q_padded[l]`` of every forward transform (:func:`forward_transform`),
    and it is the kernel bias of the LPT P -> xi_l^n rows."""

    nu_n0: float = 1.5
    """LPT: bias of the source -> correlator transforms with ``n >= 0``, at any order l
    (currently only ``theta``).  It suppresses the periodic image ``e^{-nu L}`` of their
    q -> 0 plateau."""

    nu_xi: float = -0.3
    """EPT: input bias of the P -> xi_l^n transforms with ``n != 0`` (one rfft shared by
    those rows, the kernel of row (l, n) has the bias ``nu_xi + n + 3``)."""

    nu_xi_n0: float = -1.5
    """The same for the ``n = 0`` rows of EPT and for the P_ell -> xi_ell transforms of
    ``get_xi_ells`` (both backends)."""

    nu_matrix_matter: float = -0.3
    """EPT matrix: decomposition bias of the matter P22 and P13 blocks."""

    nu_matrix_p22_bias: float = -1.0
    """EPT matrix: bias for the P22 bias-operator blocks; above -3/2 so they give ``I(k) - I(0)``."""

    nu_matrix_p13_bias: float = -1.6
    """EPT matrix: bias for the P13 bias-operator blocks."""

    npad_factor: float = 0.5
    """Padded band on each side, ``n_pad = max(1, int(npad_factor * nfft))``."""


@dataclass(frozen=True, eq=False)
class LogGrid:
    """Padded geometric grid ``x`` (core ``x[n_pad : n - n_pad]``) with spacing ``dln``.

    ``dln`` is passed explicitly for conjugate grids: they keep the input spacing,
    which their own nodes reproduce only up to rounding.
    """

    x: jnp.ndarray
    n_pad: int = 0
    dln: jnp.ndarray = None

    def __post_init__(self):
        if self.dln is None:
            object.__setattr__(self, 'dln', jnp.log(self.x[1] / self.x[0]))

    @classmethod
    def from_core(cls, x_core, n_pad):
        """Pad ``x_core`` with ``n_pad`` nodes on each side (end ratios continued)."""
        if n_pad == 0:
            return cls(x_core, 0)
        x_low = x_core[0] * ((x_core[1] / x_core[0]) ** jnp.arange(-n_pad, 0))
        x_high = x_core[-1] * ((x_core[-1] / x_core[-2]) ** jnp.arange(1, n_pad + 1))
        return cls(jnp.concatenate([x_low, x_core, x_high], axis=0), n_pad)

    @property
    def n(self):
        """Length of the full (padded) grid."""
        return self.x.shape[0]

    @property
    def n_core(self):
        """Length of the physical (unpadded) part."""
        return self.n - 2 * self.n_pad

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
        """Full, symmetrically ordered frequencies of the power-law decomposition."""
        n = self.n
        return 2 * jnp.pi / (n * self.dln) * (jnp.arange(n) - n // 2)

    def crop(self, arr):
        """Drop the padding from the last axis (``n - n_pad``, not ``-n_pad``: identity for n_pad = 0)."""
        return arr[..., self.n_pad:self.n - self.n_pad]

    def conjugate(self, lnxy):
        """Output grid ``y_j = e^{lnxy - dln} / x_{n-1-j}`` (an involution for fixed ``lnxy``)."""
        return LogGrid(jnp.exp(lnxy - self.dln) / self.x[::-1], self.n_pad, self.dln)

    def moment(self, fx, power):
        """``sum_j dln x_j^power f_j`` over the last axis: the m = 0 coefficient FFTlog sees.

        The package's only trapezoid rule on a log grid; end weights come folded into ``fx``.
        """
        return jnp.sum(self.dln * self.x ** power * fx, axis=-1)


def pad(array, grid, mode):
    """Extend ``array`` across the padded band: zeros or the end log-slopes (``'power-law'``)."""
    n_pad = grid.n_pad
    if mode == 'zero-pad':
        pad_width = [(0, 0)] * (array.ndim - 1) + [(n_pad, n_pad)]
        return jnp.pad(array, pad_width)
    if mode == 'power-law':
        # nodes rebuilt inside the trace, not grid.x: XLA rounds precomputed constants differently
        x = LogGrid.from_core(grid.x_core, n_pad).x
        array_low, array_high = get_log_extrap(grid.x_core, array, x[:n_pad], x[x.shape[0] - n_pad:])
        return jnp.concatenate([array_low, array, array_high], axis=-1)
    raise ValueError(f"unknown pad mode: {mode!r}")


def power_law_basis(x, nu):
    """The table ``x^{nu_m}``, ``nu_m = nu + i eta_full``, shape ``(n, n)``."""
    nu_m = nu + 1j * LogGrid(x, 0).eta_full
    return x[None, :] ** nu_m[:, None]


def power_law_decomposition(fx, x, nu):
    """Terms ``c_m x^{nu_m}`` (rows) of ``fx = Re sum_m c_m x^{nu_m}`` on the unpadded grid."""
    c_half = jnp.fft.rfft(fx * x ** (-nu), norm='forward')
    eta = LogGrid(x, 0).eta_full
    c_m = x[0] ** (-1j * eta) * jnp.concatenate([c_half[::-1][:-1].conj(), c_half[:-1]], axis=0)
    return c_m[:, None] * power_law_basis(x, nu)


def _mellin_g(ell, z, eps=1e-15):
    """``g_ell(z) = sqrt(pi) 2^{z-2} Gamma((ell+z)/2) / Gamma((3+ell-z)/2)``."""
    z = np.asarray(z, dtype=np.complex128)
    # nudge off the real axis, where the ratio has poles
    z = np.where(np.abs(z.imag) < eps, z + 1j * eps, z)
    g_l = np.sqrt(np.pi) * np.exp(
        np.log(2.0) * (z - 2) + loggamma(0.5 * (ell + z)) - loggamma(0.5 * (3 + ell - z))
    )
    return g_l


def hankel_setup(ell, nu, grid, lnxy=None):
    """``(lnxy, y, u_m)`` of the order-``ell`` transform from ``grid`` with bias ``nu``.

    ``u_m = e^{-i eta lnxy} g_ell(nu + i eta)``, ``y = grid.conjugate(lnxy)``; ``lnxy``
    defaults to the low-ringing phase ``dln arg(g_ell(nu + i pi / dln)) / pi``.
    """
    dln = grid.dln
    if lnxy is None:
        lnxy = dln * jnp.angle(_mellin_g(ell, nu + 1j * jnp.pi / dln)) / jnp.pi
    eta = grid.eta_half
    u_m = jnp.exp(lnxy * (-1j * eta)) * _mellin_g(ell, nu + 1j * eta)
    return lnxy, grid.conjugate(lnxy), u_m


def hankel(nu, fx_padded, x, y, u_m, n_pad, crop=True, y_pow=None):
    """One FFTlog pass: ``irfft(conj(rfft(fx_padded x^{-nu}) u_m)) y^{-nu}``.

    Leading batch axes broadcast (one source against many kernels uses one rfft).
    ``y_pow`` replaces ``y^{-nu}`` (then ``y`` may be None); ``crop=False`` keeps
    the padded output for the next transform of a chain.
    """
    c_half = jnp.fft.rfft(fx_padded * x ** (-nu))
    res = jnp.fft.irfft(jnp.conj(c_half * u_m))
    res = res * (y ** (-nu) if y_pow is None else y_pow)
    if not crop:
        return res
    n_out = res.shape[-1]
    return res[..., n_pad:n_out - n_pad]


class ForwardTransform(NamedTuple):
    """Kernels of the forward transforms ``H_l^n[f](q) = int dk k^{n+2} f(k) j_l(kq) / (2 pi^2)``.

    Built by :func:`forward_transform`, one row per (l, n).  The input is ``f / (2 pi^2)`` on the
    padded k grid with its padding and end weights already applied.  ``nu_in`` is the bias of its
    rfft, and the kernel of a row has the bias ``nu_in + n + 3`` and the low-ringing phase of
    ``(l, FFTlogSettings.nu)``, so row (l, n) lands on ``HankelGrids.q_padded[l]``.
    """
    nu_in: object                   # float (one rfft shared by the rows) or (n_rows, 1) array (one per row)
    u_m: jnp.ndarray                # (n_rows, n // 2 + 1) kernels
    y_pow: jnp.ndarray              # (n_rows, n) q^{-(nu_in + n + 3)}
    q: jnp.ndarray                  # (n_rows, n) output grids

    def row(self, i):
        """The transform of row ``i`` alone (arrays without the row axis)."""
        nu_in = self.nu_in if np.ndim(self.nu_in) == 0 else self.nu_in[i]
        return ForwardTransform(nu_in, self.u_m[i], self.y_pow[i], self.q[i])

    def apply(self, fx_padded, grid):
        """The rows of ``fx_padded`` on ``grid`` (leading batch axes broadcast as in :func:`hankel`), uncropped."""
        return hankel(self.nu_in, fx_padded, grid.x, None, self.u_m, grid.n_pad,
                      crop=False, y_pow=self.y_pow)


def forward_transform(grid, nu, ells, ns, nu_in=None, nu_kernel=None):
    """:class:`ForwardTransform` of the rows ``(ells[i], ns[i])`` on the padded grid ``grid``.

    ``nu`` is the ordinary bias whose low-ringing phase fixes the output grids.  Exactly one of
    ``nu_in`` and ``nu_kernel`` is given.  ``nu_in`` (EPT, ``get_xi_ells``) is one input bias for
    all rows, with kernels at ``nu_in + n + 3``.  ``nu_kernel`` (LPT) is one kernel bias for all
    rows, with the input bias ``nu_kernel - n - 3`` of each row (one rfft per row).
    """
    if (nu_in is None) == (nu_kernel is None):
        raise ValueError("give exactly one of nu_in and nu_kernel")
    ells = [int(l) for l in np.asarray(ells)]
    ns = [int(n) for n in np.asarray(ns)]
    u_ms, y_pows, qs = [], [], []
    for l, n in zip(ells, ns):
        lnxy, q_grid, _ = hankel_setup(l, nu, grid)
        nu_eff = nu_kernel if nu_in is None else nu_in + n + 3
        u_ms.append(hankel_setup(l, nu_eff, grid, lnxy=lnxy)[2])
        y_pows.append(q_grid.x ** (-nu_eff))
        qs.append(q_grid.x)
    if nu_in is None:
        nu_in = (nu_kernel - (jnp.asarray(ns) + 3).astype(jnp.float64))[:, None]
    return ForwardTransform(nu_in, jnp.stack(u_ms, axis=0), jnp.stack(y_pows, axis=0),
                            jnp.stack(qs, axis=0))


INPUT_SUPPORT = {'zero-pad': 'core', 'power-law': 'full'}


def trapezoid_weights(support, n_core, n_pad, dln):
    """Static weights on the padded grid: 1 inside ``support``, 1/2 on its end nodes, 0 outside.

    ``support``: 'core', 'full', 'lower_and_core' (forward outputs: q -> 0 plateau
    kept, upper band zeroed) or 'core_to_2kmax' (mode-coupling integrals).
    """
    n_pad = int(n_pad)
    n_core = int(n_core)
    idx = np.arange(n_core + 2 * n_pad)
    if support == 'core':
        on = (idx >= n_pad) & (idx < n_pad + n_core)
    elif support == 'full':
        on = idx >= 0
    elif support == 'lower_and_core':
        on = idx < n_pad + n_core
    elif support == 'core_to_2kmax':
        n2 = int(round(np.log(2.0) / float(dln)))
        on = (idx >= n_pad) & (idx < n_pad + n_core + n2)
    else:
        raise ValueError(f"unknown support: {support!r}")
    w = on.astype(np.float64)
    first, last = np.flatnonzero(on)[[0, -1]]
    # 1/2: FFTlog's m = 0 coefficient is the full-weight node sum
    w[first] = w[last] = 0.5
    return jnp.asarray(w)


class HankelGrids:
    """Padded k grid, per-order low-ringing kernels ``u_m[l]`` and q grids, and support weights.

    ``u_m[l]`` maps ``k_padded`` onto ``q_padded[l]`` and back; ``w_input``/``w_xi`` are
    the end weights of the input model and of a forward output, ``mask_xi`` its support.
    """

    def __init__(self, k, settings, pad_mode, lmax):
        self.pad_mode = pad_mode
        self.npad = max(1, int(settings.npad_factor * k.shape[0]))
        self.grid = LogGrid.from_core(k, self.npad)
        self.k_padded = self.grid.x
        setups = [hankel_setup(l, settings.nu, self.grid) for l in range(int(lmax) + 1)]
        self.lnxy = jnp.array([s[0] for s in setups])
        self.q_grids = tuple(s[1] for s in setups)
        self.q_padded = jnp.array([g.x for g in self.q_grids])
        self.u_m = jnp.array([s[2] for s in setups])
        n_core, dln = self.grid.n_core, self.grid.dln
        self.w_input = trapezoid_weights(INPUT_SUPPORT[pad_mode], n_core, self.npad, dln)
        self.w_xi = trapezoid_weights('lower_and_core', n_core, self.npad, dln)
        self.mask_xi = jnp.where(self.w_xi > 0, 1.0, 0.0)

    def pad_input(self, array):
        """Padded, end-weighted copy of a core-grid input array."""
        return pad(array, self.grid, self.pad_mode) * self.w_input
