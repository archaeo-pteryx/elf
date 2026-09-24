"""FFTlog / Mellin primitives shared by the EPT and LPT backends.

Conventions
-----------
Every logarithmic-grid transform in this package is built on one primitive: the
discrete Mellin coefficients of a function tabulated on a geometric grid
``x_i = x_0 e^{i dln}``,

    c_m = rfft( f(x) * x^{-nu} )                           (``mellin_coefficients``)

with ``nu`` the FFTlog bias.  The result is a *half* spectrum: only the
non-negative rfft frequencies are stored,

    eta_half[m] = 2 pi rfftfreq(n, d=1) / dln,   m = 0 ... n//2   (``LogGrid.eta_half``)

Two different discrete models are built on that single primitive.  They are
deliberately kept apart: they share the primitive, not the coefficient arrays.

* **Power-law decomposition (PLD) / matrix backend** (``power_law_decomposition``).
  Works on the *unpadded* grid with ``norm='forward'``, and re-expands the half
  spectrum into the full, symmetrically ordered set of complex frequencies

      eta_full[m] = 2 pi / (n dln) * (m - n//2),  m = 0 ... n-1   (``LogGrid.eta_full``)

  Because the rfft measures phases relative to the first grid node while the
  power-law expansion uses ``x`` itself as the variable, the re-expansion also
  carries the phase origin ``x[0]``::

      c_m = x_0^{-i eta_m} * [ conj(c_half[::-1][:-1]), c_half[:-1] ]
      nu_m = nu + i eta_m

  so that ``f(x) = Re sum_m c_m x^{nu_m}``.  The analytic one-loop kernels in
  ``pt_matrix`` are contracted with the terms ``c_m x^{nu_m}``.

* **Hankel / FFTlog backend** (``hankel``).  Works on a *padded* grid
  (``n_pad > 0``), keeps the half spectrum as it comes out of the rfft, and
  multiplies it by the Mellin transform of the spherical Bessel function
  (``mellin_g``) evaluated on ``eta_half``.  The output grid is fixed by the
  low-ringing phase (``low_ringing_phase``, ``output_grid``).

The two models use different transform lengths (padded vs unpadded), different
frequency layouts (half vs full, symmetric) and different normalisations
(``None`` vs ``'forward'``), so their coefficients are *not* interchangeable.

Who uses what
-------------
* ``ept.EPT`` (``method='matrix'`` and the P13 blocks of ``'hybrid'``):
  ``power_law_basis``, ``power_law_decomposition``, ``LogGrid.eta_full``.
* ``ept.EPT._set_hankel`` and ``lpt.LPT._set_hankel``: ``LogGrid.from_core``
  (via ``extend_log_grid``), ``low_ringing_phase``, ``output_grid``,
  ``hankel_kernel``, ``trapezoid_weights``.
* EPT/LPT transform methods: ``pad`` (once, on the input ``P(k)``), then
  ``hankel`` for every transform of the chain; ``grid_moment`` for the zero-lag
  constants of the padded input.

Grid arguments
--------------
The construction-time Hankel helpers (``low_ringing_phase``, ``output_grid``,
``hankel_kernel``) accept either a bare grid array or a :class:`LogGrid`.  A
backend's ``_set_hankel`` builds one ``LogGrid`` and passes it to all of them so
that ``dln`` and ``eta_half`` are computed once instead of once per kernel.

Padding and the padded band
---------------------------
Only the input ``P(k)`` is padded, once, with :func:`pad`.  Every later
transform of a chain (``P(k) -> xi(q) -> Q(k) -> D(q)`` in the LPT backend,
``P(k) -> xi(q) -> P22/P13(k)`` in the EPT backend) takes the *padded* output
of the previous one as its input, restricted to the physical support of the
quantity it represents by a static weight array from :func:`trapezoid_weights`:

(i)   the lower part of the padded band of a forward (``k -> q``) output,
      ``q < q_min``, is the physical ``q -> 0`` plateau of ``xi_l^n`` and is
      kept;
(ii)  the upper part of the padded band of a forward output, ``q > q_max``, is
      the rounding floor of an already-decayed transform and is zeroed
      (``'lower_and_core'``);
(iii) the lower part of the padded band of a backward (``q -> k``) output,
      ``k < k_min``, is Nyquist ringing amplified by ``k^{-nu}`` and is zeroed,
      while the UV side is kept out to the physical support of the quantity:
      ``[k_min, 2 k_max]`` for a mode-coupling integral such as ``Q``
      (``'core_to_2kmax'``) and the core ``[k_min, k_max]`` for anything
      carrying an explicit ``P(k)`` factor such as ``R`` (``'core'``).

``LogGrid.crop`` and ``hankel(crop=True)`` both slice
``[..., n_pad : n - n_pad]`` rather than ``[..., n_pad : -n_pad]`` so that
``n_pad = 0`` is the identity.  The weights are built from static indices only,
so they are compile-time constants and never depend on the data being
transformed.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.special import loggamma

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FFTlogSettings:
    """Numerical constants of the FFTlog pipelines of both backends.

    One instance is passed to ``EPT``/``LPT`` as ``fftlog_settings`` (``None``
    means ``FFTlogSettings()``).  The defaults are the validated values; they are
    collected here so that they are visible and can be varied in convergence
    studies, not because they are expected to be tuned per analysis.  A field
    that a backend does not use is ignored by it.
    """

    nu: float = 1.1
    """FFTlog bias of the ordinary Hankel transforms (``j_l`` kernels) in both
    backends.  It must lie inside the convergence strip of every transform it is
    used for; moving it changes the aliasing/ringing floor of all loop terms at
    the ~1e-6 level and, outside the strip, breaks them."""

    nu_residual: float = -1.7
    """EPT: bias of the backward transform with the residual kernel
    ``j_0(x) - 1`` that gives the ``m = 0`` P22 blocks ``B(k) - B(0)`` at low k.
    That kernel has its own convergence strip; the value sets the low-k accuracy
    of those blocks."""

    nu_pld: float = -0.3
    """EPT: bias of the shared forward rfft of the input ``P(k)`` from which all
    ``n != 0`` rows ``xi_l^n`` are obtained (the per-row output bias is
    ``nu_pld + n + 3``).  Changing it moves the forward transforms' error floor."""

    nu_pld_n0: float = -1.5
    """EPT: the same for the ``n = 0`` rows, whose convergence strip differs."""

    nu_matrix_matter: float = -0.3
    """EPT ``method='matrix'``/``'hybrid'``: bias of the power-law decomposition
    used for the matter P22 and P13 blocks.  The precomputed ``pt_matrix`` kernels
    are evaluated at this bias, so it must stay inside their strip."""

    nu_matrix_p22_bias: float = -1.0
    """EPT ``method='matrix'``: bias of the decomposition for the P22 blocks with
    bias operators.  It is chosen above -3/2 so that the decomposition evaluates
    ``I(k) - I(0)`` by analytic continuation; the ``b2^2`` constant is added back
    separately."""

    nu_matrix_p13_bias: float = -1.6
    """EPT ``method='matrix'``/``'hybrid'``: bias of the decomposition for the P13
    blocks with bias operators (inside their convergence strip)."""

    npad_factor: float = 0.5
    """Width of the padded band on each side of the grid, as a fraction of
    ``nfft`` (``n_pad = max(1, int(npad_factor * nfft))``).  A narrower band lets
    the periodic images of the transforms leak into the core; a wider one costs
    time with no measurable gain at the default."""

    blend_rstar: float = 0.03
    """EPT: scale of the Gaussian weight ``w = exp(-(r / blend_rstar)^2)`` that
    blends the residual-kernel (low k) and the direct (high k) estimates of the
    ``m = 0`` P22 blocks, with ``r = |B(k) - B(0)| / |B(0)|``.  A larger value
    carries the residual kernel's error, which grows like ``k^2``, further into
    high k; a smaller one uses the direct estimate where its absolute error is
    not yet negligible against ``B(k) - B(0)``."""

    q_star_factor: float = 0.1
    """LPT: the zero-lag constants of the loop correlators are read off the
    ``q -> 0`` plateau of the padded transform at the node closest to
    ``q_star = q_min * q_star_factor``.  Too close to the padded corner the
    reading picks up the ``q^{-nu}``-amplified FFT floor; too close to ``q_min``
    the ``q_star^6`` Taylor remainder grows."""


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
# Padding of the input function
# ---------------------------------------------------------------------------

def get_log_extrap(x, y, x_low, x_high):
    """Continue a tabulated function past both ends of a log grid by a power law.

    This is the single power-law extrapolation rule of the package: both
    :func:`pad` (power-law padding of the FFTlog input) and
    ``utils.get_pk``/``ir_resum.get_pk_nw`` (extrapolation of the input
    ``P(k)`` table before interpolation) call it.

    ``x`` is the input grid (ascending); ``y`` may carry leading batch axes
    and is extrapolated along its last axis.  The low-end log-slope is set by
    the pair ``(x[0], x[1])`` / ``(y[..., 0], y[..., 1])``; the high-end slope
    by ``(x[-2], x[-1])`` / ``(y[..., -2], y[..., -1])``.  ``x_low``
    (ascending, below ``x[0]``) and ``x_high`` (ascending, above ``x[-1]``)
    are the requested extrapolation nodes.  The returned ``(y_low, y_high)``
    are

        y(x_new) = y_end (x_new / x_end)^s,   s = ln(y_hi / y_lo) / ln(x_hi / x_lo),

    evaluated as ``y_end * exp(log(y_hi / y_lo) * steps)`` with
    ``steps = ln(x_new / x_end) / ln(x_hi / x_lo)`` (negative below the grid,
    positive above it).

    There is a single continuation rule, applied independently on each side:
    the log-slope is used only when both end values of that side's pair are
    positive (so the slope is real); otherwise -- a non-positive end value
    ``y_end <= 0``, or a positive ``y_end`` whose neighbour makes the ratio
    ``y_hi / y_lo`` zero, negative or underflow -- the continuation is zero.

    AD safety: the ratio is formed with a *double* ``jnp.where`` -- the
    denominator is replaced by 1.0 where it is not positive *before* dividing,
    and the result is discarded afterwards -- so a zero or subnormal endpoint
    (e.g. the IR-damped copy ``P e^{-(k/k_IR)^2}``, which underflows to exactly
    zero long before ``k_max``) gives a finite reverse-mode gradient instead of
    the ``inf``/``nan`` a single ``where`` would leave from the quotient rule.
    """
    dlnx_low = jnp.log(x[1] / x[0])
    dlnx_high = jnp.log(x[-1] / x[-2])

    y_dtype = y.dtype
    steps_low = (jnp.log(x_low / x[0]) / dlnx_low).astype(y_dtype)
    steps_high = (jnp.log(x_high / x[-1]) / dlnx_high).astype(y_dtype)

    expand = (None,) * (y.ndim - 1)
    y_low = _power_law_continuation(
        y[..., 0:1], y[..., 1:2], steps_low[expand], 'low')
    y_high = _power_law_continuation(
        y[..., -2:-1], y[..., -1:], steps_high[expand], 'high')
    return y_low, y_high


def _power_law_continuation(y_lo, y_hi, steps, side):
    """Shared rule of :func:`get_log_extrap` -- see its docstring for the maths.

    ``(y_lo, y_hi)`` are the two end values of one side, in ascending-``x``
    order; ``side`` (``'low'`` or ``'high'``) says which of them is the end
    node ``x_end``.  The arguments broadcast, so ``y_lo``/``y_hi`` may carry
    leading batch axes and a trailing length-1 axis against a ``steps``
    vector.
    """
    if side == 'low':
        y_end = y_lo
    elif side == 'high':
        y_end = y_hi
    else:
        raise ValueError(f"side must be 'low' or 'high', got {side!r}")

    den_ok = y_lo > 0.0
    # inner where: never divide by a non-positive (possibly subnormal) node
    raw = y_hi / jnp.where(den_ok, y_lo, 1.0)
    ok = den_ok & (raw > 0.0)
    # outer where: discard the replaced branch, value and gradient alike
    ratio = jnp.where(ok, raw, 1.0)
    amp = jnp.where(ok, y_end, 0.0)
    return amp * jnp.exp(jnp.log(ratio) * steps)


def pad(array_y, n_pad, mode, x=None):
    """Extend the last axis of ``array_y`` by ``n_pad`` nodes on each side.

    ``'zero-pad'`` pads with zeros (the band-limited input model);
    ``'power-law'`` continues each end with the log-slope of its two outermost
    nodes, evaluated at the new nodes by :func:`get_log_extrap` (an end whose
    two nodes are not both positive is continued by zero, the only
    continuation that is meaningful for a spectrum that has already reached
    zero at the edge of the grid).  ``'power-law'`` requires ``x``, the
    log-uniform core grid of ``array_y``'s last axis: the padded nodes sit at
    ``x[0] (x[1]/x[0])^{-n_pad..-1}`` and ``x[-1] (x[-1]/x[-2])^{1..n_pad}``,
    the same construction :func:`extend_log_grid` uses for the padded grid
    itself.
    """
    if mode == 'zero-pad':
        pad_width = [(0, 0)] * (array_y.ndim - 1) + [(n_pad, n_pad)]
        return jnp.pad(array_y, pad_width)
    if mode == 'power-law':
        if x is None:
            raise ValueError("mode='power-law' requires the core grid x")
        ratio_low = x[1] / x[0]
        ratio_high = x[-1] / x[-2]
        x_low = x[0] * (ratio_low ** jnp.arange(-n_pad, 0))
        x_high = x[-1] * (ratio_high ** jnp.arange(1, n_pad + 1))
        array_low, array_high = get_log_extrap(x, array_y, x_low, x_high)
        return jnp.concatenate([array_low, array_y, array_high], axis=-1)
    raise ValueError(f"unknown pad mode: {mode!r}")


# ---------------------------------------------------------------------------
# The single rfft definition
# ---------------------------------------------------------------------------

def mellin_coefficients(fx, x, nu, norm=None):
    """Half-spectrum Mellin coefficients ``rfft(fx * x^{-nu})``.

    This is the only place in the package where the forward rfft of a
    log-gridded function is taken.  ``fx`` may carry leading batch axes; ``nu``
    may be an array broadcastable against ``x`` (e.g. ``nu[:, None]`` for a
    batch of biases sharing one source).  ``norm`` is passed through to
    ``jnp.fft.rfft``: ``None`` for the Hankel model, ``'forward'`` for the
    PLD/matrix model.
    """
    return jnp.fft.rfft(fx * x ** (-nu), norm=norm)


# ---------------------------------------------------------------------------
# Power-law decomposition (PLD / matrix model)
# ---------------------------------------------------------------------------

def power_law_basis(x, nu):
    """The table ``x^{nu_m}``, ``nu_m = nu + i eta_full``, shape ``(n, n)``.

    Building it is the expensive part of :func:`power_law_decomposition`; a
    caller that decomposes many spectra with the same ``(x, nu)`` precomputes it
    once and passes it as ``xpow``.
    """
    nu_m = nu + 1j * LogGrid(x, 0).eta_full
    return x[None, :] ** nu_m[:, None]


def power_law_decomposition(fx, x, nu, xpow=None):
    """Power-law terms ``c_m x^{nu_m}`` of ``fx`` on the unpadded grid ``x``.

    Returns the ``(n, n)`` complex array whose row ``m`` is ``c_m x^{nu_m}``, so
    that ``fx = Re sum_m c_m x^{nu_m}``; the analytic ``pt_matrix`` kernels are
    contracted with it.  ``c_m`` are the full, symmetric PLD coefficients with
    the ``x[0]`` phase origin (see the module docstring).  ``xpow`` may hold the
    precomputed :func:`power_law_basis` ``(x, nu)``; otherwise it is built here.
    """
    c_half = mellin_coefficients(fx, x, nu, norm='forward')
    eta = LogGrid(x, 0).eta_full
    c_m = x[0] ** (-1j * eta) * jnp.concatenate([c_half[::-1][:-1].conj(), c_half[:-1]], axis=0)
    if xpow is None:
        xpow = power_law_basis(x, nu)
    return c_m[:, None] * xpow


# ---------------------------------------------------------------------------
# Hankel / FFTlog model
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
    """
    eta = _as_grid(x).eta_half
    return jnp.exp(lnxy * (-1j * eta)) * mellin_g(ell, nu + 1j * eta)


def hankel(nu, fx_padded, x, y, u_m, n_pad, crop=True, y_pow=None):
    """One FFTlog pass on an input that already lives on the padded grid ``x``.

    ``irfft(conj(rfft(fx_padded x^{-nu}) u_m)) y^{-nu}``: the Mellin
    coefficients of ``fx_padded`` with bias ``nu``, the kernel ``u_m`` from
    :func:`hankel_kernel`, and the inverse transform onto the padded output grid
    ``y``.  ``fx_padded`` is either the padded input ``P(k)`` (padded once by the
    caller with :func:`pad`) or the padded output of the previous transform of a
    chain.  ``fx_padded``, ``nu`` and ``u_m`` may carry leading batch axes that
    broadcast against each other; a single ``fx_padded`` row against a batch of
    kernels is transformed with one rfft.

    ``y_pow`` replaces ``y ** (-nu)`` where the output bias differs from the
    Mellin bias (the forward transforms of the input ``P(k)``, whose ``k^{n+3}``
    factor is absorbed into the Mellin bias) or where a precomputed factor is
    reused.  When ``y_pow`` is given, ``y`` is unused and may be ``None``.

    With ``crop=True`` the padding is removed from the last axis of the output;
    with ``crop=False`` the whole padded output is returned, which is what the
    next transform of a chain takes as its input.
    """
    c_half = mellin_coefficients(fx_padded, x, nu)
    res = jnp.fft.irfft(jnp.conj(c_half * u_m))
    res = res * (y ** (-nu) if y_pow is None else y_pow)
    if not crop:
        return res
    n_out = res.shape[-1]
    return res[..., n_pad:n_out - n_pad]


# ---------------------------------------------------------------------------
# Supports, end weights and zero-lag moments
# ---------------------------------------------------------------------------

# Support of the input P(k) under each pad mode: the band-limited model
# ('zero-pad') lives on the core, the continued one ('power-law') on the whole
# padded grid.
INPUT_SUPPORT = {'zero-pad': 'core', 'power-law': 'full'}


def trapezoid_weights(support, n_core, n_pad, dln):
    """Trapezoidal quadrature weights on the padded grid: 1 inside the
    support, 1/2 on its two end nodes, 0 outside.

    Returns a float array of length ``n_core + 2 n_pad`` (the padded transform
    length) that is 0 outside the support, 1 inside it and 1/2 on the first and
    the last node of the support.  Multiplying a padded array by it both
    restricts the array to its support and turns the FFTlog's ``m = 0``
    coefficient into a trapezoid sum over that support.  ``weights > 0`` is the
    plain support mask.  The array is built in numpy from grid indices alone, so
    it is a compile-time constant, transparent to ``jit`` and to autodiff.

    ``support``:

    ``'core'``
        ``[n_pad, n_pad + n_core)`` -- the physical grid.  The input ``P(k)`` of
        the band-limited ('zero-pad') model, and a backward output that carries
        an explicit ``P(k)`` factor (rule (iii), ``R``-type rows).  With
        ``n_pad = 0`` this is the trapezoid weight of an unpadded core array.
    ``'full'``
        the whole padded grid -- the input of the continued ('power-law')
        model, whose half weights sit on the padded corners.
    ``'lower_and_core'``
        ``[0, n_pad + n_core)`` -- a forward (``k -> q``) output: the lower part
        of the padded band ``q < q_min`` is the physical ``q -> 0`` plateau and
        is kept (rule (i)), the upper part ``q > q_max`` is zeroed (rule (ii)).
    ``'core_to_2kmax'``
        ``[n_pad, n_pad + n_core + n2)`` with ``n2 = round(log 2 / dln)`` -- the
        core plus the octave above ``k_max`` over which a mode-coupling integral
        of two modes below ``k_max`` still has support (rule (iii), ``Q`` rows).

    ``dln`` is the logarithmic spacing of the grid and may be a JAX scalar; it
    is read once, at construction time, to size the ``2 k_max`` extension.

    Why the half weights are needed
    -------------------------------
    An FFTlog returns the exact Hankel transform of the trigonometric
    interpolant of the samples on the logarithmic grid.  Its ``m = 0`` Fourier
    coefficient is ``dln sum_j a_j`` -- the *rectangle* rule, every node at full
    weight.  If the source stops at an interior node ``J`` (``a_j = 0`` for
    ``j > J``), the rectangle rule integrates half a log-cell past the cut, and
    the Euler-Maclaurin endpoint term ``(dln/2) a_J = O(dln)`` survives in the
    zero-lag constant and in every angular moment.  Halving the samples at the
    ends of the support turns that sum into the *trapezoid* rule over the same
    support, whose error is ``O(dln^2)``.  Verified (2026-09-21/23; see
    ``notes/lpt_source_accuracy_ja.pdf`` section "切断模型" and
    ``notes/ept_ja.pdf`` section "入力 padding の zero-pad 化" in the
    ``ps_1loop_jax`` repo): with the end weights the LPT six-row conditional
    error drops from ~1e-3 to <= 3e-6 and the zero-padded EPT input matches the
    power-law-continued one, whereas an unweighted zero-pad is 3 to 10 times
    worse than either.
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
    w[first] = w[last] = 0.5
    return jnp.asarray(w)


def grid_moment(fx_padded, x_padded, dln, power):
    """Zero-lag moment ``sum_j dln x_j^power f_j`` of a padded FFTlog input.

    This is the full-weight node sum that equals the ``m = 0`` Fourier
    coefficient of the trigonometric interpolant the FFTlog actually
    transforms, i.e. ``int dln x  x^power f(x)`` evaluated by the same discrete
    rule as every other moment of that input.  ``fx_padded`` is expected to
    carry the trapezoid end weights of its support already (see
    :func:`trapezoid_weights`), in which case the sum is the trapezoid rule
    over that support rather than the rectangle rule.

    The sum runs over the last axis, so ``fx_padded`` may carry leading batch
    axes; ``x_padded`` is the padded grid the input lives on.
    """
    return jnp.sum(dln * x_padded ** power * fx_padded, axis=-1)
