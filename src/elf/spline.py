"""The cubic interpolation of the package.

Cubic Hermite interpolation on strictly increasing nodes: :func:`slopes` once per
array, :func:`hermite` per query (arrays evaluated many times keep their slopes);
:func:`interp1d` is both steps.  :func:`hermite_weights` and :func:`hermite_weight_derivatives`
are the four basis weights of a query and their derivatives, shared with the kernel of the
LPT final transform (``lpt.LPT._point_kernel``).  Between fixed grids the same interpolation is a
fixed linear map, built once in numpy by :func:`stencil` (four nodes per query) or :func:`resampling`
(a band of offsets for grids of the same spacing) and applied by :func:`resample`.
"""

from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

# EPT and LPT get_pk_ells on 1 and 8 cores take 1.05-2.27x (scan_unrolled) and 2.28-5.10x (compare_all) scan's time
SEARCHSORTED_METHOD = 'scan'


def _interval_index(x, xq, ascending=False):
    """The interval i of each query, ``clip(searchsorted(x, xq, side='right') - 1, 0, n - 2)`` with ``n = len(x)``,
    so that x[i] <= xq < x[i + 1] inside the nodes and the first or last interval applies outside them.  With
    ``ascending`` the queries are 1-D and in ascending order and the nodes are searched among them, which is cheaper
    for many more queries than nodes: the nodes at or below query j are those whose first query at or above them has
    an index <= j."""
    if ascending:
        first = jnp.searchsorted(xq, x, side='left', method=SEARCHSORTED_METHOD)
        below = jnp.cumsum(jnp.zeros(xq.shape[0] + 1, jnp.int32).at[first].add(1))[:-1]
    else:
        below = jnp.searchsorted(x, xq, side='right', method=SEARCHSORTED_METHOD)
    return jnp.clip(below - 1, 0, x.shape[0] - 2)


def hermite_weights(t, h_i):
    """The weights of (y0, m0, y1, m1) at ``t = (xq - x[i]) / h_i`` in the interval i of width ``h_i``.

    The tuple (h00, h10 h_i, h01, h11 h_i) of the cubic Hermite basis h00 = 2 t^3 - 3 t^2 + 1,
    h10 = t^3 - 2 t^2 + t, h01 = -2 t^3 + 3 t^2 and h11 = t^3 - t^2, the slope weights times h_i.
    """
    t2 = t * t
    t3 = t2 * t
    return 2.0 * t3 - 3.0 * t2 + 1.0, (t3 - 2.0 * t2 + t) * h_i, -2.0 * t3 + 3.0 * t2, (t3 - t2) * h_i


def hermite_weight_derivatives(t, h_i):
    """The derivatives of :func:`hermite_weights` with respect to the query xq, as dt/dxq = 1 / h_i.

    The tuple of (6 t^2 - 6 t, (3 t^2 - 4 t + 1) h_i, -6 t^2 + 6 t, (3 t^2 - 2 t) h_i), the t derivatives of the
    four weights, each divided by h_i.
    """
    t2 = t * t
    return tuple(w / h_i for w in (6.0 * t2 - 6.0 * t, (3.0 * t2 - 4.0 * t + 1.0) * h_i, -6.0 * t2 + 6.0 * t,
                                   (3.0 * t2 - 2.0 * t) * h_i))


def slopes(x, y, axis=-1):
    """Node slopes ``m_j = (h_{j-1} s_j + h_j s_{j-1}) / (h_{j-1} + h_j)``; end secants at the ends."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    axis = axis % y.ndim
    n = x.shape[0]

    def part(a, lo, hi):
        return jax.lax.slice_in_dim(a, lo, hi, axis=axis)

    h = jnp.reshape(x[1:] - x[:-1], (n - 1,) + (1,) * (y.ndim - axis - 1))
    s = (part(y, 1, n) - part(y, 0, n - 1)) / h
    h_prev, h_next = h[:-1], h[1:]
    s_prev, s_next = part(s, 0, n - 2), part(s, 1, n - 1)
    m_mid = (h_prev * s_next + h_next * s_prev) / (h_prev + h_next)
    return jnp.concatenate([part(s, 0, 1), m_mid, part(s, n - 2, n - 1)], axis=axis)


def hermite(xq, x, y, m, axis=-1, ascending=False):
    """Cubic Hermite interpolation of ``y`` (slopes ``m``) along ``axis`` at ``xq``.

    End cubics are continued outside the grid; ``axis`` is replaced by the shape of ``xq``.
    ``ascending`` declares 1-D queries in ascending order, whose intervals :func:`_interval_index` then finds by
    searching the nodes among the queries (the same intervals).
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    m = jnp.asarray(m)
    xq = jnp.asarray(xq)
    axis = axis % y.ndim

    idx = _interval_index(x, xq, ascending)
    x0 = x[idx]
    h_i = (x[1:] - x[:-1])[idx]

    t = (xq - x0) / h_i
    w00, w10, w01, w11 = hermite_weights(t, h_i)

    tail = (1,) * (y.ndim - axis - 1)

    def c(a):
        return jnp.reshape(a, jnp.shape(a) + tail)

    # mode='clip': idx is already in range, so no out-of-bounds fill is needed
    y0 = jnp.take(y, idx, axis=axis, mode='clip')
    y1 = jnp.take(y, idx + 1, axis=axis, mode='clip')
    m0 = jnp.take(m, idx, axis=axis, mode='clip')
    m1 = jnp.take(m, idx + 1, axis=axis, mode='clip')
    return c(w00) * y0 + c(w10) * m0 + c(w01) * y1 + c(w11) * m1


def interp1d(xq, x, y, axis=-1):
    """Cubic Hermite interpolation of ``y`` along ``axis`` at ``xq``: :func:`slopes`, then :func:`hermite`."""
    return hermite(xq, x, y, slopes(x, y, axis), axis)


def stencil(xq, x):
    """``(index, weight)``, both ``(nq, 4)`` numpy arrays, of the interpolation of :func:`interp1d` between fixed
    grids: ``interp1d(xq, x, y)[j] = sum_d weight[j, d] y[index[j, d]]`` for every ``y``.  The four nodes of query j
    are i - 1, i, i + 1, i + 2 with i its interval (clipped to the nodes at the two ends, where their weight is 0).
    The weights combine the Hermite weights of (y_i, m_i, y_{i+1}, m_{i+1}) with the slopes m of :func:`slopes`,
    each a combination of three node values (two at the ends).  ``x`` ``(n,)`` and ``xq`` ``(nq,)`` are numpy
    arrays (static grids)."""
    x, xq = np.asarray(x, dtype=np.float64), np.asarray(xq, dtype=np.float64)
    n = x.shape[0]
    i = np.clip(np.searchsorted(x, xq, side='right') - 1, 0, n - 2)
    h = np.diff(x)
    t = (xq - x[i]) / h[i]
    w00, w10, w01, w11 = hermite_weights(t, h[i])
    # the slope m_j = a[j] . (y_{j-1}, y_j, y_{j+1}) of slopes(), the end secants at j = 0 and n - 1
    a = np.zeros((n, 3))
    h_prev, h_next = h[:-1], h[1:]
    a[1:-1] = np.stack([-h_next / h_prev, h_next / h_prev - h_prev / h_next, h_prev / h_next],
                       axis=1) / (h_prev + h_next)[:, None]
    a[0, 1:] = -1.0 / h[0], 1.0 / h[0]
    a[-1, :2] = -1.0 / h[-1], 1.0 / h[-1]
    weight = np.zeros((xq.shape[0], 4))                     # of y_{i-1}, y_i, y_{i+1}, y_{i+2}
    weight[:, 1] += w00
    weight[:, 2] += w01
    weight[:, 0:3] += w10[:, None] * a[i]
    weight[:, 1:4] += w11[:, None] * a[i + 1]
    index = np.clip(i[:, None] + np.arange(-1, 3), 0, n - 1)
    return index, weight


class Resampling(NamedTuple):
    """:func:`interp1d` between two fixed grids of the same spacing, as fixed weights (:func:`resampling`)."""
    weights: jnp.ndarray     # (..., nq, width), weights[..., j, d] multiplies the node first + j + d
    first: int               # the node of the first weight of query 0 (outside the nodes: a zero-padded source)
    n: int                   # number of nodes of the source grid


def resampling(xq, x):
    """The :class:`Resampling` from the nodes ``x`` ``(..., n)`` (one grid per leading index) to the queries ``xq``
    ``(nq,)``, numpy arrays.  For grids of the same spacing (FFTlog output grids differ by a shift) the four nodes
    of :func:`stencil` move with the query, so the stencils of all rows fit a band of a few offsets node - query.
    The weights of every row are placed at their offsets in one band, ``first`` being the smallest offset over all
    rows and queries, and the weights of nodes that coincide after the clipping at the ends add.  ``x`` of shape
    ``(n,)`` gives weights ``(nq, width)``, ``(r1, r2, n)`` gives ``(r1, r2, nq, width)``.
    ValueError if the band is wider than 8 (grids of different spacing)."""
    xq, x = np.asarray(xq, dtype=np.float64), np.asarray(x, dtype=np.float64)
    lead, n, nq = x.shape[:-1], x.shape[-1], xq.shape[0]
    stencils = [stencil(xq, row) for row in x.reshape(-1, n)]
    index = np.stack([s[0] for s in stencils])               # (rows, nq, 4)
    weight = np.stack([s[1] for s in stencils])
    offset = index - np.arange(nq)[None, :, None]
    first = int(offset.min())
    width = int(offset.max()) - first + 1
    if width > 8:
        raise ValueError(f'resampling: the stencils span a band of {width} offsets (more than 8), '
                         'the grids do not have the same spacing')
    weights = np.zeros((index.shape[0], nq, width))
    row, query, _ = np.indices(index.shape)
    np.add.at(weights, (row, query, offset - first), weight)
    return Resampling(jnp.asarray(weights.reshape(lead + (nq, width))), first, n)


def resample(plan, y):
    """``interp1d(xq, x, y)`` of the grids of ``plan`` for ``y`` ``(..., n)`` (leading axes broadcast against those
    of ``plan.weights``): the sum over the band of ``weights[..., d]`` times the source shifted by ``first + d``
    (zero-padded where the band leaves the nodes)."""
    nq, width = plan.weights.shape[-2:]
    lo = max(0, -plan.first)
    hi = max(0, plan.first + nq + width - 1 - plan.n)
    ypad = jnp.pad(y, [(0, 0)] * (jnp.ndim(y) - 1) + [(lo, hi)])
    start = plan.first + lo
    out = plan.weights[..., 0] * ypad[..., start:start + nq]
    for d in range(1, width):
        out = out + plan.weights[..., d] * ypad[..., start + d:start + d + nq]
    return out
