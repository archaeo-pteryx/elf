"""The cubic interpolation of the package.

Cubic Hermite interpolation on strictly increasing nodes: :func:`slopes` once per
array, :func:`hermite` per query (arrays evaluated many times keep their slopes);
:func:`interp1d` is both steps.  :func:`hermite_weights` and :func:`hermite_weight_derivatives`
are the four basis weights of a query and their derivatives, shared with the kernel of the
LPT tangent rule (``lpt_rule.point_kernel``).
"""

import jax
import jax.numpy as jnp

# EPT and LPT get_pk_ells on 1 and 8 cores take 1.05-2.27x (scan_unrolled) and 2.28-5.10x (compare_all) scan's time
SEARCHSORTED_METHOD = 'scan'


def _interval_index(x, xq):
    """The interval i of each query, ``clip(searchsorted(x, xq, side='right') - 1, 0, n - 2)`` with ``n = len(x)``,
    so that x[i] <= xq < x[i + 1] inside the nodes and the first or last interval applies outside them."""
    return jnp.clip(jnp.searchsorted(x, xq, side='right', method=SEARCHSORTED_METHOD) - 1, 0, x.shape[0] - 2)


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


def hermite(xq, x, y, m, axis=-1):
    """Cubic Hermite interpolation of ``y`` (slopes ``m``) along ``axis`` at ``xq``.

    End cubics are continued outside the grid; ``axis`` is replaced by the shape of ``xq``.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    m = jnp.asarray(m)
    xq = jnp.asarray(xq)
    axis = axis % y.ndim

    idx = _interval_index(x, xq)
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
