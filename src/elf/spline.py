"""The cubic interpolation of the package.

Cubic Hermite interpolation on strictly increasing nodes: :func:`slopes` once per
array, :func:`hermite` per query (arrays evaluated many times keep their slopes);
:func:`interp1d` is both steps.
"""

import jax
import jax.numpy as jnp


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

    idx = jnp.searchsorted(x, xq, side="right") - 1
    idx = jnp.clip(idx, 0, x.shape[0] - 2)
    x0 = x[idx]
    h_i = (x[1:] - x[:-1])[idx]

    t = (xq - x0) / h_i
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    tail = (1,) * (y.ndim - axis - 1)

    def c(a):
        return jnp.reshape(a, jnp.shape(a) + tail)

    # mode='clip': idx is already in range, so no out-of-bounds fill is needed
    y0 = jnp.take(y, idx, axis=axis, mode='clip')
    y1 = jnp.take(y, idx + 1, axis=axis, mode='clip')
    m0 = jnp.take(m, idx, axis=axis, mode='clip')
    m1 = jnp.take(m, idx + 1, axis=axis, mode='clip')
    return c(h00) * y0 + c(h10 * h_i) * m0 + c(h01) * y1 + c(h11 * h_i) * m1


def interp1d(xq, x, y, axis=-1):
    """Cubic Hermite interpolation of ``y`` along ``axis`` at ``xq``: :func:`slopes`, then :func:`hermite`."""
    return hermite(xq, x, y, slopes(x, y, axis), axis)
