import jax.numpy as jnp

def interp1d(xq, x, y):
    h, m = _cubic_spline_precompute(x, y)
    yq = _cubic_spline_eval(xq, x, y, h, m)
    return yq

def _cubic_spline_precompute(x, y):
    x = jnp.array(x)
    y = jnp.array(y)
    h = x[1:] - x[:-1]              # (n-1,)
    slopes = (y[1:] - y[:-1]) / h   # (n-1,)
    
    # endpoints
    m0 = slopes[0]
    mn = slopes[-1]

    # otherwise
    h_prev = h[:-1]
    h_next = h[1:]
    s_prev = slopes[:-1]
    s_next = slopes[1:]
    m_mid = (h_prev * s_next + h_next * s_prev) / (h_prev + h_next)

    m = jnp.concatenate([jnp.array([m0]), m_mid, jnp.array([mn])])
    return h, m

def _cubic_spline_eval(xq, x, y, h, m):
    
    xq = jnp.array(xq)

    idx = jnp.searchsorted(x, xq, side="right") - 1
    idx = jnp.clip(idx, 0, x.shape[0] - 2)

    x0 = x[idx]
    h_i = h[idx]
    y0 = y[idx]
    y1 = y[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]

    t = (xq - x0) / h_i
    t2 = t * t
    t3 = t2 * t

    h00 =  2.0 * t3 - 3.0 * t2 + 1.0
    h10 =        t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 =        t3 -        t2

    return h00 * y0 + h10 * h_i * m0 + h01 * y1 + h11 * h_i * m1
