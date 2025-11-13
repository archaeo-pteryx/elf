import jax
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

def cubic_conv1d_patch(p, t):
    f_m1, f0, f1, f2 = p[0], p[1], p[2], p[3]

    t2 = t * t
    t3 = t2 * t

    # Keys (a = -0.5) coefficients
    a0 = (-0.5 * f_m1) + (1.5 * f0) - (1.5 * f1) + (0.5 * f2)
    a1 = (f_m1)       - (2.5 * f0) + (2.0 * f1) - (0.5 * f2)
    a2 = (-0.5 * f_m1) + (0.5 * f1)
    a3 = f0

    return a0 * t3 + a1 * t2 + a2 * t + a3

def make_bicubic_spline2d(x, y, vals):
    
    x  = jnp.asarray(x)
    y  = jnp.asarray(y)
    vals = jnp.asarray(vals)

    nx, ny = x.shape[0], y.shape[0]
    assert vals.shape[0] == nx and vals.shape[1] == ny
    batch_shape = vals.shape[2:]
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    def interp_single(xq, yq):
        xq_clamped = jnp.clip(xq, x[0], x[-1])
        yq_clamped = jnp.clip(yq, y[0], y[-1])

        sx = (xq_clamped - x[0]) / dx
        sy = (yq_clamped - y[0]) / dy

        ix = jnp.floor(sx).astype(jnp.int32)
        iy = jnp.floor(sy).astype(jnp.int32)

        ix = jnp.clip(ix, 1, nx - 3)
        iy = jnp.clip(iy, 1, ny - 3)

        tx = sx - ix
        ty = sy - iy

        # extract 4x4 batch: shape (4, 4, *batch)
        start = (ix - 1, iy - 1) + (0,) * len(batch_shape)
        size  = (4, 4) + batch_shape
        patch = jax.lax.dynamic_slice(vals, start, size)

        # cubic along x-axis
        def interp_along_x(col4):
            # col4: shape (4, *batch)
            return cubic_conv1d_patch(col4, tx)  # (*batch,)
        
        tmp = jax.vmap(interp_along_x, in_axes=1, out_axes=0)(patch)

        # cubic along y-axis
        val = cubic_conv1d_patch(tmp, ty)       # (*batch,)
        return val

    def interp(xq, yq):
        xq = jnp.asarray(xq)
        yq = jnp.asarray(yq)

        xq_b, yq_b = jnp.broadcast_arrays(xq, yq)
        flat_x = xq_b.ravel()
        flat_y = yq_b.ravel()

        vals_flat = jax.vmap(interp_single)(flat_x, flat_y)  # (N, *batch)
        out = vals_flat.reshape(xq_b.shape + batch_shape)
        return out

    return interp
