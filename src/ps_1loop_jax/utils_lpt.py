import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
from functools import partial

# @jit
# def get_G00(A, B, C, coeff):
#     x = - B
#     y = C**2 / (A**2 + C**2)
    
#     res = 0
#     l = coeff.shape[0] - 1
#     for i in range(l+1):
#         for k in range(l+1):
#             res = res + coeff[k, i] * x**(l + i) * y**(l + i - k)
#     res = res * jnp.exp(x * y)
    
#     return res

@jit
def get_G00(A, B, C, coeff):
    x = -B
    y = C**2 / (A**2 + C**2)

    # l を静的に取りたいなら coeff.shape[0]-1 を使う
    l = coeff.shape[0] - 1
    i = jnp.arange(l + 1)
    k = jnp.arange(l + 1)

    # v[k] = y^(l-k)   （非負の指数だけ：y=0 でも安全）
    v = jnp.power(y[..., None], l - k)

    M = jnp.tensordot(v, coeff, axes=((-1,), (0,)))

    # x^{l+i} と y^{i} をまとめて作る
    xpow = jnp.power(x[..., None], l + i)
    ypow = jnp.power(y[..., None], i)

    res = jnp.sum(M * xpow * ypow, axis=-1) * jnp.exp(x * y)

    return res

# @jit
# def get_dGs_l(A, B, C, coeff):
#     rho2 = A**2 + C**2
#     c2 = A**2 / rho2
#     s2 = C**2 / rho2
#     Bs2 = B * s2

#     G00 = 0
#     dGdA = 0
#     dGdC = 0
#     dG2dA2 = 0
#     d2GdC2 = 0
#     d2GdAdC = 0
#     dG3dA3 = 0
#     d3GdA2dC = 0
#     d4GdA4 = 0
    
#     l = coeff.shape[0] - 1
#     for i in range(l+1):
#         for k in range(l+1):

#             n = l + i - k
#             term = coeff[k, i] * (- B)**(l + i) * s2**(n)
#             term2 = jnp.where(s2 == 0., 0., term / s2)

#             G00 = G00 + term

#             factor = - n + Bs2
#             dGdA = dGdA + factor * term

#             factor = n - Bs2
#             dGdC = dGdC + factor * term2

#             factor = -n + 2 * n * (1 + n) * c2 + Bs2 - 4 * (1 + n) * c2 * Bs2 + 2 * c2 * Bs2**2
#             dG2dA2 = dG2dA2 + factor * term

#             factor = n * (-3 + 2 * (1 + n) * c2) + (3 - 4 * c2 * (1 + n)) * Bs2 + 2 * c2 * Bs2**2
#             d2GdC2 = d2GdC2 + factor * term2

#             factor = n * (-1 + (1 + n) * c2) + (1 - 2 * (1 + n) * c2) * Bs2 + c2 * Bs2**2
#             d2GdAdC = d2GdAdC + factor * term2

#             factor = n * (1 + n) * (-3 + 2 * (2 + n) * c2) - 6 * (1 + n) * (-1 + (2 + n) * c2) * Bs2 \
#                     + 3 * (-1 + 2 * (2 + n) * c2) * Bs2**2 - 2 * c2 * Bs2**3
#             dG3dA3 = dG3dA3 + factor * term

#             factor = n + n * (1 + n) * (-5 + 2 * (2 + n) * c2) * c2 \
#                     - (1 + 2 * (1 + n) * (-5 + 3 * c2 * (2 + n)) * c2) * Bs2 \
#                     + (-5 + 6 * (2 + n) * c2) * c2 * Bs2**2 - 2 * c2**2 * Bs2**3
#             d3GdA2dC = d3GdA2dC + factor * term2
            
#             factor = n * (1 + n) * (3 + 4 * (2 + n) * (-3 + (3 + n) * c2) * c2) \
#                     - 2 * (1 + n) * (3 + 2 * c2 * (2 + n) * (-9 + 4 * c2 * (3 + n))) * Bs2 \
#                     + 3 * (1 + 4 * (2 + n) * (-3 + 2 * (3 + n) * c2) * c2) * Bs2**2 \
#                     - 4 * (-3 + 4 * (3 + n) * c2) * c2 * Bs2**3 + 4 * c2**2 * Bs2**4
#             d4GdA4 = d4GdA4 + factor * term

#     G00 = G00 * jnp.exp(- Bs2)
    
#     dGdA = dGdA * jnp.exp(- Bs2) * (2 * A / rho2)
#     dGdC = dGdC * jnp.exp(- Bs2) * (2 * c2 * jnp.sqrt(s2 / rho2))
    
#     dG2dA2 = dG2dA2 * jnp.exp(- Bs2) * (2 / rho2)
#     d2GdC2 = d2GdC2 * jnp.exp(- Bs2) * (2 * c2 / rho2)
#     d2GdAdC = d2GdAdC * jnp.exp(- Bs2) * (-4 * jnp.sqrt(c2 * s2) / rho2)
    
#     dG3dA3 = dG3dA3 * jnp.exp(- Bs2) * (-4 * jnp.sqrt(c2) / rho2**1.5)
#     d3GdA2dC = d3GdA2dC * jnp.exp(- Bs2) * (4 * jnp.sqrt(s2) / rho2**1.5)

#     d4GdA4 = d4GdA4 * jnp.exp(- Bs2) * (4 / rho2**2)

#     dGs_l = jnp.vstack([G00, dGdA, dGdC, dG2dA2, d2GdC2, d2GdAdC, dG3dA3, d3GdA2dC, d4GdA4])
#     return dGs_l

@jit
def get_dGs_l(A, B, C, coeff):

    l = coeff.shape[0] - 1

    rho2 = A**2 + C**2
    c2   = A**2 / rho2            # (...,)
    s2   = C**2 / rho2            # (...,)
    Bs2  = B * s2                 # (...,)

    # ---- 事前計算（k, i の指数依存を分離）----
    K = jnp.arange(l+1)           # (K,)
    I = jnp.arange(l+1)           # (I,)

    # v_k = s2^(l-k)   （非負の指数のみ）
    v = jnp.power(s2[..., None], l - K)                             # (..., K)

    # w_i = (-B)^(l+i) * s2^i
    w = jnp.power(-B[..., None], l + I) * jnp.power(s2[..., None], I)  # (..., I)

    # term_{k,i} = coeff[k,i] * v_k * w_i
    # 形合わせ: v[..., :, None] * coeff[None,...,:,:] * w[..., None, :]
    term  = v[..., :, None] * coeff[None, ...] * w[..., None, :]    # (..., K, I)

    # s2=0 対策の term2 = term / s2 だけ where で安全化
    inv_s2 = jnp.where(s2 == 0., 0., 1.0 / s2)                      # (...,)
    term2  = term * inv_s2[..., None, None]                         # (..., K, I)

    # n = l + i - k を (K,I) 行列で持つ（バッチには自動ブロードキャスト）
    N = (l + I[None, :] - K[:, None])                               # (K, I)

    # ブロードキャスト用
    N_b   = N[None, ...]                     # (1, K, I)
    Bs2_b = Bs2[..., None, None]             # (..., 1, 1)
    c2_b  = c2[..., None, None]              # (..., 1, 1)

    # ---- 各和を一発で計算（axis=(-2,-1) が k,i の和）----
    G00_sum   = jnp.sum(term, axis=(-2, -1))

    dGdA_sum  = jnp.sum((-N_b + Bs2_b) * term,  axis=(-2, -1))
    dGdC_sum  = jnp.sum(( N_b - Bs2_b) * term2, axis=(-2, -1))

    d2A_sum = jnp.sum(
        (-N_b + 2*N_b*(1+N_b)*c2_b + Bs2_b - 4*(1+N_b)*c2_b*Bs2_b + 2*c2_b*Bs2_b**2) * term,
        axis=(-2, -1)
    )

    d2C_sum = jnp.sum(
        (N_b*(-3 + 2*(1+N_b)*c2_b) + (3 - 4*c2_b*(1+N_b))*Bs2_b + 2*c2_b*Bs2_b**2) * term2,
        axis=(-2, -1)
    )

    dAdC_sum = jnp.sum(
        (N_b*(-1 + (1+N_b)*c2_b) + (1 - 2*(1+N_b)*c2_b)*Bs2_b + c2_b*Bs2_b**2) * term2,
        axis=(-2, -1)
    )

    d3A_sum = jnp.sum(
        ( N_b*(1+N_b)*(-3 + 2*(2+N_b)*c2_b)
        - 6*(1+N_b)*(-1 + (2+N_b)*c2_b)*Bs2_b
        + 3*(-1 + 2*(2+N_b)*c2_b)*Bs2_b**2
        - 2*c2_b*Bs2_b**3) * term,
        axis=(-2, -1)
    )

    d3A2C_sum = jnp.sum(
        ( N_b + N_b*(1+N_b)*(-5 + 2*(2+N_b)*c2_b)*c2_b
        - (1 + 2*(1+N_b)*(-5 + 3*c2_b*(2+N_b))*c2_b)*Bs2_b
        + (-5 + 6*(2+N_b)*c2_b)*c2_b*Bs2_b**2
        - 2*c2_b**2 * Bs2_b**3) * term2,
        axis=(-2, -1)
    )

    d4A_sum = jnp.sum(
        ( N_b*(1+N_b)*(3 + 4*(2+N_b)*(-3 + (3+N_b)*c2_b)*c2_b)
        - 2*(1+N_b)*(3 + 2*c2_b*(2+N_b)*(-9 + 4*c2_b*(3+N_b)))*Bs2_b
        + 3*(1 + 4*(2+N_b)*(-3 + 2*(3+N_b)*c2_b)*c2_b)*Bs2_b**2
        - 4*(-3 + 4*(3+N_b)*c2_b)*c2_b*Bs2_b**3
        + 4*c2_b**2 * Bs2_b**4) * term,
        axis=(-2, -1)
    )

    # ---- 共通の指数因子とチェーンルールの外側係数 ----
    e = jnp.exp(-Bs2)                           # (...,)

    G00   = G00_sum * e

    dGdA  = dGdA_sum  * e * (2 * A / rho2)
    dGdC  = dGdC_sum  * e * (2 * c2 * jnp.sqrt(s2 / rho2))

    d2A   = d2A_sum   * e * (2 / rho2)
    d2C   = d2C_sum   * e * (2 * c2 / rho2)
    dAdC  = dAdC_sum  * e * (-4 * jnp.sqrt(c2 * s2) / rho2)

    d3A   = d3A_sum   * e * (-4 * jnp.sqrt(c2) / (rho2**1.5))
    d3A2C = d3A2C_sum * e * ( 4 * jnp.sqrt(s2) / (rho2**1.5))

    d4A   = d4A_sum   * e * (4 / (rho2**2))

    # 先頭軸=コンポーネントにまとめる（元の vstack と同じ順）
    dGs_l = jnp.stack([G00, dGdA, dGdC, d2A, d2C, dAdC, d3A, d3A2C, d4A], axis=0)  # (9, ...)

    return dGs_l

# @partial(jit, static_argnames=['lmax'])
# def get_Gs(A, B, C, G00_coeffs, lmax=10):
#     dGs = jnp.array([get_dGs_l(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)])
#     nq = dGs.shape[2]
    
#     G00s = dGs[:,0]
#     dGdAs = dGs[:,1]
#     dGdCs = dGs[:,2]
#     d2GdA2s = dGs[:,3]
#     d2GdC2s = dGs[:,4]
#     d2GdAdCs = dGs[:,5]
#     d3GdA3s = dGs[:,6]
#     d3GdA2dCs = dGs[:,7]
#     d4GdA4s = dGs[:,8]

#     zeros = jnp.zeros((1, nq))
    
#     G00s_lm1 = jnp.vstack([zeros, G00s[:-1]])
#     G00s_lm2 = jnp.vstack([zeros, G00s_lm1[:-1]])
#     G00s_lm3 = jnp.vstack([zeros, G00s_lm2[:-1]])
#     G00s_lm4 = jnp.vstack([zeros, G00s_lm3[:-1]])
    
#     dGdAs_lm1 = jnp.vstack([zeros, dGdAs[:-1]])
#     dGdAs_lm2 = jnp.vstack([zeros, dGdAs_lm1[:-1]])
#     dGdAs_lm3 = jnp.vstack([zeros, dGdAs_lm2[:-1]])

#     dGdCs_lm1 = jnp.vstack([zeros, dGdCs[:-1]])
#     dGdCs_lm2 = jnp.vstack([zeros, dGdCs_lm1[:-1]])

#     d2GdAdCs_lm1 = jnp.vstack([zeros, d2GdAdCs[:-1]])

#     d2GdA2s_lm1 = jnp.vstack([zeros, d2GdA2s[:-1]])
#     d2GdA2s_lm2 = jnp.vstack([zeros, d2GdA2s_lm1[:-1]])

#     d3GdA3s_lm1 = jnp.vstack([zeros, d3GdA3s[:-1]])

#     Gs = jnp.zeros((5, 3, lmax+1, nq))
    
#     Gs = Gs.at[0,0].set(G00s)

#     Gs = Gs.at[1,0].set(dGdAs + 0.5 * A * G00s_lm1)
#     Gs = Gs.at[0,1].set(dGdCs + 0.5 * C * G00s_lm1)

#     Gs = Gs.at[2,0].set(d2GdA2s + A * dGdAs_lm1 + 0.5 * G00s_lm1 + 0.25 * A**2 * G00s_lm2)
#     Gs = Gs.at[0,2].set(d2GdC2s + C * dGdCs_lm1 + 0.5 * G00s_lm1 + 0.25 * C**2 * G00s_lm2)
#     Gs = Gs.at[1,1].set(d2GdAdCs + 0.5 * C * dGdAs_lm1 + 0.5 * A * dGdCs_lm1 + 0.25 * A * C * G00s_lm2)

#     Gs = Gs.at[3,0].set(d3GdA3s + 1.5 * A * d2GdA2s_lm1 + 1.5 * dGdAs_lm1 \
#                         + 0.75 * A**2 * dGdAs_lm2 + 0.75 * A * G00s_lm2 + A**3 / 8 * G00s_lm3)
#     Gs = Gs.at[2,1].set(d3GdA2dCs + 0.5 * C * d2GdA2s_lm1 + A * d2GdAdCs_lm1 + 0.5 * dGdCs_lm1 \
#                         + 0.5 * A * C * dGdAs_lm2 + 0.25 * A**2 * dGdCs_lm2 + 0.25 * C * G00s_lm2 \
#                         + A**2 * C / 8 * G00s_lm3)

#     Gs = Gs.at[4,0].set(d4GdA4s + 2 * A * d3GdA3s_lm1 + 3 * d2GdA2s_lm1 \
#                         + 1.5 * A**2 * d2GdA2s_lm2 + 3 * A * dGdAs_lm2 + 0.75 * G00s_lm2 \
#                         + 0.5 * A**3 * dGdAs_lm3 + 0.75 * A**2 * G00s_lm3 \
#                         + A**4 / 16 * G00s_lm4)
    
#     return Gs

@partial(jit, static_argnames=("lmax",))
def get_Gs(A, B, C, G00_coeffs, lmax=10):
    L = lmax + 1

    # ---- l ごとの get_dGs_l を Python ループなしで評価（fori_loop + switch）----
    # 事前に l 固有の分岐を作成（coeff を閉じ込める）
    branches = []
    for i in range(L):
        coeff_i = G00_coeffs[i]  # shape (i+1, i+1)
        def make_branch(coeff_i):
            def branch(_):
                # get_dGs_l は (9, nq) を返す想定
                return get_dGs_l(A, B, C, coeff_i)
            return branch
        branches.append(make_branch(coeff_i))

    # ループ本体：各 l で (9, nq) を返し、(L, 9, nq) に積む
    out0 = jnp.zeros((L, 9, C.shape[0]), dtype=jnp.result_type(A, B, C))  # nq = len(self._q) 相当
    # もし nq を C の shape から取れない場合は適宜置き換えてください

    def body(l, out):
        dGs_l = jax.lax.switch(l, branches, operand=None)   # (9, nq)
        return out.at[l].set(dGs_l)

    dGs = jax.lax.fori_loop(0, L, body, out0)              # (L, 9, nq)

    # ---- 軸の命名を合わせて取り出し（元コードと同じ）----
    # dGs: (L, 9, nq) -> 各成分 (L, nq)
    G00s       = dGs[:, 0, :]
    dGdAs      = dGs[:, 1, :]
    dGdCs      = dGs[:, 2, :]
    d2GdA2s    = dGs[:, 3, :]
    d2GdC2s    = dGs[:, 4, :]
    d2GdAdCs   = dGs[:, 5, :]
    d3GdA3s    = dGs[:, 6, :]
    d3GdA2dCs  = dGs[:, 7, :]
    d4GdA4s    = dGs[:, 8, :]

    # ---- シフトは pad 一回 + スライスで（vstack 多用を排除）----
    # 例：X_lm1 = [0; X[:-1]], X_lm2 = [0; 0; X[:-2]], ...
    def shift_down(x, s):
        # x: (L, nq) -> (L, nq) with s zeros inserted at top
        return jnp.pad(x, ((s, 0), (0, 0)))[:L, :]

    G00s_lm1, G00s_lm2, G00s_lm3, G00s_lm4 = (shift_down(G00s, s) for s in (1, 2, 3, 4))
    dGdAs_lm1, dGdAs_lm2, dGdAs_lm3       = (shift_down(dGdAs, s) for s in (1, 2, 3))
    dGdCs_lm1, dGdCs_lm2                   = (shift_down(dGdCs, s) for s in (1, 2))
    d2GdAdCs_lm1                            = shift_down(d2GdAdCs, 1)
    d2GdA2s_lm1, d2GdA2s_lm2               = (shift_down(d2GdA2s, s) for s in (1, 2))
    d3GdA3s_lm1                             = shift_down(d3GdA3s, 1)

    # ---- 最終合成（.at[...] を最小限に）----
    nq = G00s.shape[1]
    Gs = jnp.zeros((5, 3, L, nq), dtype=G00s.dtype)

    Gs = Gs.at[0, 0].set(G00s)

    Gs = Gs.at[1, 0].set(dGdAs + 0.5 * A * G00s_lm1)
    Gs = Gs.at[0, 1].set(dGdCs + 0.5 * C * G00s_lm1)

    Gs = Gs.at[2, 0].set(d2GdA2s + A * dGdAs_lm1 + 0.5 * G00s_lm1 + 0.25 * (A**2) * G00s_lm2)
    Gs = Gs.at[0, 2].set(d2GdC2s + C * dGdCs_lm1 + 0.5 * G00s_lm1 + 0.25 * (C**2) * G00s_lm2)
    Gs = Gs.at[1, 1].set(d2GdAdCs + 0.5 * C * dGdAs_lm1 + 0.5 * A * dGdCs_lm1
                         + 0.25 * (A * C) * G00s_lm2)

    Gs = Gs.at[3, 0].set(d3GdA3s + 1.5 * A * d2GdA2s_lm1 + 1.5 * dGdAs_lm1
                         + 0.75 * (A**2) * dGdAs_lm2 + 0.75 * A * G00s_lm2
                         + (A**3) / 8 * G00s_lm3)

    Gs = Gs.at[2, 1].set(d3GdA2dCs + 0.5 * C * d2GdA2s_lm1 + A * d2GdAdCs_lm1 + 0.5 * dGdCs_lm1
                         + 0.5 * A * C * dGdAs_lm2 + 0.25 * (A**2) * dGdCs_lm2
                         + 0.25 * C * G00s_lm2 + (A**2) * C / 8 * G00s_lm3)

    Gs = Gs.at[4, 0].set(d4GdA4s + 2 * A * d3GdA3s_lm1 + 3 * d2GdA2s_lm1
                         + 1.5 * (A**2) * d2GdA2s_lm2 + 3 * A * dGdAs_lm2 + 0.75 * G00s_lm2
                         + 0.5 * (A**3) * dGdAs_lm3 + 0.75 * (A**2) * G00s_lm3
                         + (A**4) / 16 * G00s_lm4)

    return Gs

