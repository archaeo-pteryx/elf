import os
import glob, re
from typing import NamedTuple, Optional
import numpy as _np
import sympy as sym
import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp

from . import fftlog

from . import pt_coeff
from . import pt_matrix
from .base import PowerSpectrum
from .utils import eval_power_coeffs

from . import spline


def _table_names(directory, prefix):
    """Sorted file stems ``<prefix>*`` of the ``.txt`` tables in the package directory ``directory``."""
    fnames = sorted(glob.glob(os.path.dirname(__file__) + f'/{directory}/{prefix}*.txt'))
    return [re.split('/', fname)[-1][:-4] for fname in fnames]


def _read_coeff_tables(prefix):
    """``(names, lnm, coeff_info)`` of the ``pt_coeff`` blocks ``<prefix>_l_n_m``; rows zero-padded to one length."""
    names = _table_names('pt_coeff', prefix)
    lnm, coeff_info = [], []
    for name in names:
        str_list = re.split('_', name)
        lnm.append([int(str_list[1]), int(str_list[2]), int(str_list[3])])
        coeff_info.append(pt_coeff.get_coeff_info(os.path.dirname(__file__) + f'/pt_coeff/{name}.txt'))
    max_len = max(len(lst) for lst in coeff_info)
    padded = [lst + [[0, 0, 0, 0, 0, 0, 0.0]] * (max_len - len(lst)) for lst in coeff_info]
    return names, _np.asarray(lnm), jnp.array(padded)


def get_degree_dict(name):
    """Powers of (mu, f, b1, ...) encoded in a ``pt_matrix`` kernel file name."""
    degree_name = re.split('=', name)[-1]
    str_list = re.split('_', degree_name)
    degree_dict = {}
    for s in str_list:
        string = re.split('-', s)
        key = string[0]
        val = int(string[1])
        degree_dict[key] = val
    return degree_dict


def _is_bias_block(name):
    """Whether a ``pt_matrix`` kernel belongs to the bias-operator group (a power of b2, bG2 or bGamma3).

    The other blocks form the matter group.  The two groups use their own decomposition
    biases (``FFTlogSettings.nu_matrix_*``).
    """
    degree_dict = get_degree_dict(name)
    return any(degree_dict.get(key, 0) > 0 for key in ('b2', 'bG2', 'bGamma3'))


def uv_terms():
    """``[((mu_pow, f_pow, b1_pow), coefficient)]`` of ``Z1 Z3_UV`` (exact rationals).

    ``Z1 = b1 + f mu^2``, ``Z3_UV = -61/315 b1 + ((-3/5 + 2/105 b1) f + (-16/35 - 1/3 b1) f^2) mu^2
    + (-46/105 f^2 - 1/3 f^3) mu^4``.
    """
    mu, f, b1 = sym.symbols('mu f b1')
    R = sym.Rational
    z1 = b1 + f * mu**2
    z3_uv = (-R(61, 315) * b1
             + ((-R(3, 5) + R(2, 105) * b1) * f + (-R(16, 35) - R(1, 3) * b1) * f**2) * mu**2
             + (-R(46, 105) * f**2 - R(1, 3) * f**3) * mu**4)
    return sorted(sym.Poly(sym.expand(z1 * z3_uv), mu, f, b1).terms())


def _one_monomial(degrees, nterms):
    """Table ``(nblocks, nterms, 7)`` of one monomial per block (``degrees`` rows), zero-padded to ``nterms``."""
    degrees = _np.asarray(degrees, dtype=_np.float64)
    table = _np.zeros((degrees.shape[0], nterms, 7))
    table[:, 0, :6] = degrees
    table[:, 0, 6] = 1.0
    return table


class PTTerms(NamedTuple):
    """EPT PT terms on the core k grid, which depend only on ``P_lin`` and ``h``.

    ``p22`` ``(n22, nk)``, ``p13_kernel`` ``(n13, nk)`` (P13 blocks divided by P(k) and
    by the block's static ``k^p``, ``p`` = ``EPT._k_power_13``.  In the matrix method the
    last row is the UV constant ``(1/2 pi^2) int dk P`` with p = 2).
    ``pk_int2 = (1/2 pi^2) int dk k^2 P^2`` is the b2^2 constant.  ``*_nw`` are the same
    for the no-wiggle spectrum (None without ``do_irres``).  The no-wiggle data
    ``pk_nw_data``, ``Sigma2``, ``dSigma2`` (as in the LPT ``PTTerms``) are present
    whenever the IR resummation or the 'linear_ir_resum' counterterm base needs them,
    and None otherwise.
    """

    p22: jnp.ndarray
    p13_kernel: jnp.ndarray
    pk_int2: jnp.ndarray
    p22_nw: Optional[jnp.ndarray]
    p13_kernel_nw: Optional[jnp.ndarray]
    pk_nw_data: Optional[jnp.ndarray]
    Sigma2: Optional[jnp.ndarray]
    dSigma2: Optional[jnp.ndarray]


class EPT(PowerSpectrum):

    def __init__(self,
                 kmin_fft=1e-5,
                 kmax_fft=1e2,
                 nfft=512,
                 pad_mode='zero-pad',
                 fftlog_settings=None,
                 ngauss=4,
                 ells=(0, 2, 4),
                 counterterm_base='linear_ir_resum',
                 irres_method='DST',
                 r_bao=110.0,
                 lambda_ir=0.2,
                 subtract_k0_const=False,
                 do_irres=True,
                 method='hankel',
                 ):
        """One-loop Eulerian-PT galaxy power spectrum in redshift space.

        ``Params.bias`` is (b1, b2, bG2, bGamma3); shared options as in
        :class:`base.PowerSpectrum` (``counterterm_base`` 'linear_ir_resum' or
        'linear'; ``subtract_k0_const`` drops the b2^2 k -> 0 constant).
        ``do_irres`` switches the IR resummation of tree level and loop.
        ``method``: 'hankel' (FFTlog) or 'matrix' (power-law-decomposition
        matrices, valid only for k in [3e-3, 3] h/Mpc).
        """

        if method not in ('matrix', 'hankel'):
            raise ValueError(f"method must be 'matrix' or 'hankel', got {method!r}")
        super().__init__(kmin_fft=kmin_fft, kmax_fft=kmax_fft, nfft=nfft, pad_mode=pad_mode,
                         fftlog_settings=fftlog_settings, ngauss=ngauss, ells=ells,
                         counterterm_base=counterterm_base, irres_method=irres_method,
                         r_bao=r_bao, lambda_ir=lambda_ir, subtract_k0_const=subtract_k0_const)
        if self.counterterm_base == 'zeldovich':
            raise ValueError("EPT does not support counterterm_base='zeldovich'")

        self.do_irres = do_irres
        self.method = method

        self._initialize_loop_coeff()
        if self.method == 'matrix':
            self._initialize_loop_matrix()

        # (mu, f, b1, b2, bG2, bGamma3) monomials + coefficient of each block, (nblocks, nterms, 7),
        # and the static extra power p of k of each P13 block, applied at the true k.
        if method == 'matrix':
            # one monomial per block, then the UV block: the monomials of Z1 Z3_UV, whose kernel
            # row is the constant pk_int = (1/2 pi^2) int dk P and whose p = 2 (the term
            # k^2 P(k) pk_int Z1 Z3_UV of P13, which the matrix blocks do not contain)
            uv = _np.array([[m, p, b, 0, 0, 0, float(c)] for (m, p, b), c in uv_terms()])
            self._monomials_22 = jnp.asarray(_one_monomial(self.degrees_22, 1))
            self._monomials_13 = jnp.asarray(_np.concatenate(
                [_one_monomial(self.degrees_13, uv.shape[0]), uv[None]], axis=0))
            self._k_power_13 = _np.array([0] * len(self.matrix_block_names_13) + [2])
        else:
            self._monomials_22 = self.coeff_info_22
            self._monomials_13 = self.coeff_info_13
            self._k_power_13 = _np.zeros(len(self.pkmu_coeff_names_13), dtype=int)
        self._mu_degree_max = int(max(_np.max(_np.asarray(table[..., 0]))
                                      for table in (self._monomials_22, self._monomials_13)))

    def _initialize_loop_coeff(self):
        _, lnm_22, self.coeff_info_22 = _read_coeff_tables('22')
        self.pkmu_coeff_names_13, lnm_13, self.coeff_info_13 = _read_coeff_tables('13')
        # m stays a static numpy array: it selects the m = 0 residual branch at trace time.
        self._ln_22 = jnp.asarray(lnm_22[:, :2], dtype=jnp.int32)
        self._m_22 = lnm_22[:, 2]
        self._lnm_13 = jnp.asarray(lnm_13)

        ln_pairs = [[0,0], [0,-2], [1,-1], [2,0], [2,-2], [3,-1], [4,0]]
        self._ln_list = jnp.array(ln_pairs)
        self._ln_list_static = tuple((int(l), int(n)) for l, n in ln_pairs)
        lmax = jnp.max(self._ln_list[:, 0])
        self._set_hankel(lmax)

    def _set_hankel(self, lmax):
        """Grids and kernels of the chain ``P(k) -> xi_l^n(q) -> P22/P13(k)``.

        The padded grid and end weights of the input also serve both methods' ``pk_int2``,
        the matrix UV constant and the ``get_xi_ells`` transforms.  The forward chain
        ``P(k) -> xi_l^n(q)`` (:func:`fftlog.forward_transform`) has all seven rows with
        the input bias ``nu_xi`` and again with ``nu_xi_n0``, each one rfft shared by the
        rows, kernel ``g_l`` at ``nu_in + n + 3`` with the low-ringing phase of ``nu``, so
        row (l, n) lands on ``q_padded[l]``, the conjugate grid of the backward kernel
        ``u_m[l]``.  The n = 0 rows take the ``nu_xi_n0`` result.
        """
        self._hankel = fftlog.HankelGrids(self._k, self.fftlog_settings, self.pad_mode, lmax)
        self._set_xi_ells_transform()

        ells = [l for l, _ in self._ln_list_static]
        ns = [n for _, n in self._ln_list_static]
        self._xi_forward = fftlog.forward_transform(self._hankel.grid, self.fftlog_settings.nu, ells, ns, nu_in=self.fftlog_settings.nu_xi)
        self._xi_forward_n0 = fftlog.forward_transform(self._hankel.grid, self.fftlog_settings.nu, ells, ns, nu_in=self.fftlog_settings.nu_xi_n0)
        self._xi_n0_mask = jnp.array(
            [n == 0 for _, n in self._ln_list_static], dtype=bool
        )

    def _initialize_loop_matrix(self):
        """Matrices and monomial degrees of the ``pt_matrix`` blocks, one (mu, f, bias) monomial per block."""
        self.matrix_block_names_22 = _table_names('pt_matrix', '22')
        self.matrix_block_names_13 = _table_names('pt_matrix', '13')

        is_matter_22 = [not _is_bias_block(name) for name in self.matrix_block_names_22]
        is_matter_13 = [not _is_bias_block(name) for name in self.matrix_block_names_13]

        matrix = self._set_matrix(self.matrix_block_names_22 + self.matrix_block_names_13)

        self.matrices_22 = jnp.array([matrix[name] for name in self.matrix_block_names_22])
        self.matrices_13 = jnp.array([matrix[name] for name in self.matrix_block_names_13])

        self._idx_22_matter = jnp.array([i for i, m in enumerate(is_matter_22) if m], dtype=jnp.int32)
        self._idx_22_bias = jnp.array([i for i, m in enumerate(is_matter_22) if not m], dtype=jnp.int32)
        self._idx_13_matter = jnp.array([i for i, m in enumerate(is_matter_13) if m], dtype=jnp.int32)
        self._idx_13_bias = jnp.array([i for i, m in enumerate(is_matter_13) if not m], dtype=jnp.int32)

        self.matrices_22_matter = self.matrices_22[self._idx_22_matter]
        self.matrices_22_bias = self.matrices_22[self._idx_22_bias]
        self.matrices_13_matter = self.matrices_13[self._idx_13_matter]
        self.matrices_13_bias = self.matrices_13[self._idx_13_bias]

        def get_degree_vector(name):
            d = get_degree_dict(name)
            if name in self.matrix_block_names_22:
                return [d['mu'], d['f'], d['b1'], d['b2'], d['bG2'], 0]
            else:
                return [d['mu'], d['f'], d['b1'], 0, d['bG2'], d['bGamma3']]

        self.degrees_22 = jnp.array([get_degree_vector(name) for name in self.matrix_block_names_22])
        self.degrees_13 = jnp.array([get_degree_vector(name) for name in self.matrix_block_names_13])

    def _set_matrix(self, names):
        """PT matrices of the given blocks at the decomposition bias of their group."""
        fs = self.fftlog_settings
        mat = {}
        for name in names:
            mat_file = os.path.dirname(__file__) + f'/pt_matrix/{name}.txt'
            if name.startswith('22'):
                mat[name] = pt_matrix.PTMatrix22(mat_file)
            elif name.startswith('13'):
                mat[name] = pt_matrix.PTMatrix13(mat_file)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))

        eta_m = fftlog.LogGrid(self._k, 0).eta_full
        matrix = {}

        for name in names:
            if not _is_bias_block(name):
                nu = fs.nu_matrix_matter
            elif name in self.matrix_block_names_22:
                nu = fs.nu_matrix_p22_bias
            else:
                nu = fs.nu_matrix_p13_bias

            nu_m = -0.5 * (nu + eta_m * 1j)
            if name.startswith('22'):
                nu_m1, nu_m2 = jnp.meshgrid(nu_m, nu_m)
                matrix[name] = mat[name](nu_m1, nu_m2).T
            elif name.startswith('13'):
                matrix[name] = mat[name](nu_m)

        return matrix

    @partial(jit, static_argnames=['self'])
    def get_pt_terms(self, pk_data, h):
        """The :class:`PTTerms` of ``pk_data``.

        ``h`` enters only the wiggle/no-wiggle split, computed when ``do_irres`` or the
        'linear_ir_resum' base needs it.  ``h=None`` works only without them.
        """
        k = self._k
        pk = self._pk_at(k, pk_data)
        if self.do_irres or self.counterterm_base == 'linear_ir_resum':
            pk_nw_data, Sigma2, dSigma2 = self._get_ir_data(pk_data, h)
        else:
            pk_nw_data = Sigma2 = dSigma2 = None
        if self.do_irres:
            # pk is rebuilt as pk_nw + pk_w on purpose: that sum is what the IR pipeline transforms
            pk_nw = self._pk_at(k, pk_nw_data)
            pk_w = pk - pk_nw
            pk = pk_nw + pk_w
        p22, p13_kernel = self._get_blocks(pk)
        # the end weight multiplies P^2 once
        pk_int2 = self._hankel.grid.moment(
            self._hankel.w_input * fftlog.pad(pk, self._hankel.grid, self._hankel.pad_mode)**2, 3) / (2 * jnp.pi**2)
        if self.do_irres:
            p22_nw, p13_kernel_nw = self._get_blocks(pk_nw)
        else:
            p22_nw = p13_kernel_nw = None
        return PTTerms(p22=p22, p13_kernel=p13_kernel, pk_int2=pk_int2,
                       p22_nw=p22_nw, p13_kernel_nw=p13_kernel_nw,
                       pk_nw_data=pk_nw_data, Sigma2=Sigma2, dSigma2=dSigma2)

    def _get_blocks(self, pk):
        """``(p22, p13_kernel)`` of one spectrum on the core k grid for ``self.method``.

        hankel: P13 kernel = ``k^m H_l[q^2 xi_l^n]``.  matrix: power-law decomposition,
        and the last P13 row is the UV constant ``pk_int``.
        """
        if self.method == 'matrix':
            fs = self.fftlog_settings
            p_q_matter = fftlog.power_law_decomposition(pk, self._k, fs.nu_matrix_matter)
            # nu > -3/2 on purpose: the P22 bias blocks come out as I(k) - I(0) by analytic continuation
            p_q_p22_bias = fftlog.power_law_decomposition(pk, self._k, fs.nu_matrix_p22_bias)
            p_q_p13_bias = fftlog.power_law_decomposition(pk, self._k, fs.nu_matrix_p13_bias)
            p22 = self._matrix_p22_blocks(p_q_matter, p_q_p22_bias)
            p13_kernel = self._matrix_p13_kernels(p_q_matter, p_q_p13_bias)
            pk_int = self._hankel.grid.moment(self._hankel.pad_input(pk), 1) / (2 * jnp.pi**2)
            # UV block: a row constant in k, its k^2 is applied at the true k (_k_power_13)
            uv = jnp.broadcast_to(pk_int, (1, self._nfft))
            return p22, jnp.concatenate([p13_kernel, uv], axis=0)

        xi_ln = self._get_xi_ln_array(pk)
        p22 = self._get_p22_blocks(xi_ln)
        p13_kernel = self._get_p13_kernel_blocks(xi_ln)
        return p22, p13_kernel

    def _pkmu_true(self, k_true, mu_true, sin_true, pt_terms, pk_data, params):
        """Tree level plus one loop at the true (k, mu) (``sin_true`` unused).

        ``L[P] = sum_b p22_b C22_b(mu) + P sum_b k^{p_b} p13_kernel_b C13_b(mu)`` (``p_b`` =
        ``_k_power_13``).  With ``do_irres``
        tree = ``Z1 Z1' [P_nw + e^{-D} (1 + D) P_w]``, loop = ``L[P_nw] + e^{-D} (L[P] - L[P_nw])``,
        ``D = k^2 Sigma^2_s`` (``damp_exponent``).  The b2^2 constant ``b2 b2'/2 pk_int2`` is
        added after, undamped.  Primes denote the second tracer ``params.bias_b``.

        The blocks are interpolated in ln k at the true k.
        """
        f = params.f
        bias_a = params.bias_a
        bias_b = params.bias_b
        pk, pk_nw, pk_w, damp_exponent = self._linear_at(k_true, mu_true, f, pt_terms, pk_data)   # (nk, nmu)

        # mu powers as products: mu ** p with p = 0 has a NaN derivative at mu = 0.
        mu_pows = [jnp.ones_like(mu_true)]
        for _ in range(self._mu_degree_max):
            mu_pows.append(mu_pows[-1] * mu_true)
        mu_pows = jnp.stack(mu_pows, axis=0)                          # (degree + 1, nmu)

        coeffs = []
        for table in (self._monomials_22, self._monomials_13):
            nblocks, nterms = table.shape[:2]
            mu_pow, c = eval_power_coeffs(table[:, :, :6].reshape(-1, 6), f, bias_a, bias_b)
            c = (table[:, :, 6].reshape(-1) * c)[:, None] * mu_pows[mu_pow]
            coeffs.append(jnp.sum(c.reshape(nblocks, nterms, -1), axis=1))
        C22, C13 = coeffs

        blocks = [pt_terms.p22, pt_terms.p13_kernel]
        if self.do_irres:
            blocks += [pt_terms.p22_nw, pt_terms.p13_kernel_nw]
        n22, n13 = pt_terms.p22.shape[0], pt_terms.p13_kernel.shape[0]
        stack = jnp.concatenate(blocks, axis=0)
        # (nfft, nblocks) -> (nk, nmu, nblocks)
        stack = spline.interp1d(jnp.log(k_true), self._logk_fft, stack.T, axis=0)
        if _np.any(self._k_power_13):
            # static k^p of the P13 blocks, at the true k, in the full and the no-wiggle part
            k_pow = _np.concatenate([_np.zeros(n22, dtype=int), self._k_power_13])
            k_pow = _np.tile(k_pow, len(blocks) // 2)
            for p in _np.unique(k_pow[k_pow != 0]):
                stack = jnp.where(k_pow == p, stack * k_true[..., None] ** int(p), stack)

        b1_a, b1_b = bias_a[0], bias_b[0]
        Z1_a = b1_a + f * mu_true**2
        Z1_b = b1_b + f * mu_true**2

        def loop(offset, P):
            p22 = stack[:, :, offset:offset + n22]
            p13_kernel = stack[:, :, offset + n22:offset + n22 + n13]
            return jnp.sum(p22 * C22.T, axis=-1) + P * jnp.sum(p13_kernel * C13.T, axis=-1)

        loop_full = loop(0, pk)
        if self.do_irres:
            loop_nw = loop(n22 + n13, pk_nw)
            e_damp = jnp.exp(-damp_exponent)
            pkmu = (Z1_a * Z1_b) * (pk_nw + e_damp * (1 + damp_exponent) * pk_w)
            pkmu = pkmu + loop_nw + e_damp * (loop_full - loop_nw)
        else:
            pkmu = (Z1_a * Z1_b) * pk + loop_full
        if not self.subtract_k0_const:
            # added after the IR combination: a constant contact term must not be damped
            pkmu = pkmu + (bias_a[1] * bias_b[1]) / 2. * pt_terms.pk_int2
        return pkmu

    def _merge_groups(self, idx_matter, blocks_matter, idx_bias, blocks_bias):
        """``k^3 Re[blocks]`` with the matter/bias group rows put back at ``idx_matter``/``idx_bias``."""
        out = jnp.zeros((idx_matter.shape[0] + idx_bias.shape[0], self._nfft), dtype=blocks_matter.dtype)
        out = out.at[idx_matter].set(blocks_matter).at[idx_bias].set(blocks_bias)
        return (self._k**3)[None, :] * jnp.real(out)

    def _matrix_p22_blocks(self, p_q_matter, p_q_bias):
        """Matrix-path P22 blocks from the decompositions of the matter and bias groups."""
        return self._merge_groups(
            self._idx_22_matter,
            jnp.einsum('tnm,nj,mj->tj', self.matrices_22_matter, p_q_matter, p_q_matter),
            self._idx_22_bias,
            jnp.einsum('tnm,nj,mj->tj', self.matrices_22_bias, p_q_bias, p_q_bias))

    def _matrix_p13_kernels(self, p_q_matter, p_q_bias):
        """Matrix-path P13 kernels (blocks without their outer P(k)) from the matter and bias decompositions."""
        return self._merge_groups(
            self._idx_13_matter, jnp.einsum('tn,nj->tj', self.matrices_13_matter, p_q_matter),
            self._idx_13_bias, jnp.einsum('tn,nj->tj', self.matrices_13_bias, p_q_bias))

    def _get_xi_ln_array(self, array):
        """Forward transforms ``xi_ln[l, n]`` of ``array`` on ``q_padded[l]`` (upper padded band zeroed)."""
        fx = self._hankel.pad_input(array / (2 * jnp.pi**2))
        xis = self._xi_forward.apply(fx, self._hankel.grid)
        xis_n0 = self._xi_forward_n0.apply(fx, self._hankel.grid)
        xis = jnp.where(self._xi_n0_mask[:, None], xis_n0, xis) * self._hankel.mask_xi
        xi_ln = jnp.zeros((5, 3, xis.shape[-1]))
        ls = self._ln_list[:, 0]
        ns = self._ln_list[:, 1]
        return xi_ln.at[ls, ns].set(xis)

    def _q_to_k(self, l, n, array):
        """Backward FFTlog of a source on ``q_padded[l]`` onto the core k grid."""
        # w_xi: end weight 1/2 at the support edge (FFTlog's m = 0 coefficient is the node sum)
        fx = array * self._hankel.q_padded[l]**(n + 3) * self._hankel.w_xi
        return fftlog.hankel(self.fftlog_settings.nu, fx, self._hankel.q_padded[l], self._hankel.k_padded,
                             self._hankel.u_m[l], self._hankel.npad, crop=True)

    def _get_p22_blocks(self, xi_ln):
        """P22 blocks ``(nterms, nk)`` in ``coeff_info_22`` order.

        ``B(k) = 4 pi int dq q^2 (xi_l^n)^2 j_0(kq)``; block = ``k^m B`` for m > 0 and
        ``B - C`` for m = 0, ``C = 4 pi dln sum_j w_j q_j^3 (xi_l^n)_j^2``.
        """
        def block(ln):
            l, n = ln[0], ln[1]
            xi = xi_ln[l, n]
            source = spline.interp1d(jnp.log(self._hankel.q_padded[0]),
                                     jnp.log(self._hankel.q_padded[l]), xi * xi)
            pk_ln = 4 * jnp.pi * self._q_to_k(0, 0, source)
            # end-weighted node sum of the same source, not B(k_min), which carries the output edge error
            c0 = 4 * jnp.pi * self._hankel.q_grids[0].moment(source * self._hankel.w_xi, 3)
            return pk_ln, c0

        pk_ln, c0 = jax.vmap(block)(self._ln_22)      # (nterms, nk), (nterms,)
        m = self._m_22[:, None]                        # static
        return jnp.where(m == 0, pk_ln - c0[:, None], self._k ** m * pk_ln)

    def _get_p13_kernel_blocks(self, xi_ln):
        """P13 kernel blocks ``(nterms, nk)`` in ``_lnm_13`` order: ``k^m H_l[q^2 xi_l^n]``."""

        def get_p13_kernel(term):
            l, n, m = term
            pk_ln = self._q_to_k(l, -1, xi_ln[l, n])  # (nk,)
            return (self._k ** m) * pk_ln
        return jax.vmap(get_p13_kernel)(self._lnm_13)  # (nterms, nk)
