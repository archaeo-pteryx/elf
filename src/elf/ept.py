import os
import glob, re
from typing import NamedTuple, Optional
import numpy as _np
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


# Matrix-path group tags (matter / bias operator); the biases themselves are in FFTlogSettings.
_NU_GROUP_MATTER_TAG = -0.3
_NU_GROUP_BIAS_TAG = -1.6


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


def get_nu_group_tag_from_name(name):
    """Matrix-path group tag (matter or bias operator) of a PT kernel name."""
    degree_dict = get_degree_dict(name)
    if 'b2' in degree_dict.keys():
        if degree_dict['b2'] > 0: return _NU_GROUP_BIAS_TAG
    if 'bG2' in degree_dict.keys():
        if degree_dict['bG2'] > 0: return _NU_GROUP_BIAS_TAG
    if 'bGamma3' in degree_dict.keys():
        if degree_dict['bGamma3'] > 0: return _NU_GROUP_BIAS_TAG
    return _NU_GROUP_MATTER_TAG


class Corrs(NamedTuple):
    """EPT intermediate quantities on the core k grid; depend only on ``P_lin`` and ``h``.

    ``p22`` ``(n22, nk)``, ``p13_kernel`` ``(n13, nk)`` (P13 blocks divided by P(k)),
    ``pk_int = (1/2 pi^2) int dk P`` (matrix UV term, else None),
    ``pk_int2 = (1/2 pi^2) int dk k^2 P^2`` (b2^2 constant); ``*_nw`` the same for
    the no-wiggle spectrum (``do_irres``); ``pk_nw_data``, ``Sigma2``, ``dSigma2``
    as in the LPT ``Corrs`` (None without IR resummation).
    """

    p22: jnp.ndarray
    p13_kernel: jnp.ndarray
    pk_int: Optional[jnp.ndarray]
    pk_int2: jnp.ndarray
    p22_nw: Optional[jnp.ndarray]
    p13_kernel_nw: Optional[jnp.ndarray]
    pk_int_nw: Optional[jnp.ndarray]
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

        # (mu, f, b1, b2, bG2, bGamma3) monomials + coefficient of each block, (nblocks, nterms, 7).
        def one_monomial(degrees):
            degrees = jnp.asarray(degrees, dtype=self.coeff_info_22.dtype)
            return jnp.concatenate([degrees, jnp.ones_like(degrees[:, :1])], axis=1)[:, None, :]

        self._monomials_22 = one_monomial(self.degrees_22) if method == 'matrix' else self.coeff_info_22
        self._monomials_13 = self.coeff_info_13 if method == 'hankel' else one_monomial(self.degrees_13)
        self._mu_degree_max = int(max(_np.max(_np.asarray(table[..., 0]))
                                      for table in (self._monomials_22, self._monomials_13)))

    def _initialize_loop_coeff(self):
        self.pkmu_coeff_names_22, lnm_22, self.coeff_info_22 = _read_coeff_tables('22')
        self.pkmu_coeff_names_13, lnm_13, self.coeff_info_13 = _read_coeff_tables('13')
        # m stays a static numpy array: it selects the m = 0 residual branch at trace time.
        self._ln_22 = jnp.asarray(lnm_22[:, :2], dtype=jnp.int32)
        self._m_22 = lnm_22[:, 2]
        self._lnm_13 = jnp.asarray(lnm_13)

        ln_pairs = [[0,0], [0,-2], [1,-1], [2,0], [2,-2], [3,-1], [4,0]]
        self._ln_list = jnp.array(ln_pairs)
        self._ln_list_static = tuple((int(l), int(n)) for l, n in ln_pairs)
        self._ln_index = {pair: i for i, pair in enumerate(self._ln_list_static)}
        lmax = jnp.max(self._ln_list[:, 0])
        self._set_hankel(lmax)

    def _set_hankel(self, lmax):
        """Grids and kernels of the chain ``P(k) -> xi_l^n(q) -> P22/P13(k)``.

        Forward row (l, n): one shared rfft per bias (``nu_pld``, ``nu_pld_n0`` for
        n = 0), kernel ``g_l`` at ``nu + n + 3`` with the low-ringing phase of
        ``nu``, so it lands on ``q_padded[l]``, the conjugate grid of the backward
        kernel ``u_m[l]``.
        """
        fs = self.fftlog_settings
        hg = self._hankel = fftlog.HankelGrids(self._k, fs, self.pad_mode, lmax)
        self._set_xi_ells_transform()

        def build_forward(nu):
            u_ms, q_factors = [], []
            for l, n in self._ln_list_static:
                nu_eff = nu + n + 3
                # phase (lnxy) of the ordinary nu, kernel bias nu_eff: keeps q_padded[l] conjugate to k_padded
                u_ms.append(fftlog.hankel_setup(l, nu_eff, hg.grid, lnxy=hg.lnxy[l])[2])
                q_factors.append(hg.q_padded[l] ** (-nu_eff))
            return jnp.stack(u_ms, axis=0), jnp.stack(q_factors, axis=0)

        self._xi_pld_u_m, self._xi_pld_q_factor = build_forward(fs.nu_pld)
        self._xi_pld_u_m_n0, self._xi_pld_q_factor_n0 = build_forward(fs.nu_pld_n0)
        self._xi_pld_n0_mask = jnp.array(
            [n == 0 for _, n in self._ln_list_static], dtype=bool
        )

    def _initialize_loop_matrix(self):
        self.pkmu_term_names_22 = _table_names('pt_matrix', '22')
        self.pkmu_term_names_13 = _table_names('pt_matrix', '13')

        matter = _NU_GROUP_MATTER_TAG
        is_matter_22 = [get_nu_group_tag_from_name(name) == matter for name in self.pkmu_term_names_22]
        is_matter_13 = [get_nu_group_tag_from_name(name) == matter for name in self.pkmu_term_names_13]

        matrix = self._set_matrix(self.pkmu_term_names_22 + self.pkmu_term_names_13)

        self.matrices_22 = jnp.array([matrix[name] for name in self.pkmu_term_names_22])
        self.matrices_13 = jnp.array([matrix[name] for name in self.pkmu_term_names_13])

        self._idx_22_nu1 = jnp.array([i for i, m in enumerate(is_matter_22) if m], dtype=jnp.int32)
        self._idx_22_nu2 = jnp.array([i for i, m in enumerate(is_matter_22) if not m], dtype=jnp.int32)
        self._idx_13_nu1 = jnp.array([i for i, m in enumerate(is_matter_13) if m], dtype=jnp.int32)
        self._idx_13_nu2 = jnp.array([i for i, m in enumerate(is_matter_13) if not m], dtype=jnp.int32)

        self.matrices_22_nu1 = self.matrices_22[self._idx_22_nu1]
        self.matrices_22_nu2 = self.matrices_22[self._idx_22_nu2]
        self.matrices_13_nu1 = self.matrices_13[self._idx_13_nu1]
        self.matrices_13_nu2 = self.matrices_13[self._idx_13_nu2]

        def get_degree_vector(name):
            d = get_degree_dict(name)
            if name in self.pkmu_term_names_22:
                return [d['mu'], d['f'], d['b1'], d['b2'], d['bG2'], 0]
            else:
                return [d['mu'], d['f'], d['b1'], 0, d['bG2'], d['bGamma3']]

        self.degrees_22 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_22])
        self.degrees_13 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_13])

    def _set_matrix(self, names):
        """PT matrices of the given blocks at the decomposition bias of their group."""
        fs = self.fftlog_settings
        mat = {}
        for name in names:
            mat_file = os.path.dirname(__file__) + f'/pt_matrix/{name}.txt'
            if '22' in name:
                mat[name] = pt_matrix.PTMatrix22(mat_file)
            elif '13' in name:
                mat[name] = pt_matrix.PTMatrix13(mat_file)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))

        eta_m = fftlog.LogGrid(self._k, 0).eta_full
        matrix = {}

        for name in names:
            if get_nu_group_tag_from_name(name) == _NU_GROUP_MATTER_TAG:
                nu = fs.nu_matrix_matter
            elif name in self.pkmu_term_names_22:
                nu = fs.nu_matrix_p22_bias
            else:
                nu = fs.nu_matrix_p13_bias

            nu_m = -0.5 * (nu + eta_m * 1j)
            if '22' in name:
                nu_m1, nu_m2 = jnp.meshgrid(nu_m, nu_m)
                matrix[name] = mat[name](nu_m1, nu_m2).T
            elif '13' in name:
                matrix[name] = mat[name](nu_m)
        
        return matrix
    
    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data, h):
        """The :class:`Corrs` of ``pk_data`` (``h`` enters only the wiggle/no-wiggle split)."""
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
        p22, p13_kernel, pk_int = self._get_blocks(pk)
        hg = self._hankel
        # the end weight multiplies P^2 once
        pk_int2 = hg.grid.moment(
            hg.w_input * fftlog.pad(pk, hg.grid, hg.pad_mode)**2, 3) / (2 * jnp.pi**2)
        if self.do_irres:
            p22_nw, p13_kernel_nw, pk_int_nw = self._get_blocks(pk_nw)
        else:
            p22_nw = p13_kernel_nw = pk_int_nw = None
        return Corrs(p22=p22, p13_kernel=p13_kernel, pk_int=pk_int, pk_int2=pk_int2,
                     p22_nw=p22_nw, p13_kernel_nw=p13_kernel_nw, pk_int_nw=pk_int_nw,
                     pk_nw_data=pk_nw_data, Sigma2=Sigma2, dSigma2=dSigma2)

    def _get_blocks(self, pk):
        """``(p22, p13_kernel, pk_int)`` of one spectrum on the grid for ``self.method``.

        hankel: P13 kernel = ``k^m H_l[q^2 xi_l^n]``, ``pk_int = None``; matrix:
        power-law decomposition, ``pk_int`` = constant of the P13 UV term.
        """
        if self.method == 'matrix':
            fs = self.fftlog_settings
            p_q_1 = fftlog.power_law_decomposition(pk, self._k, fs.nu_matrix_matter)
            # nu > -3/2 on purpose: the P22 bias blocks come out as I(k) - I(0) by analytic continuation
            p_q_2 = fftlog.power_law_decomposition(pk, self._k, fs.nu_matrix_p22_bias)
            p_q_3 = fftlog.power_law_decomposition(pk, self._k, fs.nu_matrix_p13_bias)
            p22 = self._get_pkmu_terms_22(p_q_1, p_q_2)
            p13_kernel = self._get_pkmu_terms_13(p_q_1, p_q_3)
            pk_int = self._hankel.grid.moment(self._hankel.pad_input(pk), 1) / (2 * jnp.pi**2)
            return p22, p13_kernel, pk_int

        xi_ln = self._get_xi_ln_array(pk)
        p22 = self._get_p22_blocks(xi_ln)
        p13_kernel = self._get_p13_kernel_blocks(xi_ln)
        return p22, p13_kernel, None

    def _pkmu_true(self, k_true, mu_true, sin_true, corrs, pk_data, params):
        """Tree level plus one loop at the true (k, mu) (``sin_true`` unused).

        ``L[P] = sum_b p22_b C22_b(mu) + P sum_b p13_kernel_b C13_b(mu)``; with ``do_irres``
        tree = ``Z1 Z1' [P_nw + e^{-D} (1 + D) P_w]``, loop = ``L[P_nw] + e^{-D} (L[P] - L[P_nw])``,
        ``D = k^2 Sigma^2_s``.  The b2^2 constant ``b2 b2'/2 pk_int2`` is added after, undamped.
        """
        f = params.f
        bias_a = params.bias
        bias_b = params.bias_b
        pk, pk_nw, pk_w, damp = self._linear_at(k_true, mu_true, f, corrs, pk_data)   # (nk, nmu)

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

        blocks = [corrs.p22, corrs.p13_kernel]
        if self.do_irres:
            blocks += [corrs.p22_nw, corrs.p13_kernel_nw]
        n22, n13 = corrs.p22.shape[0], corrs.p13_kernel.shape[0]
        stack = jnp.concatenate(blocks, axis=0)
        stack = spline.interp1d(jnp.log(k_true), self._logk_fft, stack)
        stack = jnp.moveaxis(stack, 0, -1)                             # (nk, nmu, nblocks)

        b1_a, b1_b = bias_a[0], bias_b[0]
        Z1_a = b1_a + f * mu_true**2
        Z1_b = b1_b + f * mu_true**2
        if corrs.pk_int is not None:
            Z3_UV_a = self._get_Z3_UV(mu_true, f, b1_a)
            Z3_UV_b = self._get_Z3_UV(mu_true, f, b1_b)
            Z1Z3_UV = (Z1_a * Z3_UV_b + Z1_b * Z3_UV_a) / 2

        def loop(offset, P, pk_int):
            p22 = stack[:, :, offset:offset + n22]
            p13_kernel = stack[:, :, offset + n22:offset + n22 + n13]
            out = jnp.sum(p22 * C22.T, axis=-1) + P * jnp.sum(p13_kernel * C13.T, axis=-1)
            if pk_int is not None:
                out = out + k_true**2 * P * pk_int * Z1Z3_UV
            return out

        loop_full = loop(0, pk, corrs.pk_int)
        if self.do_irres:
            loop_nw = loop(n22 + n13, pk_nw, corrs.pk_int_nw)
            e_damp = jnp.exp(-damp)
            pkmu = (Z1_a * Z1_b) * (pk_nw + e_damp * (1 + damp) * pk_w)
            pkmu = pkmu + loop_nw + e_damp * (loop_full - loop_nw)
        else:
            pkmu = (Z1_a * Z1_b) * pk + loop_full
        if not self.subtract_k0_const:
            # added after the IR combination: a constant contact term must not be damped
            pkmu = pkmu + (bias_a[1] * bias_b[1]) / 2. * corrs.pk_int2
        return pkmu

    def _merge_groups(self, idx_1, blocks_1, idx_2, blocks_2):
        """``k^3 Re[blocks]`` with the matter/bias group rows put back at ``idx_1``/``idx_2``."""
        out = jnp.zeros((idx_1.shape[0] + idx_2.shape[0], self._nfft), dtype=blocks_1.dtype)
        out = out.at[idx_1].set(blocks_1).at[idx_2].set(blocks_2)
        return (self._k**3)[None, :] * jnp.real(out)

    def _get_pkmu_terms_22(self, p_q_1, p_q_2):
        """Matrix-path P22 blocks from the decompositions of the matter and bias groups."""
        return self._merge_groups(
            self._idx_22_nu1, jnp.einsum('tnm,nj,mj->tj', self.matrices_22_nu1, p_q_1, p_q_1),
            self._idx_22_nu2, jnp.einsum('tnm,nj,mj->tj', self.matrices_22_nu2, p_q_2, p_q_2))

    def _get_pkmu_terms_13(self, p_q_1, p_q_2):
        """Matrix-path P13 kernels (blocks without their outer P(k))."""
        return self._merge_groups(
            self._idx_13_nu1, jnp.einsum('tn,nj->tj', self.matrices_13_nu1, p_q_1),
            self._idx_13_nu2, jnp.einsum('tn,nj->tj', self.matrices_13_nu2, p_q_2))

    @staticmethod
    def _get_Z3_UV(mu, f, b1):
        return - 61./315. * b1 \
            + ((- 3./5. + 2./105. * b1) * f + (- 16./35. - 1./3. * b1) * f**2) * mu**2 \
            + ((- 46./105.) * f**2 + (- 1./3.) * f**3) * mu**4

    def _get_xi_ln_array(self, array):
        """Forward transforms ``xi_ln[l, n]`` of ``array`` on ``q_padded[l]`` (upper padded band zeroed)."""
        fs = self.fftlog_settings
        hg = self._hankel
        fx = hg.pad_input(array / (2 * jnp.pi**2))
        xis = fftlog.hankel(fs.nu_pld, fx, hg.k_padded, None, self._xi_pld_u_m,
                            hg.npad, crop=False, y_pow=self._xi_pld_q_factor)
        xis_n0 = fftlog.hankel(fs.nu_pld_n0, fx, hg.k_padded, None, self._xi_pld_u_m_n0,
                               hg.npad, crop=False, y_pow=self._xi_pld_q_factor_n0)
        xis = jnp.where(self._xi_pld_n0_mask[:, None], xis_n0, xis) * hg.mask_xi
        xi_ln = jnp.zeros((5, 3, xis.shape[-1]))
        ls = self._ln_list[:, 0]
        ns = self._ln_list[:, 1]
        return xi_ln.at[ls, ns].set(xis)

    def _get_pk_ln(self, l, n, array):
        """Backward FFTlog of a source on ``q_padded[l]`` onto the core k grid."""
        hg = self._hankel
        # w_xi: end weight 1/2 at the support edge (FFTlog's m = 0 coefficient is the node sum)
        fx = array * hg.q_padded[l]**(n + 3) * hg.w_xi
        return fftlog.hankel(self.fftlog_settings.nu, fx, hg.q_padded[l], hg.k_padded,
                             hg.u_m[l], hg.npad, crop=True)

    def _get_p22_blocks(self, xi_ln):
        """P22 blocks ``(nterms, nk)`` in ``coeff_info_22`` order.

        ``B(k) = 4 pi int dq q^2 (xi_l^n)^2 j_0(kq)``; block = ``k^m B`` for m > 0 and
        ``B - C`` for m = 0, ``C = 4 pi dln sum_j w_j q_j^3 (xi_l^n)_j^2``.
        """
        hg = self._hankel

        def block(ln):
            l, n = ln[0], ln[1]
            xi = xi_ln[l, n]
            source = spline.interp1d(jnp.log(hg.q_padded[0]),
                                     jnp.log(hg.q_padded[l]), xi * xi)
            pk_ln = 4 * jnp.pi * self._get_pk_ln(0, 0, source)
            # end-weighted node sum of the same source, not B(k_min), which carries the output edge error
            c0 = 4 * jnp.pi * hg.q_grids[0].moment(source * hg.w_xi, 3)
            return pk_ln, c0

        pk_ln, c0 = jax.vmap(block)(self._ln_22)      # (nterms, nk), (nterms,)
        m = self._m_22[:, None]                        # static
        return jnp.where(m == 0, pk_ln - c0[:, None], self._k ** m * pk_ln)

    def _get_p13_kernel_blocks(self, xi_ln):
        """P13 kernel blocks ``(nterms, nk)`` in ``_lnm_13`` order: ``k^m H_l[q^2 xi_l^n]``."""

        def get_p13_kernel(term):
            l, n, m = term
            pk_ln = self._get_pk_ln(l, -1, xi_ln[l, n])  # (nk,)
            return (self._k ** m) * pk_ln
        return jax.vmap(get_p13_kernel)(self._lnm_13)  # (nterms, nk)
