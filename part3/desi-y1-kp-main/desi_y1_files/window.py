import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

try:
    from jax import numpy as jnp
    from jax.config import config; config.update('jax_enable_x64', True)
except ImportError:
    jnp = np


from pypower import utils
from pypower.utils import BaseClass


class WindowRotation(BaseClass):

    def __init__(self, wmatrix, covmatrix, attrs=None):
        self.set_wmatrix(wmatrix)
        self.set_covmatrix(covmatrix)
        self.attrs = dict(attrs or {})
        self.clear()

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def clear(self):
        self.mmatrix = np.eye(self.covmatrix.shape[0])  # default M matrix

    def set_wmatrix(self, wmatrix):
        self.kin = {proj.ell: x for proj, x in zip(wmatrix.projsin, wmatrix.xin)}
        self.kout = {proj.ell: x for proj, x in zip(wmatrix.projsout, wmatrix.xout)}
        self.wmatrix = np.array(wmatrix.value.T, dtype='f8')

        ellsin = np.concatenate([[ell] * len(x) for ell, x in self.kin.items()])
        self.mask_ellsin = {ell: ellsin == ell for ell in self.kin}
        ellsout = np.concatenate([[ell] * len(x) for ell, x in self.kout.items()])
        self.mask_ellsout = {ell: ellsout == ell for ell in self.kout}

        wm00 = self.wmatrix[np.ix_(self.mask_ellsout[0], self.mask_ellsin[0])][np.argmin(np.abs(self.kout[0] - 0.1))]  # kout = 0.1 h/Mpc
        height = np.max(wm00)  # peak height
        kmax = self.kin[0][np.argmin(np.abs(wm00 - height))]  # k at maximum
        mask_before = self.kin[0] < kmax
        mask_after = self.kin[0] >= kmax
        k1 = self.kin[0][mask_before][np.argmin(np.abs(wm00[mask_before] - height / 2.))]
        k2 = self.kin[0][mask_after][np.argmin(np.abs(wm00[mask_after] - height / 2.))]
        self.bandwidth = np.abs(k2 - k1) #/ (2. * np.sqrt(2. * np.log(2)))
        self.khalfout = {}
        for ell in self.kout:
            kin = self.kin[ell]
            wm = self.wmatrix[np.ix_(self.mask_ellsout[ell], self.mask_ellsin[ell])]
            mask_half = (wm / kin) / np.max(wm / kin, axis=1)[:, None] > 0.5
            self.khalfout[ell] = np.sum(wm * mask_half * self.kin[ell], axis=1) / np.sum(wm * mask_half, axis=1)

    def set_covmatrix(self, covmatrix):
        self.covmatrix = np.array(covmatrix)

    dtype = 'f4'

    def fit(self, Minit='momt', Mtarg=None, max_sigma_W=1000, max_sigma_R=1000, factor_diff_ell=100, csub=False, state=None):
        """Fit."""
        import jax
        import optax

        kin = np.concatenate(list(self.kin.values()))
        kout = np.concatenate(list(self.khalfout.values()))
        ellsin = np.concatenate([[ell] * len(x) for ell, x in self.kin.items()])
        ellsout = np.concatenate([[ell] * len(x) for ell, x in self.kout.items()])

        if Mtarg is None:
            Mtarg = np.eye(len(kout))

        self.csub = csub
        
        if Minit in [None, 'momt']:
            with_momt = Minit == 'momt'
            Minit = jnp.identity(len(kout), dtype=self.dtype)
            offsets = jnp.zeros((len(kout), len(kout)), dtype=self.dtype)
            max_offset = 0
            for k in range(1, 1 + max_offset):
                offsets += jnp.identity(len(kout), k=k, dtype=self.dtype) + jnp.identity(len(kout), k=-k, dtype=self.dtype)
            offsets *= ellsout[:, None] * ellsout[...]
            offsets /= np.maximum(np.sum(offsets, axis=1), 1)[:, None]
            Minit -= offsets
            if with_momt:
                mo, mt, m = [], [], []
                kincut = 0.20
                idxout = 20
                for mask_ellout in self.mask_ellsout.values():
                    rowin = self.wmatrix[mask_ellout, :][idxout, :]
                    mt.append(rowin * (kin >= kincut))
                    mo.append([row[ellsin == ell][-1] / rowin[ellsin == ell][-1] for row, ell in zip(self.wmatrix, ellsout)] * mask_ellout)
                    m.append(0.)
                #print('mt', np.sum(mt), 'mo', np.sum(mo))
                if csub:
                    Minit = (Minit, mo, mt, m)
                else:
                    Minit = (Minit, mo, mt)
        else:
            with_momt = isinstance(Minit, tuple)

        weights_wmatrix = np.empty_like(self.wmatrix)
        weights_wmatrix_denom = np.empty_like(self.wmatrix)
        eps = np.finfo(float).eps
        for io, ko in enumerate(kout):
            weights_wmatrix[io, :] = np.minimum(((kin - ko) / self.bandwidth)**2, max_sigma_W**2)
            weights_wmatrix[io, :] += factor_diff_ell * (ellsout[io] != ellsin)  # off-diagonal blocks
            #weights_wmatrix_denom[io, :] = (weights_wmatrix[io, : ] + eps) / (((kin - ko) / self.bandwidth)**2 + factor_diff_ell * (ellsout[io] != ellsin) + eps)
        
        weights_covmatrix = np.empty_like(self.covmatrix)
        weights_covmatrix_denom = np.empty_like(self.covmatrix)
        for io, ko in enumerate(kout):
            weights_covmatrix[io, :] = np.minimum(((kout - ko) / self.bandwidth)**2, max_sigma_R**2)
            weights_covmatrix[io, :] += factor_diff_ell * (ellsout[io] != ellsout)  # off-diagonal blocks
            #weights_covmatrix_denom[io, :] = (weights_covmatrix[io, :] + eps) / (((kout - ko) / self.bandwidth)**2 + factor_diff_ell * (ellsout[io] != ellsout) + eps)
        #weights_wmatrix = jax.device_put(weights_wmatrix)
        #weights_covmatrix = jax.device_put(weights_covmatrix)
        
        def softabs(x):
            return jnp.sqrt(x**2 + 1e-37)
        
        def RfromC(C):
            sig = jnp.sqrt(jnp.diag(C))
            denom = jnp.outer(sig, sig)
            return C / denom

        def loss(mmatrix):
            Wp, Cp = self.rotate(mmatrix=mmatrix)
            if with_momt: mmatrix = mmatrix[0]
            loss_W = jnp.sum(softabs(Wp * weights_wmatrix)) / jnp.sum(softabs(Wp) * (weights_wmatrix > 0))
            #loss_W = jnp.sum(softabs(Wp * weights_wmatrix)) / jnp.sum(softabs(Wp) * weights_wmatrix_denom)
            Rp = RfromC(Cp)
            loss_C = jnp.sum(softabs(Rp * weights_covmatrix)) / jnp.sum(softabs(Rp) * (weights_covmatrix > 0))
            #loss_C = jnp.sum(softabs(Rp * weights_covmatrix)) / jnp.sum(softabs(Rp) * weights_covmatrix_denom)
            loss_M = 10 * jnp.sum((jnp.sum(mmatrix, axis=1) - 1.)**2)
            #print(loss_W, loss_C, weights_wmatrix.sum(), weights_covmatrix.sum(), weights_wmatrix.shape, weights_covmatrix.shape)
            return loss_W + loss_C + loss_M

        def fit(theta, loss, init_learning_rate=1e-5, meta_learning_rate=1e-4, nsteps=100000, state=None, meta_state=None):

            self.log_info(f'Will do {nsteps} steps')
            optimizer = optax.inject_hyperparams(optax.adabelief)(learning_rate=init_learning_rate)
            meta_opt = optax.adam(learning_rate=meta_learning_rate)

            @jax.jit
            def step(theta, state):
                grads = jax.grad(loss)(theta)
                updates, state = optimizer.update(grads, state)
                theta = optax.apply_updates(theta, updates)
                return theta, state

            @jax.jit
            def outer_loss(eta, theta, state):
                # Apparently this is what inject_hyperparams allows us to do
                state.hyperparams['learning_rate'] = jnp.exp(eta)
                theta, state = step(theta, state)
                return loss(theta), (theta, state)

            # Only this jit actually matters
            @jax.jit
            def outer_step(eta, theta, meta_state, state):
                #has_aux says we're going to return the 2nd part, extra info
                grad, (theta, state) = jax.grad(outer_loss, has_aux=True)(eta, theta, state)
                meta_updates, meta_state = meta_opt.update(grad, meta_state)
                eta = optax.apply_updates(eta, meta_updates)
                return eta, theta, meta_state, state

            if state is None: state = optimizer.init(theta)
            eta = jnp.log(init_learning_rate)
            if meta_state is None: meta_state = meta_opt.init(eta)
            printstep = max(nsteps // 20, 1)
            self.log_info(f'Initial loss: {loss(theta)}')
            for i in range(nsteps):
                eta, theta, meta_state, state = outer_step(eta, theta, meta_state, state)
                if i < 2 or nsteps - i < 4 or i % printstep == 0:
                    self.log_info(f'step {i}, loss: {loss(theta)}, lr: {jnp.exp(eta)}')
            return theta, (jnp.exp(eta), meta_state, state)

        if state is None:
            self.mmatrix, self.state = fit(Minit, loss)
        else:
            self.mmatrix, self.state = fit(Minit, loss, init_learning_rate=state[0], state=state[2], meta_state=state[1])
        return self.mmatrix, self.state

    def _index_kout(self, klim):
        kout = np.concatenate(list(self.kout.values()))
        return (kout >= klim[0]) & (kout <= klim[-1])

    def rotate(self, mmatrix=None, covmatrix=None, data=None, mask_cov=None, theory=None, shotnoise=0., klim=None):
        """Return prior and precmatrix if input theory."""
        if mmatrix is None: mmatrix = self.mmatrix
        input_covmatrix = covmatrix is not None
        if not input_covmatrix: covmatrix = self.covmatrix
        with_momt = isinstance(mmatrix, tuple)
        if not hasattr(self, 'csub'):
            self.csub = isinstance(mmatrix, tuple) and len(mmatrix) > 3
        if with_momt:
            Wsub = jnp.zeros(self.wmatrix.shape, dtype=self.dtype)
            if self.csub:
                mmatrix, mo, mt, m = mmatrix
                Csub = jnp.zeros(covmatrix.shape, dtype=self.dtype)
                for mmo, mmt, mm, mask_ellout in zip(mo, mt, m, self.mask_ellsout.values()):
                    mask_mo = mask_ellout * mmo
                    Wsub += jnp.outer(mask_mo, mmt)
                    Csub[np.ix_(mask_cov, mask_cov)] += mm * jnp.outer(mask_mo, mask_mo)
            else:
                mmatrix, mo, mt = mmatrix
                Csub = 0
                for mmo, mmt, mask_ellout in zip(mo, mt, self.mask_ellsout.values()):
                    mask_mo = mask_ellout * mmo
                    Wsub += jnp.outer(mask_mo, mmt)
        else:
            Wsub = Csub = 0.
        
        def marg_precmatrix(precmatrix, mo, m):
            deriv = np.zeros((len(mo),) + precmatrix.shape[:1], dtype='f8')
            deriv[:, mask_cov if mask_cov is not None else Ellipsis] = mo
            fisher = deriv.dot(precmatrix).dot(deriv.T)
            derivp = deriv.dot(precmatrix)
            fisher += np.diag(1. / m**2)  # prior
            return precmatrix - derivp.T.dot(np.linalg.solve(fisher, derivp))
        
        def marg_covmatrix(covmatrix, mo, m):
            deriv = np.zeros((len(mo),) + covmatrix.shape[:1], dtype='f8')
            deriv[:, mask_cov if mask_cov is not None else Ellipsis] = mo
            return covmatrix + deriv.T.dot(np.diag(m**2)).dot(deriv)
            
        #print('WC', Wsub.sum(), Csub.sum())
        wmatrix_rotated = jnp.matmul(mmatrix, self.wmatrix) - Wsub
        if mask_cov is not None:
            if klim is not None:
                raise ValueError('cannot pass both mask_cov and klim')
            tmpmmatrix = np.eye(covmatrix.shape[0], dtype='f8')
            tmpmmatrix[np.ix_(mask_cov, mask_cov)] = mmatrix
            mmatrix = tmpmmatrix
        covmatrix_rotated = jnp.matmul(jnp.matmul(mmatrix, covmatrix), mmatrix.T) - Csub
        if klim is not None:
            mask_kout = self._index_kout(klim)
            wmatrix_rotated = wmatrix_rotated[mask_kout, :]
            covmatrix_rotated = covmatrix_rotated[np.ix_(mask_kout, mask_kout)]
        if data is not None:
            data = np.asarray(data).real.ravel()
            #data_rotated = np.matmul(mmatrix, data + shotnoise * self.mask_ellsout[0]) - shotnoise * self.mask_ellsout[0]
            data_rotated = np.matmul(mmatrix, data)
            if klim is not None:
                data_rotated = data_rotated[mask_kout]
                if with_momt: mo = [mmo[mask_kout] for mmo in mo]
            if theory is not None and with_momt:
                theory = np.asarray(theory).real.ravel()
                precmatrix = np.linalg.inv(covmatrix_rotated)
                deriv = np.array(mo)
                derivp = deriv.dot(precmatrix)
                fisher = derivp.dot(deriv.T)
                self.marg_prior_mo = m = np.linalg.solve(fisher, derivp.dot(data_rotated + shotnoise * self.mask_ellsout[0][mask_kout if klim is not None else Ellipsis] - np.matmul(wmatrix_rotated, theory + shotnoise * self.mask_ellsin[0])))
                offset = np.dot(m, mo)
                precmatrix = marg_precmatrix(precmatrix, mo, m)
                return wmatrix_rotated, covmatrix_rotated, data_rotated, m, offset, precmatrix
            return wmatrix_rotated, covmatrix_rotated, data_rotated
        if input_covmatrix and with_momt and hasattr(self, 'marg_prior_mo'):
            return wmatrix_rotated, covmatrix_rotated, marg_precmatrix(np.linalg.inv(covmatrix_rotated), mo, self.marg_prior_mo)
        return wmatrix_rotated, covmatrix_rotated

    @property
    def with_momt(self):
        return isinstance(self.mmatrix, tuple)

    def plot_wmatrix(self, k=0.1, ells=None, fn=None):
        k = np.ravel(k)
        if ells is None: ells = sorted(self.kout.keys())

        wmatrix_rotated = self.rotate()[0]
        alphas = np.linspace(1, 0.2, len(k))
        fig, lax = plt.subplots(len(ells), len(ells), sharey=True, figsize=(8, 6), squeeze=False)

        for iin, ellin in enumerate(ells):
            for iout, ellout in enumerate(ells):
                ax = lax[iout][iin]
                for ik, kk in enumerate(k):
                    indexout = np.abs(self.kout[ellout] - kk).argmin()
                    # Indices in approximate window matrix
                    norm = self.kin[ellin]
                    ax.semilogy(self.kin[ellin], np.abs(self.wmatrix[np.ix_(self.mask_ellsout[ellout], self.mask_ellsin[ellin])][indexout, :] / norm), alpha=alphas[ik], color='C0', label=r'$W$' if ik == 0 else '')
                    ax.semilogy(self.kin[ellin], np.abs(wmatrix_rotated[np.ix_(self.mask_ellsout[ellout], self.mask_ellsin[ellin])][indexout, :] / norm), alpha=alphas[ik], color='C1', label=r'$W^{\prime}$' if ik == 0 else '')
                ax.set_title(r'$\ell_{{\mathrm{{t}}}} = {:d} \times \ell_{{\mathrm{{o}}}} = {:d}$'.format(ellin, ellout))
                ax.set_ylim(1e-4, 2)
                ax.grid(True)
                if iout == len(ells) - 1: ax.set_xlabel(r'$k_{\mathrm{t}}$ [$h/\mathrm{Mpc}$]')
                if iin == iout == 0: lax[iout][iin].legend()
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        if fn is not None:
            utils.savefig(fn, fig=fig)
    
    def plot_covmatrix(self, ells=None, fn=None, corrcoef=True, covmatrix=None, norm=None):
        if ells is None: ells = sorted(self.kout.keys())

        covmatrix = self.rotate(covmatrix=covmatrix)[1]
        
        if corrcoef:
            stddev = np.sqrt(np.diag(covmatrix).real)
            covmatrix = covmatrix / stddev[:, None] / stddev[None, :]

        nells = len(ells)
        figsize = (6,) * 2
        xextend = 0.8
        fig, lax = plt.subplots(nrows=nells, ncols=nells, sharex=False, sharey=False, figsize=(figsize[0] / xextend, figsize[1]), squeeze=False)
        norm = norm or Normalize(vmin=covmatrix.min(), vmax=covmatrix.max())

        for ill1, ell1 in enumerate(ells):
            for ill2, ell2 in enumerate(ells):
                ax = lax[nells - 1 - ill1][ill2]
                k1, k2 = self.kout[ell1], self.kout[ell2]
                mask1, mask2 = self.mask_ellsout[ell1], self.mask_ellsout[ell2]
                mesh = ax.pcolor(k1, k2, covmatrix[np.ix_(mask1, mask2)].T, norm=norm, cmap=plt.get_cmap('RdBu'))
                if ill1 > 0: ax.xaxis.set_visible(False)
                else: ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
                if ill2 > 0: ax.yaxis.set_visible(False)
                else: ax.set_ylabel(r'$k$ [$h/\mathrm{Mpc}$]')
                text = r'$\ell = {:d} \times \ell = {:d}$'.format(ell1, ell2)
                ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black')
                ax.grid(False)

        fig.subplots_adjust(right=xextend)
        cbar_ax = fig.add_axes([xextend + 0.05, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(mesh, cax=cbar_ax)
        if fn is not None:
            utils.savefig(fn, fig=fig)
            plt.close(fig)

    def plot_compactness(self, frac=0.95, ells=None, klim=(0.02, 0.2), fn=None):
        if ells is None:
            ells = sorted(self.kout.keys())
            ells = [ells[:i + 1] for i in range(len(ells))]

        wmatrix_rotated = self.rotate()[0]
        alphas = np.linspace(0.4, 1., len(ells))
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), squeeze=True)

        def compactness(wm, ells, frac):
            weights_bf = sum(np.cumsum(np.abs(wm[np.ix_(self.mask_ellsout[ellout], self.mask_ellsin[ellin])]), axis=-1) for ellin in ells for ellout in ells)
            weights_tot = weights_bf[:, -1]
            iktmax = np.argmax(weights_bf / weights_tot[:, None] >= frac, axis=-1)
            return self.kin[0][iktmax]

        for ill, ells in enumerate(ells):
            ax.plot(self.kout[0], compactness(self.wmatrix, ells=ells, frac=frac), color='C0', alpha=alphas[ill], label=r'$\ell = {}$'.format(tuple(ells)))
            ax.plot(self.kout[0], compactness(wmatrix_rotated, ells=ells, frac=frac), color='C1', alpha=alphas[ill])

        for kk in (klim or []): ax.axvline(kk, ls=':', color='k')

        ax.set_xlabel(r'$k_{\mathrm{o}}$ [$h/\mathrm{Mpc}$]')
        ax.set_ylabel(r'$k_{\mathrm{t}}$ [$h/\mathrm{Mpc}$]')
        ax.legend()
        if fn is not None:
            utils.savefig(fn, fig=fig)
            plt.close(fig)

    def plot_rotated(self, data, shotnoise=0., ells=None, klim=(0.02, 0.2), fn=None):

        if ells is None:
            ells = sorted(self.kout.keys())

        data_rotated = self.rotate(data=data, shotnoise=shotnoise)[2]
        fig, lax = plt.subplots(1, len(ells), figsize=(8, 3), sharey=False, squeeze=False)
        lax = lax.ravel()

        for ill, ell in enumerate(ells):
            ax = lax[ill]
            ax.plot(self.kout[ell], self.kout[ell] * data[self.mask_ellsout[ell]], color='C0', label=r'$P_{\mathrm{o}}(k)$')
            ax.plot(self.kout[ell], self.kout[ell] * data_rotated[self.mask_ellsout[ell]], color='C1', label=r'$P_{\mathrm{o}}^{\prime}(k)$')
            ax.set_title(r'$\ell = {}$'.format(ell))
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            ax.set_xlim(klim)

        lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.legend()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        if fn is not None:
            utils.savefig(fn, fig=fig)
            plt.close(fig)

    def plot_validation(self, data, theory, ells=None, klim=None, covmatrix=None, shotnoise=0., marg_shotnoise=False, nobs=1, fn=None):

        if ells is None:
            ells = sorted(self.kout.keys())
    
        fig, lax = plt.subplots(2, len(ells), figsize=(3 * len(ells), 4), sharey=False, sharex=True, gridspec_kw={'height_ratios': [4, 2]}, squeeze=False)
        rotate = self.rotate(data=data, theory=theory, covmatrix=covmatrix, klim=klim, shotnoise=shotnoise)
        precmatrix_rotated = None
        try:
            wmatrix_rotated, covmatrix_rotated, data_rotated = rotate
            offset = 0.
        except ValueError:
            wmatrix_rotated, covmatrix_rotated, data_rotated, m, offset, precmatrix_rotated = rotate

        #np.save('precm.npy', precmatrix_rotated)
        std = np.diag(covmatrix_rotated)**0.5
        if precmatrix_rotated is not None:
            std = np.diag(np.linalg.inv(precmatrix_rotated))**0.5
        std /= nobs**0.5
        kout = np.concatenate(list(self.kout.values()))
        if klim is not None:
            mask_kout = (kout >= klim[0]) & (kout <= klim[-1])
        else:
            mask_kout = np.ones(len(kout), dtype='?')
        data = np.asarray(data).real.ravel()
        if klim is not None:
            data = data[mask_kout]
        data_rotated -= offset
        theory_rotated = np.matmul(wmatrix_rotated, theory + shotnoise * self.mask_ellsin[0]) - shotnoise * self.mask_ellsout[0][mask_kout]
        #theory_rotated = np.matmul(wmatrix_rotated, theory)
        #theory_rotated = theory
        
        if marg_shotnoise:  # shotnoise free
            precmatrix = np.linalg.inv(covmatrix_rotated)
            deriv = np.matmul(wmatrix_rotated, self.mask_ellsin[0])[None, :]
            derivp = deriv.dot(precmatrix)
            fisher = derivp.dot(deriv.T)
            shotnoise_value = np.linalg.solve(fisher, derivp.dot(data_rotated - theory_rotated))
            theory_rotated_shotnoise = theory_rotated + shotnoise_value.dot(deriv)

        for ill, ell in enumerate(ells):
            color = 'C{}'.format(ill)
            ax = lax[0][ill]
            kk = kout[self.mask_ellsout[ell] & mask_kout]
            mask_out = self.mask_ellsout[ell][mask_kout]
            ax.errorbar(kk, kk * data_rotated[mask_out], kk * std[mask_out], color=color, marker='.', ls='', label=r'$P_{\mathrm{o}}(k)$')
            ##ax.errorbar(kk, kk * data[mask_out], kk * std[mask_out], color=color, marker='.', ls='', label=r'$P_{\mathrm{o}}(k)$')
            ax.plot(kk, kk * np.interp(kk, self.kin[ell], theory[self.mask_ellsin[ell]]), color=color, ls=':', label=r'$P_{\mathrm{t}}(k)$')
            ax.plot(kk, kk * theory_rotated[mask_out], color=color, label=r'$W(k, k^{\prime}) P_{\mathrm{t}}(k^{\prime})$')
            if marg_shotnoise:
                ax.plot(kk, kk * theory_rotated_shotnoise[mask_out], color=color, ls='--', label=r'$W(k, k^{\prime}) (P_{\mathrm{t}}(k^{\prime}) + N)$')
            ax.set_title(r'$\ell = {}$'.format(ell))
            ax.set_xlim(klim)
            ax.grid(True)
            ax = lax[1][ill]
            ax.plot(kk, ((theory_rotated_shotnoise if marg_shotnoise else theory_rotated)[mask_out] - data_rotated[mask_out]) / std[mask_out], color=color)
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            ax.set_ylim(-2., 2.)
            ax.grid(True)

        lax[0][0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[0][0].legend()
        lax[1][0].set_ylabel(r'$\Delta P_{\ell}(k) / \sigma$')
        fig.align_ylabels()
        if fn is not None:
            utils.savefig(fn, fig=fig)
            plt.close(fig)
            
    def __getstate__(self):
        state = {name: getattr(self, name) for name in ['kin', 'kout', 'mask_ellsin', 'mask_ellsout', 'khalfout', 'wmatrix', 'covmatrix', 'mmatrix', 'marg_precmatrix', 'marg_prior_mo', 'marg_theory_offset', 'state', 'csub', 'attrs'] if hasattr(self, name)}
        # convert jax to numpy arrays
        mmatrix = []
        for im, m in enumerate(state['mmatrix']):
            if isinstance(m, list):
                m = [np.array(m) for m in m]
            else:
                m = np.array(m)
            mmatrix.append(m)
        state['mmatrix'] = tuple(mmatrix)
        state.pop('state')
        return state


class WindowRIC(BaseClass):
    
    r"""To be tuned on mocks without :math:`\theta`-cut."""

    def __init__(self, wmatrix, power_noric, power_ric, covmatrix=None, attrs=None):
        self.set_wmatrix(wmatrix)
        self.set_power(power_noric, power_ric)
        self.set_covmatrix(covmatrix)
        self.attrs = dict(attrs or {})
        self.clear()

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def set_wmatrix(self, wmatrix):
        self.kin = {proj.ell: x for proj, x in zip(wmatrix.projsin, wmatrix.xin)}
        self.kout = {proj.ell: x for proj, x in zip(wmatrix.projsout, wmatrix.xout)}
        self.wmatrix = jnp.array(wmatrix.value.T, dtype='f8')

        ellsin = np.concatenate([[ell] * len(x) for ell, x in self.kin.items()])
        self.mask_ellsin = {ell: ellsin == ell for ell in self.kin}
        ellsout = np.concatenate([[ell] * len(x) for ell, x in self.kout.items()])
        self.mask_ellsout = {ell: ellsout == ell for ell in self.kout}
        Wij = self.wmatrix[np.ix_(self.mask_ellsout[0], self.mask_ellsin[0])]
        self.W0j = Wij[0, :]
        self.Wi0 = Wij[:, 0]
        self.W00 = Wij[0, 0]

    def set_power(self, power_noric, power_ric):
        self.power_noric = np.asarray(power_noric, dtype=self.dtype)
        self.power_noric.shape = (len(self.power_noric), -1)
        self.power_ric = np.asarray(power_ric, dtype=self.dtype)
        self.power_ric.shape = (len(self.power_noric), -1)
        
    def set_covmatrix(self, covmatrix=None):
        if covmatrix is None:
            self.covmatrix = np.diag(np.var(self.power_ric - self.power_noric, axis=0, ddof=1)) #/ len(self.power_ric)
        else:
            self.covmatrix = np.array(covmatrix)

    def clear(self):
        ellsin, ellsout = list(self.kin), list(self.kout)
        amatrix = jnp.zeros((len(ellsin), len(ellsout)), dtype=self.dtype)
        #amatrix = amatrix.at[ellsin.index(0), ellsout.index(0)].set(1e-4)
        smatrix = [i * jnp.ones(amatrix.shape, dtype=self.dtype) for i in [1.]]
        #smatrix = jnp.ones_like(amatrix)
        self.asmatrix = [amatrix, smatrix]
        #K = self.kernel()
        #self.asmatrix[0] = self.asmatrix[0] * self.W00 / K.sum()

    def logprior(self, asmatrix=None):
        if asmatrix is None: asmatrix = self.asmatrix
        amatrix, smatrix = asmatrix
        return 0.#5 * jnp.sum(smatrix[-1]**2) / smatrix[-1].size
    
    def kernel(self, asmatrix=None):
        if isinstance(asmatrix, str) and asmatrix == 'gic':
            wmat = jnp.zeros_like(self.wmatrix.T)
            wmat = wmat.at[np.flatnonzero(self.mask_ellsin[0])[0], np.flatnonzero(self.mask_ellsout[0])[0]].set(1.) / self.W00
        else:
            if asmatrix is None: asmatrix = self.asmatrix
            amatrix, smatrix = asmatrix
            print(amatrix, smatrix)
            #wmat = jnp.block([[amatrix[illin, illout] * jnp.exp(-(self.kout[ellout][None, :]**2 + self.kin[ellin][:, None]**2) /  sk0**2 * smatrix[illin, illout]) for illin, ellin in enumerate(self.kin)] for illout, ellout in enumerate(self.kout)])
            sigma_kout = sigma_kin = 1e-2
            def softabs(x):
                return jnp.sqrt(x**2 + 1e-12)
            wmat = []
            for illout, ellout in enumerate(self.kout):
                row = []
                for illin, ellin in enumerate(self.kin):
                    #sk2 = (self.kout[ellout][None, :] / sigma_kout)**2 + (self.kin[ellin][:, None] / sigma_kin)**2
                    #sin, sout = smatrix[0][illin, illout]**2, smatrix[1][illin, illout]**2
                    #sk2 = (self.kout[ellout][None, :] / sigma_kout)**2 * sin  + (self.kin[ellin][:, None] / sigma_kin)**2 * sout
                    sout = smatrix[0][illin, illout]
                    sin = smatrix[0][illin, illout]
                    skout = self.kout[ellout][None, :] / sigma_kout * sout
                    skin = self.kin[ellout][:, None] / sigma_kin * sin
                    rho = 0. #0.5 * 2 / jnp.pi * jnp.arctan(smatrix[2][illin, illout])
                    sk2 = skout**2 + skin**2 - 2. * skout * skin * rho
                    sk2 /= (1 - rho**2)
                    #sk2 = ((self.kout[ellout][None, :] * sout)**2 + (self.kin[ellin][:, None] * sin)**2) #/ coeff
                    #sk2 -= self.kout[ellout][None, :] * self.kin[ellin][:, None] * sout * sin * 2 * rho / coeff # * 2
                    tmp = jnp.exp(-sk2)
                    #print(sk2)
                    #print(np.isfinite(tmp).all())
                    #tmp = jnp.zeros_like(tmp)
                    #tmp = tmp.at[0, 0].set(1.)
                    #tmp *= amatrix[illin, illout] / (self.W0j.dot(tmp).dot(self.Wi0) / self.W00)
                    tmp *= amatrix[illin, illout] #/ tmp.sum()
                    row.append(tmp)
                wmat.append(row)
            wmat = jnp.block(wmat)
        return wmat
    
    def icwmatrix(self, *args, **kwargs):
        K = self.kernel(*args, **kwargs)
        return self.wmatrix.dot(K).dot(self.wmatrix)

    def export(self, wmatrix, wmatrix_cut=None):
        if wmatrix_cut is None: wmatrix_cut = wmatrix
        di = self.__dict__
        self.set_wmatrix(wmatrix)
        K = self.kernel(asmatrix=None)
        self.__dict__.update(di)
        wmatrix = wmatrix.deepcopy()
        wmatrix.value = wmatrix_cut.value - wmatrix_cut.value.T.dot(K.dot(self.wmatrix)).T
        return wmatrix

    dtype = 'f8'

    def fit(self, init=None, state=None):
        """Fit."""
        import jax
        import optax
        invcov = jnp.linalg.inv(self.covmatrix)
        W0j = self.wmatrix[[np.flatnonzero(mask_out)[0] for mask_out in self.mask_ellsout.values()], :]
        factor_0 = 1. / np.sum(W0j**2)
        Kgic = self.kernel(asmatrix='gic')

        if init is None:
            init = self.asmatrix

        def loss(args):
            K = self.kernel(args) #- Kgic
            WricP = jnp.dot(jnp.dot(self.wmatrix, K), self.power_noric.T).T
            diff = self.power_noric - self.power_ric - WricP
            loss_K = jnp.sum(diff.dot(invcov) * diff) / diff.size  # normalize chi2 to 1.
            # Enforce Wric_0j = W_0j
            loss_p = -2. * self.logprior(args)
            #Wric0j = jnp.dot(jnp.dot(W0j, K), self.wmatrix)
            #loss_0 = factor_0 * jnp.sum(Wric0j**2)  #jnp.sum((Wric0j - W0j)**2)
            return loss_K + loss_p #+ loss_0

        """
        Wric0j = jnp.dot(jnp.dot(W0j, Kgic), self.wmatrix)
        print(loss(init), jnp.sum((Wric0j - W0j)**2), Wric0j, W0j)
        exit()
        """

        def fit(theta, loss, init_learning_rate=1e-5, meta_learning_rate=1e-4, nsteps=50000, state=None, meta_state=None):

            self.log_info(f'Will do {nsteps} steps')
            optimizer = optax.inject_hyperparams(optax.adabelief)(learning_rate=init_learning_rate)
            meta_opt = optax.adam(learning_rate=meta_learning_rate)

            @jax.jit
            def step(theta, state):
                grads = jax.grad(loss)(theta)
                updates, state = optimizer.update(grads, state)
                theta = optax.apply_updates(theta, updates)
                return theta, state

            @jax.jit
            def outer_loss(eta, theta, state):
                # Apparently this is what inject_hyperparams allows us to do
                state.hyperparams['learning_rate'] = jnp.exp(eta)
                theta, state = step(theta, state)
                return loss(theta), (theta, state)

            # Only this jit actually matters
            @jax.jit
            def outer_step(eta, theta, meta_state, state):
                # has_aux says we're going to return the 2nd part, extra info
                grad, (theta, state) = jax.grad(outer_loss, has_aux=True)(eta, theta, state)
                meta_updates, meta_state = meta_opt.update(grad, meta_state)
                eta = optax.apply_updates(eta, meta_updates)
                return eta, theta, meta_state, state

            if state is None: state = optimizer.init(theta)
            eta = jnp.log(init_learning_rate)
            if meta_state is None: meta_state = meta_opt.init(eta)
            printstep = max(nsteps // 10, 1)
            self.log_info(f'Initial loss: {loss(theta)}')
            for i in range(nsteps):
                eta, theta, meta_state, state = outer_step(eta, theta, meta_state, state)
                if i < 2 or nsteps - i < 4 or i % printstep == 0:
                    self.log_info(f'step {i}, loss: {loss(theta)}, lr: {jnp.exp(eta)}')
            return theta, (jnp.exp(eta), meta_state, state)

        if state is None:
            self.asmatrix, self.state = fit(init, loss)
        else:
            self.asmatrix, self.state = fit(init, loss, init_learning_rate=state[0], state=state[2], meta_state=state[1])
        return self.asmatrix, self.state
    
    def plot_wmatrix(self, ells=None, klim=None, ric=True, fn=None, norm=None):
        if ells is None: ells = sorted(self.kout.keys())
        
        if ric == 'kernel': wmatrix = self.kernel()
        elif ric: wmatrix = self.icwmatrix()
        else: wmatrix = self.wmatrix

        nells = len(ells)
        figsize = (6,) * 2
        xextend = 0.8
        fig, lax = plt.subplots(nrows=nells, ncols=nells, sharex=False, sharey=False, figsize=(figsize[0] / xextend, figsize[1]), squeeze=False)
        norm = norm or Normalize(vmin=wmatrix.min(), vmax=wmatrix.max())

        for ill1, ell1 in enumerate(ells):
            for ill2, ell2 in enumerate(ells):
                ax = lax[nells - 1 - ill1][ill2]
                k1, k2 = self.kout[ell1], self.kin[ell2]
                mask1, mask2 = self.mask_ellsout[ell1].copy(), self.mask_ellsin[ell2].copy()
                if klim is not None:
                    mask_k1 = (k1 >= klim[0]) & (k1 <= klim[-1])
                    mask_k2 = (k2 >= klim[0]) & (k2 <= klim[-1])
                    mask1[mask1] &= mask_k1
                    mask2[mask2] &= mask_k2
                    k1, k2 = k1[mask_k1], k2[mask_k2]
                mesh = ax.pcolor(k1, k2, wmatrix[np.ix_(mask1, mask2)].T, norm=norm, cmap=plt.get_cmap('RdBu'))
                if ill1 > 0: ax.xaxis.set_visible(False)
                else: ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
                if ill2 > 0: ax.yaxis.set_visible(False)
                else: ax.set_ylabel(r'$k$ [$h/\mathrm{Mpc}$]')
                text = r'$\ell = {:d} \times \ell = {:d}$'.format(ell1, ell2)
                ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black')
                ax.grid(False)

        fig.subplots_adjust(right=xextend)
        cbar_ax = fig.add_axes([xextend + 0.05, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(mesh, cax=cbar_ax)
        if fn is not None:
            utils.savefig(fn, fig=fig)
            plt.close(fig)

    def plot_power(self, ells=None, klim=None, covmatrix=None, fn=None):

        if ells is None:
            ells = sorted(self.kout.keys())
    
        fig, lax = plt.subplots(2, len(ells), figsize=(3 * len(ells), 4), sharey=False, sharex=True, gridspec_kw={'height_ratios': [4, 2]}, squeeze=False)

        kout = np.concatenate(list(self.kout.values()))
        if klim is not None:
            mask_kout = (kout >= klim[0]) & (kout <= klim[-1])
        else:
            mask_kout = np.ones(len(kout), dtype='?')

        if covmatrix is None: covmatrix = self.covmatrix
        std = np.diag(covmatrix)**0.5

        for ill, ell in enumerate(ells):
            color = 'C{}'.format(ill)
            ax = lax[0][ill]
            mask_out = self.mask_ellsout[ell] & mask_kout
            kk = kout[mask_out]
            power_ric = np.mean(self.power_ric, axis=0)
            power_noric = np.mean(self.power_noric, axis=0)
            #ax.plot(kk, power_ric[mask_out], color=color, linestyle='-', label=r'$P_{\mathrm{ric}}(k)$')
            #ax.plot(kk, power_noric[mask_out], color=color, linestyle='--', label=r'$P_{\mathrm{noric}}(k)$')
            ax.plot(kk, power_ric[mask_out] - power_noric[mask_out], color=color, linestyle='-', label=r'$P_{\mathrm{ric}}(k) - P_{\mathrm{noric}}(k)$')
            ax.set_title(r'$\ell = {}$'.format(ell))
            ax.set_xlim(klim)
            ax.grid(True)
            ax = lax[1][ill]
            diff = power_ric - power_noric
            ax.plot(kk, diff[mask_out] / std[mask_out], color=color)
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            ax.set_ylim(-2., 2.)
            ax.grid(True)

        lax[0][0].set_ylabel(r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
        lax[0][0].legend()
        lax[1][0].set_ylabel(r'$\Delta P_{\ell}(k) / \sigma$')
        fig.align_ylabels()
        if fn is not None:
            utils.savefig(fn, fig=fig)
            plt.close(fig)
            
    def plot_validation(self, ells=None, klim=None, covmatrix=None, fn=None):

        if ells is None:
            ells = sorted(self.kout.keys())
    
        fig, lax = plt.subplots(2, len(ells), figsize=(3 * len(ells), 4), sharey=False, sharex=True, gridspec_kw={'height_ratios': [4, 2]}, squeeze=False)
        fig.subplots_adjust(wspace=0.3)

        kout = np.concatenate(list(self.kout.values()))
        if klim is not None:
            mask_kout = (kout >= klim[0]) & (kout <= klim[-1])
        else:
            mask_kout = np.ones(len(kout), dtype='?')

        if covmatrix is None: covmatrix = self.covmatrix
        std = np.diag(covmatrix)**0.5
        K = self.kernel() #- self.kernel(asmatrix='gic')
        #print('K', K)
        #print(self.kernel())
        #print(self.kernel(asmatrix='gic'))
        #print(self.wmatrix.shape, K.shape, np.abs(K).max(), self.power_noric.shape)
        theory_diff = - self.wmatrix.dot(K).dot(np.mean(self.power_noric, axis=0))
        #print('D', theory_diff)

        for ill, ell in enumerate(ells):
            color = 'C{}'.format(ill)
            ax = lax[0][ill]
            mask_out = self.mask_ellsout[ell] & mask_kout
            kk = kout[mask_out]
            diff = np.mean(self.power_ric - self.power_noric, axis=0)
            std_diff = np.std(self.power_ric - self.power_noric, axis=0, ddof=1) / len(self.power_noric)**0.5
            #ax.errorbar(kk, diff[mask_out], std_diff[mask_out], color=color, marker='.', ls='none', label=r'$P_{\mathrm{ric}}(k) - P_{\mathrm{noric}}(k)$')
            #ax.plot(kk, diff[mask_out], color=color, linestyle=':', label=r'$P_{\mathrm{ric}}(k) - P_{\mathrm{noric}}(k)$')
            ax.fill_between(kk, diff[mask_out] - std_diff[mask_out], diff[mask_out] + std_diff[mask_out], alpha=0.4, color=color)
            ax.plot(kk, theory_diff[mask_out], color=color, label=r'$W_{\mathrm{ric}}(k, k^{\prime}) P_{\mathrm{noric}}(k^{\prime})$')
            ax.set_title(r'$\ell = {}$'.format(ell))
            ax.set_xlim(klim)
            ax.grid(True)
            ax = lax[1][ill]
            ax.plot(kk, (theory_diff[mask_out] - diff[mask_out]) / std[mask_out], color=color)
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            ax.set_ylim(-2., 2.)
            ax.grid(True)

        lax[0][0].set_ylabel(r'$\Delta P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
        lax[0][0].legend()
        lax[1][0].set_ylabel(r'$\Delta P_{\ell}(k) / \sigma$')
        fig.align_ylabels()
        if fn is not None:
            utils.savefig(fn, fig=fig)
            plt.close(fig)

    def plot_validation_gic(self, fn=None):
        from cosmoprimo.fiducial import DESI
        cosmo = DESI()
        fo = cosmo.get_fourier()
        z = 1.
        pk = fo.pk_interpolator().to_1d(z=z)
        f = fo.sigma8_z(z, of='theta_cb') / fo.sigma8_z(z, of='delta_cb')
        b1 = 2.
        beta = f / b1
        theory = []
        for ellin, kin in self.kin.items():
            kaiser = {0: (1. + 2. / 3. * beta + 1. / 5. * beta**2), 2: (4. / 3. * beta + 4. / 7. * beta**2), 4: 8. / 35 * beta**2}
            theory.append(kaiser[ellin] * b1**2 * pk(self.kin[ellin]))
        theory = np.concatenate(theory)
        icwmatrix = self.icwmatrix()
        wmatrix = self.wmatrix
        theory_w = wmatrix.dot(theory)
        theory_wric = theory_w - icwmatrix.dot(theory)
        ax = plt.gca()
        ax.plot([], [], color='k', linestyle='-', label='theory')
        ax.plot([], [], color='k', linestyle='--', label='W * theory')
        ax.plot([], [], color='k', linestyle=':', label='(W - Wric) * theory')
        for illin, (ell, kin) in enumerate(self.kin.items()):
            color = 'C{:d}'.format(illin)
            ax.plot([], [], color=color, label='$\ell = {:d}$'.format(ell))
            ax.plot(kin, theory[self.mask_ellsin[ell]], color=color, linestyle='-')
            if ell in self.mask_ellsout:
                ax.plot(self.kout[ell], theory_w[self.mask_ellsout[ell]], color=color, linestyle='--')
                ax.plot(self.kout[ell], theory_wric[self.mask_ellsout[ell]], color=color, linestyle=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        ax.set_ylabel(r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
        ax.legend()
        if fn is not None:
            fig = plt.gcf()
            utils.savefig(fn, fig=fig)
            plt.close(fig)

    def __getstate__(self):
        return {name: getattr(self, name) for name in ['kin', 'kout', 'mask_ellsin', 'mask_ellsout', 'wmatrix', 'covmatrix', 'asmatrix', 'state', 'attrs'] if hasattr(self, name)}