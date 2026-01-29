import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from .base import BaseStyle
from . import utils
from .kp3 import is_sequence, default_colors


def load_chain(fi, burnin=0.5):
    from desilike.samples import Chain
    chains = [Chain.load(ff).remove_burnin(burnin) for ff in fi]
    chain = chains[0].concatenate(chains)
    eta = 1. / 3.
    if 'qpar' in chain and 'qper' in chain:
        chain.set((chain['qpar']**eta * chain['qper']**(1. - eta)).clone(param=dict(name='qiso', derived=True, latex=r'q_{\rm iso}')))
        chain.set((chain['qpar'] / chain['qper']).clone(param=dict(name='qap', derived=True, latex=r'q_{\rm ap}')))
    if 'qiso' in chain and 'qap' in chain:
        chain.set((chain['qiso'] * chain['qap']**(1. - eta)).clone(param=dict(name='qpar', derived=True, latex=r'q_{\parallel}')))
        chain.set((chain['qiso'] * chain['qap']**(-eta)).clone(param=dict(name='qper', derived=True, latex=r'q_{\perp}')))
    return chain


def get_center_interval(data, params=['qiso']):
    if not utils.is_sequence(params):
        params = [params]
    if utils.is_sequence(data):
        chain = load_chain(data, burnin=0.5)
        center = chain.mean(params)
        interval = list(zip(*[chain.std(params)] * 2))
        return center, interval, data[0].options
    from desilike.samples import Profiles
    profiles = Profiles.load(data)
    index = profiles.bestfit.logposterior.argmax()
    center = [profiles.bestfit[param][index] for param in params]
    interval = [np.abs(profiles.interval[param]) for param in params]
    return center, interval, data.options


class KP4Style(BaseStyle):
    """
    Context for KP4 style.
    To be used as a context:

    .. code-block:: python

        with KP4Style() as style:

            style.plot_bao_diagram(data, fn='my_plot.png')

    """
    def __init__(self, **kwargs):
        """
        Initialize :class:`KP4Style`.

        Parameters
        ----------
        kwargs : dict
            Either arguments for ``mpl.rcParams``, or attributes for this class.
            Passed to :meth:`update`.
        """
        # Dictionary to contain default colors
        self.colors = default_colors
        # Dictionaries for plotting different ell
        self.points = {0: 'o', 2: 'x', 4: 'd'}
        self.linestyles = {0: '-', 2: '--', 4: '-.'}
        self.alphas = {0: 1., 2: 2. / 3., 4: 1. / 3.}
        self._rcparams = mpl.rcParams.copy()
        self.update(**kwargs)
        self.default_params = ['qpar', 'qper', 'qiso', 'qap']

    def get_tracer_label(self, tracer):
        for name in ['BGS', 'LRG+ELG', 'LRG', 'ELG', 'QSO']:
            if name in tracer:
                return name

    @utils.plotter
    def plot_legend(self, fig=None):
        """
        Plot legend.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``1`` axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if fig is None:
            fig, ax = plt.subplots(figsize=(4, 1.5))
        else:
            ax = fig.axes[0]

        fig_tmp, ax_tmp = plt.subplots()
        for key in self.colors:
            if isinstance(key, str): continue  # to keep only (tracer, zrange) pairs
            ax_tmp.hist(np.ones(1), color=self.colors[key], label='{0}, ${1[0]:.1f} < z < {1[1]:.1f}$'.format(*key))
        ax.legend(*ax_tmp.get_legend_handles_labels(), ncol=3, loc='center')
        plt.close(fig_tmp)
        # Hide the axes frame and the x/y labels
        ax.axis('off')
        return fig

    @utils.plotter
    def plot_bao_diagram(self, data, labels=None, apmode='qparqper', qlim=None, figsize=None, fig=None):
        """
        Plot BAO diagram.

        Parameters
        ----------
        data : BaseFileEntry, list
            :class:`Chain` file(s).
            Can be a list of tuples of such files, for e.g. error bar comparisons.
        
        labels : list, default=None
            Labels for error bar comparisons.

        apmode : str, default='qparqper'
            'qparqper' for radial and transverse BAO measurements.
            'qisoqap' for isotropic and anisotropic BAO measurements.
            'qiso' for isotropic BAO measurements.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``1 + len(self.ells)`` axes.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from desi_y1_files.cosmo_tools import get_bao_params
        params = get_bao_params(apmode)
        label_bao = {'qpar': r'$\alpha_{\parallel}$', 'qper': r'$\alpha_{\perp}$',
                     'qiso': r'$\alpha_{\mathrm{iso}}$', 'qap': r'$\alpha_{\mathrm{ap}}$'}
        for dd in data:
            dd = list(dd) if utils.is_sequence(dd) else [dd]
            dd = load_chain(dd[0])
            params = dd.params(name=[str(param) for param in params] if params is not None else self.default_params, varied=True)
            break

        if fig is None:
            nrows = len(params)
            fig, lax = plt.subplots(nrows, figsize=figsize, sharex=True, sharey=False, squeeze=False)
            lax = np.ravel(lax)
        else:
            lax = fig.axes

        labels = list(labels) if utils.is_sequence(labels) else [labels]
        xmain = np.arange(len(data))
        ids, colors = [], []
        for imain, dd in enumerate(data):
            dd = list(dd) if utils.is_sequence(dd) else [dd]
            xaux = np.linspace(-0.15, 0.15, len(dd))
            key = dd[0].options['tracer'][0], tuple(dd[0].options['zrange'][0])
            ids.append('{0}, {1[0]:.1f} < z < {1[1]:.1f}'.format(key[0][:3], key[1]))
            colors.append(self.colors[key])
            for iaux, dd in enumerate(dd):
                label = labels[min(iaux, len(labels) - 1)]
                dd = load_chain(dd)
                zz = dd.attrs['zeff']
                mean, std = dd.mean(params), dd.std(params)
                for ax, mean, std in zip(lax, mean, std):
                    color = 'C{:d}'.format(iaux)
                    ax.errorbar(xmain[imain] + xaux[iaux], mean, std, fmt='o', color=color, label=label if imain == 0 else None, capsize=4)
        for ax, param in zip(lax, params):
            ax.set_ylabel(label_bao[param.name])
            ax.grid(True, axis='y')
            ref = param.value
            if qlim is None:
                lim = np.mean(np.abs(np.array(ax.get_ylim()) - ref))
                ax.set_ylim(ref - lim, ref + lim)  # symmetrize
            else:
                ax.set_ylim(*qlim)
            ax.axhline(y=ref, xmin=0., xmax=1., color='k', linestyle='--')
        ax = lax[-1]
        ax.set_xticks(xmain)
        ax.set_xticklabels(ids, rotation=40, ha='right')
        for tick, color in zip(ax.xaxis.get_ticklabels(), colors): tick.set_color(color)
        if any(label is not None for label in labels): ax.legend()
        return fig
    
    @utils.plotter
    def plot_scatter(self, data, params=None, rescale=False, fig=None, **kwargs):
        """
        Scatter plot of best fits.

        Parameters
        ----------
        data : BaseFileEntry, list
            :class:`Profiles` file(s).

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``1`` axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from desilike.samples import Chain, Profiles
        from getdist import plots
        g = plots.get_subplot_plotter()
        centers, intervals = [], []
        for dd in data:
            prof = Profiles.load(dd)
            params = prof.bestfit.params(name=[str(param) for param in params] if params is not None else self.default_params, varied=True)
            ibest = prof.bestfit.logposterior.argmax()
            # Extract intervals or default to errors
            interval = prof.get('interval', None)
            if interval is None and not rescale:  # create interval from Gaussian error
                from desilike.samples import Samples
                interval = Samples()
                for array in prof.error[ibest]:
                    tmp = array[[0, 0]]
                    tmp[:1] *= -1
                    interval.set(tmp)
            if interval is not None:
                centers.append(prof.bestfit.choice(index=ibest, return_type=None))
                intervals.append(interval[None, :])
        samples = Chain(Chain.concatenate(centers))
        intervals = Chain(Chain.concatenate(intervals))
        if rescale:
            for param in params:
                array = samples[param]
                param = array.param
                mean = array.mean()
                mask = array < mean
                array[mask] = -(array[mask] - mean) / intervals[param][mask, 0]
                mask = ~mask
                array[mask] = (array[mask] - mean) / intervals[param][mask, 1]
                latex = param.latex(inline=False)
                latex = r'({0} - \langle {0} \rangle) / \sigma({0})'.format(latex)
                param.update(latex=latex, prior=None)  # to remove plot boundaries
        samples = samples.to_getdist(params=params)
        for iparam1, param1 in enumerate(params):
            for param2 in params[iparam1 + 1:]:
                g.plot_2d_scatter(samples, param1.name, param2.name, **kwargs)
                ax = g.get_axes()
                ax.grid(True)
                if rescale:
                    lim = (-3., 3.)
                    ax.set_xlim(lim)
                    ax.set_ylim(lim)
        fig = plt.gcf()
        return fig

    @utils.plotter
    def plot_whiskers(self, list_data, param='qiso', labels=None, qlim=None, fig=None, **kwargs):
        """
        Enrique's whiskers plot.

        Parameters
        ----------
        list_data : list
            List (for each tracer / zrange) of list (for each setting) of :class:`Profiles` file(s).

        param : str, default='qiso'
            Parameter to plot.

        labels : list, default=None
            List of labels for settings.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``1`` axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        ndata = len(list_data)
        nsettings = len(list_data[0])
        if fig is None:
            fig, lax = plt.subplots(1, ndata, figsize=(12, 6), squeeze=False)
        else:
            lax = fig.axes
        lax = np.ravel(lax)
        yvals = np.linspace(0, 10, nsettings)[::-1]

        for iax, (ax, ldata) in enumerate(zip(lax, list_data)):
            for i, data in enumerate(ldata):
                if data is None: continue
                center, interval, options = get_center_interval(data, params=param)
                tracer, zrange = options['tracer'], options['zrange']
                color = self.colors[tracer, tuple(zrange)]
                ax.errorbar(center, yvals[i], xerr=np.array(interval).T, marker='o', capsize=3, color=color, ms=5.0, markerfacecolor=utils.lighten_color(color), markeredgecolor=color, elinewidth=1.5)

                if i == 0:
                    ax.fill_betweenx(yvals, center[0] - interval[0][0], center[0] + interval[0][1], color=color, alpha=0.05)

            xlabel = r'$\alpha_{\rm iso}$' if param == 'qiso' else r'$\alpha_{\rm AP}$'
            ax.set_xlabel(xlabel, fontsize=20)
            ax.tick_params(axis='x', labelsize=13, rotation=45)
            if iax > 0:
                ax.axes.get_yaxis().set_visible(False)
            else:
                ax.set_yticks(yvals)
                if labels is not None: ax.set_yticklabels(labels, minor=False, rotation=0, fontsize=15)
            title = '{tracer} {zrange[0]:.1f}-{zrange[1]:.1f}'.format(tracer=self.get_tracer_label(tracer), zrange=zrange)
            ax.set_title(title, fontsize=15)
            ax.set_xlim(qlim)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1)
        return fig

    @utils.plotter
    def plot_profiles(self, list_data, param='qiso', fig=None, cl=(4, 6, 8, 10), **kwargs):
        r"""
        :math:`\chi^{2}` plot.

        Parameters
        ----------
        list_data : list
            List (for each tracer / zrange) of (wiggle, no-wiggle) :class:`Profiles` file(s).

        param : str, default='qiso'
            Parameter to plot.

        cl : list, tuple, default=(4, 8)
            Confidence levels.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``1`` axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from desilike.samples import plotting, Profiles
        ndata = len(list_data)
        if fig is None:
            ncols = min(ndata, 3)
            nrows = (ndata + ncols - 1) // ncols
            fig, lax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)
        else:
            lax = fig.axes
        lax = np.ravel(lax)
        for iax, (ax, ldata) in enumerate(zip(lax, list_data)):
            tracer, zrange = ldata[0].options['tracer'], ldata[0].options['zrange']
            color = self.colors[tracer, tuple(zrange)]
            profiles, profiles_now = [Profiles.load(data) for data in ldata]
            center, interval = get_center_interval(ldata[0], params=[param])[:2]
            param = profiles.bestfit.params()[str(param)]
            plotting.plot_profile_comparison(profiles, profiles_now, params=param, colors=color, cl=cl, fig=ax)
            sigma = (profiles_now.bestfit.chi2min - profiles.bestfit.chi2min)**0.5
            title = '{tracer} {zrange[0]:.1f}-{zrange[1]:.1f}\n'.format(tracer=self.get_tracer_label(tracer), zrange=zrange)
            chi2, size, nvaried = profiles.bestfit.chi2min, profiles.bestfit.attrs['size'], profiles.bestfit.attrs['nvaried']
            #title += r'$\chi^2 / \mathrm{{ndof}} = {chi2:.2f} / ({size:d} - {nvaried:d}) = {rchi2:.2f}$'.format(chi2=chi2, size=size, nvaried=nvaried, rchi2=chi2 / (size - nvaried)) + '\n'
            title += r'${center:.3f} \pm {std:.3f}$, ${sigma:.1f} \sigma$'.format(center=np.mean(center), std=np.mean(interval), sigma=sigma)
            #title += r'${center:.3f} \pm {std:.3f}$'.format(center=np.mean(center), std=np.mean(interval))
            ax.set_title(title, fontsize=12)
        for ax in lax[iax + 1:]:
            ax.axis('off')
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        return fig

# Function added by Chris Blake 30/1/24 for KP4 Key Paper
    def plot_correlation_multipoles_panel(self, axes, isub, data, mock=None, covariance=None, ells=(0,2,4), select=(20.,200.,4.)):
        fontsize = 9
        ax = axes[isub]
        if covariance is not None:
            covariance = covariance.load(select)
            diag = np.array_split(np.diag(covariance)**0.5, 3)
        if is_sequence(data): data = list(data)
        else: data = [data]
        tracer, zrange = data[0].options['tracer'], tuple(data[0].options['zrange'])
        data_list_poles = [d.load(select)(ell=ells, return_sep=True) for d in data]
        data_mean_s, data_mean_poles = (np.mean([poles[i] for poles in data_list_poles], axis=0) for i in (0, 1))
        if mock is not None:
            if is_sequence(mock): mock = list(mock)
            else: mock = [mock]
            mock_list_poles = [d.load(select)(ell=ells, return_sep=True) for d in mock]
            if mock_list_poles:
                mock_mean_s, mock_mean_poles = (np.mean([poles[i] for poles in mock_list_poles], axis=0) for i in (0, 1))
        for ill, ell in enumerate(ells):
            if covariance is not None:
                ax.errorbar(data_mean_s, data_mean_s**2 * data_mean_poles[ill], data_mean_s**2 * diag[ill], marker=self.points[ell], markersize=3, elinewidth=1, capsize=2, ls='none', color=self.colors[tracer, zrange], alpha=self.alphas[ell])
            else:
                ax.plot(data_mean_s, data_mean_s**2 * data_mean_poles[ill], linestyle='none', marker=self.points[ell], color=self.colors[tracer, zrange], alpha=self.alphas[ell])
            if ((mock is not None) and mock_list_poles):
                ax.plot(mock_mean_s, mock_mean_s**2 * mock_mean_poles[ill], color='k', linestyle=self.linestyles[ell], alpha=0.7, lw=1, zorder=1000)
        if ((isub == 0) or (isub == 1) or (isub == 2) or (isub == 3)):
            ax.set_ylim(-100.,100.)
        elif ((isub == 4) or (isub == 5)):
            ax.set_ylim(-60.,40.)
        elif (isub == 6):
            ax.set_ylim(-80.,80.)
        ax.tick_params(axis='x',labelsize=fontsize)
        ax.tick_params(axis='y',labelsize=fontsize)
        props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        if tracer[:3] == 'ELG':
            ax.text(120, 35, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize, fontweight='bold', verticalalignment='top', bbox=props)
        elif tracer[:3] == 'QSO':
            ax.text(120, 70, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize, fontweight='bold', verticalalignment='top', bbox=props)
        else:
            ax.text(120, 90, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,fontweight='bold', verticalalignment='top', bbox=props)
        if ((isub == 4) or (isub == 5)):
            ax.set_xlabel(r'$s$ [$h^{-1}\mathrm{Mpc}$]',fontsize=fontsize)
        else:
            ax.set_xticklabels([])
        if ((isub == 1) or (isub == 3) or (isub == 5)):
            ax.set_yticklabels([])
        if ((isub == 0) or (isub == 2) or (isub == 4)):
            ax.set_ylabel(r'$s^2 \xi_\ell(s)$ [$h^{{-2}}(\mathrm{{Mpc}})^2$]',labelpad=-1,fontsize=fontsize)
        ax.grid(True)
        return

# Function added by Chris Blake 30/1/24 for KP4 Key Paper
    def plot_power_multipoles_panel(self, axes, isub, data, mock=None, covariance=None, ells=(0,2,4), select=(0.,0.3,0.005)):
        fontsize = 9
        ax = axes[isub]
        if covariance is not None:
            covariance = covariance.load(select)
            diag = np.array_split(np.diag(covariance)**0.5, 3)
        if is_sequence(data): data = list(data)
        else: data = [data]
        tracer, zrange = data[0].options['tracer'], tuple(data[0].options['zrange'])
        data_list_poles = [d.load(select, mode='poles')(ell=ells, return_k=True, complex=False) for d in data]
        data_mean_k, data_mean_poles = (np.mean([poles[i] for poles in data_list_poles], axis=0) for i in (0, 1))
        if mock is not None:
            if is_sequence(mock): mock = list(mock)
            else: mock = [mock]
            mock_list_poles = [d.load(select, mode='poles')(ell=ells, return_k=True, complex=False) for d in mock]
            if mock_list_poles:
                mock_mean_k, mock_mean_poles = (np.mean([poles[i] for poles in mock_list_poles], axis=0) for i in (0, 1))
        for ill, ell in enumerate(ells):
            if covariance is not None:
                ax.errorbar(data_mean_k, data_mean_k * data_mean_poles[ill], data_mean_k * diag[ill], marker=self.points[ell], markersize=3, elinewidth=1, capsize=2, ls='none', color=self.colors[tracer, zrange], alpha=self.alphas[ell])
            else:
                ax.plot(data_mean_k, data_mean_k * data_mean_poles[ill], linestyle='none', marker=self.points[ell], color=self.colors[tracer, zrange], alpha=self.alphas[ell])
            if ((mock is not None) and mock_list_poles):
                ax.plot(mock_mean_k, mock_mean_k * mock_mean_poles[ill],color='k', linestyle=self.linestyles[ell], alpha=0.7, lw=1, zorder=1000)
        if ((isub == 0) or (isub == 1) or (isub == 2) or (isub == 3)):
            ax.set_ylim(-300.,1800.)
        elif ((isub == 4) or (isub == 5)):
            ax.set_ylim(-100.,600.)
        elif (isub == 6):
            ax.set_ylim(-200.,1200.)
        props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        if tracer[:3] == 'ELG':
            ax.text(0.1675, 565, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,fontweight='bold',verticalalignment='top', bbox=props)
        elif tracer[:3] == 'QSO':
            ax.text(0.1675, 1140, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,fontweight='bold',verticalalignment='top', bbox=props)
        else:
            ax.text(0.1675, 1700, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,fontweight='bold',verticalalignment='top', bbox=props)
        ax.tick_params(axis='x',labelsize=fontsize)
        ax.tick_params(axis='y',labelsize=fontsize)
        if ((isub == 4) or (isub == 5)):
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]',fontsize=fontsize)
        else:
            ax.set_xticklabels([])
        if ((isub == 1) or (isub == 3) or (isub == 5)):
            ax.set_yticklabels([])
        if ((isub == 0) or (isub == 2) or (isub == 4)):
            ax.set_ylabel(r'$k P_\ell(k)$ [$h^{{-2}}(\mathrm{{Mpc}})^2$]',labelpad=-1,fontsize=fontsize)
        ax.grid(True)
        return

class KP4StylePaper(KP4Style):

    """KP4 style for papers."""

    def __init__(self, **kwargs):
        super(KP4StylePaper, self).__init__(**{'lines.linewidth': 2, 'axes.labelsize': 14, **kwargs})


class KP4StylePresentation(KP4Style):

    """KP4 style for presentations."""

    def __init__(self, **kwargs):
        super(KP4StylePresentation, self).__init__(**{'lines.linewidth': 4, 'axes.labelsize': 18, **kwargs})
