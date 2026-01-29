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


class KP5Style(BaseStyle):
    """
    Context for KP5 style.
    To be used as a context:

    .. code-block:: python

        with KP5Style() as style:

            style.plot_bao_diagram(data, fn='my_plot.png')

    """
    def __init__(self, **kwargs):
        """
        Initialize :class:`KP5Style`.

        Parameters
        ----------
        kwargs : dict
            Either arguments for ``mpl.rcParams``, or attributes for this class.
            Passed to :meth:`update`.
        """
        # Dictionary to contain default colors
        self.colors = default_colors
        # Dictionaries for plotting different ell
        self._rcparams = mpl.rcParams.copy()
        self.update(**kwargs)
        self.default_params = ['qiso', 'qap', 'qpar', 'qper', 'df', 'dm', 'Omega_m', 'h', 'logA', 'n_s']

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
    def plot_full_shape_diagram(self, data, params=None, labels=None, figsize=None, fig=None):
        """
        Plot full shape diagram.

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
            ax.set_ylabel(param.latex(inline=True))
            ax.grid(True, axis='y')
            ref = param.value
            lim = np.mean(np.abs(np.array(ax.get_ylim()) - ref))
            ax.set_ylim(ref - lim, ref + lim)  # symmetrize
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
            interval = prof.get('interval', None)
            if interval is None:
                interval = prof.error[ibest]
                if rescale and not all(interval.attrs[param.name][key] for key in ['upper_valid', 'lower_valid'] for param in params): continue
                for array in interval:
                    tmp = array[[0, 0]]
                    tmp[:1] *= -1
                    interval.set(tmp)
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
    

class KP5StylePaper(KP5Style):

    """KP5 style for papers."""

    def __init__(self, **kwargs):
        super(KP5StylePaper, self).__init__(**{'lines.linewidth': 2, 'axes.labelsize': 14, **kwargs})


class KP5StylePresentation(KP5Style):

    """KP5 style for presentations."""

    def __init__(self, **kwargs):
        super(KP5StylePresentation, self).__init__(**{'lines.linewidth': 4, 'axes.labelsize': 18, **kwargs})