import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from .base import BaseStyle
from . import utils


def is_sequence(item):
    """Whether input item is a tuple or list."""
    return hasattr(item, '__iter__') and not isinstance(item, str)


default_colors = {'BGS': 'yellowgreen', 'BGS_BRIGHT-21.5': 'yellowgreen', ('BGS_BRIGHT-21.5', (0.1, 0.4)): 'yellowgreen',
'LRG': 'red', ('LRG', (0.4, 0.6)): 'orange', ('LRG', (0.6, 0.8)): 'orangered', ('LRG', (0.8, 1.1)): 'firebrick', 
'LRG+ELG': 'slateblue', 'LRGplusELG': 'slateblue', ('LRG+ELG_LOPnotqso', (0.8, 1.1)): 'slateblue', 'ELG': 'blue', ('ELG_LOPnotqso', (0.8, 1.1)): 'skyblue', ('ELG_LOPnotqso', (1.1, 1.6)): 'steelblue', 'QSO': 'seagreen', ('QSO', (0.8, 2.1)): 'seagreen', 'Lya': 'purple', ('Lya', (0.8, 3.5)): 'purple', ('Lya', (1.8, 4.2)): 'purple'}


args_rcparam = {'xtick.direction' : 'inout',
'ytick.direction' : 'inout',
'xtick.major.size' : 4,      # major tick size in points
'xtick.major.width' : 1,      # major tick width in points
'ytick.major.size'  : 4,      # major tick size in points
'ytick.major.width': 1,      # major tick width in points

'axes.grid'      : True,
'grid.color'    : 'grey',
'grid.linestyle' : '-',
'grid.linewidth' : 0.7,
'grid.alpha'     : 0.2,

## LEGEND
'legend.fancybox'  : True,
'legend.facecolor' : 'white',
'legend.framealpha' : 0.5,
'legend.edgecolor' : 'grey',
                

'figure.autolayout'  : False,      # When True, automatically adjust subplot # not used because of mollweide plot ...
'figure.frameon'     : True,

'figure.subplot.left'    : 0.08,  # the left side of the subplots of the figure
'figure.subplot.right'   : 0.96,   # the right side of the subplots of the figure
             
## LINES
'lines.linewidth' : 1.5,

## AXES
'axes.facecolor': 'white',
'axes.edgecolor': 'black',
'axes.formatter.use_mathtext' : True, # When True, use mathtext for scientific notation

#axes.facecolor             : eeeeee       # axes background color
#axes.edgecolor             : black        # axes edge color
'axes.linewidth': 1.,         # edge linewidth
'axes.labelcolor' : 'black',
'axes.formatter.use_mathtext' : True, # When True, use mathtext for scientific notation
'axes.labelsize' : 16,
                
## FONT
'font.family': 'serif',
'font.size': 13,
## TEXT
"text.color"          : "k",
'text.usetex'         : False,
#text.latex.preamble : \newcommand{\mathdefault}[1][]{} #With python3.9 it is now useless.
'mathtext.cal' : 'cursive',
'mathtext.rm'  : 'serif',
'mathtext.tt'  : 'monospace',
'mathtext.it'  : 'serif:italic',
'mathtext.bf'  : 'serif:italic:bold',
'mathtext.sf'  : 'sans\-serif',
'mathtext.fontset' : 'dejavusans',
#mathtext.default : it
               }

class KP3Style(BaseStyle):
    """
    Context for KP3 style.
    To be used as a context:

    .. code-block:: python

        with KP3Style() as style:

            style.plot_correlation_multipoles(data, covariance, fn='my_plot.png')

    """
    def __init__(self, **kwargs):
        """
        Initialize :class:`KP3Style`.

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
        self._rcparams = args_rcparam#mpl.rcParams.copy()
        self.update(**kwargs)

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
        """for key in self.colors:
            if isinstance(key, str): continue  # to keep only (tracer, zrange) pairs
            ax_tmp.hist(np.ones(1), color=self.colors[key], label='{0}, ${1[0]:.1f} < z < {1[1]:.1f}$'.format(*key))"""
        for ell in self.points:
            ax_tmp.plot([0, 1], [0, 1], marker=self.points[ell], color='k', linestyle='none', alpha=self.alphas[ell], label=r'$\ell = {:d}$'.format(ell))
        ax.legend(*ax_tmp.get_legend_handles_labels(), ncol=3, loc='center')
        plt.close(fig_tmp)
        # Hide the axes frame and the x/y labels
        ax.axis('off')
        return fig

    @utils.plotter
    def plot_correlation_multipoles(self, data, covariance=None, mean=True, ells=(0, 2, 4), select=(20., 200., 4.), markers=None, fig=None):
        """
        Plot correlation function multipoles.

        Parameters
        ----------
        data : BaseFileEntry
            Correlation function file(s).

        covariance : BaseFile, default=None
            Covariance matrix file.
            If ``None``, no error bars are plotted.

        ells : tuple, default=(0, 2, 4)
            Poles to plot.

        select : tuple, default=(20., 200., 4.)
            s-limits (min, max, step).

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
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        if covariance is not None:
            covariance = covariance.load(select)
            diag = np.array_split(np.diag(covariance)**0.5, 3)
        if is_sequence(data): data = list(data)
        else: data = [data]
        tracer, zrange = data[0].options['tracer'], tuple(data[0].options['zrange'])
        list_poles = [d.load(select)(ell=ells, return_sep=True) for d in data]
        mean_s, mean_poles = (np.mean([poles[i] for poles in list_poles], axis=0) for i in (0, 1))
        if markers is None: markers = ('line', 'point')
        for ill, ell in enumerate(ells):
            ls = self.linestyles[ell] if 'line' in markers else 'none'
            ps = self.points[ell] if 'point' in markers else 'none'
            if covariance is not None:
                ax.errorbar(mean_s, mean_s**2 * mean_poles[ill], mean_s**2 * diag[ill], ls='none', fmt='none', color=self.colors[tracer, zrange], alpha=self.alphas[ell])
            else:
                if mean:
                    ax.plot(mean_s, mean_s**2 * mean_poles[ill], linestyle=ls, marker=ps, color=self.colors[tracer, zrange], alpha=self.alphas[ell])
                else:
                    for (s, poles) in list_poles:
                        ax.plot(s, s**2 * poles[ill], linestyle=ls, marker=ps, color=self.colors[tracer, zrange], alpha=self.alphas[ell])
        ax.set_xlabel(r'$s$ [$h^{-1}\mathrm{Mpc}$]')
        ax.set_ylabel(r'$s^2 \xi_\ell(s)$ [$h^{{-2}}(\mathrm{{Mpc}})^2$] {0}'.format(tracer[:3]), labelpad=-2)
        ax.grid(True)
        return fig

    @utils.plotter
    def plot_power_multipoles(self, data, covariance=None, mean=True, ells=(0, 2, 4), select=(0., 0.3, 0.005), markers=None, fig=None):
        """
        Plot power spectrum multipoles.

        Parameters
        ----------
        data : BaseFileEntry, list
            Power spectrum file(s).

        covariance : BaseFile, default=None
            Covariance matrix file.
            If ``None``, no error bars are plotted.

        ells : tuple, default=(0, 2, 4)
            Poles to plot.

        select : tuple, default=(0., 0.3, 0.005)
            k-limits (min, max, step).

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
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        if covariance is not None:
            covariance = covariance.load(select)
            diag = np.array_split(np.diag(covariance)**0.5, 3)
        if is_sequence(data): data = list(data)
        else: data = [data]
        tracer, zrange = data[0].options['tracer'], tuple(data[0].options['zrange'])
        list_poles = [d.load(select, mode='poles')(ell=ells, return_k=True, complex=False) for d in data]
        mean_k, mean_poles = (np.mean([poles[i] for poles in list_poles], axis=0) for i in (0, 1))
        if markers is None: markers = ('line', 'point')
        for ill, ell in enumerate(ells):
            ls = self.linestyles[ell] if 'line' in markers else 'none'
            ps = self.points[ell] if 'point' in markers else 'none'
            if covariance is not None:
                ax.errorbar(k, k * poles[ill], k * diag[ill], ls=ls, fmt=ps, color=self.colors[tracer, zrange], alpha=self.alphas[ell])
            else:
                if mean:
                    ax.plot(mean_k, mean_k * mean_poles[ill], linestyle=ls, marker=ps, color=self.colors[tracer, zrange], alpha=self.alphas[ell])
                else:
                    for (k, poles) in list_poles:
                        ax.plot(k, k * poles[ill], linestyle=ls, marker=ps, color=self.colors[tracer, zrange], alpha=self.alphas[ell])
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        ax.set_ylabel(r'$k P_\ell(k)$ [$h^{{-2}}(\mathrm{{Mpc}})^2$] {0}'.format(tracer[:3]), labelpad=-2)
        ax.grid(True)
        return fig

    @utils.plotter
    def plot_window_power_multipoles(self, wmatrix, ells=(0, 2, 4), select=(0., 0.3), markers=None, fig=None):
        """
        Plot window matrix effect on power spectrum multipoles.

        Parameters
        ----------
        wmatriw : BaseFile, default=None
            Window matrix file.
            If ``None``, no error bars are plotted.

        ells : tuple, default=(0, 2, 4)
            Poles to plot.

        select : tuple, default=(0., 0.3, 0.005)
            k-limits (min, max, step).

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
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        from pypower import BaseMatrix
        from cosmoprimo.fiducial import DESI

        m = wmatrix if isinstance(wmatrix, BaseMatrix) else BaseMatrix.load(wmatrix)
        factorout = 5
        m.slice_x(sliceout=slice(0, len(m.xout[0]) // factorout * factorout, factorout))  # rebin observed ks to dk = 0.0O5
        m.select_x(xinlim=(0., 0.5), xoutlim=select[:2])  # apply k-cut to both input and output; some margin for input
        ellsout = [proj.ell for proj in m.projsout]
        kout = m.xout[0]
        kin = m.xin[0]

        # Generate Kaiser power spectrum
        cosmo = DESI()
        z = 1.
        pk = cosmo.get_fourier().pk_interpolator(of='delta_cb')(kin, z=z)
        b = 2.
        f = cosmo.growth_rate(z)
        shotnoise = 1 / 1e-4
        volume = 1e9
        poles = []
        poles.append((b**2 + 2. / 3. * f * b + 1. / 5. * f**2) * pk + shotnoise)
        poles.append((4. / 3. * f * b + 4. / 7. * f**2) * pk)
        poles.append(8. / 35 * f**2 * pk)
        poles = np.array(poles, dtype='f8')

        # matrix * theory (shot noise need to be window-convolved)
        cpoles = m.dot(poles, unpack=True)

        # Plot
        poles[0] -= shotnoise
        cpoles[0] -= shotnoise
        ax.plot([], [], linestyle='--', color='k', label='theory')
        ax.plot([], [], linestyle='-', color='k', label='window')
        mask = (kin >= select[0]) & (kin <= select[1])
        for ill, ell in enumerate(ellsout):
            color = 'C{:d}'.format(ill)
            ax.plot([], [], linestyle='-', color=color, label=r'$\ell = {:d}$'.format(ell))
            plt.plot(kin[mask], kin[mask] * poles[ill][mask], linestyle='--', color=color, label=None)
            plt.plot(kout, kout * cpoles[ill], linestyle='-', color=color, label=None)
        ax.legend()
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        ax.set_ylabel(r'$k P(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.grid(True)
        return fig

    @utils.plotter
    def plot_covariance_correlation_multipoles(self, covariance, ells=(0, 2, 4), select=(20., 200., 4.), stat=None, corrcoef=True, norm=None):
        """
        Plot covariance.

        Parameters
        ----------
        covariance : list, default=None
            List of mock files.
            If ``None``, no error bars are plotted.

        ells : tuple, default=(0, 2, 4)
            Poles to plot.

        select : tuple, default=(20., 200., 4.)
            s-limits (min, max, step).

        stat : str, default=None
            'residual' to plot covariance matrix for residual.

        corrcoef : bool, default=True
            If ``True``, plot correlation matrix.

        norm : matplotlib.colors.Normalize, default=None
            Scales the covariance / correlation to the canonical colormap range [0, 1] for mapping to colors.
            By default, the covariance / correlation range is mapped to the color bar range using linear scaling.

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
        from pycorr import TwoPointCorrelationFunction
        from pycorr.twopoint_estimator import ResidualTwoPointEstimator
        s, poles = [], []
        for mock in covariance:
            mock = (ResidualTwoPointEstimator if stat == 'residual' else TwoPointCorrelationFunction).load(mock).select(select)
            si, polesi = mock(ell=ells, return_sep=True, return_std=False)
            s.append(si)
            poles.append(polesi.ravel())
        s = np.mean(s, axis=0)
        mask_ell = np.concatenate([np.full(s.size, ell) for ell in ells])
        covmatrix = np.cov(poles, rowvar=False, ddof=1)

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
                s1 = s2 = s
                mask1, mask2 = mask_ell == ell1, mask_ell == ell2
                mesh = ax.pcolor(s1, s2, covmatrix[np.ix_(mask1, mask2)].T, norm=norm, cmap=plt.get_cmap('RdBu'))
                if ill1 > 0: ax.xaxis.set_visible(False)
                else: ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
                if ill2 > 0: ax.yaxis.set_visible(False)
                else: ax.set_ylabel(r'$s$ [$\mathrm{Mpc}/h$]')
                text = r'$\ell = {:d} \times \ell = {:d}$'.format(ell1, ell2)
                ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black')
                ax.grid(False)

        fig.subplots_adjust(right=xextend)
        cbar_ax = fig.add_axes([xextend + 0.05, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(mesh, cax=cbar_ax)
        return fig

    @utils.plotter
    def plot_residual_multipoles(self, data, covariance=None, mean=True, ells=(0, 2, 4), select=(20., 200., 4.), markers=None, fig=None):
        """
        Plot residual multipoles.

        Parameters
        ----------
        data : BaseFileEntry
            Correlation function file(s).

        covariance : list, default=None
            List of mock files.
            If ``None``, no error bars are plotted.

        ells : tuple, default=(0, 2, 4)
            Poles to plot.

        select : tuple, default=(20., 200., 4.)
            s-limits (min, max, step).

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
        from scipy import stats
        from pycorr.twopoint_estimator import ResidualTwoPointEstimator
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        if covariance is not None:
            covariance = [ResidualTwoPointEstimator.load(mock).select(select)(ell=ells, return_sep=False) for mock in covariance]
            mean = np.mean(covariance, axis=0)
            std = np.std(covariance, ddof=1, axis=0)
        if is_sequence(data): data = list(data)
        else: data = [data]
        std /= len(data)**0.5
        options = data[0].options
        tracer, zrange = options['tracer'], tuple(options['zrange'])
        list_poles = [ResidualTwoPointEstimator.load(d).select(select)(ell=ells, return_sep=True) for d in data]
        s, poles = (np.mean([poles[i] for poles in list_poles], axis=0) for i in (0, 1))
        if markers is None: markers = ('line', 'point')
        for ill, ell in enumerate(ells):
            ls = self.linestyles[ell] if 'line' in markers else 'none'
            ps = self.points[ell] if 'point' in markers else 'none'
            ax.plot(s, poles[ill], linestyle=ls, marker=ps, color=self.colors[tracer, zrange], alpha=1., label=r'$\ell = {:d}$'.format(ell))
            if covariance is not None:
                ax.fill_between(s, (mean[ill] - std[ill]), (mean[ill] + std[ill]), color=self.colors[tracer, zrange], alpha=0.2)
        if covariance is not None:
            diff = np.ravel(poles) - np.ravel(mean)
            cov = np.cov([mock.ravel() for mock in covariance], ddof=1, rowvar=False)
            nobs, nbins = len(covariance), diff.size
            hartlap = (nobs - nbins - 2.) / (nobs - 1.)
            prec = np.linalg.inv(cov) * hartlap * len(data)
            chi2 = diff.dot(prec).dot(diff)
            print('For {} in {} {} cut = {}, residual chi2 is {:.2f} / {:d} = {:.2f}, p-value of {:.2g}.'.format(tracer, zrange, options['region'], options['cut'], chi2, nbins, chi2 / nbins, stats.chi2.sf(chi2, nbins)))
        ax.set_xlabel(r'$s$ [$h^{-1}\mathrm{Mpc}$]')
        ax.set_ylabel(r'$(DR / RR - 1)_\ell(s)$ {0}'.format(tracer[:3]), labelpad=-2)
        ax.legend()
        ax.grid(True)
        return fig

# Function added by Chris Blake 14/1/24 for KP3 Key Paper
    def plot_map_sindec_panel(self,ras,sindec,comp,vmin,vmax,title,fig,isub):
        ps = 0.1
        ax = fig.add_subplot(4,1,isub)
        mp = plt.scatter(ras,sindec,c=comp,edgecolor='none',vmin=vmin,vmax=vmax,s=ps)
        plt.colorbar(mp,fraction=0.05,shrink=2/2.3) 
        plt.xlabel('R.A. [deg]',fontsize=14)
        decvals = np.array([-30,-15,0,15,30,60,90])
        sindecvals = np.sin(np.radians(decvals))
        plt.yticks(sindecvals,decvals)
        plt.ylabel('Dec. [deg]',fontsize=14)
        plt.xlim(-60.,300.)
        plt.ylim(sindecvals[0],sindecvals[-1])
        plt.title(title,fontsize=14)
        plt.grid()
        return

# Function added by Chris Blake 14/1/24 for KP3 Key Paper
    def plot_legend_panel(self,axes,isub):
        fontsize = 9
        ax = axes[isub]
        fig_tmp, ax_tmp = plt.subplots()
        """for key in self.colors:
            if isinstance(key, str): continue  # to keep only (tracer, zrange) pairs
            if (key[0] != 'Lya'):
                ax_tmp.hist(np.ones(1), color=self.colors[key], label='{0:.3s}, ${1[0]:.1f} < z < {1[1]:.1f}$'.format(*key))"""
        for ell in self.points:
            ax_tmp.plot([0, 1], [0, 1], marker=self.points[ell], color='k', linestyle='none', alpha=self.alphas[ell], label=r'$\ell = {:d}$ (data)'.format(ell))
        for ell in self.points:
            ax_tmp.plot([0, 1], [0, 1], color='k', linestyle=self.linestyles[ell], alpha=self.alphas[ell], label=r'$\ell = {:d}$ (mock)'.format(ell))
        handles,labels = ax_tmp.get_legend_handles_labels()
        #order = [6,7,8,9,10,11,12,0,1,2,3,4,5]
        order = [0,1,2,3,4,5]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=2,columnspacing=1,fontsize=fontsize,loc=[0.05,0.3])
        ax.axis('off')
        return

# Function added by Chris Blake 14/1/24 for KP3 Key Paper
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
        # place a text box 
        props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        if tracer[:3] == 'ELG':
             ax.text(125, 35, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,
                verticalalignment='top', bbox=props)
        elif tracer[:3] == 'QSO':
             ax.text(125, 70, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,
                verticalalignment='top', bbox=props)
        else:
            ax.text(125, 90, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,
                verticalalignment='top', bbox=props)
        
        if ((isub == 5) or (isub == 6)):
            ax.set_xlabel(r'$s$ [$h^{-1}\mathrm{Mpc}$]',fontsize=fontsize)
        else:
            ax.set_xticklabels([])
        if ((isub == 1) or (isub == 3) or (isub == 5)):
            ax.set_yticklabels([])
        if ((isub == 0) or (isub == 2) or (isub == 4) or (isub == 6)):
            ax.set_ylabel(r'$s^2 \xi_\ell(s)$ [$h^{{-2}}(\mathrm{{Mpc}})^2$]',labelpad=-1,fontsize=fontsize)
        ax.grid(True)
        return    

# Function added by Chris Blake 14/1/24 for KP3 Key Paper
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
            
        # place a text box 
        props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        if tracer[:3] == 'ELG':
            ax.text(0.175, 565, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,
                verticalalignment='top', bbox=props)
        elif tracer[:3] == 'QSO':
            ax.text(.175, 1140, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,
                verticalalignment='top', bbox=props)
        else:
            ax.text(0.175, 1700, f'{tracer[:3]}: {zrange[0]} < z < {zrange[1]}', fontsize=fontsize,
                verticalalignment='top', bbox=props)
            
        ax.tick_params(axis='x',labelsize=fontsize)
        ax.tick_params(axis='y',labelsize=fontsize)
        if ((isub == 5) or (isub == 6)):
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]',fontsize=fontsize)
        else:
            ax.set_xticklabels([])
        if ((isub == 1) or (isub == 3) or (isub == 5)):
            ax.set_yticklabels([])
        if ((isub == 0) or (isub == 2) or (isub == 4) or (isub == 6)):
            ax.set_ylabel(r'$k P_\ell(k)$ [$h^{{-2}}(\mathrm{{Mpc}})^2$]',labelpad=-1,fontsize=fontsize)
        ax.grid(True)
        return


class KP3StylePaper(KP3Style):

    """KP3 style for papers."""

    def __init__(self, **kwargs):
        super(KP3StylePaper, self).__init__(**{'lines.linewidth': 2, 'axes.labelsize': 14, **kwargs})


class KP3StylePresentation(KP3Style):

    """KP3 style for presentations."""

    def __init__(self, **kwargs):
        super(KP3StylePresentation, self).__init__(**{'lines.linewidth': 4, 'axes.labelsize': 18, **kwargs})
