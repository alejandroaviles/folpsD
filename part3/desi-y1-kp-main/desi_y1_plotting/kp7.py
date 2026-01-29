import os

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from .base import BaseStyle
from . import utils
from .kp3 import is_sequence, default_colors


class KP7Style(BaseStyle):
    """
    Context for KP7 style.
    To be used as a context:

    .. code-block:: python

        with KP7Style() as style:

            style.plot_bao_diagram(data, fn='my_plot.png')

    """
    def __init__(self, **kwargs):
        """
        Initialize :class:`KP7Style`.

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
        import seaborn as sns
        self.color_palette = [sns.color_palette('colorblind')[i] for i in [0, 1, 2, 4, 5, 3]]
        self.color_palette[1] = 'darkorange'


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
    def inset_plot_correlation_bao(self, state, slim=(50., 150.), color='C0', scale=1., spow=2, fig=None):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        ell = 0
        ill = state['ells'].index(ell)
        sep, data, std, theory, theory_nobao = state['s'][ill], state['data'][ill], state['std'][ill], state['theory'][ill], state['theory_nobao'][ill]
        indices = np.flatnonzero((sep >= slim[0]) & (sep <= slim[1]))
        if indices[0] > 0: indices = np.insert(indices, 0, indices[0] - 1)
        if indices[-1] < sep.size - 1: indices = np.insert(indices, -1, indices[-1] + 1)
        sep, data, std, theory, theory_nobao = sep[indices], data[indices], std[indices], theory[indices], theory_nobao[indices]
        ax.errorbar(sep, sep**spow * (data - theory_nobao), sep**spow * std, color=color, linestyle='none', marker='o', linewidth=2. * scale, markersize=2. * scale)
        ax.plot(sep, sep**spow * (theory - theory_nobao), color=color, linewidth=2. * scale, markersize=2. * scale)
        ax.xaxis.set_tick_params(size=scale, pad=5. * scale)
        ax.yaxis.set_tick_params(size=scale, pad=5. * scale)
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(scale * item.get_fontsize())
        if spow:
            ylabel = r'$s^{{:d}} \Delta \xi_{{{:d}}}(s)$ [$(\mathrm{{Mpc}}/h)^{{2}}$]'.format(spow, ell)
        else:
            ylabel = r'$\Delta \xi_{{{:d}}}(s)$'.format(ell)
        ax.set_ylabel(ylabel, labelpad=0.5 * scale)
        ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]', labelpad=0.5 * scale)
        ax.set_xlim(*slim)
        for spine in ax.spines.values(): spine.set_color(color)
        return fig

    @utils.plotter
    def plot_bao_diagram(self, data, cosmo_ref=None, cosmo_alt=None, label_cosmo_alt=None, label_bao=None, apmode='qparqper', insets=None, xyinsets=None, label_insets='{tracer}, ${zrange[0]:.1f} < z < {zrange[1]:.1f}$', labelsize_insets=6., zlim=(0., 2.5), qlim=(0.85, 1.15), seed=False, figsize=None, fig=None, **kwargs):
        """
        Plot BAO diagram.

        Parameters
        ----------
        data : BaseFileEntry, list
            :class:`FisherLikelihood` file(s), or dictionary mapping (tracer, zrange): LikelihoodFisher instance.

        cosmo_ref : Cosmology, default=None
            Cosmology to use as reference.
    
        cosmo_alt : Cosmology, default=None
            Cosmology to use for theory curves.

        label_cosmo_alt : str, default=None
            If ``cosmo_alt`` is provided, use this label.

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
        from desi_y1_files.cosmo_tools import convert_bao_fisher, predict_bao, get_bao_params
        params = get_bao_params(apmode)
        label_bao = label_bao or {}
        default_label_bao = {'qpar': 'radial BAO / fiducial', 'qper': 'transverse BAO / fiducial',
                             'qiso': 'isotropic BAO / fiducial', 'qap': 'anisotropic BAO / fiducial'}
    
        def _make_list(tmp, size):
            if tmp is None: tmp = []
            tmp = list(tmp)
            tmp = tmp + [None] * (max(size - len(tmp), 0))
            return tmp

        if not isinstance(data, dict):
            from desilike import LikelihoodFisher
            data = {(dd.options['tracer'], tuple(dd.options['zrange'])): LikelihoodFisher.load(dd) for dd in data}
        insets = _make_list(insets, len(data))
        xyinsets = _make_list(xyinsets, len(data))
        if isinstance(label_insets, str): label_insets = [label_insets] * len(data)

        if fig is None:
            nrows = len(params)
            fig, lax = plt.subplots(nrows, figsize=figsize, sharex=True, sharey=not any(insets), squeeze=False)
            lax = np.ravel(lax)
        else:
            lax = fig.axes

        def format_inset(inset, color):
            img = plt.imread(inset)
            from matplotlib import colors
            mask = np.any(img[..., :3] < 1., axis=-1)  # everything that is not white
            img[mask, ...] = colors.to_rgba(color)
            return img

        labelset = set()
        for ax, param in zip(lax, params):
            ax.set_ylabel(label_bao.get(param.name, default_label_bao[param.name]))
            ax.set_ylim(qlim)
            ax.set_xlim(zlim)
        lax[-1].set_xlabel('$z$')

        if cosmo_ref is None:
            from cosmoprimo.fiducial import DESI
            cosmo_ref = DESI()
            for ax in lax:
                ax.axhline(1., zlim[0], zlim[-1], color='k', linestyle='--')

        ninsets = sum(inset is not None for inset in insets)
        ninsets_up = (ninsets + 1) // 2
        ninsets_low = ninsets - ninsets_up
        height = 0.3  # height of the insets (in fraction of axis)
        figsize = fig.get_size_inches()
        width = figsize[1] / figsize[0] * height  # width of the insets, to get it ~ square even if figure is elongated
        hdiff = 0.5 - height # total white space for up row
        hoffset = 2. / 3. * hdiff  # for up row, white space of (2/3, 1/3) with respect to (middle, top)
        hspace = 0.5  # total fraction of the horizontal separation between two insets to leave on the left and right
        step_up = (1. - width) / (ninsets_up - 1 + hspace)
        step_low = (1. - width) / (ninsets_low - 1 + hspace)
        iinset = 0
        rng = None
        if seed is not False:
            rng = np.random.RandomState(seed=int(seed))
        for (key, dd), inset, xyinset, labelinset in zip(data.items(), insets, xyinsets, label_insets):
            label = key[0].split('_')[0]
            color = self.colors[key]
            zz = dd.attrs['zeff']
            try:
                dd = convert_bao_fisher(dd, apmode=apmode, scale='distance')
                ref = predict_bao(zz, apmode=apmode, cosmo=cosmo_ref, scale='distance')
            except ValueError as exc:
                if apmode == 'qisoqap':
                    dd = convert_bao_fisher(dd, apmode='qiso', scale='distance')
                    ref = predict_bao(zz, apmode='qiso', cosmo=cosmo_ref, scale='distance')
                else:
                    raise exc
            lparams = dd.params()
            mean, std = dd.mean(lparams) / ref, dd.std(lparams) / ref
            if rng is not None:
                mean = rng.normal(loc=mean, scale=std)
            if inset is not None:
                ax = lax[0]
                up = not (iinset % 2)
                step = step_up if up else step_low
                ix = (iinset // 2 + hspace / 2.) * step
                iy = (0.5 + hoffset) if up else (0.5 - height - hoffset)
                trans_axes_to_fig = ax.transAxes + fig.transFigure.inverted()  # axis -> display -> figure
                trans_fig_to_data = ax.transAxes + ax.transData.inverted()  # axis -> display -> data
                if xyinset is not None:
                    ix, iy = xyinset
                (x0, y0), (x1, y1) = trans_fig_to_data.transform([(ix, iy), (ix + width, iy + height)])
                yp = mean[0]
                up = y0 >= yp
                y0 = y0 if up else y1
                ax.plot([zz, x0], [yp, y0], color=color, linewidth=1.)
                ax.plot([zz, x1], [yp, y0], color=color, linewidth=1.)
                (x0, y0), (x1, y1) = trans_axes_to_fig.transform([(ix, iy), (ix + width, iy + height)])
                axinset = fig.add_axes([x0, y0, x1 - x0, y1 - y0])  # left, bottom, width, height
                if utils.is_path(inset) and str(inset).endswith('.npy'):
                    inset = np.load(inset, allow_pickle=True)[()]['attrs']['observable']
                if utils.is_path(inset) and str(inset).endswith('.png'):
                    inset = format_inset(inset, color)
                    axinset.imshow(inset)
                else:
                    if not isinstance(inset, dict):
                        inset = inset.attrs['observable']
                    self.inset_plot_correlation_bao(inset, color=color, fig=axinset, scale=height, **kwargs)
                    for spine in axinset.spines.values(): spine.set_linewidth(1.)
                #axlabel = labelinset.format(tracer=label, zrange=key[1])
                #axinset.text(0.05, 0.92, axlabel, color=color, fontsize=labelsize_insets, transform=axinset.transAxes)
                axlabel = labelinset.format(tracer=label, zrange=key[1])
                axinset.text(0.5, 1. + labelsize_insets / 300 if up else - labelsize_insets / 200, axlabel, va='bottom' if up else 'top', ha='center', color=color, fontsize=labelsize_insets, transform=axinset.transAxes)
                iinset += 1
            for param, mean, std in zip(lparams, mean, std):
                ax = lax[params.index(param)]
                ax.errorbar(zz, mean, std, fmt='o', color=color, markersize=4, capsize=4)
                if label in labelset: continue
                trans = ax.transData + ax.transAxes.inverted()  # data -> display -> axis
                ix, iy = trans.transform((zz, mean))
                if inset is None: ax.text(ix + 0.02, iy + 0.05, label, color=color, transform=ax.transAxes)
                labelset.add(label)

        z = np.linspace(*zlim, 1000)
        if cosmo_alt is not None:
            means = np.array(predict_bao(z, apmode=apmode, cosmo=cosmo_alt, scale='distance'))
            means /= np.array(predict_bao(z, apmode=apmode, cosmo=cosmo_ref, scale='distance'))
            for iax, (ax, mean) in enumerate(zip(lax, means)):
                ax.plot(z, mean, color='k', linestyle='-', label=label_cosmo_alt if iax == 0 else None)
            if label_cosmo_alt:
                lax[0].legend(frameon=False)
        fig.align_labels()
        return fig

    @utils.plotter
    def plot_summary_H0(self, center, error, fig=None):
        r"""
        :math:`H_{0}' plot, from Rafaela Gsponer.

        Parameters
        ----------
        center : float
            DESI :math:`H_{0}' center value.

        error : float
            DESI :math:`H_{0}' uncertainty.

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
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            ax = fig.axes[0]

        ax.set_xlabel(R"$H_{0}$ [km/s/Mpc]")
        ax.set_xlim(60, 78)
        ax.set_ylim(0, 18)
        ax.set_yticks([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [R"(Riess et al.) SH0ES", R"(Pesce et al.) Masers", R"(Blakeslee et al.) SBF", R"(Huang et al.) Miras - SNIa", R"(Freedman et al.) TRGB - SNIa", R"DESI BAO+BBN", R"(Alam et al.) eBOSS+BOSS BAO+BBN", R"(Cuceu et al.) eBOSS Ly$\alpha$ DF+BBN", R"(Simon et al.) eBOSS+BOSS DF+BBN", R"(Gsponer et al.) eBOSS+BOSS DF+BBN", R"(Ivanov et al.) BOSS DF+BBN", R"(Schoneberg et al.) eBOSS+BOSS BAO+SF+BBN", R"(Aiola et al.) ACT", R"(Dutcher et al.) SPT", "(Aghanim et al.) Planck"])

        #CMB
        ax.errorbar(67.27, 16, xerr=[[ 0.6], [0.6 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5) #Planck (1807.06209)
        ax.errorbar(68.8, 15, xerr=[[ 1.5], [1.5  ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #SPT (2101.01684)
        ax.errorbar(67.9, 14, xerr=[[ 1.5], [1.5 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #ACT (2007.07288)

        #BOSS+eBOSS
        ax.errorbar(68.3, 13, xerr=[[ 0.69], [0.66 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)    # SF+BAO+BBN eBOSS (2209.14330)
        ax.errorbar(67.9, 12, xerr=[[ 1.1], [1.1 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)    # DM BOSS (1909.05277)
        ax.errorbar(67.33, 11, xerr=[[ 1.3], [1.3 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)    # DM eBOSS +BOSS z1 + BBN (in prep)
        ax.errorbar(68.27, 10, xerr=[[ 0.85], [0.78 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  # DM eBOSS QSO + BOSS + BBN (2210.14931)
        ax.errorbar(63.2, 9, xerr=[[ 2.5], [2.5 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5,)  # DM eBOSS Lya (2209.13942)
        ax.errorbar(67.35, 8, xerr=[[ 0.97], [0.97 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #(e)BOSS BAO + BBN (2007.08991)

        #DESI
        ax.errorbar(center, 7, xerr=error, fmt='s', color='b', elinewidth=1, capsize=3, capthick=1.5)
        ax.axhline(6, linestyle='dashed', color='k')
        ax.text(x=75.5, y=6.25, s='Indirect')
        ax.text(x=75.5, y=5.25, s='Direct')

        #SNIa
        ax.errorbar(69.8, 5, xerr=[[1.9], [1.9]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #TRGB (1908.10883)
        ax.errorbar(73.3, 4, xerr=[[4], [4]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)      #Miras (1908.10883)
        ax.errorbar(73.3, 3, xerr=[[2.5], [2.5]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #SBF (2101.02221)
        ax.errorbar(73.9, 2, xerr=[[3], [3]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)       #Masers (2001.09213)
        ax.errorbar(73.04, 1, xerr=[[1.04], [1.04]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5) #SH0ES (2112.04510)

        # axes.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.2), fontsize=14)
        ax.axvspan(72, 74.08, alpha=0.05, color='blue')
        ax.axvspan(66.67, 67.87, alpha=0.05, color='red')
        return fig

    @utils.plotter
    def plot_whisker_H0(self, datasets, tension_levels, models, colors=None, markers=('s', '^', 'o', 'v', '8', 'D'), fig=None):
        r"""
        :math:`H_{0}'-tension whisker plot, from Rodrigo Calderon.
        
        Whisker plot representing the level of Gaussian tension between the different datasets/models and SHOES.

        Parameters
        ----------
        datasets : list
            List of datasets to plot (ideally matching the chain labels).

        tension_levels : list[list]
            List of tension levels for each dataset, where each element is a list containing the level of tension for each model.
        
        models : list
            List of the different models to be plotted in the y-axis.
            
        colors : list
            List of colors for each dataset.
            Optionally, default colors instead.
            
        markers : list
            List of marker styles for each dataset.
            Optionally, markers to use for representing the different datasets.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``1 + len(self.ells)`` axes.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]

        from matplotlib.lines import Line2D

        if colors is None:
            colors = [f'C{i}' for i in range(len(models))]
            
        # Plotting horizontal lines representing the level of tension X for each set of measurements
        for i, (tension, color, marker) in enumerate(zip(tension_levels, colors, markers)):
            for j, x in enumerate(tension):
                ax.plot([0, x], [j + i*0.1, j + i*0.1], linestyle='-', color=color)
                ax.plot(x, j + i*0.1, marker=marker, color=color)

        # Adding labels and title
        ax.set_yticks(range(len(models)), models)
        ax.set_ylabel('Models')
        ax.set_xlabel('Level of tension $\sigma$')
        ax.set_title(r'$H_0$ Tension with SH0ES')

        # Custom legend handles
        legend_handles = []
        for i in range(len(tension_levels)):
            line = Line2D([0, 1], [0, 0], color=colors[i], linestyle='-', linewidth=2)
            marker = Line2D([0], [0], marker=markers[i], color=colors[i], markerfacecolor=colors[i], markersize=10)
            legend_handles.append((line, marker))

        # Adding legend with custom handles
        ax.legend(legend_handles, datasets, loc='upper left', bbox_to_anchor=(1, 1), title='Data Sets')

        ax.set_xlim(0,6)
        # Displaying the plot
        ax.grid(True, alpha=0.3)
        return fig
    
    @utils.plotter()
    def plot_Hrd(self, samples, cosmo=None, colors=None, data=None, axis_H='right', scale_Hrd=1, label_cosmo=None):
        r"""
        :math:`H(z)r_d / (1+z)' plot, from Rodrigo Calderon.
        
        Expansion history plot.

        Parameters
        ----------
        samples : dict
            Dictionnary containing samples for each model.
            Keys in the dictionnary will be used for the labels in the plot for each model.

        colors : list, default=None
            List of colors for each contour/model.
            Optionally, if not provided, default colors will be used.

        data : dict, default=None
            Dictionary mapping data to :class:`FisherLikelihood`.

        axis_H : str, default='right'
            Axis for :math:`H(z) / (1+z)', 'left' or 'right'.

        scale_Hrd : int, default=1
            100 to scale :math:`H(z)r_d / (1+z)' axis.
            If ``None``, do not show :math:`H(z)r_d / (1+z)'.

        label_cosmo : str, default=None
            If ``cosmo`` is provided, use this label.
    
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``1 + len(self.ells)`` axes.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from cosmoprimo import constants
        from desi_y1_files.cosmo_tools import predict_bao, convert_bao_fisher
        z = np.linspace(0, 2.5, 100)
        factor = constants.c / 1e3
        show_Hrd = scale_Hrd is not None
        if not show_Hrd:
            axis_H = 'left'
        if axis_H == 'left':
            scale = cosmo.rs_drag / cosmo.h
        else:
            scale = scale_Hrd

        if cosmo is not None:
            DH_over_rd, DM_over_rd = predict_bao(z, cosmo=cosmo, apmode='qparqper', scale='distance')
            Hrd_theory = factor / DH_over_rd
        
        def compute_Hrd(z, samples):
            from astropy.cosmology import Flatw0waCDM
            from tqdm import tqdm
            one = np.ones(len(samples['omegam']))
            params = {'H0': one * cosmo.H0, 'omegam': None, 'w': -1. * one, 'wa': 0. * one, 'rdrag': None}
            values = np.array([samples[param] if param in samples.paramNames.list() else params[param] for param in params]).T
            return np.array([Flatw0waCDM(H0, Om0, w0=w0, wa=wa, Tcmb0=2.72548).H(z).value * rd for H0, Om0, w0, wa, rd in tqdm(values)])

        fig, lax = plt.subplots(3, 1, figsize=(6, 5.5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.})
        if colors is None:
            colors = ['C{:d}'.format(i) for i in range(len(samples))]
        ax_color = 'darkred'

        for (label, sample), ax, c in zip(samples.items(), lax, colors):
            # Create twin axes for Z on the right side
            ax2 = ax.twinx() if show_Hrd else ax
            Hrd = compute_Hrd(z, sample)
            utils.plot_fill_between(ax, z, Hrd / (1 + z) / scale, color=c, alpha=0.4, lw=1)
            if cosmo is not None:
                ax.plot(z, Hrd_theory / (1 + z) / scale, label=label_cosmo, ls='--', c=ax_color, lw=1.)

            if data is not None:
                offsets, ioff = [300, -810, 400, -900, 300], 0
                for key, dd in data.items():
                    color = self.colors[key]
                    tracer = key[0][:3]
                    if tracer in ['LRG', 'ELG']:
                        label = '{0}\n${1[0]:.1f} < z < {1[1]:.1f}$'.format(tracer, key[1])
                    elif tracer == 'Lya':
                        label = r'Ly-$\alpha$'
                    else:
                        label = tracer
                    try:
                        dd = convert_bao_fisher(dd, apmode='qparqper', scale='distance')
                    except ValueError as exc:
                        continue  # isotropic fit
                    offset = offsets[ioff]
                    ioff += 1
                    mean, std = dd.mean('qpar'), dd.std('qpar')
                    std = factor / mean**2 * std
                    mean = factor / mean
                    zeff = dd.attrs['zeff']
                    ax.errorbar(zeff, mean / (1 + zeff) / scale, std / (1 + zeff) / scale, fmt='.', color=color)
                    lax[0].text(zeff, (mean / (1 + zeff) + offset) / scale, label, color=color, horizontalalignment='center', fontsize='small')

            if cosmo is not None and label_cosmo and ax is lax[0]:
                ax.legend(fontsize='medium', loc='lower right', frameon=False)

            # Set ticks on both axes
            ax.set_xticks(np.arange(0, 2.5, 0.5))
            ax.set_xticks(np.arange(0, 2.5, 0.1), minor=True)

            Hrd_lim = (7850, 11100 - 0.5)
            if axis_H == 'left':
                ax_H, ax_Hrd = ax, ax2
            else:
                ax_H, ax_Hrd = ax2, ax
            if show_Hrd:
                ax_Hrd.set_ylim(Hrd_lim[0] / scale_Hrd, Hrd_lim[1] / scale_Hrd)
                ax_Hrd.set_yticks(np.arange(7500 / scale_Hrd, 12000 / scale_Hrd, 500 / scale_Hrd))
                ax_Hrd.set_yticks(np.arange(7500 / scale_Hrd, 12000 / scale_Hrd, 100 / scale_Hrd), minor=True)
                # ax_Hrd.grid(which='both', alpha=0.2)

            H_lim = np.array(Hrd_lim) / (cosmo.rs_drag / cosmo.h)
            ax_H.text(0.1, 73, label, fontsize='large')
            ax_H.set_ylim(*H_lim)
            ax_H.tick_params(axis='y', labelcolor=ax_color)
            ax_H.set_yticks(np.arange(55., 80., 5))
            ax_H.set_yticks(np.arange(53., 76., 1), minor=True)

        lax[-1].set_xlim(0, 2.5)
        lax[-1].set_xlabel(r'$z$', fontsize='xx-large')

        label_H = r'$H(z)r_\mathrm{d}/r_\mathrm{d}^\mathrm{Planck}/(1+z)~[\rm km\ s^{-1}Mpc^{-1}]$'
        label_Hrd = r'$H(z)r_\mathrm{{d}}{}/(1+z)~[\rm km\ s^{{-1}}]$'.format('/{}'.format(scale_Hrd) if scale_Hrd != 1 else '')
        if axis_H == 'left':
            if show_Hrd: fig.text(1.0, 0.5, label_Hrd, va='center', rotation=90, fontsize='xx-large')  # add label on the right y-axis
            fig.supylabel(label_H, color=ax_color, fontsize='xx-large')
        else:
            fig.text(1.0, 0.5, label_H, color=ax_color, va='center', rotation=90, fontsize='xx-large')
            fig.supylabel(label_Hrd, fontsize='xx-large')
        fig.tight_layout()
        return fig

    def plot_bao_data_v1(self, data=None, cosmo=None, zlim=(0.01, 2.5), ls='-', marker='.', 
                         ms=14, label_cosmo=None, label_data=False, figsize=(15, 6), fig=None):
        """
        Plot BAO data in terms of DV/rd and FAP and model, version 1 (in terms of DV/rd). From Sesh Nadathur.
        
        Parameters
        ----------
        data : dict, default=None
            Dictionary mapping data to :class:`FisherLikelihood`.
        
        cosmo : Cosmology, default=None
            Cosmology to use for DV/rd and FAP theory curves.
        
        zlim : tuple, default=(0.01, 2.5)
            z-limits.
        
        ls : str, default='-'
            Line style.
        
        marker : str, default='.'
            Marker for data points.
        
        ms : int, default=14
            Marker size.
        
        label_cosmo : str, default=None
            If ``cosmo`` is provided, use this label.
        
        figsize : tuple, default=(15, 6)
            Figure size, if ``fig`` not provided.

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
        from desi_y1_files.cosmo_tools import predict_bao, convert_bao_fisher
        z = np.linspace(zlim[0], zlim[1], 100)
        DV_over_rd, FAP = predict_bao(z, cosmo=cosmo, apmode='qisoqap', scale='distance')

        if fig is None:
            fig, lax = plt.subplots(1, 2, figsize=figsize, squeeze=True)
        else:
            lax = fig.axes

        lax[0].plot(z, DV_over_rd / z**(2 / 3), c='dimgrey', lw=1.1, label=label_cosmo, ls=ls)
        lax[1].plot(z, FAP / z, c='dimgrey', lw=1.1, label=label_cosmo, ls=ls)

        if data is not None:
            for key, dd in data.items():
                tracer = key[0][:3]
                if label_data is True:
                    if tracer in ['LRG', 'ELG']:
                        label = '{0} ${1[0]:.1f} < z < {1[1]:.1f}$'.format(tracer, key[1])
                    elif tracer == 'Lya':
                        label = r'Ly-$\alpha$'
                    else:
                        label = tracer
                else:
                    label = None

                try:
                    dd = convert_bao_fisher(dd, apmode='qisoqap', scale='distance')
                except ValueError as exc:
                    dd = convert_bao_fisher(dd, apmode='qiso', scale='distance')

                kw = dict(c=self.colors[key], marker=marker, markersize=ms, elinewidth=2, label=label)
                zeff = dd.attrs['zeff']
                mean = dd.mean('qiso') / zeff**(2 / 3)  # in fact, DV
                std = dd.std('qiso') / zeff**(2 / 3)
                eb = lax[0].errorbar(zeff, mean, yerr=std, **kw)

                if 'qap' in dd.params():  # in fact, FAP
                    mean = dd.mean('qap') / zeff
                    std = dd.std('qap') / zeff
                    eb = lax[1].errorbar(zeff, mean, yerr=std, **kw)

        if label_data or label_cosmo:
            lax[0].legend(frameon=False, numpoints=1, fontsize=14)
        lax[0].set_xlim(*zlim)
        lax[0].set_ylim(17, 21.5)
        lax[0].set_ylabel(r'$D_{\rm V}/(r_{\rm d}z^{2/3})$', fontsize=18)
        lax[1].set_xlim(*zlim)
        lax[1].set_ylabel(r'$D_{\rm M}/(zD_{\rm H})$', fontsize=18)
        for ax in lax:
            ax.set_xlabel(r'Redshift $z$', fontsize=18)
            ax.tick_params(labelsize=16)
        return fig

    def plot_bao_data_v2(self, data, cosmo_best=None, cosmo_alt=None, zlim=(0.01, 2.5), ylim=(0.9, 1.05),
                         ls='--', marker='.', ms=14, label_cosmo_alt=None, label_data=False, figsize=(15, 6)):
        """
        Plot BAO data in terms of DV/rd and FAP and model, version 2 (in terms of DV/rd / (DV/rd)_best). From Sesh Nadathur.
        
        Parameters
        ----------
        data : dict, default=None
            Dictionary mapping data to :class:`FisherLikelihood`.
        
        cosmo_best : Cosmology, default=None
            Cosmology to use for (DV/rd)_best and FAP_best.
    
        cosmo_alt : Cosmology, default=None
            Cosmology to use for DV/rd and FAP theory curves.
        
        zlim : tuple, default=(0.01, 2.5)
            z-limits.
        
        ylim : tuple, default=(0.9, 1.05)
            Limits for y-axis.
        
        ls : str, default='--'
            Line style.
        
        marker : str, default='.'
            Marker for data points.
        
        ms : int, default=14
            Marker size.
        
        label_cosmo_alt : str, default=None
            If ``cosmo_alt`` is provided, use this label.
        
        figsize : tuple, default=(15, 6)
            Figure size, if ``fig`` not provided.

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
        from cosmoprimo.fiducial import DESI
        fiducial = DESI()
        no_cosmo_best = cosmo_best is None
        if no_cosmo_best:
            import warnings
            warnings.warn('no best-fit cosmology provided, defaulting to fiducial')
            cosmo_best = fiducial
        cosmo_alt = cosmo_best if cosmo_best is None else cosmo_alt

        from desi_y1_files.cosmo_tools import predict_bao, convert_bao_fisher
        z = np.linspace(zlim[0], zlim[1], 100)

        DV_over_rd, FAP = predict_bao(z, cosmo=cosmo_best, apmode='qisoqap', scale='distance')
        DV_over_rd_alt, FAP_alt = predict_bao(z, cosmo=cosmo_alt, apmode='qisoqap', scale='distance')

        fig, lax = plt.subplots(1, 2, figsize=figsize, squeeze=True)

        if data is not None:
            # solid lines for best-fit cosmology, =1 at all redshifts
            lax[0].plot(z, np.ones_like(z), c='dimgrey', lw=1.1, ls='-')
            lax[1].plot(z, np.ones_like(z), c='dimgrey', lw=1.1, ls='-')
        # lines with desired linestyle for the alternative cosmology
        lax[0].plot(z, DV_over_rd_alt / DV_over_rd, c='dimgrey', lw=1.1, label=label_cosmo_alt, ls=ls) 
        lax[1].plot(z, FAP_alt / FAP, c='dimgrey', lw=1.1, label=label_cosmo_alt, ls=ls)

        if data is not None:
            for key, dd in data.items():
                tracer = key[0][:3]
                if label_data is True:
                    if tracer in ['LRG', 'ELG']:
                        label = '{0} ${1[0]:.1f} < z < {1[1]:.1f}$'.format(tracer, key[1])
                    elif tracer == 'Lya':
                        label = r'Ly-$\alpha$'
                    else:
                        label = tracer
                else:
                    label = None

                try:
                    dd = convert_bao_fisher(dd, apmode='qisoqap', scale='distance')
                except ValueError as exc:
                    dd = convert_bao_fisher(dd, apmode='qiso', scale='distance')

                kw = dict(c=self.colors[key], marker=marker, markersize=ms, elinewidth=2, label=label)
                zeff = dd.attrs['zeff']
                DV_over_rd, FAP = predict_bao(zeff, cosmo=cosmo_best, apmode='qisoqap', scale='distance')
                mean = dd.mean('qiso') / DV_over_rd  # in fact, DV
                std = dd.std('qiso') / DV_over_rd
                eb = lax[0].errorbar(zeff, mean, yerr=std, **kw)

                if 'qap' in dd.params():  # in fact, FAP
                    mean = dd.mean('qap') / FAP
                    std = dd.std('qap') / FAP
                    eb = lax[1].errorbar(zeff, mean, yerr=std, **kw)

        if label_data or label_cosmo_alt:
            lax[0].legend(frameon=False, numpoints=1, fontsize=14, loc='lower right')
        lax[0].set_ylim(*ylim)
        if no_cosmo_best:
            # everything has been plotted relative to DESI fiducial cosmology
            lax[0].set_ylabel(r'$(D_{\rm V}/r_{\rm d})/(D_{\rm V}/r_{\rm d})^{\rm fid}$', fontsize=18)
            lax[1].set_ylabel(r'$F_{\rm AP}/F_{\rm AP}^{\rm fid}$', fontsize=18)
        else:
            # everything has been plotted relative to provided best-fit cosmology
            lax[0].set_ylabel(r'$(D_{\rm V}/r_{\rm d})/(D_{\rm V}/r_{\rm d})^{\rm best}$', fontsize=18)
            lax[1].set_ylabel(r'$F_{\rm AP}/F_{\rm AP}^{\rm best}$', fontsize=18)

        for ax in lax:
            ax.set_xlim(*zlim)
            ax.set_xlabel(r'Redshift $z$', fontsize=18)
            ax.tick_params(labelsize=16)
            # ax.axhline(1, c='grey', ls=':')
        return fig
    

class KP7StylePaper(KP7Style):

    """KP7 style for papers."""

    def __init__(self, **kwargs):
        super(KP7StylePaper, self).__init__(**{'lines.linewidth': 2, 'axes.labelsize': 14, **kwargs})
        import seaborn as sns
        idx = [0, 1, 2, 4, 5, 3]
        self.contour_colors = list(np.array(sns.color_palette('colorblind').as_hex())[idx])
        self.contour_colors[1] = 'darkorange'
        from getdist import plots
        # GetDist settings
        settings = plots.GetDistPlotSettings()
        settings.rc_sizes()
        settings.legend_frame = False
        settings.figure_legend_frame = True
        settings.prob_label = r'$P/P_{\rm max}$'
        settings.norm_prob_label = 'Probability density'
        settings.prob_y_ticks = True
        settings.alpha_filled_add = 0.85
        settings.solid_contour_palefactor = 0.6
        settings.solid_colors = self.contour_colors
        settings.axis_marker_lw = 0.6
        settings.linewidth_contour = 1.2
        settings.alpha_factor_contour_lines = 1
        settings.colorbar_axes_fontsize = 8
        settings.axes_fontsize = 16
        settings.axes_labelsize = 18
        settings.legend_fontsize = 14
        # This is how to override default labels for parameters with specified names
        settings.param_names_for_labels = os.path.normpath(os.path.join(os.path.dirname(__file__), 'kp7.paramnames'))
        self.settings = settings
        self.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman'})
        """
        self.legend_loc = 'upper right'
        self.legend_fontsize = 20
        self.legend_frameon = True
        self.legend_edgecolor = 'none'

        self.label_fontsize = 28
        self.tick_labelsize = 20

        self.axhline_color = 'black'
        self.axvline_color = 'black'
        """


class KP7StylePresentation(KP7Style):

    """KP7 style for presentations."""

    def __init__(self, **kwargs):
        super(KP7StylePresentation, self).__init__(**{'lines.linewidth': 4, 'axes.labelsize': 18, **kwargs})
