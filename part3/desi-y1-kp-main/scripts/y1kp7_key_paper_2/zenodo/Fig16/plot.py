from matplotlib import pyplot as plt
from getdist import MCSamples, plots


class FakeMCSamples(MCSamples):
    @classmethod
    def from_getdist(cls, samples, params):
        from copy import deepcopy
        new = cls()
        new.paramNames = deepcopy(samples.paramNames)
        if hasattr(samples, 'label'):
            new.label = samples.label
        new._density1d = {
            (name, False): samples.get1DDensityGridData(name, meanlikes=False)
            for name in params
        }
        new._density2d = {
            (name1, name2, 2, False): samples.get2DDensityGridData(name1, name2, num_plot_contours=2, meanlikes=False)
            for i, name1 in enumerate(params) for name2 in params[i:] if name1 != name2
        }
        return new

    def get1DDensityGridData(self, j, meanlikes=False):
        return self._density1d[j, meanlikes]

    def get2DDensityGridData(self, j, j2, num_plot_contours=None, meanlikes=False):
        return self._density2d[j, j2, num_plot_contours, meanlikes]

    def save(self, filename):
        import numpy as np
        np.save(filename, self)

    @classmethod
    def load(cls, filename):
        import numpy as np
        return np.load(filename, allow_pickle=True)[()]


def get_settings():
    settings = plots.GetDistPlotSettings()
    settings.rc_sizes()
    settings.legend_frame = False
    settings.figure_legend_frame = False
    settings.prob_label = r'$P/P_{\rm max}$'
    settings.norm_prob_label = 'Probability density'
    settings.prob_y_ticks = True
    settings.alpha_filled_add = 0.85
    settings.solid_contour_palefactor = 0.6
    settings.axis_marker_lw = 0.6
    settings.linewidth_contour = 1.2
    settings.alpha_factor_contour_lines = 1
    settings.colorbar_axes_fontsize = 8
    settings.axes_fontsize = 16
    settings.axes_labelsize = 18
    settings.legend_fontsize = 14
    import matplotlib as mpl
    mpl.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman'})
    return settings


tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO', 'ALL']
samples_fns = [f'contours_samples_{tracer}.npy' for tracer in tracers]
samples = [FakeMCSamples.load(sample_fn) for sample_fn in samples_fns]

colors = ['yellowgreen', 'orange', 'orangered', 'firebrick', 'blue', 'seagreen', 'k']

g = plots.getSinglePlotter(width_inch=5, ratio=1 / 1.2, scaling=True)
g.settings.__dict__.update(get_settings().__dict__)
g.settings.alpha_factor_contour_lines = 0.6
g.plot_2d(samples, ['omegam', 'sigma8'], filled=True,
          alphas=0.85, colors=colors)
g.add_legend(legend_labels=['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO', 'All'],
             bbox_to_anchor=[0.67, 1.02], legend_loc='upper left')
g.add_text(r'$\Lambda$CDM', 0.05, 0.05, fontsize=20)

ax = g.get_axes()
#ax.set_xlabel(r'$\Omega_{\mathrm{m}}$')
ax.tick_params(axis='both', which='major', labelsize=g.settings.axes_fontsize)
ax.set_ylim(0.4, 1.2)
ax.set_xlim(0.1, 0.6)

plt.savefig('Omegam_sigma8_sample_all.pdf', bbox_inches='tight', dpi=360)


g = plots.getSinglePlotter(width_inch=5, ratio=1 / 1.2, scaling=True)
g.settings.__dict__.update(get_settings().__dict__)
g.settings.alpha_factor_contour_lines = 0.6
g.plot_2d(samples, ['omegam', 'H0'], filled=True,
          alphas=0.85,
          colors=colors)

g.add_legend(legend_labels=['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO', 'All'],
             bbox_to_anchor=[0.67, 1.02], legend_loc='upper left')

g.add_text(r'$\Lambda$CDM', 0.05, 0.05, fontsize=20)

ax = g.get_axes()
ax.set_ylabel(r'$H_0 \, [\mathrm{km} \, \mathrm{s}^{-1} \, \mathrm{Mpc}^{-1}]$')
ax.tick_params(axis='both', which='major', labelsize=g.settings.axes_fontsize)
ax.set_ylim(60., 80.)
ax.set_xlim(0.1, 0.6)

plt.savefig('Omegam_H0_sample_all.pdf', bbox_inches='tight', dpi=360)