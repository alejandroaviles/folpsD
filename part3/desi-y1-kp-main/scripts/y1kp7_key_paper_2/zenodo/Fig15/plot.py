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


samples_fns = [f'contours_samples_{dataset}.npy' for dataset in ['fs', 'fs_bao', 'bao']]
fs, fs_bao, bao = [FakeMCSamples.load(sample_fn) for sample_fn in samples_fns]

for samples in [fs, fs_bao, bao]:
    samples.paramNames.parWithName('H0').label = r'H_0 \, [\mathrm{km} \, \mathrm{s}^{-1} \, \mathrm{Mpc}^{-1}]'

g = plots.getSinglePlotter(width_inch=5, ratio=1 / 1.2, scaling=True)
g.settings.__dict__.update(get_settings().__dict__)
g.settings.legend_fontsize = 16

colors = ['darkorange', (0.8, 0.47058823529411764, 0.7372549019607844), (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)]

g.triangle_plot([bao, fs, fs_bao], ['omegam', 'sigma8', 'H0', 'ns'],
                contour_colors=colors,
                filled=True, legend_labels=[r'BAO post-recon (no Ly$\alpha$)', r'Full-Modelling pre-recon', r'Joint analysis'], 
                alphas=0.85)

plt.savefig('triangle_fs_bao_all_nolya.pdf', bbox_inches='tight', dpi=360)