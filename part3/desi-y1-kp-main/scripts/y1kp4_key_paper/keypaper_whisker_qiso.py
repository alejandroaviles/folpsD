import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desilike.samples import Chain, Profiles
from getdist import plots as gdplt
from pathlib import Path
plt.style.use(['enrique-science'])


version = 'v1.2/unblinded'
base_dir = Path(f'/global/homes/e/epaillas/desi/users/epaillas/Y1/iron/{version}/fits_bao/')


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

colors = {'BGS_BRIGHT-21.5': 'k', ('BGS_BRIGHT-21.5', (0.1, 0.4)): 'darkkhaki',
                       'LRG': 'red', ('LRG', (0.4, 0.6)): 'orange', ('LRG', (0.6, 0.8)): 'orangered', ('LRG', (0.8, 1.1)): 'firebrick', 
                       ('ELG_LOPnotqso', (0.8, 1.1)): 'lightskyblue', ('ELG_LOPnotqso', (1.1, 1.6)): 'steelblue',
                       ('QSO', (0.8, 2.1)): 'seagreen', ('Lya', (0.8, 3.5)): 'purple', ('LRG+ELG_LOPnotqso', (0.8, 1.1)): 'slateblue'}

zranges = {
    'BGS_BRIGHT-21.5': [[0.1, 0.4]],
    'LRG': [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
    'ELG_LOPnotqso': [[0.8, 1.1], [1.1, 1.6]],
    'LRG+ELG_LOPnotqso': [[0.8, 1.1]],
    'QSO': [[0.8, 2.1]],
}

smoothing_scales = {'BGS_BRIGHT-21.5': 15, 'LRG': 15, 'ELG_LOPnotqso': 15, 'QSO': 30, 'LRG+ELG_LOPnotqso': 15}
smoothing_scales2 = {'BGS_BRIGHT-21.5': 10, 'LRG': 10, 'ELG_LOPnotqso': 10, 'QSO': 20, 'LRG+ELG_LOPnotqso': 10}

sigmapar = {'BGS_BRIGHT-21.5':{'pre': 10.0, 'post': 8.0}, 'LRG':{'pre': 9.0, 'post': 6.0}, 'ELG_LOPnotqso':{'pre': 8.5, 'post': 6.0}, 'QSO': {'pre': 9.0, 'post': 6.0}, 'LRG+ELG_LOPnotqso': {'pre': 9.0, 'post': 6.0}}
sigmaper = {'BGS_BRIGHT-21.5':{'pre':6.5, 'post':3.0}, 'LRG':{'pre':4.5, 'post':3.0}, 'ELG_LOPnotqso':{'pre': 4.5, 'post': 3.0}, 'QSO': {'pre': 3.5, 'post': 3.0}, 'LRG+ELG_LOPnotqso': {'pre': 4.5, 'post': 3.0}}
sigmas = {'BGS_BRIGHT-21.5':{'pre': 2.0, 'post': 2.0}, 'LRG':{'pre': 2.0, 'post': 2.0}, 'ELG_LOPnotqso':{'pre': 2.0, 'post': 2.0}, 'QSO': {'pre': 2.0, 'post': 2.0}, 'LRG+ELG_LOPnotqso': {'pre': 2.0, 'post': 2.0}}

apmodes = {('BGS_BRIGHT-21.5', (0.1, 0.4)): 'qiso',
          ('LRG', (0.4, 0.6)): 'qisoqap', ('LRG', (0.6, 0.8)): 'qisoqap', ('LRG', (0.8, 1.1)): 'qisoqap', 
          ('ELG_LOPnotqso', (0.8, 1.1)): 'qiso', ('ELG_LOPnotqso', (1.1, 1.6)): 'qisoqap',
          ('QSO', (0.8, 2.1)): 'qiso',
          ('LRG+ELG_LOPnotqso', (0.8, 1.1)): 'qisoqap'}

labels = {'BGS_BRIGHT-21.5': r'{\tt BGS}', ('BGS_BRIGHT-21.5', (0.1, 0.4)): r'{\tt BGS}',
          'LRG': r'{\tt LRG}', ('LRG', (0.4, 0.6)): r'{\tt LRG1}', ('LRG', (0.6, 0.8)): r'{\tt LRG2}', ('LRG', (0.8, 1.1)): r'{\tt LRG3}', 
          ('ELG_LOPnotqso', (0.8, 1.1)): r'{\tt ELG1}', ('ELG_LOPnotqso', (1.1, 1.6)): r'{\tt ELG2}', ('LRG+ELG_LOPnotqso', (0.8, 1.1)): r'{\tt LRG3+ELG1}',
          ('QSO', (0.8, 2.1)): r'{\tt QSO}', ('Lya', (0.8, 3.5)): r'{\tt Lya}'}

profile_choices = ['baseline', 'pre-recon', 'power\nspectrum', 'NGC', 'flat prior on\n'r'$\Sigma_s, \Sigma_{\parallel}, \Sigma_{\perp}$',
                   'polynomial\nbroadband', 'RecIso', 'smaller 'r'$\Sigma_{\rm sm}$']

param = 'qiso'

fig, ax = plt.subplots(1, 8, figsize=(14, 6))
ax_idx = 0
if param == 'qap':
    profile_choices = [i for i in profile_choices if '1D' not in i]
yvals = np.linspace(0, 10, len(profile_choices))[::-1]

for tracer in ['BGS_BRIGHT-21.5', 'LRG', 'LRG+ELG_LOPnotqso', 'ELG_LOPnotqso', 'QSO']:
    damping_str_post = f'sigmas{sigmas[tracer]["post"]}_sigmapar{sigmapar[tracer]["post"]}_sigmaper{sigmaper[tracer]["post"]}'
    damping_str_pre = f'sigmas{sigmas[tracer]["pre"]}_sigmapar{sigmapar[tracer]["pre"]}_sigmaper{sigmaper[tracer]["pre"]}'
    sm = smoothing_scales[tracer]
    sm2 = smoothing_scales2[tracer]

    for zrange in zranges[tracer]:
        zmin, zmax = zrange
        apmode = apmodes[(tracer,  (zmin, zmax))]

        profile_fns = {
            'baseline': f'fits_correlation_{apmode}_pcs2/recon_IFFT_recsym_sm{sm}/profiles_{tracer}_GCcomb_{zmin}_{zmax}_{damping_str_post}.npy',
            'pre-recon': f'fits_correlation_{apmode}_pcs2/profiles_{tracer}_GCcomb_{zmin}_{zmax}_{damping_str_pre}.npy',
            'power\nspectrum': f'fits_power_{apmode}_pcs/recon_IFFT_recsym_sm{sm}/profiles_{tracer}_GCcomb_{zmin}_{zmax}_{damping_str_post}.npy',
            'NGC': f'fits_correlation_{apmode}_pcs2/recon_IFFT_recsym_sm{sm}/profiles_{tracer}_NGC_{zmin}_{zmax}_{damping_str_post}.npy',
            'flat prior on\n'r'$\Sigma_s, \Sigma_{\parallel}, \Sigma_{\perp}$': f'fits_correlation_{apmode}_pcs2/recon_IFFT_recsym_sm{sm}/profiles_{tracer}_GCcomb_{zmin}_{zmax}.npy',
            'polynomial\nbroadband': f'fits_correlation_{apmode}_power3/recon_IFFT_recsym_sm{sm}/profiles_{tracer}_GCcomb_{zmin}_{zmax}_{damping_str_post}.npy',
            'RecIso': f'fits_correlation_{apmode}_power3/recon_IFFT_reciso_sm{sm}/profiles_{tracer}_GCcomb_{zmin}_{zmax}_{damping_str_post}.npy',
            'smaller 'r'$\Sigma_{\rm sm}$': f'fits_correlation_{apmode}_pcs2/recon_IFFT_recsym_sm{sm2}/profiles_{tracer}_GCcomb_{zmin}_{zmax}_{damping_str_post}.npy',
        }
        for i, choice in enumerate(profile_choices):
            profile_fn = base_dir / profile_fns[choice]
            if not os.path.exists(profile_fn): continue
            profile = Profiles.load(profile_fn)
            idx_best = profile.bestfit.logposterior.argmax()

            c = colors[tracer, (zmin, zmax)]

            ax[ax_idx].errorbar(profile.bestfit[param][idx_best], yvals[i], xerr=profile.error[param][idx_best],
                                marker='o', capsize=3, color=c,
                                ms=5.5, markerfacecolor=lighten_color(c), markeredgecolor=c,
                                elinewidth=1.5)
            if i == 0:
                ax[ax_idx].fill_betweenx(yvals, profile.bestfit[param][idx_best] - profile.error[param][idx_best],
                                        profile.bestfit[param][idx_best] + profile.error[param][idx_best],
                                        color=c, alpha=0.075)
        xlabel = r'$\alpha_{\rm iso}$' if param == 'qiso' else r'$\alpha_{\rm AP}$'
        ax[ax_idx].set_xlabel(xlabel, fontsize=20)
        ax[ax_idx].tick_params(axis='x', labelsize=15, rotation=45)
        if ax_idx > 0:
            ax[ax_idx].axes.get_yaxis().set_visible(False)
        else:
            ax[ax_idx].set_yticks(yvals)
            ax[ax_idx].set_yticklabels(profile_choices, minor=False, rotation=0, fontsize=17)

        title = labels[(tracer, (zmin, zmax))]
        ax[ax_idx].set_title(title, fontsize=17)
        ax_idx += 1
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
plt.savefig(f"fig/Y1unblindedwhiskeraiso.pdf", bbox_inches='tight')
plt.show()