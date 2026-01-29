import numpy as np
from pathlib import Path
from desilike.samples import Chain
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from desilike.samples import Profiles

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def get_profile_bestfit(filename, param='qiso'):
    profiles = Profiles.load(filename)
    idx = profiles.bestfit.logposterior.argmax()
    bestfit = profiles.bestfit[param][idx]
    error = profiles.error[param][idx]
    return bestfit, error

def get_chain_bestfit(filename, param='qiso'):
    if isinstance(filename, list):
        chains = []
        for fn in filename:
            chains.append(Chain.load(fn))
        chain = chains[0].concatenate([chain.remove_burnin(0.5)[::10] for chain in chains])
    else:
        chain = Chain.load(filename)
        chain = chain.remove_burnin(0.5)[::10]
    bestfit = chain.argmax(param)
    error = chain.std(param)
    return bestfit, error

zranges = {
    'BGS_BRIGHT-21.5': [[0.1, 0.4]],
    'LRG': [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
    'ELG_LOPnotqso': [[0.8, 1.1], [1.1, 1.6]],
    'QSO': [[0.8, 2.1]],
}

smoothing_scales = {'BGS_BRIGHT-21.5': 15, 'LRG': 15, 'ELG_LOPnotqso': 15, 'QSO': 30,}

colors = {'BGS_BRIGHT-21.5': 'black', ('BGS_BRIGHT-21.5', (0.1, 0.4)): 'darkgoldenrod',
          'LRG': 'red', ('LRG', (0.4, 0.6)): 'orange', ('LRG', (0.6, 0.8)): 'orangered', ('LRG', (0.8, 1.1)): 'firebrick', 
          ('ELG_LOPnotqso', (0.8, 1.1)): 'lightskyblue', ('ELG_LOPnotqso', (1.1, 1.6)): 'steelblue',
          ('QSO', (0.8, 2.1)): 'seagreen', ('Lya', (0.8, 3.5)): 'purple'}

labels = {'BGS_BRIGHT-21.5': r'${\tt BGS}$', ('BGS_BRIGHT-21.5', (0.1, 0.4)): r'${\tt BGS}$',
          'LRG': r'${\tt LRG}$', ('LRG', (0.4, 0.6)): r'${\tt LRG1}$', ('LRG', (0.6, 0.8)): r'${\tt LRG2}$', ('LRG', (0.8, 1.1)): r'${\tt LRG3}$', 
          ('ELG_LOPnotqso', (0.8, 1.1)): r'${\tt ELG1}$', ('ELG_LOPnotqso', (1.1, 1.6)): r'${\tt ELG2}$',
          ('QSO', (0.8, 2.1)): r'${\tt QSO}$', ('Lya', (0.8, 3.5)): r'${\tt Lya}$'}

# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[colors[i],
#                                              lighten_color(colors[i])])

sigmapar = {'BGS_BRIGHT-21.5':{'pre': 10.0, 'post': 8.0}, 'LRG':{'pre': 9.0, 'post': 6.0}, 'ELG_LOPnotqso':{'pre': 8.5, 'post': 6.0}, 'QSO': {'pre': 9.0, 'post': 6.0}}
sigmaper = {'BGS_BRIGHT-21.5':{'pre':6.5, 'post':3.0}, 'LRG':{'pre':4.5, 'post':3.0}, 'ELG_LOPnotqso':{'pre': 4.5, 'post': 3.0}, 'QSO': {'pre': 3.5, 'post': 3.0}}
sigmas = {'BGS_BRIGHT-21.5':{'pre': 2.0, 'post': 2.0}, 'LRG':{'pre': 2.0, 'post': 2.0}, 'ELG_LOPnotqso':{'pre': 2.0, 'post': 2.0}, 'QSO': {'pre': 2.0, 'post': 2.0}}


parser = argparse.ArgumentParser()
parser.add_argument('--tracer', help='tracer to be selected', type=str, default='LRG_ffa')
parser.add_argument('--region', help='regions; by default, run on all regions', type=str, choices=['NGC','SGC', 'GCcomb'], default='GCcomb')
parser.add_argument('--zlim', help='z-limits, or options for z-limits, e.g. "highz", "lowz"', type=str, nargs='*', default=None)
parser.add_argument('--zmin', help='minimum redshift', type=float, default=0.4)
parser.add_argument('--zmax', help='maximum redshift', type=float, default=0.6)
parser.add_argument('--ells', help='multipoles to be used', type=int, nargs='*', default=[0, 2,])
parser.add_argument('--recon_algorithm', help='reconstruction method', type=str, default='')
parser.add_argument('--recon_mode', help='reconstruction convention', type=str, choices=['recsym', 'reciso'], default='')
parser.add_argument('--smoothing_radius', help='smoothing radius', type=int, default=10)
parser.add_argument('--free_damping', help='free damping parameters', action='store_true')
parser.add_argument('--npoly', help='number of polynomial terms', type=int, default=3, choices=[3, 5])
parser.add_argument('--apmode', help='AP parametrization', type=str, default='qisoqap')
parser.add_argument('--only_now', help='use no-wiggles power spectrum', action='store_true')
parser.add_argument('--nphases', help='phase of the mocks', type=int, default=25)
parser.add_argument('--fit_mean', help='fit the mean of the mocks', action='store_true')
parser.add_argument('--broadband', help='method to model broadband', type=str, default='pcs2')
parser.add_argument('--fitting_method', help='method to fit', type=str, choices=['profiling', 'sampling'], default='profiling')
args = parser.parse_args()

tracer = args.tracer
region = args.region
zmin = args.zmin
zmax = args.zmax
smin, smax = 50, 150
smoothing_radius = smoothing_scales[tracer]
color = colors[(tracer, (zmin, zmax))]
label = labels[(tracer, (zmin, zmax))]


base_dir = '/global/homes/e/epaillas/desi/users/epaillas/Y1/mocks/SecondGenMocks/AbacusSummit'

fig, ax = plt.subplots(figsize=(4, 4))
markers = ['o', 'o']
qiso_bestfit_pre = []
qiso_bestfit_post = []
qiso_error_pre = []
qiso_error_post = []
qap_bestfit_pre = []
qap_bestfit_post = []
qap_error_pre = []
qap_error_post = []
for phase in range(args.nphases):
    if args.fitting_method == 'profiling':
        profiles_dir = f'{base_dir}/v3_1/altmtl/fits_bao/{phase}/fits_correlation_{args.apmode}_pcs2/'
        profiles_fn = Path(profiles_dir) / f"profiles_{tracer}_GCcomb_{zmin}_{zmax}_sigmas{sigmas[tracer]['pre']}_sigmapar{sigmapar[tracer]['pre']}_sigmaper{sigmaper[tracer]['pre']}.npy"
        qiso_bestfit_pre_i, qiso_error_pre_i = get_profile_bestfit(profiles_fn, param='qiso')

        profiles_dir = f'{base_dir}/v3_1/altmtl/fits_bao/{phase}/fits_correlation_{args.apmode}_pcs2/recon_IFFT_recsym_sm{smoothing_radius}/'
        profiles_fn = Path(profiles_dir) / f"profiles_{tracer}_GCcomb_{zmin}_{zmax}_sigmas{sigmas[tracer]['post']}_sigmapar{sigmapar[tracer]['post']}_sigmaper{sigmaper[tracer]['post']}.npy"
        qiso_bestfit_post_i, qiso_error_post_i = get_profile_bestfit(profiles_fn, param='qiso')
    else:
        chain_dir = f'{base_dir}/v3_1/altmtl/fits_bao/{phase}/fits_correlation_{args.apmode}_pcs2/'
        chain_fn = [Path(chain_dir) / f"chain_{tracer}_GCcomb_{zmin}_{zmax}_sigmas{sigmas[tracer]['pre']}_sigmapar{sigmapar[tracer]['pre']}_sigmaper{sigmaper[tracer]['pre']}_{i}.npy" for i in range(8)]
        qiso_bestfit_pre_i, qiso_error_pre_i = get_chain_bestfit(chain_fn, param='qiso')

        chain_dir = f'{base_dir}/v3_1/altmtl/fits_bao/{phase}/fits_correlation_{args.apmode}_pcs2/recon_IFFT_recsym_sm{smoothing_radius}/'
        chain_fn = [Path(chain_dir) / f"chain_{tracer}_GCcomb_{zmin}_{zmax}_sigmas{sigmas[tracer]['post']}_sigmapar{sigmapar[tracer]['post']}_sigmaper{sigmaper[tracer]['post']}_{i}.npy" for i in range(8)]
        qiso_bestfit_post_i, qiso_error_post_i = get_chain_bestfit(chain_fn, param='qiso')

    qiso_bestfit_pre.append(qiso_bestfit_pre_i)
    qiso_bestfit_post.append(qiso_bestfit_post_i)
    qiso_error_pre.append(qiso_error_pre_i)
    qiso_error_post.append(qiso_error_post_i)

qiso_bestfit_pre = np.array(qiso_bestfit_pre)
qiso_bestfit_post = np.array(qiso_bestfit_post)
qiso_error_pre = np.array(qiso_error_pre)
qiso_error_post = np.array(qiso_error_post)

ax.scatter(qiso_bestfit_post - qiso_bestfit_pre, qiso_error_post/qiso_error_pre, s=10.0, marker='o', color=color)

ax.errorbar(np.mean(qiso_bestfit_post - qiso_bestfit_pre), np.mean(qiso_error_post/qiso_error_pre),
                xerr=np.std(qiso_bestfit_post - qiso_bestfit_pre), yerr=np.std(qiso_error_post/qiso_error_pre),
                marker='o', capsize=2.0, ls='', color=color, markeredgewidth=1.0, mfc='w')

if args.fitting_method == 'profiling':
    # read pre-recon fits
    profiles_dir = f'/global/homes/e/epaillas/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/fits_correlation_{args.apmode}_pcs2/'
    profiles_fn = Path(profiles_dir) / f"profiles_{tracer}_{region}_{zmin}_{zmax}_sigmas{sigmas[tracer]['pre']}_sigmapar{sigmapar[tracer]['pre']}_sigmaper{sigmaper[tracer]['pre']}.npy"
    qiso_bestfit_pre, qiso_error_pre = get_profile_bestfit(profiles_fn, param='qiso')

    # read post-recon fits
    profiles_dir = f'/global/homes/e/epaillas/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/fits_correlation_{args.apmode}_pcs2/recon_IFFT_recsym_sm{smoothing_radius}/'
    profiles_fn = Path(profiles_dir) / f"profiles_{tracer}_{region}_{zmin}_{zmax}_sigmas{sigmas[tracer]['post']}_sigmapar{sigmapar[tracer]['post']}_sigmaper{sigmaper[tracer]['post']}.npy"
    qiso_bestfit_post, qiso_error_post = get_profile_bestfit(profiles_fn, param='qiso')
else:
    # read pre-recon fits
    chain_dir = f'/global/homes/e/epaillas/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/fits_correlation_{args.apmode}_pcs2/'
    chain_fn = [Path(chain_dir) / f"chain_{tracer}_{region}_{zmin}_{zmax}_sigmas{sigmas[tracer]['pre']}_sigmapar{sigmapar[tracer]['pre']}_sigmaper{sigmaper[tracer]['pre']}_{i}.npy" for i in range(8)]
    qiso_bestfit_pre, qiso_error_pre = get_chain_bestfit(chain_fn, param='qiso')

    # read post-recon fits
    chain_dir = f'/global/homes/e/epaillas/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/fits_correlation_{args.apmode}_pcs2/recon_IFFT_recsym_sm{smoothing_radius}/'
    chain_fn = [Path(chain_dir) / f"chain_{tracer}_{region}_{zmin}_{zmax}_sigmas{sigmas[tracer]['post']}_sigmapar{sigmapar[tracer]['post']}_sigmaper{sigmaper[tracer]['post']}_{i}.npy" for i in range(8)]
    qiso_bestfit_post, qiso_error_post = get_chain_bestfit(chain_fn, param='qiso')

ax.plot(qiso_bestfit_post - qiso_bestfit_pre, qiso_error_post/qiso_error_pre,
            marker='*', color=color, ls='', ms=10.0, mfc='k', markeredgewidth=1.0,
            )

ax.plot(np.nan, np.nan, label='DESI', marker='*', color='k', ls='', ms=10.0, mfc='k', markeredgewidth=1.0,)
ax.errorbar(np.nan, np.nan, xerr=np.nan, yerr=np.nan,
            markeredgewidth=1.0, mfc='w', color='k',
            marker='o', capsize=2.0, ls='',
            label=r'{\tt AbacusSummit}')

ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel(r'$\alpha_{\rm iso}\, \textrm{post} - \alpha_{\rm iso}\, \textrm{pre}$', fontsize=17)
ax.set_ylabel(r'$\sigma_{\alpha_{\rm iso}}\, \textrm{post}/\sigma_{\alpha_{\rm iso}}\, \textrm{pre}$', fontsize=17)
title = f'{label}'
ax.annotate(title, xy=(0.83, 0.9), xycoords='axes fraction', fontsize=13, bbox=dict(boxstyle="round", fc='w', ec='lightgrey'))
ax.legend(frameon=True, fontsize=13, handletextpad=0.0, loc='upper left')
fig.tight_layout()
output_fn = f'fig/keypaper_qiso_scatter_{tracer}_{zmin}_{zmax}.pdf'
plt.savefig(output_fn, bbox_inches='tight')
plt.show()
