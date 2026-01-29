import numpy as np
from pathlib import Path
from desilike.samples import Chain
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
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

colors = {'BGS_BRIGHT-21.5': 'black', ('BGS_BRIGHT-21.5', (0.1, 0.4)): 'black',
          'LRG': 'red', ('LRG', (0.4, 0.6)): 'orange', ('LRG', (0.6, 0.8)): 'orangered', ('LRG', (0.8, 1.1)): 'firebrick', 
          ('ELG_LOPnotqso', (0.8, 1.1)): 'lightskyblue', ('ELG_LOPnotqso', (1.1, 1.6)): 'steelblue',
          ('QSO', (0.8, 2.1)): 'seagreen', ('Lya', (0.8, 3.5)): 'purple'}

labels = {'BGS_BRIGHT-21.5': r'${\tt BGS}$', ('BGS_BRIGHT-21.5', (0.1, 0.4)): r'${\tt BGS}$',
          'LRG': r'${\tt LRG}$', ('LRG', (0.4, 0.6)): r'${\tt LRG1}$', ('LRG', (0.6, 0.8)): r'${\tt LRG2}$', ('LRG', (0.8, 1.1)): r'${\tt LRG3}$', 
          ('ELG_LOPnotqso', (0.8, 1.1)): r'${\tt ELG1}$', ('ELG_LOPnotqso', (1.1, 1.6)): r'${\tt ELG2}$',
          ('QSO', (0.8, 2.1)): r'${\tt QSO}$', ('Lya', (0.8, 3.5)): r'${\tt Lya}$'}

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
free_damping = args.free_damping
smoothing_radius = smoothing_scales[tracer]
color = colors[(tracer, (zmin, zmax))]
label = labels[(tracer, (zmin, zmax))]


base_dir = '/global/homes/e/epaillas/desi/users/epaillas/Y1/mocks/SecondGenMocks/AbacusSummit'

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
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
        qap_bestfit_pre_i, qap_error_pre_i = get_profile_bestfit(profiles_fn, param='qap')

        profiles_dir = f'{base_dir}/v3_1/altmtl/fits_bao/{phase}/fits_correlation_{args.apmode}_pcs2/recon_IFFT_recsym_sm{smoothing_radius}/'
        profiles_fn = Path(profiles_dir) / f"profiles_{tracer}_GCcomb_{zmin}_{zmax}_sigmas{sigmas[tracer]['post']}_sigmapar{sigmapar[tracer]['post']}_sigmaper{sigmaper[tracer]['post']}.npy"
        qiso_bestfit_post_i, qiso_error_post_i = get_profile_bestfit(profiles_fn, param='qiso')
        qap_bestfit_post_i, qap_error_post_i = get_profile_bestfit(profiles_fn, param='qap')
    else:
        chain_dir = f'{base_dir}/v3_1/altmtl/fits_bao/{phase}/fits_correlation_{args.apmode}_pcs2/'
        chain_fn = [Path(chain_dir) / f"chain_{tracer}_GCcomb_{zmin}_{zmax}_sigmas{sigmas[tracer]['pre']}_sigmapar{sigmapar[tracer]['pre']}_sigmaper{sigmaper[tracer]['pre']}_{i}.npy" for i in range(8)]
        qiso_bestfit_pre_i, qiso_error_pre_i = get_chain_bestfit(chain_fn, param='qiso')
        qap_bestfit_pre_i, qap_error_pre_i = get_chain_bestfit(chain_fn, param='qap')

        chain_dir = f'{base_dir}/v3_1/altmtl/fits_bao/{phase}/fits_correlation_{args.apmode}_pcs2/recon_IFFT_recsym_sm{smoothing_radius}/'
        chain_fn = [Path(chain_dir) / f"chain_{tracer}_GCcomb_{zmin}_{zmax}_sigmas{sigmas[tracer]['post']}_sigmapar{sigmapar[tracer]['post']}_sigmaper{sigmaper[tracer]['post']}_{i}.npy" for i in range(8)]
        qiso_bestfit_post_i, qiso_error_post_i = get_chain_bestfit(chain_fn, param='qiso')
        qap_bestfit_post_i, qap_error_post_i = get_chain_bestfit(chain_fn, param='qap')

    if args.apmode == 'qisoqap':
        if np.isclose(qap_bestfit_pre_i, 0.8, rtol=5e-2) or np.isclose(qap_bestfit_pre_i, 1.2, rtol=5e-2)\
            or np.isclose(qap_bestfit_post_i, 0.8, rtol=5e-2) or np.isclose(qap_bestfit_post_i, 1.2, rtol=5e-2):
            print(f'phase {phase} has a bad fit')
            continue

    qiso_bestfit_pre.append(qiso_bestfit_pre_i)
    qiso_bestfit_post.append(qiso_bestfit_post_i)
    qiso_error_pre.append(qiso_error_pre_i)
    qiso_error_post.append(qiso_error_post_i)
    qap_bestfit_pre.append(qap_bestfit_pre_i)
    qap_bestfit_post.append(qap_bestfit_post_i)
    qap_error_pre.append(qap_error_pre_i)
    qap_error_post.append(qap_error_post_i)

qiso_bestfit_pre = np.array(qiso_bestfit_pre)
qiso_bestfit_post = np.array(qiso_bestfit_post)
qiso_error_pre = np.array(qiso_error_pre)
qiso_error_post = np.array(qiso_error_post)
qap_bestfit_pre = np.array(qap_bestfit_pre)
qap_bestfit_post = np.array(qap_bestfit_post)
qap_error_pre = np.array(qap_error_pre)
qap_error_post = np.array(qap_error_post)

ax[0].scatter(qiso_bestfit_post - qiso_bestfit_pre, qap_bestfit_post - qap_bestfit_pre, s=10.0, marker='o', color=color)
ax[1].scatter(qiso_error_post/qiso_error_pre, qap_error_post/qap_error_pre, s=10.0, marker='o', color=color)

ax[0].errorbar(np.mean(qiso_bestfit_post - qiso_bestfit_pre), np.mean(qap_bestfit_post - qap_bestfit_pre),
                xerr=np.std(qiso_bestfit_post - qiso_bestfit_pre), yerr=np.std(qap_bestfit_post - qap_bestfit_pre),
                marker='o', capsize=2.0, ls='', color=color,
                markeredgewidth=1.0, mfc='w')

ax[1].errorbar(np.mean(qiso_error_post/qiso_error_pre), np.mean(qap_error_post/qap_error_pre),
                xerr=np.std(qiso_error_post/qiso_error_pre), yerr=np.std(qap_error_post/qap_error_pre),
                markeredgewidth=1.0, mfc='w', color=color,
                marker='o', capsize=2.0, ls='')

if args.fitting_method == 'profiling':
    # read pre-recon fits
    profiles_dir = f'/global/homes/e/epaillas/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/fits_correlation_{args.apmode}_pcs2/'
    profiles_fn = Path(profiles_dir) / f"profiles_{tracer}_{region}_{zmin}_{zmax}_sigmas{sigmas[tracer]['pre']}_sigmapar{sigmapar[tracer]['pre']}_sigmaper{sigmaper[tracer]['pre']}.npy"
    qiso_bestfit_pre, qiso_error_pre = get_profile_bestfit(profiles_fn, param='qiso')
    qap_bestfit_pre, qap_error_pre = get_profile_bestfit(profiles_fn, param='qap')

    # read post-recon fits
    profiles_dir = f'/global/homes/e/epaillas/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/fits_correlation_{args.apmode}_pcs2/recon_IFFT_recsym_sm{smoothing_radius}/'
    profiles_fn = Path(profiles_dir) / f"profiles_{tracer}_{region}_{zmin}_{zmax}_sigmas{sigmas[tracer]['post']}_sigmapar{sigmapar[tracer]['post']}_sigmaper{sigmaper[tracer]['post']}.npy"
    qiso_bestfit_post, qiso_error_post = get_profile_bestfit(profiles_fn, param='qiso')
    qap_bestfit_post, qap_error_post = get_profile_bestfit(profiles_fn, param='qap')
else:
    # read pre-recon fits
    chain_dir = f'/global/homes/e/epaillas/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/fits_correlation_{args.apmode}_pcs2/'
    chain_fn = [Path(chain_dir) / f"chain_{tracer}_{region}_{zmin}_{zmax}_sigmas{sigmas[tracer]['pre']}_sigmapar{sigmapar[tracer]['pre']}_sigmaper{sigmaper[tracer]['pre']}_{i}.npy" for i in range(8)]
    qiso_bestfit_pre, qiso_error_pre = get_chain_bestfit(chain_fn, param='qiso')
    qap_bestfit_pre, qap_error_pre = get_chain_bestfit(chain_fn, param='qap')

    # read post-recon fits
    chain_dir = f'/global/homes/e/epaillas/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/fits_correlation_{args.apmode}_pcs2/recon_IFFT_recsym_sm{smoothing_radius}/'
    chain_fn = [Path(chain_dir) / f"chain_{tracer}_{region}_{zmin}_{zmax}_sigmas{sigmas[tracer]['post']}_sigmapar{sigmapar[tracer]['post']}_sigmaper{sigmaper[tracer]['post']}_{i}.npy" for i in range(8)]
    qiso_bestfit_post, qiso_error_post = get_chain_bestfit(chain_fn, param='qiso')
    qap_bestfit_post, qap_error_post = get_chain_bestfit(chain_fn, param='qap')

ax[0].plot(qiso_bestfit_post - qiso_bestfit_pre, qap_bestfit_post - qap_bestfit_pre,
            marker='*', color=color, ls='', ms=10.0, mfc='k', markeredgewidth=1.0,
            )
ax[1].plot(qiso_error_post/qiso_error_pre, qap_error_post/qap_error_pre, marker='*',
            color=color, ls='', ms=10.0, mfc='k', markeredgewidth=1.0,
            )

ax[1].plot(np.nan, np.nan, label='DESI', marker='*', color='k', ls='', ms=10.0, mfc='k', markeredgewidth=1.0,)
ax[1].errorbar(np.nan, np.nan, xerr=np.nan, yerr=np.nan,
            markeredgewidth=1.0, mfc='w', color='k',
            marker='o', capsize=2.0, ls='',
            label=r'{\tt AbacusSummit}')

xlim0 = ax[0].get_xlim()
ylim0 = ax[0].get_ylim()
xlim1 = ax[1].get_xlim()
ylim1 = ax[1].get_ylim()

ax[0].set_xlim(xlim0)
ax[0].set_ylim(ylim0)
ax[1].set_xlim(xlim1)
ax[1].set_ylim(ylim1)
for aa in ax:
    aa.tick_params(axis='both', which='major', labelsize=12)
ax[0].set_xlabel(r'$\alpha_{\rm iso}\, \textrm{post} - \alpha_{\rm iso}\, \textrm{pre}$', fontsize=17)
ax[0].set_ylabel(r'$\alpha_{\rm AP}\, \textrm{post} - \alpha_{\rm AP}\, \textrm{pre}$', fontsize=17)
ax[1].set_xlabel(r'$\sigma_{\alpha_{\rm iso}}\, \textrm{post}/\sigma_{\alpha_{\rm iso}}\, \textrm{pre}$', fontsize=17)
ax[1].set_ylabel(r'$\sigma_{\alpha_{\rm AP}}\, \textrm{post}/\sigma_{\alpha_{\rm AP}}\, \textrm{pre}$', fontsize=17)
title = f'{label}'
ax[0].annotate(title, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=14, bbox=dict(boxstyle="round", fc='w', ec='lightgrey'))
ax[1].legend(frameon=True, fontsize=14, handletextpad=0.0, loc='best')
fig.tight_layout()

output_fn = f'fig/keypaper_qisoqap_scatter_{tracer}_{zmin}_{zmax}.pdf'
plt.savefig(output_fn, bbox_inches='tight')
plt.show()
