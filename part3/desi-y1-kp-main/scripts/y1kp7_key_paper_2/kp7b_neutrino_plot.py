import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import getdist
from getdist import plots, loadMCSamples

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')
sys.path.insert(1, '../../')

from y1_fs_cosmo_tools import load_cobaya_samples, load_desilike_samples
from y1_bao_cosmo_tools import load_cobaya_samples as load_cobaya_bao_samples
from desi_y1_plotting import KP7StylePaper, utils

style = KP7StylePaper()

getdist_2D_width_inch = 5
getdist_2D_ratio = 1 / 1.2

%matplotlib inline
# ------------------------ #

style.settings.legend_frame = True
def white_legend(legend):
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('white')
    legend.get_frame().set_alpha(1.)
    
outdir = Path('./plots_v1_talk/')
outdir.mkdir(exist_ok=True)
ext = 'png'

from cosmoprimo.fiducial import DESI

cosmo = DESI()
markers = {}
for name in ['H0', 'logA', 'omega_b', 'omega_cdm', 'Omega_m', 'n_s']:
    markers[name] = cosmo[name]
markers['sigma8_m'] = cosmo.get_fourier().sigma8_m
markers['S8'] = markers['sigma8_m'] * (markers['Omega_m'] / 0.3)**0.5

ext = 'pdf'

#load chains
model = 'base_mnu'
cmb_nolens = load_cobaya_bao_samples(model=model, run='run0', dataset=['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'])
cmb = load_cobaya_samples(model=model, run='run3', dataset=['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing'])
fs_bao_cmb = load_cobaya_samples(model=model, run='run4', dataset=['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing'], convergence=True)
fs_bao = load_cobaya_samples(model=model, run='run4', dataset=['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10'], convergence=True)
fs_bao_ns = load_cobaya_samples(model=model, run='run4', dataset=['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns'], convergence=True)

model = 'base_mnu_w_wa'
fs_bao_cmb_pantheon = load_cobaya_samples(model=model, run='run3', dataset=['desi-reptvelocileptors-fs-bao-all', 'pantheonplus', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing'], convergence=True)
fs_bao_cmb_union = load_cobaya_samples(model=model, run='run3', dataset=['desi-reptvelocileptors-fs-bao-all', 'union3', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing'], convergence=True)
fs_bao_cmb_des = load_cobaya_samples(model=model, run='run3', dataset=['desi-reptvelocileptors-fs-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing'], convergence=True)

#load more chains
model = 'base_mnu'
fs_bao_cmb_camspec = load_cobaya_samples(model=model, run='run4', dataset=['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing'], convergence=True)
fs_bao_cmb_hillipop = load_cobaya_samples(model=model, run='run4', dataset=['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE', 'planck-act-dr6-lensing'], convergence=True)

# make the neutrino plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 8.5), tight_layout = True)

i = 5
hierarchy = 2

g1 = plots.get_single_plotter(width_inch=getdist_2D_width_inch, ratio=getdist_2D_ratio, scaling=True)
g1.settings = style.settings
g1.settings.legend_frame = False
g1.settings.legend_fontsize = 14

g2 = plots.get_single_plotter(width_inch=getdist_2D_width_inch, ratio=getdist_2D_ratio, scaling=True)
g2.settings = style.settings
g2.settings.legend_frame = False
g2.settings.legend_fontsize = 14
    
g1.plot_2d([fs_bao, fs_bao_ns, cmb_nolens, cmb, fs_bao_cmb], 'ns', 'mnu',
              colors=[style.color_palette[0]] * 2 + [style.color_palette[1]] * 2 + style.color_palette[2:],
              filled=[False, True, False, True, True],
              ls=['--', '-', '--', '-', '-'],
              alphas=[style.settings.alpha_filled_add] * i + [0.] * 6, ax = ax1)
g1.add_legend([r'DESI + BBN + $n_{s10}$', r'DESI + BBN + tight $n_s$ prior', 'CMB (no lensing)', 'CMB', 'DESI + CMB'][:i], ncols=1, legend_loc='upper left', ax = ax1)
ax1.set_ylabel(r'$\sum m_\nu \, [\mathrm{eV}]$')
ax1.set_xlim(0.87, 1.1)
ax1.set_ylim(0., 0.75)
ax1.annotate(r'$\Lambda$CDM', xy=(1.063, 0.7), fontsize=14)

g2.plot_2d([fs_bao, fs_bao_ns, cmb_nolens, cmb, fs_bao_cmb], 'H0', 'mnu',
          colors=[style.color_palette[0]] * 2 + [style.color_palette[1]] * 2 + style.color_palette[2:],
          filled=[False, True, False, True, True],
          ls=['--', '-', '--', '-', '-'],
          alphas=[style.settings.alpha_filled_add] * i + [0.] * 6, ax = ax2)
g2.add_legend([r'DESI + BBN + $n_{s10}$', r'DESI + tight $n_s$ prior', 'CMB (no lensing)', 'CMB', 'DESI + CMB'][:i], ncols=1, legend_loc='upper left', ax = ax2)
# ax2.set_ylabel(r'$\sum m_\nu \, [\mathrm{eV}]$')
ax2.set_ylabel('')
ax2.set_ylim(0., 0.75)
ax2.annotate(r'$\Lambda$CDM', xy=(70, 0.7), fontsize=14)

smooth1d = 3
cmb.updateSettings({'smooth_scale_1D': smooth1d})
fs_bao.updateSettings({'smooth_scale_1D': smooth1d})
fs_bao_cmb.updateSettings({'smooth_scale_1D': smooth1d})
fs_bao_cmb_camspec.updateSettings({'smooth_scale_1D': smooth1d})
fs_bao_cmb_hillipop.updateSettings({'smooth_scale_1D': smooth1d})

g3 = plots.getSinglePlotter(width_inch=getdist_2D_width_inch, ratio=getdist_2D_ratio, analysis_settings={'smooth_scale_1D': 0.9, 'fine_bins': -1})
g3.settings.__dict__.update(style.settings.__dict__)
g3.settings.legend_frame = True
g3.plot_1d([fs_bao, cmb, fs_bao_cmb, fs_bao_cmb_camspec, fs_bao_cmb_hillipop], 'mnu', colors=style.contour_colors[:2] + [style.contour_colors[2]] * 3, ls = ['-', '-', '-', '--', '-.'], ax = ax3)
g3.add_legend(['DESI', 'CMB', r'DESI + CMB ($\texttt{clik-plik}$)', r' DESI + CMB ($\texttt{clik-CamSpec}$)', r'DESI + CMB ($\texttt{LoLL-HiLL}$)'], ax = ax3)
ax3.set_xlim(0, 0.2)
ax3.set_ylim(0, 1.1)
ax3.set_xlabel(r'$\sum m_\nu \, [\mathrm{eV}]$')
ax3.annotate(r'$\Lambda$CDM', xy=(0.163, 1.005), fontsize=14, zorder=10)
if hierarchy == 1:
    ax3.axvline(0.059, 0., 1., color='grey', ls='--', lw=1.2)
    ax3.axvspan(0.059, 0.2, color='plum', alpha=0.1)
if hierarchy == 2:
    ax3.axvline(0.059, 0., 1., color='grey', ls='--', lw=1.2)
    ax3.axvline(0.1, 0., 1., color='grey', ls='--', lw=1.2)
    ax3.axvspan(0.059, 0.1, color='plum', alpha=0.1)
    ax3.axvspan(0.1, 0.3, color='cornflowerblue', alpha=0.1)

fs_bao_cmb_pantheon.label = 'DESI (FS+BAO) + CMB + PantheonPlus'
fs_bao_cmb_union.label = 'DESI (FS+BAO) + CMB + Union3'
fs_bao_cmb_des.label = 'DESI (FS+BAO) + CMB + DES-SN5YR'

g4 = plots.getSinglePlotter(width_inch=getdist_2D_width_inch, ratio=getdist_2D_ratio)
g4.settings.__dict__.update(style.settings.__dict__)
g4.settings.legend_frame = fs_bao_cmb_union
g4.plot_1d([fs_bao_cmb_pantheon, fs_bao_cmb_union, fs_bao_cmb_des], 'mnu', colors=style.contour_colors[:2] + [style.contour_colors[2]] * 3, ls = ['-', '-', '-', '--', '-.'], ax = ax4)
g4.add_legend([r'DESI + CMB + PantheonPlus', r'DESI + CMB + Union3', r'DESI + CMB + DES-SN5YR'], fontsize=12, ax = ax4)
ax4.set_xlim(0, 0.2)
ax4.set_ylim(0, 1.1)
ax4.annotate(r'$w_0w_a$CDM', xy=(0.1525, 1.02), fontsize=14)
ax4.set_xlabel(r'$\sum m_\nu \, [\mathrm{eV}]$')
ax4.set_ylabel('')
if hierarchy == 1:
    ax4.axvline(0.059, 0., 1., color='grey', ls='--', lw=1.2)
    ax4.axvspan(0.059, 0.2, color='plum', alpha=0.1)
if hierarchy == 2:
    ax4.axvline(0.059, 0., 1., color='grey', ls='--', lw=1.2)
    ax4.axvline(0.1, 0., 1., color='grey', ls='--', lw=1.2)
    ax4.axvspan(0.059, 0.1, color='plum', alpha=0.1)
    ax4.axvspan(0.1, 0.2, color='cornflowerblue', alpha=0.1)

fig.savefig(outdir / 'mnu_all.{}'.format(ext), bbox_inches='tight', dpi=360)
