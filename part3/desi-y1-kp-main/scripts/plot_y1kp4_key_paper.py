########################################################################
# Code to create the plots which appear in the KP4 Key Paper.          #
#                                                                      #
# This code created by Chris Blake 30/1/24, replacing original code    #
# by clarifying the data links at the top of the script rather than    #
# within the file manager.                                             #
#                                                                      #
# Fig.1: Post-reconstruction correlation function multipoles           #
# Fig.2: Post-reconstruction power spectrum multipoles                 #
#                                                                      #
# The main plotting codes are in desi_y1_plotting/kp4.py               #
########################################################################

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from desi_y1_plotting.kp4 import KP4StylePaper
from desi_y1_plotting import utils
from desipipe import FileManager
from desi_y1_files import io

# Plotting options
plopt = 2   # 1) Correlation function multipoles
            # 2) Power spectrum multipoles

# Global location of DESI data
stem = Path('/global/cfs/cdirs/desi/')

# Data directory for 2-pt functions
data_dir = Path('survey/catalogs/Y1/LSS/iron/LSScats/v1.1/unblinded/desipipe/2pt/recon_sm15_IFFT_recsym/')
mock_dir = Path('survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v3/altmtl/2pt/')
cov_dir_corr = Path('users/mrash/RascalC/Y1/unblinded/v1/')
cov_dir_pow = Path('users/oalves/thecovs/y1_unblinded/post/')

# Directory for outputting plots
out_dir = 'output/'

# Cases for different panels
list_tracer = ['BGS_BRIGHT-21.5','LRG','ELG_LOPnotqso']
list_tracer_zrange = ['BGS_BRIGHT-21.5','LRG','LRG','LRG','ELG_LOPnotqso','ELG_LOPnotqso']
list_zrange = [(0.1,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.1),(0.8,1.1),(1.1,1.6)]

# Data options
region = 'GCcomb'
cut = ''
tracer_nran = {'BGS_BRIGHT-21.5': 1, 'LRG': 8, 'ELG_LOPnotqso': 10}
tracer_boxsize = {'BGS_BRIGHT-21.5': 4000., 'LRG': 7000., 'ELG_LOPnotqso': 9000.}
ells = (0, 2, 4)
select_corr = (20., 200., 4.)
select_pow = (0., 0.3, 0.005)

# Plot style options from Antoine Rocher
args_rcparam = {
'xtick.direction' : 'inout',
'ytick.direction' : 'inout',
'xtick.major.size ' : 4,
'xtick.major.width' : 1,
'ytick.major.size'  : 4,
'ytick.major.width ': 1,
'axes.grid'      : True,
'grid.color'    : 'grey',
'grid.linestyle' : '-',
'grid.linewidth' : 0.7,
'grid.alpha'     : 0.2,
'legend.fancybox'  : True,
'legend.facecolor' : 'white',
'legend.framealpha' : 0.5,
'legend.edgecolor' : 'grey',
'figure.autolayout'  : False,
'figure.frameon'     : True,
'figure.subplot.left'    : 0.08,
'figure.subplot.right'   : 0.96,
'lines.linewidth' : 1.5,
'axes.facecolor': 'white',
'axes.edgecolor': 'black',
'axes.formatter.use_mathtext' : True,
'axes.linewidth': 1.,
'axes.labelcolor' : 'black',
'axes.formatter.use_mathtext' : True,
'axes.labelsize' : 8,
'font.family': 'serif',
"text.color"          : "k",
'text.usetex'         : False,
'mathtext.cal' : 'cursive',
'mathtext.rm'  : 'serif',
'mathtext.tt'  : 'monospace',
'mathtext.it'  : 'serif:italic',
'mathtext.bf'  : 'serif:bold',
'mathtext.sf'  : 'sans\-serif',
'mathtext.fontset' : 'dejavusans'}

with KP4StylePaper() as style:
    style.update(args_rcparam)

########################################################################
# Create panel plot of correlation function multipoles.                #
########################################################################

    if (plopt == 1):
    
# Create panel plot
        fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(7,7))
        ax = ax.flatten()

# Loop over different cases
        isub = -1
        for tracer, zrange in zip(list_tracer_zrange, list_zrange):
        
# Create file manager object for data
            corr_data_options = dict({'tracer': tracer, 'region': region, 'zrange': zrange, 'weighting': 'default_FKP', 'binning': 'lin', 'nran': tracer_nran[tracer], 'njack': 0, 'split': 20, 'cut': cut})
            corr_data_filename = 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy'
            fm = FileManager()
            fm.append(dict(filetype='correlation', path=stem/data_dir/corr_data_filename, options=corr_data_options))
            fdata = fm.get()

# Create file manager object for covariance
            corr_cov_options = dict({'tracer': tracer, 'region': region, 'zrange': zrange})
            corr_cov_filename = 'xi024_{tracer}_IFFT_recsym_sm15_{region}_{zrange[0]:.1f}_{zrange[1]:.1f}_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'
            fm = FileManager()
            fm.append(dict(filetype='correlation_covariance', path=stem/cov_dir_corr/corr_cov_filename, options=corr_cov_options))
            fcovariance = fm.get()

# Create file manager object for mocks
            corr_mock_options = dict({'imock': range(0, 25), 'tracer': tracer, 'region': region, 'zrange': zrange, 'weighting': 'default_FKP', 'binning': 'lin', 'nran': tracer_nran[tracer], 'njack': 0, 'split': 20, 'cut': cut})
            corr_mock_filename = 'mock{imock:d}/recon_sm15_IFFT_recsym/xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy'
            fm = FileManager()
            fm.append(dict(filetype='correlation', path=stem/mock_dir/corr_mock_filename, options=corr_mock_options))
            fmock = fm.select()
            fmock = [fi for fi in fmock if fi.exists()]

# Add new sub-plot
            isub += 1
            style.plot_correlation_multipoles_panel(axes=ax, isub=isub, data=fdata, mock=fmock, covariance=fcovariance, ells=ells, select=select_corr)

# Output plot
        fig.subplots_adjust(hspace=0.1,wspace=0.05)
        utils.savefig(out_dir + 'correlation_multipoles_all.png', fig=fig)

########################################################################
# Create panel plot of power spectrum multipoles.                      #
########################################################################

    elif (plopt == 2):

# Create panel plot
        fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(7,7))
        ax = ax.flatten()

# Loop over different cases
        isub = -1
        for tracer, zrange in zip(list_tracer_zrange, list_zrange):
        
# Create file manager object for data
            pow_data_options = dict({'tracer': tracer, 'region': region, 'zrange': zrange, 'weighting': 'default_FKP', 'binning': 'lin', 'nran': 18, 'cellsize': 6, 'boxsize': tracer_boxsize[tracer], 'cut': cut})
            pow_data_filename = 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy'
            fm = FileManager()
            fm.append(dict(filetype='power', path=stem/data_dir/pow_data_filename, options=pow_data_options))
            fdata = fm.get()

# Create file manager object for covariance
            pow_cov_options = dict({'tracer': tracer, 'region': region, 'zrange': zrange})
            pow_cov_filename = 'cov_gaussian_{tracer}_{region}_{zrange[0]:.1f}-{zrange[1]:.1f}.txt'
            fm = FileManager()
            fm.append(dict(filetype='power_covariance', path=stem/cov_dir_pow/pow_cov_filename, options=pow_cov_options))
            fcovariance = fm.get()

# Create file manager object for mocks
            pow_mock_options = dict({'imock': range(0, 25), 'tracer': tracer, 'region': region, 'zrange': zrange, 'weighting': 'default_FKP', 'binning': 'lin', 'nran': 18, 'cellsize': 6, 'boxsize': tracer_boxsize[tracer], 'cut': cut})
            pow_mock_filename = 'mock{imock:d}/recon_sm15_IFFT_recsym/pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy'
            fm = FileManager()
            fm.append(dict(filetype='power', path=stem/mock_dir/pow_mock_filename, options=pow_mock_options))
            fmock = fm.select()
            fmock = [fi for fi in fmock if fi.exists()]

# Add new sub-plot
            isub += 1
            style.plot_power_multipoles_panel(axes=ax, isub=isub, data=fdata, mock=fmock, covariance=fcovariance, ells=ells, select=select_pow)

# Output plot
        fig.subplots_adjust(hspace=0.1,wspace=0.05)
        utils.savefig(out_dir + 'power_multipoles_all.png', fig=fig)
