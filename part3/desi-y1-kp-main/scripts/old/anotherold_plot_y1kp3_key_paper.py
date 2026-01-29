########################################################################
# Code to create intitial versions of some of the plots in the KP3 Key Paper.          #
#                                                                      #
# This code created by Chris Blake 14/1/24, replacing original code    #
# by clarifying the data links at the top of the script rather than    #
# within the file manager.                                             #
#                                                                      #
# Fig.1: Completeness map                                              #
# Fig.2: Correlation function multipoles                               #
# Fig.3: Power spectrum multipoles                                     #
#                                                                      #
# The main plotting codes are in desi_y1_plotting/kp3.py               #
########################################################################

import sys
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import fitsio
from astropy.table import Table
import LSS.common_tools as common
from desi_y1_plotting.kp3 import KP3StylePaper
from desi_y1_plotting import utils
from desipipe import FileManager
from desi_y1_files import io

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--plopt", help="which plot to create",default=1,type=int)
args = parser.parse_args()

# Plotting options
plopt = args.plopt   # 1) Completeness map
            # 2) Correlation function multipoles
            # 3) Power spectrum multipoles

# Global location of DESI data
stem = Path('/global/cfs/cdirs/desi/')

# Data directory for completeness maps
comp_dir = Path('survey/catalogs/Y1/LSS/iron/LSScats/v1/')

# Data directory for 2-pt functions
data_dir = Path('survey/catalogs/Y1/LSS/iron/LSScats/v1/blinded/desipipe/2pt/') #can later switch to baseline_2pt with simpler file names
mock_dir = Path('survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v3/altmtl/baseline_2pt/') 
cov_dir_corr = Path('users/mrash/RascalC/Y1/blinded/v0.6/')
cov_dir_pow = Path('survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/pk/covariances/')

# Directory for outputting plots
out_dir = 'output/'

# Cases for different panels
list_tracer = ['BGS_BRIGHT-21.5','LRG','ELG_LOPnotqso','QSO']
list_tracer_zrange = ['BGS_BRIGHT-21.5','LRG','LRG','LRG','ELG_LOPnotqso','ELG_LOPnotqso','QSO']
list_zrange = [(0.1,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.1),(0.8,1.1),(1.1,1.6),(0.8,2.1)]

# Data options
region = 'GCcomb'
cut = ''
tracer_nran = {'BGS_BRIGHT-21.5': 1, 'LRG': 8, 'ELG_LOPnotqso': 10, 'QSO': 4}
tracer_boxsize = {'BGS_BRIGHT-21.5': 4000., 'LRG': 7000., 'ELG_LOPnotqso': 9000., 'QSO': 10000.}
ells = (0, 2, 4)
select_corr = (20., 200., 4.)
select_pow = (0., 0.3, 0.005)


args_rcparam = {'xtick.direction' : 'inout',
'ytick.direction' : 'inout',
'xtick.major.size ' : 4,      # major tick size in points
'xtick.major.width' : 1,      # major tick width in points
'ytick.major.size'  : 4,      # major tick size in points
'ytick.major.width ': 1,      # major tick width in points

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
'axes.labelsize' : 8,
                
## FONT
'font.family': 'serif',

## TEXT
"text.color"          : "k",
'text.usetex'         : False,
#text.latex.preamble : \newcommand{\mathdefault}[1][]{} #With python3.9 it is now useless.
'mathtext.cal' : 'cursive',
'mathtext.rm'  : 'serif',
'mathtext.tt'  : 'monospace',
'mathtext.it'  : 'serif:italic',
'mathtext.bf'  : 'serif:bold',
'mathtext.sf'  : 'sans\-serif',
'mathtext.fontset' : 'dejavusans',
#mathtext.default : it
               }

with KP3StylePaper() as style:
    style.update(args_rcparam)
########################################################################
# Create panel of completeness maps.                                   #
########################################################################

    if (plopt == 1):

# Create plot
        fig = plt.figure(figsize=(7,11))
        
# Loop over different tracers
        isub = 0
        for tracer in list_tracer:

# Read in completeness map data
            datfile = tracer + '_full_HPmapcut.dat.fits'
            dt = Table(fitsio.read(stem/comp_dir/datfile))
            sel_gz = common.goodz_infull(tracer[:3],dt)
            sel_obs = dt['ZWARN'] != 999999
            dt = dt[sel_obs & sel_gz]
            ras = dt['RA']
            ras[ras > 300] -= 360
            sindec = np.sin(dt['DEC']*np.pi/180.)
            comp = dt['COMP_TILE']
            
# Add new sub-plot
#            vmin = np.min(comp)
#            vmax = np.max(comp)
            vmin = 0.
            vmax = 1.
            title  = tracer[:3] + ' completeness'
            isub += 1
            style.plot_map_sindec_panel(ras,sindec,comp,vmin,vmax,title,fig,isub)
            
# Output plot
        plt.tight_layout(h_pad=2,pad=1)
        utils.savefig(out_dir + 'comptile_all.png', fig=fig)

########################################################################
# Create panel plot of correlation function multipoles.                #
########################################################################

    elif (plopt == 2):
    
# Create panel plot
        fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(7,9))
        ax = ax.flatten()

# Creat legend
        isub = 7
        style.plot_legend_panel(ax,isub)

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
            corr_cov_filename = 'xi024_{tracer}_{region}_{zrange[0]:.1f}_{zrange[1]:.1f}_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'
            fm = FileManager()
            fm.append(dict(filetype='correlation_covariance', path=stem/cov_dir_corr/corr_cov_filename, options=corr_cov_options))
            fcovariance = fm.get()

# Create file manager object for mocks
            corr_mock_options = dict({'imock': range(0, 25), 'tracer': tracer, 'region': region, 'zrange': zrange, 'weighting': 'default_FKP', 'binning': 'lin', 'nran': tracer_nran[tracer], 'njack': 0, 'split': 20, 'cut': cut})
            corr_mock_filename = 'mock{imock:d}/xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy'
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

    elif (plopt == 3):

# Create panel plot
        fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(7,9))
        ax = ax.flatten()

# Create legend
        isub = 7
        style.plot_legend_panel(ax,isub)

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
            pow_cov_options = dict({'tracer': tracer, 'region': region, 'zrange': zrange, 'weighting': 'default_FKP', 'binning': 'lin'})
            pow_cov_filename = 'cov_gaussian_pre_{tracer}_{region}_{zrange[0]:.1f}_{zrange[1]:.1f}_{weighting}_{binning}.txt'
            fm = FileManager()
            fm.append(dict(filetype='power_covariance', path=stem/cov_dir_pow/pow_cov_filename, options=pow_cov_options))
            fcovariance = fm.get()

# Create file manager object for mocks
            pow_mock_options = dict({'imock': range(0, 25), 'tracer': tracer, 'region': region, 'zrange': zrange, 'weighting': 'default_FKP', 'binning': 'lin', 'nran': 18, 'cellsize': 6, 'boxsize': tracer_boxsize[tracer], 'cut': cut})
            pow_mock_filename = 'mock{imock:d}/pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy'
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
