"""Produce the plots for the KP3 key paper."""

import os
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt

from desipipe import setup_logging
from desi_y1_plotting import KP3StylePaper, utils
from desi_y1_files import get_data_file_manager, get_abacus_file_manager, get_raw_abacus_file_manager, get_box_abacus_file_manager, get_ez_file_manager, get_box_ez_file_manager, get_glam_file_manager, get_zsnap_from_z


if __name__ == '__main__':

    #setup_logging()
    #todo = ['abacus']
    #todo = ['ez']
    #todo = ['glam']
    #todo = ['residual']
    #todo = ['comparison_abacus']
    #todo = ['comparison_box_abacus']
    #todo = ['comparison_abacus_real']
    #todo = ['comparison_abacus_fa']
    #todo = ['comparison_ez_ric']
    #todo = ['residual_abacus']
    #todo = ['residual_abacus_fa']
    #todo = ['wmatrix_validation_raw']
    #todo = ['comparison_cutsky_raw']
    #todo = ['wmatrix_validation_raw_thetacut']
    #todo = ['comparison_cutsky_ez']
    #todo = ['wmatrix_validation_ez']
    #todo = ['counts']
    #todo = ['unblinding']
    #todo = ['wmatrix_validation_abacus']
    todo = ['wmatrix_validation_thetacut_abacus']
    #todo += ['wmatrix_validation_thetacut_ez']
    #todo = ['data_rotated']
    
    version = 'v1.4'
    fm = get_data_file_manager(conf='unblinded') + get_abacus_file_manager() + get_ez_file_manager() + get_glam_file_manager()
    
    # Output directory
    #outdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/KP3plots/KeyPaper/'
    for fc in fm.select(id='catalog_data_y1', version=version).iter(intersection=False):
        outdir = Path(os.path.dirname(fc)) / 'plots_kp3'
        break
    setup_logging()

    with KP3StylePaper() as style:

        if 'data' in todo:
            for ft in fm.select(id='correlation_y1', weighting='default_FKP', version=version).iter(exclude=['zrange'], intersection=False):  # we are iterating on tracer, cut, etc. --- except zrange
                fig = plt.figure()
                for options in ft.iter_options():  # here we are iterating on zrange: stack on the same figure
                    fdata = fm.get(id='correlation_y1', **options, check_exists=True)
                    fcovariance = fm.get(id='covariance_correlation_y1', **options, ignore=['version'], check_exists=True, raise_error=False)
                    fabacus = fm.select(id='correlation_abacus_y1', **options, fa='altmtl', ignore=['weighting', 'version'], raise_error=False)
                    fabacus = [fi for fi in fabacus if fi.exists()]
                    style.plot_correlation_multipoles(fdata, covariance=fcovariance, fig=fig, markers='point')
                    if fabacus:
                        style.plot_correlation_multipoles(fabacus, fig=fig, markers='line')
                utils.savefig(outdir / 'correlation_multipoles_{tracer}_{region}{cut}.png'.format(**fdata.foptions), fig=fig)
                plt.close(fig)

            for ft in fm.select(id='power_y1', weighting='default_FKP', version=version).iter(exclude=['zrange'], intersection=False):
                fig = plt.figure()
                for options in ft.iter_options():
                    fdata = fm.get(id='power_y1', **options, check_exists=True)
                    fcovariance = fm.get(id='covariance_power_y1', **options, ignore=['version'], check_exists=True, raise_error=False)
                    fabacus = fm.select(id='power_abacus_y1', **options, fa='altmtl', ignore=['weighting', 'version'], raise_error=False)
                    fabacus = [fi for fi in fabacus if fi.exists()]
                    style.plot_power_multipoles(fdata, covariance=fcovariance, fig=fig, markers='point')
                    if fabacus:
                        style.plot_power_multipoles(fabacus, fig=fig, markers='line')
                # foptions is just the "formatted option" to pass to the string
                utils.savefig(outdir / 'power_multipoles_{tracer}_{region}{cut}.png'.format(**fdata.foptions), fig=fig)
                plt.close(fig)

            style.plot_legend(fn=os.path.join(outdir, 'legend.png'))

        if 'abacus' in todo:
            options = dict(weighting='default_FKP', version='v3', fa='ffa', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'])
            """
            for fi in fm.select(id='wmatrix_power_merged_abacus_y1', **options).iter(intersection=False):
                fig = style.plot_window_power_multipoles(fi)
                utils.savefig(outdir / 'window_power_multipoles_abacus_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi.foptions), fig=fig)
                plt.close(fig)
            """

            for fi in fm.select(id='power_abacus_y1', **options).iter(exclude=['imock'], intersection=False):
                fi = list(fi)
                fig = style.plot_power_multipoles(fi, mean=False, markers='line')
                utils.savefig(outdir / 'power_multipoles_abacus_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)

            for fi in fm.select(id='power_recon_abacus_y1', **options).iter(exclude=['imock'], intersection=False):
                fi = list(fi)
                fig = style.plot_power_multipoles(fi, mean=False, markers='line')
                utils.savefig(outdir / 'power_multipoles_abacus_{mode}_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)

            for fi in fm.select(id='correlation_abacus_y1', **options).iter(exclude=['imock'], intersection=False):
                fi = list(fi)
                fig = style.plot_correlation_multipoles(fi, mean=False, markers='line')
                utils.savefig(outdir / 'correlation_multipoles_abacus_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)

            for fi in fm.select(id='correlation_recon_abacus_y1', **options).iter(exclude=['imock'], intersection=False):
                fi = list(fi)
                fig = style.plot_correlation_multipoles(fi, mean=False, markers='line')
                utils.savefig(outdir / 'correlation_multipoles_abacus_{mode}_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)

        if 'ez' in todo:
            options = dict(weighting='default_FKP', version='v1', fa='ffa', tracer=['LRG', 'LRG+ELG_LOPnotqso', 'ELG_LOPnotqso', 'QSO'][1:2], imock=list(range(1, 200)))
            """
            for fi in fm.select(id='wmatrix_power_merged_ez_y1', **options).iter(intersection=False):
                fig = style.plot_window_power_multipoles(fi)
                utils.savefig(outdir / 'window_power_multipoles_ez_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi.foptions), fig=fig)
                plt.close(fig)
            """
            """
            for fi in fm.select(id='power_ez_y1', **options).iter(exclude=['imock'], intersection=False):
                fi = [fi for fi in fi if fi.exists()]
                print(len(fi))
                fig = style.plot_power_multipoles(fi, mean=False, markers='line')
                utils.savefig(outdir / 'power_multipoles_ez_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)
            """
            """
            for fi in fm.select(id='power_recon_ez_y1', **options).iter(exclude=['imock'], intersection=False):
                fi = [fi for fi in fi if fi.exists()]
                print(len(fi))
                fig = style.plot_power_multipoles(fi, mean=False, markers='line')
                utils.savefig(outdir / 'power_multipoles_ez_{mode}_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)
            """
            """
            for fi in fm.select(id='correlation_ez_y1', **options).iter(exclude=['imock'], intersection=False):
                #fi = list(fi)
                fi = [fi for fi in fi if fi.exists()]
                fig = style.plot_correlation_multipoles(fi, mean=False, markers='line')
                utils.savefig(outdir / 'correlation_multipoles_ez_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)
            """
            for fi in fm.select(id='correlation_recon_ez_y1', njack=0, **options).iter(exclude=['imock'], intersection=False):
                fi = [fi for fi in fi if fi.exists()]
                for ff in fi:
                    ff.load()
                    print(ff)
                fig = style.plot_correlation_multipoles(fi, mean=False, markers='line')
                utils.savefig(outdir / 'correlation_multipoles_ez_{mode}_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)
        
        if 'glam' in todo:
            dfm = get_data_file_manager(conf='unblinded')
            fm = get_glam_file_manager()
            options = dict(weighting='default_FKP', version='v1', fa='ffa', tracer=['QSO'])

            for fi in fm.select(id='power_glam_y1', **options).iter(exclude=['imock'], intersection=False):
                fi = [fi for fi in fi if fi.exists()]
                fig = plt.figure()
                options = dict(fi[0].options)
                options['version'] = 'v1.5'
                options.pop('nran')
                print(options)
                fdata = dfm.get(id='power_y1', **options, ignore=True)
                print(fdata)
                style.plot_power_multipoles(fdata, fig=fig, markers='point')
                fig = style.plot_power_multipoles(fi, mean=False, markers='line', fig=fig)
                utils.savefig(outdir / 'power_multipoles_glam_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)

        if 'residual' in todo:
            options = dict(weighting='default_FKP', version='v1', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], cut=[None, ('theta', 0.05)])
            for fd in fm.select(id='correlation_y1', **options).iter(intersection=False):
                options = fd.options.copy()
                options.pop('nran')
                fi = [fi for fi in fm.select(id='correlation_ez_y1', **options, ignore=True).iter(intersection=False) if fi.exists()]
                fig = style.plot_residual_multipoles(fd, covariance=fi, select=(50., 150., 10.), ells=(0, 2, 4))
                utils.savefig(outdir / 'residual_multipoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)

        """
        if 'unblinding' in todo:
            for ft in fm.select(id='correlation_recon_y1', weighting='default_FKP', version=version, cut=None, region='GCcomb').iter(exclude=['zrange'], intersection=False):  # we are iterating on tracer, cut, etc. --- except zrange
                fig = plt.figure()
                for options in ft.iter_options():  # here we are iterating on zrange: stack on the same figure
                    fdata = fm.get(id='correlation_recon_y1', **options, check_exists=True)
                    fcovariance = fm.get(id='covariance_correlation_recon_y1', **options, ignore=['version'], check_exists=True, raise_error=False)
                    #fabacus = fm.select(id='correlation_abacus_recon_y1', **options, fa='altmtl', ignore=['weighting', 'version'], raise_error=False)
                    #fabacus = [fi for fi in fabacus if fi.exists()]
                    style.plot_correlation_multipoles(fdata, covariance=fcovariance, fig=fig, markers='point')
                    #if fabacus:
                    #    style.plot_correlation_multipoles(fabacus, fig=fig, markers='line')
                utils.savefig(outdir / 'correlation_recon_blinded_multipoles_{tracer}_{region}{cut}.png'.format(**fdata.foptions), fig=fig)
                plt.close(fig)
        """
        if 'unblinding' in todo:
            for ft in fm.select(id='correlation_recon_y1', weighting='default_FKP', version=version, region='GCcomb').iter(exclude=['cut'], intersection=False):  # we are iterating on tracer, cut, etc. --- except zrange
                fig = plt.figure()
                lim = (30., 180., 4.)
                fdata = ft.get(id='correlation_recon_y1', cut=None, check_exists=True)
                data = fdata.load().select(lim)
                data_cut = ft.get(id='correlation_recon_y1', cut=('theta', 0.05), check_exists=True).load().select(lim)
                s, corr = data(ells=0, return_sep=True)
                plt.plot(s, s**2 * corr, color='C0', linestyle='-')
                #s, corr = data_cut(ells=0, return_sep=True)
                #plt.plot(s, s**2 * corr, color='C0', linestyle='--')    
                utils.savefig(outdir / 'correlation_recon_cut_multipoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fdata.foptions), fig=fig)
                plt.close(fig)
        
        def plot_comparison_correlation_function_multipoles(all_data, ells=(0, 2, 4), select=(0., 200., 4.), labels=None, linestyles=None):
            fig, lax = plt.subplots(2, len(ells), figsize=(4 * len(ells), 4), sharey=False, sharex=True, gridspec_kw={'height_ratios': [4, 2]}, squeeze=False)
            if linestyles is None: linestyles = ['-', '--', ':', '-.']

            all_s, all_poles, all_list_poles, all_std = [], [], [], []

            for list_data in all_data:
                list_s, list_poles = [], []
                for data in list_data:
                    dd = data.select(select)
                    ss, pp = dd(ell=ells, return_sep=True, return_std=False)
                    #list_s.append(ss)
                    list_s.append(dd.sepavg(axis=0, method='mid'))
                    list_poles.append(pp)
                all_s.append(np.mean(list_s, axis=0))
                all_list_poles.append(list_poles)
                all_poles.append(np.mean(list_poles, axis=0))
                all_std.append(np.std(list_poles, ddof=1, axis=0) / len(list_data)**0.5)

            for ill, ell in enumerate(ells):
                color = 'C{}'.format(ill)
                ax = lax[0][ill]
                for s, poles, list_poles, linestyle, label in zip(all_s, all_poles, all_list_poles, linestyles, labels):
                    ax.plot(s, s**2 * poles[ill], color=color, linestyle=linestyle, label=label)
                    #for poles in list_poles:
                    #    ax.plot(s, s**2 * poles[ill], color=color, linestyle=linestyle, alpha=0.2)
                ax.set_title(r'$\ell = {}$'.format(ell))
                ax.grid(True)
                ax = lax[1][ill]
                for s, poles, std, linestyle in list(zip(all_s, all_poles, all_std, linestyles))[1:]:
                    ax.plot(s, (poles[ill] - all_poles[0][ill]) / std[ill], color=color, linestyle=linestyle)
                ax.set_xlabel(r'$s$ [$\mathrm{Mpc} / h$]')
                ax.set_ylim(-2., 2.)
                ax.grid(True)

            lax[0][0].set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
            lax[0][0].legend()
            lax[1][0].set_ylabel(r'$\Delta \xi_{\ell}(s) / \sigma$')
            fig.align_ylabels()
            return fig

        def plot_comparison_power_spectrum_multipoles(all_data, ells=(0, 2, 4), select=(0.02, 0.5, 0.005), labels=None, linestyles=None, colors=None):
            fig, lax = plt.subplots(2, len(ells), figsize=(4 * len(ells), 4), sharey=False, sharex=True, gridspec_kw={'height_ratios': [4, 2]}, squeeze=False)
            if linestyles is None: linestyles = ['-', '--', ':', '-.']

            all_k, all_poles, all_std = [], [], []

            for idata, list_data in enumerate(all_data):
                list_k, list_poles = [], []
                for data in list_data:
                    kk, pp = data.select(select)(ell=ells, return_k=True)
                    list_k.append(kk)
                    list_poles.append(pp)
                list_poles = np.array(list_poles)
                all_k.append(np.mean(list_k, axis=0))
                all_poles.append(np.mean(list_poles, axis=0))
                ndata = len(list_data)**0.5
                if True: #idata == 0:
                    list_poles_ref = list_poles
                    all_std.append(np.std(list_poles, ddof=1, axis=0) / ndata)
                else:
                    all_std.append(np.std(list_poles[-1] - list_poles_ref, ddof=1, axis=0) / ndata)

            for ill, ell in enumerate(ells):
                ax = lax[0][ill]
                for i, (k, poles, linestyle, label) in enumerate(zip(all_k, all_poles, linestyles, labels)):
                    if colors is None: color = 'C{:d}'.format(ill)
                    else: color = colors[i]
                    ax.plot(k, k * poles[ill], color=color, linestyle=linestyle, label=label)
                ax.set_title(r'$\ell = {}$'.format(ell))
                ax.grid(True)
                ax = lax[1][ill]
                for i, (k, poles, std, linestyle) in list(enumerate(zip(all_k, all_poles, all_std, linestyles)))[1:]:
                    if colors is None: color = 'C{:d}'.format(ill)
                    else: color = colors[i]
                    ax.plot(k, (poles[ill] - all_poles[0][ill]) / std[ill], color=color, linestyle=linestyle)
                ax.set_xlabel(r'$k$ [$h / \mathrm{Mpc}$]')
                ax.set_ylim(-2., 2.)
                ax.grid(True)

            lax[0][0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
            lax[0][0].legend()
            lax[1][0].set_ylabel(r'$\Delta P_{\ell}(k) / \sigma$')
            fig.align_ylabels()
            return fig
        
        if 'comparison_abacus_real' in todo:

            fma = get_abacus_file_manager() + get_raw_abacus_file_manager() + get_box_abacus_file_manager()
            imock = range(25)
            for fi in fma.select(id='correlation_abacus_y1', fa='complete', cut=None, region='GCcomb', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v3_1').iter(exclude='imock', intersection=False):
                options = fi.get(imock=0).options
                xi_raw_snapshot = [fma.get(id='correlation_raw_abacus_y1', **{**options, 'version': 'v3_1', 'catalog': 'rsd-no', 'weighting': '', 'imock': iimock}, ignore=True).load() for iimock in imock]
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                print(tracer, region, zrange)
                zsnap = get_zsnap_from_z(tracer=tracer, z=zrange)
                xi_box = [fma.get(id='correlation_box_abacus_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, ignore=True).load() for iimock in imock for los in [None]]
                fig = plot_comparison_correlation_function_multipoles([xi_box, xi_raw_snapshot], select=(10., 150., 4.), labels=['box', 'raw'])
                utils.savefig(outdir / 'comparison_correlation_abacus_box_real_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)

        if 'comparison_abacus' in todo:

            fma = get_abacus_file_manager() + get_raw_abacus_file_manager() + get_box_abacus_file_manager()
            imock = range(25)
            """
            for fi in fma.select(id='correlation_abacus_y1', fa='complete', cut=None, region='GCcomb', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v4', weighting='default_FKP').iter(exclude='imock', intersection=False):
                options = fi.get(imock=0).options
                xi_complete = [fi.get(imock=iimock).load() for iimock in imock]
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                zsnap = get_zsnap_from_z(tracer=tracer, z=zrange)
                #zsnap = get_zsnap_from_z(tracer=tracer, z=(0.8, 1.1))
                xi_box = [[fma.get(id='correlation_box_abacus_y1', version='v1.1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, ignore=True).load() for iimock in imock] for los in ['x', 'y', 'z']]
                xi_box = [fn for xi in xi_box for fn in xi]
                labels = ['box', 'complete']
                fig = plot_comparison_correlation_function_multipoles([xi_box, xi_complete], select=(10., 150., 4.), labels=labels)
                utils.savefig(outdir / 'comparison_correlation_abacus_box_complete_v4_weighting_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            """
            """
            for fi in fma.select(id='correlation_abacus_y1', fa='complete', cut=None, region='GCcomb', weighting='default_FKP', version='v4_1', tracer=['LRG', 'ELG_LOPnotqso', 'QSO']).iter(exclude=['imock'], intersection=False):
                options = fi.get(imock=0).options
                xi_complete = [fi.get(imock=iimock).load() for iimock in imock]
                #xi_complete_thetacut = [fma.get(id='correlation_abacus_y1', **dict(options, cut=('theta', 0.05), imock=iimock)).load() for iimock in imock]
                #xi_complete_window2 = [fma.get(id='correlation_abacus_y1', **dict(options, version='test_eb', catalog='rsd-snapshot', imock=iimock)).load() for iimock in imock]
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                zsnap = get_zsnap_from_z(tracer=tracer, z=zrange)
                #zsnap = get_zsnap_from_z(tracer=tracer, z=(0.8, 1.1))
                xi_box = [[fma.get(id='correlation_box_abacus_y1', version='v1.1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, ignore=True).load() for iimock in imock] for los in ['x', 'y', 'z']]
                xi_box = [fn for xi in xi_box for fn in xi]
                labels = ['box', 'complete']
                fig = plot_comparison_correlation_function_multipoles([xi_box, xi_complete], select=(10., 150., 4.), labels=labels)
                utils.savefig(outdir / 'comparison_correlation_abacus_box_complete_v4_1_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            """
            for fi in fma.select(id='correlation_raw_abacus_y1', cut=None, region='GCcomb', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v4_1').iter(exclude=['imock', 'weighting', 'catalog', 'zshuffled'], intersection=False):
                options = fi.get(weighting='', catalog='standard', zshuffled=False, imock=0).options
                xi_raw = [fi.get(weighting='', catalog='standard', zshuffled=False, imock=iimock).load() for iimock in imock]
                xi_masked = [fi.get(weighting='default_FKP', catalog='mask', zshuffled=False, imock=iimock).load() for iimock in imock if fi.get(weighting='default_FKP', catalog='mask', zshuffled=False, imock=iimock).exists()]
                xi_shuffled = [fi.get(weighting='default_FKP', catalog='standard', zshuffled=True, imock=iimock).load() for iimock in imock if fi.get(weighting='default_FKP', catalog='standard', zshuffled=True, imock=iimock).exists()]
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                zsnap = get_zsnap_from_z(tracer=tracer, z=zrange)
                xi_box = [[fma.get(id='correlation_box_abacus_y1', version='v1.1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, ignore=True).load() for iimock in imock] for los in ['x', 'y', 'z']]
                xi_box = [fn for xi in xi_box for fn in xi]
                labels = ['box', 'raw', 'raw, masked', 'shuffled']
                fig = plot_comparison_correlation_function_multipoles([xi_box, xi_raw, xi_masked, xi_shuffled], select=(10., 150., 4.), labels=labels)
                utils.savefig(outdir / 'comparison_correlation_abacus_box_raw_v4_1_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)

            for fi in fma.select(id='power_raw_abacus_y1', cut=None, region='GCcomb', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v4_1', weighting='', catalog='standard', zshuffled=False).iter(exclude='imock', intersection=False):
                options = fi.get(imock=0).options
                pk_raw = [fi.get(imock=iimock).load() for iimock in imock]
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                zsnap = get_zsnap_from_z(tracer=tracer, z=zrange)
                pk_box = [[fma.get(id='power_box_abacus_y1', version='v1.1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, ignore=True).load() for iimock in imock] for los in ['x', 'y', 'z']]
                pk_box = [fn for xi in pk_box for fn in xi]
                labels = ['box', 'raw']
                fig = plot_comparison_power_spectrum_multipoles([pk_box, pk_raw], select=(0.005, 0.3, 0.005), labels=labels)
                utils.savefig(outdir / 'comparison_power_abacus_box_raw_v4_1_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            """
            for fi in fma.select(id='correlation_abacus_y1', fa='complete', cut=None, region='GCcomb', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'][1:2], version='v3_1', weighting='default_FKP', zrange=(1.1, 1.6)).iter(exclude='imock', intersection=False):
                options = fi.get(imock=0).options
                xi_complete = [fi.get(imock=iimock).load() for iimock in imock]
                xi_complete_no_fkp = [fma.get(id='correlation_abacus_y1', **{**options, 'weighting': 'default', 'imock': iimock}).load() for iimock in imock]
                #xi_complete_no_fkp = [fma.get(id='correlation_abacus_y1', **{**options, 'weighting': 'default_FKP', 'version': 'v3_1_c', 'imock': iimock}).load() for iimock in imock]
                xi_complete_zcut = [fma.get(id='correlation_abacus_y1', **{**options, 'weighting': 'default_FKP', 'version': 'v3_1_b', 'zrange': (1.1, 1.58), 'imock': iimock}).load() for iimock in imock]
                #xi_complete = xi_complete_zcut
                xi_raw_current = [fma.get(id='correlation_raw_abacus_y1', **{**options, 'version': 'v3_1', 'catalog': 'rsd-z', 'weighting': '', 'imock': iimock}, ignore=True).load() for iimock in imock]
                xi_raw_current_snapshot = [fma.get(id='correlation_raw_abacus_y1', **{**options, 'version': 'v3_1', 'catalog': 'rsd-snapshot', 'weighting': '', 'imock': iimock}, ignore=True).load() for iimock in imock]
                #xi_raw_snapshot = [fma.get(id='correlation_raw_abacus_y1', **{**options, 'catalog': 'rsd-snapshot', 'weighting': '', 'imock': iimock}, ignore=True).load() for iimock in imock]
                xi_raw_snapshot = [fma.get(id='correlation_raw_abacus_y1', **{**options, 'version': 'v3_1_b', 'catalog': 'standard', 'weighting': '', 'imock': iimock}, ignore=True).load() for iimock in imock]
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                print(tracer, region, zrange, len(xi_raw_snapshot))
                #print(tracer, zrange, get_zsnap_from_z(tracer=tracer, z=zrange))
                zsnap = get_zsnap_from_z(tracer=tracer, z=zrange)
                #zsnap = get_zsnap_from_z(tracer=tracer, z=(0.8, 1.1))
                xi_box = [[fma.get(id='correlation_box_abacus_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, ignore=True).load() for iimock in imock] for los in ['x', 'y', 'z']]
                #labels = ['raw snapshot-rsd (new)', 'box los-x', 'box los-y', 'box los-z']
                #fig = plot_comparison_correlation_function_multipoles([xi_raw_snapshot] + xi_box, select=(10., 150., 4.), labels=labels)
                xi_box = [fn for xi in xi_box for fn in xi]
                #labels = ['box', 'complete', 'raw snapshot-rsd (current)', 'raw (current)']
                #fig = plot_comparison_correlation_function_multipoles([xi_box, xi_complete, xi_raw_current_snapshot, xi_raw_snapshot], select=(10., 150., 4.), labels=labels)
                #labels = ['box', 'complete', 'complete no FKP', 'raw (current)']
                labels = ['box', 'complete', 'complete no FKP', 'complete $z < 1.58$']
                fig = plot_comparison_correlation_function_multipoles([xi_box, xi_complete, xi_complete_no_fkp, xi_complete_zcut], select=(10., 150., 4.), labels=labels)
                #utils.savefig(outdir / 'comparison_correlation_abacus_box_los_raw_new_complete_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                utils.savefig(outdir / 'comparison_correlation_abacus_box_complete_zcut_weighting_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            """

        if 'comparison_box_abacus' in todo:

            fma = get_box_abacus_file_manager()
            imock = range(25)
            llos = ['x', 'y', 'z']
            #llos = [None]
            for fi in fma.select(id='correlation_box_abacus_y1', version='v1.1', tracer=['LRG', 'ELG_LOPnotqso', 'QSO']).iter(exclude=['imock', 'los'], intersection=False):
                options = fi.get(imock=0, los='x').options
                xi_box = [fma.get(id='correlation_box_abacus_y1', **{**options, 'version': 'v1', 'los': los, 'imock': iimock}).load() for los in llos for iimock in imock]
                xi_box_v11 = [fma.get(id='correlation_box_abacus_y1', **{**options, 'version': 'v1.1', 'los': los, 'imock': iimock}).load() for los in llos for iimock in imock]
                labels = ['box', 'box v1.1']
                fig = plot_comparison_correlation_function_multipoles([xi_box, xi_box_v11], select=(10., 150., 4.), labels=labels)
                utils.savefig(outdir / 'comparison_correlation_abacus_box_version_{tracer}_z{zsnap:.1f}.png'.format(**options), fig=fig)
                plt.close(fig)

        if 'comparison_abacus_fa' in todo:

            fma = get_abacus_file_manager() + get_box_abacus_file_manager()
            imock = range(25)
            """
            for options in fma.select(id='correlation_abacus_y1', region='GCcomb', cut=('theta', 0.05), tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], njack=0).iter_options(exclude=['imock', 'fa', 'version'], intersection=False):
                xi_complete = [fma.get(id='correlation_abacus_y1', **options, fa='complete', version='v4_2', imock=iimock).load() for iimock in imock]
                xi_ffa = [fma.get(id='correlation_abacus_y1', **options, fa='ffa', version='v4_2', imock=iimock).load() for iimock in imock]
                xi_altmtl = [fma.get(id='correlation_abacus_y1', **options, fa='altmtl', version='v4_2', imock=iimock).load() for iimock in imock]
                print(len(xi_complete), len(xi_ffa), len(xi_altmtl))
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                print(tracer, region, zrange)
                fig = plot_comparison_correlation_function_multipoles([xi_complete, xi_ffa, xi_altmtl], select=(10., 150., 4.), labels=['complete', 'ffa', 'altmtl'])
                utils.savefig(outdir / 'comparison_correlation_abacus_complete_ffa_altmtl_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            """
            """
            for options in fma.select(id='power_abacus_y1', region='GCcomb', cut=('theta', 0.05), tracer=['LRG', 'ELG_LOPnotqso', 'QSO'][1:2], zrange=(1.1, 1.6)).iter_options(exclude=['imock', 'fa', 'version'], intersection=False):
                version = 'v4_2'
                pk_complete = [fma.get(id='power_abacus_y1', **options, fa='complete', version=version, imock=iimock).load() for iimock in imock]
                pk_ffa = [fma.get(id='power_abacus_y1', **options, fa='ffa', version=version, imock=iimock).load() for iimock in imock]
                pk_altmtl = [fma.get(id='power_abacus_y1', **options, fa='altmtl', version=version, imock=iimock).load() for iimock in imock]
                print(len(pk_complete), len(pk_ffa), len(pk_altmtl))
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                print(tracer, region, zrange)
                fig = plot_comparison_power_spectrum_multipoles([pk_complete, pk_ffa, pk_altmtl], select=(0.02, 0.2, 0.005), labels=['complete', 'ffa', 'altmtl'])
                utils.savefig(outdir / 'comparison_power_abacus_complete_ffa_altmtl_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            """
            imock = range(4)
            for options in fma.select(id='power_abacus_y1', region='GCcomb', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'][1:2], zrange=(1.1, 1.6)).iter_options(exclude=['imock', 'fa', 'version', 'cut'], intersection=False):
                version = 'v4_2'
                cut = ('theta', 0.05)

                pk_complete_cut = [fma.get(id='power_abacus_y1', **options, fa='complete', cut=cut, version=version, imock=iimock).load() for iimock in imock]
                pk_altmtl_cut = [fma.get(id='power_abacus_y1', **options, fa='altmtl', cut=cut, version=version, imock=iimock).load() for iimock in imock]
                pk_ffa_cut = [fma.get(id='power_abacus_y1', **options, fa='ffa', cut=cut, version=version, imock=iimock).load() for iimock in imock]

                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                fig = plot_comparison_power_spectrum_multipoles([pk_complete_cut, pk_ffa_cut, pk_altmtl_cut], select=(0.02, 0.2, 0.005), labels=[r'complete $\theta = 0.05$', r'ffa $\theta = 0.05$', r'altmtl $\theta = 0.05$'])
                utils.savefig(outdir / 'comparison_power_abacus_thetacut0.05_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            
            imock = range(4)
            for options in fma.select(id='power_abacus_y1', region='GCcomb', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'][1:2], zrange=(1.1, 1.6)).iter_options(exclude=['imock', 'fa', 'version', 'cut'], intersection=False):
                version = 'v4_2'
                cut2 = ('theta', 0.07)
                pk_complete_cut2 = [fma.get(id='power_abacus_y1', **options, fa='complete', cut=cut2, version=version, imock=iimock).load() for iimock in imock]
                pk_altmtl_cut2 = [fma.get(id='power_abacus_y1', **options, fa='altmtl', cut=cut2, version=version, imock=iimock).load() for iimock in imock]
                pk_ffa_cut2 = [fma.get(id='power_abacus_y1', **options, fa='ffa', cut=cut2, version=version, imock=iimock).load() for iimock in imock]

                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                fig = plot_comparison_power_spectrum_multipoles([pk_complete_cut2, pk_ffa_cut2, pk_altmtl_cut2], select=(0.02, 0.2, 0.005), labels=[r'complete $\theta = 0.07$', r'ffa $\theta = 0.07$', r'altmtl $\theta = 0.07$'])
                utils.savefig(outdir / 'comparison_power_abacus_thetacut0.07_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)

        if 'residual_abacus' in todo:
            fma = get_raw_abacus_file_manager() + get_ez_file_manager()

            options = dict(weighting='', version='v3_1_b', catalog='standard', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], region='GCcomb', cut=None)
            for fd in fma.select(id='correlation_raw_abacus_y1', **options).iter(exclude=['imock'], intersection=False):
                options = fd.options.copy()
                options.pop('nran')
                options.pop('imock')
                print([str(ff) for ff in fd])
                fi = list(fm.select(id='correlation_ez_y1', **{**options, 'weighting': 'default_FKP', 'fa': 'ffa', 'version': 'v1'}, ignore=True).iter(intersection=False))
                """
                fig = style.plot_covariance_correlation_multipoles(covariance=fi, select=(10., 200., 4.), ells=(0, 2, 4))
                utils.savefig(outdir / 'covariance_ez_multipoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                fig = style.plot_covariance_correlation_multipoles(covariance=fi, stat='residual', select=(10., 200., 10.), ells=(0, 2, 4))
                utils.savefig(outdir / 'covariance_residual_ez_multipoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                """
                fig = style.plot_residual_multipoles(fd, covariance=fi, select=(10., 200., 10.), ells=(0, 2, 4))
                utils.savefig(outdir / 'residual_abacus_multipoles_raw_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fd[0].foptions), fig=fig)
                plt.close(fig)

        if 'residual_abacus_fa' in todo:
            fma = get_abacus_file_manager() + get_ez_file_manager()

            options = dict(weighting='default_FKP', version='v3_1', fa='altmtl', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], region='GCcomb', cut=[None, ('theta', 0.05)])
            for fd in fma.select(id='correlation_abacus_y1', **options).iter(exclude=['imock'], intersection=False):
                options = fd.options.copy()
                options.pop('nran')
                options.pop('imock')
                print([str(ff) for ff in fd])
                fi = list(fm.select(id='correlation_ez_y1', **{**options, 'fa': 'ffa', 'version': 'v1'}, ignore=True).iter(intersection=False))
                """
                fig = style.plot_covariance_correlation_multipoles(covariance=fi, select=(10., 200., 4.), ells=(0, 2, 4))
                utils.savefig(outdir / 'covariance_ez_multipoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                fig = style.plot_covariance_correlation_multipoles(covariance=fi, stat='residual', select=(10., 200., 10.), ells=(0, 2, 4))
                utils.savefig(outdir / 'covariance_residual_ez_multipoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                """
                fig = style.plot_residual_multipoles(fd, covariance=fi, select=(10., 200., 10.), ells=(0, 2, 4))
                utils.savefig(outdir / 'residual_abacus_multipoles_{fa}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fd[0].foptions), fig=fig)
                plt.close(fig)

        def plot_wmatrix_validation(data, theory, wmatrix, klim=None, covmatrix=None, shotnoise=True, marg_shotnoise=True, nobs=None, ells=None, fn=None):
            ellsin = [0, 2, 4]
            kout = data[0].k
            if ells is None:
                ells = data[0].ells
            if nobs is None: nobs = len(data)
            kin = wmatrix.xin[0]
            #print(kin.max())
            wmatrix = wmatrix.value.T
            print(shotnoise, marg_shotnoise)
            mean_shotnoise = np.mean([dd.shotnoise for dd in data], axis=0) * shotnoise
            data = np.mean([dd(ell=ells, complex=False, remove_shotnoise=True) for dd in data], axis=0)
            kth = theory[0].k
            theory = np.mean([dd(ell=ellsin, complex=False, remove_shotnoise=True) for dd in theory], axis=0)
            #theory[ells.index(0)] += mean_shotnoise
            mask = ~np.isnan(theory).any(axis=0)
            kth, theory = kth[mask], theory[..., mask]
            theory_interp = np.array([np.interp(kin, kth, theory[ill], left=0., right=0.) + (ell == 0) * mean_shotnoise for ill, ell in enumerate(ellsin)])
            theory_rotated = wmatrix.dot(theory_interp.ravel()).reshape(len(ellsin), -1)
            std = (np.diag(covmatrix).reshape(len(ells), -1) / nobs)**0.5
            #theory[ells.index(0)] -= mean_shotnoise
            theory_rotated[ells.index(0)] -= mean_shotnoise

            fig, lax = plt.subplots(2, len(ells), figsize=(3 * len(ells), 4), sharey=False, sharex=True, gridspec_kw={'height_ratios': [4, 2]}, squeeze=False)
            
            if marg_shotnoise:  # shotnoise free

                precmatrix = np.linalg.inv(covmatrix)
                mask_shotnoise_in = np.concatenate([(ell == 0) * np.ones(len(kin), dtype='?') for ell in ellsin])
                #mask_shotnoise_out = np.concatenate([(ell == 0) * np.ones(len(kout), dtype='?') for ell in ells])
                deriv = np.matmul(wmatrix, mask_shotnoise_in)[None, :]
                derivp = deriv.dot(precmatrix)
                fisher = derivp.dot(deriv.T)
                shotnoise_value = np.linalg.solve(fisher, derivp.dot(data.ravel() - theory_rotated.ravel()))
                theory_rotated_shotnoise = theory_rotated.copy()
                theory_rotated_shotnoise += shotnoise_value.dot(deriv).reshape(len(ellsin), -1)
            
            for ill, ell in enumerate(ells):
                color = 'C{}'.format(ill)
                ax = lax[0][ill]
                ax.errorbar(kout, kout * data[ill], kout * std[ill], color=color, marker='.', ls='', label=r'$P_{\mathrm{o}}(k)$')
                ax.plot(kout, kout * np.interp(kout, kth, theory[ill], left=0., right=0.), color=color, ls=':', label=r'$P_{\mathrm{t}}(k)$')
                if marg_shotnoise:
                    ax.plot(kout, kout * theory_rotated_shotnoise[ill], color=color, ls='--', label=r'$W(k, k^{\prime}) (P_{\mathrm{t}}(k^{\prime}) + N)$')
                else:
                    ax.plot(kout, kout * theory_rotated[ill], color=color, label=r'$W(k, k^{\prime}) P_{\mathrm{t}}(k^{\prime})$')
                ax.set_title(r'$\ell = {}$'.format(ell))
                ax.set_xlim(klim)
                ax.grid(True)
                ax = lax[1][ill]
                diff = (theory_rotated_shotnoise if marg_shotnoise else theory_rotated)[ill] - data[ill]
                ax.plot(kout, diff / std[ill], color=color)
                ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
                ax.set_ylim(-2., 2.)
                ax.grid(True)

            lax[0][0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
            lax[0][0].legend()
            lax[1][0].set_ylabel(r'$\Delta P_{\ell}(k) / \sigma$')
            fig.align_ylabels()
            if fn is not None:
                utils.savefig(fn, fig=fig)
                plt.close(fig)
    
        if 'wmatrix_validation_abacus' in todo:

            imock = list(range(25))
            #kinlim = (0., 0.5)
            kinlim = (0., np.inf)
            koutlim = (0.02, 0.2)
            kinrebin = 1
            koutrebin = 5
            region = ['NGC', 'SGC', 'GCcomb'][2:]
            dfm = get_data_file_manager()
            tfm = get_box_abacus_file_manager()
            fm = get_abacus_file_manager()
            fmr = get_raw_abacus_file_manager()

            for corrected in ['', '_corrected']:

                for tracer in ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']:
                    for fi in fm.select(id='wmatrix_power_merged_abacus_y1', fa='complete', cut=None, region=region, weighting='default_FKP', version='v1' if 'BGS' in tracer else 'v4_2', tracer=tracer).iter(intersection=False):
                        fid = fi.id
                        options = fi.options
                        wmatrix = fi.load()
                        wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)

                        fcovariance = dfm.get(id='covariance_y1', **{**options, 'region': 'GCcomb', 'cut': None, 'source': 'thecov', 'version': 'v1.2', 'observable': 'power'}, ignore=True)
                        covmatrix = fcovariance.load().view(xlim=koutlim)

                        def get_power(data):
                            data = data.load()
                            koutrebin = 1 if corrected else 5
                            return data[:(data.shape[0] // koutrebin) * koutrebin:koutrebin].select(koutlim)

                        data = [get_power(fm.get(id='power{}_abacus_y1'.format(corrected), **options, imock=iimock, ignore=True)) for iimock in imock]
                        #data = [get_power(fmr.get(id='power_raw_abacus_y1', **{**options, 'version': 'v4_1_complete', 'catalog': 'standard', 'zshuffled': True, 'weighting': 'default'}, imock=iimock, ignore=True)) for iimock in imock]
                        tracer, zrange, region = options['tracer'], options['zrange'], options['region']
                        zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                        theory = [tfm.get(id='power_box_abacus_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, version='v0.1' if 'BGS' in tracer else 'v1.1', ignore=True).load() for iimock in imock for los in ['x', 'y', 'z']]

                        fig = plot_wmatrix_validation(data, theory, wmatrix, covmatrix=covmatrix, shotnoise=False, marg_shotnoise=True, nobs=25)
                        utils.savefig(outdir / 'wmatrix_validation{}_abacus_complete_v4_2_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(corrected, tracer=tracer, region=region, zrange=zrange), fig=fig)
                        plt.close(fig)

        if 'wmatrix_validation_thetacut_abacus' in todo:

            imock = list(range(25))
            #kinlim = (0., 0.5)
            kinlim = (0., np.inf)
            koutlim = (0.02, 0.2)
            kinrebin = 1
            region = ['NGC', 'SGC', 'GCcomb'][2:]
            dfm = get_data_file_manager()
            tfm = get_box_abacus_file_manager()
            fm = get_abacus_file_manager()
            fmr = get_raw_abacus_file_manager()
            
            for corrected in ['', '_corrected']:

                for rotated in ['', '_rotated']:

                    for fa in ['complete', 'ffa', 'altmtl']: #[:1]:
                        for tracer in ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']: #[-1:]:
                            for fi in fm.select(id='wmatrix_power_merged{}_abacus_y1'.format(rotated), fa=fa, cut=('theta', 0.05), region=region, weighting='default_FKP', version='v1' if 'BGS' in tracer else 'v4_2', tracer=tracer).iter(intersection=False):
                                fid = fi.id
                                options = dict(fi.options)
                                wmatrix = fi.load()
                                koutrebin = 1 if rotated else 5
                                wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)

                                cov_options = {**options, 'region': 'GCcomb', 'source': 'thecov', 'version': 'v1.2', 'observable': 'power'}
                                ells = [0, 2] if rotated else [0, 2, 4]
                                fcovariance = fm.get(id='covariance{}_abacus_y1'.format(rotated), **cov_options, ignore=True)
                                covmatrix = fcovariance.load().view(xlim=koutlim, projs=ells)
                                #print(covmatrix.shape, wmatrix.shape)
                                #exit()

                                def get_power(data):
                                    data = data.load()
                                    koutrebin = 1 if rotated or corrected else 5
                                    return data[:(data.shape[0] // koutrebin) * koutrebin:koutrebin].select(koutlim)

                                data = [get_power(fm.get(id='power{}{}_abacus_y1'.format(rotated, corrected), **options, imock=iimock, ignore=True)) for iimock in imock]
                                #data = [get_power(fmr.get(id='power_raw_abacus_y1', **{**options, 'version': 'v4_1_complete', 'catalog': 'standard', 'zshuffled': True, 'weighting': 'default'}, imock=iimock, ignore=True)) for iimock in imock]
                                tracer, zrange, region = options['tracer'], options['zrange'], options['region']
                                zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                                theory = [tfm.get(id='power_box_abacus_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, version='v0.1' if 'BGS' in tracer else 'v1.1', ignore=True).load() for iimock in imock for los in ['x', 'y', 'z']]

                                fig = plot_wmatrix_validation(data, theory, wmatrix, covmatrix=covmatrix, shotnoise=False, marg_shotnoise=not rotated, ells=ells, nobs=25)
                                #fig = plot_wmatrix_validation(data, theory, wmatrix, covmatrix=covmatrix, shotnoise=True, marg_shotnoise=True, nobs=25)
                                utils.savefig(outdir / 'wmatrix_validation{}{}_abacus_{}_thetacut0.05_v4_2_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(rotated, corrected, fa, tracer=tracer, region=region, zrange=zrange), fig=fig)
                                plt.close(fig)

        if 'wmatrix_validation_thetacut_ez' in todo:

            #kinlim = (0., 0.5)
            kinlim = (0., np.inf)
            koutlim = (0.01, 0.2)
            kinrebin = 1
            region = ['NGC', 'SGC', 'GCcomb'][2:]
            dfm = get_data_file_manager()
            tfm = get_box_ez_file_manager()
            fm = get_ez_file_manager()

            for rotated in ['', '_rotated']:

                koutrebin = 1 if rotated else 5

                for fa in ['ffa']:
                    for tracer in ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']:
                        for fi in fm.select(id='wmatrix_power_merged{}_ez_y1'.format(rotated), fa=fa, cut=('theta', 0.05), region=region, weighting='default_FKP', version='v1ric', tracer=tracer).iter(intersection=False):
                            for version in ['v1ric', 'v1noric']:
                                fid = fi.id
                                options = dict(fi.options)
                                wmatrix = fi.load()
                                wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)

                                cov_options = {**options, 'region': 'GCcomb', 'source': 'thecov', 'version': 'v1.2', 'observable': 'power'}
                                ells = [0, 2] if rotated else [0, 2, 4]
                                fcovariance = fm.get(id='covariance{}_ez_y1'.format(rotated), **cov_options, ignore=True)
                                covmatrix = fcovariance.load().view(xlim=koutlim, projs=ells)
                                #print(covmatrix.shape, wmatrix.shape)
                                #exit()

                                def get_power(data):
                                    data = data.load()
                                    return data[:(data.shape[0] // koutrebin) * koutrebin:koutrebin].select(koutlim)

                                options.pop('nran')
                                data = [get_power(fm.get(id='power{}_ez_y1'.format(rotated), **{**options, 'version': version}, imock=iimock, ignore=True)) for iimock in range(1, 51)]
                                #data = [get_power(fmr.get(id='power_raw_abacus_y1', **{**options, 'version': 'v4_1_complete', 'catalog': 'standard', 'zshuffled': True, 'weighting': 'default'}, imock=iimock, ignore=True)) for iimock in imock]
                                tracer, zrange, region = options['tracer'], options['zrange'], options['region']
                                zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                                theory = [tfm.get(id='power_box_ez_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, version='v1', ignore=True).load() for iimock in range(1, 11) for los in ['x', 'y', 'z']]

                                fig = plot_wmatrix_validation(data, theory, wmatrix, covmatrix=covmatrix, shotnoise=False, marg_shotnoise=not rotated, ells=ells, nobs=25)
                                utils.savefig(outdir / 'wmatrix_validation{}_ez_{}_thetacut0.05_{version}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(rotated, fa, version=version, tracer=tracer, region=region, zrange=zrange), fig=fig)
                                plt.close(fig)
                        
        if 'comparison_cutsky_raw' in todo:

            from desi_y1_files.file_manager import get_abacus_file_manager_test
            imock = list(range(25))
            fma = get_abacus_file_manager()
            fma2 = get_abacus_file_manager_test()
            fm = get_raw_abacus_file_manager()

            """
            for fi in fm.select(id='power_raw_abacus_y1', cut=None, region=['NGC'], tracer=['LRG', 'ELG_LOPnotqso', 'QSO'][1:2], version='v4_1', weighting='default_FKP', catalog='standard', zshuffled=False, zrange=(1.1, 1.6)).iter(exclude='imock', intersection=False):
                options = fi.get(imock=0).options
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                pk_raw = [fi.get(imock=iimock).load() for iimock in imock]
                #pk_raw_aurelio = [fm.get(id='power_raw_abacus_y1', **{**options, 'version': 'v3_1', 'catalog': 'rsd-snapshot', 'weighting': '', 'imock': iimock}, ignore=True).load() for iimock in imock]
                pk_raw_mask = [fm.get(id='power_raw_abacus_y1', **{**options, 'catalog': 'mask', 'imock': iimock}, ignore=True).load() for iimock in imock]
                #pk_raw_zshuffled = [fm.get(id='power_raw_abacus_y1', **{**options, 'catalog': 'mask', 'zshuffled': True, 'imock': iimock}, ignore=True).load() for iimock in imock]
                #options = {'region': 'GCcomb', 'version': 'v4_1', 'tracer': 'LRG', 'imock': 0, 'zrange': (0.4, 0.6), 'weighting': 'default_FKP', 'binning': 'lin', 'cut': None, 'cellsize': 6.0, 'boxsize': 7000.0, 'nran': 18}
                pk_complete = [fma.get(id='power_abacus_y1', **{**options, 'fa': 'complete', 'imock': iimock}, ignore=True).load() for iimock in imock]
                #pk_complete2 = [fma2.get(id='power_abacus_y1', **{**options, 'fa': 'complete', 'imock': iimock, 'zrange': (1.12, 1.58), 'weighting': 'default'}, ignore=True).load() for iimock in imock]
                pk_complete_zshuffled = [fm.get(id='power_raw_abacus_y1', **{**options, 'version': 'v4_1_complete', 'catalog': 'standard', 'zshuffled': True, 'imock': iimock, 'weighting': 'default'}, ignore=True).load() for iimock in imock]
                fig = plot_comparison_power_spectrum_multipoles([pk_raw_mask, pk_complete, pk_complete_zshuffled], select=(0.005, 0.3, 0.005), labels=['raw, masked', 'complete', 'complete z-shuffled'])
                utils.savefig(outdir / 'comparison_power_abacus_cutsky_raw_v4_1_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            """
            for fi in fm.select(id='power_raw_abacus_y1', cut=None, region=['GCcomb'], tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v4_1', weighting='default_FKP', catalog='standard', zshuffled=False).iter(exclude='imock', intersection=False):
                options = fi.get(imock=0).options
                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                pk_raw = [fi.get(imock=iimock).load() for iimock in imock]
                pk_raw_aurelio = [fm.get(id='power_raw_abacus_y1', **{**options, 'version': 'v3_1', 'catalog': 'rsd-snapshot', 'weighting': '', 'imock': iimock}, ignore=True).load() for iimock in imock]
                pk_complete = [fma.get(id='power_abacus_y1', **{**options, 'version': 'v4_1fixran', 'fa': 'complete', 'imock': iimock}, ignore=True).load() for iimock in imock]
                fig = plot_comparison_power_spectrum_multipoles([pk_raw, pk_raw_aurelio, pk_complete], select=(0.005, 0.3, 0.005), labels=['raw', 'raw Aurelio', 'complete'])
                utils.savefig(outdir / 'comparison_power_abacus_cutsky_raw_v4_1fixran_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
                
        if 'wmatrix_validation_raw' in todo:

            imock = range(25)
            kinlim = (0., 0.5)
            koutlim = (0.01, 0.2)
            kinrebin = 1
            koutrebin = 5
            ells = (0, 2, 4)
            region = ['NGC', 'SGC', 'GCcomb']
            dfm = get_data_file_manager()
            tfm = get_box_abacus_file_manager()
            fm = get_raw_abacus_file_manager()

            """
            for fi in fm.select(id='wmatrix_power_merged_raw_abacus_y1', cut=None, region=region, weighting='default_FKP', version='v4_1', tracer=['ELG_LOPnotqso'], zrange=(1.1, 1.6), catalog='mask', zshuffled=False).iter(intersection=False):
                fid = fi.id
                options = fi.options
                wmatrix = fi.load()
                wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)
                print(options)
                fcovariance = dfm.get(id='covariance_y1', **{**options, 'region': 'GCcomb', 'cut': None, 'source': 'thecov', 'version': 'v1.2', 'observable': 'power'}, ignore=True)
                covmatrix = fcovariance.load().view(xlim=koutlim)
                
                def get_power(data):
                    data = data.load()
                    return data[:(data.shape[0] // koutrebin) * koutrebin:koutrebin].select(koutlim)

                data = [get_power(fm.get(id='power_raw_abacus_y1', **{**options, 'version': 'v4_1'}, imock=iimock, ignore=True)) for iimock in imock]
                tracer, zrange, region = options['tracer'], options['zrange'], options['region']
                zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                theory = [tfm.get(id='power_box_abacus_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, version='v1.1', ignore=True).load() for iimock in imock for los in ['x', 'y', 'z']]

                fig = plot_wmatrix_validation(data, theory, wmatrix, covmatrix=covmatrix)
                utils.savefig(outdir / 'wmatrix_validation_raw_mask_v4_1_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            """
            for fi in fm.select(id='wmatrix_power_merged_raw_abacus_y1', cut=None, region=region, weighting='default_FKP', version='v4_1', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], catalog='standard', zshuffled=False).iter(intersection=False):
                fid = fi.id
                options = fi.options
                wmatrix = fi.load()
                wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)
                print(options)
                fcovariance = dfm.get(id='covariance_y1', **{**options, 'region': 'GCcomb', 'cut': None, 'source': 'thecov', 'version': 'v1.2', 'observable': 'power'}, ignore=True)
                covmatrix = fcovariance.load().view(xlim=koutlim)
                
                def get_power(data):
                    data = data.load()
                    return data[:(data.shape[0] // koutrebin) * koutrebin:koutrebin].select(koutlim)

                data = [get_power(fm.get(id='power_raw_abacus_y1', **{**options, 'version': 'v4_1'}, imock=iimock, ignore=True)) for iimock in imock]
                tracer, zrange, region = options['tracer'], options['zrange'], options['region']
                zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                theory = [tfm.get(id='power_box_abacus_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, version='v1.1', ignore=True).load() for iimock in imock for los in ['x', 'y', 'z']]

                fig = plot_wmatrix_validation(data, theory, wmatrix, covmatrix=covmatrix)
                utils.savefig(outdir / 'wmatrix_validation_raw_v4_1_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)

        if 'wmatrix_validation_raw_thetacut' in todo:

            imock = range(25)
            kinlim = (0., 0.5)
            koutlim = (0.01, 0.2)
            kinrebin = 1
            koutrebin = 5
            ells = (0, 2, 4)
            region = ['NGC', 'SGC', 'GCcomb']
            dfm = get_data_file_manager()
            tfm = get_box_abacus_file_manager()
            fm = get_raw_abacus_file_manager()

            for fi in fm.select(id='wmatrix_power_merged_raw_abacus_y1', cut=('theta', 0.05), region=region, weighting='default_FKP', version='v4_1', tracer=['LRG'], zrange=(0.6, 0.8), catalog='standard', zshuffled=False).iter(intersection=False):
                fid = fi.id
                options = fi.options
                wmatrix = fi.load()
                wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)
                print(options)
                fcovariance = dfm.get(id='covariance_y1', **{**options, 'region': 'GCcomb', 'cut': None, 'source': 'thecov', 'version': 'v1.2', 'observable': 'power'}, ignore=True)
                covmatrix = fcovariance.load().view(xlim=koutlim)
                
                def get_power(data):
                    data = data.load()
                    return data[:(data.shape[0] // koutrebin) * koutrebin:koutrebin].select(koutlim)

                data = [get_power(fm.get(id='power_raw_abacus_y1', **{**options, 'weighting': 'default', 'version': 'v4_1'}, imock=iimock, ignore=True)) for iimock in imock]
                tracer, zrange, region = options['tracer'], options['zrange'], options['region']
                zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                theory = [tfm.get(id='power_box_abacus_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, version='v1.1', ignore=True).load() for iimock in imock for los in ['x', 'y', 'z']]

                fig = plot_wmatrix_validation(data, theory, wmatrix, covmatrix=covmatrix)
                utils.savefig(outdir / 'wmatrix_validation_raw_thetacut_v4_1_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)

        if 'wmatrix_validation_ez' in todo:

            imock = range(1, 51)
            kinlim = (0., 0.5)
            koutlim = (0.01, 0.2)
            kinrebin = 1
            koutrebin = 5
            ells = (0, 2, 4)
            region = ['GCcomb']
            dfm = get_data_file_manager()
            tfm = get_box_ez_file_manager()
            fm = get_ez_file_manager()

            for fi in fm.select(id='wmatrix_power_merged_ez_y1', cut=None, region=region, weighting='default_FKP', version='v1', tracer=['LRG', 'ELG_LOPnotqso', 'QSO']).iter(intersection=False):
                fid = fi.id
                options = fi.options
                wmatrix = fi.load()
                wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)
                print(options)
                fcovariance = dfm.get(id='covariance_y1', **{**options, 'region': 'GCcomb', 'cut': None, 'source': 'thecov', 'version': 'v1.2', 'observable': 'power'}, ignore=True)
                covmatrix = fcovariance.load().view(xlim=koutlim)
                
                def get_power(data):
                    data = data.load()
                    return data[:(data.shape[0] // koutrebin) * koutrebin:koutrebin].select(koutlim)

                options.pop('nran', None)
                print(options, imock)
                data = [get_power(fm.get(id='power_ez_y1', **{**options, 'version': 'v1noric'}, imock=iimock, ignore=True)) for iimock in imock]
                tracer, zrange, region = options['tracer'], options['zrange'], options['region']
                zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                theory = [tfm.get(id='power_box_ez_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, version='v1', ignore=True).load() for iimock in range(1, 11) for los in ['x', 'y', 'z']]

                fig = plot_wmatrix_validation(data, theory, wmatrix, covmatrix=covmatrix)
                utils.savefig(outdir / 'wmatrix_validation_ez_v1_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
                
        if 'counts' in todo:

            tracer = 'ELG_LOPnotqso'
            region = 'NGC'
            zrange = (1.1, 1.6)
            imock = range(25)
            fma = get_abacus_file_manager()
            fmr = get_raw_abacus_file_manager()

            xi_complete = [fma.get(id='correlation_abacus_y1', **{'tracer': tracer, 'zrange': zrange, 'region': region, 'version': 'v4_1', 'fa': 'complete', 'cut': None, 'njack': 0, 'imock': iimock}, ignore=True).load() for iimock in imock]
            xi_raw = [fmr.get(id='correlation_raw_abacus_y1', **{'tracer': tracer, 'zrange': zrange, 'region': region, 'version': 'v4_1', 'catalog': 'standard', 'zshuffled': False, 'weighting': 'default_FKP', 'cut': None, 'njack': 0, 'imock': iimock}, ignore=True).load() for iimock in imock]
            
            from matplotlib import pyplot as plt
            ax = plt.gca()
            imock, isbin = 1, 18
            s, mu = xi_complete[imock].sepavg(axis=0), xi_complete[imock].sepavg(axis=1)
            ax.plot(mu, xi_complete[imock].R1R2.wcounts[isbin], label='complete')
            s, mu = xi_raw[imock].sepavg(axis=0), xi_raw[imock].sepavg(axis=1)
            ax.plot(mu, xi_raw[imock].R1R2.wcounts[isbin], label='raw')
            ax.set_xlabel('$\mu$')
            ax.legend()
            fig = plt.gcf()
            utils.savefig(outdir / 'counts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
            plt.close(fig)
            """
            ax = plt.gca()
            xi = xi[0][:10, ::1]
            s, mu = xi.sepavg(axis=0), xi.sepavg(axis=1)
            print(xi.corr.shape)
            ax.pcolormesh(s, mu, xi.corr.T)
            fig = plt.gcf()
            utils.savefig(outdir / 'counts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
            plt.close(fig)
            """

        if 'comparison_cutsky_ez' in todo:

            fma = get_ez_file_manager()
            tfm = get_box_ez_file_manager()
            imock = list(range(1, 1001))[:50]
            for options in fm.select(id='power_ez_y1', region='GCcomb', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'][1:2], cut=None).iter_options(exclude=['imock', 'version'], intersection=False):

                tracer, zrange, region = options['tracer'], options['zrange'], options['region']
                zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                #pk_box = [tfm.get(id='power_box_ez_y1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, version='v1', ignore=True).load() for iimock in imock for los in ['x', 'y', 'z']]
                pk_v1 = [fma.get(id='power_ez_y1', **options, version='v1', imock=iimock).load() for iimock in imock]
                pk_ric = [fma.get(id='power_ez_y1', **options, version='v1ric', imock=iimock).load() for iimock in imock]
                pk_noric = [fma.get(id='power_ez_y1', **options, version='v1noric', imock=iimock).load() for iimock in imock]

                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                fig = plot_comparison_power_spectrum_multipoles([pk_v1, pk_ric, pk_noric], select=(0.0, 0.2, 0.005), labels=['v1', 'RIC', 'no RIC'])
                utils.savefig(outdir / 'comparison_power_ez_cutsky_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)
            
        if 'comparison_ez_ric' in todo:

            fma = get_ez_file_manager()
            imock = list(range(1, 1001))[:25]
            for options in fm.select(id='power_ez_y1', region='GCcomb', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], cut=('theta', 0.05)).iter_options(exclude=['imock', 'version'], intersection=False):

                pk = [fma.get(id='power_ez_y1', **options, version='v1', imock=iimock).load() for iimock in imock]
                pk_ric = [fma.get(id='power_ez_y1', **options, version='v1ric', imock=iimock).load() for iimock in imock]
                pk_noric = [fma.get(id='power_ez_y1', **options, version='v1noric', imock=iimock).load() for iimock in imock]

                tracer, region, zrange = options['tracer'], options['region'], options['zrange']
                fig = plot_comparison_power_spectrum_multipoles([pk, pk_ric, pk_noric], select=(0.0, 0.2, 0.005), labels=['v1', 'RIC', 'no RIC'])
                utils.savefig(outdir / 'comparison_power_ez_ric_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, region=region, zrange=zrange), fig=fig)
                plt.close(fig)

        if 'data_rotated' in todo:
            
            def plot_power_spectrum(data, linestyles=None, labels=None, klim=None, covmatrix=None, fn=None):

                if linestyles is None:
                    linestyles = ('-', '--', ':', '-.')
                tracer, zrange = data[0].options['tracer'], data[0].options['zrange']
                data = [dd.load() for dd in data]
                if klim is not None:
                    data = [dd.select(klim) for dd in data]
                kout = data[0].k
                ells = data[0].ells

                std = None
                if covmatrix is not None:
                    covmatrix = covmatrix.load()
                    covmatrix = covmatrix.xmatch((data[0].kedges[:-1] + data[0].kedges[1:]) / 2., select_projs=True)
                    if klim is not None:
                        covmatrix = covmatrix.select(xlim=klim[:2])
                    std = [covmatrix.std(projs=ell) for ell in ells]

                #fig, lax = plt.subplots(1, len(ells), figsize=(3 * len(ells), 4), sharey=True, sharex=True, squeeze=False)
                fig, lax = plt.subplots(1, 1, figsize=(6, 4), sharey=True, sharex=True, squeeze=False)
                lax = lax.ravel()

                for ill, ell in enumerate(ells):
                    color = 'C{}'.format(ill)
                    #ax = lax[ill]
                    ax = lax[0]
                    
                    if std is not None:
                        ax.errorbar(kout, kout * data[0](ell=ell, complex=False), kout * std[ill], color=color, ls='')
                    for dd, linestyle, label in zip(data, linestyles, labels):
                        ax.plot(dd.k, dd.k * dd(ell=ell, complex=False), color=color, linestyle=linestyle, label=label if ill == 0 else None)
                   
                    ax.set_title('{} in ${:.1f} < z < {:.1f}$'.format(tracer, *zrange))
                    ax.set_xlim(klim[:2])
                    ax.grid(True)
                    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

                lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
                lax[0].legend(ncol=2, framealpha=0., loc=1)
                fig.align_ylabels()
                if fn is not None:
                    utils.savefig(fn, fig=fig)
                    plt.close(fig)

            for ft in fm.select(id='power_y1', weighting='default_FKP', version='v1.5', region='GCcomb', cut=('theta', 0.05)).iter(intersection=False):
                options = dict(ft.options)
                fig = plt.figure()
                fdata = fm.get(id='power_y1', **options, check_exists=True)
                fcovariance = fm.get(id='covariance_y1', **options, observable='power', source='thecov', ignore=['version'], check_exists=True, raise_error=False)
                fdata_corrected = fm.get(id='power_corrected_y1', **options, check_exists=True)
                fdata_rotated = fm.get(id='power_rotated_y1', **options, check_exists=True)
                fdata_rotated_corrected = fm.get(id='power_rotated_corrected_y1', **options, check_exists=True)
                fn = outdir / 'power_multipoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fdata.foptions)
                plot_power_spectrum([fdata, fdata_corrected, fdata_rotated, fdata_rotated_corrected], labels=['raw', 'corrected', 'rotated', 'rotated + corrected'], covmatrix=fcovariance, klim=(0., 0.2, 0.005), fn=fn)