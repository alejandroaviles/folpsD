"""Produce the plots for the KP4 key paper."""

import os

from pathlib import Path
from matplotlib import pyplot as plt

from desipipe import setup_logging
from desi_y1_plotting import KP4StylePaper, utils
from desi_y1_plotting.kp4 import load_chain
from desi_y1_files import get_data_file_manager, get_abacus_file_manager, get_ez_file_manager


if __name__ == '__main__':

    setup_logging()
    #todo = ['mean_abacus']
    #todo = ['abacus']
    #todo = ['mean_ez']
    #todo = ['data']
    #todo = ['unblinding']
    todo = ['lrg+elg']
    
    version = 'v1.2'
    fm = get_data_file_manager(conf='unblinded') + get_abacus_file_manager() + get_ez_file_manager()
    
    # Output directory
    #outdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/KP4plots/KeyPaper/'
    for fc in fm.select(id='power_y1', version=version).iter(intersection=False):
        outdir = Path(os.path.dirname(os.path.dirname(os.path.dirname(fc)))) / 'plots_kp4'
        break

    with KP4StylePaper() as style:

        if 'mean_abacus' in todo:
            fms = fm.select(id='chains_bao_recon_mean_abacus_y1', weighting='default_FKP', version='v3', fa='altmtl', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], region=['GCcomb'], template='bao-qisoqap', broadband=['pcs', 'pcs2'], cut=None)
            for options in fms.iter_options(intersection=False, exclude=['tracer', 'zrange', 'ichain', 'observable', 'broadband', 'smoothing_radius', 'sigmas', 'lim']):
                fi = []
                for options in fms.select(**options).iter_options(intersection=False, exclude=['ichain', 'observable', 'broadband', 'lim']):  # iterate on tracer, zrange
                    if options['zrange'] == (0.8, 1.6): continue
                    fi.append(tuple(fms.select(**options).iter(intersection=False, exclude=['ichain']))) # iterate on observable, broadband
                    labels = ['{} {}'.format(ff.options['observable'][0], ff.options['broadband'][0]) for ff in fi[-1]]
                    for ff in fi[-1][0]: foptions = ff.foptions
                fig = style.plot_bao_diagram(fi, labels=labels, apmode='qisoqap')
                utils.savefig(outdir / 'bao_recon_abacus_{fa}_{region}_precscale{precscale:d}{cut}.png'.format(**foptions), fig=fig)
                plt.close(fig)
            """
            fms = fm.select(id='chains_bao_recon_mean_abacus_y1', weighting='default_FKP', version='v2', fa='altmtl', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], region=['GCcomb'], template='bao-qisoqap', cut=None)
            for options in fms.iter_options(intersection=False, exclude=['tracer', 'zrange', 'ichain', 'broadband', 'smoothing_radius', 'sigmas']):
                fi = []
                for options in fms.select(**options).iter_options(intersection=False, exclude=['ichain', 'broadband']):  # iterate on tracer, zrange
                    if options['zrange'] == (0.8, 1.6): continue
                    fi.append(tuple(fms.select(**options).iter(intersection=False, exclude=['ichain']))) # iterate on broadband
                    labels = [ff.options['broadband'][0] for ff in fi[-1]]
                    for ff in fi[-1][0]: foptions = ff.foptions
                fig = style.plot_bao_diagram(fi, labels=labels, apmode='qisoqap')
                utils.savefig(outdir / 'bao_recon_abacus_{observable}_{fa}_{region}_precscale{precscale:d}{cut}.png'.format(**foptions), fig=fig)
                plt.close(fig)
            """
        if 'data' in todo:
            fms = fm.select(id='chains_bao_recon_y1', weighting='default_FKP', version='v1', region=['GCcomb'], template='bao-qisoqap', broadband=['pcs', 'pcs2'], cut=None)
            for options in fms.iter_options(intersection=False, exclude=['tracer', 'zrange', 'ichain', 'observable', 'broadband', 'smoothing_radius', 'sigmas', 'lim']):
                fi = []
                for options in fms.select(**options).iter_options(intersection=False, exclude=['ichain', 'observable', 'broadband', 'lim']):  # iterate on tracer, zrange
                    fi.append(tuple(fms.select(**options).iter(intersection=False, exclude=['ichain']))) # iterate on observable, broadband
                    labels = ['{} {}'.format(ff.options['observable'][0], ff.options['broadband'][0]) for ff in fi[-1]]
                    for ff in fi[-1][0]: foptions = ff.foptions
                fig = style.plot_bao_diagram(fi, labels=labels, apmode='qisoqap')
                utils.savefig(outdir / 'bao_recon_data_{region}{cut}.png'.format(**foptions), fig=fig)
                plt.close(fig)
            
        if 'abacus' in todo:
            fms = fm.select(id='profiles_bao_recon_abacus_y1', weighting='default_FKP', version='v3', fa='altmtl', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], region=['GCcomb'], template='bao-qisoqap', cut=None)
            for fi in fms.iter(intersection=False, exclude=['imock']):
                fi = list(fi)
                fig = style.plot_scatter(fi, params=['qiso', 'qap'], rescale=True)
                utils.savefig(outdir / 'bao_recon_abacus_{observable}_{theory}_{template}_{broadband}_{fa}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                plt.close(fig)

        if 'mean_ez' in todo:
            fms = fm.select(id='chains_bao_recon_mean_ez_y1', weighting='default_FKP', version='v1', fa='ffa', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], region=['GCcomb'], template='bao-qisoqap', cut=None)
            for options in fms.iter_options(intersection=False, exclude=['tracer', 'zrange', 'ichain', 'observable', 'broadband', 'smoothing_radius', 'sigmas', 'lim']):
                fi = []
                for options in fms.select(**options).iter_options(intersection=False, exclude=['ichain', 'observable', 'broadband', 'lim']):  # iterate on tracer, zrange
                    fi.append(tuple(fms.select(**options).iter(intersection=False, exclude=['ichain']))) # iterate on observable, broadband
                    labels = ['{} {}'.format(ff.options['observable'][0], ff.options['broadband'][0]) for ff in fi[-1]]
                    for ff in fi[-1][0]: foptions = ff.foptions
                fig = style.plot_bao_diagram(fi, labels=labels, apmode='qisoqap')
                utils.savefig(outdir / 'bao_recon_ez_{fa}_{region}_precscale{precscale:d}{cut}.png'.format(**foptions), fig=fig)
                plt.close(fig)

        from desi_y1_files.file_manager import get_bao_baseline_fit_setup
        """
        def get_list_options(tracer, zrange):
            from desi_y1_files.file_manager import get_bao_baseline_fit_setup, list_zrange
            list_options = {}
            ref_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=False)
            list_options['baseline'] = ('bao_recon', {**ref_options, 'region': 'GCcomb'})
            #list_options['$d\\beta \\in [0.25, 1.75]$'] = ('bao_recon', {**ref_options, 'region': 'GCcomb', 'dbeta': (0.25, 1.75)})
            ref_iso_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=True)
            list_options['1D fit'] = ('bao_recon', {**ref_iso_options, 'region': 'GCcomb'})
            list_options['now-qiso'] = ('bao_recon', {**ref_iso_options, 'region': 'GCcomb', 'template': 'bao-now-qiso'})
            #list_options['pre-recon'] = ('bao', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=False, iso=False), 'region': 'GCcomb'})
            #list_options['power\nspectrum'] = ('bao_recon', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='power', recon=True, iso=False), 'region': 'GCcomb'})
            ##list_options['power\nspectrum'] = ('bao_recon', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='power', recon=True, iso=True), 'region': 'GCcomb'})
            list_options['NGC'] = ('bao_recon', {**ref_options, 'region': 'NGC'})
            list_options['polynomial\nbroadband'] = ('bao_recon', {**ref_options, 'broadband': 'power3', 'region': 'GCcomb'})
            list_options['flat prior\non $\Sigma_{s}, \Sigma_{\parallel}, \Sigma_{\perp}$'] = ('bao_recon', {**ref_options, 'sigmas': {'sigmas': None, 'sigmapar': None, 'sigmaper': None}, 'region': 'GCcomb'})
            #list_options = {name: list_options[name] for name in ['1D fit']}
            #toret += [options for ft, options in list_options.values()]
            for options in list_options.values(): options[1]['version'] = version
            return list_options
        """
        def get_list_options(tracer, zrange):
            from desi_y1_files.file_manager import get_bao_baseline_fit_setup, list_zrange
            list_options = {}
            ref_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=None)
            list_options['baseline'] = ('bao_recon', {**ref_options, 'region': 'GCcomb'})
            #list_options['$d\\beta \\in [0.25, 1.75]$'] = ('bao_recon', {**ref_options, 'region': 'GCcomb', 'dbeta': (0.25, 1.75)})
            ref_iso_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=True)
            list_options['1D fit'] = ('bao_recon', {**ref_iso_options, 'region': 'GCcomb'})
            list_options['now-qiso'] = ('bao_recon', {**ref_iso_options, 'region': 'GCcomb', 'template': 'bao-now-qiso'})
            list_options['pre-recon'] = ('bao', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=False, iso=None), 'region': 'GCcomb'})
            #list_options['power\nspectrum'] = ('bao_recon', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='power', recon=True, iso=False), 'region': 'GCcomb'})
            ##list_options['power\nspectrum'] = ('bao_recon', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='power', recon=True, iso=True), 'region': 'GCcomb'})
            list_options['NGC'] = ('bao_recon', {**ref_options, 'region': 'NGC'})
            list_options['polynomial\nbroadband'] = ('bao_recon', {**ref_options, 'broadband': 'power3', 'region': 'GCcomb'})
            list_options['fixed\nbroadband'] = ('bao_recon', {**ref_options, 'broadband': 'fixed', 'region': 'GCcomb'})
            list_options['flat prior\non $\Sigma_{s}, \Sigma_{\parallel}, \Sigma_{\perp}$'] = ('bao_recon', {**ref_options, 'sigmas': {'sigmas': None, 'sigmapar': None, 'sigmaper': None}, 'region': 'GCcomb'})
            #list_options = {name: list_options[name] for name in ['1D fit']}
            #toret += [options for ft, options in list_options.values()]
            for options in list_options.values(): options[1]['version'] = version
            return list_options

        if 'unblinding' in todo:
            version = 'v1.2'
            list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8)], 'LRG+ELG_LOPnotqso': [(0.8, 1.1)], 'ELG_LOPnotqso': [(1.1, 1.6)], 'QSO': [(0.8, 2.1)]}

            list_data = []
            for tracer, zranges in list_zrange.items():
                for zrange in zranges:
                    data, labels = [], []
                    for label, (fit_type, options) in get_list_options(tracer, zrange).items():
                        if '1D' in label or 'now' in label: continue
                        #data.append(list(fm.select(id='chains_{}_y1'.format(fit_type), version=version, **options)))
                        fi = fm.get(id='profiles_{}_y1'.format(fit_type), **options, ignore=True)
                        if not fi.exists(): fi = None
                        data.append(fi)
                        labels.append(label)
                    list_data.append(data)
            fig = style.plot_whiskers(list_data, param='qiso', qlim=None, labels=labels)
            fig.set_size_inches(15, 6)
            utils.savefig(outdir / 'bao_whiskers_qiso.png')

            list_data = []
            for tracer, zranges in list_zrange.items():
                for zrange in zranges:
                    if 'BGS' in tracer or 'QSO' in tracer: continue
                    data, labels = [], []
                    for label, (fit_type, options) in get_list_options(tracer, zrange).items():
                        if '1D' in label or 'qiso' in label: continue
                        #data.append(list(fm.select(id='chains_{}_y1'.format(fit_type), version=version, **options)))
                        fi = fm.get(id='profiles_{}_y1'.format(fit_type), **options, ignore=True)
                        if not fi.exists(): fi = None
                        data.append(fi)
                        labels.append(label)
                    list_data.append(data)
            fig = style.plot_whiskers(list_data, param='qap', qlim=None, labels=labels)
            fig.set_size_inches(15, 6)
            utils.savefig(outdir / 'bao_whiskers_qap.png')

            list_data = []
            for tracer, zranges in list_zrange.items():
                for zrange in zranges:
                    data = []
                    list_options = get_list_options(tracer, zrange)
                    data.append(fm.get(id='profiles_bao_recon_y1', **list_options['1D fit'][1], ignore=True))
                    data.append(fm.get(id='profiles_bao_recon_y1', **list_options['now-qiso'][1], ignore=True))
                    list_data.append(data)
            fn = outdir / 'bao_profile.png'
            style.plot_profiles(list_data, param='qiso', fn=fn)

            chains, labels = [], []
            from desilike.samples import Chain, plotting
            for tracer, zranges in list_zrange.items():
                for zrange in zranges:
                    options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True)
                    chain = load_chain(fm.select(id='chains_bao_recon_y1', version=version, **options, ignore=True))[::10]
                    if 'qap' not in chain: continue
                    chains.append(chain)
                    labels.append('{} {zrange[0]:.1f} - {zrange[1]:.1f}'.format(tracer[:3], zrange=zrange))
            fn = outdir / 'bao_contours_qparqper.png'
            plotting.plot_triangle(chains, params=['qpar', 'qper'], labels=labels, filled=True, fn=fn)
            """
            chains, labels = [], []
            from desilike.samples import Chain, plotting
            for tracer, zranges in list_zrange.items():
                for zrange in zranges:
                    options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=False)
                    chain = load_chain(fm.select(id='chains_bao_recon_y1', version=version, **options, ignore=True))[::10]
                    chains.append(chain)
                    labels.append('{} {zrange[0]:.1f} - {zrange[1]:.1f}'.format(tracer[:3], zrange=zrange))
            fn = outdir / 'bao_contours_qparqper_all.png'
            plotting.plot_triangle(chains, params=['qpar', 'qper'], labels=labels, filled=False, fn=fn)
            
            chains, labels = [], []
            from desilike.samples import Chain, plotting
            for tracer, zranges in list_zrange.items():
                for zrange in zranges:
                    options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True)
                    chain = load_chain(fm.select(id='chains_bao_recon_y1', version=version, **options, ignore=True))[::10]
                    if 'qap' not in chain: continue
                    chains.append(chain)
                    labels.append('{} {zrange[0]:.1f} - {zrange[1]:.1f}'.format(tracer[:3], zrange=zrange))
            fn = outdir / 'bao_contours_qisoqap.png'
            plotting.plot_triangle(chains, params=['qiso', 'qap'], labels=labels, filled=True, fn=fn)
            """

        if 'lrg+elg' in todo:
            import numpy as np
            from desilike.samples import Profiles
            imock = list(range(25))
            dirname = Path('/global/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe')
            profiles = {}
            profiles['lrg+elg'] = [Profiles.load(dirname / 'v3_1_lrg+elg/altmtl/fits_2pt/mock{:d}/fits_correlation_dampedbao_bao-qisoqap_pcs2/recon_IFFT_recsym_sm15/profiles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150_2-50-150.npy'.format(imock)) for imock in imock]
            profiles['lrg'] = [Profiles.load(dirname / 'v3_1/altmtl/fits_2pt/mock{:d}/fits_correlation_dampedbao_bao-qisoqap_pcs2/recon_IFFT_recsym_sm15/profiles_LRG_GCcomb_z0.8-1.1_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150_2-50-150.npy'.format(imock)) for imock in imock]
            profiles_data = {}
            dirname = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/desipipe/fits_2pt/fits_correlation_dampedbao_bao-qisoqap_pcs2/recon_IFFT_recsym_sm15')
            profiles_data['lrg+elg'] = Profiles.load(dirname / 'profiles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150_2-50-150.npy')
            profiles_data['lrg'] = Profiles.load(dirname / 'profiles_LRG_GCcomb_z0.8-1.1_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150_2-50-150.npy')
            style.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman'})
            for param in ['qiso', 'qap']:
                param = profiles['lrg'][0].bestfit.params()[param]
                values = {name: [p.bestfit.choice(params=[param], return_type='nparray') for p in profiles[name]] for name in profiles}
                errors = {name: [np.abs(p.interval[param]).mean() for p in profiles[name]] for name in profiles}
                print('std({})'.format(param), ' '.join(['{} = {:.4f}'.format(name, np.std(value)) for name, value in values.items()]))
                print('mean(sigma({}))'.format(param), ' '.join(['{} = {:.4f}'.format(name, np.mean(value)) for name, value in errors.items()]))
                values_data = {name: profiles_data[name].bestfit.choice(params=[param], return_type='nparray') for name in profiles_data}
                errors_data = {name: np.abs(profiles_data[name].interval[param]).mean() for name in profiles_data}
                fig, lax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'wspace': 0.3})
                ax = lax[0]
                ax.scatter(values['lrg'], values['lrg+elg'], color='C0', label='Abacus altmtl')
                ax.scatter(values_data['lrg'], values_data['lrg+elg'], color='C1', label='data v1.2')
                ax.legend()
                label = param.latex(inline=False)
                ax.set_xlabel('${}$ for LRG'.format(label))
                ax.set_ylabel('${}$ for LRG + ELG'.format(label))
                xlim = ax.get_xlim()
                ax.plot(xlim, xlim, color='k', linestyle='--')
                ax.set_xlim(xlim)
                ax.set_ylim(xlim)
                ax = lax[1]
                ax.scatter(errors['lrg'], errors['lrg+elg'], color='C0', label='Abacus altmtl')
                ax.scatter(errors_data['lrg'], errors_data['lrg+elg'], color='C1', label='data v1.2')
                ax.legend()
                label = param.latex(inline=False)
                ax.set_xlabel('$\sigma({})$ for LRG'.format(label))
                ax.set_ylabel('$\sigma({})$ for LRG + ELG'.format(label))
                xlim = ax.get_xlim()
                ax.plot(xlim, xlim, color='k', linestyle='--')
                ax.set_xlim(xlim)
                ax.set_ylim(xlim)
                fig.savefig('abacus_mocks_lrg+elg_{}.pdf'.format(param))
                plt.tight_layout()
                plt.close(fig)
                
                
                