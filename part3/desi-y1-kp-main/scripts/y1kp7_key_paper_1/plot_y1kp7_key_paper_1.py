"""
Produce the plots for the KP7 key paper #1.

We may want to organize the script in sections corresponding to the paper:
- expansion_history
- dark_energy
- hubble_tension
- neutrinos

There is an example of how to read chains (in getdist format) in 'examples'.
Look at ``y1_bao_cosmo_tools.py`` to load cobaya samples (:func:`load_cobaya_samples`).
Please add your plotting scripts, and if you feel like it you can add your name as comments.
"""

import os
import sys
import glob

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo.fiducial import DESI
from desipipe import setup_logging
sys.path.insert(1, '../')
from y1_bao_cosmo_tools import load_cobaya_samples
from desi_y1_plotting import KP7StylePaper, utils


if __name__ == '__main__':

    from pathlib import Path

    setup_logging()

    theory = 'camb'
    #todo = ['examples']
    todo = ['H0']
    todo = ['whisker_H0']
    todo = ['summary_H0']
    todo = ['comparisons']
    todo = ['dark_energy']
    todo = ['data']
    todo = ['investigate_H0']
    todo = ['tests']
    todo = ['press', 'press2']
    todo = ['dark_energy']
    todo = ['expansion_history']
    todo = ['bao_versions']
    todo = ['press4']
    todo = ['bao_likelihoods']
    todo = ['bao_versions_v15']
    #todo = ['fsigma8']
    outdir = Path('./plots/')
    outdir.mkdir(exist_ok=True)
    from getdist import plots

    with KP7StylePaper() as style:

        if 'tests' in todo:
            if 0:
                # DESY5
                from desilike.samples import Chain, ParameterCollection
                ref = {}
                dirname = '/global/cfs/cdirs/desi/science/cpe/y1kp7/checks/DESY5SN'
                values = np.loadtxt(os.path.join(dirname, 'flcdm_SN_emcee.txt'), unpack=True)[:1]
                params = ParameterCollection({'omegam': {'prior': {'limits': [0.1, 0.9]}}})
                ref['base'] = Chain(values, params=params)
                values = np.loadtxt(os.path.join(dirname, 'fwcdm_SN_emcee.txt'), unpack=True)[:2]
                params['w'] = {'prior': {'limits': [-10., 5.]}}
                #ref['base_w'] = Chain(values, params=params)
                values = np.loadtxt(os.path.join(dirname, 'fw0wacdm_SN_emcee.txt'), unpack=True)[:3]
                params['wa'] = {'prior': {'limits': [-20., 10.]}}
                chain = Chain(values, params=params)
                chain['w'].param.update(prior={'limits': [-3., 1.]})
                chain['wa'].param.update(prior={'limits': [-3., 2.]})
                chain = chain[(chain['w'] > -3) & (chain['w'] < 1) & (chain['wa'] > -3) & (chain['wa'] < 2)]
                ref['base_w_wa'] = chain
                ref = {name: chain.remove_burnin(0.3)[::10] for name, chain in ref.items()}
                for model, ref in ref.items():
                    names = ref.names()
                    ref = ref.to_getdist(label='ref')
                    test = load_cobaya_samples(theory=theory, model=model, dataset=['desy5'], label='test')
                    g = plots.get_subplot_plotter()
                    # Modify the plot_2d call to include the colors argument
                    g.settings = style.settings
                    g.triangle_plot([test, ref], names)
                    fn = outdir / 'check_desy5_{}.png'.format(model)
                    plt.tight_layout()
                    plt.savefig(fn, bbox_inches='tight', dpi=360)
    
            if 0:
                model = 'base_w_wa'
                for dataset in ['pantheonplus', 'union3', 'desy5']:
                    desilike = load_cobaya_samples(theory=theory, model=model, dataset=[dataset], label='desilike')
                    cobaya = load_cobaya_samples(theory=theory, model=model, dataset=[dataset], run='test0', label='cobaya')
                    g = plots.get_subplot_plotter()
                    # Modify the plot_2d call to include the colors argument
                    g.settings = style.settings
                    #g.triangle_plot([desilike, cobaya], ['omegam', 'w', 'wa'])
                    #fn = outdir / 'check_{}_{}.png'.format(dataset, model)
                    desilike.addDerived(desilike['w'] + desilike['wa'], name='wwa', label='$w_0 + w_a$', range=[-3, 0])
                    cobaya.addDerived(cobaya['w'] + cobaya['wa'], name='wwa', label='$w_0 + w_a$', range=[-3, 0])
                    g.triangle_plot([desilike, cobaya], ['w', 'wa', 'wwa'])
                    fn = outdir / 'check_{}_{}.png'.format(dataset, model)
                    plt.tight_layout()
                    plt.savefig(fn, bbox_inches='tight', dpi=360)
                
            if 0:
                #tracer = load_cobaya_samples(theory=theory, model='base', dataset=['desi-bao-bgs', 'desi-bao-lrg', 'desi-bao-elg', 'desi-bao-qso', 'schoneberg2024-bbn'])
                #split = load_cobaya_samples(theory=theory, model='base', dataset=['desi-bao-bgs', 'desi-bao-lrg-z0', 'desi-bao-lrg-z1', 'desi-bao-lrg-z2', 'desi-bao-elg', 'desi-bao-qso', 'schoneberg2024-bbn'])
                desi = load_cobaya_samples(theory=theory, model='base', dataset=['desi-bao-bgs', 'desi-bao-lrg', 'desi-bao-elg', 'desi-bao-qso', 'desi-eboss-bao-lya', 'schoneberg2024-bbn'])
                desi_best = load_cobaya_samples(theory=theory, model='base', dataset=['desi-sdss-bao-best', 'schoneberg2024-bbn'])
                print(1. - desi_best.std('H0') / desi.std('H0'))
                g = plots.get_single_plotter(width_inch=7, ratio=1, scaling=True)
                # Modify the plot_2d call to include the colors argument
                g.settings = style.settings
                g.plot_2d([desi, desi_best], 'H0', 'omegam', filled=True)
                legend = g.add_legend(['DESI + BBN', 'DESI + SDSS + BBN'], legend_loc='upper right')
                fn = outdir / 'H0_omegam.pdf'
                plt.tight_layout()
                plt.savefig(fn, bbox_inches='tight', dpi=360)

            if 0:
                desi = load_cobaya_samples(theory=theory, model='base', dataset=['mock-fiducial-desi-bao-bgs', 'mock-fiducial-desi-bao-lrg', 'mock-fiducial-desi-bao-elg', 'mock-fiducial-desi-bao-qso', 'schoneberg2024-bbn'])
                desi_best = load_cobaya_samples(theory=theory, model='base', dataset=['sdss-bao-dr7-mgs', 'sdss-bao-dr12-lrg', 'mock-fiducial-desi-bao-lrg-z1', 'mock-fiducial-desi-bao-lrg-z2', 'mock-fiducial-desi-bao-elg', 'mock-fiducial-desi-bao-qso', 'schoneberg2024-bbn'])
                print(1. - desi_best.std('H0') / desi.std('H0'))
                g = plots.get_single_plotter(width_inch=7, ratio=1, scaling=True)
                # Modify the plot_2d call to include the colors argument
                g.settings = style.settings
                g.plot_2d([desi, desi_best], 'H0', 'omegam', filled=True)
                legend = g.add_legend(['DESI + BBN', 'DESI + SDSS + BBN'], legend_loc='upper right')
                fn = outdir / 'H0_omegam_mock_fiducial.pdf'
                plt.tight_layout()
                plt.savefig(fn, bbox_inches='tight', dpi=360)

            if 0:
                desi = load_cobaya_samples(theory=theory, model='base', dataset=['desi-bao-all', 'schoneberg2024-bbn'])
                desi_best = load_cobaya_samples(theory=theory, model='base', dataset=['desi-sdss-bao-best', 'schoneberg2024-bbn'])
                desi_best_lowz = load_cobaya_samples(theory=theory, model='base', dataset=['sdss-bao-dr7-mgs', 'sdss-bao-dr12-lrg', 'desi-bao-lrg-z1', 'desi-bao-lrg-z2', 'desi-bao-elg', 'desi-bao-qso', 'desi-bao-lya', 'schoneberg2024-bbn'])
                print(1. - desi_best_lowz.std('H0') / desi.std('H0'), 1. - desi_best.std('H0') / desi_best_lowz.std('H0'), 1. - desi_best.std('H0') / desi.std('H0'))
                g = plots.get_single_plotter(width_inch=7, ratio=1, scaling=True)
                # Modify the plot_2d call to include the colors argument
                g.settings = style.settings
                g.plot_2d([desi, desi_best, desi_best_lowz], 'H0', 'omegam', filled=True)
                legend = g.add_legend(['DESI + BBN', 'DESI + SDSS + BBN', 'DESI + SDSS lowZ + BBN'], legend_loc='upper right')
                fn = outdir / 'H0_omegam.pdf'
                plt.tight_layout()
                plt.savefig(fn, bbox_inches='tight', dpi=360)

            if 0:
                desi_planck = load_cobaya_samples(theory=theory, model='base', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], label='DESI + Planck T\&P')
                desi_planck_lensing = load_cobaya_samples(theory=theory, model='base', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'], label='DESI + Planck')
                print(desi_planck_lensing['tau'].min(), desi_planck_lensing['tau'].max())
                g = plots.get_subplot_plotter()
                # Modify the plot_2d call to include the colors argument
                g.settings = style.settings
                g.triangle_plot([desi_planck, desi_planck_lensing], ['H0', 'ombh2', 'omch2', 'logA', 'tau'])
                fn = outdir / 'importance_lensing.pdf'
                plt.tight_layout()
                plt.savefig(fn, bbox_inches='tight', dpi=360)

            if 1:
                desi_planck = load_cobaya_samples(theory=theory, model='base', dataset=['desi-bao-all'], label='DESI + Planck T\&P')
                desi_planck_lensing = load_cobaya_samples(theory=theory, model='base', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'], label='DESI + Planck T\&P')
                g = plots.get_subplot_plotter()
                # Modify the plot_2d call to include the colors argument
                g.settings = style.settings
                g.triangle_plot([desi_planck, desi_planck_lensing], ['H0', 'ombh2', 'omch2', 'logA', 'tau'])
                fn = outdir / 'test.pdf'
                plt.tight_layout()
                plt.savefig(fn, bbox_inches='tight', dpi=360)
                
            if 0:
                desi = load_cobaya_samples(theory=theory, model='base', dataset=['desi-bao-all', 'schoneberg2024-bbn'])
                desi_best = load_cobaya_samples(theory=theory, model='base', dataset=['desi-sdss-bao-best', 'schoneberg2024-bbn'])
                desi_best_theta = load_cobaya_samples(theory=theory, model='base', dataset=['desi-sdss-bao-best', 'planck2018-thetastar'])
                desi_best_rdrag = load_cobaya_samples(theory=theory, model='base', dataset=['desi-sdss-bao-best', 'planck2018-rdrag'])
                g = plots.get_single_plotter(width_inch=7, ratio=1, scaling=True)
                # Modify the plot_2d call to include the colors argument
                g.settings = style.settings
                g.plot_2d([desi, desi_best, desi_best_theta, desi_best_rdrag], 'H0', 'omegam', filled=True)
                legend = g.add_legend(['DESI + BBN', 'DESI + SDSS + BBN', r'DESI + SDSS + $\theta_{\ast}$', r'DESI + SDSS + $r_{\mathrm{d}}$'], legend_loc='upper right')
                fn = outdir / 'H0_omegam.pdf'
                plt.tight_layout()
                plt.savefig(fn, bbox_inches='tight', dpi=360)
            
            if 0:
                desi = load_cobaya_samples(theory=theory, model='base_w_wa', dataset=['desi-bao-all'])
                desi_fiducial = load_cobaya_samples(theory=theory, model='base_w_wa', dataset=['mock-fiducial-desi-bao-all'])
                g = plots.get_single_plotter(width_inch=7, ratio=1, scaling=True)
                # Modify the plot_2d call to include the colors argument
                g.settings = style.settings
                g.plot_2d([desi_fiducial, desi], 'w', 'wa', filled=True)
                legend = g.add_legend(['DESI fiducial', 'DESI'], legend_loc='upper right')
                ax = g.get_axes()
                ax.set_xlabel(r'$w_0$', fontsize=g.settings.axes_labelsize, color='black')
                ax.set_ylabel(r'$w_a$', fontsize=g.settings.axes_labelsize, color='black')
                ax.tick_params(axis='both', which='major', labelsize=g.settings.axes_fontsize)
                ax.axhline(y=0, color='black', linestyle='--')
                ax.axvline(x=-1, color='black', linestyle='--')
                ax.set_xlim([-1.2, -0.35])
                ax.set_ylim([-2.5, 1.5])
                fn = outdir / 'w0_wa.pdf'
                plt.tight_layout()
                plt.savefig(fn, bbox_inches='tight', dpi=360)
        
        if 'press' in todo:
            from y1_bao_cosmo_tools import load_bao_fisher
            from desi_y1_files.cosmo_tools import predict_bao
            from cosmoprimo.fiducial import Planck2018FullFlatLCDM
            from cosmoprimo.utils import DistanceToRedshift
            desi = DESI(engine='camb')
            cosmo_ref = desi.clone(Omega_m=1.)
            #bestfit = load_cobaya_samples(model='base', dataset=['desi-bao-all'], source='', sampler='iminuit')
            #desi_best = desi.clone(N_eff=3.044, Omega_m=bestfit['omegam']).solve('H0', lambda cosmo: cosmo.rs_drag, target=bestfit['H0rdrag'] / 100, limits=[50, 100])
            #assert np.allclose(desi_best.Omega0_m, bestfit['omegam']) and np.allclose(desi_best.rs_drag, bestfit['H0rdrag'] / 100)
            #bestfit = load_cobaya_samples(model='base_w_wa', dataset=['desi-bao-all'], source='', sampler='iminuit')
            #desi_best_w_wa = desi.clone(N_eff=3.044, w0_fld=bestfit['w'], wa_fld=bestfit['wa'], Omega_m=bestfit['omegam']).solve('H0', lambda cosmo: cosmo.rs_drag, target=bestfit['H0rdrag'] / 100, limits=[50, 100])

            bestfit = load_cobaya_samples(model='base', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            bestfit = {name: bestfit.mean(name) for name in ['omegam', 'H0']}
            desi_best = desi.clone(N_eff=3.044, Omega_m=bestfit['omegam'])

            bestfit = load_cobaya_samples(model='base_w_wa', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            bestfit = {name: bestfit.mean(name) for name in ['omegam', 'H0', 'w', 'wa']}
            print(bestfit)
            desi_best_w_wa = desi.clone(N_eff=3.044, w0_fld=bestfit['w'], wa_fld=bestfit['wa'], Omega_m=bestfit['omegam'])
            
            # load BAO fit data from file
            tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1)), ('Lya', (1.8, 4.2))]

            data = {tracer: load_bao_fisher(*tracer, scale=None, return_type=None) for tracer in tracers}
            #apmode, zlim, qlim, figsize = 'qiso', (0., 2.5), (0.63, 1.25), (8, 4)
            #xyinsets = [(0.04, 0.5), (0.16, 0.05), (0.23, 0.6), (0.35, 0.15), (0.43, 0.67), (0.55, 0.25), (0.82, 0.3)]
            apmode, zlim, qlim, figsize = 'qiso', (0., 2.5), (0.6, 1.3), (8, 4)
            xyinsets = [(0.04, 0.47), (0.16, 0.05), (0.23, 0.56), (0.35, 0.15), (0.43, 0.63), (0.55, 0.25), (0.82, 0.3)]
            fig = style.plot_bao_diagram(data, cosmo_ref=cosmo_ref, apmode=apmode, insets=list(data.values()), xyinsets=xyinsets, zlim=zlim, qlim=qlim, label_bao={'qiso': 'DESI / (No Dark Energy)'}, spow=0, figsize=figsize)
            lax = fig.axes
            for ax in lax[1:]:
                ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
                ax.set_xlabel(None)
                ax.set_ylabel(None)      
            z = np.linspace(*zlim, 1000)[1:]  # to avoid 0
            ref = np.array(predict_bao(z, apmode=apmode, cosmo=cosmo_ref, scale='distance'))
            means = np.array(predict_bao(z, apmode=apmode, cosmo=desi_best, scale='distance')) / ref
            for iax, (ax, mean) in enumerate(zip(lax, means)):
                ax.plot(z, mean, color='k', linestyle='-', linewidth=1., label=r'DESI ($\Lambda\rm CDM$)')
            means = np.array(predict_bao(z, apmode=apmode, cosmo=desi_best_w_wa, scale='distance')) / ref
            for iax, (ax, mean) in enumerate(zip(lax, means)):
                ax.plot(z, mean, color='k', linestyle='--', linewidth=1., label=r'DESI ($w_{0}w_{a}\rm CDM$)')
            #means = np.array(predict_bao(z, apmode=apmode, cosmo=Planck2018FullFlatLCDM(engine='camb'), scale='distance')) / ref
            #for iax, (ax, mean) in enumerate(zip(lax, means)):
            #    ax.plot(z, mean, color='k', linestyle='--', linewidth=0.5, label=r'Planck ($\Lambda\rm CDM$)')
            ax.legend(frameon=False)
            ax.set_xlabel('redshift $z$')
            ax2 = ax.twiny()
            dt = lambda z: desi.time(0.) - desi.time(z)
            t2z = DistanceToRedshift(dt)
            ax2.set_xlim(zlim)
            ticks = np.arange(*dt(zlim), 1)
            tick_labels = ['{:.0f}'.format(t) for t in ticks]
            ax2.set_xticks(t2z(ticks), tick_labels)
            ax2.set_xlabel('lookback time [Gyr]')
            fn = outdir / 'bao_isotropic_diagram.pdf'
            plt.savefig(fn, bbox_inches='tight', dpi=360)

        if 'press2' in todo:
            from y1_bao_cosmo_tools import load_bao_fisher
            from desi_y1_files.cosmo_tools import predict_bao
            from cosmoprimo.fiducial import Planck2018FullFlatLCDM
            from cosmoprimo.utils import DistanceToRedshift
            desi = DESI(engine='camb')
            cosmo_ref = desi.clone(N_eff=3.044)
            #bestfit = load_cobaya_samples(model='base', dataset=['desi-bao-all'], source='', sampler='iminuit')
            #desi_best = desi.clone(N_eff=3.044, Omega_m=bestfit['omegam']).solve('H0', lambda cosmo: cosmo.rs_drag, target=bestfit['H0rdrag'] / 100, limits=[50, 100])
            #assert np.allclose(desi_best.Omega0_m, bestfit['omegam']) and np.allclose(desi_best.rs_drag, bestfit['H0rdrag'] / 100)
            #bestfit = load_cobaya_samples(model='base_w_wa', dataset=['desi-bao-all'], source='', sampler='iminuit')
            #desi_best_w_wa = desi.clone(N_eff=3.044, w0_fld=bestfit['w'], wa_fld=bestfit['wa'], Omega_m=bestfit['omegam']).solve('H0', lambda cosmo: cosmo.rs_drag, target=bestfit['H0rdrag'] / 100, limits=[50, 100])

            bestfit = load_cobaya_samples(model='base', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            desi_best = desi.clone(N_eff=3.044, omega_cdm=bestfit.mean('omch2'), omega_b=bestfit.mean('ombh2'), H0=bestfit.mean('H0'), logA=bestfit.mean('logA'), n_s=bestfit.mean('ns'), tau_reio=bestfit.mean('tau'))
            cosmo_ref = desi_best

            bestfit = load_cobaya_samples(model='base_w_wa', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            desi_best_w_wa = desi.clone(N_eff=3.044, omega_cdm=bestfit.mean('omch2'), omega_b=bestfit.mean('ombh2'), H0=bestfit.mean('H0'), w0_fld=bestfit.mean('w'), wa_fld=bestfit.mean('wa'), logA=bestfit.mean('logA'), n_s=bestfit.mean('ns'), tau_reio=bestfit.mean('tau'))
            #cosmo_ref = desi_best.solve('H0', lambda cosmo: cosmo.rs_drag, target=desi_best_w_wa.rs_drag, limits=[50, 100])
            #print(cosmo_ref.Omega0_m, desi_best.Omega0_m, cosmo_ref.rs_drag, desi_best_w_wa.rs_drag)
            
            # load BAO fit data from file
            tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1)), ('Lya', (1.8, 4.2))]

            apmode, zlim, qlim, figsize = 'qiso', (0., 2.5), (0.7, 1.3), (8, 5)
            data = {tracer: load_bao_fisher(*tracer, scale=None, return_type=None) for tracer in tracers}
            fig = style.plot_bao_diagram(data, cosmo_ref=cosmo_ref, apmode=apmode, insets=list(data.values()), zlim=zlim, qlim=qlim, label_insets='{tracer}', labelsize_insets=9., label_bao={'qiso': r'Distance measurement relative to $\Lambda\rm CDM$'}, spow=0, figsize=figsize)
            lax = fig.axes
            for ax in lax[1:]:
                ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
                ax.set_xlabel(None)
                ax.set_ylabel(None)
            z = np.linspace(*zlim, 1000)[1:]  # to avoid 0
            ref = np.array(predict_bao(z, apmode=apmode, cosmo=cosmo_ref, scale='distance'))
            means = np.array(predict_bao(z, apmode=apmode, cosmo=cosmo_ref, scale='distance')) / ref
            for iax, (ax, mean) in enumerate(zip(lax, means)):
                ax.plot(z, mean, color='k', linestyle='-', linewidth=1.)
            means = np.array(predict_bao(z, apmode=apmode, cosmo=desi_best_w_wa, scale='distance')) / ref
            for iax, (ax, mean) in enumerate(zip(lax, means)):
                ax.plot(z, mean, color='k', linestyle='--', linewidth=1., label=r'DESI $w_{0}w_{a}\rm CDM$')
            #means = np.array(predict_bao(z, apmode=apmode, cosmo=Planck2018FullFlatLCDM(engine='camb'), scale='distance')) / ref
            #for iax, (ax, mean) in enumerate(zip(lax, means)):
            #    ax.plot(z, mean, color='k', linestyle='--', linewidth=0.5, label=r'Planck ($\Lambda\rm CDM$)')
            ax.legend(loc=(0.75, 0.4), frameon=False)
            #ax.set_xlabel('redshift $z$')
            #ax2 = ax.twiny()
            dt = lambda z: cosmo_ref.time(0.) - cosmo_ref.time(z)
            t2z = DistanceToRedshift(dt)
            ticks = np.arange(*dt(zlim), 1)
            print(dt(0.1), dt(0.5), dt(1.), dt(2.), dt(100.))
            tick_labels = ['{:.0f}'.format(t) for t in ticks]
            ax.set_xticks(t2z(ticks), tick_labels)
            ax.set_xlabel('lookback time [billions of years]')
            fn = outdir / 'bao_isotropic_diagram_lcdm.pdf'
            plt.savefig(fn, bbox_inches='tight', dpi=360)

        if 'press3' in todo:
            from y1_bao_cosmo_tools import load_bao_fisher
            from desi_y1_files.cosmo_tools import predict_bao
            from cosmoprimo.fiducial import Planck2018FullFlatLCDM
            from cosmoprimo.utils import DistanceToRedshift
            desi = DESI(engine='camb')
            cosmo_ref = desi.clone(N_eff=3.044)
            #bestfit = load_cobaya_samples(model='base', dataset=['desi-bao-all'], source='', sampler='iminuit')
            #desi_best = desi.clone(N_eff=3.044, Omega_m=bestfit['omegam']).solve('H0', lambda cosmo: cosmo.rs_drag, target=bestfit['H0rdrag'] / 100, limits=[50, 100])
            #assert np.allclose(desi_best.Omega0_m, bestfit['omegam']) and np.allclose(desi_best.rs_drag, bestfit['H0rdrag'] / 100)
            #bestfit = load_cobaya_samples(model='base_w_wa', dataset=['desi-bao-all'], source='', sampler='iminuit')
            #desi_best_w_wa = desi.clone(N_eff=3.044, w0_fld=bestfit['w'], wa_fld=bestfit['wa'], Omega_m=bestfit['omegam']).solve('H0', lambda cosmo: cosmo.rs_drag, target=bestfit['H0rdrag'] / 100, limits=[50, 100])

            bestfit = load_cobaya_samples(model='base', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            desi_best = desi.clone(N_eff=3.044, omega_cdm=bestfit.mean('omch2'), omega_b=bestfit.mean('ombh2'), H0=bestfit.mean('H0'), logA=bestfit.mean('logA'), n_s=bestfit.mean('ns'), tau_reio=bestfit.mean('tau'))
            cosmo_ref = desi_best

            bestfit = load_cobaya_samples(model='base_w_wa', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            desi_best_w_wa = desi.clone(N_eff=3.044, omega_cdm=bestfit.mean('omch2'), omega_b=bestfit.mean('ombh2'), H0=bestfit.mean('H0'), w0_fld=bestfit.mean('w'), wa_fld=bestfit.mean('wa'), logA=bestfit.mean('logA'), n_s=bestfit.mean('ns'), tau_reio=bestfit.mean('tau'))
            #cosmo_ref = desi_best.solve('H0', lambda cosmo: cosmo.rs_drag, target=desi_best_w_wa.rs_drag, limits=[50, 100])
            #print(cosmo_ref.Omega0_m, desi_best.Omega0_m, cosmo_ref.rs_drag, desi_best_w_wa.rs_drag)
            
            # load BAO fit data from file
            tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1)), ('Lya', (1.8, 4.2))]

            apmode, zlim, qlim, figsize = 'qiso', (0., 2.5), (0.8, 1.2), (8, 5)
            data = {tracer: load_bao_fisher(*tracer, scale=None, return_type=None) for tracer in tracers}
            fig = style.plot_bao_diagram(data, cosmo_ref=cosmo_ref, apmode=apmode, insets=list(data.values()), zlim=zlim, qlim=qlim, label_insets='{tracer}', labelsize_insets=9., label_bao={'qiso': r'$(D_\mathrm{V} / r_\mathrm{d}) / (D_\mathrm{V} / r_\mathrm{d})^\mathrm{best}$'}, spow=0, figsize=figsize)
            lax = fig.axes
            for ax in lax[1:]:
                ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
                ax.set_xlabel(None)
                ax.set_ylabel(None)
            z = np.linspace(*zlim, 1000)[1:]  # to avoid 0
            ref = np.array(predict_bao(z, apmode=apmode, cosmo=cosmo_ref, scale='distance'))
            means = np.array(predict_bao(z, apmode=apmode, cosmo=cosmo_ref, scale='distance')) / ref
            for iax, (ax, mean) in enumerate(zip(lax, means)):
                ax.plot(z, mean, color='k', linestyle='-', linewidth=1.)
            means = np.array(predict_bao(z, apmode=apmode, cosmo=desi_best_w_wa, scale='distance')) / ref
            for iax, (ax, mean) in enumerate(zip(lax, means)):
                ax.plot(z, mean, color='k', linestyle='--', linewidth=1., label=r'DESI $w_{0}w_{a}\rm CDM$')
            ax.legend(loc=(0.75, 0.4), frameon=False)       
            ax.set_xlabel('redshift $z$')
            ax2 = ax.twiny()
            dt = lambda z: desi.time(0.) - desi.time(z)
            t2z = DistanceToRedshift(dt)
            ax.set_xlim(zlim)
            ticks = np.arange(*dt(zlim), 1)
            tick_labels = ['{:.0f}'.format(t) for t in ticks]
            ax2.set_xticks(t2z(ticks), tick_labels)
            ax2.set_xlabel('lookback time [Gyr]')
            fn = outdir / 'bao_isotropic_diagram_z.pdf'
            plt.savefig(fn, bbox_inches='tight', dpi=360)
          
        if 'press4' in todo:
            from y1_bao_cosmo_tools import load_bao_fisher
            from desi_y1_files.cosmo_tools import predict_bao
            from cosmoprimo.fiducial import Planck2018FullFlatLCDM
            from cosmoprimo.utils import DistanceToRedshift
            desi = DESI(engine='camb')
            cosmo_ref = desi.clone(N_eff=3.044)
            #bestfit = load_cobaya_samples(model='base', dataset=['desi-bao-all'], source='', sampler='iminuit')
            #desi_best = desi.clone(N_eff=3.044, Omega_m=bestfit['omegam']).solve('H0', lambda cosmo: cosmo.rs_drag, target=bestfit['H0rdrag'] / 100, limits=[50, 100])
            #assert np.allclose(desi_best.Omega0_m, bestfit['omegam']) and np.allclose(desi_best.rs_drag, bestfit['H0rdrag'] / 100)
            #bestfit = load_cobaya_samples(model='base_w_wa', dataset=['desi-bao-all'], source='', sampler='iminuit')
            #desi_best_w_wa = desi.clone(N_eff=3.044, w0_fld=bestfit['w'], wa_fld=bestfit['wa'], Omega_m=bestfit['omegam']).solve('H0', lambda cosmo: cosmo.rs_drag, target=bestfit['H0rdrag'] / 100, limits=[50, 100])

            bestfit = load_cobaya_samples(model='base', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            desi_best = desi.clone(N_eff=3.044, omega_cdm=bestfit.mean('omch2'), omega_b=bestfit.mean('ombh2'), H0=bestfit.mean('H0'), logA=bestfit.mean('logA'), n_s=bestfit.mean('ns'), tau_reio=bestfit.mean('tau'))
            cosmo_ref = desi_best

            bestfit = load_cobaya_samples(model='base_w_wa', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            desi_best_w_wa = desi.clone(N_eff=3.044, omega_cdm=bestfit.mean('omch2'), omega_b=bestfit.mean('ombh2'), H0=bestfit.mean('H0'), w0_fld=bestfit.mean('w'), wa_fld=bestfit.mean('wa'), logA=bestfit.mean('logA'), n_s=bestfit.mean('ns'), tau_reio=bestfit.mean('tau'))
            #print(bestfit.mean('H0'), bestfit.mean('omegam'), bestfit.mean('w'), bestfit.mean('wa'))
            #cosmo_ref = desi_best.solve('H0', lambda cosmo: cosmo.rs_drag, target=desi_best_w_wa.rs_drag, limits=[50, 100])
            #print(cosmo_ref.Omega0_m, desi_best.Omega0_m, cosmo_ref.rs_drag, desi_best_w_wa.rs_drag)
            
            # load BAO fit data from file
            tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1)), ('Lya', (1.8, 4.2))]

            apmode, zlim, qlim, figsize = 'qiso', (0., 2.5), (0.8, 1.2), (8, 5)
            data = {tracer: load_bao_fisher(*tracer, scale=None, return_type=None) for tracer in tracers}
            fig = style.plot_bao_diagram(data, cosmo_ref=cosmo_ref, apmode=apmode, insets=list(data.values()), zlim=zlim, qlim=qlim, label_insets='{tracer}', labelsize_insets=9., label_bao={'qiso': r'mesure de distance relative au $\Lambda\rm CDM$'}, spow=0, figsize=figsize)
            lax = fig.axes
            for ax in lax[1:]:
                ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
                ax.set_xlabel(None)
                ax.set_ylabel(None)
            z = np.linspace(*zlim, 1000)[1:]  # to avoid 0
            ref = np.array(predict_bao(z, apmode=apmode, cosmo=cosmo_ref, scale='distance'))
            means = np.array(predict_bao(z, apmode=apmode, cosmo=cosmo_ref, scale='distance')) / ref
            for iax, (ax, mean) in enumerate(zip(lax, means)):
                ax.plot(z, mean, color='k', linestyle='-', linewidth=1.)
            means = np.array(predict_bao(z, apmode=apmode, cosmo=desi_best_w_wa, scale='distance')) / ref
            for iax, (ax, mean) in enumerate(zip(lax, means)):
                ax.plot(z, mean, color='k', linestyle='--', linewidth=1., label=r'$w_{0}w_{a}\rm CDM$')
            ax.legend(loc=(0.77, 0.4), frameon=False)       
            ax2 = ax.twiny()
            ax2.set_xlabel('décalage spectral $z$')
            dt = lambda z: cosmo_ref.time(0.) - cosmo_ref.time(z)
            t2z = DistanceToRedshift(dt)
            ax2.set_xlim(zlim)
            ticks = np.arange(*dt(zlim), 1)
            tick_labels = ['{:.0f}'.format(t) for t in ticks]
            ax.set_xticks(t2z(ticks), tick_labels)
            ax.set_xlabel("temps dans le passé [milliards d'années]")
            fn = outdir / 'bao_isotropic_diagram_z_french.png'
            plt.savefig(fn, bbox_inches='tight', dpi=360)
            print(desi_best_w_wa.time(0.), desi_best.time(0.), desi.time(0.)) 
        
        if 'fsigma8' in todo:
            desi = DESI(engine='camb')
            cosmo_ref = desi.clone(N_eff=3.044)
            bestfit = load_cobaya_samples(model='base', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            desi_best = desi.clone(N_eff=3.044, omega_cdm=bestfit.mean('omch2'), omega_b=bestfit.mean('ombh2'), H0=bestfit.mean('H0'), logA=bestfit.mean('logA'), n_s=bestfit.mean('ns'))
    
            bestfit = load_cobaya_samples(model='base_w_wa', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
            desi_best_w_wa = desi.clone(N_eff=3.044, omega_cdm=bestfit.mean('omch2'), omega_b=bestfit.mean('ombh2'), H0=bestfit.mean('H0'), w0_fld=bestfit.mean('w'), wa_fld=bestfit.mean('wa'), logA=bestfit.mean('logA'), n_s=bestfit.mean('ns'))

            z = [.15, .38, .51, .7, .85, 1.48]
            values = [.53, .497, .459, .473, .315, .462]
            errors = [.16, .045, .038, .041, .095, .045]
            zz = np.linspace(0., 1.6, 100)
            ax = plt.gca()
            ax.plot(zz, desi_best.get_fourier().sigma8_z(zz, of='theta_cb'), label='best $\Lambda$CDM')
            ax.plot(zz, desi_best_w_wa.get_fourier().sigma8_z(zz, of='theta_cb'), label='best $w_{0}w_{a}$CDM')
            ax.errorbar(z, values, errors, fmt='ko', label='SDSS')
            ax.set_xlabel('z')
            ax.set_ylabel(r'$f\sigma_8(z)$')
            ax.legend()
            ax.set_xlim(zz[0], zz[-1])
            fn = outdir / 'fsigma8.pdf'
            plt.savefig(fn, bbox_inches='tight', dpi=360)

        if 'bao_likelihoods' in todo:
            from y1_bao_cosmo_tools import load_bao_chain, load_bao_fisher
            from desilike.samples import plotting
            tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1))]
            from desi_y1_files.cosmo_tools import convert_bao_fisher
            if 0:
                for tracer, zrange in tracers:
                    chain, profiles = load_bao_chain(tracer, zrange, return_profiles=True)
                    iso = 'qpar' not in chain
                    params = ['qiso'] if iso else ['qiso', 'qap']
                    fisher_rotated = load_bao_fisher(tracer, zrange, with_syst=False, scale=None, return_type=None, apmode='qisoqap')
                    fisher = chain.to_fisher(params=params)
                    fn = outdir / 'chain_bao_qisoqap_gaussian_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.pdf'.format(tracer=tracer, zrange=zrange)
                    g = plots.get_subplot_plotter(width_inch=6)
                    g.settings.num_plot_contours = 3
                    plotting.plot_triangle([chain, fisher, fisher_rotated, profiles], g=g, labels=['samples', 'Fisher', 'Fisher rotated', 'profiles'], legend_loc='upper right', params=params, title_limit=0, fn=fn)

            if 1:
                for tracer, zrange in tracers[1:2]:
                    chain, profiles = load_bao_chain(tracer, zrange, return_profiles=True)
                    params = ['qpar', 'qper']
                    fisher = chain.to_fisher(params=params)
                    fisher2 = chain.to_fisher(params=['qiso', 'qap'])
                    print(fisher2.covariance())
                    exit()
                    fisher_rotated = convert_bao_fisher(fisher2, apmode='qparqper', scale=None, eta=1. / 3.)
                    fn = outdir / 'chain_bao_qparqper_gaussian_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.pdf'.format(tracer=tracer, zrange=zrange)
                    g = plots.get_subplot_plotter(width_inch=6)
                    g.settings.num_plot_contours = 3
                    plotting.plot_triangle([chain, fisher, fisher_rotated], g=g, labels=['samples', 'Fisher', 'Fisher rotated', 'profiles'], legend_loc='upper right', params=params, title_limit=0, fn=fn)

            if 0:
                fisher = load_bao_fisher('Lya', (1.8, 4.2), apmode='qparqper', with_syst=True, scale=None, return_type=None)
                for tracer, zrange in tracers:
                    chain, profiles = load_bao_chain(tracer, zrange, return_profiles=True)
                    chain2 = chain[chain.shape[0]//2:]
                    iso = 'qpar' not in chain
                    apmode = 'qiso' if iso else 'qparqper'
                    fisher = load_bao_fisher(tracer, zrange, apmode=apmode, with_syst=False, scale=None, return_type=None)
                    fisher_syst = load_bao_fisher(tracer, zrange, apmode=apmode, scale=None, return_type=None)
                    params = ['qiso'] if iso else ['qpar', 'qper']
                    print({param: fisher.std(param) for param in params})
                    print({param: fisher_syst.std(param) / fisher.std(param) for param in params})
                    #print([chain2.std(param) / chain.std(param) for param in params])
                    fisher_rotated = load_bao_fisher(tracer, zrange, with_syst=False, scale=None, return_type=None, apmode='qisoqap')
                    fisher_rotated_syst = load_bao_fisher(tracer, zrange, with_syst=True, scale=None, return_type=None, apmode='qisoqap')
                    print({param.name: fisher_rotated.std(param) for param in fisher_rotated.params()})
                    print({param.name: fisher_rotated_syst.std(param) / fisher_rotated.std(param) for param in fisher_rotated.params()})
                    fn = outdir / 'chain_bao_gaussian_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, zrange=zrange)
                    g = plots.get_subplot_plotter()
                    g.settings = style.settings
                    g.settings.num_plot_contours = 3
                    #plotting.plot_triangle([chain, chain2, fisher, fisher_syst], g=g, labels=['samples', 'half samples', 'Gaussian', 'Gaussian + Syst'], params=params)
                    #fig = plt.gcf()
                    #fig.suptitle(r'{tracer} in ${zrange[0]:.1f} < z < {zrange[1]:.1f}$'.format(tracer=tracer, zrange=zrange), fontsize=16)
                    #utils.savefig(fn)
            
            if 0:
                from desilike.samples import Chain
                for tracer, zrange in tracers:
                    fn = '/global/cfs/cdirs/desicollab/science/cpe/y1kp7/v1/wrong/data/chain_bao_{}_GCcomb_z{:.1f}-{:.1f}.npy'.format('LRG' if zrange == (0.8, 1.1) else tracer, *zrange)
                    chain_v1 = Chain.load(fn)[::10]
                    chain_v12 = load_bao_chain(tracer, zrange)
                    iso = 'qpar' not in chain_v12
                    params = ['qiso'] if iso else ['qpar', 'qper']
                    fn = outdir / 'chain_bao_version_gaussian_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png'.format(tracer=tracer, zrange=zrange)
                    g = plots.get_subplot_plotter()
                    g.settings = style.settings
                    g.settings.num_plot_contours = 3
                    plotting.plot_triangle([chain_v1, chain_v12], g=g, labels=['v1', 'v1.2'], params=params)
                    #fig = plt.gcf()
                    #fig.suptitle(r'{tracer} in ${zrange[0]:.1f} < z < {zrange[1]:.1f}$'.format(tracer=tracer, zrange=zrange), fontsize=16)
                    utils.savefig(fn)

        if 'bao_versions' in todo:

            from desilike.samples import Chain, plotting

            def load_bao_chain(fi, burnin=0.5):
                chains = [Chain.load(ff).remove_burnin(burnin)[::10].select(name=['qpar', 'qper', 'qiso', 'qap']) for ff in fi]
                chain = chains[0].concatenate(chains)
                eta = 1. / 3.
                if 'qpar' in chain and 'qper' in chain:
                    chain.set((chain['qpar']**eta * chain['qper']**(1. - eta)).clone(param=dict(name='qiso', derived=True, latex=r'q_{\rm iso}')))
                    chain.set((chain['qpar'] / chain['qper']).clone(param=dict(name='qap', derived=True, latex=r'q_{\rm ap}')))
                if 'qiso' in chain and 'qap' in chain:
                    chain.set((chain['qiso'] * chain['qap']**(1. - eta)).clone(param=dict(name='qpar', derived=True, latex=r'q_{\parallel}')))
                    chain.set((chain['qiso'] * chain['qap']**(-eta)).clone(param=dict(name='qper', derived=True, latex=r'q_{\perp}')))
                z = chain.attrs['zeff']
                from desi_y1_files.cosmo_tools import predict_bao
                DH_over_rd_fid, DM_over_rd_fid = predict_bao(z, apmode='qparqper', scale='distance', eta=eta)
                DV_over_rd_fid, FAP_fid = predict_bao(z, apmode='qisoqap', scale='distance', eta=eta)
                chain.set((chain['qiso'] * DV_over_rd_fid).clone(param=dict(name='DV_over_rd', derived=True, latex=r'D_{\mathrm{V}} / r_{\mathrm{d}}')))
                if 'qper' in chain:
                    chain.set((chain['qper'] * DM_over_rd_fid).clone(param=dict(name='DM_over_rd', derived=True, latex=r'D_{\mathrm{M}} / r_{\mathrm{d}}')))
                    chain.set((chain['qpar'] * DH_over_rd_fid).clone(param=dict(name='DH_over_rd', derived=True, latex=r'D_{\mathrm{H}} / r_{\mathrm{d}}')))
                if 'qap' in chain:
                    chain.set((1. / chain['qap'] * FAP_fid).clone(param=dict(name='FAP', derived=True, latex=r'F_{\mathrm{AP}}')))
                return chain

            tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1))]
            
            from desi_y1_files import get_data_file_manager, get_bao_baseline_fit_setup
            dfm = get_data_file_manager(conf='unblinded')
            
            for tracer, zrange in tracers:
                options = dict(get_bao_baseline_fit_setup(tracer, zrange=zrange), version='v1.2')
                fchains = [str(fi) for fi in dfm.select(id='chains_bao_recon_y1', **options, ignore=True)]
                chain_v12 = load_bao_chain(fchains)
                
                tracer_v1 = 'LRG' if zrange == (0.8, 1.1) else tracer
                options = dict(get_bao_baseline_fit_setup(tracer_v1, zrange=zrange), version='v1')
                fchains = fchains_sigma = [str(fi) for fi in dfm.select(id='chains_bao_recon_y1', **options, ignore=True)]
                if 'BGS' in tracer:
                    fchains_sigma = [fi.replace('sigmapar-8.0-2.0_sigmaper-3.0-1.0', 'sigmapar-6.0-2.0_sigmaper-3.0-1.0') for fi in fchains]
                chain_v1 = load_bao_chain(fchains_sigma)
                
                fchains = [fi.replace('fits_2pt', 'fits_2pt/new_model') for fi in fchains]
                chain_v1_new_model = load_bao_chain(fchains)
                
                iso = 'qpar' not in chain_v12
                params = ['qiso'] if iso else ['qiso', 'qap']
                
                g = plots.get_subplot_plotter()
                g.settings = style.settings
                g.settings.num_plot_contours = 3
                print(tracer, zrange, 'new vs old v1', [(chain_v1_new_model.mean(param) - chain_v1.mean(param)) / chain_v1_new_model.std(param) for param in params])
                print(tracer, zrange, 'v12 vs new v1', [(chain_v12.mean(param) - chain_v1_new_model.mean(param)) / chain_v12.std(param) for param in params])
                print(tracer, zrange, 'v12 vs old v1', [(chain_v12.mean(param) - chain_v1.mean(param)) / chain_v12.std(param) for param in params])
                continue
                plotting.plot_triangle([chain_v1, chain_v1_new_model, chain_v12], g=g, params=params, labels=['v1', 'v1 new model', 'v1.2'], legend_labels=[] if iso else None)
                plt.gcf().suptitle('{} in ${:.1f} < z < {:.1f}$'.format(tracer.split('_')[0], *zrange), y=1.05)
                fn = outdir / 'comparison_bao_version_{}_z{:.1f}-{:.1f}.png'.format(tracer, *zrange)
                plt.tight_layout()
                plt.savefig(fn, bbox_inches='tight', dpi=360)

        if 'bao_versions_v15' in todo:

            from desilike.samples import Chain, plotting

            tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1))]
            base_dir = '/global/cfs/cdirs/desicollab/science/cpe/y1kp7/bao/'

            for tracer, zrange in tracers:
                chain_v12 = Chain.load(os.path.join(base_dir, 'data_internal', 'chain_bao_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy').format(tracer=tracer, zrange=zrange))
                chain_v15 = Chain.load(os.path.join(base_dir, 'data_v1.5', 'chain_bao_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy').format(tracer=tracer, zrange=zrange))
                
                iso = 'qpar' not in chain_v12
                params = ['qiso'] if iso else ['qiso', 'qap']
                
                g = plots.get_subplot_plotter()
                g.settings = style.settings
                g.settings.num_plot_contours = 3
                print(tracer, zrange, 'v15 vs v12', [(chain_v15.mean(param) - chain_v12.mean(param)) / chain_v12.std(param) for param in params])
                for param in chain_v12.params(basename=params):
                    param._latex = param._latex.replace('q', r'\alpha')
                plotting.plot_triangle([chain_v12, chain_v15], g=g, params=params, labels=['v1.2', 'v1.5'], legend_labels=[] if iso else None)
                plt.gcf().suptitle('{} in ${:.1f} < z < {:.1f}$'.format(tracer.split('_')[0], *zrange), y=1.05)
                fn = outdir / 'comparison_bao_version_{}_z{:.1f}-{:.1f}.png'.format(tracer, *zrange)
                plt.tight_layout()
                plt.savefig(fn, bbox_inches='tight', dpi=360)
                
        if 'comparisons' in todo:
            chain = load_cobaya_samples(theory=theory, model='base_mnu', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], source='', label='main')
            hanyuz = load_cobaya_samples(theory=theory, model='base_mnu', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], source='hanyuz', label='hanyuz')
            print(chain.getGelmanRubinEigenvalues()[-1], hanyuz.getGelmanRubinEigenvalues()[-1])
            g = plots.get_subplot_plotter()
            # Modify the plot_2d call to include the colors argument
            g.settings = style.settings
            g.triangle_plot([chain, hanyuz], ['H0', 'ombh2', 'omch2', 'logA', 'mnu'])
            fn = outdir / 'comparison_mnu_hanyuz.pdf'
            plt.tight_layout()
            plt.savefig(fn, bbox_inches='tight', dpi=360)

            """
            chain = load_cobaya_samples(theory=theory, model='base_mnu_w_wa', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], source='', label='main')
            hanyuz = load_cobaya_samples(theory=theory, model='base_mnu_w_wa', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], source='hanyuz', label='hanyuz')
            g = plots.get_subplot_plotter()
            # Modify the plot_2d call to include the colors argument
            g.settings = style.settings
            g.triangle_plot([chain, hanyuz], ['H0', 'ombh2', 'omch2', 'logA', 'w', 'wa', 'mnu'])
            fn = outdir / 'comparison_mnu_w_wa_hanyuz.pdf'
            plt.tight_layout()
            plt.savefig(fn, bbox_inches='tight', dpi=360)
            """
            """
            chain = load_cobaya_samples(theory=theory, model='base_w_wa', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], source='', label='main')
            kushal = load_cobaya_samples(theory=theory, model='base_w_wa', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], source='kushal', label='kushal')
            g = plots.get_subplot_plotter()
            # Modify the plot_2d call to include the colors argument
            g.settings = style.settings
            g.triangle_plot([chain, kushal], ['H0', 'ombh2', 'omch2', 'logA', 'w', 'wa'])
            fn = outdir / 'comparison_w_wa_kushal.pdf'
            plt.tight_layout()
            plt.savefig(fn, bbox_inches='tight', dpi=360)

            chain = load_cobaya_samples(theory=theory, model='base_omegak_w_wa', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], source='', label='main')
            kushal = load_cobaya_samples(theory=theory, model='base_omegak_w_wa', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], source='kushal', label='kushal')
            g = plots.get_subplot_plotter()
            # Modify the plot_2d call to include the colors argument
            g.settings = style.settings
            g.triangle_plot([chain, kushal], ['H0', 'ombh2', 'omch2', 'logA', 'omk', 'w', 'wa'])
            fn = outdir / 'comparison_omegak_w_wa_kushal.pdf'
            plt.tight_layout()
            plt.savefig(fn, bbox_inches='tight', dpi=360)
            """

        if 'examples' in todo:
            # Example from Jiaming Pan
            desi = load_cobaya_samples(theory=theory, model='base_w_wa', dataset=['desi-bao-all'])
            desi_planck = load_cobaya_samples(theory=theory, model='base_w_wa', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'])
            g = plots.get_single_plotter(width_inch=7, ratio=1, scaling=True)
            # Modify the plot_2d call to include the colors argument
            g.settings = style.settings
            g.plot_2d([desi, desi_planck], 'w', 'wa', filled=True)
            legend = g.add_legend(['DESI', 'DESI + Planck T\&P'], legend_loc='upper right')
            ax = g.get_axes()
            ax.set_xlabel(r'$w_0$', fontsize=g.settings.axes_labelsize, color='black')
            ax.set_ylabel(r'$w_a$', fontsize=g.settings.axes_labelsize, color='black')
            ax.tick_params(axis='both', which='major', labelsize=g.settings.axes_fontsize)
            ax.axhline(y=0, color='black', linestyle='--')
            ax.axvline(x=-1, color='black', linestyle='--')
            ax.set_xlim([-1.2, -0.35])
            ax.set_ylim([-2.5, 1.5])
            fn = outdir / 'w0_wa.pdf'
            plt.tight_layout()
            plt.savefig(fn, bbox_inches='tight', dpi=360)

        if 'data' in todo:
            # load BAO fit data from file
            from y1_bao_cosmo_tools import load_bao_fisher
            tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1)), ('Lya', (1.8, 4.2))]
            for tracer in tracers:
                try:
                    fisher = load_bao_fisher(*tracer, apmode='qparqper', scale='distance', return_type=None)
                except:
                    fisher = load_bao_fisher(*tracer, apmode='qiso', scale='distance', return_type=None)
                print(tracer, fisher.attrs['zeff'])
                print(fisher.to_stats(tablefmt='pretty'))
            
            # data / model plot by Sesh Nadathur
            from y1_bao_cosmo_tools import load_bao_fisher
            from cosmoprimo.fiducial import Planck2018FullFlatLCDM
            desi = DESI(engine='camb')
            planck = Planck2018FullFlatLCDM(engine='camb')
            #bestfit = load_cobaya_samples(model='base', dataset=['desi-bao-all', 'schoneberg2024-bbn'], source='', sampler='iminuit')
            #desi_best = desi.clone(N_eff=3.044, omega_cdm=bestfit['omch2'], omega_b=bestfit['ombh2'], H0=bestfit['H0'])
            bestfit = load_cobaya_samples(model='base', dataset=['desi-bao-all'], source='', sampler='iminuit')
            desi_best = desi.clone(N_eff=3.044, Omega_m=bestfit['omegam']).solve('H0', lambda cosmo: cosmo.rs_drag, target=bestfit['H0rdrag'] / 100, limits=[50, 100])
            assert np.allclose(desi_best.Omega0_m, bestfit['omegam']) and np.allclose(desi_best.rs_drag, bestfit['H0rdrag'] / 100)

            # load BAO fit data from file
            tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1)), ('Lya', (1.8, 4.2))]

            data = {tracer: load_bao_fisher(*tracer, scale=None, return_type=None) for tracer in tracers}
            fig = style.plot_bao_data_v1(data=None, cosmo=desi_best, label_cosmo=r'best fit $\Lambda$CDM')
            fig = style.plot_bao_data_v1(data=data, cosmo=planck, label_data=False, label_cosmo=r'Planck 2020', ls='--', fig=fig)
            fn = outdir / 'data_upper_panel.pdf'
            plt.savefig(fn, bbox_inches='tight', dpi=360)
            fig = style.plot_bao_data_v2(data=data, cosmo_best=desi_best, cosmo_alt=planck, label_cosmo_alt=r'Planck 2020', label_data=False, ls='--')
            fn = outdir / 'data_lower_panel.pdf'
            plt.savefig(fn, bbox_inches='tight', dpi=360)

        if 'expansion_history' in todo:
            from cosmoprimo.fiducial import Planck2018FullFlatLCDM
            planck = Planck2018FullFlatLCDM(engine='camb')
            # H(z)rd/(1+z) plot by Rodrigo Calderon
            from y1_bao_cosmo_tools import load_bao_fisher
            #from y1_cosmo_tools import compute_Hrd
            models = {r'$\Lambda$CDM': 'base', r'$w$CDM': 'base_w', r'$w_0w_a$CDM': 'base_w_wa'}
            chains = {label: load_cobaya_samples(theory=theory, model=model, dataset=['desi-bao-all']) for label, model in models.items()}
            # Extract the measurements of the quantity H(z)rd for the different tracers
            tracers = [('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('Lya', (1.8, 4.2))]
            #data = {tracer: load_bao_fisher(*tracer, scale=None, return_type=None) for tracer in tracers}
            data = None
            # Actual plotting
            fn = outdir / 'Hrd.pdf'
            style.plot_Hrd(chains, colors=style.color_palette, data=data, cosmo=planck, label_cosmo=r'Planck ($\Lambda\rm CDM$)', axis_H='right', scale_Hrd=100, fn=fn)

        if 'dark_energy' in todo:
            #  Table by Tianke Zhuang
            from y1_bao_cosmo_tools import make_table
            models = {r'$\Lambda$CDM': 'base', r'$w$CDM': 'base_w', r'$w_0w_a$CDM': 'base_w_wa',
                      #r'$\Lambda$CDM+$\sum m_\nu$': 'base_mnu', r'$\Lambda$CDM+$\Omega_k$': 'base_omegak',
                      # Add more models with corresponding latex labels, if needed!
                      }
            datasets = {'DESI': ['desi-bao-all'], 'DESI+SNIa+CMB': ['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], 'DESI+SNIa+CMB': ['desi-bao-all', 'pantheonplus', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']}
            samples = {(label_model, label_dataset): load_cobaya_samples(model=model, dataset=dataset) for label_dataset, dataset in datasets.items() for label_model, model in models.items()}
            fn = outdir / 'table.tex'
            make_table(samples, params=('omegam', 'w0', 'wa', 'H0', 'H0rd'), add_rule=True, fn=fn)

        if 'hubble_tension' in todo:
            pass

        if 'investigate_H0' in todo:
            samples_rdrag = load_cobaya_samples(theory='camb', model='base', dataset=['desi-bao-all', 'planck2018-rdrag'], label='rdrag')
            samples_bbn = load_cobaya_samples(theory='camb', model='base', dataset=['desi-bao-all', 'schoneberg2024-bbn'], label='bbn')
            samples_bbn_cmb = load_cobaya_samples(theory='camb', model='base', run='test0', dataset=['desi-bao-all', 'schoneberg2024-bbn'], label='bbn - CMB basis')
            samples_rdrag_shifted = load_cobaya_samples(theory='camb', model='base', run='test0', dataset=['desi-bao-all', 'planck2018-rdrag-shifted'], label='rdrag-shifted')
            print(samples_bbn.getLatex(params=['H0'], limit=1, err_sig_figs=2), samples_bbn_cmb.getLatex(params=['H0'], limit=1, err_sig_figs=2), samples_rdrag.getLatex(params=['H0'], limit=1, err_sig_figs=2), samples_rdrag_shifted.getLatex(params=['H0'], limit=1, err_sig_figs=2))
            print(samples_bbn.std('H0') / samples_bbn.mean('H0'), samples_bbn.std('H0rdrag') / samples_bbn.mean('H0rdrag'), samples_rdrag.std('H0') / samples_rdrag.mean('H0'), samples_rdrag.std('H0rdrag') / samples_rdrag.mean('H0rdrag'))
            print(samples_bbn.std('H0') / samples_bbn.mean('H0') / (samples_rdrag.std('H0') / samples_rdrag.mean('H0')))
            g = plots.get_subplot_plotter()
            # Modify the plot_2d call to include the colors argument
            g.settings = style.settings
            g.triangle_plot([samples_bbn, samples_bbn_cmb, samples_rdrag, samples_rdrag_shifted], ['H0', 'H0rd', 'rd', 'ombh2', 'omegam'])
            fn = outdir / 'comparison_H0.pdf'
            plt.tight_layout()
            plt.savefig(fn, bbox_inches='tight', dpi=360)
            
            samples_rdrag = load_cobaya_samples(theory='camb', model='base', dataset=['mock-fiducial-desi-bao-all', 'planck2018-rdrag'], run='test0', label='rdrag')
            samples_bbn = load_cobaya_samples(theory='camb', model='base', dataset=['mock-fiducial-desi-bao-all', 'schoneberg2024-bbn'], run='test0', label='bbn')
            print(samples_bbn.std('H0') / samples_bbn.mean('H0'), samples_bbn.std('rdrag') / samples_bbn.mean('rdrag'), samples_bbn.std('H0rdrag') / samples_bbn.mean('H0rdrag'))
            print(samples_rdrag.std('H0') / samples_rdrag.mean('H0'), samples_rdrag.std('H0rdrag') / samples_rdrag.mean('H0rdrag'))
            print(samples_bbn.std('H0') / samples_bbn.mean('H0') / (samples_rdrag.std('H0') / samples_rdrag.mean('H0')))
            g = plots.get_subplot_plotter()
            # Modify the plot_2d call to include the colors argument
            g.settings = style.settings
            g.triangle_plot([samples_bbn, samples_rdrag], ['H0', 'H0rd', 'rd', 'ombh2', 'omegam'])
            fn = outdir / 'comparison_H0_fiducial.png'
            plt.tight_layout()
            plt.savefig(fn, bbox_inches='tight', dpi=360)

        if 'summary_H0' in todo:
            # By Rafaela Gsponer
            chain = load_cobaya_samples(theory=theory, model='base', dataset=['desi-sdss-bao-best_schoneberg2024-bbn'])
            center, error = chain.mean('H0'), chain.std('H0')
            fn = outdir / 'H0_summary.pdf'
            style.plot_summary_H0(center, error, fn=fn)

        if 'whisker_H0' in todo:
            # Whisker plot by Rodrigo Calderon
            models = {r'$\Lambda$CDM': 'base', r'$w$CDM': 'base_w', r'$w_0w_a$CDM': 'base_w_wa',
                      r'$\Lambda$CDM+$\sum m_\nu$': 'base_mnu',r'$\Lambda$CDM+$\Omega_k$': 'base_omegak',
                      # Add more models with corresponding latex labels, if needed!
                     } 
            # Load the DESI+BBN chains for all models
            desi_bbn = {label: load_cobaya_samples(theory=theory, model=model, dataset=['desi-bao-all_schoneberg2024-bbn']) for label, model in models.items()}

            # Load the DESI+SDSS+BBN chains for all models
            desi_sdss_bbn = {label: load_cobaya_samples(theory=theory, model=model, dataset=['desi-sdss-bao-best_schoneberg2024-bbn']) for label, model in models.items()}

            # Load the DESI+Planck chains for all models
            desi_planck = {label: load_cobaya_samples(theory=theory, model=model, dataset=['desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck2018-highl-plik-TTTEEE']) for label, model in models.items()}

            # Chains and corresponding labels for each data combinations
            chains_datasets = {'DESI+BBN': desi_bbn,
                               'DESI+SDSS+BBN': desi_sdss_bbn,
                               'DESI+Planck': desi_planck}

            # Compute tension levels for each of the datasets & models
            tension_levels = [np.array([utils.compute_tension(chain.mean(['H0']), chain.std(['H0']))[0] for k, chain in chains.items()]) for dataset, chains in chains_datasets.items()]
            # Actual plotting
            fn = outdir / 'H0_whisker.pdf'
            style.plot_whisker_H0(list(chains_datasets), tension_levels, list(models), colors=style.color_palette, fn=fn) 
        
        if 'neutrinos' in todo:
            pass
