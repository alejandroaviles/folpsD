"""Produce the plots for the KP7 key paper."""

import os
import glob

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from desilike.samples import Chain, plotting
from cosmoprimo.fiducial import DESI
from desipipe import setup_logging
from desi_y1_plotting import KP7StylePaper, utils
from desi_y1_files import get_data_file_manager, get_cosmo_file_manager

from y1_cosmo_tools import load_chain, load_planck_chain

    
def plot_H0(center, error):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize= (8, 5), sharex="col", sharey="row", gridspec_kw={"hspace": 0})
    axes.set_xlabel(R"$H_{0}$ [km/s/Mpc]")
    axes.set_xlim(60, 78)
    axes.set_ylim(0, 18)
    axes.set_yticks([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [R"(Riess et al.) SH0ES", R"(Pesce et al.) Masers", R"(Blakeslee et al.) SBF", R"(Huang et al.) Miras - SNIa", R"(Freedman et al.) TRGB - SNIa", R"DESI BAO+BBN", R"(Alam et al.) eBOSS+BOSS BAO+BBN", R"(Cuceu et al.) eBOSS Ly$\alpha$ DF+BBN", R"(Simon et al.) eBOSS+BOSS DF+BBN", R"(Gsponer et al.) eBOSS+BOSS DF+BBN", R"(Ivanov et al.) BOSS DF+BBN", R"(Schoneberg et al.) eBOSS+BOSS BAO+SF+BBN", R"(Aiola et al.) ACT", R"(Dutcher et al.) SPT", "(Aghanim et al.) Planck"])

    #CMB
    axes.errorbar(67.27, 16, xerr=[[ 0.6], [0.6 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5) #Planck (1807.06209)
    axes.errorbar(68.8, 15, xerr=[[ 1.5], [1.5  ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #SPT (2101.01684)
    axes.errorbar(67.9, 14, xerr=[[ 1.5], [1.5 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #ACT (2007.07288)

    #BOSS+eBOSS
    axes.errorbar(68.3, 13, xerr=[[ 0.69], [0.66 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)    # SF+BAO+BBN eBOSS (2209.14330)
    axes.errorbar(67.9, 12, xerr=[[ 1.1], [1.1 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)    # DM BOSS (1909.05277)
    axes.errorbar(67.33, 11, xerr=[[ 1.3], [1.3 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)    # DM eBOSS +BOSS z1 + BBN (in prep)
    axes.errorbar(68.27, 10, xerr=[[ 0.85], [0.78 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  # DM eBOSS QSO + BOSS + BBN (2210.14931)
    axes.errorbar(63.2, 9, xerr=[[ 2.5], [2.5 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5,)  # DM eBOSS Lya (2209.13942)
    axes.errorbar(67.35, 8, xerr=[[ 0.97], [0.97 ]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #(e)BOSS BAO + BBN (2007.08991)

    #DESI
    axes.errorbar(center, 7, xerr=error, fmt='s', color='b', elinewidth=1, capsize=3, capthick=1.5)

    plt.axhline(6, linestyle='dashed', color='k')
    plt.text(x=75.5, y=6.25, s='Indirect')
    plt.text(x=75.5, y=5.25, s='Direct')

    #SNIa
    axes.errorbar(69.8, 5, xerr=[[ 1.9], [1.9]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #TRGB (1908.10883)
    axes.errorbar(73.3, 4, xerr=[[ 4], [4]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)      #Miras (1908.10883)
    axes.errorbar(73.3, 3, xerr=[[ 2.5], [2.5]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)  #SBF (2101.02221)
    axes.errorbar(73.9, 2, xerr=[[3], [3]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5)       #Masers (2001.09213)
    axes.errorbar(73.04, 1, xerr=[[1.04], [1.04]], fmt='s', color='k', elinewidth=1, capsize=3, capthick=1.5) #SH0ES (2112.04510)

    # axes.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.2), fontsize=14)
    plt.axvspan(72, 74.08, alpha=0.05, color='blue')
    plt.axvspan(66.67, 67.87, alpha=0.05, color='red')


if __name__ == '__main__':

    setup_logging()
    
    todo = []
    #todo = ['bao_diagram', 'bao_triangle', 'shapefit_triangle', 'direct_triangle', 'direct_shapefit_triangle', 'planck2018_triangle']
    #todo = ['bao_diagram_final']
    #todo = ['bao_compression_triangle']
    #todo = ['gelman-rubin']
    #todo = ['direct_shapefit_triangle']
    #todo = ['direct_shapefit_planck_triangle']
    #todo = ['bao_full_shape_triangle']
    #todo = ['unblinding']
    todo = ['sn']

    if True: #'unblinding' in todo:
        version, conf = 'v1', 'wrong'
    else:
        version, conf = 'test', {}
    #version, conf = 'v1', 'blinded'
    dfm = get_data_file_manager(conf=conf)
    cfm = get_cosmo_file_manager(version=version, conf=conf)
    cfm_fid = get_cosmo_file_manager(version=version, conf='mock')
    fiducial = DESI()

    # Output directory
    for fc in cfm.select(id='gaussian_bao_y1').iter(intersection=False):
        outdir = Path(os.path.dirname(os.path.dirname(fc))) / 'plots_kp7'
        break
    
    with KP7StylePaper() as style:
        
        if 'unblinding' in todo:

            for fi in cfm.select(id='chain_cosmological_inference_y1', code='cobaya', datasets=['desi-bao-gaussian'], model=['base']).iter(intersection=False):
                chain = load_chain(fi)
                bestfit = chain.choice(index='argmax', input=True)
                bestfit = {}
                cosmo = fiducial.clone(base='input', **bestfit)
                data = cfm.select(id='gaussian_bao_y1', **fi.options, ignore=True).iter(intersection=False)
                data_selected = [dd for dd in data if any(dd.options['tracer'].lower()[:3] in dataset for dataset in fi.options['datasets'])]  # if not in dataset, all tracers are kept
                data_selected = [dd for dd in data if (dd.options['tracer'][:3], dd.options['zrange']) != ('ELG', (0.8, 1.1))]
                if data_selected: data = data_selected
                fn = os.path.join(outdir, 'bao_anisotropic_diagram_{code}_{model}_{datasets}.png'.format(**fi.foptions))
                zlim, qlim = (0., 2.5), (0.85, 1.15)
                fig, lax = plt.subplots(2, figsize=(8, 5), sharex=True, sharey=False, gridspec_kw={'height_ratios': (4, 1)}, squeeze=False)
                fig.subplots_adjust(hspace=0.05)
                style.plot_bao_diagram(data, apmode='qisoqap', label_bao={'qap': 'anisotropic'}, insets=data, zlim=zlim, qlim=qlim, fig=fig, fn=fn)
                fn = os.path.join(outdir, 'bao_isotropic_diagram_{code}_{model}_{datasets}.png'.format(**fi.foptions))
                zlim, qlim, figsize = (0., 2.5), (0.85, 1.15), (8, 4)
                fig = style.plot_bao_diagram(data, apmode='qiso', insets=data, zlim=zlim, qlim=qlim, figsize=figsize)
                ax = fig.axes[0]
                plt.text(0.02, 0.98, 'error bars are preliminary', transform=ax.transAxes, fontsize=6, color='black', alpha=1., ha='left', va='top')
                utils.savefig(fn, fig=fig)
                #chain.to_stats(fn=os.path.join(outdir, 'bao_{code}_{model}_{datasets}.txt'.format(**fi.foptions)))
            style.plot_legend(fn=outdir / 'legend.png')

            fid = 'chain_cosmological_inference_y1'
            from getdist import plots
            
            bao = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian', 'bbn-omega_b'], model=['base']))
            plot_H0(bao['H0'].mean(), bao['H0'].std(ddof=1))
            fn = outdir / 'hubble_tension.png'
            plt.tight_layout()
            plt.savefig(fn, bbox_inches='tight', pad_inches=0.1, dpi=200)

            # w/wa
            bao_cmb_fid = load_chain(cfm_fid.get(id=fid, code='importance-planck', datasets=['desi-bao-gaussian', 'planck2018'], model=['base_w_wa'])).to_getdist(params=['w0_fld', 'wa_fld'])
            bao_cmb = load_chain(cfm.get(id=fid, code='importance-planck', datasets=['desi-bao-gaussian', 'planck2018'], model=['base_w_wa'])).to_getdist(params=['w0_fld', 'wa_fld'])
            g = plots.get_single_plotter(width_inch=4, ratio=1)
            g.plot_2d([bao_cmb_fid, bao_cmb], 'w0_fld', 'wa_fld', filled=True)
            g.add_legend(['CMB T&P + BAO fiducial', 'CMB T&P + BAO'], legend_loc='upper right')
            ax = g.get_axes()
            fn = outdir / 'w0_wa.png'
            plt.savefig(fn)
            
            # Omega_k
            bao = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian'], model=['base_omegak'])).to_getdist()
            sn = load_chain(cfm.get(id=fid, code='cobaya', datasets=['pantheon'], model=['base_omegak'])).to_getdist()
            cmb = load_planck_chain(model='base_omegak', params=['Omega_m', 'Omega_k', 'Omega_Lambda']).to_getdist()
            g = plots.get_single_plotter(width_inch=4, ratio=1)
            g.plot_2d([cmb, sn, bao], 'Omega_m', 'Omega_Lambda', filled=True)
            g.add_legend(['CMB T&P', 'SN', 'BAO'], legend_loc='upper right')
            ax = g.get_axes()
            ax.plot([0., 1.], [1., 0.], color='k', linestyle='--')
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            ax.text(0.1, 0.1, 'o$\Lambda$CDM', transform=ax.transAxes, fontsize=18)
            fn = outdir / 'omegam_omegal.png'
            plt.savefig(fn, dpi=200)

            cmb = load_planck_chain(model='base_omegak', params=['Omega_m', 'Omega_k', 'Omega_Lambda']).to_getdist()
            bao_cmb = load_chain(cfm.get(id=fid, code='importance-planck', datasets=['desi-bao-gaussian', 'planck2018'], model=['base_omegak']))
            bao_cmb_desi = bao_cmb.deepcopy().to_getdist(params=['Omega_m', 'Omega_k', 'Omega_Lambda'])
            bao_cmb.aweight[...] = 1.
            bao_cmb_sdss = bao_cmb.to_getdist(params=['Omega_m', 'Omega_k', 'Omega_Lambda'])
            g = plots.get_single_plotter(width_inch=4, ratio=1)
            #g.plot_2d([cmb, bao_cmb_sdss, bao_cmb_desi], 'Omega_m', 'Omega_k', filled=True)
            #g.add_legend(['CMB T&P', 'CMB T&P + BAO SDSS', 'CMB T&P + BAO'], legend_loc='lower right')
            g.plot_2d([cmb, bao_cmb_desi], 'Omega_m', 'Omega_k', filled=True)
            g.add_legend(['CMB T&P', 'CMB T&P + BAO'], legend_loc='upper right')
            #ax = g.get_axes()
            #ax.text(0.1, 0.1, 'o$\Lambda$CDM')
            fn = outdir / 'omegam_omegak.png'
            plt.savefig(fn, dpi=200)

            # w
            bao = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian'], model=['base_w'])).to_getdist(params=['Omega_m', 'w0_fld'])
            sn = load_chain(cfm.get(id=fid, code='cobaya', datasets=['pantheon'], model=['base_w'])).to_getdist(params=['Omega_m', 'w0_fld'])
            cmb = load_planck_chain(model='base_w', params=['Omega_m', 'w0_fld']).to_getdist()
            g = plots.get_single_plotter(width_inch=4, ratio=1)
            g.plot_2d([cmb, sn, bao], 'Omega_m', 'w0_fld', filled=True)
            g.add_legend(['CMB T&P', 'SN', 'BAO'], legend_loc='lower right')
            ax = g.get_axes()
            ax.axhline(y=-1., color='k', linestyle='--')
            ax.set_ylim(-3., 0.)
            ax.text(0.1, 0.1, '$w$CDM', transform=ax.transAxes, fontsize=18)
            fn = outdir / 'omegam_w.png'
            plt.savefig(fn, dpi=200)

            # mnu
            bao_cmb = load_chain(cfm.get(id=fid, code='importance-planck', datasets=['desi-bao-gaussian', 'planck2018'], model=['base_mnu']))
            bao_cmb_desi = bao_cmb.deepcopy().to_getdist(params=['Omega_m', 'm_ncdm'])
            bao_cmb.aweight[...] = 1.
            bao_cmb_sdss = bao_cmb.to_getdist(params=['Omega_m', 'm_ncdm'])
            cmb = load_planck_chain(model='base_mnu', params=['Omega_m', 'm_ncdm']).to_getdist(params=['Omega_m', 'm_ncdm'])
            g = plots.get_single_plotter(width_inch=4, ratio=1)
            g.plot_2d([cmb, bao_cmb_sdss, bao_cmb_desi], 'Omega_m', 'm_ncdm', filled=True)
            g.add_legend(['CMB T&P', 'CMB T&P + SDSS BAO', 'CMB T&P + BAO'], legend_loc='upper right')
            #g.plot_2d([cmb, bao_cmb_desi], 'Omega_m', 'm_ncdm', filled=True)
            #g.add_legend(['CMB T&P', 'CMB T&P + BAO'], legend_loc='upper right')
            ax = g.get_axes()
            ax.set_ylabel(r'$\sum m_{\nu}$ [eV]')
            ax.text(0.1, 0.1, '$\nu$$\Lambda$CDM')
            fn = outdir / 'omegam_mnu.png'
            plt.savefig(fn)

            # H0
            bao_lowz = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian-bgs-lrg-elg-qso', 'bbn-omega_b'], model=['base'])).to_getdist(params=['Omega_m', 'H0'])
            #bao_highz = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian-elg-qso-lya', 'bbn-omega_b'], model=['base'])).to_getdist(params=['Omega_m', 'H0'])
            bao = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian', 'bbn-omega_b'], model=['base'])).to_getdist(params=['Omega_m', 'H0'])
            g = plots.get_single_plotter(width_inch=4, ratio=1)
            g.plot_2d([bao_lowz, bao], 'Omega_m', 'H0', filled=True)
            g.add_legend(['BAO (w/out Lya) + BBN', 'BAO + BBN'], legend_loc='upper right')
            ax = g.get_axes()
            g.add_y_bands(73.04, 1.04)
            ax.text(0.15, 73.04, 'SHOES', va='center', ha='left', fontsize=18)
            ax.set_ylabel('$H_{0}$ [km/s/Mpc]')
            ax.text(0.1, 0.1, '$\Lambda$CDM', va='center', ha='left', transform=ax.transAxes, fontsize=18)
            ax.set_xlim(0.1, 0.5)
            ax.set_ylim(64., 76.)
            fn = outdir / 'omegam_h0.png'
            plt.savefig(fn, dpi=200)

            bao_lowz = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian-bgs-lrg', 'bbn-omega_b'], model=['base'])).to_getdist(params=['Omega_m', 'H0'])
            bao_highz = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian-elg-qso-lya', 'bbn-omega_b'], model=['base'])).to_getdist(params=['Omega_m', 'H0'])
            bao = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian', 'bbn-omega_b'], model=['base'])).to_getdist(params=['Omega_m', 'H0'])
            g = plots.get_single_plotter(width_inch=4, ratio=1)
            g.plot_2d([bao_lowz, bao_highz, bao], 'Omega_m', 'H0', filled=True)
            g.add_legend(['BAO (BGS/LRG) + BBN', 'BAO (ELG/QSO/Lya) + BBN', 'BAO + BBN'], legend_loc='upper right')
            ax = g.get_axes()
            g.add_y_bands(73.04, 1.04)
            ax.text(0.15, 73.04, 'SHOES', va='center', ha='left', fontsize=18)
            ax.set_ylabel('$H_{0}$ [km/s/Mpc]')
            ax.text(0.1, 0.1, '$\Lambda$CDM', va='center', ha='left', transform=ax.transAxes, fontsize=18)
            ax.set_xlim(0.1, 0.5)
            ax.set_ylim(64., 78.)
            fn = outdir / 'omegam_h0_lowz_highz.png'
            plt.savefig(fn, dpi=200)

            if False:
                chains, labels = [], []
                for tracer in ['bgs', 'lrg', 'elg', 'qso', 'lya']:
                    chain = load_chain(cfm.get(id=fid, code='cobaya', datasets=['desi-bao-gaussian-{}'.format(tracer), 'bbn-omega_b'], model=['base']))
                    chains.append(chain.to_getdist())
                    labels.append('{} + BBN'.format(tracer.upper()))
                g = plots.get_single_plotter(width_inch=6, ratio=1)
                g.plot_2d(chains, 'Omega_m', 'H0', filled=True)
                g.add_legend(labels, legend_loc='upper right', ncols=2)
                ax = g.get_axes()
                g.add_y_bands(73.04, 1.04)
                ax.text(0.15, 73.04, 'SHOES', va='center', ha='left', fontsize=18)
                ax.set_ylabel('$H_{0}$ [km/s/Mpc]')
                ax.text(0.1, 0.1, '$\Lambda$CDM', va='center', ha='left', transform=ax.transAxes, fontsize=18)
                ax.set_xlim(0.1, 0.5)
                ax.set_ylim(64., 78.)
                fn = outdir / 'omegam_h0_all.png'
                plt.savefig(fn, dpi=200)
            
        
        if 'bao_diagram' in todo:
            for fi in cfm.select(id='chain_cosmological_inference_y1', code='cobaya', datasets=['desi-bao-gaussian'], model=['base', 'base_omegak', 'base_w', 'base_w_wa']).iter(intersection=False):
                chain = load_chain(fi)
                bestfit = chain.choice(index='argmax', input=True)
                cosmo = fiducial.clone(base='input', **bestfit)
                data = cfm.select(id='gaussian_bao_y1', **fi.options, ignore=True).iter(intersection=False)
                data_selected = [dd for dd in data if any(dd.options['tracer'].lower()[:3] in dataset for dataset in fi.options['datasets'])]  # if not in dataset, all tracers are kept
                data_selected = [dd for dd in data if not (dd.options['tracer'].lower().startswith('elg') and tuple(dd.options['zrange']) == (0.8, 1.1))]
                if data_selected: data = data_selected
                fn = os.path.join(outdir, 'bao_diagram_{code}_{model}_{datasets}.png'.format(**fi.foptions))
                style.plot_bao_diagram(data, cosmo=cosmo, label_cosmo='bestfit', apmode='qparqper', insets=[], figsize=(8, 6), fn=fn)
                for seed in [False, 42]:
                    is_random = bool(seed)
                    fn = os.path.join(outdir, 'bao_isotropic_diagram_{code}_{model}_{datasets}'.format(**fi.foptions)) + ('_random.png' if is_random else '.png')
                    zlim, figsize = (0., 2.), (8, 4)
                    if 'lya' in [dd.options['tracer'].lower()[:3] for dd in data]:
                        zlim, figsize = (0., 2.5), (11, 4)
                    fig = style.plot_bao_diagram(data, cosmo=cosmo, label_cosmo='bestfit', apmode='qiso', insets=data, zlim=zlim, seed=seed, figsize=(9, 4.5))
                    for ax in fig.axes[1:]:
                        ax.set_xlabel(ax.get_xlabel() + ' *')
                        ax.set_ylabel(ax.get_ylabel() + ' *')
                    ax = fig.axes[0]
                    ax.get_legend().remove()
                    watermark = 'Preliminary/Blinded'
                    if is_random: watermark += '\nData points scattered randomly'
                    plt.text(0.5, 0.5, watermark, transform=ax.transAxes, fontsize=30, color='gray', alpha=0.4, ha='center', va='center', rotation=30)
                    plt.text(0.02, 0.98, '(*) distances and/or correlations have been shifted as part of the blinding process', transform=ax.transAxes, fontsize=6, color='black', alpha=1., ha='left', va='top')
                    utils.savefig(fn, fig=fig)
            style.plot_legend(fn=outdir / 'legend.png')

        if 'bao_diagram_final' in todo:
            fi = cfm.get(id='chain_cosmological_inference_y1', code='cobaya', datasets=['desi-bao-gaussian'], model=['base'])
            #bestfit = chain.choice(index='argmax', input=True)
            #cosmo = fiducial.clone(base='input', **bestfit)
            data = cfm.select(id='gaussian_bao_y1', **fi.options, ignore=True).iter(intersection=False)
            data_selected = [dd for dd in data if any(dd.options['tracer'].lower()[:3] in dataset for dataset in fi.options['datasets'])]  # if not in dataset, all tracers are kept
            data_selected = [dd for dd in data if not (dd.options['tracer'].lower().startswith('elg') and tuple(dd.options['zrange']) == (0.8, 1.1))]
            if data_selected: data = data_selected
            fn = os.path.join(outdir, 'bao_diagram_{datasets}.png'.format(**fi.foptions))
            for seed in [False, 42]:
                is_random = bool(seed)
                fn = os.path.join(outdir, 'bao_isotropic_diagram_{datasets}'.format(**fi.foptions)) + ('_random.png' if is_random else '.png')
                zlim, figsize = (0., 2.), (8, 4)
                if 'lya' in [dd.options['tracer'].lower()[:3] for dd in data]:
                    zlim, figsize = (0., 2.5), (11, 4)
                fig = style.plot_bao_diagram(data, apmode='qiso', insets=data, zlim=zlim, seed=seed, figsize=(9, 4.5))
                for ax in fig.axes[1:]:
                    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
                    #ax.set_xlabel(ax.get_xlabel() + ' *')
                    #ax.set_ylabel(ax.get_ylabel() + ' *')
                ax = fig.axes[0]
                watermark = 'Preliminary/Blinded'
                if is_random: watermark += '\nData points scattered randomly'
                plt.text(0.5, 0.5, watermark, transform=ax.transAxes, fontsize=30, color='gray', alpha=0.4, ha='center', va='center', rotation=30)
                utils.savefig(fn, fig=fig)
                utils.savefig(fn.replace('.png', '.pdf'), fig=fig)
            style.plot_legend(fn=outdir / 'legend.png')
            
        if 'bao_triangle' in todo:
            for model in ['base', 'base_omegak', 'base_w', 'base_w_wa']:
                code = 'cobaya'

                externals = []
                labels = ['DESI point tracers', 'DESI', 'DESI + Pantheon']
                datasets = [['desi-bao-gaussian-bgs-lrg-elg-qso'] + externals, ['desi-bao-gaussian'] + externals, ['desi-bao-gaussian', 'pantheon'] + externals]
                chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for datasets in datasets]
                params = chains[0].params(basename=['Omega_m', 'Omega_k', 'w0_fld', 'wa_fld'])
                markers = {name: fiducial[name] for name in params.names()}
                plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_distance_{code}_{model}.png'.format(code=code, model=model))

                externals = ['bbn-omega_b']
                labels = ['DESI point tracers + BBN', 'DESI + BBN', 'DESI + Pantheon + BBN'][1:]
                datasets = [['desi-bao-gaussian-bgs-lrg-elg-qso'] + externals, ['desi-bao-gaussian'] + externals, ['desi-bao-gaussian', 'pantheon'] + externals][1:]
                chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for datasets in datasets]
                params = chains[0].params(basename=['H0', 'Omega_m', 'Omega_k', 'w0_fld', 'wa_fld'])
                markers = {name: fiducial[name] for name in params.names()}
                plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_H0_BBN_{code}_{model}.png'.format(code=code, model=model))

        if 'shapefit_triangle' in todo:
            code = 'cobaya'
            datasets = ['desi-shapefit-gaussian', 'bbn-omega_b']

            models = ['base_w_wa', 'base_w', 'base']
            labels = ['$w_{0} - w_{a}$ free', '$w_{0}$ free', 'base']
            chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for model in models]
            params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'])
            markers = {name: fiducial[name] for name in params.names()}
            plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_shapefit_{code}_base_w_wa_{datasets}.png'.format(code=code, datasets='_'.join(datasets)))

            models = ['base_omegak', 'base']
            labels = ['$\Omega_{k}$ free', 'base']
            chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for model in models]
            params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'])
            markers = {name: fiducial[name] for name in params.names()}
            plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_shapefit_{code}_base_omegak_{datasets}.png'.format(code=code, datasets='_'.join(datasets)))

            models = ['base_mnu', 'base']
            labels = ['$m_{ncdm}$ free', 'base']
            chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for model in models]
            params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'])
            markers = {name: fiducial[name] for name in params.names()}
            plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_shapefit_{code}_base_mnu_{datasets}.png'.format(code=code, datasets='_'.join(datasets)))

        if 'direct_triangle' in todo:
            code = 'cobaya'
            datasets = ['desi-full-shape-power', 'bbn-omega_b']
            models = ['base_w_wa', 'base_w', 'base']
            labels = ['$w_{0} - w_{a}$ free', '$w_{0}$ free', 'base']
            chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for model in models]
            params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'])
            markers = {name: fiducial[name] for name in params.names()}
            plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_direct_{code}_base_w_wa_{datasets}.png'.format(code=code, datasets='_'.join(datasets)))

            code = 'cobaya'
            datasets = [['desi-full-shape-power'], ['desi-full-shape-power', 'bbn-omega_b']]
            models = ['base']
            labels = ['base, no BBN prior', 'base, BBN prior']
            chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model='base', datasets=datasets)) for datasets in datasets]
            params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'])
            markers = {name: fiducial[name] for name in params.names()}
            plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_direct_{code}_base_bbnprior.png'.format(code=code))

        if 'direct_shapefit_triangle' in todo:
            code = 'cobaya'
            #for model in ['base_ns-fixed', 'base_w_ns-fixed', 'base_w_wa_ns-fixed'][:0] + ['base', 'base_w', 'base_w_wa'][:2]:
            for model in ['base_ns-fixed', 'base_mnu_ns-fixed'] + ['base', 'base_mnu']:
                labels = ['shapefit', 'direct']
                datasets = [['desi-shapefit-gaussian', 'bbn-omega_b'], ['desi-full-shape-power', 'bbn-omega_b']]
                chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for datasets in datasets]
                params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'], varied=True)
                markers = {name: fiducial[name] for name in params.names()}
                plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_direct_shapefit_{code}_{model}.png'.format(code=code, model=model))

        if 'bao_compression_triangle' in todo:
            code = 'cobaya'
            for model in ['base', 'base_w', 'base_w_wa'][:2]:
                labels = ['compression', 'combined', 'moving template']
                chains = []
                chains.append(load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=['desi-bao-gaussian-all-params'])))
                chains.append(load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=['desi-bao-correlation-recon-all-params'])))
                chains.append(load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=['desi-bao-correlation-recon-direct-all-params'])))
                params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'], varied=True)
                markers = {name: fiducial[name] for name in params.names()}
                plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_bao_compression_{code}_{model}.png'.format(code=code, model=model))

        if 'planck2018_triangle' in todo:
            from desilike.likelihoods.cmb import read_planck2018_chain
            for model in ['base', 'base_omegak', 'base_w']:
                code = 'importance-planck'
                labels = ['Planck 2018', 'Planck 2018 + DESI BAO', 'Planck 2018 + DESI BAO point tracers', 'Planck 2018 + DESI ShapeFit']
                datasets = [['desi-bao-gaussian', 'planck2018'], ['desi-bao-gaussian-bgs-lrg-elg-qso', 'planck2018'], ['desi-shapefit-gaussian', 'planck2018']]
                chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for datasets in datasets]
                params = chains[-1].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'])
                params = [param for param in params if param.varied or param.derived]
                chains = [read_planck2018_chain(basename='{}_plikHM_TTTEEE_lowl_lowE'.format(model), weights='cmb_only', params=params)] + chains
                markers = {param.name: fiducial[param.name] for param in params}
                plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_bao_shapefit_{code}_{model}.png'.format(code=code, model=model))

        if 'gelman-rubin' in todo:
            from desilike.samples import plotting
            model = 'base'
            code = 'cobaya'
            datasets = [['desi-full-shape-power', 'bbn-omega_b']]
            datasets += [['desi-full-shape-power-bao-correlation-recon', 'bbn-omega_b']]
            for datasets in datasets:
                chains = load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets), concatenate=False)
                for tracer in ['cosmo', 'BGS_0', 'LRG_0', 'LRG_1', 'LRG_2', 'ELG_0', 'ELG_1', 'QSO_0']:
                    if tracer == 'cosmo':
                        params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'], varied=True)
                    else:
                        params = chains[0].params(name='*{}*'.format(tracer), varied=True, derived=False)
                    fn = outdir / 'gelman-rubin_{code}_{model}_{datasets}_{tracer}.png'.format(code=code, model=model, datasets='_'.join(datasets), tracer=tracer)
                    nsteps = min(chain.size for chain in chains)
                    slices = np.arange(5000, nsteps, 500)
                    fig = plotting.plot_gelman_rubin(chains, params=params, multivariate=True, threshold=1e-2, slices=slices, offset=-1)
                    ax = fig.axes[0]
                    ax.legend(ncol=2, loc=4)
                    ax.set_yscale('log')
                    plotting.savefig(fn)

        if 'bao_full_shape_triangle' in todo:
            code = 'cobaya'
            for model in ['base', 'base_w', 'base_w_wa'][:2]:
                labels = ['full-shape (direct) + BBN', 'BAO + BBN', 'BAO + full-shape (direct) + BBN']
                datasets = [['desi-full-shape-power', 'bbn-omega_b'], ['desi-bao-correlation-recon', 'bbn-omega_b'], ['desi-full-shape-power-bao-correlation-recon', 'bbn-omega_b']]
                chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for datasets in datasets]
                params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'], varied=True)
                markers = {name: fiducial[name] for name in params.names()}
                plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_bao_full_shape_{code}_{model}.png'.format(code=code, model=model))

        if 'direct_shapefit_planck_triangle' in todo:
            code = 'cobaya'
            for model in ['base', 'base_w', 'base_w_wa'][:2]:
                labels = ['Planck2018-lite', 'shapefit + Planck2018-lite', 'direct + Planck2018-lite']
                datasets = [['planck2018-lite'], ['desi-shapefit-gaussian', 'planck2018-lite'], ['desi-full-shape-power', 'planck2018-lite']]
                chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for datasets in datasets]
                params = chains[0].params(basename=['H0', 'Omega_b', 'Omega_m', 'Omega_k', 'logA', 'n_s', 'm_ncdm', 'w0_fld', 'wa_fld'], varied=True)
                markers = {name: fiducial[name] for name in params.names()}
                plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_direct_shapefit_planck2018-lite_{code}_{model}.png'.format(code=code, model=model))
        
        if 'sn' in todo:
            code = 'cobaya'
            for model in ['base', 'base_w', 'base_w_wa'][:1]:
                labels = ['Pantheon', 'Pantheon+', 'Union3']
                datasets = [['pantheon'], ['pantheon+'], ['union3']]
                chains = [load_chain(cfm.get(id='chain_cosmological_inference_y1', code=code, model=model, datasets=datasets)) for datasets in datasets]
                params = chains[0].params(basename=['H0', 'Omega_m'], varied=True)
                markers = {name: fiducial[name] for name in params.names()}
                plotting.plot_triangle(chains, [str(param) for param in params], markers=markers, labels=labels, filled=True, fn=outdir / 'triangle_sn_{code}_{model}.png'.format(code=code, model=model))
            