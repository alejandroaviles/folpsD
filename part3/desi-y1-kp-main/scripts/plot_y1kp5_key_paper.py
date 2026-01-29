"""Produce the plots for the KP5 key paper."""

import os

from pathlib import Path
from matplotlib import pyplot as plt

from desipipe import setup_logging
from desi_y1_plotting import KP5StylePaper, utils
from desi_y1_files import get_data_file_manager, get_abacus_file_manager


if __name__ == '__main__':

    setup_logging()
    fm = get_data_file_manager() + get_abacus_file_manager()
    todo = ['mean_abacus']
    #todo = ['abacus']
    cut = [('theta', 0.05)]
    # Output directory
    #outdir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/KP5plots/KeyPaper/'
    for fc in fm.select(id='catalog_data_y1', version='test').iter(intersection=False):
        outdir = Path(os.path.dirname(fc)) / 'plots_kp5'
        break

    with KP5StylePaper() as style:

        if 'mean_abacus' in todo:
            fms = fm.select(id='chains_full_shape_mean_abacus_y1', weighting='default_FKP', version='v2', fa='altmtl', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], region=['GCcomb'], cut=cut, precscale=1)
            for options in fms.iter_options(intersection=False, exclude=['tracer', 'zrange', 'ichain', 'observable', 'lim']):
                fi = []
                for options in fms.select(**options).iter_options(intersection=False, exclude=['ichain', 'observable', 'lim']):  # iterate on tracer, zrange
                    if options['zrange'] == (0.8, 1.6): continue
                    fi.append(tuple(fms.select(**options).iter(intersection=False, exclude=['ichain']))) # iterate on observable
                    labels = [ff.options['observable'][0] for ff in fi[-1]]
                    for ff in fi[-1][0]: foptions = ff.foptions
                fig = style.plot_full_shape_diagram(fi, labels=labels)
                utils.savefig(outdir / 'full_shape_abacus_{theory}_{template}_{fa}_{region}_precscale{precscale:d}{cut}.png'.format(**foptions), fig=fig)
                plt.close(fig)
        
        if 'abacus' in todo:
            fms = fm.select(id='profiles_full_shape_abacus_y1', weighting='default_FKP', version='v2', fa='altmtl', tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], region=['GCcomb'], cut=cut)
            for fi in fms.iter(intersection=False, exclude=['imock']):
                fi = list(fi)
                fig = style.plot_scatter(fi, rescale=True)
                utils.savefig(outdir / 'full_shape_abacus_{observable}_{theory}_{template}_{fa}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.png'.format(**fi[0].foptions), fig=fig)
                exit()
                plt.close(fig)