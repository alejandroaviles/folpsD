import os
from pathlib import Path

from desipipe import FileManager
from . import io  # to import covariance / correlation function files


list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'LRG+ELG_LOPnotqso': [(0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'QSO': [(0.8, 2.1)]}
#list_smoothing_radius = {'BGS_BRIGHT-21.5': [15.], 'LRG': [10.], 'ELG_LOPnotqso': [10.], 'QSO': [30.]}
list_smoothing_radius = {'BGS_BRIGHT-21.5': [15.], 'LRG': [15.], 'LRG+ELG_LOPnotqso': [15.], 'ELG_LOPnotqso': [15.], 'QSO': [30.]}
#list_nran = {'BGS_BRIGHT-21.5': 1, 'LRG': 5, 'ELG_LOPnotqso': 6, 'QSO': 3}
list_nran = {'BGS_BRIGHT-21.5': 1, 'LRG': 8, 'LRG+ELG_LOPnotqso': 10, 'ELG_LOPnotqso': 10, 'QSO': 4}
# fix boxsize to have same number of modes in NGC and SGC; at least 1.5x maximum extend of the survey, except for QSO (1.2x)
# these are not the exact box sizes; boxsize = ceil(nmesh * cellsize)
list_boxsize = {'BGS_BRIGHT-21.5': 4000., 'LRG': 7000., 'LRG+ELG_LOPnotqso': 9000., 'ELG_LOPnotqso': 9000., 'QSO': 10000.}
# AJR prefers putting everything at the top to make editing easy
weighting = ['default', 'default_FKP', 'default_SYS1_FKP', 'default_SYSIMLIN_FKP', 'default_SYSRF_FKP', 'default_SYSSN_FKP']
# max number of randoms
nranmax = 18


def get_fit_setup(tracer, ells=None, observable_name='power', theory_name='velocileptors', return_list=None):
    if ells is None:
        ells = (0, 2)
        if 'bao' in theory_name:
            ells = (0, 2)
    post = 'post' in theory_name
    if tracer.startswith('BGS'):
        b0 = 1.34
        if 'bao' in theory_name:
            klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
            slim = {ell: [50., 150., 4.] for ell in ells}
        else:
            klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
            slim = {ell: [32., 150., 4.] for ell in ells}
        #sigmapar, sigmaper = 9., 4.5
        #if post: sigmapar, sigmaper = 6., 3.
        sigmapar, sigmaper = 10., 6.5
        if post: sigmapar, sigmaper = 8., 3.
    if tracer.startswith('LRG'):
        b0 = 1.34
        if 'bao' in theory_name:
            klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
            slim = {ell: [50., 150., 4.] for ell in ells}
        else:
            klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
            slim = {ell: [30., 150., 4.] for ell in ells}
        sigmapar, sigmaper = 9., 4.5
        if post: sigmapar, sigmaper = 6., 3.
    if tracer.startswith('LRG+ELG'):
        b0 = 1.34 * (1.6 / 2.)
        if 'bao' in theory_name:
            klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
            slim = {ell: [50., 150., 4.] for ell in ells}
        else:
            klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
            slim = {ell: [30., 150., 4.] for ell in ells}
        sigmapar, sigmaper = 9., 4.5
        if post: sigmapar, sigmaper = 6., 3.
    if tracer.startswith('ELG'):
        b0 = 0.722
        if 'bao' in theory_name:
            klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
            slim = {ell: [50., 150., 4.] for ell in ells}
        else:
            klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
            slim = {ell: [27., 150., 4.] for ell in ells}
        sigmapar, sigmaper = 8.5, 4.5
        if post: sigmapar, sigmaper = 6., 3.
    if tracer.startswith('QSO'):
        b0 = 1.137
        if 'bao' in theory_name:
            klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
            slim = {ell: [50., 150., 4.] for ell in ells}
        else:
            klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
            slim = {ell: [25., 150., 4.] for ell in ells}
        sigmapar, sigmaper = 9., 3.5
        if post: sigmapar, sigmaper = 6., 3.
    #if 'bao' in theory_name:
    #    klim = {ell: [0.02, 0.2, 0.005] for ell in ells}
    if 'power' in observable_name:
        lim = klim
    if 'corr' in observable_name:
        lim = slim
    if 'bao' in theory_name and 'qiso' in theory_name and 'qap' not in theory_name:
        lim = {0: lim[0]}
    sigmas = {'sigmas': (2., 2.), 'sigmapar': (sigmapar, 2.), 'sigmaper': (sigmaper, 1.)}
    toret = {'b0': b0, 'lim': lim, 'sigmas': sigmas}
    if return_list is None:
        return toret
    if isinstance(return_list, str):
        return toret[return_list]
    return [toret[name] for name in return_list]


def get_baseline_recon_setup(tracer=None, zrange=tuple()):
    smoothing_radius = {'BGS_BRIGHT-21.5': 15., 'LRG': 15., 'LRG+ELG_LOPnotqso': 15., 'ELG_LOPnotqso': 15., 'QSO': 30.}
    toret = {'mode': 'recsym', 'algorithm': 'IFFT', 'smoothing_radius': smoothing_radius[tracer], 'recon_weighting': 'default', 'recon_zrange': None, 'zrange': zrange}
    if 'QSO' in tracer:
        toret['recon_zrange'] = (0.8, 2.1)
    if 'LRG+ELG' in tracer:
        toret['recon_zrange'] = (0.8, 1.1)
    return toret


def get_baseline_2pt_setup(tracer=None, zrange=tuple(), observable=None, recon=None):
    toret = {'weighting': 'default_FKP', 'cut': ('theta', 0.05)}
    if observable == 'correlation':
        toret.update({'split': 20., 'nran': list_nran[tracer], 'njack': 0})
    if recon:
        toret.update(get_baseline_recon_setup(tracer=tracer, zrange=zrange))
        toret.update({'cut': None})
    toret.pop('cut')
    return toret


def is_baseline_2pt_setup(tracer=None, zrange=tuple(), observable=None, **options):
    recon = False
    if any(name in options for name in ['mode', 'algorithm', 'smoothing_radius', 'recon_weighting', 'recon_zrange']): recon = True
    if any(name in options for name in ['split', 'njack']): observable = 'correlation'
    default_options = get_baseline_2pt_setup(tracer=tracer, zrange=zrange, observable=observable, recon=recon)
    from desipipe.file_manager import in_options
    #print(default_options, options, [in_options(value, [options[name]]) for name, value in default_options.items() if name in options])
    return all(in_options(value, [options[name]]) for name, value in default_options.items() if name in options)


def get_bao_baseline_fit_setup(tracer, zrange=tuple(), observable=None, recon=None, iso=None):
    if iso is None:
        if tracer.startswith('BGS') or (tracer.startswith('ELG') and 'LRG' not in tracer and tuple(zrange) == (0.8, 1.1)) or tracer.startswith('QSO'):
            iso = True
        else:
            iso = False
    if iso:
        template = 'bao-qiso'
    else:
        template = 'bao-qisoqap'
    if recon is None:
        recon = True
    if observable is None:
        observable = 'correlation'
    post = '-post' if recon else ''
    lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template + post, return_list=['lim', 'sigmas'])
    if 'correlation' in observable:
        broadband = 'pcs2'
    else:
        broadband = 'pcs'
    toret = get_baseline_2pt_setup(tracer=tracer, zrange=zrange, observable=observable, recon=recon)
    toret.update({'template': template, 'theory': 'dampedbao', 'observable': observable, 'lim': lim, 'sigmas': sigmas, 'dbeta': None, 'broadband': broadband, 'tracer': tracer, 'zrange': tuple(zrange), 'region': 'GCcomb', 'covmatrix': 'rascalc' if 'corr' in observable else 'thecov', 'cut': None})
    return toret


def get_fs_baseline_fit_setup(tracer, zrange=tuple(), observable=None):
    if observable is None:
        observable = 'power'
    lim = get_fit_setup(tracer, theory_name='velocileptors', observable_name=observable, return_list='lim')
    return {'theory': 'reptvelocileptors', 'observable': observable, 'lim': lim, 'freedom': 'max', 'tracer': tracer, 'zrange': tuple(zrange), 'weighting': 'default_FKP', 'covmatrix': 'rascalc' if 'corr' in observable else 'ezmock', 'syst': 'rotation-hod-photo', 'wmatrix': 'rotated', 'cut': ('theta', 0.05)}


def get_data_file_manager(conf='unblinded', **kwargs):

    file_manager = FileManager(environ={name: os.environ.get(name, '.') for name in ['DESICFS']}, **kwargs)
    if not conf: conf = ''
    list_smoothing_radius = {'BGS_BRIGHT-21.5': [10., 15.], 'LRG': [10., 15.], 'LRG+ELG_LOPnotqso': [10., 15.], 'ELG_LOPnotqso': [10., 15.], 'QSO': [20., 30.]}

    for tracer, zrange in list_zrange.items():

        for version in ['test', 'v1', 'v1.1', 'v1.2', 'v1.3', 'v1.4', 'v1.5'][3:]:
            
            if version not in ['v1.2', 'v1.3', 'v1.4', 'v1.5'] and tracer == 'LRG+ELG_LOPnotqso':
                continue

            if version == 'test':
                list_nran = {'BGS_BRIGHT-21.5': 1, 'LRG': 5, 'LRG+ELG_LOPnotqso': 6, 'ELG_LOPnotqso': 6, 'QSO': 3}
            else:
                list_nran = {'BGS_BRIGHT-21.5': 1, 'LRG': 8, 'LRG+ELG_LOPnotqso': 10, 'ELG_LOPnotqso': 10, 'QSO': 4}

            #if tracer != 'LRG': continue
            base_options = {'region': ['NGC', 'SGC'], 'version': [version], 'tracer': tracer}
            nran = list_nran[tracer]
            catalog_dir = Path('${DESICFS}/survey/catalogs/Y1/LSS/iron/LSScats/{version}')
            #catalog_dir = Path('/dvs_ro/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/{version}')
            if version not in ['v1.3', 'v1.4', 'v1.5']:
                catalog_dir = catalog_dir / conf
            if version == 'test':
                base_dir = Path('${DESICFS}/survey/catalogs/Y1/LSS/iron/LSScats/{version}') / conf
                data_dir = base_dir
                baseline_data_dir = base_dir / 'baseline_2pt/'
                base_fits_dir = base_dir / 'fits'
            else:
                base_dir = Path('${DESICFS}/survey/catalogs/Y1/LSS/iron/LSScats/{version}') / conf / 'desipipe' #/ 'test2'
                data_dir = base_dir / '2pt'
                baseline_data_dir = base_dir / 'baseline_2pt'
                base_fits_dir = base_dir / 'fits_2pt' #/ 'new_model'
                if version == 'v1.2':
                    base_cov_dir = Path('${DESICFS}/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/desipipe/cov_2pt')
                    base_forfit_dir = Path('${DESICFS}/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/desipipe/forfit_2pt')
                else:
                    base_cov_dir = Path('${DESICFS}/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt')
                    base_forfit_dir = Path('${DESICFS}/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt')

            # Catalogs
            file_manager.append(dict(description='Y1 data catalogs',
                                     id='catalog_data_y1',
                                     filetype='catalog',
                                     path=catalog_dir / '{tracer}_{region}_clustering.dat.fits',
                                     options=base_options))

            file_manager.append(dict(description='Y1 randoms catalogs',
                                     id='catalog_randoms_y1',
                                     filetype='catalog',
                                     path=catalog_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                     options={**base_options, 'iran': range(0, nranmax)}))

            recon_options = {**base_options, 'mode': ['recsym'], 'algorithm': ['IFFT'], 'recon_zrange': [None] + zrange, 'recon_weighting': ['default', 'default_FKP'], 'smoothing_radius': list_smoothing_radius[tracer]}
            recon_base = 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}{recon_zrange}{recon_weighting}/'
            recon_dir = data_dir / recon_base
            baseline_recon_dir = baseline_data_dir / 'recon_{mode}{recon_zrange}{recon_weighting}/'
            recon_foptions = {}
            recon_foptions['recon_zrange'] = ['' if zr is None else '_z{zrange[0]:.1f}-{zrange[1]:.1f}'.format(zrange=zr) for zr in recon_options['recon_zrange']]
            recon_foptions['recon_weighting'] = ['' if w == 'default' else '_{}'.format(w) for w in recon_options['recon_weighting']]
            file_manager.append(dict(description='Y1 data catalogs post-reconstruction',
                                     id='catalog_data_recon_y1',
                                     filetype='catalog',
                                     path=recon_dir / '{tracer}_{region}_clustering.dat.fits',
                                     link=baseline_recon_dir / '{tracer}_{region}_clustering.dat.fits',
                                     options={**recon_options, 'mode': ['recsym', 'reciso']},
                                     foptions=recon_foptions))

            file_manager.append(dict(description='Y1 randoms catalogs post-reconstruction',
                                     id='catalog_randoms_recon_y1',
                                     filetype='catalog',
                                     path=recon_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                     link=baseline_recon_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                     options={**recon_options, 'mode': ['recsym', 'reciso'], 'iran': range(0, nranmax)},
                                     foptions=recon_foptions))

            meas_options = {**base_options, 'zrange': zrange, 'region': ['NGC', 'SGC', 'GCcomb', 'N', 'S'], 'weighting': weighting, 'binning': ['lin'], 'cut': {None: '', ('rp', 2.5): '_rpcut2.5', ('theta', 0.05): '_thetacut0.05'}}
            #corr_options = {**meas_options, 'nran': [nran] + ([min(nran, 4)] if nran != 4 else []), 'split': [20., None], 'njack': [0, 60]}
            #corr_options = {**meas_options, 'nran': [nran], 'split': [20., None], 'njack': [0, 60]}
            #corr_options = {**meas_options, 'nran': [nran, 2], 'split': [20., None], 'njack': [0, 60]}
            #corr_foptions = {'split': ['_split20', '_splitinf']}
            corr_options = {**meas_options, 'nran': [nran], 'split': [20.], 'njack': [0, 60]}
            corr_foptions = {'split': ['_split20']}
            file_manager.append(dict(description='Correlation functions smu of blinded data',
                                     id='correlation_y1',
                                     filetype='correlation',
                                     path=data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}{split}{cut}.npy',
                                     link=baseline_data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=corr_options,
                                     foptions=corr_foptions))

            power_options = {**meas_options, 'cellsize': [6.], 'boxsize': [list_boxsize[tracer]], 'nran': [nranmax]}
            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_y1',
                                     filetype='power',
                                     path=data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=power_options))

            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_rotated_y1',
                                     filetype='power',
                                     path=data_dir / 'pk/rotated/pkpoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'pk/rotated/pkpoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=power_options))
            
            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_corrected_y1',
                                     filetype='power',
                                     path=data_dir / 'pk/corrected/pkpoles_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'pk/corrected/pkpoles_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=power_options))

            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_rotated_corrected_y1',
                                     filetype='power',
                                     path=data_dir / 'pk/rotated_corrected/pkpoles_rotated_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'pk/rotated_corrected/pkpoles_rotated_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=power_options))
            
            #for zr, recon_zr in zip([zrange, [None]], [[None], recon_options['recon_zrange']]):
            for zr, recon_zr in zip([zrange] + [[zr] for zr in zrange], [[None]] + [[zr] for zr in zrange]):
                corr_recon_options = {**recon_options, **corr_options, 'mode': ['recsym', 'reciso'], 'zrange': zr, 'recon_zrange': recon_zr}
                recon_foptions['zrange'] = ['' if zr is None else '_z{zrange[0]:.1f}-{zrange[1]:.1f}'.format(zrange=zr) for zr in corr_recon_options['zrange']]
                recon_foptions['recon_zrange'] = ['' if zr is None else '_z{zrange[0]:.1f}-{zrange[1]:.1f}'.format(zrange=zr) for zr in corr_recon_options['recon_zrange']]
                file_manager.append(dict(description='Correlation functions smu of blinded data post-reconstruction',
                                         id='correlation_recon_y1',
                                         filetype='correlation',
                                         path=recon_dir / 'xi/smu/allcounts_{tracer}_{region}{zrange}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}{split}{cut}.npy',
                                         link=baseline_recon_dir / 'xi/smu/allcounts_{tracer}_{region}{zrange}{cut}.npy',
                                         options=corr_recon_options,
                                         foptions={**corr_foptions, **recon_foptions}))

                power_recon_options = {**recon_options, **power_options, 'mode': ['recsym', 'reciso'], 'zrange': zr, 'recon_zrange': recon_zr}
                file_manager.append(dict(description='Power spectrum multipoles of blinded data post-reconstruction',
                                         id='power_recon_y1',
                                         filetype='power',
                                         path=recon_dir / 'pk/pkpoles_{tracer}_{region}{zrange}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_recon_dir / 'pk/pkpoles_{tracer}_{region}{zrange}{cut}.npy',
                                         options=power_recon_options,
                                         foptions=recon_foptions))

            # pk window
            file_manager.append(dict(description='Power spectrum window function of blinded data',
                                     id='window_power_y1',
                                     path=data_dir / 'pk/window_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}_boxscale{boxscale:.0f}{cut}.npy',
                                     options={**power_options, 'boxscale': [20., 5., 1.]}))

            file_manager.append(dict(description='Power spectrum window matrix of blinded data',
                                     id='wmatrix_power_y1',
                                     filetype='wmatrix',
                                     path=data_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=power_options))

            file_manager.append(dict(description='Rotation of power spectrum window function',
                                     id='rotation_wmatrix_power_y1',
                                     path=data_dir / 'pk/rotated/rotation_wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     options=power_options))

            file_manager.append(dict(description='Rotated power spectrum window matrix of blinded data',
                                     id='wmatrix_power_rotated_y1',
                                     filetype='wmatrix',
                                     path=data_dir / 'pk/rotated/wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'pk/rotated/wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=power_options))
            
            if version in ['v1.2', 'v1.5']:  #CHANGED
                observables = ['power', 'power+correlation-recon', 'power+bao-recon', 'shapefit+bao-recon']
                for observable in observables:
                    forfit_options = {**base_options, 'zrange': zrange, 'region': 'GCcomb', 'observable': observable, 'syst': '', 'klim': {None: ''}}
                    forfit_foptions = {}
                    if 'power' in observable:
                        forfit_options['syst'] = {'rotation': '_syst-rotation', 'rotation-hod-photo': '_syst-rotation-hod-photo', 'hod-photo': '_syst-hod-photo', 'hod': '_syst-hod', '': '_syst-no'}
                    if 'power' in observable or 'shapefit' in observable:
                        forfit_options['klim'] = [{0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]}]
                        if 'power' in observable:
                            forfit_options['klim'].append({0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005], 4: [0.02, 0.2, 0.005]})
                        tlim = '{:d}-{:.2f}-{:.2f}'
                        forfit_foptions['klim'] = ['_klim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in forfit_options['klim']]
                    file_manager.append(dict(description='Covariance matrix, observable and window matrix for fit',
                                             author='',
                                             id='forfit_y1',
                                             filetype='covariance',
                                             path=base_forfit_dir / 'forfit_{observable}{syst}{klim}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                             options=forfit_options,
                                             foptions=forfit_foptions))
                forfit_options['tracer'], forfit_options['zrange'] = 'Lya', (1.8, 4.2)
                forfit_options['observable'] = 'bao-recon'
                forfit_options.pop('klim', None)
                file_manager.append(dict(description='Covariance matrix, observable and window matrix for fit',
                                             author='',
                                             id='forfit_y1',
                                             filetype='covariance',
                                             path=base_forfit_dir / 'forfit_{observable}{syst}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                             options=forfit_options))
                
                # 2-pt covariances
                sources = ['rascalc', 'thecov', 'ezmock', 'hybrid'][:-1]
                if 'LRG+ELG' in tracer:
                    sources = sources[:2]
                for source in sources:
                    if source == 'ezmock':
                        cov_version = 'v1'
                        observables = ['power', 'power-recon', 'correlation', 'correlation-recon', 'power+correlation-recon', 'power+bao-recon', 'shapefit+bao-recon', 'power+correlation']
                    elif source == 'hybrid':
                        cov_version = 'v1'
                        observables = ['power+correlation-recon', 'power+bao-recon']
                    else:
                        cov_version = 'v1.2'
                        if source == 'rascalc':
                            observables = ['correlation', 'correlation-recon']
                        else:
                            observables = ['power', 'power-recon']
                    for observable in observables:
                        cov_options = {**meas_options, 'region': 'GCcomb' if source in ['thecov', 'hybrid'] or any(name in observable for name in ['bao-recon', 'shapefit']) else ['NGC', 'SGC', 'GCcomb'], 'weighting': 'default_FKP', 'source': [source], 'version': cov_version, 'observable': observable}
                        if source == 'rascalc':
                            if version == 'v1.2':
                                cov_options['version'] = 'v1.2' if 'recon' in observable else 'v0.6'
                            else:
                                cov_options['version'] = 'v1.5'
                        cov_options['cut'] = {None: '', ('theta', 0.05): '_thetacut0.05'}
                        if not any(name in observable.split('+') for name in ['power', 'correlation']):
                            cov_options['cut'] = {None: ''}

                        cov_options['klim'] = {None: ''}
                        cov_foptions = {}
                        if 'shapefit' in observable:
                            cov_options['klim'] = [{0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]}, {0: [0.02, 0.12, 0.005], 2: [0.02, 0.12, 0.005]}][:1]
                            tlim = '{:d}-{:.2f}-{:.2f}'
                            cov_foptions['klim'] = ['_klim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in cov_options['klim']]
                        file_manager.append(dict(description='Covariance matrix',
                                                 author='mrash, oalves',
                                                 id='covariance_y1',
                                                 filetype='covariance',
                                                 path=base_cov_dir / '{source}/{version}/covariance_{observable}{klim}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                 options=cov_options,
                                                 foptions=cov_foptions))
                        
                        if 'power' in observable:
                            #pk_cov_options = {**cov_options, 'marg': {False: '', 'rotation': '_marg-rotation'}}
                            file_manager.append(dict(description='Covariance matrix',
                                                     author='mrash, oalves',
                                                     id='covariance_rotated_y1',
                                                     filetype='covariance',
                                                     path=base_cov_dir / '{source}/{version}/rotated/covariance_rotated_marg-no_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                     options=cov_options))

                            if observable != 'power': continue
                            cov_options['source'] = ['syst']
                            cov_options['syst'] = ['rotation', 'photo', 'hod']
                            cov_options['version'] = 'v1.5'
                            #cov_options.pop('marg')
                            file_manager.append(dict(description='Covariance matrix',
                                                     author='ntbfin',
                                                     id='covariance_syst_y1',
                                                     filetype='covariance',
                                                     path=base_cov_dir / '{source}/{version}/covariance_syst-{syst}_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                     options=cov_options))
                            file_manager.append(dict(description='Covariance matrix',
                                                     author='ntbfin',
                                                     id='covariance_syst_rotated_y1',
                                                     filetype='covariance',
                                                     path=base_cov_dir / '{source}/{version}/rotated/covariance_syst-{syst}_rotated_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                     options=cov_options))

                template_options = {**meas_options, 'syst': ['ric', 'aic', 'photo'], 'observable': ['power']}
                file_manager.append(dict(description='Systematic template',
                                         id='template_syst_y1',
                                         filetype='template',
                                         path=base_dir / 'templates_2pt/template_{syst}_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                         options=template_options))

                file_manager.append(dict(description='Systematic template',
                                         id='template_syst_rotated_y1',
                                         filetype='template',
                                         path=base_dir / 'templates_2pt/rotated/template_rotated_{syst}_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                         options=template_options))

            footprint_options = {**base_options, 'zrange': zrange, 'region': ['NGC', 'SGC', 'GCcomb']}
            file_manager.append(dict(description='Footprint of input data',
                                     id='footprint_y1',
                                     path=base_fits_dir / 'footprints/footprints_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                     options=footprint_options))

            for observable in ['correlation', 'power', 'power+bao-recon']:
                # full shape fits
                fits_dir = base_fits_dir / 'fits_{observable}_{theory}_{template}_freedom-{freedom}{emulator}'
                if observable == 'correlation': tlim = '{:d}-{:.0f}-{:.0f}'
                else: tlim = '{:d}-{:.2f}-{:.2f}'
                lim = get_fit_setup(tracer, observable_name=observable, return_list='lim')                
                profile_options = {**meas_options, 'observable': [observable], 'theory': ['velocileptors', 'reptvelocileptors', 'folpsax'], 'template': ['shapefit-qisoqap', 'direct_ns-fixed', 'direct', 'fixed'], 'freedom': ['max'], 'covmatrix': ['thecov' if 'power' in observable else 'rascalc', 'ezmock'], 'wmatrix': [''], 'syst': ['', 'photo'], 'lim': [lim], 'emulator': {False: '_noemu', True: ''}}
                #if observable == 'power': profile_options['lim'].append({0: [0.02, 0.12, 0.005], 2: [0.02, 0.12, 0.005]})
                foptions = {**{}, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']]}, 'wmatrix': [''], 'syst': ['', '_syst-photo']}
                if 'power' in observable:
                    profile_options['wmatrix'] += ['rotated']
                    foptions['wmatrix'] += ['_wmat-rotated']
                    profile_options['syst'] += ['hod', 'hod-photo', 'rotation-hod-photo']
                    foptions['syst'] += ['_syst-hod', '_syst-hod-photo', '_syst-rotation-hod-photo']
                        
                file_manager.append(dict(description='Full shape fits (profiles) to blinded data',
                                         id='profiles_full_shape_y1',
                                         filetype='profiles',
                                         path=fits_dir / 'profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{lim}.npy',
                                         options=profile_options,
                                         foptions=foptions))
                chain_options = {**profile_options, 'ichain': range(8)}
                file_manager.append(dict(description='Full shape fits (chains) to blinded data',
                                         id='chains_full_shape_y1',
                                         filetype='chain',
                                         path=fits_dir / 'chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{lim}_{ichain:d}.npy',
                                         options=chain_options,
                                         foptions=foptions))
                file_manager.append(dict(description='Full shape fits (chains) to blinded data',
                                         id='chains_full_shape_importance_y1',
                                         filetype='chain',
                                         path=fits_dir / 'chain_importance_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{lim}_{ichain:d}.npy',
                                         options=chain_options,
                                         foptions=foptions))
                emulator_options = {name: value for name, value in profile_options.items() if name not in ['region', 'weighting', 'binning', 'cut']}
                file_manager.append(dict(description='Emulator for full shape fits to blinded data',
                                         id='emulator_full_shape_y1',
                                         path=fits_dir / 'emulator_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                         options=emulator_options))
                # bao fits
                fits_dir = base_fits_dir / 'fits_{observable}_{theory}_{template}_{broadband}'
                for template in ['bao-qisoqap', 'bao-qiso', 'bao-now-qiso']:
                    lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template, return_list=['lim', 'sigmas'])
                    profile_options = {**meas_options, 'observable': [observable], 'theory': ['dampedbao'], 'template': [template], 'broadband': ['power3', 'pcs' if observable == 'power' else 'pcs2', 'fixed'], 'lim': [lim], 'sigmas': [sigmas, {key: None for key in sigmas}], 'covmatrix': ['thecov' if observable == 'power' else 'rascalc', 'ezmock'], 'dbeta': {None: '', (0.25, 1.75): '_dbeta-0.25-1.75'}}
                    foptions = {'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas'][:-1]] + ['sigmas-flat_sigmapar-flat_sigmaper-flat']}
                    file_manager.append(dict(description='BAO fits (profiles) to blinded data',
                                             id='profiles_bao_y1',
                                             filetype='profiles',
                                             path=fits_dir / 'profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}{dbeta}_{lim}.npy',
                                             options=profile_options,
                                             foptions=foptions))
                    chain_options = {**profile_options, 'ichain': range(8)}
                    file_manager.append(dict(description='BAO fits (chains) to blinded data',
                                             id='chains_bao_y1',
                                             filetype='chain',
                                             path=fits_dir / 'chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}{dbeta}_{lim}_{ichain:d}.npy',
                                             options=chain_options,
                                             foptions=foptions))

                    lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template + '-post', return_list=['lim', 'sigmas'])
                    profile_options = {**recon_options, **profile_options, 'lim': [lim], 'sigmas': [sigmas, {key: None for key in sigmas}]}
                    # bao fits post-reconstruction
                    foptions = {'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas'][:-1]] + ['sigmas-flat_sigmapar-flat_sigmaper-flat']}
                    file_manager.append(dict(description='BAO fits (profiles) to blinded data post-reconstruction',
                                             id='profiles_bao_recon_y1',
                                             filetype='profiles',
                                             path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}{dbeta}_{lim}.npy',
                                             options=profile_options,
                                             foptions=foptions))
                    chain_options = {**profile_options, 'ichain': range(8)}
                    file_manager.append(dict(description='BAO fits (chains) to blinded data post-reconstruction',
                                             id='chains_bao_recon_y1',
                                             filetype='chain',
                                             path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}{dbeta}_{lim}_{ichain:d}.npy',
                                             options=chain_options,
                                             foptions=foptions))

    return file_manager


def get_box_ez_file_manager(**kwargs):

    file_manager = FileManager(environ={name: os.environ.get(name, '.') for name in ['DESICFS']}, **kwargs)
    nmocks = 50
    list_zsnap = {'BGS_BRIGHT-21.5': [0.200], 'LRG': [0.500, 0.800, 1.100], 'ELG_LOPnotqso': [0.950, 1.100, 1.325], 'QSO': [1.100, 1.400, 1.700]}

    for tracer, zsnap in list_zsnap.items():

        versions = ['v1']

        for version in versions:
            base_options = {'version': version, 'tracer': tracer, 'zsnap': zsnap, 'imock': range(0, nmocks)}
            base_foptions = {}
            base_foptions = {'tracer': [{'BGS_BRIGHT-21.5': 'BGS', 'ELG_LOPnotqso': 'ELG'}.get(tracer, tracer)]}

            root_dir = Path('${DESICFS}')
            #root_dir = Path('/dvs_ro/cfs/cdirs/desi/')
            size = 2 if 'BGS' in tracer else 6
            catalog_dir = root_dir / 'cosmosim/SecondGenMocks/EZmock/CubicBox_{:d}Gpc/'.format(size) / '{tracer}/z{zsnap:.3f}'
            base_dir = Path('${DESICFS}') / 'cosmosim/SecondGenMocks/EZmock/CubicBox_{:d}Gpc/'.format(6) / 'desipipe/{version}'  # 2 is not writable

            data_dir = base_dir / '2pt/mock{imock:d}_los-{los}'
            merged_dir = base_dir.parent / 'wmatrix'
            baseline_data_dir = base_dir / 'baseline_2pt/mock{imock:d}_los-{los}'
            baseline_merged_dir = merged_dir
            base_fits_dir = base_dir / 'fits_2pt/mock{imock:d}_los-{los}'
            mean_fits_dir = base_dir / 'fits_2pt'

            path_data = catalog_dir / '{imock:04d}/EZmock_{tracer}_z{zsnap:.3f}_AbacusSummit_base_c000_ph000_{imock:04d}.{isub:d}.fits.gz'
            file_manager.append(dict(description='Y1 data box catalogs',
                                     id='catalog_data_box_ez_y1',
                                     filetype='catalog',
                                     path=path_data,
                                     options={**base_options, 'isub': range(64)},
                                     foptions=base_foptions))

            recon_options = {**base_options, 'los': ['x', 'y', 'z', None], 'mode': ['recsym'], 'algorithm': ['IFFT'], 'smoothing_radius': list_smoothing_radius[tracer]}
            file_manager.append(dict(description='Y1 data catalogs post-reconstruction',
                                     id='catalog_data_recon_box_ez_y1',
                                     filetype='catalog',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_clustering.dat.fits',
                                     options={**recon_options, 'mode': ['recsym', 'reciso']},
                                     foptions=base_foptions))

            file_manager.append(dict(description='Y1 randoms catalogs post-reconstruction',
                                     id='catalog_randoms_recon_box_ez_y1',
                                     filetype='catalog',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_{iran:d}_clustering.ran.fits',
                                     options={**recon_options, 'mode': ['recsym', 'reciso'], 'iran': range(0, nranmax)},
                                     foptions=base_foptions))

            meas_options = {**base_options, 'boxsize': [size * 1000.], 'los': ['x', 'y', 'z', None], 'binning': ['lin']}
            corr_options = {**meas_options}
            file_manager.append(dict(description='Correlation functions smu of blinded data',
                                     id='correlation_box_ez_y1',
                                     filetype='correlation',
                                     path=data_dir / 'xi/smu/allcounts_{tracer}_z{zsnap:.4f}_{binning}.npy',
                                     link=baseline_data_dir / 'xi/smu/allcounts_{tracer}_z{zsnap:.4f}_{binning}.npy',
                                     options=corr_options,
                                     foptions=base_foptions))

            power_options = {**meas_options, 'cellsize': [4.]}
            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_box_ez_y1',
                                     filetype='power',
                                     path=data_dir / 'pk/pkpoles_{tracer}_z{zsnap:.4f}_{binning}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}.npy',
                                     link=baseline_data_dir / 'pk/pkpoles_{tracer}_z{zsnap:.4f}.npy',
                                     options=power_options,
                                     foptions=base_foptions))

            corr_recon_options = {**recon_options, **corr_options, 'nran': [nranmax]}
            file_manager.append(dict(description='Correlation functions smu of blinded data post-reconstruction',
                                     id='correlation_recon_box_ez_y1',
                                     filetype='correlation',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/xi/smu/allcounts_{tracer}_z{zsnap:.4f}_{binning}_nran{nran:d}_split{split:.0f}.npy',
                                     link=baseline_data_dir / 'recon_{mode}/xi/smu/allcounts_{tracer}_z{zsnap:.4f}.npy',
                                     options=corr_recon_options,
                                     foptions=base_foptions))

            power_recon_options = {**recon_options, **power_options, 'nran': [nranmax]}
            file_manager.append(dict(description='Power spectrum multipoles of blinded data post-reconstruction',
                                     id='power_recon_box_ez_y1',
                                     filetype='power',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/pk/pkpoles_{tracer}_z{zsnap:.4f}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}.npy',
                                     link=baseline_data_dir / 'recon_{mode}/pk/pkpoles_{tracer}_z{zsnap:.4f}.npy',
                                     options=power_recon_options,
                                     foptions=base_foptions))

            # pk window
            merged_power_options = dict(power_options)
            merged_power_options.pop('imock')
            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_merged_box_ez_y1',
                                     filetype='power',
                                     path=merged_dir / 'pkpoles_{tracer}_z{zsnap:.4f}_{binning}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}.npy',
                                     options=merged_power_options,
                                     foptions=base_foptions))

            file_manager.append(dict(description='Power spectrum window matrix of blinded data',
                                     id='wmatrix_power_merged_box_ez_y1',
                                     filetype='wmatrix',
                                     path=merged_dir / 'wmatrix_{binning}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}.npy',
                                     link=baseline_merged_dir / 'wmatrix.npy',
                                     options=merged_power_options,
                                     foptions=base_foptions))

    return file_manager


def get_ez_file_manager(**kwargs):

    file_manager = FileManager(environ={name: os.environ.get(name, '.') for name in ['DESICFS']}, **kwargs)
    #list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'LRG+ELG_LOPnotqso': [(0.8, 1.1)], 'QSO': [(0.8, 2.1)]}
    #list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)]}
    for tracer, zrange in list_zrange.items():

        #if tracer != 'LRG': continue
        nran = list_nran[tracer]

        versions = ['v1', 'v1ric', 'v1noric']

        for version in versions:

            list_fa = ['ffa']

            for fa in list_fa:

                root_dir = Path('${DESICFS}/survey/catalogs/Y1/mocks/SecondGenMocks/')
                #root_dir = Path('/dvs_ro/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/')
                #root_dir = Path('/pscratch/sd/a/adematti/desipipe')
                if 'BGS' in tracer:
                    base_dir = root_dir / 'EZmock/desipipe/BGS_{version}/{fa}'
                else:
                    base_dir = root_dir / 'EZmock/desipipe/{version}/{fa}'
                #data_dir = base_dir / '2pt/mock{imock:d}'
                data_dir = base_dir / '2pt/mock{imock:d}'
                merged_dir = base_dir / '2pt/merged'
                #merged_dir = base_dir / '2pt/merged2'
                base_cov_dir = Path(str(base_dir).replace('{version}', 'v1')) / 'cov_2pt'
                base_forfit_dir = base_cov_dir.parent / 'forfit_2pt'

                baseline_data_dir = base_dir / 'baseline_2pt/mock{imock:d}'
                baseline_merged_dir = base_dir / 'baseline_2pt/merged'
                base_fits_dir = base_dir / 'fits_2pt/mock{imock:d}'
                mean_fits_dir = base_dir / 'fits_2pt'

                root_dir = Path('${DESICFS}')
                #root_dir = Path('/dvs_ro/cfs/cdirs/desi/')
                # Catalogs
                if fa == 'ffa':
                    if 'BGS' in tracer:
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/FFA_BGS/mock{imock:d}/'
                    else:
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/FFA/mock{imock:d}/'
                    path_data = catalog_dir / '{tracer}_ffa_{region}_clustering.dat.fits'
                    path_randoms = catalog_dir / '{tracer}_ffa_{region}_{iran:d}_clustering.ran.fits'

                nmocks = 1000
                for ric in ['ric', 'noric']:
                    if version == 'v1' + ric:
                        nmocks = 100
                        #catalog_dir = Path(str(base_dir / 'noric').replace('/global/', '/dvs_ro/')) / 'mock{imock:d}'
                        if 'BGS' in tracer:
                            #catalog_dir = Path('/pscratch/sd/a/adematti/desipipe/EZmock/desipipe/BGS_v1/ffa/{}/mock{{imock:d}}'.format(ric))
                            catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/desipipe/BGS_v1/ffa/{}/mock{{imock:d}}'.format(ric)
                        else:
                            #catalog_dir = Path('/pscratch/sd/a/adematti/desipipe/EZmock/desipipe/v1/ffa/{}/mock{{imock:d}}'.format(ric))
                            catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/desipipe/v1/ffa/{}/mock{{imock:d}}'.format(ric)
                        path_randoms = catalog_dir / '{tracer}_ffa_{region}_{iran:d}_clustering.ran.fits'

                base_options = {'region': ['NGC', 'SGC'], 'version': [version], 'tracer': tracer, 'fa': [fa], 'imock': range(1, nmocks + 1)}
                base_foptions = {}
                if fa in ['complete', 'ffa']:
                    base_foptions = {'tracer': [{'BGS_BRIGHT-21.5': 'BGS', 'ELG_LOPnotqso': 'ELG_LOP', 'LRG+ELG_LOPnotqso': 'LRG+ELG_LOP'}.get(tracer, tracer)]}

                file_manager.append(dict(description='Y1 data catalogs',
                                         id='catalog_data_ez_y1',
                                         filetype='catalog',
                                         path=path_data,
                                         options=base_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Y1 randoms catalogs',
                                         id='catalog_randoms_ez_y1',
                                         filetype='catalog',
                                         path=path_randoms,
                                         options={**base_options, 'iran': range(0, nranmax)},
                                         foptions=base_foptions))

                recon_options = {**base_options, 'mode': ['recsym'], 'algorithm': ['IFFT'], 'smoothing_radius': list_smoothing_radius[tracer]}
                file_manager.append(dict(description='Y1 data catalogs post-reconstruction',
                                         id='catalog_data_recon_ez_y1',
                                         filetype='catalog',
                                         path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_{region}_clustering.dat.fits',
                                         link=baseline_data_dir / 'recon_{mode}/{tracer}_{region}_clustering.dat.fits',
                                         options={**recon_options, 'mode': ['recsym', 'reciso']},
                                         foptions=base_foptions))

                file_manager.append(dict(description='Y1 randoms catalogs post-reconstruction',
                                         id='catalog_randoms_recon_ez_y1',
                                         filetype='catalog',
                                         path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                         link=baseline_data_dir / 'recon_{mode}/{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                         options={**recon_options, 'mode': ['recsym', 'reciso'], 'iran': range(0, nran)},
                                         foptions=base_foptions))

                meas_options = {**base_options, 'zrange': zrange, 'region': ['NGC', 'SGC', 'GCcomb'], 'weighting': 'default_FKP', 'binning': ['lin'], 'cut': {None: '', ('theta', 0.05): '_thetacut0.05'}}
                corr_options = {**meas_options, 'nran': [nran], 'split': [20.], 'njack': [0, 60]}
                file_manager.append(dict(description='Correlation functions smu of blinded data',
                                         id='correlation_ez_y1',
                                         filetype='correlation',
                                         path=data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=corr_options,
                                         foptions=base_foptions))

                power_options = {**meas_options, 'cellsize': [6.], 'boxsize': [list_boxsize[tracer]], 'nran': [nran]}
                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_ez_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_rotated_ez_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/rotated/pkpoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/rotated/pkpoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_corrected_ez_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/corrected/pkpoles_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/corrected/pkpoles_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))
                
                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_rotated_corrected_ez_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/rotated_corrected/pkpoles_rotated_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/rotated_corrected/pkpoles_rotated_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))

                corr_recon_options = {**recon_options, **corr_options, 'cut': {None: ''}}
                file_manager.append(dict(description='Correlation functions smu of blinded data post-reconstruction',
                                         id='correlation_recon_ez_y1',
                                         filetype='correlation',
                                         path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'recon_{mode}/xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=corr_recon_options,
                                         foptions=base_foptions))

                power_recon_options = {**recon_options, **power_options, 'cut': {None: ''}}
                file_manager.append(dict(description='Power spectrum multipoles of blinded data post-reconstruction',
                                         id='power_recon_ez_y1',
                                         filetype='power',
                                         path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'recon_{mode}/pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_recon_options,
                                         foptions=base_foptions))

                # pk window
                merged_options = dict(base_options)
                merged_options.pop('imock')
                file_manager.append(dict(description='Y1 data catalogs',
                                         id='catalog_data_merged_ez_y1',
                                         filetype='catalog',
                                         path=merged_dir / '{tracer}_{region}_clustering.dat.fits',
                                         options=merged_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Y1 randoms catalogs',
                                         id='catalog_randoms_merged_ez_y1',
                                         filetype='catalog',
                                         path=merged_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                         options={**merged_options, 'iran': range(0, nranmax)},
                                         foptions=base_foptions))

                merged_power_options = dict(power_options)
                merged_power_options.pop('imock')
                merged_power_options['nran'] = nranmax
                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_merged_ez_y1',
                                         filetype='power',
                                         path=merged_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum window function of blinded data',
                                         id='window_power_merged_ez_y1',
                                         path=merged_dir / 'pk/window_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}_boxscale{boxscale:.0f}{cut}.npy',
                                         options={**merged_power_options, 'boxscale': [20., 5., 1.]},
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum window matrix of blinded data',
                                         id='wmatrix_power_merged_ez_y1',
                                         filetype='wmatrix',
                                         path=merged_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_merged_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Rotation of power spectrum window function',
                                         id='rotation_wmatrix_power_merged_ez_y1',
                                         path=merged_dir / 'pk/rotated/rotation_wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Rotated power spectrum window matrix of blinded data',
                                         id='wmatrix_power_merged_rotated_ez_y1',
                                         filetype='wmatrix',
                                         path=merged_dir / 'pk/rotated/wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_merged_dir / 'pk/rotated/wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                if version == 'v1':
                    observables = ['power']
                    for observable in observables:
                        forfit_options = {**base_options, 'zrange': zrange, 'region': 'GCcomb', 'observable': observable, 'syst': '', 'klim': {None: ''}}
                        forfit_options.pop('imock')
                        forfit_foptions = dict(base_foptions)
                        if 'power' in observable:
                            forfit_options['syst'] = {'rotation': '_syst-rotation', 'rotation-hod-photo': '_syst-rotation-hod-photo'}
                            forfit_options['klim'] = [{0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]}, {0: [0.02, 0.12, 0.005], 2: [0.02, 0.12, 0.005]}]
                            tlim = '{:d}-{:.2f}-{:.2f}'
                            forfit_foptions['klim'] = ['_klim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in forfit_options['klim']]
                        file_manager.append(dict(description='Covariance matrix, observable and window matrix for fit',
                                                 author='',
                                                 id='forfit_ez_y1',
                                                 filetype='covariance',
                                                 path=base_forfit_dir / 'forfit_{observable}{syst}{klim}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                                 options=forfit_options,
                                                 foptions=forfit_foptions))

                        # 2-pt covariances
                        sources = ['rascalc', 'thecov', 'ezmock']
                        if 'LRG+ELG' in tracer:
                            sources = sources[:2]
                        for source in sources:
                            if source == 'ezmock':
                                cov_version = 'v1'
                                observables = ['power', 'power-recon', 'correlation', 'correlation-recon', 'power+correlation-recon', 'power+bao-recon']
                            else:
                                cov_version = 'v1.2'
                                if source == 'rascalc':
                                    observables = ['correlation', 'correlation-recon']
                                else:
                                    observables = ['power', 'power-recon']
                            for observable in observables:
                                cov_options = {**meas_options, 'region': 'GCcomb' if source in ['thecov', 'hybrid'] or any(name in observable for name in ['bao-recon', 'shapefit']) else ['NGC', 'SGC', 'GCcomb'], 'weighting': 'default_FKP', 'source': [source], 'version': cov_version, 'observable': observable}
                                cov_options.pop('imock')
                                if source == 'rascalc': cov_options['version'] = 'v1.5'  #'v1.2' if 'recon' in observable else 'v0.6'
                                cov_options['cut'] = {None: '', ('theta', 0.05): '_thetacut0.05'}
                                if not any(name in observable.split('+') for name in ['power', 'correlation']):
                                    cov_options['cut'] = {None: ''}

                                file_manager.append(dict(description='Covariance matrix',
                                                         author='mrash, oalves',
                                                         id='covariance_ez_y1',
                                                         filetype='covariance',
                                                         path=base_cov_dir / '{source}/{version}/covariance_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                         options=cov_options,
                                                         foptions=base_foptions))

                                if 'power' in observable:
                                    #cov_options = {**cov_options, 'cut': {('theta', 0.05): '_thetacut0.05'}, 'marg': {False: '', 'rotation': '_marg-rotation', 'rotation-syst': '_marg-rotation-syst'}}
                                    cov_options = {**cov_options, 'cut': {('theta', 0.05): '_thetacut0.05'}}
                                    file_manager.append(dict(description='Covariance matrix',
                                                             author='mrash, oalves',
                                                             id='covariance_rotated_ez_y1',
                                                             filetype='covariance',
                                                             path=base_cov_dir / '{source}/{version}/rotated/covariance_rotated_marg-no_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                             options=cov_options,
                                                             foptions=base_foptions))

                                if observable != 'power': continue
                                cov_options['source'] = ['syst']
                                cov_options['syst'] = ['rotation', 'photo', 'hod']
                                cov_options['version'] = 'v1'
                                #cov_options.pop('marg')
                                file_manager.append(dict(description='Covariance matrix',
                                                         author='ntbfin',
                                                         id='covariance_syst_ez_y1',
                                                         filetype='covariance',
                                                         path=base_cov_dir / '{source}/{version}/covariance_syst-{syst}_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                         options=cov_options,
                                                         foptions=base_foptions))
                                file_manager.append(dict(description='Covariance matrix',
                                                         author='ntbfin',
                                                         id='covariance_syst_rotated_ez_y1',
                                                         filetype='covariance',
                                                         path=base_cov_dir / '{source}/{version}/rotated/covariance_syst-{syst}_rotated_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                         options=cov_options,
                                                         foptions=base_foptions))

                    template_options = {**meas_options, 'syst': ['ric', 'aic', 'photo'], 'observable': ['power']}
                    template_options.pop('imock')
                    file_manager.append(dict(description='Systematic template',
                                             id='template_syst_ez_y1',
                                             filetype='template',
                                             path=base_dir / 'templates_2pt/template_{syst}_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                             options=template_options,
                                             foptions=base_foptions))

                    file_manager.append(dict(description='Systematic template',
                                             id='template_syst_rotated_ez_y1',
                                             filetype='template',
                                             path=base_dir / 'templates_2pt/rotated/template_rotated_{syst}_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                             options=template_options,
                                             foptions=base_foptions))


                    file_manager.append(dict(description='RIC of power spectrum',
                                             id='ric_wmatrix_power_merged_ez_y1',
                                             path=merged_dir / 'pk/ric/ric_wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                             options=merged_power_options,
                                             foptions=base_foptions))

                    file_manager.append(dict(description='Power spectrum window matrix of blinded data with RIC',
                                             id='wmatrix_ric_power_merged_ez_y1',
                                             filetype='wmatrix',
                                             path=merged_dir / 'pk/ric/wmatrix_ric_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                             link=baseline_merged_dir / 'pk/wmatrix_ric_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                             options=merged_power_options,
                                             foptions=base_foptions))

                    file_manager.append(dict(description='Rotation of power spectrum window function',
                                             id='rotation_wmatrix_ric_power_merged_ez_y1',
                                             path=merged_dir / 'pk/rotated/rotation_wmatrix_ric_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                             options=merged_power_options,
                                             foptions=base_foptions))

                    file_manager.append(dict(description='Rotated power spectrum window matrix of blinded data',
                                             id='wmatrix_ric_power_merged_rotated_ez_y1',
                                             filetype='wmatrix',
                                             path=merged_dir / 'pk/rotated/wmatrix_ric_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                             link=baseline_merged_dir / 'pk/rotated/wmatrix_ric_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                             options=merged_power_options,
                                             foptions=base_foptions))

            for mean in [True, False]:
                tmp_meas_options = dict(meas_options)
                tmp_recon_options = dict(recon_options)
                tmp_base_fits_dir = base_fits_dir
                if mean:
                    tmp_meas_options.pop('imock')
                    tmp_recon_options.pop('imock')
                    tmp_meas_options['precscale'] = tmp_recon_options['precscale'] = [1, 1000]  #nmocks]
                    tmp_base_fits_dir = mean_fits_dir / 'mean_precscale{precscale:d}'
                mean = 'mean_' if mean else ''

                for observable in ['correlation', 'power']:
                    # full shape fits
                    fits_dir = tmp_base_fits_dir / 'fits_{observable}_{theory}_{template}_freedom-{freedom}{emulator}'
                    if observable == 'correlation': tlim = '{:d}-{:.0f}-{:.0f}'
                    else: tlim = '{:d}-{:.2f}-{:.2f}'
                    lim = get_fit_setup(tracer, observable_name=observable, return_list='lim')       
                    profile_options = {**tmp_meas_options, 'observable': [observable], 'theory': ['velocileptors', 'reptvelocileptors'], 'template': ['shapefit-qisoqap', 'direct_ns-fixed', 'direct', 'fixed'], 'freedom': ['max'], 'covmatrix': ['thecov' if observable == 'power' else 'rascalc', 'ezmock'], 'wmatrix': [''], 'syst': [''], 'lim': [lim], 'emulator': {False: '_noemu', True: ''}}
                    if observable == 'power': profile_options['lim'].append({0: [0.02, 0.12, 0.005], 2: [0.02, 0.12, 0.005]})
                    foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']]}, 'wmatrix': [''], 'syst': ['']}
                    if observable == 'power':
                        profile_options['wmatrix'] += ['rotated']
                        foptions['wmatrix'] += ['_wmat-rotated']
                        profile_options['syst'] += ['rotation']
                        foptions['syst'] += ['_syst-rotation']
                        profile_options['syst'] += ['rotation-hod-photo']
                        foptions['syst'] += ['_syst-rotation-hod-photo']

                    file_manager.append(dict(description='Full shape fits (profiles) to blinded data',
                                             id='profiles_full_shape_{}ez_y1'.format(mean),
                                             filetype='profiles',
                                             path=fits_dir / 'profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{lim}.npy',
                                             options=profile_options,
                                             foptions=foptions))
                    chain_options = {**profile_options, 'ichain': range(8)}
                    file_manager.append(dict(description='Full shape fits (chains) to blinded data',
                                             id='chains_full_shape_{}ez_y1'.format(mean),
                                             filetype='chain',
                                             path=fits_dir / 'chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{lim}_{ichain:d}.npy',
                                             options=chain_options,
                                             foptions=foptions))
                    emulator_options = {name: value for name, value in profile_options.items() if name not in ['region', 'weighting', 'binning', 'cut']}
                    if mean:
                        file_manager.append(dict(description='Emulator for full shape fits to blinded data',
                                                 id='emulator_full_shape_mean_ez_y1',
                                                 path=fits_dir / 'emulators/emulator_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                                 options=emulator_options,
                                                 foptions=base_foptions))
                    # bao fits
                    fits_dir = tmp_base_fits_dir / 'fits_{observable}_{theory}_{template}_{broadband}'
                    for template in ['bao-qisoqap', 'bao-qiso']:
                        lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template, return_list=['lim', 'sigmas'])
                        profile_options = {**tmp_meas_options, 'observable': [observable], 'theory': ['dampedbao'], 'template': [template], 'broadband': ['power3', 'pcs' if observable == 'power' else 'pcs2'], 'covmatrix': ['thecov' if observable == 'power' else 'rascalc', 'ezmock'], 'sigmas': [sigmas], 'lim': [lim]}
                        foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas']]}}
                        file_manager.append(dict(description='BAO fits (profiles) to blinded data',
                                                 id='profiles_bao_{}ez_y1'.format(mean),
                                                 filetype='profiles',
                                                 path=fits_dir / 'profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}_{lim}.npy',
                                                 options=profile_options,
                                                 foptions=foptions))
                        chain_options = {**profile_options, 'ichain': range(8)}
                        file_manager.append(dict(description='BAO fits (chains) to blinded data',
                                                 id='chains_bao_{}ez_y1'.format(mean),
                                                 filetype='chain',
                                                 path=fits_dir / 'chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}_{lim}_{ichain:d}.npy',
                                                 options=chain_options,
                                                 foptions=foptions))

                        lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template + '-post', return_list=['lim', 'sigmas'])
                        profile_options = {**tmp_recon_options, **profile_options, 'lim': [lim], 'sigmas': [sigmas]}
                        foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas']]}}
                        # bao fits post-reconstruction
                        file_manager.append(dict(description='BAO fits (profiles) to blinded data post-reconstruction',
                                                 id='profiles_bao_recon_{}ez_y1'.format(mean),
                                                 filetype='profiles',
                                                 path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}_{lim}.npy',
                                                 options=profile_options,
                                                 foptions=foptions))
                        chain_options = {**profile_options, 'ichain': range(8)}
                        file_manager.append(dict(description='BAO fits (chains) to blinded data post-reconstruction',
                                                 id='chains_bao_recon_{}ez_y1'.format(mean),
                                                 filetype='chain',
                                                 path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}_{lim}_{ichain:d}.npy',
                                                 options=chain_options,
                                                 foptions=foptions))

    file_manager.update(ro=('/global/', '/dvs_ro/'))
    return file_manager


def get_glam_file_manager(**kwargs):

    file_manager = FileManager(environ={name: os.environ.get(name, '.') for name in ['DESICFS']}, **kwargs)
    list_zrange = {'QSO': [(0.8, 2.1)]}

    for tracer, zrange in list_zrange.items():

        #if tracer != 'LRG': continue
        nran = list_nran[tracer]

        versions = ['v1']
        nmocks = 1000

        for version in versions:

            list_fa = ['ffa']

            for fa in list_fa:

                root_dir = Path('/global/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/')
                base_dir = root_dir / 'GLAM/desipipe/{version}/{fa}'
                #data_dir = base_dir / '2pt/mock{imock:d}'
                data_dir = base_dir / '2pt/mock{imock:d}'
                merged_dir = base_dir / '2pt/merged'
                #merged_dir = base_dir / '2pt/merged2'
                base_cov_dir = Path(str(base_dir).replace('{version}', 'v1')) / 'cov_2pt'

                baseline_data_dir = base_dir / 'baseline_2pt/mock{imock:d}'
                baseline_merged_dir = base_dir / 'baseline_2pt/merged'
                base_fits_dir = base_dir / 'fits_2pt/mock{imock:d}'
                mean_fits_dir = base_dir / 'fits_2pt'

                # Catalogs
                if fa == 'ffa':
                    catalog_dir = Path('/pscratch/sd/v/vaisakh/GLAMmain/FFA_GLAM_outputs/Y1/SecondGenMocks/FFA_GLAM/SecondGenMocks/GLAM/mock{imock:d}/')
                    path_data = catalog_dir / '{tracer}_ffa_{region}_clustering.dat.fits'
                    path_randoms = catalog_dir / '{tracer}_ffa_{region}_{iran:d}_clustering.ran.fits'

                base_options = {'region': ['NGC', 'SGC'], 'version': [version], 'tracer': tracer, 'fa': [fa], 'imock': range(1, nmocks + 1)}
                base_foptions = {}
                if fa in ['complete', 'ffa']:
                    base_foptions = {'tracer': [{'BGS_BRIGHT-21.5': 'BGS', 'ELG_LOPnotqso': 'ELG_LOP', 'LRG+ELG_LOPnotqso': 'LRG+ELG_LOP'}.get(tracer, tracer)]}

                file_manager.append(dict(description='Y1 data catalogs',
                                         id='catalog_data_glam_y1',
                                         filetype='catalog',
                                         path=path_data,
                                         options=base_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Y1 randoms catalogs',
                                         id='catalog_randoms_glam_y1',
                                         filetype='catalog',
                                         path=path_randoms,
                                         options={**base_options, 'iran': range(0, nranmax)},
                                         foptions=base_foptions))

                recon_options = {**base_options, 'mode': ['recsym'], 'algorithm': ['IFFT'], 'smoothing_radius': list_smoothing_radius[tracer]}
                file_manager.append(dict(description='Y1 data catalogs post-reconstruction',
                                         id='catalog_data_recon_glam_y1',
                                         filetype='catalog',
                                         path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_{region}_clustering.dat.fits',
                                         link=baseline_data_dir / 'recon_{mode}/{tracer}_{region}_clustering.dat.fits',
                                         options={**recon_options, 'mode': ['recsym', 'reciso']},
                                         foptions=base_foptions))

                file_manager.append(dict(description='Y1 randoms catalogs post-reconstruction',
                                         id='catalog_randoms_recon_glam_y1',
                                         filetype='catalog',
                                         path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                         link=baseline_data_dir / 'recon_{mode}/{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                         options={**recon_options, 'mode': ['recsym', 'reciso'], 'iran': range(0, nran)},
                                         foptions=base_foptions))

                meas_options = {**base_options, 'zrange': zrange, 'region': ['NGC', 'SGC', 'GCcomb'], 'weighting': 'default_FKP', 'binning': ['lin'], 'cut': {None: '', ('theta', 0.05): '_thetacut0.05'}}
                corr_options = {**meas_options, 'nran': [nran], 'split': [20.], 'njack': [0, 60]}
                file_manager.append(dict(description='Correlation functions smu of blinded data',
                                         id='correlation_glam_y1',
                                         filetype='correlation',
                                         path=data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=corr_options,
                                         foptions=base_foptions))

                power_options = {**meas_options, 'cellsize': [6.], 'boxsize': [list_boxsize[tracer]], 'nran': [nran]}
                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_glam_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_rotated_glam_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/rotated/pkpoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/rotated/pkpoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))

                corr_recon_options = {**recon_options, **corr_options, 'cut': {None: ''}}
                file_manager.append(dict(description='Correlation functions smu of blinded data post-reconstruction',
                                         id='correlation_recon_glam_y1',
                                         filetype='correlation',
                                         path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'recon_{mode}/xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=corr_recon_options,
                                         foptions=base_foptions))

                power_recon_options = {**recon_options, **power_options, 'cut': {None: ''}}
                file_manager.append(dict(description='Power spectrum multipoles of blinded data post-reconstruction',
                                         id='power_recon_glam_y1',
                                         filetype='power',
                                         path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'recon_{mode}/pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_recon_options,
                                         foptions=base_foptions))

                # pk window
                merged_options = dict(base_options)
                merged_options.pop('imock')
                file_manager.append(dict(description='Y1 data catalogs',
                                         id='catalog_data_merged_glam_y1',
                                         filetype='catalog',
                                         path=merged_dir / '{tracer}_{region}_clustering.dat.fits',
                                         options=merged_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Y1 randoms catalogs',
                                         id='catalog_randoms_merged_glam_y1',
                                         filetype='catalog',
                                         path=merged_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                         options={**merged_options, 'iran': range(0, nranmax)},
                                         foptions=base_foptions))

                merged_power_options = dict(power_options)
                merged_power_options.pop('imock')
                merged_power_options['nran'] = nranmax
                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_merged_glam_y1',
                                         filetype='power',
                                         path=merged_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum window function of blinded data',
                                         id='window_power_merged_glam_y1',
                                         path=merged_dir / 'pk/window_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}_boxscale{boxscale:.0f}{cut}.npy',
                                         options={**merged_power_options, 'boxscale': [20., 5., 1.]},
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum window matrix of blinded data',
                                         id='wmatrix_power_merged_glam_y1',
                                         filetype='wmatrix',
                                         path=merged_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_merged_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Rotation of power spectrum window function',
                                         id='rotation_wmatrix_power_merged_glam_y1',
                                         path=merged_dir / 'pk/rotated/rotation_wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Rotated power spectrum window matrix of blinded data',
                                         id='wmatrix_power_merged_rotated_glam_y1',
                                         filetype='wmatrix',
                                         path=merged_dir / 'pk/rotated/wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_merged_dir / 'pk/rotated/wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

            for mean in [True, False]:
                tmp_meas_options = dict(meas_options)
                tmp_recon_options = dict(recon_options)
                tmp_base_fits_dir = base_fits_dir
                if mean:
                    tmp_meas_options.pop('imock')
                    tmp_recon_options.pop('imock')
                    tmp_meas_options['precscale'] = tmp_recon_options['precscale'] = [1, 1000]  #nmocks]
                    tmp_base_fits_dir = mean_fits_dir / 'mean_precscale{precscale:d}'
                mean = 'mean_' if mean else ''

                for observable in ['correlation', 'power']:
                    # full shape fits
                    fits_dir = tmp_base_fits_dir / 'fits_{observable}_{theory}_{template}_freedom-{freedom}'
                    if observable == 'correlation': tlim = '{:d}-{:.0f}-{:.0f}'
                    else: tlim = '{:d}-{:.2f}-{:.2f}'
                    lim = get_fit_setup(tracer, observable_name=observable, return_list='lim')
                    profile_options = {**tmp_meas_options, 'observable': [observable], 'theory': ['velocileptors', 'reptvelocileptors'], 'template': ['shapefit-qisoqap', 'direct_ns-fixed', 'direct', 'fixed'], 'freedom': ['max'], 'covmatrix': ['thecov' if observable == 'power' else 'rascalc', 'ezmock'], 'lim': [lim]}
                    foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']]}}
                    file_manager.append(dict(description='Full shape fits (profiles) to blinded data',
                                             id='profiles_full_shape_{}glam_y1'.format(mean),
                                             filetype='profiles',
                                             path=fits_dir / 'profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{lim}.npy',
                                             options=profile_options,
                                             foptions=foptions))
                    chain_options = {**profile_options, 'ichain': range(8)}
                    file_manager.append(dict(description='Full shape fits (chains) to blinded data',
                                             id='chains_full_shape_{}glam_y1'.format(mean),
                                             filetype='chain',
                                             path=fits_dir / 'chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{lim}_{ichain:d}.npy',
                                             options=chain_options,
                                             foptions=foptions))
                    emulator_options = {name: value for name, value in profile_options.items() if name not in ['region', 'weighting', 'binning', 'cut']}
                    if mean:
                        file_manager.append(dict(description='Emulator for full shape fits to blinded data',
                                                 id='emulator_full_shape_mean_glam_y1',
                                                 path=fits_dir / 'emulators/emulator_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                                 options=emulator_options,
                                                 foptions=base_foptions))
                    # bao fits
                    fits_dir = tmp_base_fits_dir / 'fits_{observable}_{theory}_{template}_{broadband}'
                    for template in ['bao-qisoqap', 'bao-qiso']:
                        lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template, return_list=['lim', 'sigmas'])
                        profile_options = {**tmp_meas_options, 'observable': [observable], 'theory': ['dampedbao'], 'template': [template], 'broadband': ['power3', 'pcs' if observable == 'power' else 'pcs2'], 'covmatrix': ['thecov' if observable == 'power' else 'rascalc', 'ezmock'], 'sigmas': [sigmas], 'lim': [lim]}
                        foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas']]}}
                        file_manager.append(dict(description='BAO fits (profiles) to blinded data',
                                                 id='profiles_bao_{}glam_y1'.format(mean),
                                                 filetype='profiles',
                                                 path=fits_dir / 'profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}_{lim}.npy',
                                                 options=profile_options,
                                                 foptions=foptions))
                        chain_options = {**profile_options, 'ichain': range(8)}
                        file_manager.append(dict(description='BAO fits (chains) to blinded data',
                                                 id='chains_bao_{}glam_y1'.format(mean),
                                                 filetype='chain',
                                                 path=fits_dir / 'chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}_{lim}_{ichain:d}.npy',
                                                 options=chain_options,
                                                 foptions=foptions))

                        lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template + '-post', return_list=['lim', 'sigmas'])
                        profile_options = {**tmp_recon_options, **profile_options, 'lim': [lim], 'sigmas': [sigmas]}
                        foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas']]}}
                        # bao fits post-reconstruction
                        file_manager.append(dict(description='BAO fits (profiles) to blinded data post-reconstruction',
                                                 id='profiles_bao_recon_{}glam_y1'.format(mean),
                                                 filetype='profiles',
                                                 path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}_{lim}.npy',
                                                 options=profile_options,
                                                 foptions=foptions))
                        chain_options = {**profile_options, 'ichain': range(8)}
                        file_manager.append(dict(description='BAO fits (chains) to blinded data post-reconstruction',
                                                 id='chains_bao_recon_{}glam_y1'.format(mean),
                                                 filetype='chain',
                                                 path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}_cov-{covmatrix}_{sigmas}_{lim}_{ichain:d}.npy',
                                                 options=chain_options,
                                                 foptions=foptions))

    return file_manager


def get_box_abacus_file_manager(**kwargs):

    file_manager = FileManager(environ={name: os.environ.get(name, '.') for name in ['DESICFS']}, **kwargs)
    nmocks = 25
    list_zsnap = {'BGS_BRIGHT-21.5': [0.200], 'LRG': [0.500, 0.800, 1.100], 'ELG_LOPnotqso': [0.950, 1.100, 1.325], 'QSO': [1.100, 1.400, 1.700]}

    for tracer, zsnap in list_zsnap.items():

        versions = ['v1', 'v1.1']
        if 'BGS' in tracer: versions = ['v0.1']

        for version in versions:
            base_options = {'version': version, 'tracer': tracer, 'zsnap': zsnap, 'imock': range(0, nmocks)}
            base_foptions = {}
            base_foptions = {'tracer': [{'BGS_BRIGHT-21.5': 'BGS', 'ELG_LOPnotqso': 'ELG'}.get(tracer, tracer)]}

            if 'BGS' in tracer:
                catalog_dir = Path('${DESICFS}/cosmosim/SecondGenMocks/AbacusSummit/CubicBox/{tracer}/{version}/z{zsnap:.3f}')
                base_dir = Path('${DESICFS}/cosmosim/SecondGenMocks/CubicBox/desipipe/BGS_{version}/')
            else:
                catalog_dir = Path('${DESICFS}/cosmosim/SecondGenMocks/CubicBox/{tracer}/z{zsnap:.3f}')
                base_dir = Path('${DESICFS}/cosmosim/SecondGenMocks/CubicBox/desipipe/{version}/')
            
            data_dir = base_dir / '2pt/mock{imock:d}_los-{los}'
            merged_dir = base_dir.parent / 'wmatrix'
            baseline_data_dir = base_dir / 'baseline_2pt/mock{imock:d}_los-{los}'
            baseline_merged_dir = merged_dir
            base_fits_dir = base_dir / 'fits_2pt/mock{imock:d}_los-{los}'
            mean_fits_dir = base_dir / 'fits_2pt'
            
            if 'BGS' in tracer:
                path_data = catalog_dir / 'AbacusSummit_base_c000_ph{imock:03d}/{tracer}_box_ph{imock:03d}.fits'
                file_manager.append(dict(description='Y1 data box catalogs',
                                         id='catalog_data_box_abacus_y1',
                                         filetype='catalog',
                                         path=path_data,
                                         options=base_options,
                                         foptions=base_foptions))
            
            elif version == 'v1':
                path_data = catalog_dir / 'AbacusSummit_base_c000_ph{imock:03d}/{tracer}_real_space.sub{isub:d}.fits.gz'
                file_manager.append(dict(description='Y1 data box catalogs',
                                         id='catalog_data_box_abacus_y1',
                                         filetype='catalog',
                                         path=path_data,
                                         options={**base_options, 'isub': range(64)},
                                         foptions=base_foptions))
            else:
                path_data = catalog_dir / 'AbacusSummit_base_c000_ph{imock:03d}/{tracer}_real_space.fits'
                file_manager.append(dict(description='Y1 data box catalogs',
                                         id='catalog_data_box_abacus_y1',
                                         filetype='catalog',
                                         path=path_data,
                                         options=base_options,
                                         foptions=base_foptions))

            if 'BGS' in tracer:
                base_options['catalog'] = 'rmag21.5'
            recon_options = {**base_options, 'los': ['x', 'y', 'z', None], 'mode': ['recsym'], 'algorithm': ['IFFT'], 'smoothing_radius': list_smoothing_radius[tracer]}
            file_manager.append(dict(description='Y1 data catalogs post-reconstruction',
                                     id='catalog_data_recon_box_abacus_y1',
                                     filetype='catalog',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_clustering.dat.fits',
                                     options={**recon_options, 'mode': ['recsym', 'reciso']},
                                     foptions=base_foptions))

            file_manager.append(dict(description='Y1 randoms catalogs post-reconstruction',
                                     id='catalog_randoms_recon_box_abacus_y1',
                                     filetype='catalog',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_{iran:d}_clustering.ran.fits',
                                     options={**recon_options, 'mode': ['recsym', 'reciso'], 'iran': range(0, nranmax)},
                                     foptions=base_foptions))

            meas_options = {**base_options, 'boxsize': [2000.], 'los': ['x', 'y', 'z', None], 'binning': ['lin']}
            corr_options = {**meas_options}
            file_manager.append(dict(description='Correlation functions smu of blinded data',
                                     id='correlation_box_abacus_y1',
                                     filetype='correlation',
                                     path=data_dir / 'xi/smu/allcounts_{tracer}_z{zsnap:.4f}_{binning}.npy',
                                     link=baseline_data_dir / 'xi/smu/allcounts_{tracer}_z{zsnap:.4f}_{binning}.npy',
                                     options=corr_options,
                                     foptions=base_foptions))

            power_options = {**meas_options, 'cellsize': [2.]}
            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_box_abacus_y1',
                                     filetype='power',
                                     path=data_dir / 'pk/pkpoles_{tracer}_z{zsnap:.4f}_{binning}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}.npy',
                                     link=baseline_data_dir / 'pk/pkpoles_{tracer}_z{zsnap:.4f}.npy',
                                     options=power_options,
                                     foptions=base_foptions))

            corr_recon_options = {**recon_options, **corr_options, 'nran': [nranmax]}
            file_manager.append(dict(description='Correlation functions smu of blinded data post-reconstruction',
                                     id='correlation_recon_box_abacus_y1',
                                     filetype='correlation',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/xi/smu/allcounts_{tracer}_z{zsnap:.4f}_{binning}_nran{nran:d}_split{split:.0f}.npy',
                                     link=baseline_data_dir / 'recon_{mode}/xi/smu/allcounts_{tracer}_z{zsnap:.4f}.npy',
                                     options=corr_recon_options,
                                     foptions=base_foptions))

            power_recon_options = {**recon_options, **power_options, 'nran': [nranmax]}
            file_manager.append(dict(description='Power spectrum multipoles of blinded data post-reconstruction',
                                     id='power_recon_box_abacus_y1',
                                     filetype='power',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/pk/pkpoles_{tracer}_z{zsnap:.4f}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}.npy',
                                     link=baseline_data_dir / 'recon_{mode}/pk/pkpoles_{tracer}_z{zsnap:.4f}.npy',
                                     options=power_recon_options,
                                     foptions=base_foptions))

            # pk window
            merged_power_options = dict(power_options)
            merged_power_options.pop('imock')
            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_merged_box_abacus_y1',
                                     filetype='power',
                                     path=merged_dir / 'pkpoles_{tracer}_z{zsnap:.4f}_{binning}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}.npy',
                                     options=merged_power_options,
                                     foptions=base_foptions))

            file_manager.append(dict(description='Power spectrum window matrix of blinded data',
                                     id='wmatrix_power_merged_box_abacus_y1',
                                     filetype='wmatrix',
                                     path=merged_dir / 'wmatrix_{binning}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}.npy',
                                     link=baseline_merged_dir / 'wmatrix.npy',
                                     options=merged_power_options,
                                     foptions=base_foptions))

            for mean in [True, False]:
                tmp_meas_options = dict(meas_options)
                tmp_recon_options = dict(recon_options)
                tmp_base_fits_dir = base_fits_dir
                if mean:
                    tmp_meas_options.pop('imock')
                    tmp_recon_options.pop('imock')
                    tmp_meas_options['precscale'] = tmp_recon_options['precscale'] = [1, nmocks]
                    tmp_base_fits_dir = mean_fits_dir / 'mean_precscale{precscale:d}'
                mean = 'mean_' if mean else ''

                for observable in ['correlation', 'power']:
                    # full shape fits
                    fits_dir = tmp_base_fits_dir / 'fits_{observable}_{theory}_{template}_freedom-{freedom}'
                    if observable == 'correlation': tlim = '{:d}-{:.0f}-{:.0f}'
                    else: tlim = '{:d}-{:.2f}-{:.2f}'
                    lim = get_fit_setup(tracer, observable_name=observable, return_list='lim')
                    profile_options = {**tmp_meas_options, 'observable': [observable], 'theory': ['velocileptors'], 'template': ['shapefit-qisoqap', 'direct_ns-fixed', 'direct'], 'freedom': ['max'], 'lim': [lim]}
                    foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']]}}
                    file_manager.append(dict(description='Full shape fits (profiles) to blinded data',
                                             id='profiles_full_shape_{}box_abacus_y1'.format(mean),
                                             filetype='profiles',
                                             path=fits_dir / 'profiles_{tracer}_z{zsnap:.4f}_{lim}.npy',
                                             options=profile_options,
                                             foptions=foptions))
                    chain_options = {**profile_options, 'ichain': range(8)}
                    file_manager.append(dict(description='Full shape fits (chains) to blinded data',
                                             id='chains_full_shape_{}box_abacus_y1'.format(mean),
                                             filetype='chain',
                                             path=fits_dir / 'chain_{tracer}_z{zsnap:.4f}_{lim}_{ichain:d}.npy',
                                             options=chain_options,
                                             foptions=foptions))
                    emulator_options = {name: value for name, value in profile_options.items() if name not in ['region', 'weighting', 'binning', 'cut']}
                    if mean:
                        file_manager.append(dict(description='Emulator for full shape fits to blinded data',
                                                 id='emulator_full_shape_mean_box_abacus_y1',
                                                 path=fits_dir / 'emulators/emulator_{tracer}_z{zsnap:.4f}.npy',
                                                 options=emulator_options,
                                                 foptions=base_foptions))
                    # bao fits
                    fits_dir = tmp_base_fits_dir / 'fits_{observable}_{theory}_{template}_{broadband}'
                    for template in ['bao-qisoqap', 'bao-qiso']:
                        lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template, return_list=['lim', 'sigmas'])
                        profile_options = {**tmp_meas_options, 'observable': [observable], 'theory': ['dampedbao'], 'template': [template], 'broadband': ['power3', 'pcs' if observable == 'power' else 'pcs2'], 'lim': [lim], 'sigmas': [sigmas]}
                        foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas']]}}
                        file_manager.append(dict(description='BAO fits (profiles) to blinded data',
                                                 id='profiles_bao_{}box_abacus_y1'.format(mean),
                                                 filetype='profiles',
                                                 path=fits_dir / 'profiles_{tracer}_z{zsnap:.4f}_{sigmas}_{lim}.npy',
                                                 options=profile_options,
                                                 foptions=foptions))
                        chain_options = {**profile_options, 'ichain': range(8)}
                        file_manager.append(dict(description='BAO fits (chains) to blinded data',
                                                 id='chains_bao_{}box_abacus_y1'.format(mean),
                                                 filetype='chain',
                                                 path=fits_dir / 'chain_{tracer}_z{zsnap:.4f}_{sigmas}_{lim}_{ichain:d}.npy',
                                                 options=chain_options,
                                                 foptions=foptions))

                        lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template + '-post', return_list=['lim', 'sigmas'])
                        profile_options = {**tmp_recon_options, **profile_options, 'lim': [lim], 'sigmas': [sigmas]}
                        foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas']]}}
                        # bao fits post-reconstruction
                        file_manager.append(dict(description='BAO fits (profiles) to blinded data post-reconstruction',
                                                 id='profiles_bao_recon_{}box_abacus_y1'.format(mean),
                                                 filetype='profiles',
                                                 path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/profiles_{tracer}_z{zsnap:.4f}_{sigmas}_{lim}.npy',
                                                 options=profile_options,
                                                 foptions=foptions))
                        chain_options = {**profile_options, 'ichain': range(8)}
                        file_manager.append(dict(description='BAO fits (chains) to blinded data post-reconstruction',
                                                 id='chains_bao_recon_{}box_abacus_y1'.format(mean),
                                                 filetype='chain',
                                                 path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/chain_{tracer}_z{zsnap:.4f}_{sigmas}_{lim}_{ichain:d}.npy',
                                                 options=chain_options,
                                                 foptions=foptions))

    return file_manager


def get_raw_abacus_file_manager(**kwargs):

    file_manager = FileManager(environ={name: os.environ.get(name, '.') for name in ['DESICFS']}, **kwargs)
    nmocks = 25
    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'QSO': [(0.8, 2.1)]}

    for tracer, zrange in list_zrange.items():

        for version in ['v3_1', 'v4_1', 'v4_1_complete']:
            #if tracer != 'LRG': continue
            nran = list_nran[tracer]
            base_options = {'region': ['NGC', 'SGC'], 'version': version, 'tracer': tracer, 'imock': range(0, nmocks)}
            base_foptions = {'tracer': [{'ELG_LOPnotqso': 'ELG_LOP'}.get(tracer, tracer)]}
            
            base_dir = Path('${DESICFS}/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/{version}/raw')
            
            data_dir = base_dir / '2pt{zshuffled}/mock{imock:d}'
            merged_dir = base_dir / '2pt{zshuffled}/merged'
            baseline_data_dir = base_dir / 'baseline_2pt/mock{imock:d}'
            baseline_merged_dir = base_dir / 'baseline_2pt/merged'
            base_fits_dir = base_dir / 'fits_2pt/mock{imock:d}'
            mean_fits_dir = base_dir / 'fits_2pt'

            # Catalogs
            if version == 'v3_1':
                catalog_dir = Path('${DESICFS}/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v3_1/raw')
                path_data = catalog_dir / 'raw_{tracer}_{imock:d}.fits'
                path_shuffled_randoms = path_randoms = catalog_dir / '{tracer}randoms/random_{tracer}_{iran:d}.fits'
            elif version in ['v3_1_b', 'v4_1']:
                catalog_dir = base_dir / 'catalogs'
                catalog_dir = Path('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/raw/catalogs/')
                path_data = catalog_dir / 'mock{imock:d}/{tracer}_raw_{region}_clustering.dat.fits'
                path_randoms = catalog_dir / 'randoms/{tracer}_raw_{region}_{iran:d}_clustering.ran.fits'
                path_shuffled_randoms = catalog_dir / 'mock{imock:d}/{tracer}_raw_{region}_{iran:d}_clustering.ran.fits'

            elif version in ['v4_1_complete']:
                catalog_dir = base_dir / 'catalogs'
                #catalog_dir = Path('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1')
                path_data = catalog_dir / 'mock{imock:d}/{tracer}_complete_{region}_clustering.dat.fits'
                path_randoms = catalog_dir / 'mock{imock:d}/{tracer}_complete_{region}_{iran:d}_clustering.ran.fits'
                catalog_shuffled_dir = Path('${DESICFS}/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1_complete/raw/catalogs/')
                path_shuffled_randoms = catalog_shuffled_dir / 'mock{imock:d}/{tracer}_raw_{region}_{iran:d}_clustering.ran.fits'

            options = dict(base_options)
            options.pop('region')
            options.pop('imock')
            file_manager.append(dict(description='Healpix mask',
                                     id='mask_healpix',
                                     path=base_dir / 'mask_healpix_{tracer}.npy',
                                     options=options,
                                     foptions=base_foptions))
            
            options = {**base_options, 'region': ['ALL']}
            foptions = dict(base_foptions)
            if version == 'v3_1':
                foptions.update({'tracer': [{'LRG': 'lrgs', 'ELG_LOPnotqso': 'elg_lop', 'QSO': 'qsos'}.get(tracer, tracer)]})
            if version == 'v4_1_complete':
                options['region'] = ['NGC', 'SGC']
            file_manager.append(dict(description='Y1 data catalogs',
                                     id='catalog_data_raw_abacus_y1',
                                     filetype='catalog',
                                     path=path_data,
                                     options=options,
                                     foptions=foptions))

            if version == 'v3_1':
                foptions.update({'tracer': [{'LRG': 'lrg', 'ELG_LOPnotqso': 'elglop', 'QSO': 'qso'}.get(tracer, tracer)]})
            file_manager.append(dict(description='Y1 randoms catalogs',
                                     id='catalog_randoms_raw_abacus_y1',
                                     filetype='catalog',
                                     path=path_shuffled_randoms,
                                     options={**options, 'zshuffled': [True], 'iran': range(0, nranmax)},
                                     foptions=foptions))

            if version != 'v4_1_complete':
                options.pop('imock')
            file_manager.append(dict(description='Y1 randoms catalogs',
                                     id='catalog_randoms_raw_abacus_y1',
                                     filetype='catalog',
                                     path=path_randoms,
                                     options={**options, 'zshuffled': [False], 'iran': range(0, nranmax)},
                                     foptions=foptions))

            base_options = {**base_options, 'zshuffled': [False, True], 'catalog': ['rsd-z', 'rsd-snapshot', 'rsd-no', 'standard', 'mask']}
            base_foptions['zshuffled'] = ['', '_zshuffled']
            recon_options = {**base_options, 'mode': ['recsym'], 'algorithm': ['IFFT'], 'smoothing_radius': list_smoothing_radius[tracer]}
            file_manager.append(dict(description='Y1 data catalogs post-reconstruction',
                                     id='catalog_data_recon_raw_abacus_y1',
                                     filetype='catalog',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_{region}_clustering.dat.fits',
                                     link=baseline_data_dir / 'recon_{mode}/{tracer}_{region}_clustering.dat.fits',
                                     options={**recon_options, 'mode': ['recsym', 'reciso']},
                                     foptions=base_foptions))

            file_manager.append(dict(description='Y1 randoms catalogs post-reconstruction',
                                     id='catalog_randoms_recon_raw_abacus_y1',
                                     filetype='catalog',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                     link=baseline_data_dir / 'recon_{mode}/{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                     options={**recon_options, 'mode': ['recsym', 'reciso'], 'iran': range(0, nranmax)},
                                     foptions=base_foptions))

            meas_options = {**base_options, 'zrange': zrange, 'region': ['NGC', 'SGC', 'GCcomb'], 'weighting': ['', 'default', 'default_FKP'], 'binning': ['lin'], 'cut': {None: '', ('theta', 0.05): '_thetacut0.05'}}
            corr_options = {**meas_options, 'nran': [nran], 'split': [20.], 'njack': [0]}
            base_foptions['weighting'] = ['', '_default', '_default_FKP']
            file_manager.append(dict(description='Correlation functions smu of blinded data',
                                     id='correlation_raw_abacus_y1',
                                     filetype='correlation',
                                     path=data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{catalog}{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=corr_options,
                                     foptions=base_foptions))

            power_options = {**meas_options, 'cellsize': [6.], 'boxsize': [list_boxsize[tracer]], 'nran': [nranmax]}
            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_raw_abacus_y1',
                                     filetype='power',
                                     path=data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{catalog}{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=power_options,
                                     foptions=base_foptions))

            corr_recon_options = {**recon_options, **corr_options, 'cut': {None: ''}}
            file_manager.append(dict(description='Correlation functions smu of blinded data post-reconstruction',
                                     id='correlation_recon_raw_abacus_y1',
                                     filetype='correlation',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{catalog}{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'recon_{mode}/xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=corr_recon_options,
                                     foptions=base_foptions))

            power_recon_options = {**recon_options, **power_options, 'cut': {None: ''}}
            file_manager.append(dict(description='Power spectrum multipoles of blinded data post-reconstruction',
                                     id='power_recon_raw_abacus_y1',
                                     filetype='power',
                                     path=data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}/pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{catalog}{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     link=baseline_data_dir / 'recon_{mode}/pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=power_recon_options,
                                     foptions=base_foptions))

            # pk window
            merged_options = dict(base_options)
            merged_options.pop('imock')
            merged_options.pop('catalog')
            #merged_options.pop('zshuffled')
            merged_options['region'] = 'ALL'
            file_manager.append(dict(description='Y1 data catalogs',
                                     id='catalog_data_merged_raw_abacus_y1',
                                     filetype='catalog',
                                     path=merged_dir / '{tracer}_{region}_clustering.dat.fits',
                                     options=merged_options,
                                     foptions=base_foptions))

            file_manager.append(dict(description='Y1 randoms catalogs',
                                     id='catalog_randoms_merged_raw_abacus_y1',
                                     filetype='catalog',
                                     path=merged_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                     options={**merged_options, 'iran': range(0, nranmax)},
                                     foptions=base_foptions))

            merged_power_options = dict(power_options)
            merged_power_options.pop('imock')
            merged_power_options['nran'] = nranmax
            file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                     id='power_merged_raw_abacus_y1',
                                     filetype='power',
                                     path=merged_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{catalog}{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     options=merged_power_options,
                                     foptions=base_foptions))

            file_manager.append(dict(description='Power spectrum window function of blinded data',
                                     id='window_power_merged_raw_abacus_y1',
                                     path=merged_dir / 'pk/window_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{catalog}{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}_boxscale{boxscale:.0f}{cut}.npy',
                                     options={**merged_power_options, 'boxscale': [20., 5., 1.]},
                                     foptions=base_foptions))

            file_manager.append(dict(description='Power spectrum window matrix of blinded data',
                                     id='wmatrix_power_merged_raw_abacus_y1',
                                     filetype='wmatrix',
                                     path=merged_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{catalog}{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                     link=baseline_merged_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                     options=merged_power_options,
                                     foptions=base_foptions))
    return file_manager


def get_abacus_file_manager(**kwargs):

    file_manager = FileManager(environ={name: os.environ.get(name, '.') for name in ['DESICFS']}, **kwargs)
    nmocks = 25
    #list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(1.1, 1.6), (1.1, 1.58)], 'QSO': [(0.8, 2.1)]}
    list_smoothing_radius = {'BGS_BRIGHT-21.5': [10., 15.], 'LRG': [10., 15.], 'LRG+ELG_LOPnotqso': [10., 15.], 'ELG_LOPnotqso': [10., 15.], 'QSO': [20., 30.]}
    #list_smoothing_radius = {'BGS_BRIGHT-21.5': [10.], 'LRG': [10.], 'ELG_LOPnotqso': [10.], 'QSO': [20.]}

    for tracer, zrange in list_zrange.items():

        for version in ['v3', 'v3_1_window', 'test_eb', 'v4', 'v3_1', 'v3_1_lrg+elg', 'v4_1', 'v4_1fixran', 'v1', 'v2', 'v4_2'][-5:]:
            #if tracer != 'LRG': continue
            nran = list_nran[tracer]
            list_fa = ['complete', 'ffa', 'altmtl']
            if version in ['v2']:
                list_fa = ['altmtl']
            for fa in list_fa:
                base_options = {'region': ['NGC', 'SGC'], 'version': version, 'tracer': tracer, 'fa': [fa], 'imock': range(0, nmocks)}
                base_foptions = {}
                if fa in ['complete', 'ffa']:
                    base_foptions = {'tracer': [{'ELG_LOPnotqso': 'ELG_LOP'}.get(tracer, tracer)]}

                root_dir = Path('${DESICFS}/survey/catalogs/Y1/mocks/SecondGenMocks/')
                #root_dir = Path('/pscratch/sd/a/adematti/desipipe')
                if 'BGS' in tracer:
                    base_dir = root_dir / 'AbacusSummitBGS/desipipe/{version}/{fa}{catalog}'
                else:
                    base_dir = root_dir / 'AbacusSummit/desipipe/{version}/{fa}{catalog}'
                
                data_dir = base_dir / '2pt/mock{imock:d}'
                merged_dir = base_dir / '2pt/merged'
                #merged_dir = base_dir / '2pt/merged2'
                base_cov_dir = Path(str(base_dir).replace('{version}', 'v1' if 'BGS' in tracer else 'v4_2')) / 'cov_2pt'
                base_forfit_dir = base_cov_dir.parent / 'forfit_2pt'

                baseline_data_dir = base_dir / 'baseline_2pt/mock{imock:d}'
                baseline_merged_dir = base_dir / 'baseline_2pt/merged'
                base_fits_dir = base_dir / 'fits_2pt/mock{imock:d}'
                mean_fits_dir = base_dir / 'fits_2pt'

                #root_dir = Path('${DESICFS}')
                root_dir = Path('/dvs_ro/cfs/cdirs/desi/')
                # Catalogs
                if version == 'v3_1_lrg+elg':
                    catalog_dir = root_dir / '/users/dvalcin/EZMOCKS/Overlap/SecondGen/FITS_FILES/AbacusSummit_v3_1/mock{imock:d}/'
                    path_data = catalog_dir / '{tracer}_{region}_clustering.dat.fits'
                    path_randoms = catalog_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits'
                elif 'BGS' in tracer and version in ['v1']:
                    if fa == 'altmtl':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummitBGS/altmtl{imock:d}/mock{imock:d}/LSScats'
                        path_data = catalog_dir / '{tracer}_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits'
                    elif fa == 'ffa':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummitBGS/FFA/mock{imock:d}'
                        path_data = catalog_dir / '{tracer}_ffa_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{tracer}_ffa_{region}_{iran:d}_clustering.ran.fits'
                    elif fa == 'complete':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummitBGS/mock{imock:d}'
                        path_data = catalog_dir / '{tracer}_complete_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{tracer}_complete_{region}_{iran:d}_clustering.ran.fits'
                elif 'BGS' in tracer and version in ['v2']:
                    if fa == 'altmtl':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummitBGS_v2/altmtl{imock:d}/mock{imock:d}/LSScats'
                        path_data = catalog_dir / '{tracer}_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits'
                elif version in ['v3', 'v3_1', 'v3_1_window', 'test_eb', 'v4', 'v4_1', 'v4_1fixran', 'v4_2']:
                    if fa == 'altmtl':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_{version}/altmtl{imock:d}/mock{imock:d}/LSScats'
                        path_data = catalog_dir / '{tracer}_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits'
                    elif fa == 'ffa':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_{version}/mock{imock:d}'
                        path_data = catalog_dir / '{tracer}_ffa_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{tracer}_ffa_{region}_{iran:d}_clustering.ran.fits'
                    elif fa == 'complete':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_{version}/mock{imock:d}'
                        path_data = catalog_dir / '{tracer}_complete_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{tracer}_complete_{region}_{iran:d}_clustering.ran.fits'
                elif version == 'v2':
                    if fa == 'altmtl':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/altmtl{imock:d}/mock{imock:d}/LSScats'
                        path_data = catalog_dir / '{tracer}_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits'
                    elif fa == 'ffa':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock{imock:d}'
                        if 'BGS' in tracer:
                            catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummitBGS/SecondGenMocks/AbacusSummit'
                        path_data = catalog_dir / '{version}' / '{tracer}_ffa_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{version}' / '{tracer}_ffa_{region}_{iran:d}_clustering.ran.fits'
                    elif fa == 'complete':
                        catalog_dir = root_dir / 'survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/mock{imock:d}'
                        path_data = catalog_dir / '{tracer}_complete_gtlimaging_{region}_clustering.dat.fits'
                        path_randoms = catalog_dir / '{tracer}_complete_gtlimaging_{region}_{iran:d}_clustering.ran.fits'

                file_manager.append(dict(description='Y1 data catalogs',
                                         id='catalog_data_abacus_y1',
                                         filetype='catalog',
                                         path=path_data,
                                         options=base_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Y1 randoms catalogs',
                                         id='catalog_randoms_abacus_y1',
                                         filetype='catalog',
                                         path=path_randoms,
                                         options={**base_options, 'iran': range(0, nranmax)},
                                         foptions=base_foptions))
                if version == 'test_eb':
                    base_options['catalog'] = ['rsd-z', 'rsd-snapshot', 'rsd-no', 'standard']
                    base_foptions['catalog'] = ['_rsd-z', '_rsd-snapshot', '_rsd-no', '']
                else:
                    base_options['catalog'] = ['standard']
                    base_foptions['catalog'] = ['']

                recon_options = {**base_options, 'mode': ['recsym'], 'algorithm': ['IFFT'], 'recon_zrange': [None] + zrange, 'recon_weighting': ['default'], 'smoothing_radius': list_smoothing_radius[tracer]}
                
                recon_dir = data_dir / 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}{recon_zrange}{recon_weighting}/'
                baseline_recon_dir = baseline_data_dir / 'recon_{mode}{recon_zrange}{recon_weighting}/'
                recon_foptions = dict(base_foptions)
                recon_foptions['recon_zrange'] = ['' if zr is None else '_z{zrange[0]:.1f}-{zrange[1]:.1f}'.format(zrange=zr) for zr in recon_options['recon_zrange']]
                recon_foptions['recon_weighting'] = ['' if w == 'default' else '_{}'.format(w) for w in recon_options['recon_weighting']]
            
                file_manager.append(dict(description='Y1 data catalogs post-reconstruction',
                                         id='catalog_data_recon_abacus_y1',
                                         filetype='catalog',
                                         path=recon_dir / '{tracer}_{region}_clustering.dat.fits',
                                         link=baseline_recon_dir / '{tracer}_{region}_clustering.dat.fits',
                                         options={**recon_options, 'mode': ['recsym', 'reciso']},
                                         foptions=recon_foptions))

                file_manager.append(dict(description='Y1 randoms catalogs post-reconstruction',
                                         id='catalog_randoms_recon_abacus_y1',
                                         filetype='catalog',
                                         path=recon_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                         link=baseline_recon_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                         options={**recon_options, 'mode': ['recsym', 'reciso'], 'iran': range(0, nranmax)},
                                         foptions=recon_foptions))

                meas_options = {**base_options, 'zrange': zrange, 'region': ['NGC', 'SGC', 'GCcomb'], 'weighting': 'default_FKP', 'binning': ['lin'], 'cut': {None: '', ('theta', 0.05): '_thetacut0.05'}}
                meas_options['cut'] = {None: '', ('theta', 0.05): '_thetacut0.05'} #, ('theta', 0.07): '_thetacut0.07'}
                #if fa == 'complete':
                #    meas_options['weighting'] = ['default_FKP', 'default']
                corr_options = {**meas_options, 'nran': [nran], 'split': [20.], 'njack': [0, 60]}
                file_manager.append(dict(description='Correlation functions smu of blinded data',
                                         id='correlation_abacus_y1',
                                         filetype='correlation',
                                         path=data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'xi/smu/allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=corr_options,
                                         foptions=base_foptions))

                power_options = {**meas_options, 'cellsize': [6.], 'boxsize': [list_boxsize[tracer]], 'nran': [nranmax]}
                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_abacus_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_rotated_abacus_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/rotated/pkpoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/rotated/pkpoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))
            
                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_corrected_abacus_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/corrected/pkpoles_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/corrected/pkpoles_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))
                
                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_rotated_corrected_abacus_y1',
                                         filetype='power',
                                         path=data_dir / 'pk/rotated_corrected/pkpoles_rotated_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_data_dir / 'pk/rotated_corrected/pkpoles_rotated_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=power_options,
                                         foptions=base_foptions))


                #for zr, recon_zr in zip([zrange, [None]], [[None], recon_options['recon_zrange']]):
                for zr, recon_zr in zip([zrange] + [[zr] for zr in zrange], [[None]] + [[zr] for zr in zrange]):
                    corr_recon_options = {**recon_options, **corr_options, 'mode': ['recsym', 'reciso'], 'zrange': zr, 'recon_zrange': recon_zr}
                    recon_foptions['zrange'] = ['' if zr is None else '_z{zrange[0]:.1f}-{zrange[1]:.1f}'.format(zrange=zr) for zr in corr_recon_options['zrange']]
                    recon_foptions['recon_zrange'] = ['' if zr is None else '_z{zrange[0]:.1f}-{zrange[1]:.1f}'.format(zrange=zr) for zr in corr_recon_options['recon_zrange']]
                    file_manager.append(dict(description='Correlation functions smu of blinded data post-reconstruction',
                                             id='correlation_recon_abacus_y1',
                                             filetype='correlation',
                                             path=recon_dir / 'xi/smu/allcounts_{tracer}_{region}{zrange}_{weighting}_{binning}_nran{nran:d}_njack{njack:d}_split{split:.0f}{cut}.npy',
                                             link=baseline_recon_dir / 'xi/smu/allcounts_{tracer}_{region}{zrange}{cut}.npy',
                                             options=corr_recon_options,
                                             foptions=recon_foptions))

                    power_recon_options = {**recon_options, **power_options, 'mode': ['recsym', 'reciso'], 'zrange': zr, 'recon_zrange': recon_zr}
                    file_manager.append(dict(description='Power spectrum multipoles of blinded data post-reconstruction',
                                             id='power_recon_abacus_y1',
                                             filetype='power',
                                             path=recon_dir / 'pk/pkpoles_{tracer}_{region}{zrange}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                             link=baseline_recon_dir / 'pk/pkpoles_{tracer}_{region}{zrange}{cut}.npy',
                                             options=power_recon_options,
                                             foptions=recon_foptions))

                # pk window
                merged_options = dict(base_options)
                merged_options.pop('imock')
                file_manager.append(dict(description='Y1 data catalogs',
                                         id='catalog_data_merged_abacus_y1',
                                         filetype='catalog',
                                         path=merged_dir / '{tracer}_{region}_clustering.dat.fits',
                                         options=merged_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Y1 randoms catalogs',
                                         id='catalog_randoms_merged_abacus_y1',
                                         filetype='catalog',
                                         path=merged_dir / '{tracer}_{region}_{iran:d}_clustering.ran.fits',
                                         options={**merged_options, 'iran': range(0, nranmax)},
                                         foptions=base_foptions))

                merged_power_options = dict(power_options)
                merged_power_options.pop('imock')
                merged_power_options['nran'] = nranmax
                file_manager.append(dict(description='Power spectrum multipoles of blinded data',
                                         id='power_merged_abacus_y1',
                                         filetype='power',
                                         path=merged_dir / 'pk/pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum window function of blinded data',
                                         id='window_power_merged_abacus_y1',
                                         path=merged_dir / 'pk/window_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}_boxscale{boxscale:.0f}{cut}.npy',
                                         options={**merged_power_options, 'boxscale': [20., 5., 1.]},
                                         foptions=base_foptions))

                file_manager.append(dict(description='Power spectrum window matrix of blinded data',
                                         id='wmatrix_power_merged_abacus_y1',
                                         filetype='wmatrix',
                                         path=merged_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_merged_dir / 'pk/wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Rotation of power spectrum window function',
                                         id='rotation_wmatrix_power_merged_abacus_y1',
                                         path=merged_dir / 'pk/rotated/rotation_wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Rotated power spectrum window matrix of blinded data',
                                         id='wmatrix_power_merged_rotated_abacus_y1',
                                         filetype='wmatrix',
                                         path=merged_dir / 'pk/rotated/wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_merged_dir / 'pk/wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Rotation of power spectrum window function',
                                         id='rotation_wmatrix_ric_power_merged_abacus_y1',
                                         path=merged_dir / 'pk/rotated/rotation_wmatrix_ric_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Rotated power spectrum window matrix of blinded data',
                                         id='wmatrix_ric_power_merged_rotated_abacus_y1',
                                         filetype='wmatrix',
                                         path=merged_dir / 'pk/rotated/wmatrix_ric_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}_nran{nran:d}_cellsize{cellsize:.0f}_boxsize{boxsize:.0f}{cut}.npy',
                                         link=baseline_merged_dir / 'pk/rotated/wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{cut}.npy',
                                         options=merged_power_options,
                                         foptions=base_foptions))

                if version == 'v1':
                    observables = ['power']
                    for observable in observables:
                        forfit_options = {**base_options, 'zrange': zrange, 'region': 'GCcomb', 'observable': observable, 'syst': '', 'klim': {None: ''}}
                        forfit_options.pop('imock')
                        forfit_foptions = dict(base_foptions)
                        if 'power' in observable:
                            forfit_options['syst'] = {'rotation': '_syst-rotation'}
                            forfit_options['klim'] = [{0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]}, {0: [0.02, 0.12, 0.005], 2: [0.02, 0.12, 0.005]}]
                            tlim = '{:d}-{:.2f}-{:.2f}'
                            forfit_foptions['klim'] = ['_klim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in forfit_options['klim']]
                        file_manager.append(dict(description='Covariance matrix, observable and window matrix for fit',
                                                 author='',
                                                 id='forfit_abacus_y1',
                                                 filetype='covariance',
                                                 path=base_forfit_dir / 'forfit_{observable}{syst}{klim}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                                 options=forfit_options,
                                                 foptions=forfit_foptions))

                    # 2-pt covariances
                    sources = ['rascalc', 'thecov', 'ezmock']
                    if 'LRG+ELG' in tracer:
                        sources = sources[:2]
                    for source in sources:
                        if source == 'ezmock':
                            cov_version = 'v1'
                            observables = ['power', 'power-recon', 'correlation', 'correlation-recon', 'power+correlation-recon', 'power+bao-recon']
                        else:
                            cov_version = 'v1.2'
                            if source == 'rascalc':
                                observables = ['correlation', 'correlation-recon']
                            else:
                                observables = ['power', 'power-recon']
                        for observable in observables:
                            cov_options = {**meas_options, 'region': 'GCcomb' if source in ['thecov', 'hybrid'] or any(name in observable for name in ['bao-recon', 'shapefit']) else ['NGC', 'SGC', 'GCcomb'], 'weighting': 'default_FKP', 'source': [source], 'version': cov_version, 'observable': observable}
                            cov_options.pop('imock')
                            if source == 'rascalc': cov_options['version'] = 'v1.5'  #'v1.2' if 'recon' in observable else 'v0.6'
                            cov_options['cut'] = {None: '', ('theta', 0.05): '_thetacut0.05'}
                            if not any(name in observable.split('+') for name in ['power', 'correlation']):
                                cov_options['cut'] = {None: ''}

                            file_manager.append(dict(description='Covariance matrix',
                                                     author='mrash, oalves',
                                                     id='covariance_abacus_y1',
                                                     filetype='covariance',
                                                     path=base_cov_dir / '{source}/{version}/covariance_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                     options=cov_options,
                                                     foptions=base_foptions))

                            if 'power' in observable:
                                #cov_options = {**cov_options, 'cut': {('theta', 0.05): '_thetacut0.05'}, 'marg': {False: '', 'rotation': '_marg-rotation', 'rotation-syst': '_marg-rotation-syst'}}
                                cov_options = {**cov_options, 'cut': {('theta', 0.05): '_thetacut0.05'}}
                                file_manager.append(dict(description='Covariance matrix',
                                                         author='mrash, oalves',
                                                         id='covariance_rotated_abacus_y1',
                                                         filetype='covariance',
                                                         path=base_cov_dir / '{source}/{version}/rotated/covariance_rotated_marg-no_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                         options=cov_options,
                                                         foptions=base_foptions))

                            if observable != 'power': continue
                            cov_options['source'] = ['syst']
                            cov_options['syst'] = ['rotation', 'photo', 'hod']
                            cov_options['version'] = 'v1'
                            #cov_options.pop('marg')
                            file_manager.append(dict(description='Covariance matrix',
                                                     author='ntbfin',
                                                     id='covariance_syst_abacus_y1',
                                                     filetype='covariance',
                                                     path=base_cov_dir / '{source}/{version}/covariance_syst-{syst}_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                     options=cov_options,
                                                     foptions=base_foptions))
                            file_manager.append(dict(description='Covariance matrix',
                                                     author='ntbfin',
                                                     id='covariance_syst_rotated_abacus_y1',
                                                     filetype='covariance',
                                                     path=base_cov_dir / '{source}/{version}/rotated/covariance_syst-{syst}_rotated_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                                     options=cov_options,
                                                     foptions=base_foptions))

                template_options = {**meas_options, 'syst': ['ric', 'aic', 'photo'], 'observable': ['power']}
                template_options.pop('imock')
                file_manager.append(dict(description='Systematic template',
                                         id='template_syst_abacus_y1',
                                         filetype='template',
                                         path=base_dir / 'templates_2pt/template_{syst}_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                         options=template_options,
                                         foptions=base_foptions))

                file_manager.append(dict(description='Systematic template',
                                         id='template_syst_rotated_abacus_y1',
                                         filetype='template',
                                         path=base_dir / 'templates_2pt/rotated/template_rotated_{syst}_{observable}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}_{binning}{cut}.npy',
                                         options=template_options,
                                         foptions=base_foptions))
                                
                for mean in [True, False]:
                    tmp_meas_options = dict(meas_options)
                    tmp_recon_options = dict(recon_options)
                    tmp_base_fits_dir = base_fits_dir
                    if mean:
                        tmp_meas_options.pop('imock')
                        tmp_recon_options.pop('imock')
                        tmp_meas_options['precscale'] = tmp_recon_options['precscale'] = [1, nmocks]
                        tmp_base_fits_dir = mean_fits_dir / 'mean_precscale{precscale:d}'
                    mean = 'mean_' if mean else ''

                    for observable in ['correlation', 'power']:
                        # full shape fits
                        fits_dir = tmp_base_fits_dir / 'fits_{observable}_{theory}_{template}_freedom-{freedom}'
                        if observable == 'correlation': tlim = '{:d}-{:.0f}-{:.0f}'
                        else: tlim = '{:d}-{:.2f}-{:.2f}'
                        lim = get_fit_setup(tracer, observable_name=observable, return_list='lim')
                        profile_options = {**tmp_meas_options, 'observable': [observable], 'theory': ['velocileptors', 'reptvelocileptors'], 'template': ['shapefit-qisoqap', 'direct_ns-fixed', 'direct', 'fixed'], 'freedom': ['max'], 'covmatrix': ['thecov' if observable == 'power' else 'rascalc', 'ezmock'], 'wmatrix': [''], 'syst': ['', 'rotation'], 'lim': [lim]}
                        foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']]}, 'wmatrix': [''], 'syst': ['', '_syst-photo']}
                        if observable == 'power':
                            profile_options['wmatrix'] += ['rotated']
                            foptions['wmatrix'] += ['_wmat-rotated']
                        file_manager.append(dict(description='Full shape fits (profiles) to blinded data',
                                                 id='profiles_full_shape_{}abacus_y1'.format(mean),
                                                 filetype='profiles',
                                                 path=fits_dir / 'profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{lim}.npy',
                                                 options=profile_options,
                                                 foptions=foptions))
                        chain_options = {**profile_options, 'ichain': range(8)}
                        file_manager.append(dict(description='Full shape fits (chains) to blinded data',
                                                 id='chains_full_shape_{}abacus_y1'.format(mean),
                                                 filetype='chain',
                                                 path=fits_dir / 'chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{lim}_{ichain:d}.npy',
                                                 options=chain_options,
                                                 foptions=foptions))
                        emulator_options = {name: value for name, value in profile_options.items() if name not in ['region', 'weighting', 'binning', 'cut']}
                        if mean:
                            file_manager.append(dict(description='Emulator for full shape fits to blinded data',
                                                     id='emulator_full_shape_mean_abacus_y1',
                                                     path=fits_dir / 'emulators/emulator_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                                     options=emulator_options,
                                                     foptions=base_foptions))
                        # bao fits
                        fits_dir = tmp_base_fits_dir / 'fits_{observable}_{theory}_{template}_{broadband}'
                        for template in ['bao-qisoqap', 'bao-qiso']:
                            lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template, return_list=['lim', 'sigmas'])
                            profile_options = {**tmp_meas_options, 'observable': [observable], 'theory': ['dampedbao'], 'template': [template], 'broadband': ['power3', 'pcs' if observable == 'power' else 'pcs2', 'fixed'], 'lim': [lim], 'sigmas': [sigmas]}
                            foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas']]}}
                            file_manager.append(dict(description='BAO fits (profiles) to blinded data',
                                                     id='profiles_bao_{}abacus_y1'.format(mean),
                                                     filetype='profiles',
                                                     path=fits_dir / 'profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{sigmas}_{lim}.npy',
                                                     options=profile_options,
                                                     foptions=foptions))
                            chain_options = {**profile_options, 'ichain': range(8)}
                            file_manager.append(dict(description='BAO fits (chains) to blinded data',
                                                     id='chains_bao_{}abacus_y1'.format(mean),
                                                     filetype='chain',
                                                     path=fits_dir / 'chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{sigmas}_{lim}_{ichain:d}.npy',
                                                     options=chain_options,
                                                     foptions=foptions))

                            lim, sigmas = get_fit_setup(tracer, observable_name=observable, theory_name=template + '-post', return_list=['lim', 'sigmas'])
                            profile_options = {**tmp_recon_options, **profile_options, 'lim': [lim], 'sigmas': [sigmas]}
                            foptions = {**base_foptions, **{'lim': ['lim_' + '_'.join([tlim.format(key, *value) for key, value in lim.items()]) for lim in profile_options['lim']], 'sigmas': ['sigmas-{sigmas[0]:.1f}-{sigmas[1]:.1f}_sigmapar-{sigmapar[0]:.1f}-{sigmapar[1]:.1f}_sigmaper-{sigmaper[0]:.1f}-{sigmaper[1]:.1f}'.format(**sigmas) for sigmas in profile_options['sigmas']]}}
                            # bao fits post-reconstruction
                            file_manager.append(dict(description='BAO fits (profiles) to blinded data post-reconstruction',
                                                     id='profiles_bao_recon_{}abacus_y1'.format(mean),
                                                     filetype='profiles',
                                                     path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/profiles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{sigmas}_{lim}.npy',
                                                     options=profile_options,
                                                     foptions=foptions))
                            chain_options = {**profile_options, 'ichain': range(8)}
                            file_manager.append(dict(description='BAO fits (chains) to blinded data post-reconstruction',
                                                     id='chains_bao_recon_{}abacus_y1'.format(mean),
                                                     filetype='chain',
                                                     path=fits_dir / 'recon_{algorithm}_{mode}_sm{smoothing_radius:.0f}/chain_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{weighting}{cut}{syst}_cov-{covmatrix}{wmatrix}_{sigmas}_{lim}_{ichain:d}.npy',
                                                     options=chain_options,
                                                     foptions=foptions))

    return file_manager


def get_cosmo_setup(cosmo=None, model='base', dataset='bao', convert='desilike', return_extra=False):
    if cosmo is None:
        from cosmoprimo.fiducial import DESI
        cosmo = DESI()

    if not isinstance(dataset, str):
        dataset = ' '.join(dataset)

    params = {'Omega_cdm': {'prior': {'limits': [0.01, 0.9]}, 'ref': {'dist': 'norm', 'loc': cosmo['Omega_cdm'], 'scale': 0.006}, 'delta': 0.015, 'latex': '\Omega_{cdm}'},
              'Omega_b': {'prior': {'limits': [0.001, 0.3]}, 'ref': {'dist': 'norm', 'loc': cosmo['Omega_b'], 'scale': 0.0006}, 'delta': 0.003, 'latex': '\Omega_{b}'},
              'H0': {'prior': {'limits': [20., 100]}, 'ref': {'dist': 'norm', 'loc': cosmo['H0'], 'scale': 0.5}, 'delta': 3., 'latex': 'H_{0}'},
              #'Omega_k': {'prior': {'limits': [-0.8, 0.8]}, 'ref': {'dist': 'norm', 'loc': cosmo['Omega_k'], 'scale': 0.0065}, 'delta': 0.01, 'latex': '\Omega_{k}'},
              #'w0_fld': {'prior': {'limits': [-3., 1.]}, 'ref': {'dist': 'norm', 'loc': cosmo['w0_fld'], 'scale': 0.08}, 'delta': 0.1, 'latex': 'w_{0}'},  # 0.1
              #'wa_fld': {'prior': {'limits': [-3., 2.]}, 'ref': {'dist': 'norm', 'loc': cosmo['wa_fld'], 'scale': 0.3}, 'delta': 0.15, 'latex': 'w_{a}'},  # 0.15

              'Omega_k': {'prior': {'limits': [-0.8, 0.8]}, 'ref': {'dist': 'norm', 'loc': cosmo['Omega_k'], 'scale': 0.001}, 'delta': 0.05, 'latex': '\Omega_{k}'},
              'w0_fld': {'prior': {'limits': [-3., 1.]}, 'ref': {'dist': 'norm', 'loc': cosmo['w0_fld'], 'scale': 0.01}, 'delta': 0.2, 'latex': 'w_{0}'},  # 0.1
              'wa_fld': {'prior': {'limits': [-3., 2.]}, 'ref': {'dist': 'norm', 'loc': cosmo['wa_fld'], 'scale': 0.01}, 'delta': 0.3, 'latex': 'w_{a}'},  # 0.15

              'logA': {'prior': {'limits': [1.61, 3.91]}, 'ref': {'dist': 'norm', 'loc': cosmo['logA'], 'scale': 0.014}, 'delta': 0.05, 'latex': '\ln(10^{10} A_{s})'},
              'n_s': {'prior': {'limits': [0.8, 1.2]}, 'ref': {'dist': 'norm', 'loc': cosmo['n_s'], 'scale': 0.0042}, 'delta': 0.005, 'latex': 'n_{s}'},
              'tau_reio': {'prior': {'limits': [0.01, 0.8]}, 'ref': {'dist': 'norm', 'loc': cosmo['tau_reio'], 'scale': 0.008}, 'delta': 0.01, 'latex': r'\tau'},
              'm_ncdm': {'prior': {'limits': [0., 5.]}, 'ref': {'dist': 'norm', 'loc': cosmo['m_ncdm_tot'], 'scale': 0.02}, 'delta': [0.31, 0.15, 0.15], 'latex': 'm_{ncdm}'},
              'sigma8_m': {'derived': True, 'latex': '\sigma_{8, m}'},
              'Omega_m': {'derived': True, 'latex': '\Omega_{m}'},
              'Omega_Lambda': {'derived': True, 'latex': '\Omega_{\Lambda}'},
              'rs_drag': {'derived': True, 'latex': 'r_{d}'}}
    extra = {}

    from desilike import ParameterCollection
    params = ParameterCollection(params)
    if 'base' in model:
        for param in params.select(name=['w0_fld', 'wa_fld', 'Omega_k', 'tau_reio', 'm_ncdm']): param.update(fixed=True)
    if 'mnu' in model:
        for param in params.select(name=['m_ncdm']): param.update(fixed=False)
    if 'omegak' in model:
        for param in params.select(name=['Omega_k']): param.update(fixed=False)
    if 'w' in model:
        for param in params.select(name=['w0_fld']): param.update(fixed=False)
    if 'wa' in model:
        for param in params.select(name=['wa_fld']): param.update(fixed=False)

    if any(name in dataset for name in ['bao', 'pantheon', 'union3']) and not any(name in dataset for name in ['all-params', 'direct', 'shapefit', 'full', 'planck']):
        for param in params.select(name=['logA', 'tau_reio', 'n_s']): param.update(fixed=True)
        params.pop('sigma8_m', None)

    if 'planck2018' in dataset:
        for param in params.select(name=['tau_reio']): param.update(fixed=False)

    if 'ns-fixed' in model:
        for param in params.select(name=['n_s']): param.update(fixed=True)

    if 'cobaya' in convert:
        from desilike.bindings.cobaya import desilike_to_cobaya_params
        extra = {name: cosmo[name] for name in ['N_ncdm', 'N_ur']}
        if 'w' in model:
            extra.update(dict(Omega_Lambda=0., fluid_equation_of_state='CLP', use_ppf='yes'))
        else:
            del params['w0_fld'], params['wa_fld']
        params.pop('sigma8_m', None)
        params = desilike_to_cobaya_params(params, engine='camb' if 'camb' in convert else 'classy')
    if return_extra:
        return params, extra
    #for param in params: param.update(fixed=True)
    #for param in params.select(name='wa_fld'): param.update(fixed=False)
    return params


def get_cosmo_file_manager(base_dir=None, chain_dir=None, version='test', conf={}, **kwargs):  #conf={'w0_fld': -0.5, 'wa_fld': -1.}, **kwargs):
    if base_dir is None:
        base_dir = '${DESICFS}/science/cpe/y1kp7/'
    if chain_dir is None:
        chain_dir = base_dir
    # Let's stick with Planck structure: model/dataset/
    if isinstance(conf, dict):
        fconf = '_'.join(['_'.join([key, f'{value:.2g}']) for key, value in conf.items()])
        fconf = 'mock{}_blinded_y1'.format(('_' if fconf else '') + fconf)
    else:
        fconf = str(conf)
    from pathlib import Path
    data_dir = Path(base_dir)  / '{version}/{conf}/data'
    chain_dir = Path(chain_dir) / '{version}/{conf}'
    file_manager = FileManager(environ={name: os.environ.get(name, '.') for name in ['DESICFS']}, **kwargs)

    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'QSO': [(0.8, 2.1)], 'Lya': [(1.8, 4.2)]}
    for tracer in list_zrange:
        for zrange in list_zrange[tracer]:
            options = {'version': version, 'conf': [conf], 'tracer': tracer, 'region': 'GCcomb', 'zrange': zrange}
            foptions = {'conf': fconf}
            if 'Lya' in tracer:
                file_manager.append(dict(description='BAO Gaussian likelihood from Y1 data',
                                         id='gaussian_bao_y1',
                                         path=data_dir / 'gaussian_bao_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                         options=options,
                                         foptions=foptions))
                continue

            options['weighting'] = ['default_FKP']
            bao_options = {**options, **{key: [value] for key, value in get_bao_baseline_fit_setup(tracer, zrange).items()}}
            fs_options = {**options, **{key: [value] for key, value in get_fs_baseline_fit_setup(tracer, zrange).items()}}

            file_manager.append(dict(description='BAO chain from Y1 data',
                                     id='chain_bao_y1',
                                     path=data_dir / 'chain_bao_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                     options=bao_options,
                                     foptions=foptions))
            file_manager.append(dict(description='BAO Gaussian likelihood from Y1 data',
                                     id='gaussian_bao_y1',
                                     path=data_dir / 'gaussian_bao_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                     options=bao_options,
                                     foptions=foptions))
            for compression_options in [bao_options, {**fs_options, 'template': 'shapefit'}]:
                file_manager.append(dict(description='Compression measurement from Y1 data',
                                         id='gaussian_compression_y1',
                                         path=data_dir / '{observable}_{theory}_{template}_gaussian_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                         options=compression_options,
                                         foptions=foptions))
                file_manager.append(dict(description='Compression measurement from Y1 data',
                                         id='profiles_compression_y1',
                                         filetype='profiles',
                                         path=data_dir / 'profiles_{observable}_{theory}_{template}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                         options=compression_options,
                                         foptions=foptions))
                file_manager.append(dict(description='Compression measurement from Y1 data',
                                         id='chains_compression_y1',
                                         filetype='chain',
                                         path=data_dir / 'chain_{observable}_{theory}_{template}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{ichain:d}.npy',
                                         options={**compression_options, 'ichain': range(8)},
                                         foptions=foptions))
                file_manager.append(dict(description='Compression emulator',
                                         id='compression_emulator_y1',
                                         path=data_dir / 'emulator_{observable}_{theory}_{template}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                         options=compression_options,
                                         foptions=foptions))
            bao_options.pop('template', '')
            fs_options.pop('template', '')
            file_manager.append(dict(description='Correlation function multipoles of Y1 data post-reconstruction',
                                     filetype='correlation',
                                     id='correlation_recon_y1',
                                     path=data_dir / 'correlation_recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                     options=bao_options,
                                     foptions=foptions))
            file_manager.append(dict(description='Covariance matrix for correlation function multipoles post-reconstruction',
                                     author='oalves',
                                     id='covariance_correlation_recon_y1',
                                     filetype='correlation_covariance',
                                     path=data_dir / 'covariance_correlation_recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.txt',
                                     options=bao_options,
                                     foptions=foptions))
            file_manager.append(dict(description='Power spectrum multipoles of Y1 data',
                                     filetype='power',
                                     id='power_y1',
                                     path=data_dir / 'power_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                     options=fs_options,
                                     foptions=foptions))
            file_manager.append(dict(description='Covariance matrix for power spectrum multipoles',
                                     author='oalves',
                                     id='covariance_power_y1',
                                     filetype='power_covariance',
                                     path=data_dir / 'covariance_power_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.txt',
                                     options=fs_options,
                                     foptions=foptions))
            file_manager.append(dict(description='Power spectrum window matrix',
                                     id='wmatrix_power_y1',
                                     filetype='wmatrix',
                                     path=data_dir / 'wmatrix_power_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy',
                                     options=fs_options,
                                     foptions=foptions))

    list_code = ['cobaya', 'desilike']
    #list_externals = {'desi-bao-gaussian': [[], ['bbn-omega_b'], ['pantheon'], ['pantheon', 'bbn-omega_b'], ['pantheon', 'planck2018']], 'desi-bao-gaussian-bgs-lrg-elg-qso': [[], ['bbn-omega_b'], ['pantheon']], 'desi-full-shape-power': [['bbn-omega_b'], ['planck2018']], 'desi-full-shape-power-bgs-lrg-elg-qso': [['bbn-omega_b']], 'desi-full-shape-power-bgs': [['bbn-omega_b']], 'desi-full-shape-power-lrg_0': [['bbn-omega_b']]}
    """
    list_datasets = [['desi-bao-gaussian'], ['desi-bao-gaussian', 'bbn-omega_b'], ['desi-bao-gaussian', 'pantheon'], ['desi-bao-gaussian', 'pantheon', 'bbn-omega_b'], ['desi-bao-gaussian', 'pantheon+'], ['desi-bao-gaussian', 'pantheon+', 'bbn-omega_b']]
    list_datasets += [['desi-bao-gaussian-bgs-lrg-elg-qso'], ['desi-bao-gaussian-bgs-lrg-elg-qso', 'bbn-omega_b'], ['desi-bao-gaussian-bgs-lrg-elg-qso', 'pantheon']]
    list_datasets += [['desi-shapefit-gaussian', 'bbn-omega_b']]
    list_datasets += [['desi-full-shape-power'], ['desi-full-shape-power', 'bbn-omega_b']]
    list_datasets += [['desi-full-shape-power-bgs_0', 'desi-full-shape-power-lrg_0', 'desi-full-shape-power-lrg_1', 'desi-full-shape-power-lrg_2', 'desi-full-shape-power-elg_0', 'desi-full-shape-power-elg_1', 'desi-full-shape-power-qso_0', 'bbn-omega_b']]
    list_datasets += [['desi-bao-correlation-recon'], ['desi-bao-correlation-recon', 'bbn-omega_b'], ['desi-full-shape-power-bao-correlation-recon', 'bbn-omega_b']]
    list_datasets += [['desi-bao-correlation-recon-all-params'], ['desi-bao-gaussian-all-params']]
    list_datasets += [['desi-bao-correlation-recon-direct-all-params']]
    list_datasets += [['desi-bao-correlation-recon-direct-all-params-lrg_0']]
    list_datasets += ['desi-bao-correlation-recon-lrg_0'], ['desi-bao-correlation-recon-lrg_0', 'bbn-omega_b']
    list_datasets += [['planck2018']]
    list_datasets += [['planck2018-lite']]
    list_datasets += [['desi-bao-gaussian', 'planck2018']]
    list_datasets += [['desi-bao-gaussian-bgs-lrg-elg-qso', 'planck2018']]
    list_datasets += [['desi-shapefit-gaussian', 'planck2018']]
    list_datasets += [['desi-full-shape-power', 'planck2018']]
    list_datasets += [['desi-shapefit-gaussian', 'planck2018-lite']]
    list_datasets += [['desi-full-shape-power', 'planck2018-lite']]
    """
    list_datasets = []
    list_datasets.append(['desi-bao-gaussian'])
    list_datasets.append(['desi-bao-gaussian-bgs'])
    list_datasets.append(['desi-bao-gaussian-lrg'])
    list_datasets.append(['desi-bao-gaussian-elg'])
    list_datasets.append(['desi-bao-gaussian-qso'])
    list_datasets.append(['desi-bao-gaussian-lya'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg-elg-qso'])
    list_datasets.append(['desi-bao-gaussian-elg-qso-lya'])
    list_datasets.append(['desi-bao-gaussian-lrg-elg-qso-lya'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg'])

    list_datasets.append(['desi-bao-gaussian', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-bgs', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-lrg', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-lrg_0'])
    list_datasets.append(['desi-bao-gaussian-lrg_1'])
    list_datasets.append(['desi-bao-gaussian-lrg_2'])
    list_datasets.append(['desi-bao-gaussian-elg', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-qso', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-lya', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg-elg-qso', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-elg-qso-lya', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-lrg-elg-qso-lya', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian', 'planck2018'])

    list_datasets.append(['desi-bao-gaussian', 'pantheon'])
    list_datasets.append(['desi-bao-gaussian', 'pantheon+'])
    list_datasets.append(['desi-bao-gaussian', 'union3'])
    list_datasets.append(['pantheon'])
    list_datasets.append(['pantheon', 'bbn-omega_b'])
    list_datasets.append(['pantheon+'])
    list_datasets.append(['pantheon+', 'bbn-omega_b'])
    list_datasets.append(['union3'])
    list_datasets.append(['union3', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian', 'pantheon', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian', 'pantheon+', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian', 'union3', 'bbn-omega_b'])

    list_datasets.append(['desi-full-shape-power', 'bbn-omega_b'])

    list_emulator = {'planck2018': 'taylor', 'planck2018-lite': 'taylor', 'desi-shapefit-gaussian': 'taylor', 'desi-full-shape-power': 'taylor-velocileptors', 'desi-full-shape-power-bao-correlation-recon': 'taylor-velocileptors'}
    list_emulator.update({'desi-full-shape-power-bgs_0': 'taylor-velocileptors', 'desi-full-shape-power-lrg_0': 'taylor-velocileptors', 'desi-full-shape-power-lrg_1': 'taylor-velocileptors', 'desi-full-shape-power-lrg_2': 'taylor-velocileptors', 'desi-full-shape-power-elg_0': 'taylor-velocileptors', 'desi-full-shape-power-elg_1': 'taylor-velocileptors', 'desi-full-shape-power-qso_0': 'taylor-velocileptors'})
    list_emulator.update({'desi-bao-correlation-recon-direct-all-params': 'taylor'})
    model = ['base', 'base_mnu', 'base_omegak', 'base_w', 'base_w_wa']
    model += ['base_ns-fixed', 'base_mnu_ns-fixed', 'base_omegak_ns-fixed', 'base_w_ns-fixed', 'base_w_wa_ns-fixed']
    for code in list_code:
        for datasets in list_datasets:
            emulators = [{dataset: list_emulator[dataset] for dataset in datasets if dataset in list_emulator}]
            options = {'version': version, 'conf': [conf], 'code': code, 'model': model, 'datasets': [datasets], 'emulators': emulators}
            foptions = {'conf': fconf}
            foptions['datasets'] = ['_'.join(datasets)]
            foptions['emulators'] = ['_' + '_'.join(['-'.join([key, value]) for key, value in emulator.items()]) if emulator else '' for emulator in emulators]
            if 'cobaya' in code:
                file_manager.append(dict(description='Cosmological inference with Y1 data',
                                     id='chain_cosmological_inference_y1',
                                     #path=str(chain_dir / '{code}/{model}/{datasets}/chain{emulators}'),
                                     path=str(chain_dir / '{code}/{model}/{datasets}/chain{emulators}') + ('_simple_fixed-dbeta-sigma' if 'desi-full-shape-power-bao-correlation-recon' in '_'.join(datasets) else ''),
                                     options=options,
                                     foptions=foptions))
            if 'desilike' in code:
                file_manager.append(dict(description='Cosmological inference with Y1 data',
                                         id='profiles_cosmological_inference_y1',
                                         path=chain_dir / '{code}/{model}/{datasets}/profiles{emulators}.npy',
                                         options=options,
                                         foptions=foptions))
                
                file_manager.append(dict(description='Cosmological inference with Y1 data',
                                     id='chain_cosmological_inference_y1',
                                     #path=str(chain_dir / '{code}/{model}/{datasets}/chain{emulators}'),
                                     path=str(chain_dir / '{code}/{model}/{datasets}/chain{emulators}') + ('_simple_fixed-dbeta-sigma' if 'desi-full-shape-power-bao-correlation-recon' in '_'.join(datasets) else ''),
                                     options=options,
                                     foptions=foptions))
    for datasets in list_datasets:
        dataset_planck = 'planck2018'
        if dataset_planck in datasets:
            emulators = [{dataset: list_emulator[dataset] for dataset in datasets if dataset in list_emulator and dataset != dataset_planck}]
            options = {'version': version, 'conf': [conf], 'code': 'importance-planck', 'model': model, 'datasets': [datasets], 'emulators': emulators}
            foptions = {'conf': fconf}
            foptions['datasets'] = ['_'.join(datasets)]
            foptions['emulators'] = ['_' + '_'.join(['-'.join([key, value]) for key, value in emulator.items() if value is not None]) if emulator else '' for emulator in emulators]
            file_manager.append(dict(description='Cosmological inference with Y1 data using importance sampling',
                                     id='chain_cosmological_inference_y1',
                                     path=chain_dir / '{code}/{model}/{datasets}/chain{emulators}',
                                     options=options,
                                     foptions=foptions))
    for dataset, emulator in list_emulator.items():
        namespace, fnamespace = [None], ['']
        fdataset = dataset
        theory = [None]
        fdesi_datasets = ['desi-full-shape-power', 'desi-shapefit-gaussian'] + ['desi-bao-correlation-recon-direct-all-params']
        if any(fdataset in dataset for fdataset in fdesi_datasets):
            base = ''
            if 'power' in dataset: base = 'power_'
            elif 'correlation-recon' in dataset: base = 'correlation_recon_'
            namespace, fnamespace = [], []
            for tracer, zranges in list_zrange.items():
                if 'Lya' in tracer: continue
                for iz, zrange in enumerate(zranges):
                    namespace.append('{base}{tracer}_{iz}'.format(base=base, tracer=tracer[:3], iz=iz))
                    fnamespace.append('_{tracer}_{iz}'.format(base=base, tracer=tracer[:3], iz=iz))
            for fdataset in fdesi_datasets:
                if fdataset in dataset: break
            theory = emulator.replace('taylor-', '')
            if not theory: theory = [None]
        options = {'version': version, 'conf': [conf], 'code': 'desilike', 'model': 'base_omegak_w_wa_mnu', 'emulator': emulator, 'dataset': dataset, 'namespace': namespace, 'theory': theory}
        foptions = {'conf': fconf, 'dataset': fdataset, 'namespace': fnamespace}
        file_manager.append(dict(description='Emulator for cosmological inference for Y1 data',
                                 id='emulator_y1',
                                 path=chain_dir / '{code}/{model}/emulators/emulator_{dataset}{namespace}_{emulator}.npy',
                                 options=options,
                                 foptions=foptions))

    return file_manager