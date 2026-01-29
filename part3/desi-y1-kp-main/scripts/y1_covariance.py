
def load_covariance_matrix(covariance_fn, binning, select=None, ells=None):

    import numpy as np

    def cut_matrix(cov, xcov, ellscov, xlim):
        """Cut a matrix based on specified indices and returns the resulting submatrix."""
        import numpy as np
        edges = None

        if isinstance(xcov, tuple):
            xmin, xmax, xstep = xcov
            if xmax is None:
                xmax = xmin + xstep * (len(cov) // len(ellscov))
            xcov = np.arange(xmin + xstep / 2., xmax + xstep / 2., xstep)
            edges = np.arange(xmin, xmax + xstep / 2., xstep)

        assert len(cov) == len(xcov) * len(ellscov), 'Input matrix {} has size {}, different than {} x {}'.format(covariance_fn, len(cov), len(xcov), len(ellscov))
        indices = []
        for ell, xlim in xlim.items():
            index = ellscov.index(ell) * len(xcov) + np.arange(len(xcov))
            if xlim is not None:
                index = index[(xcov >= xlim[0]) & (xcov <= xlim[1])]
            indices.append(index)
        indices = np.concatenate(indices, axis=0)
        return cov[np.ix_(indices, indices)], xcov, edges, ellscov

    covariance = np.loadtxt(covariance_fn)
    ellscov = (0, 2, 4)

    if ells is None:
        ells = ellscov

    if select is None:
        select = dict.fromkeys(ells)
    elif not isinstance(select, dict):
        select = {ell: select for ell in ells}

    return cut_matrix(covariance, binning, ellscov, select)


def export_rascalc_covariance():
    import glob
    import logging
    from desilike.observables import ObservableCovariance
    from desi_y1_files import get_data_file_manager

    logger = logging.getLogger('Covariance')
    fm = get_data_file_manager(conf='unblinded')

    for fcov in fm.select(id='covariance_y1', source='rascalc').iter(intersection=False):
        if 'recon' in fcov.options['observable']:  # post-recon
            covariance_fn = '/dvs_ro/cfs/cdirs/desi/users/mrash/RascalC/Y1/unblinded/{version}/xi024_{tracer}_IFFT_recsym_sm*_{region}_{zrange[0]:.1f}_{zrange[1]:.1f}_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'
        else:
            covariance_fn = '/dvs_ro/cfs/cdirs/desi/users/mrash/RascalC/Y1/unblinded/{version}/xi024_{tracer}_{region}_{zrange[0]:.1f}_{zrange[1]:.1f}_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'
            #covariance_fn = '/dvs_ro/cfs/cdirs/desi/users/mrash/RascalC/Y1/blinded/v0.6/xi024_{tracer}_{region}_{zrange[0]:.1f}_{zrange[1]:.1f}_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'
        covariance_fn = covariance_fn.format(**fcov.options)
        fn = glob.glob(covariance_fn)
        if not fn:
            print('Covariance {} does not exist.'.format(covariance_fn))
            continue
        assert len(fn) == 1, covariance_fn
        fn = fn[0]
        binning = (20., 200., 4.)
        value, x, edges, ells = load_covariance_matrix(fn, binning)
        covariance = ObservableCovariance(value=value, observables=[{'name': 'correlation', 'x': [x] * len(ells), 'edges': [edges] * len(ells), 'projs': list(ells)}])
        fcov.save(covariance)
        #fcov.load()

    
def export_thecov_covariance():
    import glob
    import logging
    from desilike.observables import ObservableCovariance
    from desi_y1_files import get_data_file_manager

    logger = logging.getLogger('Covariance')

    fm = get_data_file_manager(conf='unblinded')
    for fcov in fm.select(id='covariance_y1', source='thecov').iter(intersection=False):
        if 'recon' in fcov.options['observable']:  # post-recon
            covariance_fn = '/dvs_ro/cfs/cdirs/desi/users/oalves/thecovs/y1_unblinded/post/v1.2/cov_gaussian_{tracer}_{region}_{zrange[0]:.1f}-{zrange[1]:.1f}.txt'
        else:
            covariance_fn = '/dvs_ro/cfs/cdirs/desi/users/oalves/thecovs/y1_blinded/pre/v1.2/cov_gaussian_{tracer}_{region}_{zrange[0]:.1f}-{zrange[1]:.1f}.txt'
        covariance_fn = covariance_fn.format(**fcov.options)
        fn = glob.glob(covariance_fn)
        if not fn:
            print('Covariance {} does not exist.'.format(covariance_fn))
            continue
        assert len(fn) == 1, covariance_fn
        fn = fn[0]
        binning = (0., None, 0.005)
        value, x, edges, ells = load_covariance_matrix(fn, binning)
        covariance = ObservableCovariance(value=value, observables=[{'name': 'power', 'x': [x] * len(ells), 'edges': [edges] * len(ells), 'projs': list(ells)}])
        fcov.save(covariance)
        #fcov.load()


def export_ezmock_covariance(only_2pt=True):
    import logging
    import numpy as np
    from desi_y1_files import get_ez_file_manager
    from desilike.observables import ObservableCovariance
    from desilike.samples import Profiles
    from pypower import PowerSpectrumMultipoles
    from pycorr import TwoPointCorrelationFunction
    from desi_y1_files import get_data_file_manager, get_bao_baseline_fit_setup, get_fs_baseline_fit_setup
    
    logger = logging.getLogger('Covariance')

    fm = get_data_file_manager(conf='unblinded')
    fm_ez = get_ez_file_manager()    
    for fcov in fm.select(id='covariance_y1', source='ezmock').iter(intersection=False):
        #if fcov.options['tracer'] != 'LRG': continue
        #if fcov.options['observable'] not in ['shapefit+bao-recon']: continue
        if 'shapefit' not in fcov.options['observable']: continue
        if 'bao-recon' not in fcov.options['observable']: continue
        #if 'shapefit' in fcov.options['observable'] and 'ELG' in fcov.options['tracer'] and fcov.options['zrange'] == (0.8, 1.1): continue
        #if 'LRG' not in fcov.options['tracer'] or fcov.options['zrange'] != (0.8, 1.1): continue
        #if fcov.options['region'] != 'GCcomb': continue
        has_compressed = any(name in fcov.options['observable'] for name in ['shapefit', 'bao'])
        if only_2pt and has_compressed: continue
        if (not only_2pt) and not has_compressed: continue
        #if fcov.options['region'] == 'GCcomb': continue
        observables = fcov.options['observable'].split('+')
        logger.info('Computing covariance for observables {}.'.format(observables))
        covariances = {}
        imock = list(range(1, 1001))
        shotnoise = {}
        for observable in observables:
            options = dict(fcov.options)
            if observable in ['power', 'correlation', 'power-recon', 'correlation-recon']:
                covariances[observable] = []
                if 'recon' in observable: options['cut'] = None
                for fi in fm_ez.select(id='{}_ez_y1'.format(observable.replace('-', '_')), **options, njack=0, imock=imock, ignore=True):
                    ells = (0, 2, 4)
                    if 'power' in observable:
                        power = PowerSpectrumMultipoles.load(fi).select((0., 0.4, 0.005))
                        covariances[observable].append({'value': power(ells, complex=False), 'x': [power.k] * len(ells), 'edges': [power.edges[0]] * len(ells), 'weights': [power.nmodes] * len(ells), 'projs': ells})
                        shotnoise[observable] = power.shotnoise
                        #if np.isnan(power(ells, complex=False)).any(): raise ValueError

                    if 'correlation' in observable:
                        corr = TwoPointCorrelationFunction.load(fi).select((0., 200., 4.))
                        weights = np.mean(corr.R1R2.normalized_wcounts(), axis=-1)  # mean over mu
                        sep, xi = corr(ell=ells, ignore_nan=True, return_sep=True, return_std=False)
                        covariances[observable].append({'value': xi, 'x': [sep] * len(ells), 'edges': [corr.edges[0]] * len(ells), 'projs': ells})
                        #if np.isnan(xi).any(): raise ValueError
            elif observable in ['bao-recon']:
                observable = 'bao-recon'
                covariances[observable] = []
                this_options = {**options, **get_bao_baseline_fit_setup(tracer=options['tracer'], zrange=options['zrange'], observable='correlation', recon=True, iso=None)}
                #print(this_options)
                #print(fm_ez.select(id='profiles_bao_recon_ez_y1', **this_options, imock=imock, ignore=True))
                for fi in fm_ez.select(id='profiles_bao_recon_ez_y1', **this_options, imock=imock, ignore=True):
                    profiles = Profiles.load(fi)

                    #print(fi)
                    iso_bao = 'qap' not in fi.options['template']
                    #print(profiles.bestfit['qiso'], profiles.bestfit['qap'], np.argmax(profiles.bestfit.logposterior), profiles.error['qiso'], profiles.error['qap'])
                    #assert profiles.bestfit.attrs['is_valid']
                    params = profiles.bestfit.params(varied=True).select(basename=['qiso'] if iso_bao else ['qiso', 'qap'])
                    value = profiles.bestfit.choice(index='argmax', params=params, return_type='nparray')
                    assert len(profiles.bestfit) == 9
                    obs = {'value': value, 'projs': params.basenames()}
                    if not iso_bao:
                        qiso, qap = value
                        obs['value'] = [qiso * qap**(2. / 3.), qiso * qap**(-1. / 3.)]
                        obs['projs'] = ['qpar', 'qper']
                    covariances[observable].append(obs)
            elif observable in ['shapefit']:
                observable = 'shapefit'
                covariances[observable] = []
                this_options = {**options, **get_fs_baseline_fit_setup(tracer=options['tracer'], zrange=options['zrange'], observable='power'), 'template': 'shapefit-qisoqap'}
                this_options['lim'] = options['klim']
                this_options['syst'] = 'rotation-hod-photo'
                this_options['emulator'] = False
                for fi in fm_ez.select(id='profiles_full_shape_ez_y1', **this_options, imock=imock, ignore=True):
                    profiles = Profiles.load(fi)
                    #assert profiles.bestfit.attrs['is_valid']
                    #for param in profiles.bestfit.params(solved=True):
                    #    assert param.derived == '.best'
                    #print(-2. * profiles.bestfit.logposterior.max() / profiles.bestfit.attrs['ndof'])
                    params = profiles.bestfit.params(varied=True).select(basename=['qiso', 'qap', 'df', 'dm']).sort(['qiso', 'qap', 'df', 'dm'])
                    value = profiles.bestfit.choice(index='argmax', params=params, return_type='nparray')
                    obs = {'value': value, 'projs': params.basenames()}
                    covariances[observable].append(obs)
            else:
                raise ValueError('unknown observable {}')

        covariance = ObservableCovariance.from_observations(covariances)
        if shotnoise:
            for observable, value in shotnoise.items():
                #print(observable, np.mean(value))
                covariance.observables(observable).attrs['shotnoise'] = np.mean(value)
        assert covariance.nobs == len(imock)
        covariance.save(fcov)

    
def export_hybrid_covariance():
    import os
    import logging
    import numpy as np
    from desi_y1_files import get_data_file_manager

    logger = logging.getLogger('Covariance')

    fm = get_data_file_manager(conf='unblinded')
    
    def get_std_corr(cov):
        var = np.diag(cov)
        std = np.sqrt(var)
        corr = cov / (std[:, None] * std)
        return std, corr
    
    for fcov in fm.select(id='covariance_y1', source='hybrid').iter(intersection=False):
        options = dict(fcov.options)
        options.pop('version')
        observables = options['observable'].split('+')
        if len(observables) == 1: continue
        covez = fm.get(id='covariance_y1', **{**options, 'source': 'ezmock'}).load()
        covez = covez.select(observables='power*', xlim=(0., 0.4, 0.005))  # range for thecov
        covez = covez.select(observables='corr*', xlim=(20., 200., 4.))  # range for rascalc
        cov = covez.view()
        std, corr = get_std_corr(cov)
        options.pop('source')
        for iobs, observable in enumerate(observables):
            if 'bao' not in observable:
                source = 'thecov' if 'power' in observable else 'rascalc'
                ana_options = {**options, 'source': source, 'observable': observable}
                if 'recon' in observable: ana_options['cut'] = None
                covana = fm.get(id='covariance_y1', **ana_options).load()
                observable = covez.observables(iobs)
                x = observable.xavg(method='mid', projs=observable.projs)
                covana = covana.xmatch(x=x, projs=observable.projs, method='mid').view()
                stdana, corrana = get_std_corr(covana)
                index = covez._index(observables=observable)
                std[index] = stdana
                corr[np.ix_(index, index)] = corrana
        value = corr * (std[:, None] * std)
        covez = covez.clone(value=value)
        covez.plot(fn=os.path.splitext(fcov)[0] + '.png')
        covez.save(fcov)
    
    
def rotate(of=None, only_2pt=True):
    import numpy as np
    from y1_data_2pt_tools import postprocess_rotate_wmatrix
    from desi_y1_files import get_data_file_manager, get_abacus_file_manager, get_ez_file_manager, WindowRotation
    dfm = get_data_file_manager(conf='unblinded')
    #tfm = get_data_file_manager(conf='unblinded')
    if of is None: of = ['data', 'abacus', 'ez']

    def run(fm, dfm, fs, of):
        # fm: where pk are
        # dfm: where covariance is
        # fs: where rotation is
        # Save standard covariance
        if False: #'data' not in of:
            for fi in get_data_file_manager(conf='unblinded').select(id='covariance_y1', ignore=True).iter(intersection=False):
                cov_options = fi.options
                if fi.exists():
                    covariance = fi.load()
                    for fi in dfm.select(id='covariance_{}_y1'.format(of), **cov_options, ignore=True).iter(intersection=False):
                        covariance.save(fi)

        if of == 'data': of = ''
        else: of = '_{}'.format(of)
        for fi in fs:
            fid = fi.id
            options = dict(fi.options)
            options.pop('nran')
            cov_options = dict(options)
            #cov_options.pop('source')
            cov_options.pop('version')
            for cov_options in dfm.select(id='covariance_rotated{}_y1'.format(of), **cov_options, ignore=True).iter_options(intersection=False):
                print(cov_options)
                #if cov_options['source'] != 'ezmock': continue
                if 'power' not in cov_options['observable'].split('+'): continue
                has_compressed = any(name in cov_options['observable'] for name in ['shapefit', 'bao'])
                if only_2pt and has_compressed: continue
                if (not only_2pt) and not has_compressed: continue
                #cov_options.pop('observable')
                #print(dfm.select(id='covariance_y1', **cov_options, ignore=True))
                covariance = dfm.get(id='covariance{}_y1'.format(of), **cov_options, ignore=True)
                #covariance.load().save(fm.get(id='covariance_y1', **cov_options, ignore=True))
                #output_covariance = fm.get(id='covariance_rotated{}_y1'.format(of), **{**cov_options, 'marg': False}, ignore=True)
                #output_covariance_marg = fm.get(id='covariance_rotated{}_y1'.format(of), **{**cov_options, 'marg': 'rotation'}, ignore=True)
                output_covariance = dfm.get(id='covariance_rotated{}_y1'.format(of), **cov_options, ignore=True)
                output_covariance_marg = None
                output_covariance_syst = None
                if cov_options['observable'] == 'power' and cov_options['source'] == 'ezmock':
                    cov_options.pop('version')
                    output_covariance_syst = dfm.get(id='covariance_syst_rotated{}_y1'.format(of), **{**cov_options, 'source': 'syst', 'syst': 'rotation'}, ignore=True)
                #output_covariance = output_covariance_marg = output_covariance_syst = None
                postprocess_rotate_wmatrix(fi, covariance=covariance, output_covariance=output_covariance, output_covariance_marg=output_covariance_marg, output_covariance_syst=output_covariance_syst)
            continue
            if of:
                limock = fm.select(id='power{}_y1'.format(of), **options, ignore=True).options['imock']
                data = [fm.get(id='power{}_y1'.format(of), **options, imock=iimock, ignore=True) for iimock in limock]
                data_rotated = [fm.get(id='power_rotated{}_y1'.format(of), **options, imock=iimock, ignore=True) for iimock in limock]
            else:
                data = [fm.get(id='power_y1', **options, ignore=True)]
                data_rotated = [fm.get(id='power_rotated_y1', **options, ignore=True)]
            postprocess_rotate_wmatrix(fi, power=data, output_power_marg=data_rotated)

    cut = ('theta', 0.05)
    if 'data' in of:
        #fm = dfm
        if 'blinded' in of:
            fm = get_data_file_manager(conf='blinded').select(version='v1.2')
        else:
            fm = get_data_file_manager(conf='unblinded').select(version='v1.5')
        fs = fm.select(id='rotation_wmatrix_power_y1', region='GCcomb', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], weighting='default_FKP', cut=cut).iter(intersection=False)
        run(fm, dfm, fs, of='data')
    if 'abacus' in of:
        fm = get_abacus_file_manager()
        fm = fm.select(tracer=['BGS_BRIGHT-21.5'], version='v1') + fm.select(tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v4_2')
        fs = fm.select(id='rotation_wmatrix_power_merged_abacus_y1', region='GCcomb', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], cut=cut).iter(intersection=False)
        fm = get_abacus_file_manager()
        run(fm, fm, fs, of='abacus')
    if 'ez' in of:
        fm = get_ez_file_manager()
        # v1 rotation is based on v1noric EZmocks
        fs = fm.select(id='rotation_wmatrix_power_merged_ez_y1', region='GCcomb', version='v1', #version='v1ric',
                       tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], cut=cut).iter(intersection=False)
        run(fm, fm, fs, of='ez')


if __name__ == '__main__':
    """
    1) run 2pt measurements for data and mocks (EZ and abacus).
    2) run this script with ``only_2pt = True``.
    3) run y1_systematic_templates with ``only_2pt = True`` to build 'forfit_*' files, including systematic covariances.
    4) run EZmock BAO fits / shapefit
    5) run this script with ``only_2pt = False``.
    6) run y1_systematic_templates with ``only_2pt = False`` to build 'forfit_*' files for power + BAO recon combination, including systematic covariances.
    """

    from desipipe import setup_logging

    setup_logging()

    todo = ['ezmock', 'rascalc', 'thecov', 'hybrid', 'rotate'][0:]
    only_2pt = False

    if only_2pt:

        if 'rascalc' in todo:
            export_rascalc_covariance()

        if 'thecov' in todo:
            export_thecov_covariance()

        if 'hybrid' in todo:
            export_hybrid_covariance()
            
    if 'ezmock' in todo:
        export_ezmock_covariance(only_2pt=only_2pt)

    if 'rotate' in todo:
        rotate(of='data', only_2pt=only_2pt)
        rotate(of='abacus', only_2pt=only_2pt)
        rotate(of='ez', only_2pt=only_2pt)
