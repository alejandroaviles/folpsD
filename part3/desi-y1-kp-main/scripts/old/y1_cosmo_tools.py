version, conf = 'v1', 'wrong'
#version, conf = 'test', {}

from y1_data_fits_tools import get_observable_likelihood


def load_chain(fi, burnin=0.5, concatenate=True):
    import glob
    from desilike.samples import Chain
    fi = str(fi)
    fi_desilike = glob.glob(fi + '_*.npy')
    if fi_desilike:
        chains = [Chain.load(ff).remove_burnin(burnin) for ff in fi_desilike]
    else:
        from cobaya.output import load_samples
        chains = [Chain.from_getdist(chain.to_getdist()).remove_burnin(burnin) for chain in load_samples(fi, combined=False)]
    conversions = {'ln_A_s_1e10': 'logA'}
    for chain in chains:
        for name, newname in conversions.items():
            if name in chain:
                chain[newname] = chain.pop(name)
    if concatenate:
        chains = chains[0].concatenate(chains)
    return chains


def load_planck_chain(model, dataset='plikHM_TTTEEE_lowl_lowE', params=None):
    from desilike.likelihoods.cmb import read_planck2018_chain

    basename = model + '_' + dataset
    return read_planck2018_chain(basename=basename, params=params)


def get_desilike_desi_gaussian_compression_likelihood(cosmo=None, tracers=None, save_emulator=False, emulator_fn=None, template_name='bao', version=version, conf=conf):

    from desi_y1_files.file_manager import get_cosmo_file_manager, get_cosmo_setup
    fm = get_cosmo_file_manager(version=version, conf=conf)

    from desilike import LikelihoodFisher
    from desilike.observables.galaxy_clustering import BAOCompressionObservable, ShapeFitCompressionObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    Observable = {'bao': BAOCompressionObservable, 'shapefit': ShapeFitCompressionObservable}[template_name]
    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'QSO': [(0.8, 2.1)], 'Lya': [(1.8, 4.2)]}

    if cosmo is None:
        from desilike.theories import Cosmoprimo
        cosmo = Cosmoprimo(fiducial='DESI')
        cosmo.init.params = get_cosmo_setup(model='base', dataset=template)

    observables, namespaces = [], []
    id = 'gaussian_bao_y1' if 'bao' in template_name else 'gaussian_compression_y1'
    #template = template_name
    #id = 'gaussian_compression_y1'
    for fi in fm.select(id=id, ignore=True).iter(intersection=False):
        iz = list_zrange[fi.options['tracer']].index(fi.options['zrange'])
        tracer = fi.options['tracer'][:3]
        if (tracer.upper(), fi.options['zrange']) == ('ELG', (0.8, 1.1)): continue
        namespace = '{tracer}_{iz}'.format(tracer=tracer, iz=iz)
        if tracers:
            if '_' in tracers:
                if namespace.lower() not in tracers: continue
            elif namespace[:3].lower() not in tracers: continue
        fisher = LikelihoodFisher.load(fi)
        observable = Observable(data=fisher, covariance=fisher, cosmo=cosmo, quantities=fisher.params(), fiducial='DESI', z=fisher.attrs['zeff'])
        observables.append(observable)
        namespaces.append(namespace)

    if emulator_fn is True:
        dataset = {'bao': 'desi-bao-gaussian', 'shapefit': 'desi-shapefit-gaussian'}[template_name]
        emulator_fn = {namespace: fm.get(id='emulator_y1', dataset=dataset, namespace=namespace) for namespace in namespaces}

    if save_emulator:
        cosmo.init.params['tau_reio'].update(fixed=True)
        for observable in observables: observable()
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator(observables, engine=TaylorEmulatorEngine(order=4, method='finite'))
        emulator.set_samples()
        emulator.fit()
        observables = emulator.to_calculator()
        if emulator.mpicomm.rank == 0:
            if isinstance(emulator_fn, dict):
                for observable, namespace in zip(observables, namespaces):
                    observable.save(emulator_fn[namespace])
            else:
                observable.save(emulator_fn)
    elif emulator_fn is not None:
        from desilike.emulators import EmulatedCalculator
        if isinstance(emulator_fn, dict):
            observables = [EmulatedCalculator.load(emulator_fn[namespace]) if namespace in emulator_fn else None for namespace in namespaces]
        else:
            observables = EmulatedCalculator.load(emulator_fn)
    likelihoods = []
    for observable, namespace in zip(observables, namespaces):
        for param in getattr(cosmo, 'all_params', []):
            if param in observable.init.params:
                observable.init.params.set(param)  # to set fixed
        likelihood = ObservablesGaussianLikelihood(observables=[observable], name=f'desi_{template_name}_{namespace}')
        likelihoods.append(likelihood)
    return likelihoods


def get_desilike_desi_2pt_likelihood(cosmo=None, tracers=None, observable_name='power', get_observable_likelihood=get_observable_likelihood, save_emulator=False, emulator_fn=None, solve=True, **kwargs):

    from desi_y1_files.file_manager import get_cosmo_file_manager, get_cosmo_setup

    fm = get_cosmo_file_manager()

    if cosmo is None:
        from desilike.theories import Cosmoprimo
        cosmo = Cosmoprimo(fiducial='DESI')
        cosmo.init.params = get_cosmo_setup(model='base', dataset='full-shape')

    if tracers is not None:
        tracers = tracers.lower()

    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'QSO': [(0.8, 2.1)], 'Lya': [(1.8, 4.2)]}

    #params = {}
    observables, namespaces, latex_namespaces = [], [], []
    for fi in fm.select(id='{}_y1'.format(observable_name), ignore=True).iter(intersection=False):
        iz = list_zrange[fi.options['tracer']].index(fi.options['zrange'])
        tracer = fi.options['tracer'][:3]
        #if (tracer, fi.options['zrange']) == ('ELG', (0.8, 1.1)): continue
        namespace = '{tracer}_{iz}'.format(tracer=tracer, iz=iz)
        if tracers:
            if '_' in tracers:
                if namespace.lower() not in tracers: continue
            elif namespace[:3].lower() not in tracers: continue
        namespace = '_'.join([observable_name, namespace])
        di = {'power': 'P(k)', 'correlation': r'\xi(s)', 'power_recon': 'P(k)', 'correlation_recon': r'\xi(s)'}
        latex_namespace = '{}, {}, {}'.format(di.get(observable_name, observable_name.replace('\_', '_')), tracer, iz)
        latex_namespaces.append(latex_namespace)
        covariance = fm.get(id='covariance_{}_y1'.format(observable_name), **fi.options, ignore=True)
        if 'power' in observable_name:
            wmatrix = fm.get(id='wmatrix_power_y1', **fi.options, ignore=True)
        else:
            wmatrix = None
        theory_name = fi.options.get('theory', 'velocileptors')
        template_name = 'bao-cosmo' if 'bao' in theory_name else 'direct'
        #template_name = 'direct'
        #attrs = fi.load().attrs
        #params.update(attrs['cosmo_params'])
        #for param, value in attrs['nuisance_params'].items():
        #    params['.'.join([namespace, param])] = value
        likelihood = get_observable_likelihood(data=fi, covariance=covariance, wmatrix=wmatrix, theory_name=theory_name, template_name=template_name, cosmo=cosmo, **{**fi.options, **kwargs})
        observables.append(likelihood.observables[0])
        namespaces.append(namespace)

    if emulator_fn is True:
        dataset = {'power': 'desi-full-shape-power', 'correlation_recon': 'desi-bao-correlation-recon'}[observable_name]
        emulator_fn = {namespace: fm.get(id='emulator_y1', dataset=dataset, theory=theory_name, namespace=namespace) for namespace in namespaces}

    ptname = 'template' if 'bao' in theory_name else 'pt'

    if save_emulator:
        cosmo.init.params['tau_reio'].update(fixed=True)
        for observable in observables: observable()
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator([getattr(observable.wmatrix.theory, ptname) for observable in observables], engine=TaylorEmulatorEngine(order=4, method='finite'))
        emulator.set_samples()
        emulator.fit()
        pts = emulator.to_calculator()
        if emulator.mpicomm.rank == 0:
            if isinstance(emulator_fn, dict):
                for pt, namespace in zip(pts, namespaces):
                    pt.save(emulator_fn[namespace])
            else:
                emulator.save(emulator_fn)
    elif emulator_fn is not None:
        from desilike.emulators import EmulatedCalculator
        if isinstance(emulator_fn, dict):
            pts = [EmulatedCalculator.load(emulator_fn[namespace]) if namespace in emulator_fn else None for namespace in namespaces]
        else:
            pts = EmulatedCalculator.load(emulator_fn)
    else:
        pts = [None] * len(observables)

    from desilike.likelihoods import ObservablesGaussianLikelihood
    likelihoods = []
    for observable, namespace, latex_namespace, pt in zip(observables, namespaces, latex_namespaces, pts):
        theory = observable.wmatrix.theory
        if pt is not None:
            theory.init[ptname] = pt
        else:
            pt = getattr(theory, ptname)
        for param in getattr(cosmo, 'all_params', []):
            if param in pt.init.params:
                pt.init.params.set(param)  # to set fixed
        for param in theory.init.params:
            param.update(namespace=namespace)
            latex = param.latex(namespace=latex_namespace, inline=False)
            param.update(latex=latex)
        for param in theory.init.params.select(basename=['dbeta', 'sigmapar', 'sigmaper', 'sigmas']):
            param.update(fixed=True)
        if solve:
            for param in theory.init.params.select(basename=['alpha*', 'sn*', 'c*']):
                #if param.varied: param.update(derived='.auto')
                if param.varied: param.update(derived='.best')
            for param in theory.init.params.select(basename=['*l*_*']):
                if param.varied: param.update(derived='.prec')

        likelihood = ObservablesGaussianLikelihood(observables=[observable], name=f'desi_{namespace}')
        likelihoods.append(likelihood)
    return likelihoods


def get_desilike_desi_combined_likelihood(emulator_fn=None, observable_names=['power', 'correlation_recon'], get_desilike_desi_2pt_likelihood=get_desilike_desi_2pt_likelihood, **kwargs):
    from desilike.likelihoods import ObservablesGaussianLikelihood
    split_likelihoods = [get_desilike_desi_2pt_likelihood(emulator_fn=emulator_fn, observable_name=observable_name, **kwargs) for observable_name in observable_names]
    likelihoods = []
    for split_likelihoods in zip(*split_likelihoods):  # same number of likelihoods in each
        observables, namespaces = [], []
        for split_likelihood in split_likelihoods:
            observable = split_likelihood.observables[0]
            observables.append(observable)
            for param in observable.wmatrix.theory.init.params:
                if param.namespace:
                    namespaces.append(param.namespace)
                    break
        namespace = '_'.join(namespaces)
        likelihood = ObservablesGaussianLikelihood(observables=observables, name=f'desi_{namespace}')  # no common covariance for now
        likelihoods.append(likelihood)
    return likelihoods


def get_desilike_planck_likelihood(dataset='planck2018', cosmo=None, save_emulator=False, emulator_fn=None):

    from desi_y1_files.file_manager import get_cosmo_setup

    from desilike.theories import Cosmoprimo
    from desilike.likelihoods.cmb import (BasePlanck2018GaussianLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood,
                                          TTHighlPlanck2018PlikLikelihood, TTTEEEHighlPlanck2018PlikLikelihood,
                                          TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood, LensingPlanck2018ClikLikelihood, FullGridPlanck2018GaussianLikelihood)

    if cosmo is None:
        from desilike.theories import Cosmoprimo
        cosmo = Cosmoprimo(fiducial='DESI')
        cosmo.init.params = get_cosmo_setup(model='base', dataset='planck')

    likelihoods, namespaces = [], []
    dataset = dataset.lower()
    if not any(name in dataset for name in ['ttteee', 'tt', 'lowl', 'lowe', 'lensing']):
        dataset = dataset + '-ttteeelowllowe'

    if 'gaussian' in dataset:
        model = 'base'
        if 'w0_fld' in cosmo.varied_params:
            model += '_w'
            if 'wa_fld' in cosmo.varied_params: model += '_wa'
        if 'Omega_k' in cosmo.varied_params: model += '_omegak'
        if 'm_ncdm' in cosmo.varied_params: model += '_mnu'
        basename = model + '_plikHM'
        if 'ttteee' in dataset: basename += '_TTTEEEE'
        elif 'tt' in dataset: basename += '_TT'
        if 'lowl' in dataset: basename += '_lowl'
        if 'lowe' in dataset: basename += '_lowE'
        if 'lensing' in dataset: basename += '_lensing'
        if model != 'base': basename += '_BAO'
        return FullGridPlanck2018GaussianLikelihood(cosmo=cosmo, basename=basename, weights='cmb_only')
    lite = 'lite' in dataset
    if 'ttteee' in dataset:
        likelihoods.append(TTTEEEHighlPlanck2018PlikLiteLikelihood() if lite else TTTEEEHighlPlanck2018PlikLikelihood())
        namespaces.append('ttteee')
    elif 'tt' in dataset:
        likelihoods.append(TTHighlPlanck2018PlikLiteLikelihood() if lite else TTHighlPlanck2018PlikLiteLikelihood())
        namespaces.append('tt')
    if 'lowl' in dataset:
        likelihoods.append(TTLowlPlanck2018ClikLikelihood())
        namespaces.append('lowl')
    if 'lowe' in dataset:
        likelihoods.append(EELowlPlanck2018ClikLikelihood())
        namespaces.append('lowe')
    if 'lensing' in dataset:
        likelihoods.append(LensingPlanck2018ClikLikelihood())
        namespaces.append('lensing')

    for likelihood in likelihoods:
        likelihood.init.update(cosmo=cosmo)

    if save_emulator:
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator([likelihood.theory for likelihood in likelihoods], engine=TaylorEmulatorEngine(order=3, method='finite'))
        emulator.set_samples()
        emulator.fit()
        theories = emulator.to_calculator()
        if emulator.mpicomm.rank == 0:
            if isinstance(emulator_fn, dict):
                for theory, namespace in zip(theories, namespaces):
                    theory.save(emulator_fn[namespace])
            else:
                emulator.save(emulator_fn)
    elif emulator_fn is not None:
        from desilike.emulators import EmulatedCalculator
        if isinstance(emulator_fn, dict):
            theories = [EmulatedCalculator.load(emulator_fn[namespace]) if namespace in emulator_fn else None for namespace in namespaces]
        else:
            theories = EmulatedCalculator.load(emulator_fn)
    else:
        theories = [None] * len(likelihoods)

    for likelihood, theory in zip(likelihoods, theories):
        if theory is not None:
            likelihood.init.update(theory=theory)
            for param in getattr(cosmo, 'all_params', []):
                if param in theory.init.params:
                    theory.init.params.set(param)  # to set fixed

    return likelihoods


def get_single_desilike_likelihood(dataset='bao-gaussian', convert='desilike', emulator_fn=None,
                                   get_desilike_desi_gaussian_compression_likelihood=get_desilike_desi_gaussian_compression_likelihood,
                                   #get_desilike_desi_full_shape_likelihood=get_desilike_desi_full_shape_likelihood,
                                   get_desilike_desi_combined_likelihood=get_desilike_desi_combined_likelihood,
                                   get_desilike_planck_likelihood=get_desilike_planck_likelihood, solve=True, **kwargs):

    import re
    from desilike.likelihoods import BaseGaussianLikelihood
    tracers = []
    for tracer in ['bgs', 'lrg', 'elg', 'qso', 'lya']:
        for traceri in re.findall(f'({tracer}_\d|{tracer})', dataset.lower()):
            tracers.append(traceri)
    if tracers: tracers = '-'.join(tracers)
    else: tracers = None

    dataset = dataset.lower()
    if 'desi-bao-gaussian' in dataset:
        get_desilike_likelihood = get_desilike_desi_gaussian_compression_likelihood
        kwargs.setdefault('tracers', tracers)
        kwargs['template_name'] = 'bao'

    elif 'desi-shapefit-gaussian' in dataset:
        get_desilike_likelihood = get_desilike_desi_gaussian_compression_likelihood
        kwargs.setdefault('tracers', tracers)
        kwargs.setdefault('emulator_fn', emulator_fn)
        kwargs['template_name'] = 'shapefit'

    elif 'desi-full-shape-power-bao-correlation-recon' in dataset:  # keep this in front, otherwise following if are triggered first
        get_desilike_likelihood = get_desilike_desi_combined_likelihood
        kwargs.setdefault('tracers', tracers)
        kwargs.setdefault('emulator_fn', emulator_fn)
        kwargs.setdefault('solve', solve)
        kwargs.setdefault('observable_names', ['power', 'correlation_recon'])

    elif 'desi-full-shape' in dataset:
        get_desilike_likelihood = get_desilike_desi_combined_likelihood
        kwargs.setdefault('tracers', tracers)
        kwargs.setdefault('emulator_fn', emulator_fn)
        kwargs.setdefault('solve', solve)
        kwargs.setdefault('observable_names', ['power'])

    elif 'desi-bao-correlation-recon' in dataset:
        get_desilike_likelihood = get_desilike_desi_combined_likelihood
        kwargs.setdefault('tracers', tracers)
        kwargs.setdefault('emulator_fn', emulator_fn)
        kwargs.setdefault('solve', solve)
        kwargs.setdefault('observable_names', ['correlation_recon'])

    else:
        
        def get_desilike_sn_likelihood(Likelihood):

            def get_desilike_likelihood(**kwargs):
                likelihood = Likelihood(**kwargs)
                for param in likelihood.init.params.select(basename=['Mb', 'dM']):
                    param.update(prior=None, derived='.prec')
                return likelihood

            return get_desilike_likelihood

        from desi_y1_files.file_manager import get_cosmo_setup
        if dataset == 'bbn-omega_b':
            from desilike.likelihoods.bbn import BBNOmegaBLikelihood
            get_desilike_likelihood = BBNOmegaBLikelihood
        elif dataset == 'union3':
            from desilike.likelihoods.supernovae import Union3SNLikelihood
            get_desilike_likelihood = get_desilike_sn_likelihood(Union3SNLikelihood)
        elif dataset == 'pantheon+shoes':
            from desilike.likelihoods.supernovae import PantheonPlusSHOESSNLikelihood
            get_desilike_likelihood = PantheonPlusSHOESSNLikelihood
        elif dataset == 'pantheon+':
            from desilike.likelihoods.supernovae import PantheonPlusSNLikelihood
            get_desilike_likelihood = get_desilike_sn_likelihood(PantheonPlusSNLikelihood)
        elif dataset == 'pantheon':
            from desilike.likelihoods.supernovae import PantheonSNLikelihood
            get_desilike_likelihood = get_desilike_sn_likelihood(PantheonSNLikelihood)
        elif dataset == 'riess2020h0':
            from desilike.likelihoods.hubble import Riess2020H0Likelihood
            get_desilike_likelihood = Riess2020H0Likelihood
        elif dataset == 'riess2020mb':
            from desilike.likelihoods.hubble import Riess2020MbLikelihood
            get_desilike_likelihood = Riess2020MbLikelihood
        elif 'planck2018' in dataset:
            kwargs['dataset'] = dataset
            kwargs.setdefault('emulator_fn', emulator_fn)
            get_desilike_likelihood = get_desilike_planck_likelihood
        else:
            raise ValueError('unknown external likelihood {}'.format(dataset))

    if convert == 'cobaya':
        from desilike.bindings.cobaya import CobayaLikelihoodFactory
        #if kwargs.get('emulator_fn', None) is None:
        kwargs['cosmo'] = 'external'

        likelihoods = get_desilike_likelihood(**kwargs)
        nlikelihoods = len(likelihoods) if isinstance(likelihoods, list) else -1

        if nlikelihoods > -1:

            [likelihood.all_params for likelihood in likelihoods]  # initialization, to set likelihood.name

            def get_single_desilike_likelihood(ilike=0):
                return likelihoods[ilike]

            from desilike.bindings.base import get_likelihood_params
            toret = []
            for ilike in range(nlikelihoods):
                kw_like = {'ilike': ilike}
                cosmo_params, nuisance_params = get_likelihood_params(get_single_desilike_likelihood(**kw_like), derived=0)
                if ilike > 0:
                    for param in list(nuisance_params):
                        if param.derived is True and not param.namespace:  # avoid duplicates of sigma8_m, Omega_m
                            del nuisance_params[param]
                toret.append(CobayaLikelihoodFactory(get_single_desilike_likelihood, name_like=likelihoods[ilike].name, kw_like=kw_like, params=nuisance_params))
            
            #return [CobayaLikelihoodFactory(get_single_desilike_likelihood, name_like=likelihoods[ilike].name, kw_like={'ilike': ilike}, params=True) for ilike in range(nlikelihoods)]
            return toret

        likelihoods.all_params  # initialization, to set likelihood.name
        
        from desilike.bindings.base import get_likelihood_params

        def get_single_desilike_likelihood():
            return likelihoods

        return CobayaLikelihoodFactory(get_single_desilike_likelihood, name_like=likelihoods.name, kw_like={}, params=True)

    return get_desilike_likelihood(**kwargs)


def get_desilike_likelihoods(dataset='desi-bao-gaussian', convert='desilike', emulator_fn=None, cosmo=None,
                             get_single_desilike_likelihood=get_single_desilike_likelihood, **kwargs):

    from desi_y1_files.file_manager import get_cosmo_setup

    if cosmo is None:
        from desilike.theories import Cosmoprimo
        cosmo = Cosmoprimo(fiducial='DESI')
        cosmo.init.params = get_cosmo_setup(model='base', dataset=dataset)

    isscalar = isinstance(dataset, str)
    if not isscalar:
        emulator_fns = dict(emulator_fn or {})
        likelihoods = []
        for dataset in dataset:
            emulator_fn = emulator_fns.get(dataset, None)
            if isinstance(emulator_fn, dict) and set(emulator_fn.keys()) == {None}:
                for emulator_fn in emulator_fn.values(): break
            likelihood = get_single_desilike_likelihood(dataset=dataset, convert=convert, cosmo=cosmo, emulator_fn=emulator_fn, **kwargs)
            if isinstance(likelihood, list):
                likelihoods += likelihood
            else:
                likelihoods.append(likelihood)
        return likelihoods
    return get_single_desilike_likelihood(dataset=dataset, convert=convert, cosmo=cosmo, emulator_fn=emulator_fn, **kwargs)


def get_bao_rsd_forecasts(version='test', region='GCcomb', apmode='qparqper'):

    from desilike.theories.galaxy_clustering import (BAOPowerSpectrumTemplate, SimpleBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerPowerSpectrumMultipoles, StandardPowerSpectrumTemplate, SimpleTracerPowerSpectrumMultipoles)
    from desilike.likelihoods.galaxy_clustering import SNWeightedPowerSpectrumLikelihood
    from desilike import Fisher
    from desi_y1_files.file_manager import get_data_file_manager, list_zrange, get_fit_setup
    from y1_data_2pt_tools import get_footprint

    dfm = get_data_file_manager().select(version=version)

    fishers_bao, fishers_rsd = [], []

    def get_recon_factor(template, shotnoise, **params):
        """Reconstruction damping factor."""
        from scipy import special
        from scipy.interpolate import UnivariateSpline
        from desilike.theories.galaxy_clustering import SimpleTracerPowerSpectrumMultipoles
        k_pivot, mu_pivot = 0.14, 0.6
        np = [0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 6.0, 10.0]
        r = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5]
        spline = UnivariateSpline(np, r, k=3, s=0., ext='const', check_finite=False)
        theory = SimpleTracerPowerSpectrumMultipoles(k=[k_pivot], template=template)
        poles = theory(**params)
        np = sum(poles[ill] * special.legendre(ell)(mu_pivot) for ill, ell in enumerate(theory.ells)) / shotnoise
        return spline(np / 0.1734)

    for tracer, zranges in list_zrange.items():
        data = dfm.select(id='catalog_data_y1', tracer=tracer)
        all_randoms = [dfm.select(id='catalog_randoms_y1', tracer=tracer, iran=iran) for iran in range(1)]
        for zrange in zranges:
            footprint = get_footprint(data, all_randoms=all_randoms, region='GCcomb', zrange=zrange)
            b0 = get_fit_setup(tracer, return_list='b0')
            cosmo, z = footprint.cosmo, footprint.attrs['zeff']
            fo = cosmo.get_fourier()
            template = StandardPowerSpectrumTemplate(z=z, fiducial=cosmo, apmode=apmode)
            s, s0 = fo.sigma8_z(z, of='delta_cb'), fo.sigma8_z(0., of='delta_cb')
            b1 = b0 / (s / s0)  # prescription for linear bias
            r = get_recon_factor(template, footprint.shotnoise, b1=b1)
            f = fo.sigma8_z(z, of='theta_cb') / s
            sigmaper = r * 9.4 * (s / 0.9)
            sigmapar = (1. + f) * sigmaper
            #sigmaper, sigmapar = {'LRG': (2.6, 6.6)}.get(tracer, (sigmaper, sigmapar))
            # print(r, b1, b1 * fo.sigma8_z(z, of='delta_cb'), sigmaper, sigmapar, footprint.size, footprint.area, footprint.attrs['zeff'])
            params = {'b1': b1, 'sigmapar': sigmapar, 'sigmaper': sigmaper}  # fiducial model parameters
            covariance_params = {'b1': b1, 'sigmapar': 0., 'sigmaper': 0.}  # fiducial covariance parameters (simple Kaiser model)
            template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI', apmode=apmode)
            #theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
            theory = SimpleBAOWigglesTracerPowerSpectrumMultipoles(template=template)  # this BAO model shifts wiggles only
            for param in theory.params.select(basename='al*'):
                param.update(value=0., fixed=True)  # fixing broadband parameters
            # For klim=(0.01, 0.5), we only use the information from the BAO feature in the power spectrum
            likelihood = SNWeightedPowerSpectrumLikelihood(theories=theory, data=params, covariance=covariance_params, footprints=footprint, klim=(0.01, 0.5))
            fisher = Fisher(likelihood)  # initializing Fisher
            fisher_bao = fisher(**params).view(params=['qpar', 'qper'] if apmode == 'qparqper' else ['qiso', 'qap'])  # computing Fisher prediction at fiducial parameters
            template = StandardPowerSpectrumTemplate(z=z, fiducial='DESI', apmode=apmode)  # here we use a standard power spectrum template, to vary f
            theory = SimpleTracerPowerSpectrumMultipoles(template=template)  # this is a damped Kaiser model
            # For klim=(0.01, 0.1), we use the RSD signal (f, b varied)
            likelihood = SNWeightedPowerSpectrumLikelihood(theories=theory, data=params, covariance=covariance_params, footprints=footprint, klim=(0.01, 0.1))
            for param in likelihood.all_params:
                if param.basename not in ['df', 'b1']: param.update(fixed=True)  # fixing all parameters (including shot noise) except f and b1
            fisher = Fisher(likelihood)  # initializing Fisher
            fisher_rsd = fisher(**params)  # computing Fisher prediction at fiducial parameters
            attrs = {'zeff': z, 'tracer': tracer, 'zrange': zrange}
            fisher_bao.attrs = fisher_rsd.attrs = attrs
            fishers_bao.append(fisher_bao)
            fishers_rsd.append(fisher_rsd)

    return fishers_bao, fishers_rsd


def save_local(version=version, conf=conf):
    import os
    import shutil

    import numpy as np
    from scipy import special
    from desi_y1_files import get_data_file_manager, get_cosmo_file_manager
    from desi_y1_files.file_manager import get_cosmo_setup
    from desi_y1_files.cosmo_tools import get_bao_apmode, get_bao_params, predict_bao
    from desilike import LikelihoodFisher
    from desilike.samples import Chain, Profiles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable
    from pypower import PowerSpectrumMultipoles

    dfm = get_data_file_manager(conf='wrong')
    cfm = get_cosmo_file_manager(version=version, conf=conf)

    def get_bao(fisher, conf=''):
        apmode = get_bao_apmode(fisher.params())
        params = get_bao_params(apmode)
        if isinstance(conf, dict):
            from cosmoprimo.fiducial import DESI
            fiducial = DESI()
            cosmo = fiducial.clone(**conf)
            center = predict_bao(fisher.attrs['zeff'], apmode=apmode, cosmo=cosmo)
        elif 'mock' in conf:
            center = [1.] * len(params)
        else:
            center = fisher.mean(params)
        assert not np.allclose(center, 1.)
        return fisher.clone(center=center, params=params)

    def get_pre_power(fpoles, fprofiles, conf=''):
        poles = fpoles.load()
        attrs = dict(poles.attrs)
        if not isinstance(conf, dict) and not 'mock' in conf:
            return poles
        profiles = Profiles.load(fprofiles)
        observable = TracerPowerSpectrumMultipolesObservable.from_state(profiles.attrs['observable'])
        if not all(len(k) == len(observable.k[0]) for k in observable.k):
            raise NotImplementedError('multipole-dependent k-cut not implemented')
        if isinstance(conf, dict):
            from desilike.theories import Cosmoprimo
            cosmo = Cosmoprimo(fiducial='DESI')
            cosmo.init.params = get_cosmo_setup(model='base_omegak_w_wa_mnu', dataset='full-shape')
            options = fprofiles.options
            data = dfm.get(id='power_y1', **options, ignore=True)
            wmatrix = dfm.get(id='wmatrix_power_y1', **options, ignore=True)
            likelihood = get_observable_likelihood(data=data, wmatrix=wmatrix, covariance=np.eye(len(observable.flatdata)), theory_name=options['theory'], template_name='direct', cosmo=cosmo, solve=False, **options)
            observable = likelihood.observables[0]
            #conf = {**profiles.bestfit.choice(input=True, index='argmax'), **conf}
            conf = {**profiles.bestfit.choice(params=['b1'], index='argmax'), **conf}
            #conf['b1'] += 1.  # velocileptors -> velocileptors
            #print(conf)
            print(conf)
            observable(**conf)
            attrs['cosmo_params'] = {name: value for name, value in conf.items() if name in cosmo.init.params}
            attrs['nuisance_params'] = {name: value for name, value in conf.items() if name not in attrs['cosmo_params']}
        shotnoise_nonorm = poles.shotnoise
        power_nonorm = np.array_split(observable.flattheory, len(observable.ells))
        power_nonorm[0] += shotnoise_nonorm
        nmodes = np.ones_like(power_nonorm[0], dtype='i8')
        for edges in observable.kedges: break
        return PowerSpectrumMultipoles(edges=edges, modes=observable.k[0], power_nonorm=power_nonorm, nmodes=nmodes, shotnoise_nonorm=shotnoise_nonorm, ells=observable.ells, attrs=attrs)

    def get_post_correlation(fcorr, fprofiles, conf=''):
        corr = fcorr.load()
        attrs = dict(corr.D1D2.attrs)
        if not isinstance(conf, dict) and not 'mock' in conf:
            return poles
        profiles = Profiles.load(fprofiles)
        observable = TracerCorrelationFunctionMultipolesObservable.from_state(profiles.attrs['observable'])
        if not all(len(s) == len(observable.s[0]) for s in observable.s):
            raise NotImplementedError('multipole-dependent s-cut not implemented')
        if isinstance(conf, dict):
            from desilike.theories import Cosmoprimo
            cosmo = Cosmoprimo(fiducial='DESI')
            cosmo.init.params = get_cosmo_setup(model='base_omegak_w_wa_mnu', dataset='full-shape')
            options = fprofiles.options
            data = dfm.get(id='correlation_recon_y1', **options, ignore=True)
            likelihood = get_observable_likelihood(data=data, covariance=np.eye(len(observable.flatdata)), theory_name=options['theory'], template_name='direct', cosmo=cosmo, solve=False, **options)
            observable = likelihood.observables[0]
            #conf = {**profiles.bestfit.choice(input=True, index='argmax'), **conf}
            conf = {**profiles.bestfit.choice(params=['b1'], index='argmax'), **conf}
            #conf['b1'] += 1.  # velocileptors -> velocileptors
            observable(**conf)
            attrs['cosmo_params'] = {name: value for name, value in conf.items() if name in cosmo.init.params}
            attrs['nuisance_params'] = {name: value for name, value in conf.items() if name not in attrs['cosmo_params']}
        theory = np.array_split(observable.flattheory, len(observable.ells))
        corr = corr.deepcopy()
        for sedges in observable.sedges: break
        corr = corr.select((sedges[0], sedges[-1], sedges[1] - sedges[0]))
        mu = corr.sepavg(axis=1, method='mid')
        mask_nan = np.isnan(corr.corr)
        corr.D1D2.wcounts[...] = 1. + sum(theory[ill][:, None] * special.legendre(ell)(mu) for ill, ell in enumerate(observable.ells))
        corr.D1S2.wcounts[...] = corr.S1D2.wcounts[...] = corr.S1S2.wcounts[...] = corr.R1R2.wcounts[...] = 1.
        for name in corr.count_names:
            getattr(corr, name).wcounts[mask_nan] = np.nan
        corr.D1D2.attrs['theory'] = {'s': observable.s, 'corr': theory}
        return corr
    
    def load_bao_chain(fi, burnin=0.5):
        from desilike.samples import Chain
        chains = [Chain.load(ff).remove_burnin(burnin) for ff in fi]
        chain = chains[0].concatenate(chains)
        eta = 1. / 3.
        if 'qpar' in chain and 'qper' in chain:
            chain.set((chain['qpar']**eta * chain['qper']**(1. - eta)).clone(param=dict(name='qiso', derived=True, latex=r'q_{\rm iso}')))
            chain.set((chain['qpar'] / chain['qper']).clone(param=dict(name='qap', derived=True, latex=r'q_{\rm ap}')))
        if 'qiso' in chain and 'qap' in chain:
            chain.set((chain['qiso'] * chain['qap']**(1. - eta)).clone(param=dict(name='qpar', derived=True, latex=r'q_{\parallel}')))
            chain.set((chain['qiso'] * chain['qap']**(-eta)).clone(param=dict(name='qper', derived=True, latex=r'q_{\perp}')))
        return chain

    # BAO compression
    for fi in cfm.select(id='gaussian_bao_y1').iter(intersection=False):
        if 'Lya' in fi.options['tracer']:
            # BAO Lya
            #fn = '/global/cfs/cdirs/desi/science/cpe/lya/mock_y1/lyalya_lyalya__lyalya_lyalyb__lyalya_qso__lyalyb_qso-z_2.3.fits'
            #import fitsio
            #table = fitsio.read(fn, ext=2)
            #quantities = [{'ap': 'qpar', 'at': 'qper'}[name] for name in table['names']]
            #fisher = LikelihoodFisher(center=[1., 1.], params=quantities, hessian=-np.linalg.inv(table['covariance']), attrs={'zeff': 2.3})
            """
            quantities = ['qpar', 'qper']
            std = np.array([0.015, 0.019])
            corr = -0.439
            corr = np.array([[1., corr], [corr, 1.]])
            cov = corr * (std[:, None] * std)
            fisher = LikelihoodFisher(center=[1., 1.], params=quantities, hessian=-np.linalg.inv(cov), attrs={'zeff': 2.35, 'qiso': {'mean': 1., 'std': 0.009}})
            """
            fn = '/global/cfs/projectdirs/desi/users/acuceu/notebooks_perl/desi-y1/cobaya/bao_data/DESI-Y1.dat'
            z, center = np.loadtxt(fn, comments='#', usecols=[0, 1], unpack=True)
            assert np.allclose(z, 2.34)
            z = z[0]
            fn = '/global/cfs/projectdirs/desi/users/acuceu/notebooks_perl/desi-y1/cobaya/bao_data/DESI-Y1.cov'
            cov = np.loadtxt(fn, comments='#', usecols=[0, 1], unpack=True)
            quantities = ['qpar', 'qper']
            from cosmoprimo.fiducial import DESI
            from cosmoprimo import constants
            fiducial = DESI()
            DM_over_rd_fid = fiducial.comoving_angular_distance(z) / fiducial.rs_drag
            DH_over_rd_fid = (constants.c / 1e3) / (100. * fiducial.efunc(z)) / fiducial.rs_drag
            fid = np.array([DH_over_rd_fid, DM_over_rd_fid])
            center /= fid
            cov /= fid[:, None] * fid
            fisher = LikelihoodFisher(center=center, params=quantities, hessian=-np.linalg.inv(cov), attrs={'zeff': z, 'qiso': {'mean': 1.000, 'std': 0.011}})

            import pandas as pd
            table = pd.read_csv('data/blinded-lya-correlations-y1-5-2-0-0-v2.csv')
            attrs = {}
            attrs['ells'] = (0,)
            attrs['s'] = [table['R_MPCH']]
            tracer = 'LYALYB_LYA+LYALYB_QSO'
            attrs['data'] = [table['XI_{}'.format(tracer)]]
            attrs['std'] = [table['ERR_{}'.format(tracer)]]
            attrs['theory'] = [table['MODEL_{}'.format(tracer)]]
            attrs['theory_nobao'] = [table['MODEL_NO_BAO_{}'.format(tracer)]]
            fisher.attrs['observable'] = attrs
        else:
            doptions = fi.options
            print(doptions)
            #doptions = {key: value for key, value in doptions.items() if key in ['template', 'theory', 'observable', 'lim', 'sigmas', 'broadband', 'tracer', 'zrange', 'weighting', 'cut', 'mode']}
            #doptions = {key: value for key, value in doptions.items() if key in []}
            fchains = list(dfm.select(id='chains_bao_recon_y1', **doptions, ignore=True))
            chain = load_bao_chain(fchains)
            cfm.get(id='chain_bao_y1', **fi.options).save(chain)
            #fisher = chain.to_fisher(params=chain.params(basename=['qpar', 'qper', 'qiso', 'qap'], varied=True, derived=False))
            # Use qpar, qper is anisotropic, else qiso
            fisher = chain.to_fisher(params=['qpar', 'qper'] if 'qpar' in chain else ['qiso'])
            for name in ['qiso', 'qap', 'qpar', 'qper']:  # for plots
                if name in chain:
                    fisher.attrs[name] = {'mean': float(chain[name].mean()), 'std': float(chain[name].std())}
            fprofiles = dfm.get(id='profiles_bao_recon_y1', **doptions, ignore=True)
            profiles = Profiles.load(fprofiles)
            #fisher = fisher.clone(center=profiles.bestfit.choice(params=fisher.params(), index='argmax', return_type='nparray'))
            fisher.attrs.update(profiles.attrs)  # adds observable
        fisher = get_bao(fisher, conf=fi.options.get('conf', ''))
        fisher.save(fi)
    """
    # Pre-recon full shape
    # TODO: save pre-cut power spectrum / covariance matrix / window matrix, in final format (no complex structure)
    for fi in cfm.select(id='power_y1').iter(intersection=False):
        doptions = {**fi.options, **dict(weighting='default_FKP', region=['GCcomb'], observable='power', cellsize=6., theory='velocileptors', template='fixed')}
        fpoles = dfm.get(id='power_y1', **doptions, ignore=True)
        fprofiles = dfm.get(id='profiles_full_shape_y1', **doptions, ignore=True)
        poles = get_pre_power(fpoles, fprofiles, conf=fi.options.get('conf', ''))
        fi.save(poles)
        shutil.copy(dfm.get(id='covariance_power_y1', **{**doptions, 'cut': None, 'version': 'v0.6'}, ignore=True), cfm.get(id='covariance_power_y1', **fi.options, ignore=True))
        shutil.copy(dfm.get(id='wmatrix_power_y1', **doptions, ignore=True), cfm.get(id='wmatrix_power_y1', **fi.options, ignore=True))

    for fi in cfm.select(id='correlation_recon_y1').iter(intersection=False):
        doptions = {**fi.options, **dict(weighting='default_FKP', region=['GCcomb'], observable='correlation', theory='dampedbao', template='fixed')}
        fcorr = dfm.get(id='correlation_recon_y1', **doptions, ignore=True)
        fprofiles = dfm.get(id='profiles_bao_recon_y1', **doptions, ignore=True)
        corr = get_post_correlation(fcorr, fprofiles, conf=fi.options.get('conf', ''))
        fi.save(corr)
        shutil.copy(dfm.get(id='covariance_correlation_recon_y1', **{**doptions, 'cut': None, 'version': 'v0.6'}, ignore=True), cfm.get(id='covariance_correlation_recon_y1', **fi.options, ignore=True))
    """


def emulate_desilike(output, get_desilike_likelihoods=get_desilike_likelihoods, dataset=None, model=None, **kwargs):
    from desi_y1_files.file_manager import get_cosmo_setup
    from desilike.theories import Cosmoprimo

    cosmo = Cosmoprimo(fiducial='DESI')
    if model is None:
        model = output.options['model']
    if dataset is None:
        dataset = output.options['dataset']
        namespace = output.options.get('namespace', None)
        if namespace is not None:
            dataset = '-'.join([dataset, namespace])
            output = {namespace: str(output)}
    cosmo.init.params = get_cosmo_setup(model=model, dataset=dataset)
    get_desilike_likelihoods(dataset=dataset, save_emulator=True, emulator_fn=output, cosmo=cosmo, **kwargs)
    return output


def measure_speed(output=None, niterations=5, get_desilike_likelihoods=get_desilike_likelihoods, emulator_fn=None, **kwargs):
    import time
    import numpy as np
    from desi_y1_files.file_manager import get_cosmo_setup
    from desilike.theories import Cosmoprimo

    def get_option(name):
        if name in kwargs: return kwargs.pop(name)
        if output is None: return None
        return output.options[name]

    datasets = get_option('datasets')
    model = get_option('model')
    emulators = get_option('emulators')

    if emulator_fn is True:
        emulator_fn = {dataset: True for dataset in emulators}

    from pyrecon.utils import MemoryMonitor
    with MemoryMonitor() as mem:
        cosmo = Cosmoprimo(fiducial='DESI', engine='class')
        cosmo.init.params = get_cosmo_setup(model=model, dataset=datasets)
        likelihoods = get_desilike_likelihoods(datasets, emulator_fn=emulator_fn, cosmo=cosmo, **kwargs)
        likelihood = sum(likelihoods)

        rng = np.random.RandomState(seed=42)
        for i in range(2):
            params = {param.name: param.ref.sample(random_state=rng) for param in likelihood.varied_params}
            likelihood(**params)
        if likelihood.mpicomm.rank == 0:
            likelihood.log_info('Varied parameters: {}.'.format(likelihood.varied_params.names()))
        likelihood.runtime_info.pipeline._set_speed(niterations=niterations, override=False, seed=42)

    t0 = time.time()
    for i in range(niterations):
        params = {param.name: param.ref.sample(random_state=rng) for param in likelihood.varied_params}
        print(likelihood(**params))
    dt = time.time() - t0
    if likelihood.mpicomm.rank == 0:
        likelihood.log_info('Likelihood evaluation time: {:.3f} s.'.format(dt / niterations))


def sample_desilike(output, resume=True, get_desilike_likelihoods=get_desilike_likelihoods, emulator_fn=None, test=False, **kwargs):
    import os
    from desi_y1_files.file_manager import get_cosmo_setup
    from desilike.theories import Cosmoprimo
    from desilike.samplers import MCMCSampler, ZeusSampler, PocoMCSampler

    datasets = output.options['datasets']
    model = output.options['model']
    emulators = output.options.get('emulators', {})
    if emulator_fn is True:
        emulator_fn = {dataset: True for dataset in emulators}

    cosmo = Cosmoprimo(fiducial='DESI')
    cosmo.init.params = get_cosmo_setup(model=model, dataset=datasets)
    likelihoods = get_desilike_likelihoods(datasets, emulator_fn=emulator_fn, cosmo=cosmo, **kwargs)
    likelihood = sum(likelihoods)

    chains = cosmo.mpicomm.size
    resume = False
    save_fn = ['{}_{:d}.npy'.format(output.filepath, ichain + 1) for ichain in range(chains)]
    if resume and all(os.path.isfile(fi) for fi in save_fn):
        chains = save_fn

    #sampler = MCMCSampler(likelihood, chains=chains, seed=42, oversample_power=0., proposal_scale=1.9, learn={'max_eigen_gr': 30.}, save_fn=save_fn)
    sampler = ZeusSampler(likelihood, chains=chains, seed=42, save_fn=save_fn)
    #sampler = PocoMCSampler(likelihood, chains=chains, seed=42, save_fn=save_fn)
    if test:
        chains = sampler.run(check={'max_eigen_gr': 0.03}, check_every=40 * len(likelihood.varied_params), max_iterations=10)
        return
    chains = sampler.run(check={'max_eigen_gr': 0.03}, check_every=40 * len(likelihood.varied_params))
    base_save_fn = output.filepath
    if sampler.mpicomm.rank == 0:
        from desilike.samples import plotting
        for ichain, chain in enumerate(chains):
            chain.write_getdist(base_save_fn, ichain=ichain)
        chain = chains[0].concatenate([chain.remove_burnin(0.5) for chain in chains])
        try:
            for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
                chain.to_stats(tablefmt=tablefmt, fn=base_save_fn + '_stats.' + ext)
            plotting.plot_triangle(chain, fn=base_save_fn + '_triangle.png')
        except:
            import traceback
            traceback.print_exc()


def profile_desilike(output, get_desilike_likelihoods=get_desilike_likelihoods, emulator_fn=None, **kwargs):
    import os
    from desi_y1_files.file_manager import get_cosmo_setup
    from desilike.theories import Cosmoprimo
    from desilike.profilers import MinuitProfiler

    datasets = output.options['datasets']
    model = output.options['model']
    emulators = output.options.get('emulators', {})

    if emulator_fn is True:
        emulator_fn = {dataset: True for dataset in emulators}

    cosmo = Cosmoprimo(fiducial='DESI')
    cosmo.init.params = get_cosmo_setup(model=model, dataset=datasets)
    likelihoods = get_desilike_likelihoods(datasets, emulator_fn=emulator_fn, cosmo=cosmo, **kwargs)
    likelihood = sum(likelihoods)
    #for param in likelihood.all_params.select(basename=['Omega_cdm', 'Omega_b']): param.update(fixed=True)
    profiler = MinuitProfiler(likelihood, save_fn=output.filepath, seed=42)
    profiles = profiler.maximize(niterations=3)
    #profiler.interval(['Mb'])
    for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
        profiles.to_stats(tablefmt=tablefmt, fn=os.path.splitext(output)[0] + '_stats.' + ext)


def plot_bestfit(samples, save_fn='base', emulator_fn=True, **kwargs):
    import os
    from desi_y1_files.file_manager import get_cosmo_setup
    from desilike.theories import Cosmoprimo
    from desilike.samples import Profiles, Chain

    datasets = samples.options['datasets']
    model = samples.options['model']
    emulators = samples.options.get('emulators', {})

    if emulator_fn is True:
        emulator_fn = {dataset: True for dataset in emulators}

    cosmo = Cosmoprimo(fiducial='DESI')
    cosmo.init.params = get_cosmo_setup(model=model, dataset=datasets)
    likelihoods = get_desilike_likelihoods(datasets, emulator_fn=emulator_fn, cosmo=cosmo, **kwargs)
    likelihood = sum(likelihoods)
    for param in likelihood.all_params.select(solved=True):
        param.update(derived='.best')

    if 'profiles' in samples.id:
        values = Profiles.load(samples).bestfit.choice(index='argmax', input=True)
    else:  # samples
        chain = load_chain(samples)
        values = chain.choice(index='argmax', input=True)
    for i in range(2):  # 1: compute the gradient, 2: update solved parameters
        likelihood(**values)
    for likelihood in likelihood.likelihoods:
        for observable in getattr(likelihood, 'observables', []):
            plot = getattr(observable, 'plot', None)
            if plot is not None:
                for param in observable.all_params:
                    if param.namespace:
                        namespace = param.namespace
                        break
                plot(fn=f'{save_fn}_{namespace}.png')


def sample_cobaya(output, resume=True, get_desilike_likelihoods=get_desilike_likelihoods, emulator_fn=None, test=False, **kwargs):
    import os
    import glob
    from desilike import mpi
    from desi_y1_files.file_manager import get_cosmo_setup

    datasets = output.options['datasets']
    model = output.options['model']
    emulators = output.options.get('emulators', {})
    save_fn = output.filepath

    if emulator_fn is True:
        emulator_fn = {dataset: True for dataset in emulators}

    desilike_datasets = [dataset for dataset in datasets if any(name in dataset.lower() for name in ['desi', 'bbn-omega_b', 'pantheon+', 'pantheon+shoes', 'union3']) or emulators.get(dataset, None) is not None]
    desilike_likelihoods = get_desilike_likelihoods(desilike_datasets, convert='cobaya', emulator_fn=emulator_fn, **kwargs)
    likelihoods = {likelihood.__name__: likelihood for likelihood in desilike_likelihoods}
    likelihoods.update({dataset: {} for dataset in datasets if dataset not in desilike_datasets})
    renames = {'pantheon': 'sn.pantheon'}
    for name, rename in renames.items():
        if name in likelihoods:
            likelihoods[rename] = likelihoods.pop(name)
    mpicomm = mpi.COMM_WORLD
    # No magic here, this is all Cobaya stuff
    params, extra_args = get_cosmo_setup(model=model, dataset=datasets, convert='cobaya-classy', return_extra=True)
    
    covmat = covmat_params = None
    if 'planck2018' in '_'.join(datasets):
        from desilike.likelihoods.cmb import read_planck2018_chain
        basename = model + '_plikHM_TTTEEE_lowl_lowE'
        if model != 'base': basename += '_BAO'
        covmat_params = [name for name, value in params.items() if isinstance(value, dict)]  # varied params
        chain = read_planck2018_chain(basename=basename, weights='cmb_only', params=covmat_params)
        covmat_params = chain.names(varied=True)
        covmat = chain.covariance(covmat_params)
        # covmat seems to decrease the acceptance rate dramatically...
        """
        covmat_params = chain.names(name=list(params.keys()), varied=True)
        covmat = chain.covariance(covmat_params)
        print(covmat_params)
        import numpy as np
        print(np.diag(covmat)**0.5)
        """

    theory = {'classy': {'extra_args': extra_args}}
    sampler = {'mcmc': {#'drag': True,
                        'oversample_power': 0., #0.4,
                        'proposal_scale': 1.9,
                        #'covmat': covmat,
                        #'covmat_params': covmat_params,
                        'Rminus1_stop': 0.03, #0.03, #0.05, #1.,
                        'Rminus1_cl_stop': 0.2, #0.4, #1., #2.,
                        'seed': 42,
                        'max_tries': 1000}}

    info = {'theory': theory, 'likelihood': likelihoods, 'params': params, 'sampler': sampler, 'output': save_fn}  #, 'stop_at_error': True}

    # Cobaya tries loading with dill first, here: https://github.com/CobayaSampler/cobaya/blob/f21ed21059a6c37aa141fb465cc7c7a99b053063/cobaya/output.py#L478
    # This works well, but yaml cannot cope with the dynamic likelihood in https://github.com/CobayaSampler/cobaya/blob/f21ed21059a6c37aa141fb465cc7c7a99b053063/cobaya/output.py#L488, therefore crashes.
    # Let's ignore all previous info at once
    def reload_updated_info(self, cache=False):
        """
        Reloads and returns the version of the input file updated with defaults.

        If none is found, returns ``None`` without raising an error.

        If ``cache=True``, the loaded input will be cached for future calls.
        """
        self._old_updated_info = None
        return None

    def set_checkpoint_info(self, checkpoint_info):
        for k, v in checkpoint_info['sampler'][self.get_name()].items():
            setattr(self, k, v)
        # check if convergence parameters changed, and if so converged=False
        old_info = self.output.get_updated_info(use_cache=True)
        if old_info is None:
            self.converged = False
        else:
            if self.converge_info_changed(old_info["sampler"][self.get_name()], self._updated_info):
                self.converged = False

    from cobaya import run
    from cobaya.log import LoggedError
    from cobaya.output import OutputReadOnly as Output
    from cobaya.sampler import Sampler
    reload_updated_info_bak = Output.reload_updated_info
    Output.reload_updated_info = reload_updated_info
    Sampler.set_checkpoint_info = set_checkpoint_info
    mpicomm.Barrier()
    try:
        for fn in glob.glob(save_fn + '*.lock*'): os.remove(fn)
    except FileNotFoundError:
        pass

    success = False
    if test:
        info.pop('output', None)
        info['sampler'] = {'evaluate': {}}
        info['stop_at_error'] = True
    try:
        updated_info, mcmc = run(info, force=not bool(resume), resume=bool(resume))
        success = True
    except LoggedError as err:
        pass
    # Did it work? (e.g. did not get stuck)
    success = all(mpicomm.allgather(success))

    if not success:
        raise RuntimeError('Sampling failed!')

    Output.reload_updated_info = reload_updated_info_bak
    if test:
        return

    if mpicomm.rank == 0:
        from desilike.samples import Chain, plotting
        from cobaya.output import load_samples
        chains = [Chain.from_getdist(chain.to_getdist()) for chain in load_samples(save_fn, combined=False)]
        for ichain, chain in enumerate(chains):
            chain.save('{}_{:d}.npy'.format(save_fn, ichain + 1))  # cobaya starts at 1
        chain = chains[0].concatenate([chain.remove_burnin(0.5) for chain in chains])
        try:
            for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
                chain.to_stats(tablefmt=tablefmt, fn=save_fn + '_stats.' + ext)
            plotting.plot_triangle(chain, fn=save_fn + '_triangle.png')
        except:
            import traceback
            traceback.print_exc()

    return output


def importance_planck(output, get_desilike_likelihoods=get_desilike_likelihoods, emulator_fn=None, **kwargs):
    import os

    from desi_y1_files.file_manager import get_cosmo_setup
    from desilike.theories import Cosmoprimo
    from desilike.samplers import ImportanceSampler

    datasets = output.options['datasets']
    model = output.options['model']
    emulators = output.options.get('emulators', {})

    if emulator_fn is True:
        emulator_fn = {dataset: True for dataset in emulators}

    cosmo = Cosmoprimo(fiducial='DESI')
    cosmo.init.params = get_cosmo_setup(model=model, dataset=datasets)
    dataset_planck = 'planck2018'
    datasets_noplanck = []
    for dataset in datasets:
        if 'planck' in dataset: dataset_planck = dataset
        else: datasets_noplanck.append(dataset)

    from desilike.likelihoods.cmb import read_planck2018_chain

    basename = model + '_plikHM_TTTEEE_lowl_lowE'
    if 'lensing' in 'panck2018': basename += '_lensing'
    if model != 'base': basename += '_BAO'
    chain = read_planck2018_chain(basename=basename, weights='cmb_only', params=cosmo.varied_params)
    likelihood = sum(get_desilike_likelihoods(datasets_noplanck, emulator_fn=emulator_fn, cosmo=cosmo, **kwargs))
    save_fn = '{}_{:d}.npy'.format(output.filepath, 1)
    sampler = ImportanceSampler(likelihood, chain, save_fn=save_fn)
    chains = sampler.run()
    base_save_fn = output.filepath
    if sampler.mpicomm.rank == 0:
        from desilike.samples import Chain, plotting
        for ichain, chain in enumerate(chains):
            chain.write_getdist(base_save_fn, ichain=ichain)
        chain = chains[0].concatenate(chains)
        try:
            for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
                chain.to_stats(tablefmt=tablefmt, fn=base_save_fn + '_stats.' + ext)
            plotting.plot_triangle(chain, fn=base_save_fn + '_triangle.png')
        except:
            pass


def test_emulator_accuracy(femulator, output=None, resume=True, params=None, plot=None, get_desilike_likelihoods=get_desilike_likelihoods, **kwargs):
    import logging
    import numpy as np
    from matplotlib import pyplot as plt
    from desi_y1_files.file_manager import get_cosmo_setup
    from desilike.theories import Cosmoprimo

    model, dataset = femulator.options['model'], femulator.options['dataset']
    logger = logging.getLogger('TestEmulator')
    cosmo = Cosmoprimo(fiducial='DESI')
    cosmo.init.params = get_cosmo_setup(model=model, dataset=dataset)

    likelihood = get_desilike_likelihoods(dataset=dataset, save_emulator=False, cosmo=cosmo, **kwargs)
    emulated_likelihood = get_desilike_likelihoods(dataset=dataset, save_emulator=False, emulator_fn=femulator, cosmo=cosmo, **kwargs)
    for like in [likelihood, emulated_likelihood]:
        like.all_params
        for ll in like.likelihoods: ll.init.params['{}.flattheory'.format(ll.name)] = {'derived': True}
    mpicomm = likelihood.mpicomm

    if output is None:
        scale = 2.
        from desilike.samples import Samples
        params = likelihood.varied_params.select(basename=params or cosmo.init.params.basenames())
        npoints = 2
        if mpicomm.rank == 0:
            samples = Samples([np.full(npoints * len(params), param.value) for param in params], params=params)
            for iparam, param in enumerate(params):
                limits = [max(param.value - scale * param.proposal, param.prior.limits[0]), min(param.value + scale * param.proposal, param.prior.limits[1])]
                samples[param][npoints * iparam:npoints * (iparam + 1)] = limits
        likelihood.runtime_info.pipeline.mpicalculate(**{name: samples[name] if mpicomm.rank == 0 else None for name in params.names()})
        if mpicomm.rank == 0:
            samples.update(likelihood.runtime_info.pipeline.derived)
    else:
        save_fn = output.filepath
        if resume and os.path.isfile(save_fn):
            samples = save_fn
        from desilike.samplers import QMCSampler
        sampler = QMCSampler(likelihood, samples, save_fn=save_fn)
        samples = sampler.run(niterations=100)

    emulated_likelihood.runtime_info.pipeline.mpicalculate(**{name: samples[name] if mpicomm.rank == 0 else None for name in params.names()})
    emulated_samples = emulated_likelihood.runtime_info.pipeline.derived
    if mpicomm.rank == 0:
        #loglikelihood = samples[likelihood._param_loglikelihood][()]
        #emulated_loglikelihood = emulated_samples[emulated_likelihood._param_loglikelihood][()]
        #diff = 2. * np.abs(loglikelihood - emulated_loglikelihood)
        diff = 0
        for like in likelihood.likelihoods:
            tmp = emulated_samples['{}.flattheory'.format(like.name)] - samples['{}.flattheory'.format(like.name)]
            if hasattr(like, 'precision'):
                tmp = np.sum(np.sum(tmp[..., None] * like.precision, axis=1) * tmp, axis=-1)
            else:
                tmp = np.sum(tmp**2, axis=-1)
            diff += tmp
        argsort = np.argsort(diff)[::-1]
        logger.info('Max chi2 difference is {:.4f}'.format(diff.max()))
        npoints = min(10, len(diff))
        points, rpoints = [], []
        params = samples.params(varied=True, derived=False)
        for index in argsort[:npoints]:
            points.append(str({'diff': diff[index], **{param.name: samples[param][index] for param in params}}))
            rpoints.append(str({'diff': diff[index], **{param.name: (samples[param][index] - param.value) / param.proposal for param in params}}))
        logger.info('{:d} points with larger chi2 difference:\n{}'.format(npoints, '\n'.join(points)))
        logger.info('Corresponding distance w.r.t. central value in proposal units:\n{}'.format('\n'.join(rpoints)))
        if plot:
            if 'full-shape' in dataset:
                for like, emulated_like in zip(likelihood.likelihoods, emulated_likelihood.likelihoods):
                    observable, emulated_observable = like.observables[0], emulated_like.observables[0]
                    ax = plt.gca()
                    for index in argsort[:npoints]:
                        observable(**{param.name: samples[param][index] for param in params})
                        emulated_observable(**{param.name: samples[param][index] for param in params})
                        for ill, ell in enumerate(observable.ells):
                            ax.plot(observable.k[ill], observable.k[ill] * observable.theory[ill], color='C{:d}'.format(ill), linestyle='-')
                            ax.plot(emulated_observable.k[ill], emulated_observable.k[ill] * emulated_observable.theory[ill], color='C{:d}'.format(ill), linestyle='--')
                    plt.savefig('{}_{}.png'.format(plot, like.name))
                    plt.close(plt.gcf())
            if 'planck' in dataset:
                for like, emulated_like in zip(likelihood.likelihoods, emulated_likelihood.likelihoods):
                    like()
                    fig = {key: plt.subplots(nrows=2, sharex=True, sharey=False)[0] for key in like.theory.cls}
                    for index in argsort[:npoints]:
                        like(**{param.name: samples[param][index] for param in params})
                        emulated_like(**{param.name: samples[param][index] for param in params})
                        for key in like.theory.cls:
                            lax = fig[key].axes
                            ells = np.arange(len(like.theory.cls[key]))
                            lax[0].plot(ells, ells * (ells + 1) * like.theory.cls[key], color='k', linestyle='-')
                            lax[0].plot(ells, ells * (ells + 1) * emulated_like.theory.cls[key], color='k', linestyle='--')
                            if key[1] == key[0]:  # auto
                                lax[1].plot(ells, (emulated_like.theory.cls[key] - like.theory.cls[key]) / like.theory.cls[key], color='k', linestyle='-')
                            else:  # cross
                                lax[1].plot(ells, emulated_like.theory.cls[key] - like.theory.cls[key], color='k', linestyle='-')
                    for key, fig in fig.items():
                        fig.subplots_adjust(hspace=0.1)
                        fig.savefig('{}_{}_{}.png'.format(plot, like.name, key))
                        plt.close(fig)


def test_emulate_comoprimo():
    from desi_y1_files.file_manager import get_cosmo_setup
    from desilike.theories import Cosmoprimo

    cosmo = Cosmoprimo(fiducial='DESI')
    cosmo.init.params = get_cosmo_setup(model='base_omegak_w_wa_mnu', dataset='planck')
    cosmo.init.params['sigma8_m'] = {'derived': True}
    from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine
    emulator = Emulator(cosmo, engine=TaylorEmulatorEngine(order=4, method='finite'))
    emulator.set_samples()
    emulator.fit()
    emulator.to_calculator()
    if emulator.mpicomm.rank == 0:
        emulator.log_info('Done.')

if __name__ == '__main__':

    import logging
    from desipipe import setup_logging
    from desipipe.file_manager import BaseFile

    setup_logging()

    logger = logging.getLogger('CosmoTools')

    #export_fisher()
    todo = []
    #todo = ['cosmoprimo']
    todo = ['forecasts']
    #todo = ['bao']
    #todo = ['full-shape']
    #todo = ['planck']
    #todo = ['bao-importance']
    #todo = ['full-shape', 'accuracy-full-shape']
    #todo = ['planck', 'accuracy-planck'][1:]
    #todo = ['local']
    #code = 'desilike'
    code = 'cobaya'
    
    if 'local' in todo:
        save_local()

    if 'bao' in todo:
        output = BaseFile('_tests/{code}/chain_bao', options=dict(code=code, model='base_omegak', datasets=['desi-bao-gaussian-bgs-lrg']))
        if code == 'cobaya': sample_cobaya(output, resume=True)
        else: sample_desilike(output, resume=True)

    if 'bao-importance' in todo:
        output = BaseFile('_tests/{code}/chain_importance_bao', options=dict(code=code, model='base_w_wa', datasets=['planck2018', 'desi-bao-gaussian-bgs-lrg']))
        importance_planck(output)

    if 'full-shape' in todo:
        dataset = 'desi-full-shape-power-bgs'
        emulator_fn = BaseFile('_tests/emulator_full_shape.npy', options=dict(model='base_omegak_w_wa_mnu', dataset=dataset))
        emulate_desilike(emulator_fn)
        #emulator_fn = {dataset: emulator_fn}
        #output = BaseFile('_tests/{code}/chain_full_shape', options=dict(code=code, model='base_omegak', datasets=[dataset, 'bbn-omega_b']))
        #if code == 'cobaya': sample_cobaya(output, emulator_fn=emulator_fn, resume=True)
        #else: sample_desilike(output, emulator_fn=emulator_fn, resume=True)

    if 'accuracy-full-shape' in todo:
        dataset = 'desi-full-shape-power-bgs'
        emulator_fn = BaseFile('_tests/emulator_full_shape.npy', options=dict(model='base_omegak_w_wa_mnu', dataset=dataset))
        measure_speed(niterations=5, model='base_omegak_w_wa_mnu', datasets=[dataset], emulator_fn={dataset: emulator_fn})
        #test_emulator_accuracy(emulator_fn, params=['w0_fld'], plot='accuracy')

    if 'planck' in todo:
        dataset = 'planck2018'
        emulator_fn = BaseFile('_tests/emulator_planck.npy', options=dict(model='base_omegak_w_wa_mnu', dataset=dataset))
        emulate_desilike(emulator_fn)

    if 'accuracy-planck' in todo:
        dataset = 'planck2018'
        emulator_fn = BaseFile('_tests/emulator_planck.npy', options=dict(model='base_omegak_w_wa_mnu', dataset=dataset))
        test_emulator_accuracy(emulator_fn, plot='accuracy')

    if 'cosmoprimo' in todo:
        test_emulate_comoprimo()

    if 'forecasts' in todo:
        from desilike import LikelihoodFisher
        from desi_y1_files.file_manager import get_cosmo_file_manager
        cfm = get_cosmo_file_manager()
        for fisher_forecast in get_bao_rsd_forecasts(region='GCcomb', apmode='qisoqap')[0]:
            tracer, zrange = fisher_forecast.attrs['tracer'], fisher_forecast.attrs['zrange']
            fisher_y1 = cfm.get(id='gaussian_bao_y1', tracer=tracer, zrange=zrange)
            fisher_y1 = LikelihoodFisher.load(fisher_y1)
            logger.info('For tracer = {} in {}, forecast:\n{}'.format(tracer, zrange, fisher_forecast.to_stats(tablefmt='pretty')))
            logger.info('For tracer = {} in {}, data:\n{}'.format(tracer, zrange, fisher_y1.to_stats(tablefmt='pretty')))
            logger.info('For tracer = {} in {}; data / forecast errors: {}'.format(tracer, zrange, fisher_y1.std() / fisher_forecast.std(fisher_y1.params())))
