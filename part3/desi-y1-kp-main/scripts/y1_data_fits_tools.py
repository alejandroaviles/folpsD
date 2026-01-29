from y1_data_2pt_tools import get_footprint


def get_desilike_covariance_matrix(observable, footprint):
    if footprint is None:
        raise ValueError('provide footprint to estimate covariance matrix')

    observable = observable.deepcopy()
    observable()
    z = observable.wmatrix.theory.z

    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles
    from desilike.observables.galaxy_clustering import ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood

    template = FixedPowerSpectrumTemplate(z=z)
    theory = (LPTVelocileptorsTracerPowerSpectrumMultipoles if 'power' in observable.__class__.__name__.lower() else LPTVelocileptorsTracerCorrelationFunctionMultipoles)(template=template)
    observable.init.update(theory=theory)

    covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)
    observable.init.update(covariance=covariance())
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    for param in likelihood.all_params.select(name=['alpha*', 'sn*', 'c*']):
        if param.varied: param.update(derived='.auto')

    from desilike.profilers import MinuitProfiler

    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize(niterations=5)
    return covariance(**profiles.bestfit.choice(input=True))


def get_observable_likelihood(data=None, covariance=None, footprint=None, wmatrix=None, theory_name='velocileptors', ells=None, template_name='shapefit', solve='.auto', save_emulator=False, emulator_fn=None, get_desilike_covariance_matrix=get_desilike_covariance_matrix, cosmo=None, z=None, **kwargs):

    """Return the power spectrum likelihood, optionally computing the emulator (if ``save_emulator``)."""
    import os
    import numpy as np
    from desi_y1_files.file_manager import get_fit_setup
    from desipipe.file_manager import FileEntryCollection

    def get_template(template_name='standard'):

        """A simple wrapper that returns the template of interest."""

        from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, StandardPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, WiggleSplitPowerSpectrumTemplate, BandVelocityPowerSpectrumTemplate, DirectPowerSpectrumTemplate, BAOPowerSpectrumTemplate
        if 'standard' in template_name:
            template = StandardPowerSpectrumTemplate()
        elif 'shapefit' in template_name:
            template = ShapeFitPowerSpectrumTemplate()
            if 'qisoqap' in template_name:
                template.init.update(apmode='qisoqap')
                template.init.params['qiso'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
                template.init.params['qap'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
                template.init.params['df'].update(delta=0.05)
                # limit the range of dm (otherwise it's easy to have bimodial posterior)
                template.init.params['dm'].update(prior={'limits': [-0.8, 0.8]})
            if 'dn' in template_name:
                template.init.params['dn'].update(fixed=False, prior={'dist': 'norm', 'loc': 0., 'scale': 0.2})
        
        elif 'direct' in template_name or 'base' in template_name:
            template = DirectPowerSpectrumTemplate()
            template.init.params['omega_b'].update(prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
            template.init.params['omega_cdm'].update(delta=0.01)
            template.init.params['logA'].update(delta=0.07)
            template.init.params['h'].update(prior={'dist': 'uniform', 'limits': [0.1, 1.0]})
            template.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 10 * 0.0042}, delta=0.01)
            #template.init.params['sigma8_m'] = {'derived': True, 'latex': r'\sigma_8'}
            #template.init.params['Omega_m'] = {'derived': True, 'latex': r'\Omega_m'}
            #template.init.params['rs_drag'] = {'derived': True, 'latex': r'r_s'}
            if 'ns-fixed' in template_name:
                template.init.params['n_s'].update(fixed=True)
            if 'ns-planck3' in template_name:
                template.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 3 * 0.0042})
            if 'ns-004' in template_name:
                template.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.04})
            if '_w' in template_name:
                template.init.params['w0_fld'].update(fixed=False)
            if '_w_wa' in template_name:
                template.init.params['wa_fld'].update(fixed=False)
        elif 'bao' in template_name:
            template = BAOPowerSpectrumTemplate(z=z) #, with_now='wallish2018')
            #template.init.params['df'].update(fixed=False, prior={'dist': 'norm', 'loc': 1., 'scale': 1.})
            #template.init.params['df'].update(fixed=False, prior={'limits': [0.7, 1.3]})
            if 'cosmo' in template_name:
                template.init.update(apmode='bao')
            if 'now' in template_name:
                template.init.update(only_now=True)
        elif template_name == 'fixed':
            template = FixedPowerSpectrumTemplate()
        if 'qisoqap' in template_name:
            template.init.update(apmode='qisoqap')
            #for param in template.init.params.select(basename=['qiso', 'qap']):
            #    param.update(prior={'limits': [0.6, 1.4]})
        elif 'qiso' in template_name:
            template.init.update(apmode='qiso')
        return template

    def get_tracer_label(tracer):
        return tracer.split('_')[0].replace('+', 'plus')
    
    # change fixed parameters here
    def get_theory(theory_name='velocileptors', observable_name='power', freedom=None, recon=None, prior_basis='physical', tracer=None, ells=(0, 2, 4)):

        tracer = get_tracer_label(tracer)

        """A simple wrapper that returns the theory of interest."""

        from desilike.theories.galaxy_clustering import (LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerCorrelationFunctionMultipoles,
                                                         PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles, FOLPSTracerPowerSpectrumMultipoles, FOLPSTracerCorrelationFunctionMultipoles, FOLPSAXTracerPowerSpectrumMultipoles, FOLPSAXTracerCorrelationFunctionMultipoles, DampedBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles)
        from desilike.theories.galaxy_clustering.full_shape import FOLPSv2TracerPowerSpectrumMultipoles

        if 'bird' in theory_name:
            theory = (PyBirdTracerPowerSpectrumMultipoles if observable_name == 'power' else PyBirdTracerCorrelationFunctionMultipoles)(freedom=freedom)
        elif 'folps' in theory_name:
            if 'folpsax' in theory_name:
                theory = (FOLPSAXTracerPowerSpectrumMultipoles if observable_name == 'power' else FOLPSAXTracerCorrelationFunctionMultipoles)()
            elif 'folpsv2' in theory_name:
                if observable_name != 'power':
                    raise NotImplementedError('FOLPSv2TracerPowerSpectrumMultipoles only implemented for observable_name="power"')
                theory = FOLPSv2TracerPowerSpectrumMultipoles()
            else:
                theory = (FOLPSTracerPowerSpectrumMultipoles if observable_name == 'power' else FOLPSTracerCorrelationFunctionMultipoles)()
            theory.init.update(freedom=freedom, prior_basis=prior_basis, tracer=tracer)
            for param in theory.init.params.select(basename=['bs', 'b3']):
                param.update(prior=dict(limits=[-50., 50.]))
            # kwargs.update(mu=3)  # using 3 mu points in [0, 1] to reproduce FOLPS, by default it is 6
        elif 'velo' in theory_name:
            if 'rept' in theory_name:
                theory = (REPTVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else REPTVelocileptorsTracerCorrelationFunctionMultipoles)()
            else:
                theory = (LPTVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else LPTVelocileptorsTracerCorrelationFunctionMultipoles)()
            theory.init.update(freedom=freedom, prior_basis=prior_basis, tracer=tracer)
        elif 'bao' in theory_name:
            theory = (DampedBAOWigglesTracerPowerSpectrumMultipoles if observable_name == 'power' else DampedBAOWigglesTracerCorrelationFunctionMultipoles)(**(recon or {}))

        # nuisance parameters
        if 4 not in ells:
            order = 4
            for param in theory.init.params.select(basename=['al{:d}*_*'.format(order), 'bl{:d}*_*'.format(order), 'alpha{:d}*'.format(order), 'sn{:d}*'.format(order)]): param.update(fixed=True)

        return theory
    
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, CutskyFootprint, SystematicTemplatePowerSpectrumMultipoles, SystematicTemplateCorrelationFunctionMultipoles
    from desilike.likelihoods import ObservablesGaussianLikelihood
    if isinstance(data, (tuple, list, FileEntryCollection)):
        dd = data[0]
        data = [str(dd) for dd in data]
    else:
        dd = data
        data = str(data)
    observable_name = str(dd.filetype)
    options = {**dd.options, **kwargs}
    tracer = options['tracer']
    recon = {'mode': options.get('mode', ''), 'smoothing_radius': options.get('smoothing_radius', 15.)}
    freedom = options.get('freedom', None)
    precscale = options.get('precscale', 1.)
    systematic_templates = options.get('syst', None)
    dd = dd.load()

    b0, lim, sigmas = get_fit_setup(tracer, ells=ells, observable_name=observable_name, theory_name=theory_name, return_list=['b0', 'lim', 'sigmas'])

    lim = kwargs.pop('lim', lim)
    #lim = {ell: [0., 0.4, 0.005] for ell in [0, 2, 4]}  # HACK
    
    sigmas = kwargs.pop('sigmas', sigmas)
    dbeta = kwargs.pop('dbeta', None)
    if z is None:
        if footprint is not None:
            footprint = CutskyFootprint.load(footprint)
            z = footprint.zavg
        else:
            if observable_name == 'power':
                attrs = dd.attrs
            if observable_name == 'correlation':
                attrs = dd.D1D2.attrs
            if 'zeff' in attrs: z = attrs['zeff']

    from cosmoprimo.fiducial import DESI
    fiducial = DESI()
    b1 = b0 / fiducial.growth_factor(z)

    # Load theory
    ells = tuple(lim)
    theory = get_theory(theory_name=theory_name, observable_name=observable_name, recon=recon, freedom=freedom, ells=ells, tracer=tracer)
    theory()
    if 'bao' in theory_name:
        if save_emulator:
            raise ValueError('No need to build an emulator for the BAO model!')
        emulator_fn = None
        broadband = kwargs.pop('broadband', 'pcs')
        if broadband == 'fixed':
            for param in theory.init.params.select(basename=['al*_*', 'bl*_*']):
                param.update(fixed=True)
        else:
            theory.init.update(broadband=broadband)
        if 4 not in ells:
            for param in theory.init.params.select(basename=['*l4_*']):
                param.update(fixed=True)
        if 2 not in ells:
            for param in theory.init.params.select(basename='*l2_*'):
                param.update(fixed=True)
            theory.init.params['dbeta'].update(fixed=True)
        for param in theory.init.params.select(basename=list(sigmas)):
            value = sigmas[param.basename]
            if value is None:
                kw = {'prior': {'limits': [0., 20.]}, 'fixed': False}
            elif isinstance(value, (tuple, list)):
                loc, scale = value
                kw = {'value': loc, 'prior': {'dist': 'norm', 'loc': loc, 'scale': scale, 'limits': [0., 20.]}, 'fixed': False}
            else:
                kw = {'value': value, 'prior': None, 'fixed': True}
            param.update(**kw)
        if dbeta is not None:
            theory.init.params['dbeta'].update(prior=dict(limits=tuple(dbeta)))

    if 'b1p' in theory.init.params:  # physical
        b1p = b1 * fiducial.sigma8_z(z)
        theory.init.params['b1p'].update(value=b1p, ref=dict(dist='norm', loc=b1p, scale=0.1))
    else:
        theory.init.params['b1'].update(value=b1, ref=dict(dist='norm', loc=b1, scale=0.1))
        
    template = get_template(template_name=template_name)
    template.init.update(z=z)
    if cosmo is not None:
        template.init.update(cosmo=cosmo)
    if save_emulator or emulator_fn is None:  # No emulator available (yet)
        theory.init.update(template=template)
    else:  # Load emulator
        from desilike.emulators import EmulatedCalculator
        #calculator = EmulatedCalculator.load('/global/cfs/cdirs/desi/users/ruiyang/Y1/iron/v1.5/emulators/velocileptors/EPT/emulator_SF_BGS.npy')
        calculator = EmulatedCalculator.load(emulator_fn)
        theory.init.update(pt=calculator)
        for param in template.init.params:
            if param in calculator.init.params:
                calculator.init.params.set(param)

    if covariance is not None and hasattr(covariance, 'load'):
        import desi_y1_files  # to define the covariance matrix files
        covariance = covariance.load()
    #if isinstance(covariance, (tuple, list, FileEntryCollection)):
    #    covariance = [str(mm) for mm in covariance]  # str otherwise error with glob and BaseFile

    templates = {}

    if observable_name == 'power':
        wmatrix = wmatrix.load()
        rotation = wmatrix.attrs.get('rotation', {})

        priors = {}
        if False: #systematic_templates:  #systematic_templates is not None:
            dirname = '/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/templates_2pt/rotated/'
            cut = '_thetacut0.05' if options.get('cut', None) else ''
            region = options.get('region', 'GCcomb')
            zrange = options.get('zrange', None)
            template_fns = {syst: os.path.join(dirname, 'template_rotated_{}_power_{}_{}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin{cut}.npy'.format(syst, tracer, region, zrange=zrange, cut=cut)) for syst in ['ric', 'aic', 'photo']}
            from desi_y1_files.systematic_template import PolynomialTemplate
            templates['syst_0'] = PolynomialTemplate.load(template_fns['photo'])
            priors['syst_0'] = {'dist': 'norm', 'loc': 0., 'scale': 0.2}
            templates['syst_1'] = PolynomialTemplate.load(template_fns['aic'])
            priors['syst_1'] = 1.
            templates['syst_2'] = PolynomialTemplate.load(template_fns['ric'])
            priors['syst_2'] = 1.
            """
            region = options.get('region', 'GCcomb')
            zrange = options.get('zrange', None)
            basename = '/global/cfs/cdirs/desi/users/ruiyang/systemplate/baseline/power_systcorr'
            ttracer = None
            if 'ELG' in tracer: ttracer = 'ELG'
            if 'QSO' in tracer: ttracer = 'QSO'
            if ttracer is not None:
                from desi_y1_files.systematic_template import PolynomialTemplate
                template_fn = '{}_poly_{}_{zrange[0]:.1f}_{zrange[1]:.1f}_{region}.npy'.format(basename, ttracer, zrange=zrange, region=region)
                templates['syst_0'] = PolynomialTemplate.load(template_fn)
                #templates['syst_0'] = np.zeros(72, dtype='f8')
                priors['syst_0'] = {'dist': 'norm', 'loc': 0., 'scale': 0.2}
                template_fn = '{}_mock_{}_{zrange[0]:.1f}_{zrange[1]:.1f}_{region}.npy'.format(basename, ttracer, zrange=zrange, region=region)
                #template_fn = '/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v1.4/unblinded/desipipe/templates_2pt/template_aic_power_QSO_GCcomb_z0.8-2.1_default_FKP_lin.npy'
                templates['syst_1'] = PolynomialTemplate.load(template_fn)
                ##templates['syst_1'] = np.zeros(72, dtype='f8')
                priors['syst_1'] = 1.
            """

        if False: #rotation:
            """
            masks = []
            for ill, ell in enumerate(wmatrix.projsout):
                ell = ell.ell
                if ell in lim: mask = (wmatrix.xout[ill] >= lim[ell][0]) & (wmatrix.xout[ill] <= lim[ell][1])
                else: mask = np.zeros_like(wmatrix.xout[ill], dtype='?')
                masks.append(mask)
            mask = np.concatenate(masks)
            """
            if covariance is not None:
                covariance = covariance.view(observables=covariance.observables('power*'), return_type=None)
                for ell, xlim in lim.items(): covariance = covariance.select(xlim=xlim, projs=ell)
                covariance = covariance.view(projs=list(lim))
                wmatrix = wmatrix.select_proj(projsout=[(ell, None) for ell in lim])
                mask = wmatrix.index_x(axis='out', xlim=xlim, concatenate=True)

            for ill, ell in enumerate(wmatrix.projsout):
                ell = ell.ell
                if ell in lim: 
                    templates['mo{:d}'.format(ell)] = rotation['mo'][ill][mask]
                    p = abs(rotation['marg_prior_mo'][ill])
                    priors['mo{:d}'.format(ell)] = {'dist': 'norm', 'loc': 0., 'scale': p}
            #mmatrix = rotation['mmatrix'][np.ix_(mask, mask)]
            #covariance = mmatrix.dot(covariance).dot(mmatrix.T)

        if templates:
            systematic_templates = SystematicTemplatePowerSpectrumMultipoles(templates=templates)
            for param in systematic_templates.init.params: param.update(derived=solve)
            for param, prior in priors.items():
                param = systematic_templates.init.params[param]
                if isinstance(prior, dict): param.update(prior=prior)
                else: param.update(value=prior, fixed=True, derived=False)
        else:
            systematic_templates = None
            #covariance.nobs = None
        observable = TracerPowerSpectrumMultipolesObservable(klim=lim, data=data, covariance=covariance, wmatrix=wmatrix, kin=np.arange(0.001, 0.35, 0.001), theory=theory, systematic_templates=systematic_templates)
        #observable = TracerPowerSpectrumMultipolesObservable(klim=lim, data=data, covariance=covariance, wmatrix=wmatrix, kinlim=(0.001, 0.35), theory=theory, systematic_templates=systematic_templates)
        #observable()
        #print(observable.flatdata.sum(), observable.shotnoise, observable.k, observable.wmatrix.matrix_full.shape, observable.wmatrix.matrix_full.sum(), observable.wmatrix.kin, observable.wmatrix.ellsin)
        #exit()
        #observable = TracerPowerSpectrumMultipolesObservable(klim=lim, data=data, covariance=covariance, wmatrix=wmatrix.load(), kinlim=(0., 0.35), theory=theory, systematic_templates=systematic_templates)

    if observable_name == 'correlation':
        #if covariance is not None:
        #    covariance = covariance.view(observables=covariance.observables('corr*'), return_type=None)
        #    for ell, xlim in lim.items(): covariance = covariance.select(xlim=xlim, projs=ell)
            #covariance = covariance.view(projs=list(lim))
        #fiber_collisions = None
        # Fiber collisions already taken into account internally in the mu-bin mask
        #if rpcut is not None:
        #    from desilike.observables.galaxy_clustering import TopHatFiberCollisionsCorrelationFunctionMultipoles
        #    fiber_collisions = TopHatFiberCollisionsCorrelationFunctionMultipoles(Dfc=rpcut, with_uncorrelated=False, mu_range_cut=True)
        if templates:
            systematic_templates = SystematicTemplateCorrelationFunctionMultipoles(templates=templates)
            for param in systematic_templates.init.params: param.update(derived=solve)
        else:
            systematic_templates = None
        wmatrix = {'resolution': 1}  # on top of RR 1 Mpc/h binning
        #from pycorr import TwoPointCorrelationFunction
        #data = TwoPointCorrelationFunction.load(data)[::4]
        #wmatrix = {'resolution': 4}
        observable = TracerCorrelationFunctionMultipolesObservable(slim=lim, data=data, covariance=covariance, wmatrix=wmatrix, theory=theory, ignore_nan=True, systematic_templates=systematic_templates)

    if covariance is None:
        if observable.mpicomm.rank == 0:
            observable.log_info('Covariance matrix not found, creating one on-the-fly')
        if footprint is None:
            raise ValueError('provide footprint to estimate covariance matrix')
        covariance = get_desilike_covariance_matrix(observable, footprint)
        observable.init.update(covariance=covariance)

    likelihood = ObservablesGaussianLikelihood(observables=[observable], scale_covariance=1. / precscale)  # likelihood is a callable that returns the log-posterior
    likelihood()
    #print(likelihood.all_params['omega_b'].prior)
    #np.save('tmpcov.npy', likelihood.covariance)
    #np.save('tmpprec.npy', np.linalg.inv(likelihood.precision))

    if save_emulator:  # Compute and save emulator
        likelihood()  # to set up k-ranges for the emulator
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=2))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)
    elif 'direct' in template_name and cosmo is not None:  # external cosmo
        template.init.update(cosmo=cosmo)
    # likelihood.all_params gives access to the parameters of the likelihood pipeline
    if solve:
        for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*'] + ['al*_*', 'bl*_*']):
            param.update(derived=solve)
        #for param in likelihood.all_params.select(basename=['sn*']):
        #    param.update(derived='.prec')
    all_params = likelihood.all_params.names(solved=True)
    if likelihood.mpicomm.rank == 0:
        likelihood.log_info('Use analytic marginalization for {}.'.format(all_params))

    return likelihood


def get_joint_observable_likelihood(data=None, covariance=None, footprint=None, wmatrix=None, theory_name='velocileptors', ells=None, template_name='shapefit', observable_name=['power'], solve='.auto', save_emulator=False, emulator_fn=None, get_desilike_covariance_matrix=get_desilike_covariance_matrix, cosmo=None, z=None, **kwargs):

    """Return the power spectrum likelihood, optionally computing the emulator (if ``save_emulator``)."""
    import os
    import numpy as np
    from desi_y1_files.file_manager import get_fit_setup
    from desipipe.file_manager import FileEntryCollection

    def get_template(template_name='standard'):

        """A simple wrapper that returns the template of interest."""

        from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, StandardPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, WiggleSplitPowerSpectrumTemplate, BandVelocityPowerSpectrumTemplate, DirectPowerSpectrumTemplate, BAOPowerSpectrumTemplate
        if 'standard' in template_name:
            template = StandardPowerSpectrumTemplate()
        elif 'shapefit' in template_name:
            template = ShapeFitPowerSpectrumTemplate()
            if 'qisoqap' in template_name:
                template.init.update(apmode='qisoqap')
                template.init.params['qiso'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
                template.init.params['qap'].update(delta=0.02, prior={'limits': [0.8, 1.2]})
                template.init.params['df'].update(delta=0.05)
                # limit the range of dm (otherwise it's easy to have bimodial posterior)
                template.init.params['dm'].update(prior={'limits': [-0.8, 0.8]})
            if 'dn' in template_name:
                template.init.params['dn'].update(fixed=False, prior={'dist': 'norm', 'loc': 0., 'scale': 0.2})

        elif 'direct' in template_name or 'base' in template_name:
            template = DirectPowerSpectrumTemplate()
            template.init.params['omega_b'].update(prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
            template.init.params['omega_cdm'].update(delta=0.01)
            template.init.params['logA'].update(delta=0.07)
            template.init.params['h'].update(prior={'dist': 'uniform', 'limits': [0.1, 1.0]})
            template.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 10 * 0.0042}, delta=0.01)
            #template.init.params['sigma8_m'] = {'derived': True, 'latex': r'\sigma_8'}
            #template.init.params['Omega_m'] = {'derived': True, 'latex': r'\Omega_m'}
            #template.init.params['rs_drag'] = {'derived': True, 'latex': r'r_s'}
            if 'ns-fixed' in template_name:
                template.init.params['n_s'].update(fixed=True)
            if 'ns-planck3' in template_name:
                template.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 3 * 0.0042})
            if 'ns-004' in template_name:
                template.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.04})
            if '_w' in template_name:
                template.init.params['w0_fld'].update(fixed=False)
            if '_w_wa' in template_name:
                template.init.params['wa_fld'].update(fixed=False)
        elif 'bao' in template_name:
            template = BAOPowerSpectrumTemplate(z=z) #, with_now='wallish2018')
            #template.init.params['df'].update(fixed=False, prior={'dist': 'norm', 'loc': 1., 'scale': 1.})
            #template.init.params['df'].update(fixed=False, prior={'limits': [0.7, 1.3]})
            if 'cosmo' in template_name:
                template.init.update(apmode='bao')
            if 'now' in template_name:
                template.init.update(only_now=True)
        elif template_name == 'fixed':
            template = FixedPowerSpectrumTemplate()
        if 'qisoqap' in template_name:
            template.init.update(apmode='qisoqap')
            #for param in template.init.params.select(basename=['qiso', 'qap']):
            #    param.update(prior={'limits': [0.6, 1.4]})
        elif 'qiso' in template_name:
            template.init.update(apmode='qiso')
        return template

    def get_tracer_label(tracer):
        return tracer.split('_')[0].replace('+', 'plus')

    # change fixed parameters here
    def get_theory(theory_name='velocileptors', observable_name='power', freedom=None, recon=None, prior_basis='physical', tracer=None, ells=(0, 2, 4)):

        tracer = get_tracer_label(tracer)

        """A simple wrapper that returns the theory of interest."""

        from desilike.theories.galaxy_clustering import (LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerCorrelationFunctionMultipoles,
                                                         PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles, FOLPSTracerPowerSpectrumMultipoles, FOLPSTracerCorrelationFunctionMultipoles, FOLPSAXTracerPowerSpectrumMultipoles, FOLPSAXTracerCorrelationFunctionMultipoles, DampedBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles)
        from desilike.theories.galaxy_clustering.full_shape import FOLPSv2TracerPowerSpectrumMultipoles

        if 'bird' in theory_name:
            theory = (PyBirdTracerPowerSpectrumMultipoles if observable_name == 'power' else PyBirdTracerCorrelationFunctionMultipoles)(freedom=freedom)
        elif 'folps' in theory_name:
            if 'folpsax' in theory_name:
                theory = (FOLPSAXTracerPowerSpectrumMultipoles if observable_name == 'power' else FOLPSAXTracerCorrelationFunctionMultipoles)()
            elif 'folpsv2' in theory_name:
                if observable_name != 'power':
                    raise NotImplementedError('FOLPSv2TracerPowerSpectrumMultipoles only implemented for observable_name="power"')
                theory = FOLPSv2TracerPowerSpectrumMultipoles()
            else:
                theory = (FOLPSTracerPowerSpectrumMultipoles if observable_name == 'power' else FOLPSTracerCorrelationFunctionMultipoles)()
            theory.init.update(freedom=freedom, prior_basis=prior_basis, tracer=tracer)
            for param in theory.init.params.select(basename=['bs', 'b3']):
                param.update(prior=dict(limits=[-50., 50.]))
            # kwargs.update(mu=3)  # using 3 mu points in [0, 1] to reproduce FOLPS, by default it is 6
        elif 'velo' in theory_name:
            if 'rept' in theory_name:
                theory = (REPTVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else REPTVelocileptorsTracerCorrelationFunctionMultipoles)()
            else:
                theory = (LPTVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else LPTVelocileptorsTracerCorrelationFunctionMultipoles)()
            theory.init.update(freedom=freedom, prior_basis=prior_basis, tracer=tracer)
        elif 'bao' in theory_name:
            theory = (DampedBAOWigglesTracerPowerSpectrumMultipoles if observable_name == 'power' else DampedBAOWigglesTracerCorrelationFunctionMultipoles)(**(recon or {}))

        # nuisance parameters
        if 4 not in ells:
            order = 4
            for param in theory.init.params.select(basename=['al{:d}*_*'.format(order), 'bl{:d}*_*'.format(order), 'alpha{:d}*'.format(order), 'sn{:d}*'.format(order)]): param.update(fixed=True)

        return theory

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, CutskyFootprint, SystematicTemplatePowerSpectrumMultipoles, SystematicTemplateCorrelationFunctionMultipoles
    from desilike.likelihoods import ObservablesGaussianLikelihood

    list_data, list_observable_name = data, observable_name
    observables = []
    for data, observable_name in zip(list_data, list_observable_name):
        if isinstance(data, (tuple, list, FileEntryCollection)):
            doptions = data[0].options
            data = [str(dd) for dd in data]
        else:
            dd = data
            data = str(data)
        options = {**dd.options, **kwargs}
        tracer = options['tracer']
        recon = {'mode': options.get('mode', ''), 'smoothing_radius': options.get('smoothing_radius', 15.)}
        freedom = options.get('freedom', None)
        precscale = options.get('precscale', 1.)
        systematic_templates = options.get('syst', None)
        dd = dd.load()

        if 'power' in observable_name or 'correlation' in observable_name:
            b0, lim, sigmas = get_fit_setup(tracer, ells=ells, observable_name=observable_name, theory_name=theory_name, return_list=['b0', 'lim', 'sigmas'])
            lim = kwargs.pop('lim', lim)
            #lim = {ell: [0., 0.4, 0.005] for ell in [0, 2, 4]}  # HACK

            sigmas = kwargs.pop('sigmas', sigmas)
            dbeta = kwargs.pop('dbeta', None)
            if z is None:
                if footprint is not None:
                    footprint = CutskyFootprint.load(footprint)
                    z = footprint.zavg
                else:
                    if observable_name == 'power':
                        attrs = dd.attrs
                    if observable_name == 'correlation':
                        attrs = dd.D1D2.attrs
                    if 'zeff' in attrs: z = attrs['zeff']

            from cosmoprimo.fiducial import DESI
            fiducial = DESI()
            b1 = b0 / fiducial.growth_factor(z)

            # Load theory
            ells = tuple(lim)
            theory = get_theory(theory_name=theory_name, observable_name=observable_name, recon=recon, freedom=freedom, ells=ells, tracer=tracer)
            theory()
            if 'bao' in theory_name:
                if save_emulator:
                    raise ValueError('No need to build an emulator for the BAO model!')
                emulator_fn = None
                broadband = kwargs.pop('broadband', 'pcs')
                if broadband == 'fixed':
                    for param in theory.init.params.select(basename=['al*_*', 'bl*_*']):
                        param.update(fixed=True)
                else:
                    theory.init.update(broadband=broadband)
                if 4 not in ells:
                    for param in theory.init.params.select(basename=['*l4_*']):
                        param.update(fixed=True)
                if 2 not in ells:
                    for param in theory.init.params.select(basename='*l2_*'):
                        param.update(fixed=True)
                    theory.init.params['dbeta'].update(fixed=True)
                for param in theory.init.params.select(basename=list(sigmas)):
                    value = sigmas[param.basename]
                    if value is None:
                        kw = {'prior': {'limits': [0., 20.]}, 'fixed': False}
                    elif isinstance(value, (tuple, list)):
                        loc, scale = value
                        kw = {'value': loc, 'prior': {'dist': 'norm', 'loc': loc, 'scale': scale, 'limits': [0., 20.]}, 'fixed': False}
                    else:
                        kw = {'value': value, 'prior': None, 'fixed': True}
                    param.update(**kw)
                if dbeta is not None:
                    theory.init.params['dbeta'].update(prior=dict(limits=tuple(dbeta)))

            if 'b1p' in theory.init.params:  # physical
                b1p = b1 * fiducial.sigma8_z(z)
                theory.init.params['b1p'].update(value=b1p, ref=dict(dist='norm', loc=b1p, scale=0.1))
            else:
                theory.init.params['b1'].update(value=b1, ref=dict(dist='norm', loc=b1, scale=0.1))

            template = get_template(template_name=template_name)
            template.init.update(z=z)
            if cosmo is not None:
                template.init.update(cosmo=cosmo)
            if save_emulator or emulator_fn is None:  # No emulator available (yet)
                theory.init.update(template=template)
            else:  # Load emulator
                from desilike.emulators import EmulatedCalculator
                #calculator = EmulatedCalculator.load('/global/cfs/cdirs/desi/users/ruiyang/Y1/iron/v1.5/emulators/velocileptors/EPT/emulator_SF_BGS.npy')
                calculator = EmulatedCalculator.load(emulator_fn)
                theory.init.update(pt=calculator)
                for param in template.init.params:
                    if param in calculator.init.params:
                        calculator.init.params.set(param)

        templates = {}

        if observable_name == 'power':
            wmatrix = wmatrix.load()
            rotation = wmatrix.attrs.get('rotation', {})
            priors = {}

            if templates:
                systematic_templates = SystematicTemplatePowerSpectrumMultipoles(templates=templates)
                for param in systematic_templates.init.params: param.update(derived=solve)
                for param, prior in priors.items():
                    param = systematic_templates.init.params[param]
                    if isinstance(prior, dict): param.update(prior=prior)
                    else: param.update(value=prior, fixed=True, derived=False)
            else:
                systematic_templates = None
                #covariance.nobs = None
            observable = TracerPowerSpectrumMultipolesObservable(klim=lim, data=data, wmatrix=wmatrix, kin=np.arange(0.001, 0.35, 0.001), theory=theory, systematic_templates=systematic_templates)
            observables.append(observable)

        if observable_name == 'correlation':
            if templates:
                systematic_templates = SystematicTemplateCorrelationFunctionMultipoles(templates=templates)
                for param in systematic_templates.init.params: param.update(derived=solve)
            else:
                systematic_templates = None
            wmatrix = {'resolution': 1}  # on top of RR 1 Mpc/h binning
            #from pycorr import TwoPointCorrelationFunction
            #data = TwoPointCorrelationFunction.load(data)[::4]
            #wmatrix = {'resolution': 4}
            observable = TracerCorrelationFunctionMultipolesObservable(slim=lim, data=data, wmatrix=wmatrix, theory=theory, ignore_nan=True, systematic_templates=systematic_templates)
            observables.append(observable)

        if observable_name == 'bao-recon':
            from jax import numpy as jnp
            from desilike import BaseCalculator
            class BAOObservable(BaseCalculator):

                def initialize(self, data=None):
                    observable = data
                    self.flatdata = observable.view()
                    self.quantities = observable.projs

                def calculate(self, **params):
                    if 'qpar' not in params:
                        params['qpar'] = params['qiso'] * params['qap']**(2. / 3.)
                    if 'qper' not in params:
                        params['qper'] = params['qiso'] * params['qap']**(-1. / 3.)
                    self.flattheory = jnp.array([params[quantity] for quantity in self.quantities])

                def to_array(self):
                    from desilike.observables import ObservableArray
                    return ObservableArray(value=self.flatdata, projs=self.quantities, name='bao-recon')

            observable = BAOObservable(data=dd.observables(observables='bao-recon'))
            observable.init.params = template.all_params.select(basename=['q*']).copy()
            observables.append(observable)

    if covariance is not None and hasattr(covariance, 'load'):
        import desi_y1_files  # to define the covariance matrix files
        covariance = covariance.load()
    #if isinstance(covariance, (tuple, list, FileEntryCollection)):
    #    covariance = [str(mm) for mm in covariance]  # str otherwise error with glob and BaseFile

    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=covariance, scale_covariance=1. / precscale)  # likelihood is a callable that returns the log-posterior
    likelihood()
    print(likelihood.flatdiff.shape)
    #print(likelihood.all_params['omega_b'].prior)
    #np.save('tmpcov.npy', likelihood.covariance)
    #np.save('tmpprec.npy', np.linalg.inv(likelihood.precision))

    if save_emulator:  # Compute and save emulator
        likelihood()  # to set up k-ranges for the emulator
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=2))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)
    elif 'direct' in template_name and cosmo is not None:  # external cosmo
        template.init.update(cosmo=cosmo)
    # likelihood.all_params gives access to the parameters of the likelihood pipeline
    if solve:
        for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*'] + ['al*_*', 'bl*_*']):
            param.update(derived=solve)
        #for param in likelihood.all_params.select(basename=['sn*']):
        #    param.update(derived='.prec')
    all_params = likelihood.all_params.names(solved=True)
    if likelihood.mpicomm.rank == 0:
        likelihood.log_info('Use analytic marginalization for {}.'.format(all_params))

    return likelihood


def profile(output, get_observable_likelihood=get_observable_likelihood, get_joint_observable_likelihood=get_joint_observable_likelihood, **kwargs):
    import os
    import numpy as np
    from desilike.profilers import MinuitProfiler
    options = {**output.options, **kwargs}
    if options.get('observable_name', []):
        likelihood = get_joint_observable_likelihood(save_emulator=False, solve='.best', **options)
    else:
        likelihood = get_observable_likelihood(save_emulator=False, solve='.best', **options)
    #likelihood()
    #likelihood = get_observable_likelihood(save_emulator=False, **{**output.options, **kwargs})
    #likelihood()
    profiler = MinuitProfiler(likelihood, seed=42)
    costly = 'sigmas' not in likelihood.all_params and not kwargs.get('emulator_fn', '') 
    profiles = profiler.maximize(niterations=profiler.mpicomm.size if costly else 10)
    params = likelihood.varied_params.select(basename=['qiso', 'qap', 'qpar', 'qper', 'df', 'dm', 'Omega_m', 'h', 'logA', 'n_s'])
    if params and not costly:
        profiles = profiler.interval(params=params)
    #if output.options.get('template', '') in ['bao-qiso', 'bao-now-qiso']:
    #    profiles = profiler.profile(params=['qiso'], grid=np.linspace(0.8, 1.2, 100))
    attrs = {'options': output.options, 'zeff': likelihood.observables[0].wmatrix.theory.z}
    for name in ['data', 'wmatrix', 'covariance']:
        if name in kwargs: attrs[name] = str(kwargs[name])
    profiles.attrs.update(attrs)
    mpicomm = profiler.mpicomm
    #if mpicomm.rank == 0:
    #    profiles.save(output)
    print(profiles.to_stats(tablefmt='pretty'))
    likelihood(**profiles.bestfit.choice(input=True))
    if mpicomm.rank == 0:
        for observable in likelihood.observables[:1]:
            basename = os.path.splitext(output)[0]
            observable.plot(fn=basename + '.png')
            isbao = 'bao' in observable.wmatrix.theory.__class__.__name__.lower()
            if isbao:
                observable.plot_bao(fn=basename + '_bao.png')
            state = observable.__getstate__()
            for name in ['data', 'theory', 'std', 'covariance'] + (['theory_nobao'] if isbao else []):
                 state[name] = getattr(observable, name)
            for name in ['flattheory', 'shotnoise']:
                if name in state: state[name] = np.array(state[name])
            for name in ['theory', 'theory_nobao']:
                if name in state: state[name] = [np.array(array) for array in state[name]]
            nowindow = {'ells': observable.wmatrix.ellsin}
            names = ['s', 'corr'] if hasattr(observable.wmatrix.theory, 's') else ['k', 'power']
            nowindow.update({name: np.array(getattr(observable.wmatrix.theory, name)) for name in names})
            state['nowindow'] = nowindow
            profiles.attrs['observable'] = state
            profiles.save(output)
        for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
            profiles.to_stats(tablefmt=tablefmt, fn=os.path.splitext(output)[0] + '_stats.' + ext)
    return profiles


def sample(output, get_observable_likelihood=get_observable_likelihood, get_joint_observable_likelihood=get_joint_observable_likelihood, resume=False, **kwargs):
    import os
    import numpy as np
    from desilike.samplers import EmceeSampler, MCMCSampler, NUTSSampler
    options = {**output[0].options, **kwargs}
    if options.get('observable_name', []):
        likelihood = get_joint_observable_likelihood(save_emulator=False, solve='.best', **options)
    else:
        likelihood = get_observable_likelihood(save_emulator=False, solve='.best', **options)
    for param in likelihood.all_params.select(basename=['al*_*', 'bl*_*']):
        if param.varied: param.update(derived='.prec')
    isbao = 'sigmas' in likelihood.all_params
    save_fn = output.filepaths
    chains = None
    if resume: chains = save_fn
    #print(likelihood.all_params['omega_b'].prior)
    #print(likelihood())
    #print(likelihood({'qpar': 0.8000828814318354, 'qper': 1.005391875086447, 'dm': 0.008204776042503782, 'df': 0.9962172246685627, 'b1p': 1.0898722879825258, 'b2p': -0.18053865857765816, 'bsp': 2.0391770526505844}))
    #exit()
    #import jax
    #print(jax.default_backend(), jax.devices())
    if isbao:
        sampler = EmceeSampler(likelihood, chains=chains, nwalkers=4 * len(likelihood.varied_params), seed=42, save_fn=save_fn)
        chains = sampler.run(min_iterations=200, max_iterations=100000, check={'max_eigen_gr': 0.005})
    else:
        if not kwargs.get('emulator_fn', ''):
            sampler = MCMCSampler(likelihood, chains=chains, drag=False, learn={'max_eigen_gr': 1., 'min_eigen_gr': 0.0, 'every': '40 * ndim'}, seed=42, save_fn=save_fn)
            chains = sampler.run(min_iterations=200, max_iterations=100000, check_every=50, check={'max_eigen_gr': 0.01})
        else:
            sampler = NUTSSampler(likelihood, seed=42, save_fn=save_fn, ref_scale=0.01)
            chains = sampler.run(min_iterations=1000, check={'max_eigen_gr': 0.02, 'min_ess': 300})
        #chains = sampler.run(min_iterations=1000, check={'max_eigen_gr': 0.1, 'min_ess': 200})
        
    #sampler = NUTSSampler(likelihood, step_size=0.3, chains=chains, seed=42, save_fn=save_fn)
    #sampler = NUTSSampler(likelihood, step_size=0.01, covariance=save_fn, adaptation=False, seed=42, save_fn=save_fn, ref_scale=0.01)
    #sampler = NUTSSampler(likelihood, seed=42, save_fn=save_fn, ref_scale=0.01)
    #from desilike.samplers import MCMCSampler
    #sampler = MCMCSampler(likelihood, chains=chains, seed=42, save_fn=save_fn)
    #chains = sampler.run(min_iterations=1000, check={'max_eigen_gr': 0.03, 'min_ess': 300})
    #chains = sampler.run(min_iterations=200, max_iterations=100000, check={'max_eigen_gr': 0.01})
    mpicomm = sampler.mpicomm
    choice = None
    if mpicomm.rank == 0:
        attrs = {'options': output.options, 'zeff': likelihood.observables[0].wmatrix.theory.z}
        for name in ['data', 'wmatrix', 'covariance']:
            if name in kwargs: attrs[name] = str(kwargs[name])
        for chain, fn in zip(chains, sampler.save_fn):
            chain.attrs.update(attrs)
            chain.save(fn)
        chain = chains[0].concatenate([chain.remove_burnin(0.5)[::10] for chain in chains])
        choice = chain.choice(index='argmax', input=True)
    likelihood(**mpicomm.bcast(choice, root=0))
    base_save_fn = os.path.splitext(save_fn[0])[0][:-2]
    if mpicomm.rank == 0:
        for observable in likelihood.observables:
            basename = base_save_fn
            observable.plot(fn=basename + '.png')
            isbao = 'bao' in observable.wmatrix.theory.__class__.__name__.lower()
            if isbao:
                observable.plot_bao(fn=basename + '_bao.png')
            state = observable.__getstate__()
            for name in ['data', 'theory', 'std', 'covariance'] + (['theory_nobao'] if isbao else []):
                 state[name] = getattr(observable, name)
            for name in ['flattheory']:
                if name in state: state[name] = np.array(state[name])
            for name in ['theory', 'theory_nobao']:
                if name in state: state[name] = [np.array(array) for array in state[name]]
            for chain, fn in zip(chains, sampler.save_fn):
                chain.attrs['observable'] = state
                chain.save(fn)
        chain = chain.sample_solved()  # sample parameters that are marginalized over
        from desilike.samples import plotting
        try:
            for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
                chain.to_stats(tablefmt=tablefmt, fn=base_save_fn + '_stats.' + ext)
            plotting.plot_triangle(chain, fn=base_save_fn + '_triangle.png')
        except:
            import traceback
            traceback.print_exc()
    return chains


def importance(chains, output, get_observable_likelihood=get_observable_likelihood, **kwargs):
    import os
    import numpy as np
    from desilike.samplers import ImportanceSampler
    likelihood = get_observable_likelihood(save_emulator=False, solve='.marg', **{**output[0].options, **kwargs})
    chains = [chain.load().remove_burnin(0.3).ravel() for chain in chains]
    factor = max(min(5, np.mean([chain.shape[0] for chain in chains]).astype('i4') // 1000), 1)
    #factor = max(10, np.mean([chain.shape[0] for chain in chains]).astype('i4') // 500)
    chains = [chain[::factor] for chain in chains]
    chain_init = chains[0].concatenate([chain for chain in chains])
    save_fn = output.filepaths
    #for chain in chains:
    #    max_logposterior = chain.logposterior[np.isfinite(chain.logposterior)].max()
    #    chain.aweight[...] *= np.exp(max_logposterior - chain.logposterior)
    sampler = ImportanceSampler(likelihood, chains=chains, save_fn=save_fn)
    chains_imp = sampler.run(subtract_input=True)
    base_save_fn = os.path.splitext(save_fn[0])[0][:-2]
    if sampler.mpicomm.rank == 0:
        chain_imp = chains_imp[0].concatenate(chains_imp)
        chain_init = chain_init.sample_solved()  # sample parameters that are marginalized over
        chain_imp = chain_imp.sample_solved()
        from desilike.samples import plotting
        try:
            for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
                chain_imp.to_stats(tablefmt=tablefmt, fn=base_save_fn + '_stats.' + ext)
            plotting.plot_triangle([chain_init, chain_imp], labels=['initial', 'importance'], fn=base_save_fn + '_triangle.png')
        except:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':

    # To run with srun -n 64 python y1_data_fits_tools.py

    todo = ['emulator', 'profiling']
    #todo = ['sampling']
    #todo = ['test']

    from desipipe import setup_logging
    from desipipe.file_manager import BaseFile

    setup_logging()
    fit_type = 'power_full_shape'
    #fit_type = 'correlation_bao_desilikecov'
    #fit_type = 'correlation_bao_recon'
    #fit_type = 'correlation_baoiso_recon'

    if fit_type == 'power_full_shape':
        kwargs = {'theory_name': 'velocileptors', 'template_name': 'shapefit-qisoqap'}
        systs = '/global/cfs/cdirs/desi/users/ruiyang/systemplate/systcorr_mock_SecondGenY1_LRG_0.4_0.6_power_rpcut2.5_SN.npy'
        kwargs['data'] = BaseFile('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v1/blinded/desipipe/baseline_2pt/pk/pkpoles_LRG_GCcomb_z0.4-0.6.npy', filetype='power', options=dict(tracer='LRG', systs=systs))
        kwargs['wmatrix'] = BaseFile('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v1/blinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_GCcomb_z0.4-0.6.npy', filetype='wmatrix')
        kwargs['covariance'] = BaseFile('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/pk/covariances/cov_gaussian_pre_LRG_GCcomb_0.4_0.6_default_FKP_lin.txt', filetype='power_covariance')
        kwargs['emulator_fn'] = './_tests/emulator_LRG_0.4_0.6.npy'
    elif fit_type == 'correlation_bao_desilikecov':
        todo = [td for td in todo if td not in ['emulator']]
        kwargs = {'theory_name': 'resumbao', 'template_name': 'bao-qisoqap'}
        kwargs['data'] = BaseFile('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/xi/smu/allcounts_LRG_GCcomb_0.4_0.6_default_FKP_lin_njack0_nran4_split20.npy', filetype='correlation', options=dict(tracer='LRG'))
        all_data = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_{region}_clustering.dat.fits', filetype='catalog', options={'region': region}) for region in ['NGC', 'SGC']]
        footprint = BaseFile('_tests/footprint_LRG_{zrange[0]:.1f}_{zrange[1]:.1f}.npy', options={'region': 'GCcomb', 'zrange': [0.4, 0.6]})
        get_footprint(all_data, output=footprint)
        kwargs['footprint'] = footprint
    elif fit_type == 'correlation_bao_recon':
        todo = [td for td in todo if td not in ['emulator']]
        kwargs = {'theory_name': 'resumbao', 'template_name': 'bao-qisoqap', 'z': 0.5}
        kwargs['data'] = BaseFile('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/recon_sm10/xi/smu/allcounts_LRG_IFTrecsym_GCcomb_0.4_0.6_default_FKP_lin_njack0_nran4_split20.npy', filetype='correlation', options=dict(tracer='LRG', mode='recsym', smoothing_radius=10.))
        kwargs['covariance'] = BaseFile('/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/blinded/v0.6/xi024_LRG_IFTrecsym_sm10_NGC_0.4_0.6_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt', filetype='correlation_covariance', options=dict())
    elif fit_type == 'correlation_baoiso_recon':
        todo = [td for td in todo if td not in ['emulator']]
        kwargs = {'theory_name': 'resumbao', 'template_name': 'bao-qiso', 'z': 0.5}
        kwargs['data'] = BaseFile('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/recon_sm10/xi/smu/allcounts_LRG_IFTrecsym_GCcomb_0.4_0.6_default_FKP_lin_njack0_nran4_split20.npy', filetype='correlation', options=dict(tracer='LRG', mode='recsym', smoothing_radius=10.))
        kwargs['covariance'] = BaseFile('/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/blinded/v0.6/xi024_LRG_IFTrecsym_sm10_NGC_0.4_0.6_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt', filetype='correlation_covariance', options=dict())
        kwargs['lim'] = {0: [50., 150., 4.]}

    if 'emulator' in todo:
        likelihood = get_observable_likelihood(**kwargs, save_emulator=True)
        likelihood.mpicomm.barrier()  # just to wait all processes are done

    if 'profiling' in todo:
        from desilike.profilers import MinuitProfiler, ScipyProfiler
        likelihood = get_observable_likelihood(**kwargs)
        profiler = MinuitProfiler(likelihood, seed=42, save_fn='_tests/profiles.npy')
        profiler.maximize(niterations=10)

    if 'sampling' in todo:
        from desilike.samplers import EmceeSampler, ZeusSampler
        likelihood = get_observable_likelihood(**kwargs)
        save_fn = ['_tests/chain_{:d}.npy'.format(i) for i in range(1)]
        chains = len(save_fn)
        sampler = EmceeSampler(likelihood, chains=chains, nwalkers=40, seed=42)  #, save_fn=save_fn)
        sampler.run(min_iterations=2000, check={'max_eigen_gr': 0.03})

    if 'test' in todo:
        from desi_y1_files import get_data_file_manager, get_bao_baseline_fit_setup
        fm = get_data_file_manager(conf='unblinded')
        tracer, zrange = 'LRG+ELG_LOPnotqso', (0.8, 1.1)
        fit_type = 'bao_recon'
        options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon='recon' in fit_type, iso=False)
        options['region'] = 'GCcomb'
        options['version'] = 'v1.2'
        observable = options['observable']
        fdata = fm.get(id='{}{}_y1'.format(observable, '_recon' if 'recon' in fit_type else ''), **options, ignore=True)
        fwmatrix = None
        if observable == 'power':
            fwmatrix = fm.get(id='wmatrix_power_y1', **options, ignore=True)
        # For now, for power spectrum let's just take pre- for post-
        coptions = {**options, 'cut': None}
        coptions.pop('version')
        fcovariance = fm.get(id='covariance_{}{}_y1'.format(observable, '_recon' if 'recon' in fit_type else ''), **coptions, ignore=True)
        kwargs = dict(data=fdata, covariance=fcovariance, theory_name=options['theory'], template_name=options['template'], wmatrix=fwmatrix)
        profiles = fm.get(id='profiles_{}_y1'.format(fit_type), **options, ignore=True)
        kwargs.update(profiles.options)
        likelihood = get_observable_likelihood(solve=True, **kwargs)
        likelihood()
        profiles = profiles.load()
        #print(profiles.bestfit.choice(input=True))
        likelihood(**profiles.bestfit.choice(input=True))
        #likelihood(**profiles.bestfit.choice(input=True))
        likelihood.observables[0].plot(fn='tmp.png')
        print(likelihood.observables[0].theory[0])
        likelihood.observables[0].plot_bao(fn='tmp_bao.png')
        print(profiles.attrs['observable']['theory'][0])
        print(profiles.attrs['data'])
        
        #output = BaseFile('tmp.fits', filetype='profiles')
        #profiles2 = profile(output, get_observable_likelihood=get_observable_likelihood, **kwargs)
        #print(profiles2.attrs['observable']['theory'][0])
