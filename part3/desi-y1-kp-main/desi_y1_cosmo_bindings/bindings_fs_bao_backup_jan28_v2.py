from pathlib import Path
import sys
import numpy as np


# Force use of the local desilike checkout (with FOLPSv2) instead
# of the version installed in the environment.
LOCAL_DESILIKE_ROOT = "/global/cfs/cdirs/desicollab/users/hernannb/__FOLPS_tutorial/desilike"
if LOCAL_DESILIKE_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_DESILIKE_ROOT)
# Remove any previously imported desilike so that the local one
# takes precedence when re-importing.
if 'desilike' in sys.modules:
    del sys.modules['desilike']
import desilike, inspect
print(inspect.getfile(desilike))

data_dir = Path(__file__).parent / 'fs_bao_data'

list_zrange = [('BGS_BRIGHT-21.5', 0, (0.1, 0.4)), ('LRG', 0, (0.4, 0.6)), ('LRG', 1, (0.6, 0.8)), ('LRG', 2, (0.8, 1.1)), ('ELG_LOPnotqso', 1, (1.1, 1.6)), ('QSO', 0, (0.8, 2.1)), ('Lya', 0, (1.8, 4.2))]


def dataset_fn(tracer, zrange, observable_name='power+bao-recon', data_name='', klim=None, covsyst='rotation-hod-photo'):
    if data_name: data_name += '_'
    if 'power' in observable_name:
        observable_name += '_syst-{}'.format(covsyst)
    if not any(name in observable_name for name in ['power', 'shapefit']): klim = None
    if klim is None:
        klim = ''
    elif tuple(klim) == (0.02, 0.2):
        klim = '_klim_0-0.02-0.20_2-0.02-0.20'
    elif tuple(klim) == (0.02, 0.12):
        klim = '_klim_0-0.02-0.12_2-0.02-0.12'
    if observable_name == 'shapefit-joint':
        klim = ''
        observable_name = 'shapefit_power+bao-recon_syst-rotation-hod-photo'
        return data_dir / f'covariance_{observable_name}{klim}_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
    return data_dir / f'{data_name}forfit_{observable_name}{klim}_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'


def get_tracer_label(tracer):
    return tracer.split('_')[0].replace('+', 'plus')


def get_theory(theory_name='velocileptors', observable_name='power', freedom=None, prior_basis='physical', tracer=None, ells=(0, 2, 4)):

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
        for param in theory.init.params.select(basename=['bs*', 'b3*']):
            param.update(prior=dict(limits=[-50., 50.]))
        # Custom priors for FOLPSv2 in the "standard" (Eulerian) basis
        if 'folpsv2' in theory_name and prior_basis == 'standard':
            width_EFT = 240.
            width_SN0 = 100.
            width_SN2 = 100.
            params = theory.init.params
            params['b1'].update(prior={'dist': 'uniform', 'limits': [1e-5, 10.]})
            params['b2'].update(prior={'dist': 'uniform', 'limits': [-50., 50.]})
            params['bs'].update(prior={'dist': 'norm', 'loc': 0., 'scale': 20.})
            for name, width in [('alpha0', width_EFT), ('alpha2', width_EFT), ('alpha4', width_EFT)]:
                params[name].update(prior={'dist': 'norm', 'loc': 0., 'scale': width})
            for name, width in [('sn0', width_SN0), ('sn2', width_SN2)]:
                params[name].update(prior={'dist': 'norm', 'loc': 0., 'scale': width})
        # kwargs.update(mu=3)  # using 3 mu points in [0, 1] to reproduce FOLPS, by default it is 6
    elif 'velo' in theory_name:
        if 'rept' in theory_name:
            theory = (REPTVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else REPTVelocileptorsTracerCorrelationFunctionMultipoles)(freedom=freedom, prior_basis=prior_basis, tracer=tracer)
        else:
            theory = (LPTVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else LPTVelocileptorsTracerCorrelationFunctionMultipoles)(freedom=freedom, prior_basis=prior_basis, tracer=tracer)
    elif 'bao' in theory_name:
        theory = (DampedBAOWigglesTracerPowerSpectrumMultipoles if observable_name == 'power' else DampedBAOWigglesTracerCorrelationFunctionMultipoles)()

    # nuisance parameters
    for order in [4, 2]:
        if order not in ells:
            for param in theory.init.params.select(basename=['al{:d}*_*'.format(order), 'bl{:d}*_*'.format(order), 'alpha{:d}*'.format(order), 'sn{:d}*'.format(order)]): param.update(fixed=True)

    return theory


# change smin and kmax here
def get_fit_setup(tracer, zrange=None, theory_name='velocileptors', return_list=None):

    klim = {0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]} #, 4: [0.02, 0.1, 0.005]} # hexadecapole kmax
    kin = np.arange(0.001, 0.35, 0.001)  # window range used for convolution
    slim = {0: [30., 150., 4.], 2: [30., 150., 4.], 4: [40., 150., 4.]}
    sin = None
    iso = any(name in tracer for name in ['BGS', 'QSO'])
    sigmapar, sigmaper, sigmas = None, None, None
    if 'bao' in theory_name:
        ells = (0,) if iso else (0, 2)
        klim = {ell: [0.02, 0.3, 0.005] for ell in ells}
        #slim = {ell: [50., 150., 4.] for ell in ells}
        slim = {ell: [80., 130., 4.] for ell in ells}  # restricted s-range
        sigmapar, sigmaper, sigmas = (6., 2.), (3., 1.), (2., 2.)
        if 'BGS' in tracer:
            sigmapar, sigmaper = (8., 2.), (3., 1.)

    b1 = {'BGS': 1.5, 'LRG+ELG': 1.6, 'LRG': 2.0, 'ELG': 1.2, 'QSO': 2.1}
    for tr, b in b1.items():
        if tr in tracer:
            b1 = b
            break

    di = {'sigmapar': sigmapar, 'sigmaper': sigmaper, 'sigmas': sigmas, 'b1': b1, 'klim': klim, 'slim': slim, 'kin': kin}
    if return_list is None:
        return di
    return [di[name] for name in return_list]


from desilike.likelihoods import BaseLikelihood

class CustomPrior(BaseLikelihood):

    name = 'custom_prior'
    _calculate_with_namespace = True  # to get parameters with namespace in calculate

    def initialize(self, params=None, **kwargs):
        if params is not None:
            params = params.select(basename=['alpha*', 'sn*'])
            self.init.params.update(params)
        super().initialize(**kwargs)
        self.flatdata = []

    def calculate(self, **params):
        # Add custom prior into calculate
        self.loglikelihood = 0.
        for param, value in params.items():
            self.loglikelihood += -0.5 * 1e4 * (value - 0.)**2


def DESIFSLikelihood(tracers=None, cosmo=None, theory_name='reptvelocileptors', data_name='', klim=(0.02, 0.2), observable_name='power+bao-recon', freedom='max', solve='auto', custom_prior=False, jit=False, save_emulator=False, emulator_fn=False, covsyst='rotation-hod-photo'):

    """Return the (pre-recon) power spectrum (x post-recon BAO parameters) likelihood, optionally computing the emulator (if ``save_emulator``)."""
    from desilike import mpi

    if data_name == 'data':
        data_name = ''
    if solve and not solve.startswith('.'):
        solve = '.' + solve
    if custom_prior:
        solve = False
    #jit = True

    from cosmoprimo.fiducial import DESI
    from desilike.observables import ObservableCovariance
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, SystematicTemplatePowerSpectrumMultipoles
    from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, BAOPowerSpectrumTemplate
    from desilike.theories import Cosmoprimo
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike import LikelihoodFisher

    if emulator_fn:
        emulators_dir = Path(__file__).parent / 'fs_bao_emulators'
        if isinstance(emulator_fn, str):  # model
            emulators_dir = emulators_dir / emulator_fn
        emulator_fn = str(emulators_dir / 'emulator_{observable_name}_{theory_name}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy')
        from desilike.emulators import EmulatedCalculator
    fiducial = DESI()

    if cosmo is None:
        cosmo = DirectPowerSpectrumTemplate(fiducial='DESI').cosmo
        cosmo.init.params['sigma8_m'] = {'derived': True, 'latex': '\sigma_8'}  # derive sigma_8
        cosmo.init.params['tau_reio'].update(fixed=True)
        cosmo.init.params['omega_b'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
        cosmo.init.update(engine='class')

    # Choose prior basis for PT model
    # For FOLPSv2 we want the standard (Eulerian) basis with custom priors;
    # otherwise keep the original physical basis.
    if 'folpsv2' in str(theory_name):
        prior_basis = 'standard'
    else:
        prior_basis = 'physical'
    klim = tuple(klim)
    
    list_zrange = [('BGS_BRIGHT-21.5', 0, (0.1, 0.4)), ('LRG', 0, (0.4, 0.6)), ('LRG', 1, (0.6, 0.8)), ('LRG', 2, (0.8, 1.1)), ('ELG_LOPnotqso', 1, (1.1, 1.6)), ('QSO', 0, (0.8, 2.1)), ('Lya', 0, (1.8, 4.2))]

    this_zrange = []
    for tracer, iz, zrange in list_zrange:
        tracer_label = get_tracer_label(tracer)
        namespace = '{tracer}_z{iz}'.format(tracer=tracer_label, iz=iz)
        if tracers is not None and namespace.lower() not in tracers: continue
        #if 'lrg_z0' not in namespace.lower(): continue
        this_zrange.append((tracer, iz, zrange, namespace))

    order = 4
    likelihoods = []
    theories = []
    pt = None

    for tracer, iz, zrange, namespace in this_zrange:

        tracer_label = get_tracer_label(tracer)
        
        tracer_has_no_fs = 'Lya' in tracer_label

        observables = []
        ### Full shape part ###
        if 'power' in observable_name and not tracer_has_no_fs:
            #print(dataset_fn(data_name, tracer, zrange, observable_name='pre-power'))
            data = ObservableCovariance.load(dataset_fn(tracer, zrange, observable_name='power+bao-recon', data_name=data_name, klim=klim)).observables(observables='power')
            #wmatrix = dataset_fn('wmatrix', tracer, zrange, observable_name='pre-power')
            b1, = get_fit_setup(tracer, zrange=zrange, return_list=['b1'])
            template = DirectPowerSpectrumTemplate()
            template.init.update(z=data.attrs['zeff'], fiducial=fiducial, cosmo=cosmo)  # same cosmo for all templates
            theory = get_theory(theory_name=theory_name, observable_name='power', freedom=freedom, prior_basis=prior_basis, tracer=tracer_label, ells=data.projs)
            theory.init.update(template=template)
            if 'REPTVelocileptors' in theory.__class__.__name__ and (not bool(emulator_fn)) and (len(this_zrange) > 1):  # share same perturbation theory
                theory.init.update(pt=pt, z=data.attrs['zeff'])
                pt = theory.pt
                theory.init.update(pt=pt)
            if 'b1p' in theory.init.params:  # physical
                b1p = b1 * fiducial.sigma8_z(data.attrs['zeff'])
                theory.init.params['b1p'].update(value=b1p, ref=dict(dist='norm', loc=b1p, scale=0.1))
            else:
                theory.init.params['b1'].update(value=b1, ref=dict(dist='norm', loc=b1, scale=0.1))
            # Define observable
            observable = TracerPowerSpectrumMultipolesObservable(data=data, theory=theory, wmatrix=data.attrs['wmatrix'], kin=data.attrs['kin'], ellsin=data.attrs['ellsin'], wshotnoise=data.attrs['wshotnoise'])
            templates, priors = {}, {}
            for syst in ['rotation', 'photo']:
                if syst not in covsyst:  # add parameters
                    tmp = ObservableCovariance.load(dataset_fn(tracer, zrange, observable_name='power+bao-recon', data_name=data_name, klim=klim)).attrs[syst]
                    ntemplates = len(tmp['templates'])
                    names = ['{}_{:d}'.format(syst, i) for i in range(ntemplates)]
                    templates.update({name: template for name, template in zip(names, tmp['templates'])})
                    priors.update({name: tmp['prior'][i] for i, name in enumerate(names)})
            if templates:
                systematic_templates = SystematicTemplatePowerSpectrumMultipoles(templates=templates)
                for name in templates:
                    param = systematic_templates.init.params[name]
                    param.update(derived=solve, prior={'dist': 'norm', 'loc': 0., 'scale': priors[name]**0.5}, namespace='pre_{}'.format(namespace), latex=param.latex(namespace=r'\mathrm{{pre}}, \mathrm{{{}}}, {:d}'.format(tracer_label, iz), inline=False))
                observable.init.update(systematic_templates=systematic_templates)
                
            #observable()
            #print(data.x(), observable.flatdata.sum(), observable.shotnoise, observable.k, observable.wmatrix.matrix_full.shape, observable.wmatrix.matrix_full.sum(), observable.wmatrix.kin, observable.wmatrix.ellsin)
            #exit()
            observables.append(observable)

            if True:
                # Compute or swap in PT emulator
                emu_fn = emulator_fn.format(observable_name='power', theory_name=theory_name + '_prior-basis-{}'.format(prior_basis), tracer=tracer, zrange=zrange) if emulator_fn else None
                if save_emulator:  # Compute and save emulator
                    if not emu_fn:
                        raise ValueError('provide emulator_fn')
                    observable()  # to set up k-ranges for the emulator
                    from desilike.emulators import Emulator, TaylorEmulatorEngine
                    theory = observable.wmatrix.theory
                    #theory.init.update(ells=(0, 2, 4))  # train emulator on all multipoles
                    emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=order))
                    emulator.set_samples()
                    emulator.fit()
                    emulator.save(emu_fn)
                    #theory.init.update(pt=emulator.to_calculator(), ells=list(klim))  # reset multipoles
                elif emu_fn: # Load emulator
                    calculator = EmulatedCalculator.load(emu_fn)
                    # Update emulator with cosmo
                    if cosmo is not None:
                        for param in cosmo.init.params:
                            if param in calculator.init.params:
                                calculator.init.params.set(param)
                    theory.init.update(pt=calculator)

            # Update namespace of bias parameters (to have one parameter per tracer / z-bin)
            for param in theory.init.params:
                # Update latex just to have better labels
                param.update(namespace='pre_{}'.format(namespace),
                             latex=param.latex(namespace=r'\mathrm{{pre}}, \mathrm{{{}}}, {:d}'.format(tracer_label, iz), inline=False))
            theories.append(theory)
        
        elif 'shapefit' in observable_name and not tracer_has_no_fs:
            from desilike.observables.galaxy_clustering import ShapeFitCompressionObservable
            forfit = ObservableCovariance.load(dataset_fn(tracer, zrange, observable_name='shapefit+bao-recon' if observable_name == 'shapefit' else observable_name, data_name=data_name, klim=klim)).select(observables='shapefit', select_observables=True)
            covariance, data = forfit.view(), forfit.observables()[0]
            observable = ShapeFitCompressionObservable(data=data.view(), covariance=covariance, cosmo=cosmo, quantities=data.projs, fiducial=fiducial, z=data.attrs['zeff'])
            #observable.init.update(cosmo=cosmo)
            flatdata = observable.flatdata
 
            if True:
                emu_fn = emulator_fn.format(observable_name='shapefit', theory_name='shapefit', tracer=tracer, zrange=zrange) if emulator_fn else None
                if save_emulator:
                    if not emu_fn:
                        raise ValueError('provide emulator_fn')
                    observable()
                    from desilike.emulators import Emulator, TaylorEmulatorEngine
                    temp = observable
                    emulator = Emulator(observable, engine=TaylorEmulatorEngine(method='finite', order=order))
                    emulator.set_samples()
                    emulator.fit()
                    emulator.save(emu_fn)
                    observable = emulator.to_calculator()
                elif emu_fn: # Load emulator
                    calculator = EmulatedCalculator.load(emu_fn)
                    # Update emulator with cosmo
                    for param in cosmo.init.params:
                        if param in calculator.init.params:
                            calculator.init.params.set(param)
                    observable = calculator

            observable.flatdata = flatdata
            observables.append(observable)

        ### BAO part ###
        if 'correlation-recon' in observable_name and not tracer_has_no_fs:
            data = ObservableCovariance.load(dataset_fn(tracer, zrange, observable_name='power+correlation-recon', data_name=data_name, klim=klim)).observables(observables='correlation-recon')
            b1, sigmapar, sigmaper, sigmas = get_fit_setup(tracer, zrange=zrange, theory_name='bao', return_list=['b1', 'sigmapar', 'sigmaper', 'sigmas'])
            template = BAOPowerSpectrumTemplate()
            template.init.update(z=data.attrs['zeff'], fiducial=fiducial, apmode='bao', cosmo=cosmo)  # same cosmo for all templates
            theory = get_theory(theory_name='dampedbao', observable_name='correlation', ells=data.projs)
            theory.init.update(template=template, broadband='pcs2')
            theory.init.params['b1'].update(value=b1, ref=dict(dist='norm', loc=b1, scale=1.))  # just to help the fit
            for name, sigma in {'sigmapar': sigmapar, 'sigmaper': sigmaper, 'sigmas': sigmas}.items():
                theory.init.params[name].update(value=sigma[0], prior=dict(dist='norm', loc=sigma[0], scale=sigma[1]), fixed=False)
            #theory.init.params['dbeta'].update(fixed=False if 2 in slim else True)
            # Define observable
            observable = TracerCorrelationFunctionMultipolesObservable(data=data, theory=theory)
            observables.append(observable)
            
            if True:
                # Compute or swap in qpar / qper emulator
                emu_fn = emulator_fn.format(observable_name='correlation-recon', theory_name='dampedbao', tracer=tracer, zrange=zrange) if emulator_fn else None
                if save_emulator:  # Compute and save emulator
                    if not emu_fn:
                        raise ValueError('provide emulator_fn')
                    observable()  # to set up k-ranges for the emulator
                    from desilike.emulators import Emulator, TaylorEmulatorEngine
                    theory = observable.wmatrix.theory
                    params = cosmo.init.params.deepcopy()
                    #cosmo.init.params.pop('sigma8_m', None)
                    temp = theory.pt.template
                    temp.all_params.pop('sigma8_m')
                    for param in temp.all_params.select(basename=['logA', 'n_s']):
                        param.update(fixed=True)
                    emulator = Emulator(temp, engine=TaylorEmulatorEngine(method='finite', order=order))
                    emulator.set_samples()
                    emulator.fit()
                    emulator.save(emu_fn)
                    theory.init.update(template=emulator.to_calculator())  # reset multipoles
                elif emu_fn: # Load emulator
                    calculator = EmulatedCalculator.load(emu_fn)
                    # Update emulator with cosmo
                    for param in cosmo.init.params:
                        if param in calculator.init.params:
                            calculator.init.params.set(param)
                    theory.init.update(template=calculator)
            
            for param in theory.init.params:
                param.update(namespace='post_{}'.format(namespace),
                             latex=param.latex(namespace=r'\mathrm{{post}}, \mathrm{{{}}}, {:d}'.format(tracer_label, iz), inline=False))
            theories.append(theory)

        elif 'bao-recon' in observable_name or ('shapefit-joint' in observable_name and tracer_has_no_fs):
            from desilike.observables.galaxy_clustering import BAOCompressionObservable
            forfit = ObservableCovariance.load(dataset_fn(tracer, zrange, observable_name='bao-recon' if tracer_has_no_fs else 'power+bao-recon', data_name=data_name, klim=klim)).select(observables='bao-recon', select_observables=True)
            covariance, data = forfit.view(), forfit.observables()[0]            
            observable = BAOCompressionObservable(data=data.view(), covariance=covariance, cosmo=cosmo, quantities=data.projs, fiducial=fiducial, z=data.attrs['zeff'])
            #observable.init.update(cosmo=cosmo)
            flatdata = observable.flatdata
 
            if True:
                emu_fn = emulator_fn.format(observable_name='bao-recon', theory_name='bao', tracer=tracer, zrange=zrange) if emulator_fn else None
                if save_emulator:
                    if not emu_fn:
                        raise ValueError('provide emulator_fn')
                    observable()
                    from desilike.emulators import Emulator, TaylorEmulatorEngine
                    temp = observable
                    temp.all_params.pop('sigma8_m', None)
                    for param in temp.all_params.select(basename=['logA', 'n_s']):
                        param.update(fixed=True)
                    emulator = Emulator(observable, engine=TaylorEmulatorEngine(method='finite', order=order))
                    emulator.set_samples()
                    emulator.fit()
                    emulator.save(emu_fn)
                    observable = emulator.to_calculator()
                elif emu_fn: # Load emulator
                    calculator = EmulatedCalculator.load(emu_fn)
                    # Update emulator with cosmo
                    for param in cosmo.init.params:
                        if param in calculator.init.params:
                            calculator.init.params.set(param)
                    observable = calculator
            observable.flatdata = flatdata
            observables.append(observable)
            #theories.append(observable)

        ### Likelihood ###
        if observables:
            covariance = None
            if not tracer_has_no_fs and observable_name != 'bao-recon':
                if observable_name == 'shapefit':
                    covariance = ObservableCovariance.load(dataset_fn(tracer, zrange, observable_name='shapefit+bao-recon', data_name=data_name, klim=klim, covsyst=covsyst)).select(observables='shapefit', select_observables=True)
                else:
                    covariance = ObservableCovariance.load(dataset_fn(tracer, zrange, observable_name=observable_name, data_name=data_name, klim=klim, covsyst=covsyst))
            likelihood = ObservablesGaussianLikelihood(observables=observables, name=namespace, covariance=covariance)
            likelihoods.append(likelihood)

    #likelihood = sum(likelihoods)  # likelihood is a callable that returns the log-posterior
    if len(likelihoods) > 1: likelihood = sum(likelihoods)
    else: likelihood = likelihoods[0]  # to avoid duplicate loglikelihood derived parameter in cobaya when using multiple likelihoods independently

    if isinstance(cosmo, str) and cosmo == 'external':
        likelihood.mpicomm = mpi.COMM_SELF
    # Analytic marginalisation
    #return likelihood
    for param in likelihood.all_params.select(basename=['al*_*', 'bl*_*']):
        if param.varied: param.update(derived='.prec')  # remove mode from data precision matrix once for all

    if solve:
        for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*']):
            if param.varied: param.update(derived=solve)

        if likelihood.mpicomm.rank == 0:
            likelihood.log_info('Use analytic marginalization for {}.'.format(likelihood.all_params.names(solved=True)))

    if custom_prior:
        likelihood = likelihood + CustomPrior(params=likelihood.all_params)

    if jit and theories:
        from desilike.base import jit
        likelihood = jit(likelihood, index=theories)

    return likelihood



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Y1 FS cosmological likelihoods')
    parser.add_argument('--todo', type=str, nargs='*', required=False, default=['bindings'], choices=['copy', 'bindings', 'emulate', 'test', 'sampling'])
    parser.add_argument('--theory-name', type=str, required=False, default='folpsv2',
                        choices=['folpsax', 'folpsv2', 'reptvelocileptors', 'velocileptors'],
                        help='Perturbation theory model to use when running in emulate mode.')
    args = parser.parse_args()

    from desilike import utils, setup_logging
    from desilike.bindings import CobayaLikelihoodGenerator, CosmoSISLikelihoodGenerator, MontePythonLikelihoodGenerator

    chains_dir = 'desilike'

    if 'copy' in args.todo:

        from desilike import utils, setup_logging
        from desilike.observables import ObservableCovariance

        #setup_logging()
        utils.mkdir(data_dir)
        
        from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable
        from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, FOLPSTracerPowerSpectrumMultipoles, FOLPSAXTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles
        from desilike.likelihoods import ObservablesGaussianLikelihood
        from desilike.profilers import MinuitProfiler
        from desilike.samples import Chain

        import sys
        sys.path.insert(0, '../scripts')  # for load_cobaya_bao_samples

        in_dir = Path('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/')
        center = {'qiso': 1., 'qap': 1., 'qpar': 1., 'qper': 1., 'df': 1., 'dm': 0.}
        for tracer, _, zrange in list_zrange:
            for observable_name in (['bao-recon'] if 'Lya' in tracer else ['power', 'power+bao-recon', 'power+correlation-recon', 'shapefit+bao-recon', 'shapefit-joint']):  # we need power-only covariance for fs-only
                if 'power' in observable_name or 'shapefit' in observable_name:
                    klims = [(0.02, 0.2), (0.02, 0.12)][:1]
                else:
                    klims = [None]
                for klim in klims:
                    for covsyst in ['hod', 'hod-photo', 'rotation-hod-photo']:  # last must be 'rotation-hod-photo' for what follows
                        fn = dataset_fn(tracer, zrange, observable_name=observable_name, klim=klim, covsyst=covsyst)
                        covariance = ObservableCovariance.load(in_dir / fn.name)
                        covariance.save(fn)
                    continue  # skip mocks
                    for data_name in ['mock-fiducial', 'mock-cmblens']:
                        from cosmoprimo.fiducial import DESI
                        fiducial = DESI()
                        if data_name == 'mock-cmblens':
                            from y1_bao_cosmo_tools import load_cobaya_samples as load_cobaya_bao_samples
                            chain = load_cobaya_bao_samples(model='base', run='run0', dataset=['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], label='CMB lensing', add=['planck-act-dr6-lensing'], source='jiamingp')
                            bestfit = {'h': chain.mean('H0') / 100., 'omega_b': chain.mean('ombh2'), 'omega_cdm': chain.mean('omch2'), 'logA': chain.mean('logA'), 'n_s': chain.mean('ns'), 'tau_reio': chain.mean('tau')}
                            # print(bestfit, {name: fiducial[name] for name in bestfit})
                            fiducial = fiducial.clone(**bestfit)

                        # We create synthetic datasets
                        for data in covariance.observables():
                            if any(name in data.name for name in ['bao', 'shapefit']):
                                for iproj, proj in enumerate(data.projs):
                                    data._value[iproj][...] = center[proj]
                            elif data.name == 'power':
                                theory = get_theory(theory_name='reptvelocileptors', observable_name='power', freedom='max', prior_basis='standard', tracer=get_tracer_label(tracer), ells=list(data.projs))
                                template = FixedPowerSpectrumTemplate(fiducial=fiducial, z=data.attrs['zeff'])
                                #template()
                                #print(data_name, template.pk_dd.sum())
                                theory.init.update(template=template)
                                observable = TracerPowerSpectrumMultipolesObservable(data=data, theory=theory, wmatrix=data.attrs['wmatrix'], kin=data.attrs['kin'], ellsin=data.attrs['ellsin'], wshotnoise=data.attrs['wshotnoise'], covariance=covariance.view(observables='power'))
                                likelihood = ObservablesGaussianLikelihood(observables=observable)
                                for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*']):
                                    if param.varied: param.update(derived='.auto')
                                likelihood.all_params['b3'].update(fixed=True)
                                profiler = MinuitProfiler(likelihood, seed=42)
                                profiles = profiler.maximize()
                                values = {param.name: param.value for param in likelihood.all_params.select(input=True)}
                                b1 = values['b1'] = profiles.bestfit.choice()['b1']
                                params = [param for param in likelihood.all_params.select(input=True) if param.prior.dist == 'uniform']
                                values.update(profiles.bestfit.choice(index='argmax', params=params))
                                likelihood(values)
                                observable.plot(fn=str(dataset_fn(tracer, zrange, observable_name=data.name)).replace('.npy', '.png'))
                                data._value = observable.theory
                            elif data.name == 'correlation-recon':
                                b1, sigmapar, sigmaper, sigmas = get_fit_setup(tracer, zrange=zrange, theory_name='bao', return_list=['b1', 'sigmapar', 'sigmaper', 'sigmas'])
                                theory = get_theory(theory_name='dampedbao', observable_name='correlation', ells=list(data.projs))
                                theory.init.update(template=FixedPowerSpectrumTemplate(z=data.attrs['zeff']), broadband='pcs2')
                                observable = TracerCorrelationFunctionMultipolesObservable(data=data, theory=theory, covariance=covariance.view(observables='correlation-recon'))
                                values = {param.name: param.value for param in observable.all_params.select(input=True)}
                                values['b1'] = b1
                                values['sigmapar'], values['sigmaper'], values['sigmas'] = sigmapar[0], sigmaper[0], sigmas[0]
                                observable(values)
                                observable.plot(fn=str(dataset_fn(tracer, zrange, observable_name='correlation-recon')).replace('.npy', '.png'))
                                data._value = observable.theory
                            else:
                                raise ValueError
                        fn = dataset_fn(tracer, zrange, observable_name=observable_name, klim=klim, data_name=data_name)
                        covariance.save(fn)

    if 'bindings' in args.todo:
        setup_logging('info')
        for observable in ['fs_bao', 'fs', 'shapefit', 'shapefit_bao', 'shapefit_joint'][-1:]:
            for data in ['', '_synthetic'][:1]:
                name_likes, kw_likes = [], []
                tracers = {'all': None, 'all_nolya': ['bgs_z0', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'elg_z1', 'qso_z0'], 'bgs': ['bgs_z0'], 'lrg': ['lrg_z0', 'lrg_z1', 'lrg_z2'], 'elg': ['elg_z1'], 'qso': ['qso_z0'], 'lya': ['lya_z0']}
                for tracer in ['all', 'all_nolya', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'elg', 'qso']:
                    name_like = 'desi{}_{}_{}'.format(data, observable, tracer)
                    kw_like = {'cosmo': 'external', 'data_name': data.replace('_', ''), 'observable_name': {'fs_bao': 'power+bao-recon', 'fs': 'power', 'shapefit_bao': 'shapefit+bao-recon', 'shapefit_nodm_bao': 'shapefit-nodm+bao-recon', 'shapefit_nodf_bao': 'shapefit-nodf+bao-recon', 'shapefit': 'shapefit', 'shapefit_joint': 'shapefit-joint'}[observable], 'tracers': tracers.get(tracer, [tracer]), 'theory_name': 'reptvelocileptors'}
                    name_likes.append(name_like)
                    kw_likes.append(kw_like)
                    fn = 'desi{}_{}.py'.format(data, observable)
                kw_cobaya = {'klim': (0.02, 0.2)}
                if 'fs' in observable:
                    kw_cobaya.update({'solve': '.marg', 'theory_name': 'reptvelocileptors', 'jit': True, 'emulator_fn': False})
                CobayaLikelihoodGenerator()([DESIFSLikelihood] * len(name_likes), name_likes, kw_like=kw_likes, kw_cobaya=kw_cobaya, fn=fn)
                CosmoSISLikelihoodGenerator()([DESIFSLikelihood] * len(name_likes), name_likes, kw_like=kw_likes, fn=fn)
                MontePythonLikelihoodGenerator()([DESIFSLikelihood] * len(name_likes), name_likes, kw_like=kw_likes, fn=fn)
    if 'emulate' in args.todo:
        setup_logging()
        # Generate PT emulator for the chosen full-shape theory
        # e.g. --theory-name folpsax  or  --theory-name folpsv2
        likelihood = DESIFSLikelihood(theory_name=args.theory_name, emulator_fn='emulator', save_emulator=True)

    if 'test' in args.todo:
        setup_logging()

        if False:
            likelihood = DESIFSLikelihood()
            likelihood()

        if False:
            kw = dict(tracers='elg_z1', theory_name='reptvelocileptors', observable_name='power+bao-recon', solve=False)
            likelihood = DESIFSLikelihood(**kw)
            params = {'h': 0.6760835707650561, 'omega_cdm': 0.11983408283859458, 'omega_b': 0.022467153280715104, 'logA': 3.0577164179897123, 'n_s': 0.963916555826162, 'pre_ELG_z1.b1p': 0.5595813405737191, 'pre_ELG_z1.b2p': 1.5792128155073915, 'pre_ELG_z1.bsp': 0.7674347291529088, 'pre_ELG_z1.alpha0p': -0.4694743859349521, 'pre_ELG_z1.alpha2p': 0.5425600435859647, 'pre_ELG_z1.sn0p': -0.46341769281246226, 'pre_ELG_z1.sn2p': -0.46572975357025687}
            print(likelihood(params), likelihood.loglikelihood)
            
            kw = dict(tracers='elg_z1', theory_name='reptvelocileptors', observable_name='power+bao-recon', custom_prior=True)
            likelihood = DESIFSLikelihood(**kw)
            print(likelihood(params), likelihood.loglikelihood)

        if False:
            from cosmoprimo.fiducial import DESI
            from desilike.bindings.cobaya import CobayaLikelihoodFactory

            for klim in [(0.02, 0.2), (0.02, 0.12)]:
                kw = dict(cosmo='external', theory_name='reptvelocileptors', solve='.auto', klim=klim)
                CobayaLikelihood = CobayaLikelihoodFactory(DESIFSLikelihood, params=True, kw_like=kw)

                cosmo = DESI()
                # No magic here, this is all Cobaya stuff
                params = {'Omega_m': {'prior': {'min': 0.1, 'max': 1.},
                                      'ref': {'dist': 'norm', 'loc': 0.3, 'scale': 0.01},
                                      'latex': '\Omega_\mathrm{m}'},
                          'm_ncdm': cosmo['m_ncdm_tot'],
                          'H0': cosmo['H0'],
                          'A_s': cosmo['A_s'],
                          'n_s': cosmo['n_s'],
                          'tau_reio': cosmo['tau_reio']}

                info = {'params': params,
                        'likelihood': {'bao_likelihood': CobayaLikelihood},
                        'theory': {'classy': {'extra_args': {'N_ncdm': cosmo['N_ncdm'], 'N_ur': cosmo['N_ur']}}},
                        'sampler': {'mcmc': {'drag': False,
                                             'oversample_power': 0.4,
                                             'proposal_scale': 1.9,
                                             'Rminus1_stop': 0.01,
                                             'Rminus1_cl_stop': 0.2,
                                             'seed': 42,
                                             'max_tries': 1000}}}
                from cobaya import run
                updated_info, mcmc = run(info, test=True, force=True, resume=False)
        
        if False:
            kw = dict(theory_name='reptvelocileptors', solve='.auto')
            likelihood = DESIFSLikelihood(**kw, jit=True)
            likelihood()
            likelihood2 = DESIFSLikelihood(**kw, jit=False)
            from desilike.base import jit
            niterations = 5
            rng = np.random.RandomState(seed=42)
            #likelihood = jit(likelihood, index=[calculator for calculator in likelihood.runtime_info.pipeline.calculators if calculator.__class__.__name__.startswith('FOLPSAXPowerSpectrum')])
            for i in range(niterations):
                params = {param.name: param.ref.sample(random_state=rng) for param in likelihood2.varied_params if param.namespace}
                like1 = likelihood(params)
                like2 = likelihood2(params)
                print(like2, like1, like2 - like1)
            exit()
            likelihood.runtime_info.pipeline._set_speed()
            import time
            t0 = time.time()
            for i in range(niterations):
                likelihood({param.name: param.ref.sample(random_state=rng) for param in likelihood2.varied_params})
            print((time.time() - t0) / niterations)
        
        if False:
            likelihood = DESIFSLikelihood(tracers=None, theory_name='reptvelocileptors', solve='.best', jit=True)
            likelihood.runtime_info.pipeline._set_speed()
            rng = np.random.RandomState(seed=42)
            niterations = 5
            import time
            t0 = time.time()
            for i in range(niterations):
                likelihood({param.name: param.ref.sample(random_state=rng) for param in likelihood.varied_params})
            print((time.time() - t0) / niterations)
            
        if False:
            likelihood = DESIFSLikelihood(tracers=['lrg_z0', 'lya_z0'], solve='.best', jit=False)
            likelihood2 = DESIFSLikelihood(tracers=['lrg_z0', 'lya_z0'], solve='.best', jit=True)
            params = {param.name: param.ref.sample() for param in likelihood.varied_params}
            (l1, d1), (l2, d2) = likelihood(params, return_derived=True), likelihood2(params, return_derived=True)
            assert np.allclose(l2, l1, atol=0.)
            for param in d1.params():
                assert np.allclose(d2[param], d1[param])
            for param in likelihood.varied_params: param.update(prior=None)
            _, derived = likelihood2(return_derived=True)
            print(derived)
            likelihood.runtime_info.pipeline._set_speed()
            likelihood2.runtime_info.pipeline._set_speed()

    if 'sampling' in args.todo:
        setup_logging('info')
        likelihood_names = ['fs_bao', 'pantheonplus']

        from desilike.theories import Cosmoprimo
        from desilike.samplers import ZeusSampler, EmceeSampler

        cosmo = Cosmoprimo(fiducial='DESI')
        cosmo.init.params = {'Omega_m': {'prior': {'limits': [0.1, 0.9]}, 'ref': {'dist': 'norm', 'loc': 0.3, 'scale': 0.002}, 'latex': '\Omega_m'},
                            'omega_b': {'prior': {'dist': 'norm', 'loc': 0.02236, 'scale': 0.0005}, 'latex': '\omega_b'},
                            'H0':  {'prior': {'limits': [20., 100.]}, 'ref': {'dist': 'norm', 'loc': 70., 'scale': 1.}, 'latex': 'H_{0}'}}
        likelihoods = []

        if 'fs_bao' in likelihood_names:
            likelihood = DESIFSLikelihood(cosmo=cosmo, tracers=['lrg_z0', 'elg_z1'])
            likelihood()
            likelihoods.append(likelihood)

        if 'pantheonplus' in likelihood_names:
            from desilike.likelihoods.supernovae import PantheonPlusSNLikelihood
            likelihood = PantheonPlusSNLikelihood(cosmo=cosmo)
            from desilike.install import Installer
            installer = Installer()
            installer(likelihood)
            likelihoods.append(likelihood)

        chains_dir = Path(__file__).parent / 'chains_{}'.format('_'.join(likelihood_names))
        sampler = EmceeSampler(sum(likelihoods), nwalkers=60, seed=42, save_fn=Path(__file__).parent / 'chain.1.npy')
        sampler.run(check_every=10, check={'max_eigen_gr': 0.02})