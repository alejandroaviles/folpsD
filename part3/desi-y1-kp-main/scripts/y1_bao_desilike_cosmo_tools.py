"""
Run cosmological inference with Y1.
Authors:
- Uendert Andrade
- Arnaud de Mattia
- Peter Taylor, DESI - DES combinations
- Hanyu Zhang
- Kushal Lodha
- Nhat-Minh Nguyen

Configs (datasets, models) are listed in :func:`yield_configs`: these correspond to the cobaya samples available on disk.
To load cobaya samples, use :func:`load_cobaya_samples` in getdist format.
To add a source where the find samples, look at :func:`get_cobaya_output`. Just add the new source, following ['hanyuz', 'kushal', 'jiamingp'].
To add aliases for parameter names, look at ``list_renames`` in :func:`load_cobaya_samples`. There you can also set default derived parameters.

See ``y1_bao_desilike_cosmo.py`` for the script running chains with :mod:`desipipe`.
"""

import os
from pathlib import Path

base_dir = Path(os.getenv('SCRATCH')) / 'tests_desilike_cosmo/'



class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_parameterization(dataset):
    """Return parameterization type ('thermodynamics' when absolute BAO scale information, 'background' (no absolute BAO scale information), 'cmb', 'cmb-lensing')."""
    if not isinstance(dataset, str):
        dataset = '_'.join(dataset)
    if ('fs' in dataset or 'shapefit' in dataset or 'standard' in dataset or 'desy3' in dataset) and 'tt' not in dataset.lower():
        return 'cmb-pk'
    if any(name in dataset for name in ['bbn', 'thetastar']):
        return 'thermodynamics'
    if 'planck2018-rdrag' in dataset:
        return 'background'
    if 'planck' in dataset:
        if 'lensing' in dataset:
            return 'cmb-lensing'
        return 'cmb'
    return 'background'

def yield_configs(importance=False, models=None, **kwargs):
    for config in []:
        yield config


def sort_dataset(dataset, yield_configs=yield_configs):
    """Sort input dataset."""
    for config in yield_configs(importance=False):
        if set(dataset) == set(config['dataset']):
            return config['dataset']
    return dataset


def make_table(samples, params, add_rule=False, fn=None):
    """
    Create table, by Tianke Zhuang.

    Parameters
    ----------
    samples : dict
        Dictionary mapping (model, dataset) to samples.

    params : list
        List of parameters to record in the table.

    add_rule : bool, default=False
        If ``True``, add rules (lines) between rows.

    fn : str, Path, default=None
        If provided, save table to this path.
    """
    from astropy.io import ascii
    from astropy.table import Table
    labels = {}
    for param in params:
        for sample in samples.values():
            par = sample.paramNames.parWithName(param)
            if par is not None:
                labels[param] = par.label
                break
    labels = [labels.get(param, param) for param in params]
    names = ['model', 'dataset'] + labels
    table = Table(names=names, dtype=['S70'] * len(names))

    for (model, dataset), sample in samples.items():
        table.add_row()
        row = table[-1]
        row['model'] = model
        row['dataset'] = dataset

        for param, label in zip(params, labels):
            par = sample.paramNames.parWithName(param)
            if par is not None:
                row[label] = '${}$'.format(sample.getLatex(params=[par.name], limit=1, err_sig_figs=2)[1][0])

        for colname in table.colnames:
            if not row[colname]: row[colname] = '--'

    import io
    stout = io.StringIO()
    ascii.write(table, stout, format='latex')
    table = stout.getvalue()
    if add_rule:
        table = table.split('\n')
        new = []
        for ind, row in enumerate(table):
            new.append(row)
            if ind == 1:
                new.append('\\toprule')
            elif ind < (len(table) - 3):
                new.append('\\midrule')
        table = '\n'.join(new)
    if fn:
        with open(fn, 'w') as file:
            file.write(table)
    return table


def get_desilike_cosmo(model='base', engine='classy', dataset='desi-fs-bao-all', get_parameterization=get_parameterization):
    """Return cobaya ``params`` dictionary, given input model, theory and dataset."""
    params, extra = {}, {}
    if not isinstance(dataset, str):
        dataset = '_'.join(dataset)
    
    def fix(name):
        di = dict(params[name])
        di['value'] = params[name]['ref']['loc']
        di['fixed'] = True
        params[name] = di

    # First define all parameters (even if redundant), then select
    params['logA'] = {'prior': {'limits': [1.61, 3.91]},
                      'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.01},
                      'latex': r'\ln(10^{10} A_\mathrm{s})',
                      'delta': 0.07,
                      'drop': True}
    params['A_s'] = {'derived': '1e-10*jnp.exp({logA})',
                    'latex': r'A_\mathrm{s}'}
    params['n_s'] = {'prior': {'limits': [0.8, 1.2]}, 
                    'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004},
                    'latex': r'n_\mathrm{s}',
                    'delta': 0.01}
    params['theta_MC_100'] = {'prior': {'limits': [0.5, 10.]},
                              'ref': {'dist': 'norm', 'loc': 1.04109, 'scale': 0.0004},
                              'latex': r'100\theta_\mathrm{MC}', 'delta': 0.001}
    params['H0'] = {'derived': True, 'latex': 'H_0'}
    params['omega_b'] = {'prior': {'limits': [0.005, 0.1]},
                        'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001},
                        'latex': r'\Omega_\mathrm{b} h^2',
                        'delta': 0.0015}

    params['omega_cdm'] = {'prior': {'limits': [0.001, 0.99]},
                           'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001},
                           'latex': r'\Omega_\mathrm{c} h^2',
                           'delta': 0.01}
    params['Omega_m'] = {'derived': True, 'latex': r'\Omega_\mathrm{m}'}
    # FIXME
    #params['omega_m'] = {'derived': 'lambda Omega_m, H0: Omega_m * (H0/100)**2',
    #                     'latex': r'\Omega_\mathrm{m} h^2'}
    params['tau_reio'] = {'prior': {'limits': [0.01, 0.8]},
                          'ref': {'dist': 'norm', 'loc': 0.0544, 'scale': 0.006},
                          'proposal': 0.003,
                          'latex': r'\tau_\mathrm{reio}',
                          'delta': 0.01}
    params['m_ncdm'] = {'prior': {'limits': [0., 5.]},
                        'ref': {'dist': 'norm', 'loc': 0.06, 'scale': 0.05},
                        'proposal': 0.01,
                        'latex': r'\sum m_\nu',
                        'delta': [0.31, 0.15, 0.15]}
    params['N_eff'] = {'prior': {'limits': [0.05, 10.]},
                        'ref': {'dist': 'norm', 'loc': 3.044, 'scale': 0.05},
                        'latex': r'N_\mathrm{eff}',
                        'delta': 0.2}
    params['w0_fld'] = {'prior': {'limits': [-3., 1.]},
                        'ref': {'dist': 'norm', 'loc': -1., 'scale': 0.05},
                        'latex': r'w_{0}',
                        'delta': 0.1}
    params['wa_fld'] = {'prior': {'limits': [-3., 2.]},
                        'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.2},
                        'latex': r'w_{a}',
                        'delta': 0.2}
    params['Omega_k'] = {'prior': {'limits': [-0.3, 0.3]},
                         'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.01},
                         'latex': r'\Omega_\mathrm{k}',
                         'delta': 0.05}
    #params['zrei'] = {'latex': r'z_\mathrm{reio}'}
    #params['YHe'] = {'latex': r'Y_\mathrm{P}'}
    # FIXME
    #params['sigma8_m'] = {'derived': True, 'latex': r'\sigma_8'}
    params['age'] = {'derived': True, 'latex': r'{\rm{Age}}/\mathrm{Gyr}'}
    params['rs_drag'] = {'derived': True, 'latex': r'r_\mathrm{d}'}
    params['z_drag'] = {'derived': True, 'latex': r'z_\mathrm{d}'}
    # FIXME
    #params['H0rdrag'] = {'derived': 'lambda H0, rdrag: H0 * rs_drag',
    #                     'latex': r'H_0 r_\mathrm{d}'}
    # extra
    #extra['fluid_equation_of_state'] = 'CLP'
    #extra['use_ppf'] = 'yes'
    #extra['N_ncdm'] = 3
    extra['neutrino_hierarchy'] = 'degenerate'
    extra['m_ncdm'] = 0.
    #extra['extra_params'] = {'deg_ncdm': 3}
    # Neff = 3.044
    extra['extra_params'] = {}
    # precision settings for lensing
    if 'lensing' in dataset:
        extra['non_linear'] = 'hmcode'
        extra['ellmax_cl'] = 4000
        if 'class' in engine:
            extra['extra_params']['nonlinear_min_k_max'] = 20
            extra['extra_params']['accurate_lensing'] = 1
            extra['extra_params']['delta_l_max'] = 800
        else: # camb
            extra['extra_params']['lens_margin'] = 1250
            extra['extra_params']['lens_potential_accuracy'] = 4
            extra['extra_params']['lSampleBoost'] = 1
            extra['extra_params']['lAccuracyBoost'] = 1
    if 'mnu' not in model:
        fix('m_ncdm')
        extra['neutrino_hierarchy'] = None
    if 'nnu' not in model:
        #params.pop('N_eff')
        fix('N_eff')
    if 'omegak' not in model:
        fix('Omega_k')
    #params['Omega_Lambda'] = {'derived': True}
    if 'w' not in model:
        params.pop('w0_fld')
        params.pop('wa_fld')
        #params.pop('Omega_Lambda')
        #extra.pop('fluid_equation_of_state')
        #extra.pop('use_ppf')
    elif 'wa' not in model:
        fix('wa_fld')
    params.pop('theta_MC_100')
    params['H0'] = {'prior': {'limits': [20, 100]},
                    'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.5},
                    'latex': r'H_0',
                    'delta': 3.}
    if 'pk' in get_parameterization(dataset):
        fix('tau_reio')
        """
        params.pop('theta_MC_100')
        params['H0'] = {'prior': {'limits': [20, 100]},
                        'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
                        'latex': r'H_0',
                        'delta': 3.}
        """
    if 'cmb' not in get_parameterization(dataset): # background only
        for name in ['logA', 'n_s', 'tau_reio']: fix(name)
        for name, di in list(params.items()):  # remove derived perturb quantities
            if any(n in name for n in ['sigma8', 's8']): params.pop(name)
        """
        params.pop('theta_MC_100')
        params['H0'] = {'prior': {'limits': [20, 100]},
                        'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
                        'latex': r'H_0',
                        'delta': 3.}
        """
        if get_parameterization(dataset) == 'background':
            fix('H0')
            fix('omega_b')
            params.pop('omega_cdm')
            if 'bao' in dataset:
                raise NotImplementedError
                params['rdrag'] = {'prior': {'limits': [10., 1000.]},
                                  'ref': {'dist': 'norm', 'loc': 147.09, 'scale': 1.},
                                  'proposal': 1.,
                                  'latex': r'r_\mathrm{d}'}
            params['Omega_m'] = {'prior': {'limits': [0.01, 0.99]},
                                 'ref': {'dist': 'norm', 'loc': 0.3152, 'scale': 0.001},
                                 'proposal': 0.0005,
                                 'latex': r'\Omega_\mathrm{m}',
                                 'delta': 0.03}

    if 'bbn' in dataset:  # FIXME
        import numpy as np
        mean = np.array([0.02196, 2.944])
        cov = np.array([[4.03112260e-07, 7.30390042e-05], [7.30390042e-05, 4.52831584e-02]])
        icov = np.linalg.inv(cov)
        loc = mean[0] - icov[0, 1] / icov[0, 0] * (3.044 - mean[1])
        scale = icov[0, 0]**(-0.5)
        params['omega_b']['prior'] = {'dist': 'norm', 'loc': loc, 'scale': scale}
        #params['omega_b']['prior'] = {'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5}
        #params['omega_b']['prior'] = {'dist': 'norm', 'loc': 0.02237, 'scale': (3.025e-7)**0.5}
    if 'ns10' in dataset:  # FIXME
        params['n_s']['prior'] = {'dist': 'norm', 'loc': 0.9649, 'scale': 10 * 0.0042}
    elif 'ns' in dataset:
        params['n_s']['prior'] = {'dist': 'norm', 'loc': 0.9649, 'scale': 0.0042}
    extra['engine'] = engine
    from desilike.theories import Cosmoprimo
    cosmo = Cosmoprimo() #fiducial='DESI')
    cosmo.init.update(extra)
    cosmo.init.params = params
    return cosmo


def get_desilike_likelihoods(dataset='desi-fs-bao-all', cosmo=None, solve='marg', **kwargs):

    if cosmo is None:
        from desilike.theories import Cosmoprimo
        cosmo = Cosmoprimo(fiducial='DESI')

    import sys
    from pathlib import Path

    import desi_y1_files
    local_pythonpath = Path(desi_y1_files.__file__).parent.parent
    sys.path.insert(0, str(local_pythonpath / 'desi_y1_cosmo_bindings'))  # to add bindings_bao.py, bindings_fs_bao.py
    from bindings_bao import DESICompressedBAOLikelihood as DESICompressedBAOLikelihoodDR1

    likelihood_renames, likelihood_mapping = {}, {}
    tracers = {'all': None, 'all_nolya': ['bgs_z0', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'elg_z1', 'qso_z0'], 'bgs': ['bgs_z0'], 'lrg': ['lrg_z0', 'lrg_z1', 'lrg_z2'], 'elg': ['elg_z1'], 'qso': ['qso_z0'], 'lya': ['lya_z0']}
    ## DESI BAO
    for tracer in ['all', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrgpluselg', 'elg', 'qso', 'lya']:
        likelihood_mapping[f"desi-dr1-bao-{tracer.replace('_', '-')}"] = (DESICompressedBAOLikelihoodDR1, {'tracers': tracers.get(tracer, [tracer]), 'version': ''})
        likelihood_mapping[f"desi-dr1-v1.5-bao-{tracer.replace('_', '-')}"] = (DESICompressedBAOLikelihoodDR1, {'tracers': tracers.get(tracer, [tracer]), 'version': 'v1.5'})
    del sys.modules['bindings_bao']
    import desi_y3_files
    local_pythonpath = Path(desi_y3_files.__file__).parent.parent
    sys.path.insert(0, str(local_pythonpath / 'desi_y3_cosmo_bindings'))  # to add bindings_bao.py, bindings_fs_bao.py
    from bindings_bao import DESICompressedBAOLikelihood as DESICompressedBAOLikelihoodDR2

    likelihood_renames, likelihood_mapping = {}, {}
    tracers = {'all': None, 'all_nolya': ['bgs_z0', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'elg_z1', 'qso_z0'], 'bgs': ['bgs_z0'], 'lrg': ['lrg_z0', 'lrg_z1', 'lrg_z2'], 'elg': ['elg_z1'], 'qso': ['qso_z0'], 'lya': ['lya_z0']}
    ## DESI BAO
    for tracer in ['all', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrgpluselg', 'elg', 'qso', 'lya']:
        likelihood_mapping[f"desi-dr2-bao-{tracer.replace('_', '-')}"] = (DESICompressedBAOLikelihoodDR2, {'tracers': tracers.get(tracer, [tracer])})

    from desilike.likelihoods.bbn import Schoneberg2024BBNLikelihood
    from desilike.likelihoods.supernovae import PantheonPlusSNLikelihood, Union3SNLikelihood, DESY5SNLikelihood

    def get_desilike_sn_likelihood(Likelihood):

        def get_desilike_likelihood(**kwargs):
            likelihood = Likelihood(**kwargs)
            for param in likelihood.init.params.select(basename=['Mb', 'dM']):
                param.update(prior=None, derived='.prec')
            return likelihood

        return get_desilike_likelihood

    likelihood_mapping['schoneberg2024-bbn'] = Schoneberg2024BBNLikelihood
    likelihood_mapping['pantheonplus'] = get_desilike_sn_likelihood(PantheonPlusSNLikelihood)
    likelihood_mapping['union3'] = get_desilike_sn_likelihood(Union3SNLikelihood)
    likelihood_mapping['desy5sn'] = get_desilike_sn_likelihood(DESY5SNLikelihood)
    
    from desilike.likelihoods.cmb import (TTHighlPlanck2018PlikLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTHighlPlanck2018PlikUnbinnedLikelihood, TTTEEEHighlPlanck2018PlikLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikUnbinnedLikelihood, LensingPlanck2018ClikLikelihood, TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood, TTLowlPlanck2018Likelihood, EELowlPlanck2018Likelihood, TTTEEEHighlPlanck2018LiteLikelihood, TTTEEEHighlPlanckNPIPECamspecLikelihood, TTTEEEHighlPlanck2020HillipopLikelihood, EELowlPlanck2020LollipopLikelihood, ACTDR6LensingLikelihood)
    
    # the official 2018 clik likelihoods
    likelihood_mapping['planck2018-lowl-TT-clik'] = TTLowlPlanck2018ClikLikelihood
    likelihood_mapping['planck2018-lowl-EE-clik'] = EELowlPlanck2018ClikLikelihood
    # native python implementation
    likelihood_mapping['planck2018-lowl-TT'] = TTLowlPlanck2018Likelihood
    likelihood_mapping['planck2018-lowl-EE'] = EELowlPlanck2018Likelihood
    # plikHM high-temperature
    likelihood_mapping['planck2018-highl-plik-TT'] = TTHighlPlanck2018PlikLikelihood
    # plikHM temperature+polarization
    likelihood_mapping['planck2018-highl-plik-TTTEEE'] = TTTEEEHighlPlanck2018PlikLikelihood
    # official clik code lensing
    likelihood_mapping['planck2018-lensing-clik'] = LensingPlanck2018ClikLikelihood
    # native python implementation
    #likelihood_mapping['planck2018-lensing'] = 'planck_2018_lensing.native'
    likelihood_mapping['planck2018-highl-plik-TTTEEE-lite'] = TTTEEEHighlPlanck2018PlikLiteLikelihood
    likelihood_mapping['planck2018-highl-TTTEEE-lite'] = TTTEEEHighlPlanck2018LiteLikelihood
    #likelihood_mapping['planck-NPIPE-highl-CamSpec-TTTEEE'] = (TTTEEEHighlPlanckNPIPECamspecLikelihood, {'proj_order': 60})
    likelihood_mapping['planck-NPIPE-highl-CamSpec-TTTEEE'] = (TTTEEEHighlPlanckNPIPECamspecLikelihood, {})
    likelihood_mapping['planck2020-lollipop-lowlE'] = (EELowlPlanck2020LollipopLikelihood, {'proj_order': 60}) 
    likelihood_mapping['planck2020-hillipop-TTTEEE'] = (TTTEEEHighlPlanck2020HillipopLikelihood, {'proj_order': 60})
    likelihood_mapping['planck-act-dr6-lensing'] = (ACTDR6LensingLikelihood, {'lens_only': False, 'variant': 'actplanck_baseline'})

    if isinstance(dataset, str):
        dataset = [dataset]
    likelihoods = []
    for ds in dataset:
        if 'bbn' in ds: continue # FIXME
        if 'planck2018-ns10' in ds: continue # FIXME
        if 'planck2018-ns' in ds: continue # FIXME
        for ds in likelihood_renames.get(ds, [ds]):
            like = likelihood_mapping[ds]
            if isinstance(like, tuple):
                likelihoods.append(like[0](cosmo=cosmo, **kwargs, **like[1]))
            else:
                likelihoods.append(like(cosmo=cosmo, **kwargs))
    return likelihoods


def get_desilike_output(model='base', theory='capse', dataset='desi-bao-all', sampler='mcmc', add='', remove='', check_exists=False, base_dir=base_dir, run='auto', suffix=None, sort_dataset=sort_dataset, **kwargs):
    """
    Return desilike base output path, given input model, theory, dataset.
    See :func:`yield_configs` below to know the models and datasets than can be considered.

    Parameters
    ----------
    model : str, default='base'
        Cosmological model: 'base', 'base_w_wa', etc.

    theory : str, default='camb'
        'camb' or 'classy'.

    dataset : str, list, default='desi-bao-all'
        Datasets to consider, e.g. ['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']

    add : str, list, default=''
        In case of importance sampling, datasets (likelihoods) that are added.

    remove : str, list, default=''
        In case of importance sampling, datasets (likelihoods) that are removed.

    check_exists : bool, default=False
        If ``True``, check if file exists, if not return ``None``.

    Returns
    -------
    output : str
        Output path to desilike samples.
    """
    import glob
    from pathlib import Path

    source = None
    base_dir = Path(base_dir)
    if not isinstance(dataset, str):
        dataset = sort_dataset(dataset)
        dataset = '_'.join(dataset)
    if not isinstance(add, str):
        add = '_'.join(add)
    if not isinstance(remove, str):
        remove = '_'.join(remove)
    importance = '_' if add or remove else ''
    if add:
        importance += 'add_' + add
    if remove:
        importance += 'remove_' + remove
    main_dir = base_dir / 'fs'
    if source == 'bao':
        main_dir = base_dir / 'bao'
        source = None
    if source is None:
        source = 'main'
    main_dir = main_dir / 'desilike'
    if kwargs.get('emulator_fn', ''):
        main_dir = main_dir / 'emulated'
    
    bestfit = 'minuit' in sampler
    if sampler.endswith('-likelihood'):
        bestfit = 'likelihood'
        sampler = sampler.replace('-likelihood', '')
    if sampler.endswith('-posterior'):
        bestfit = 'posterior'
        sampler = sampler.replace('-posterior', '')

    def get_output(run):
        output = str(main_dir / f'{sampler}/{theory}/{run}/{model}/{dataset}{importance}')
        if bestfit:
            output = output + '/bestfit'
        else:
            output = output + '/chain'
        return output

    if run == 'auto':
        for run in ['run1']:
            output = get_output(run)
            if bool(glob.glob(output + '*.npy')):
                break
    else:
        output = get_output(run)

    if check_exists:
        if bool(glob.glob(output + '*.npy')):
            return output
        raise ValueError('no samples found for model = {}, dataset = {}, add = {}! (tried {})'.format(model, dataset, add, output))
    if suffix is not None:
        if bestfit:
            output = output + '.npy'
        else:
            output = output + '_{}.npy'.format(suffix)
    return output


def load_desilike_samples(sampler='mcmc', skip=0.5, thin=1, label=None, getdist=None, ichain=None, **config):
    """
    Load desilike samples.

    Parameters
    ----------
    skip : float, default=None
        Fraction of samples to skip (burnin).
        Defaults to 0.3 if not importance sampling, else 0.

    thin : int, default=1
        Thin samples by this factor (helps decorrelate samples for nicer plots).

    label : str, default=None
        Label for GetDist.

    bestfit : str, default=None
        'iminuit' to set GetDist best fit.

    **config : dict
        See :func:`get_desilike_output` for arguments.

    Returns
    -------
    samples : getdist.MCSamples
    """
    bestfit = 'minuit' in sampler
    print('Loading {}'.format(get_desilike_output(sampler=sampler, **config, suffix=None)))
    from desilike.samples import Chain, Profiles
    if bestfit:
        profiles = Profiles.load(get_desilike_output(sampler=sampler, **config, suffix=True))
        if getdist:
            return profiles.bestfit.choice(index='argmax')
        return profiles

    if getdist is None: getdist = True
    from cosmoprimo import constants
    if ichain is None: ichains = list(range(4))
    else: ichains = [ichain]
    output = [get_desilike_output(sampler=sampler, **config, suffix=ichain) for ichain in ichains]
    chain = Chain.concatenate([Chain.load(fn).remove_burnin(skip)[::thin] for fn in output])

    chain.set(chain['H0'].clone(value=chain['H0'] * chain['rs_drag'], param={'basename': 'H0rdrag', 'derived': True, 'latex': r'H_0 r_\mathrm{d}'}))
    chain.set(chain['H0rdrag'].clone(value=chain['H0rdrag'] / (constants.c * 1e-3), param={'basename': 'H0rdragc', 'derived': True, 'latex': r'H_0 r_\mathrm{d}/c'}))
    chain.set(chain['H0rdrag'].clone(value=chain['H0rdrag'] / 100, param={'basename': 'hrdrag', 'derived': True, 'latex': r'r_\mathrm{d} h'}))
    try:
        chain.set(chain['Omega_m'].clone(value=chain['sigma8_m'] * (chain['Omega_m'] / 0.3)**0.5, param={'basename': 'S8', 'derived': True, 'latex': r'S_8'}))
    except KeyError:
        pass

    if getdist:
        params = [str(param) for param in chain.params() if param.varied or (param.derived and not param.solved)]
        #params = [param for param in params if 'loglikelihood' not in param and 'logprior' not in param and 'logposterior' not in param]
        samples = chain.to_getdist(params=params, label=label)
        #add_derived_getdist(samples)
        chain = samples

    return chain


def emulate_desilike(model='base', theory='class', dataset='desi-fs-bao-all', get_desilike_likelihoods=get_desilike_likelihoods, get_desilike_cosmo=get_desilike_cosmo, **kwargs):

    cosmo = get_desilike_cosmo(model=model, engine=theory, dataset=dataset)
    get_desilike_likelihoods(dataset=dataset, save_emulator=True, emulator_fn=model, cosmo=cosmo, **kwargs)


def measure_speed(model='base', theory='capse', dataset='desi-fs-bao-all', niterations=5, get_desilike_likelihoods=get_desilike_likelihoods, get_desilike_cosmo=get_desilike_cosmo, **kwargs):
    import time
    import numpy as np

    from pyrecon.utils import MemoryMonitor
    with MemoryMonitor() as mem:
        cosmo = get_desilike_cosmo(model=model, engine=theory, dataset=dataset)
        likelihoods = get_desilike_likelihoods(dataset=dataset, cosmo=cosmo, **kwargs)
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


def sample_desilike(model='base', theory='capse', dataset='desi-bao-all', get_desilike_likelihoods=get_desilike_likelihoods, test=False, resume=False, sampler='mcmc', get_parameterization=get_parameterization, get_desilike_output=get_desilike_output, get_desilike_cosmo=get_desilike_cosmo, run='run0', **kwargs):
    import os
    from desilike.samplers import MCMCSampler, NUTSSampler, HMCSampler, ZeusSampler, PocoMCSampler
    from desilike.samples import Chain

    cosmo = get_desilike_cosmo(model=model, engine=theory, dataset=dataset)
    likelihoods = get_desilike_likelihoods(dataset=dataset, cosmo=cosmo, **kwargs)
    likelihood = sum(likelihoods)
    #likelihood = likelihoods[0]
    likelihood()

    chains = cosmo.mpicomm.size
    resume = False
    chains = 4

    output = [get_desilike_output(model=model, theory=theory, dataset=dataset, sampler=sampler, suffix=ichain, run=run, **kwargs) for ichain in range(chains)]
    save_fn = output

    base_save_fn = save_fn[0][:-len('_0.npy')]
    if resume and all(os.path.isfile(fi) for fi in save_fn):
        chains = save_fn
    elif sampler == 'mcmc':
        #covariance = [get_desilike_output(model=model, theory=theory, dataset=['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10'], sampler='nuts', suffix=ichain, run='run2', **{**kwargs, 'emulator_fn': model}) for ichain in range(chains)]
        covariance = [get_desilike_output(model=model, theory=theory, dataset=dataset, sampler='mcmc', suffix=ichain, run=run) for ichain in range(chains)]
        covariance = Chain.concatenate([Chain.load(fn).remove_burnin(0.5) for fn in covariance])
        #covariance = None
        sampler = MCMCSampler(likelihood, chains=chains, covariance=covariance, oversample_power=0.4, drag=False, learn={'max_eigen_gr': 10., 'every': '40 * ndim'}, seed=42, save_fn=save_fn)
    elif sampler == 'nuts':
        covariance = [get_desilike_output(model=model, theory=theory, dataset=dataset, sampler='mcmc', suffix=ichain, run=run) for ichain in range(chains)]
        covariance = Chain.concatenate([Chain.load(fn).remove_burnin(0.5) for fn in covariance])
        sampler = NUTSSampler(likelihood, chains=chains, seed=42, save_fn=save_fn, covariance=covariance, adaptation=False)
    elif sampler == 'hmc':
        sampler = HMCSampler(likelihood, chains=chains, seed=42, save_fn=save_fn, num_integration_steps=10, step_size=0.03)
    elif sampler == 'zeus':
        sampler = ZeusSampler(likelihood, chains=chains, seed=42, save_fn=save_fn)
    elif sampler == 'pocomc':
        sampler = PocoMCSampler(likelihood, chains=chains, seed=42, save_fn=save_fn)
    if test:
        #chains = sampler.run(check={'max_eigen_gr': 0.03}, check_every=40 * len(likelihood.varied_params), max_iterations=10)
        chains = sampler.run(max_iterations=10, check={'max_eigen_gr': 0.02, 'min_ess': 100})
        return
    #chains = sampler.run(check={'max_eigen_gr': 0.03}, check_every=40 * len(likelihood.varied_params))
    is_cmb = 'cmb' in get_parameterization(dataset)
    #chains = sampler.run(min_iterations=1000, check={'max_eigen_gr': 0.02, 'max_cl_diag_gr': 0.2 if is_cmb else 0.02, 'min_ess': 300})
    chains = sampler.run(min_iterations=1000, check={'max_eigen_gr': 0.01, 'max_cl_diag_gr': 0.2 if is_cmb else 0.02, 'min_ess': 100}, check_every=50)
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
        try: print_margestats(base_save_fn, fn=True)
        except: pass


def profile_desilike(model='base', theory='capse', dataset='desi-fs-bao-all', start=None, get_desilike_likelihoods=get_desilike_likelihoods, get_desilike_cosmo=get_desilike_cosmo, get_desilike_output=get_desilike_output, run='run0', profile_params=None, test=False, **kwargs):
    import os
    from desilike.profilers import MinuitProfiler

    cosmo = get_desilike_cosmo(model=model, engine=theory, dataset=dataset)
    likelihoods = get_desilike_likelihoods(dataset=dataset, cosmo=cosmo, solve='.best', **kwargs)
    likelihood = sum(likelihoods)
    #likelihood = likelihoods[0]

    if start is not None:
        for name in start:
            param = likelihood.all_params[name]
            if param.varied:
                param.update(ref={'dist': 'norm', 'loc': start[name], 'scale': 1e-2 * param.proposal})
                print(param, param.ref)
    output = get_desilike_output(model=model, theory=theory, dataset=dataset, sampler='minuit', suffix=True, run=run, **kwargs)
    #for param in likelihood.all_params.select(basename=['Omega_cdm', 'Omega_b']): param.update(fixed=True)
    profiler = MinuitProfiler(likelihood, save_fn=output, seed=42)
    profiles = profiler.maximize(niterations=max(likelihood.mpicomm.size, 4))
    print(profiles.to_stats(tablefmt='pretty'))
    #profiler.interval(params=cosmo.varied_params)
    if profile_params is not None:
        profiler.profile(params=profile_params, size=30, cl=2.5)
        for cl in [1, 2]: profiler.contour(params=profile_params, cl=cl, size=40)
    #profiler.interval(['Mb'])
    for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
        profiles.to_stats(tablefmt=tablefmt, fn=os.path.splitext(output)[0] + '_stats.' + ext)