"""
Run cosmological inference with Y1 FS.
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

See ``y1_fs_cosmo.py`` for the script running chains with :mod:`desipipe`.
"""

from pathlib import Path


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


base_dir = Path('/global/cfs/cdirs/desicollab/users/hernannb/__FOLPS_tutorial/part3/chains')
#base_dir = Path('/global/cfs/cdirs/desicollab/science/cpe/y1kp7')
covmat_dir = Path('/global/cfs/cdirs/desicollab/science/cpe/y3_bao_cosmo/bao_v1p1') # Use covariances from previous Y3 fits (no need to use Y1 anymore)


def get_cobaya_likelihoods(dataset):
    """Return cobaya likelihood given input dataset (as list of strings, e.g. ['desi-bao-lrg', 'pantheonplus'])."""
    from pathlib import Path
    import desi_y1_files
    local_pythonpath = Path(desi_y1_files.__file__).parent.parent
    local_path = str(local_pythonpath / 'desi_y1_cosmo_bindings' / 'cobaya_likelihoods')
    local_pythonpath = str(local_pythonpath)

    likelihood_renames, likelihood_mapping = {}, {}
    
    ## DESI BAO
    for tracer in ['all', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrgpluselg', 'elg', 'qso', 'lya']:
        likelihood_mapping[f"desi-bao-{tracer.replace('_', '-')}"] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.bao_likelihoods.desi_bao_{tracer}': {'python_path': local_pythonpath, 'path': local_path}}
    ## v1.5
    ## DESI BAO
    for tracer in ['all', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'lrgpluselg', 'elg', 'qso', 'lya']:
        likelihood_mapping[f"desi-v1.5-bao-{tracer.replace('_', '-')}"] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.bao_likelihoods_v1_5.desi_bao_{tracer}': {'python_path': local_pythonpath, 'path': local_path}}

    likelihood_renames['desi-v1.5-bao-all-nolya'] = ['desi-v1.5-bao-bgs', 'desi-v1.5-bao-lrg-z0', 'desi-v1.5-bao-lrg-z1', 'desi-v1.5-bao-lrg-z2', 'desi-v1.5-bao-elg', 'desi-v1.5-bao-qso']

    ## DESI FS
    for observable in ['fs', 'fs_bao']:
        for data in ['mock-cmblens-', '']:
            for tracer in ['all', 'all_nolya', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'elg', 'qso', 'lya']:
                theory = 'reptvelocileptors'
                likelihood_mapping[f"{data}desi-{theory}-{observable}-{tracer}".replace('_', '-')] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods.desi_{observable}_{tracer}': {'python_path': local_pythonpath, 'data_name': data[:-1]}}
                likelihood_mapping[f"{data}desi-{theory}-prior3-{observable}-{tracer}".replace('_', '-')] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods.desi_{observable}_{tracer}': {'python_path': local_pythonpath, 'data_name': data[:-1], 'scale_eft_priors': 3.}}

    for observable in ['fs', 'fs_bao']:
        for data in ['']:
            for tracer in ['all', 'all_nolya', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'elg', 'qso', 'lya']:
                for theory in ['folpsax', 'folpsv2']:
                    # For now we only support the "all" tracer combination
                    # via dedicated Cobaya bindings that fix the PT theory
                    # internally (desi_fs_bao_all_folpsax / folpsv2).
                    if tracer == 'all' and observable == 'fs_bao':
                        if theory == 'folpsax':
                            module_name = 'desi_y1_cosmo_bindings.cobaya_bindings.desi_fs_bao_all_folpsax'
                        else:  # folpsv2
                            module_name = 'desi_y1_cosmo_bindings.cobaya_bindings.desi_fs_bao_all_folpsv2'
                        likelihood_mapping[f"{data}desi-{theory}-{observable}-{tracer}".replace('_', '-')] = {module_name: {'python_path': local_pythonpath, 'stop_at_error': False}}
                    # Other tracer/theory combinations can be added here with
                    # additional bindings if needed.

    for observable in ['shapefit', 'shapefit_bao', 'shapefit_joint', 'shapefit_nodm_bao', 'shapefit_nodf_bao']:
        for tracer in ['all', 'all_nolya', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'elg', 'qso', 'lya']:
            likelihood_mapping[f"{data}desi-{observable}-{tracer}".replace('_', '-')] = {f'desi_y1_cosmo_bindings.cobaya_bindings.desi_{observable}_{tracer}': {'python_path': local_pythonpath, 'stop_at_error': False}}

    ## SDSS BAO
    likelihood_mapping['sdss-bao-dr7-mgs'] = 'bao.sdss_dr7_mgs'
    likelihood_mapping['sdss-bao-dr12-lrg'] = 'bao.sdss_dr12_lrg_bao_dmdh'
    likelihood_mapping['sdss-bao-dr16-lrg'] = 'bao.sdss_dr16_lrg_bao_dmdh'

    ## DESI x eBOSS Lya BAO
    tracer = 'lya'
    likelihood_mapping[f'desi-eboss-bao-{tracer}'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.bao_likelihoods.desi_eboss_bao_{tracer}': {'python_path': local_pythonpath, 'path': local_path}}
    
    ## Best BAO
    likelihood_renames['desi-sdss-bao-best'] = ['sdss-bao-dr7-mgs', 'sdss-bao-dr12-lrg', 'desi-bao-lrg-z1', 'desi-bao-lrgpluselg', 'desi-bao-elg', 'desi-bao-qso', 'desi-eboss-bao-lya']

    ## DES-Y3
    likelihood_mapping['desy3shear'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.desy3_likelihoods.shear': {'python_path': local_pythonpath, 'path': str(Path(local_path) / 'desy3_data'), 'use_Weyl': True, 'speed': 1}}
    likelihood_mapping['desy3joint'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.desy3_likelihoods.joint': {'python_path': local_pythonpath, 'path': str(Path(local_path) / 'desy3_data'), 'use_Weyl': True, 'speed': 1}}
    
    ## BBN-Schoneberg
    likelihood_mapping['schoneberg2024-bbn'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.bbn_likelihoods.schoneberg2024': {'python_path': local_pythonpath}}
    likelihood_mapping['schoneberg2024-bbn-fixed-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.bbn_likelihoods.schoneberg2024_fixed_nnu': {'python_path': local_pythonpath}}

    ## Planck2018 CMB priors on theta_star and rdrag
    likelihood_mapping['planck2018-thetastar-fixed-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.thetastar_planck2018_fixed_nnu': {'python_path': local_pythonpath}}
    likelihood_mapping['planck2018-thetastar-varied-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.thetastar_planck2018_varied_nnu': {'python_path': local_pythonpath}}
    likelihood_mapping['planck2018-thetastar-fixed-marg-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.thetastar_planck2018_fixed_marg_nnu': {'python_path': local_pythonpath}}
    likelihood_mapping['planck2018-rdrag-fixed-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.rdrag_planck2018_fixed_nnu': {'python_path': local_pythonpath}}
    #likelihood_mapping['planck2018-rdrag-shifted'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.rdrag_planck2018_shifted': {'python_path': local_pythonpath}}

    ## Planck2018 CMB priors on ns
    likelihood_mapping['planck2018-ns10'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.ns10_planck2018': {'python_path': local_pythonpath}}
    likelihood_mapping['planck2018-ns'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.ns_planck2018': {'python_path': local_pythonpath}}
    ## SN
    likelihood_mapping['pantheon'] = 'sn.pantheon'
    #likelihood_mapping['pantheonplus'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.sn_likelihoods.pantheonplus': {'python_path': local_pythonpath}}
    #likelihood_mapping['union3'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.sn_likelihoods.union3': {'python_path': local_pythonpath}}
    #likelihood_mapping['desy5sn'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.sn_likelihoods.desy5': {'python_path': local_pythonpath}}
    """
    likelihood_mapping['pantheonplus'] = 'sn.pantheonplus'
    likelihood_mapping['union3'] = 'sn.union3'
    likelihood_mapping['desy5sn'] = 'sn.desy5'
    """
    likelihood_mapping['pantheonplus'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.sn_likelihoods_internal.pantheonplus': {'python_path': local_pythonpath}}
    likelihood_mapping['union3'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.sn_likelihoods_internal.union3': {'python_path': local_pythonpath}}
    likelihood_mapping['desy5sn'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.sn_likelihoods_internal.desy5': {'python_path': local_pythonpath}}
    
    likelihood_mapping['pantheonplus-zmin0.1'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.sn_likelihoods_internal.pantheonplus': {'python_path': local_pythonpath, 'zmin': 0.1}}
    likelihood_mapping['union3-zmin0.1'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.sn_likelihoods_internal.union3': {'python_path': local_pythonpath, 'zmin': 0.1, 'stop_at_error': True}}
    likelihood_mapping['desy5sn-zmin0.1'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.sn_likelihoods_internal.desy5': {'python_path': local_pythonpath, 'zmin': 0.1}}
    #likelihood_mapping['pantheonplus'] = {'desi_y1_cosmo_bindings.cobaya_bindings.pantheonplus': {'stop_at_error': False, 'python_path': local_pythonpath}}
    #likelihood_mapping['union3'] = {'desi_y1_cosmo_bindings.cobaya_bindings.union3': {'stop_at_error': False, 'python_path': local_pythonpath}}
    #likelihood_mapping['desy5sn'] = {'desi_y1_cosmo_bindings.cobaya_bindings.desy5sn': {'stop_at_error': False, 'python_path': local_pythonpath}}

    ## CMB (Planck 2018)
    # the official 2018 clik likelihoods
    likelihood_mapping['planck2018-lowl-TT-clik'] = 'planck_2018_lowl.TT_clik'
    likelihood_mapping['planck2018-lowl-EE-clik'] = 'planck_2018_lowl.EE_clik'
    # native python implementation
    likelihood_mapping['planck2018-lowl-TT'] = 'planck_2018_lowl.TT'
    likelihood_mapping['planck2018-lowl-EE'] = 'planck_2018_lowl.EE'
    # plikHM high-temperature
    likelihood_mapping['planck2018-highl-plik-TT'] = 'planck_2018_highl_plik.TT'
    # plikHM temperature+polarization
    likelihood_mapping['planck2018-highl-plik-TTTEEE'] = 'planck_2018_highl_plik.TTTEEE'
    # planck 2018 CamSpec likelihoods
    likelihood_mapping['planck2018-highl-CamSpec-TT-clik'] = 'planck_2018_highl_CamSpec.TT'
    likelihood_mapping['planck2018-highl-CamSpec-TTTEEE-clik'] = 'planck_2018_highl_CamSpec.TTTEEE'
    # native python implementation - planck 2018 CamSpec likelihoods
    likelihood_mapping['planck2018-highl-CamSpec-TT'] = 'planck_2018_highl_CamSpec.TT_native'
    likelihood_mapping['planck2018-highl-CamSpec-TTTEEE'] = 'planck_2018_highl_CamSpec.TTTEEE_native'
    # official clik code lensing
    likelihood_mapping['planck2018-lensing-clik'] = 'planck_2018_lensing.clik'
    # native python implementation
    likelihood_mapping['planck2018-lensing'] = 'planck_2018_lensing.native'
    
    likelihood_mapping['planck2018-highl-plik-TTTEEE-lite'] = 'planck_2018_highl_plik.TTTEEE_lite'

    ## More recent Planck likelihoods
    # native Python versions of high-CamSpec likelihoods
    likelihood_mapping['planck2018-highl-CamSpec2021-TT'] = 'planck_2018_highl_CamSpec2021.TT'
    likelihood_mapping['planck2018-highl-CamSpec2021-TTTEEE'] = 'planck_2018_highl_CamSpec2021.TTTEEE'
    # latest native python NPIPE (PR4) CamSpec high-ell likelihoods
    likelihood_mapping['planck-NPIPE-highl-CamSpec-TT'] = 'planck_NPIPE_highl_CamSpec.TT'
    likelihood_mapping['planck-NPIPE-highl-CamSpec-TE'] = 'planck_NPIPE_highl_CamSpec.TE'
    likelihood_mapping['planck-NPIPE-highl-CamSpec-EE'] = 'planck_NPIPE_highl_CamSpec.EE'
    likelihood_mapping['planck-NPIPE-highl-CamSpec-TTTEEE'] = 'planck_NPIPE_highl_CamSpec.TTTEEE'
    # NPIPE lensing
    likelihood_mapping['planckpr4lensing'] = 'planckpr4lensing.PlanckPR4Lensing'
    likelihood_mapping['planckpr4lensingmarged'] = 'planckpr4lensing.PlanckPR4LensingMarged'
    # lollipop / hillipop
    likelihood_mapping['planck2020-lollipop-lowlE'] = 'planck_2020_lollipop.lowlE'
    likelihood_mapping['planck2020-lollipop-lowlB'] = 'planck_2020_lollipop.lowlB'
    likelihood_mapping['planck2020-lollipop-lowlEB'] = 'planck_2020_lollipop.lowlEB'
    likelihood_mapping['planck2020-hillipop-TT'] = 'planck_2020_hillipop.TT'
    likelihood_mapping['planck2020-hillipop-TE'] = 'planck_2020_hillipop.TE'
    likelihood_mapping['planck2020-hillipop-EE'] = 'planck_2020_hillipop.EE'
    likelihood_mapping['planck2020-hillipop-TTTEEE'] = 'planck_2020_hillipop.TTTEEE'
    ## ACT DR6 lensing likelihood
    likelihood_mapping['act-dr6-lensing'] = {'act_dr6_lenslike.ACTDR6LensLike': {'lens_only': False, 'variant': 'act_baseline', 'lmax': 4000, 'version': 'v1.2'}}
    likelihood_mapping['planck-act-dr6-lensing'] = {'act_dr6_lenslike.ACTDR6LensLike': {'lens_only': False, 'variant': 'actplanck_baseline', 'lmax': 4000, 'version': 'v1.2'}}
    # Add more mappings here
    
    if isinstance(dataset, str):
        dataset = [dataset]
    likelihoods = {}
    for ds in dataset:
        for ds in likelihood_renames.get(ds, [ds]):
            like = likelihood_mapping[ds]
            if isinstance(like, dict):  # e.g. act-dr6
                likelihoods.update(like)
            elif like:
                likelihoods[like] = None
    if 'planck-act-dr6-lensing' in dataset and 'tt' not in '_'.join(dataset).lower():
        likelihoods['act_dr6_lenslike.ACTDR6LensLike'].update(lens_only=True)
    return likelihoods


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


def get_cobaya_params(model='base', theory='camb', dataset='desi-synthetic-fs-bao-all', get_parameterization=get_parameterization):
    """Return cobaya ``params`` dictionary, given input model, theory and dataset."""
    params, extra = {}, {}
    if not isinstance(dataset, str):
        dataset = '_'.join(dataset)
    
    def fix(name):
        di = dict(params[name])
        di['value'] = params[name]['ref']['loc']
        di.pop('prior', None)
        di.pop('ref', None)
        di.pop('proposal', None)
        params[name] = di

    if 'classy' in theory:
        # First define all parameters (even if redundant), then select
        params['logA'] = {'prior': {'min': 1.61, 'max': 3.91},
                          'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001},
                          'proposal': 0.001,
                          'latex': r'\ln(10^{10} A_\mathrm{s})',
                          'drop': True}
        params['A_s'] = {'value': 'lambda logA: 1e-10*np.exp(logA)',
                        'latex': r'A_\mathrm{s}'}
        params['n_s'] = {'prior': {'min': 0.8, 'max': 1.2}, 
                        'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004},
                        'proposal': 0.002,
                        'latex': r'n_\mathrm{s}'}
        params['theta_s_100'] = {'prior': {'min': 0.5, 'max': 10.},
                                  'ref': {'dist': 'norm', 'loc': 1.04109, 'scale': 0.0004},
                                  'proposal': 0.0002, 'latex': r'100\theta_\mathrm{s}',
                                  'renames': 'theta'}
        params['H0'] = {'latex': 'H_0'}
        params['omega_b'] = {'prior': {'min': 0.005, 'max': 0.1},
                            'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001},
                            'proposal': 0.0001,
                            'latex': r'\Omega_\mathrm{b} h^2'}
        params['omega_cdm'] = {'prior': {'min': 0.001, 'max': 0.99},
                               'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001},
                               'proposal': 0.0005,
                               'latex': r'\Omega_\mathrm{c} h^2'}
        params['Omega_m'] = {'latex': r'\Omega_\mathrm{m}'}
        params['omega_m'] = {'derived': 'lambda Omega_m, H0: Omega_m * (H0/100)**2',
                             'latex': r'\Omega_\mathrm{m} h^2'}
        params['tau_reio'] = {'prior': {'min': 0.01, 'max': 0.8},
                              'ref': {'dist': 'norm', 'loc': 0.0544, 'scale': 0.006},
                              'proposal': 0.003,
                              'latex': r'\tau_\mathrm{reio}'}
        params['m_ncdm'] = {'prior': {'min': 0., 'max': 5.},
                            'ref': {'dist': 'norm', 'loc': 0.06, 'scale': 0.05},
                            'proposal': 0.01,
                            'latex': r'\sum m_\nu',
                            'renames': 'mnu'}
        params['N_eff'] = {'prior': {'min': 0.05, 'max': 10.},
                            'ref': {'dist': 'norm', 'loc': 3.044, 'scale': 0.05},
                            'proposal': 0.05,
                            'latex': r'N_\mathrm{eff}',
                            'renames': 'nnu'}
        params['w0_fld'] = {'prior': {'min': -3., 'max': 1.},
                            'ref': {'dist': 'norm', 'loc': -1., 'scale': 0.02},
                            'proposal': 0.02,
                            'latex': r'w_{0}'}
        params['wa_fld'] = {'prior': {'min': -3., 'max': 2.},
                            'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.05},
                            'proposal': 0.05,
                            'latex': r'w_{a}'}
        params['Omega_k'] = {'prior': {'min': -0.3, 'max': 0.3},
                             'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.01},
                             'proposal': 0.01,
                             'latex': r'\Omega_\mathrm{k}'}
        params['zrei'] = {'latex': r'z_\mathrm{reio}'}
        params['YHe'] = {'latex': r'Y_\mathrm{P}'}
        params['sigma8'] = {'latex': r'\sigma_8'}
        params['s8h5'] = {'derived': 'lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)',
                          'latex': r'\sigma_8/h^{0.5}'}
        params['s8omegamp5'] = {'derived': 'lambda sigma8, omegam: sigma8*omegam**0.5',
                                'latex': r'\sigma_8 \Omega_\mathrm{m}^{0.5}'}
        params['s8omegamp25'] = {'derived': 'lambda sigma8, omegam: sigma8*omegam**0.25',
                                 'latex': r'\sigma_8 \Omega_\mathrm{m}^{0.25}'}
        params['A'] = {'derived': 'lambda As: 1e9*As', 'latex': r'10^9 A_\mathrm{s}'}
        params['clamp'] = {'derived': 'lambda As, tau: 1e9*As*np.exp(-2*tau)',
                           'latex': r'10^9 A_\mathrm{s} e^{-2\tau}'}
        params['age'] = {'latex': r'{\rm{Age}}/\mathrm{Gyr}'}
        params['rs_drag'] = {'latex': r'r_\mathrm{d}', 'renames': 'rdrag'}
        params['z_d'] = {'latex': r'z_\mathrm{d}', 'renames': 'zdrag'}
        params['H0rdrag'] = {'derived': 'lambda H0, rdrag: H0 * rdrag',
                             'latex': r'H_0 r_\mathrm{d}'}
        # extra
        extra['fluid_equation_of_state'] = 'CLP'
        extra['use_ppf'] = 'yes'
        extra['N_ncdm'] = 1
        extra['deg_ncdm'] = 3
        # Neff = 3.044
        extra['N_ur'] = 0.00441
        # precision settings for lensing
        if 'lensing' in dataset:
            extra['non linear'] = 'hmcode'
            extra['nonlinear_min_k_max'] = 20
            extra['accurate_lensing'] = 1
            extra['delta_l_max'] = 800
        if 'mnu' not in model:
            fix('m_ncdm')
            extra['deg_ncdm'] = 1
            extra['N_ur'] = 2.0308
        if 'nnu' in model:
            extra.pop('N_ur')
        else:
            params.pop('N_eff')
        if 'omegak' not in model:
            fix('Omega_k')
        params['Omega_Lambda'] = 0.
        if 'w' not in model:
            params.pop('w0_fld')
            params.pop('wa_fld')
            params.pop('Omega_Lambda')
            extra.pop('fluid_equation_of_state')
            extra.pop('use_ppf')
        elif 'wa' not in model:
            fix('wa_fld')
        #if 'bbn' in dataset:
        #    params['omega_b']['prior'] = {'loc': 0.02218, 'scale': 0.00055}
        if 'pk' in get_parameterization(dataset):
            fix('tau_reio')
            params.pop('theta_s_100')
            params['H0'] = {'prior': {'min': 20, 'max': 100},
                            'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
                            'latex': r'H_0'}
        if 'cmb' not in get_parameterization(dataset): # background only
            for name in ['logA', 'n_s', 'tau_reio']: fix(name)
            for name, di in list(params.items()):  # remove derived perturb quantities
                if any(n in name for n in ['sigma8', 's8']): params.pop(name)
            params.pop('theta_s_100')
            params.pop('omega_cdm')
            params['H0'] = {'prior': {'min': 20, 'max': 100},
                            'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
                            'latex': r'H_0'}
            if get_parameterization(dataset) == 'background':
                fix('H0')
                fix('omega_b')
                if 'bao' in dataset:
                    params['rdrag'] = {'prior': {'min': 10., 'max': 1000.},
                                      'ref': {'dist': 'norm', 'loc': 147.09, 'scale': 1.},
                                      'proposal': 1.,
                                      'latex': r'r_\mathrm{d}'}
            params['Omega_m'] = {'prior': {'min': 0.01, 'max': 0.99},
                                 'ref': {'dist': 'norm', 'loc': 0.3152, 'scale': 0.001},
                                 'proposal': 0.0005,
                                 'latex': r'\Omega_\mathrm{m}'}
    
    else:  # camb or isitgr

        params['logA'] = {'prior': {'min': 1.61, 'max': 3.91},
                          'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001},
                          'proposal': 0.001,
                          'latex': r'\ln(10^{10} A_\mathrm{s})',
                          'drop': True}
        params['As'] = {'value': 'lambda logA: 1e-10*np.exp(logA)',
                        'latex': r'A_\mathrm{s}'}
        params['ns'] = {'prior': {'min': 0.8, 'max': 1.2}, 
                        'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004},
                        'proposal': 0.002,
                        'latex': r'n_\mathrm{s}'}
        params['theta_MC_100'] = {'prior': {'min': 0.5, 'max': 10.},
                                  'ref': {'dist': 'norm', 'loc': 1.04109, 'scale': 0.0004},
                                  'proposal': 0.0002, 'latex': r'100\theta_\mathrm{MC}',
                                  'drop': True,
                                  'renames': 'theta'}
        params['cosmomc_theta'] = {'value': 'lambda theta_MC_100: 1.e-2*theta_MC_100',
                                   'derived': False}
        params['H0'] = {'latex': r'H_0'}
        params['ombh2'] = {'prior': {'min': 0.005, 'max': 0.1},
                           'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001},
                           'proposal': 0.0001,
                           'latex': r'\Omega_\mathrm{b} h^2'}
        params['omch2'] = {'prior': {'min': 0.001, 'max': 0.99},
                           'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001},
                           'proposal': 0.0005,
                           'latex': r'\Omega_\mathrm{c} h^2'}
        params['tau'] = {'prior': {'min': 0.01, 'max': 0.8},
                         'ref': {'dist': 'norm', 'loc': 0.0544, 'scale': 0.006},
                         'proposal': 0.003,
                         'latex': r'\tau_\mathrm{reio}'}
        params['mnu'] = {'prior': {'min': 0., 'max': 5.},
                         'ref': {'dist': 'norm', 'loc': 0.06, 'scale': 0.05},
                         'proposal': 0.01,
                         'latex': r'\sum m_\nu'}
        if 'mnu059' in model or 'mnu_nhcamb' in model:
            params['mnu']['prior']['min'] = 0.059
            #params['mnu']['prior']['min'] = 0.05885
        if 'mnu100' in model:
            params['mnu']['prior']['min'] = 0.100
            #params['mnu']['prior']['min'] = 0.0995
        if  'mnu_ihcamb' in model:
            params['mnu']['prior']['min'] = 0.1003  # because of +1e-6 in https://github.com/cmbant/CAMB/blob/9ae4620a31475e863e8f35620a0ec729c745483f/fortran/model.f90#L285, mnu min is 0.10026808450517906 (otherwise https://github.com/cmbant/CAMB/blob/9ae4620a31475e863e8f35620a0ec729c745483f/fortran/MathUtils.f90#L275)
        if 'mnu_nh3' in model or 'mnu_ih3' in model:
            # https://arxiv.org/pdf/2203.14247.pdf
            # NuFit 5.1
            deltamnu21sq = (7.42e-5, 0.21e-5)
            deltamnu31sq = (2.510e-3, 0.027e-3)
            deltamnu32sq = (-2.490e-3, 0.027e-3)

            # in camb, PDG 2015
            # https://github.com/cmbant/CAMB/blob/9ae4620a31475e863e8f35620a0ec729c745483f/fortran/constants.f90#L73
            #deltamnu21sq = (7.54e-5, 0.)
            #deltamnu31sq = (2.46e-3, 0.)
            #deltamnu32sq = (- deltamnu31sq[0] - deltamnu21sq[0], 0.)

            params['mnul'] = params.pop('mnu')
            params['mnul']['latex'] = r'm_{\nu, L}'
            params['mnul']['drop'] = True
            dmnu2_fixed = 'mnu_nh3_dmnu2fixed' in model or 'mnu_ih3_dmnu2fixed' in model
            params['deltamnu21sq'] = {'prior': {'dist': 'norm', 'loc': deltamnu21sq[0], 'scale': deltamnu21sq[1]}, 'drop': True}
            #params['deltamnu21sq']['ref'] = params['deltamnu21sq']['prior']
            if dmnu2_fixed:
                params['deltamnu21sq']['value'] = params['deltamnu21sq'].pop('prior')['loc']
            if 'nh3' in model:
                # m1 < m2 << m3
                params['deltamnu31sq'] = {'prior': {'dist': 'norm', 'loc': deltamnu31sq[0], 'scale': deltamnu31sq[1]}, 'drop': True}
                params['deltamnu31sq']['ref'] = params['deltamnu31sq']['prior']
                value = 'lambda mnul, deltamnu21sq, deltamnu31sq: [mnul, (mnul**2 + deltamnu21sq)**0.5, (mnul**2 + deltamnu31sq)**0.5]'
                nunames = [1, 2, 3]
                if dmnu2_fixed:
                    params['deltamnu31sq']['value'] = params['deltamnu31sq'].pop('prior')['loc']
            elif 'ih3' in model:
                # m3 << m1 < m2
                params['deltamnu32sq'] = {'prior': {'dist': 'norm', 'loc': deltamnu32sq[0], 'scale': deltamnu32sq[1]}, 'drop': True}
                params['deltamnu32sq']['ref'] = params['deltamnu32sq']['prior']
                value = 'lambda mnul, deltamnu21sq, deltamnu32sq: [mnul, (mnul**2 - deltamnu32sq - deltamnu21sq)**0.5, (mnul**2 - deltamnu32sq)**0.5]'
                nunames = [3, 1, 2]
                if dmnu2_fixed:
                    params['deltamnu32sq']['value'] = params['deltamnu32sq'].pop('prior')['loc']

            params['mnulist'] = {'value': value, 'latex': r'\sum m_\nu', 'drop': True, 'derived': False}
            params['mnu'] = {'value': 'lambda mnulist: sum(mnulist)', 'latex': r'\sum m_\nu'}
            params['nu_mass_fractions'] = {'value': 'lambda mnulist: list(mnulist / np.sum(mnulist))', 'derived': False}
            #params['nu_mass_fractions'] = {'value': 'lambda mnulist: [0.23508277, 0.76491723]', 'derived': False}  # TOREMOVE
            #params['nu_mass_fractions'] = {'value': 'lambda mnulist: [0.23508277 - 0.1, 0.1, 0.76491723]', 'derived': False}  # TOREMOVE
            
            for imu in range(1, 1 + len(nunames)):
                iname = nunames.index(imu)
                params['mnu{:d}'.format(imu)] = {'derived': 'lambda mnulist: mnulist[{:d}]'.format(iname), 'latex': r'm_{{\nu, {:d}}}'.format(imu), 'drop': True}

            params['num_nu_massless'] = {'value': 'lambda nnu: nnu - 3.044', 'derived': False}
            params['nu_mass_degeneracies'] = {'value': 'lambda nnu: [3.044 / {0:d}] * {0:d}'.format(len(nunames)), 'derived': False}

            #params['nu_mass_degeneracies'] = {'value': 'lambda nnu: [2 * 3.044 / 3, 3.044 / 3]', 'derived': False}
            #params['num_nu_massless'] = {'value': 'lambda: 0.00439674503298604', 'derived': False}
            #params['nu_mass_degeneracies'] = {'value': 'lambda nnu: [1.01320163, 1.01320163, 1.01320163]', 'derived': False}

            
        if 'mnu_nh2' in model or 'mnu_ih2' in model:
            # https://arxiv.org/pdf/2203.14247.pdf
            # NuFit 5.1
            #deltamnu21sq = (7.42e-5, 0.21e-5)
            #deltamnu31sq = (2.510e-3, 0.027e-3)
            #deltamnu32sq = (-2.490e-3, 0.027e-3)

            # in camb, PDG 2015
            # https://github.com/cmbant/CAMB/blob/9ae4620a31475e863e8f35620a0ec729c745483f/fortran/constants.f90#L73
            deltamnu21sq = (7.54e-5, 0.)
            deltamnu31sq = (2.46e-3, 0.)
            deltamnu32sq = (- deltamnu31sq[0] - deltamnu21sq[0], 0.)

            params['mnul'] = params.pop('mnu')
            params['mnul']['latex'] = r'm_{\nu, L}'
            params['mnul']['drop'] = True
            dmnu2_fixed = 'mnu_nh2_dmnu2fixed' in model or 'mnu_ih2_dmnu2fixed' in model
            params['deltamnu21sq'] = {'prior': {'dist': 'norm', 'loc': deltamnu21sq[0], 'scale': deltamnu21sq[1]}, 'drop': True}
            #params['deltamnu21sq']['ref'] = params['deltamnu21sq']['prior']
            if dmnu2_fixed:
                params['deltamnu21sq']['value'] = params['deltamnu21sq'].pop('prior')['loc']
            if 'nh2' in model:
                # m1 < m2 << m3
                params['deltamnu31sq'] = {'prior': {'dist': 'norm', 'loc': deltamnu31sq[0], 'scale': deltamnu31sq[1]}, 'drop': True}
                params['deltamnu31sq']['ref'] = params['deltamnu31sq']['prior']
                #value = 'lambda mnul, deltamnu21sq, deltamnu31sq: [mnul + (mnul**2 + deltamnu21sq)**0.5, (mnul**2 + deltamnu31sq)**0.5]'
                params['smnu'] = {'value': 'lambda mnul, deltamnu21sq, deltamnu31sq: mnul + (mnul**2 + deltamnu21sq)**0.5 + (mnul**2 + deltamnu31sq)**0.5', 'drop': True, 'derived': False}
                value = 'lambda mnul, smnu: [2 * mnul, smnu - 2 * mnul]'
                nunames = [1, 2]
                if dmnu2_fixed:
                    params['deltamnu31sq']['value'] = params['deltamnu31sq'].pop('prior')['loc']
            elif 'ih2' in model:
                # m3 << m1 < m2
                params['deltamnu32sq'] = {'prior': {'dist': 'norm', 'loc': deltamnu32sq[0], 'scale': deltamnu32sq[1]}, 'drop': True}
                params['deltamnu32sq']['ref'] = params['deltamnu32sq']['prior']
                #value = 'lambda mnul, deltamnu21sq, deltamnu32sq: [(mnul**2 - deltamnu32sq - deltamnu21sq)**0.5 + (mnul**2 - deltamnu32sq)**0.5, mnul]'
                params['smnu'] = {'value': 'lambda mnul, deltamnu21sq, deltamnu31sq: (mnul**2 - deltamnu32sq - deltamnu21sq)**0.5 + (mnul**2 - deltamnu32sq)**0.5 + mnul', 'drop': True, 'derived': False}
                value = 'lambda mnul, deltamnu21sq, deltamnu31sq, smnu: [2 * (mnul**2 - deltamnu32sq - deltamnu21sq)**0.5, smnu - 2 * (mnul**2 - deltamnu32sq - deltamnu21sq)**0.5]'
                nunames = [1, 2]
                if dmnu2_fixed:
                    params['deltamnu32sq']['value'] = params['deltamnu32sq'].pop('prior')['loc']

            params['mnulist'] = {'value': value, 'latex': r'\sum m_\nu', 'drop': True, 'derived': False}
            params['mnu'] = {'value': 'lambda mnulist: sum(mnulist)', 'latex': r'\sum m_\nu'}
            params['nu_mass_fractions'] = {'value': 'lambda mnulist: list(mnulist / np.sum(mnulist))', 'derived': False}
            #params['nu_mass_fractions'] = {'value': 'lambda mnulist: [0.27085477, 0.72914523]', 'derived': False}  # TOREMOVE
            
            for imu in range(1, 1 + len(nunames)):
                iname = nunames.index(imu)
                params['mnu{:d}'.format(imu)] = {'derived': 'lambda mnulist: mnulist[{:d}]'.format(iname), 'latex': r'm_{{\nu, {:d}}}'.format(imu), 'drop': True}

            params['num_nu_massless'] = {'value': 'lambda nnu: nnu - 3.044', 'derived': False}
            params['nu_mass_degeneracies'] = {'value': 'lambda nnu: [2 * 3.044 / 3, 3.044 / 3]', 'derived': False}
            
        params['nnu'] = {'prior': {'min': 0.05, 'max': 10.},
                         'ref': {'dist': 'norm', 'loc': 3.044, 'scale': 0.05},
                         'proposal': 0.05,
                         'latex': r'N_\mathrm{eff}'}
        params['w'] = {'prior': {'min': -3., 'max': 1.},
                       'ref': {'dist': 'norm', 'loc': -1., 'scale': 0.02},
                       'proposal': 0.02,
                       'latex': r'w_{0}'}
        params['wa'] = {'prior': {'min': -3., 'max': 2.},
                       'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.05},
                       'proposal': 0.05,
                       'latex': r'w_{a}'}
        params['omk'] = {'prior': {'min': -0.3, 'max': 0.3},
                         'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.01},
                         'proposal': 0.01,
                         'latex': r'\Omega_\mathrm{k}'}
        params['omegam'] = {'latex': r'\Omega_\mathrm{m}'}
        params['omegamh2'] = {'derived': 'lambda omegam, H0: omegam*(H0/100)**2',
                              'latex': r'\Omega_\mathrm{m} h^2'}
        params['omegal'] = {'latex': r'\Omega_\Lambda'}
        params['zrei'] = {'latex': r'z_\mathrm{reio}'}
        params['YHe'] = {'latex': r'Y_\mathrm{P}'}
        params['Y_p'] = {'latex': r'Y_P^\mathrm{BBN}'}
        params['DHBBN'] = {'derived': 'lambda DH: 10**5*DH',
                           'latex': r'10^5 \mathrm{D}/\mathrm{H}'}
        params['sigma8'] = {'latex': r'\sigma_8'}
        params['s8h5'] = {'derived': 'lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)',
                          'latex': r'\sigma_8/h^{0.5}'}
        params['s8omegamp5'] = {'derived': 'lambda sigma8, omegam: sigma8*omegam**0.5',
                                'latex': r'\sigma_8 \Omega_\mathrm{m}^{0.5}'}
        params['s8omegamp25'] = {'derived': 'lambda sigma8, omegam: sigma8*omegam**0.25',
                                 'latex': r'\sigma_8 \Omega_\mathrm{m}^{0.25}'}
        params['A'] = {'derived': 'lambda As: 1e9*As', 'latex': r'10^9 A_\mathrm{s}'}
        params['clamp'] = {'derived': 'lambda As, tau: 1e9*As*np.exp(-2*tau)',
                           'latex': r'10^9 A_\mathrm{s} e^{-2\tau}'}
        params['age'] = {'latex': r'\mathrm{Age}/\mathrm{Gyr}'}
        params['rdrag'] = {'latex': r'r_\mathrm{d}'}
        params['zdrag'] = {'latex': r'z_\mathrm{d}'}
        params['H0rdrag'] = {'derived': r'lambda H0, rdrag: H0 * rdrag',
                             'latex': r'H_0 r_\mathrm{d}'}
        #params['thetastar'] = {'latex': r'\theta_\ast'}
        # extra
        extra['bbn_predictor'] = 'PArthENoPE_880.2_standard.dat'
        extra['dark_energy_model'] = 'ppf'
        extra['theta_H0_range'] = [20, 100]
        extra['num_massive_neutrinos'] = 3
        # precision settings for lensing
        if 'lensing' in dataset:
            extra['halofit_version'] = 'mead2016'
            extra['lmax'] = 4000
            extra['lens_margin'] = 1250
            extra['lens_potential_accuracy'] = 4
            extra['AccuracyBoost'] = 1
            extra['lSampleBoost'] = 1
            extra['lAccuracyBoost'] = 1
        if 'mnu_nhcamb' in model:
            extra['neutrino_hierarchy'] = 'normal'
        if 'mnu_ihcamb' in model:
            extra['neutrino_hierarchy'] = 'inverted'
        if 'mnu_nh3' in model or 'mnu_ih3' in model:
            extra['share_delta_neff'] = False
            extra['nu_mass_numbers'] = [1] * len(nunames)
            extra['num_nu_massive'] = sum(extra['nu_mass_numbers'])
        if 'mnu_nh2' in model or 'mnu_ih2' in model:
            extra['share_delta_neff'] = False
            extra['nu_mass_numbers'] = [2, 1]
            extra['num_nu_massive'] = sum(extra['nu_mass_numbers'])

        if 'nnu' not in model:
            fix('nnu')
        if 'mnu' not in model:
            fix('mnu')
            extra['num_massive_neutrinos'] = 1
        if 'omegak' not in model:
            fix('omk')
        if 'w' not in model:
            fix('w')
            fix('wa')
        elif 'wa' not in model:
            fix('wa')
        if 'pk' in get_parameterization(dataset):
            fix('tau')
            params.pop('theta_MC_100')
            params.pop('cosmomc_theta')
            extra.pop('theta_H0_range')
            params['H0'] = {'prior': {'min': 20, 'max': 100},
                            'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
                            'latex': r'H_0'}
        if 'cmb' not in get_parameterization(dataset): # background only
            for name in ['logA', 'ns', 'tau']: fix(name)
            for name, di in list(params.items()):  # remove derived perturb quantities
                if any(n in name for n in ['sigma8', 's8']): params.pop(name)
            params.pop('theta_MC_100')
            params.pop('cosmomc_theta')
            if get_parameterization(dataset) == 'thermodynamics' and 'thetastar' in dataset:
                params['theta_s_100'] = {'prior': {'min': 0.5, 'max': 10.},
                                         'ref': {'dist': 'norm', 'loc': 1.04109, 'scale': 0.0004},
                                         'proposal': 0.0002, 'latex': r'100\theta_\mathrm{s}'}
                params['thetastar'] = {'value': 'lambda theta_s_100: 1.e-2*theta_s_100', 'derived': False}
            else:
                extra.pop('theta_H0_range')
                params['H0'] = {'prior': {'min': 20, 'max': 100},
                                'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
                                'latex': r'H_0'}
                if get_parameterization(dataset) == 'background':
                    fix('ombh2')
                    if 'rdrag' not in dataset: fix('H0')
                    if 'bao' in dataset:
                        #for name in ['H0', 'ombh2', 'omch2']: params.pop(name)  # to avoid confusion
                        params['hrdrag'] = {'prior': {'min': 10., 'max': 1000.},
                                             'ref': {'dist': 'norm', 'loc': 99.079, 'scale': 1.},
                                             'proposal': 1.,
                                             'latex': r'hr_\mathrm{d}'}
                        params['rdrag'] = {'value': 'lambda hrdrag, H0: 100 * hrdrag / H0',
                                           'latex': r'r_\mathrm{d}'}

                params['omm'] = {'prior': {'min': 0.01, 'max': 0.99},
                                 'ref': {'dist': 'norm', 'loc': 0.3152, 'scale': 0.001},
                                 'proposal': 0.0005,
                                 'drop': True}
                params['omch2'] = {'value': 'lambda omm, mnu, ombh2, H0: omm*(H0/100)**2 - mnu / 93.14 - ombh2',
                                   'latex': r'\Omega_\mathrm{c} h^2'}
        
        if 'isitgr' in theory:
            if 'desy3' not in dataset:
                extra.update(NonLinear='NonLinear_none')
            if model == 'base_mu_sigma':
                extra.update(MG_parameterization='muSigma', AccuracyBoost=1.1)
                params['mu0'] = {'prior': {'min': -3, 'max': 3},
                                 'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.1},
                                 'proposal': 0.01,
                                 'latex': r'\mu_0'}
                params['Sigma0'] = {'prior': {'min': -3, 'max': 3},
                                    'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.1},
                                    'proposal': 0.01,
                                    'latex': r'\Sigma_0'}
            elif model == 'base_mu_eta':
                extra.update(MG_parameterization='mueta', AccuracyBoost=1.1)
                params['E11'] = {'prior': {'min': -5, 'max': 5},
                                 'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.1},
                                 'proposal': 0.01,
                                 'latex': r'E_{11}'}
                params['mu0'] = {'latex': r'\mu_0'}
                params['Sigma0'] = {'latex': r'\Sigma_0'}
                params['E22'] = {'prior': {'min': -5, 'max': 5},
                                 'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.1},
                                 'proposal': 0.01,
                                 'latex': r'E_{21}'}

    return params, extra


def yield_configs(importance=False, models=None, **kwargs):
    for config in []:
        yield config


def sort_dataset(dataset, yield_configs=yield_configs):
    """Sort input dataset."""
    for config in yield_configs(importance=False):
        if set(dataset) == set(config['dataset']):
            return config['dataset']
    return dataset


def get_cobaya_output(model='base', theory='camb', dataset='desi-fs-bao-all', sampler='cobaya', add='', remove='', source=None, check_exists=False, base_dir=base_dir, run='auto', suffix=True, sort_dataset=sort_dataset):
    """
    Return cobaya base output path, given input model, theory, dataset.
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

    source : str, default='main'
        If 'auto', try to find available chains, from ['main'].
        Else, one source from this list.

    check_exists : bool, default=False
        If ``True``, check if file exists, if not return ``None``.

    Returns
    -------
    output : str
        Cobaya output path to samples.
    """
    import glob
    from pathlib import Path

    base_dir = Path(base_dir)
    if not isinstance(dataset, str):
        dataset = sort_dataset(dataset)
        dataset = '_'.join(dataset)
    if not isinstance(add, str):
        add = '_'.join(add)
    if not isinstance(remove, str):
        remove = '_'.join(remove)
    importance = ''
    if remove:
        importance += '_remove_' + remove
    if add:
        importance += '_add_' + add
    main_dir = base_dir / 'fs'
    if source == 'bao':
        main_dir = base_dir / 'bao'
        source = None
    if source is None:
        source = 'main'
    
    bestfit = 'cobaya' not in sampler
    if sampler.endswith('-likelihood'):
        bestfit = 'likelihood'
        sampler = sampler.replace('-likelihood', '')
    if sampler.endswith('-posterior'):
        bestfit = 'posterior'
        sampler = sampler.replace('-posterior', '')

    def get_output(run):
        output = str(main_dir / f'{"cobaya" if "cobaya" in sampler else sampler}/{theory}/{run}/{model}/{dataset}{importance}')
        if bestfit:
            output = output + '/bestfit'
        else:
            output = output + '/chain'
        if suffix:
            if bestfit:
                if bestfit is not True:
                    output = output + ('.minimum' if bestfit == 'posterior' else '.bestfit')
            elif importance:
                output = output + '.post.importance'
        return output

    if run == 'auto':
        for run in ['run4', 'run3']:
            # In run3 there was a bug in velocileptors, when rescaling the PT calculation at one z-bin to the other z-bins. It would show up mostly for cases where the k-dependence on the effect to be constrained depends on the redshift z. For mnu = 0.06 eV, I think this is totally negligible. In the run4 directory, I have relaunched DESI-only chains just to make sure, and chains involving mnu. I think the impact may just be visible for DESI-only constraints on mnu.
            output = get_output(run)
            if bool(glob.glob(output + '*.txt')):
                break
    else:
        output = get_output(run)

    if check_exists:
        if bool(glob.glob(output + '*.txt')):
            return output
        raise ValueError('no samples found for model = {}, dataset = {}, add = {}! (tried {})'.format(model, dataset, add, output))
    return output


def exists_cobaya_output(*args, **kwargs):
    """Whether samples exist, see :func:`get_cobaya_output` for input arguments."""
    import glob
    output = get_cobaya_output(*args, **kwargs)
    return len(glob.glob(output + '*.*.txt')) >= 4


def get_cobaya_covmat(model='base', theory='camb', dataset='desi-bao-all', run='run3', base_dir=covmat_dir, get_parameterization=get_parameterization, get_cobaya_output=get_cobaya_output):
    """Return cobaya proposal matrix."""
    import os
    from pathlib import Path
    if 'cmb' not in get_parameterization(dataset):
        return None
    #if run == 'test':
    #    return None
    if not isinstance(dataset, str):
        dataset = '_'.join(dataset)

    if '-prior3' in dataset:
        covmat = str(get_cobaya_output(model=model, theory=theory, dataset=dataset.replace('-prior3', ''), sampler='cobaya', add='', remove='', base_dir=base_dir, run='run3', suffix=False)) + '.covmat'
        return covmat
    if model == 'base_mnu':
        covmat = f'/global/cfs/cdirs/desicollab/science/cpe/y1kp7/fs/cobaya/camb/run4/base_mnu/desi-reptvelocileptors-fs-bao-all_schoneberg2024-bbn_planck2018-ns10/chain.covmat'
    
    if run == 'run4':
        if dataset.endswith('planck2018-ns'):
            dataset = dataset.replace('planck2018-ns', 'planck2018-ns10')
        covmat = str(get_cobaya_output(model=model, theory=theory, dataset=dataset, sampler='cobaya', remove='', base_dir=base_dir, run='run3', suffix=False)) + '.covmat'
        if not os.path.exists(covmat):
            covmat = str(get_cobaya_output(model='base', theory=theory, dataset=dataset, sampler='cobaya', remove='', base_dir=base_dir, run='run3', suffix=False)) + '.covmat'
        assert os.path.exists(covmat), covmat
        return covmat

    if dataset == 'planck2018-lowl-TT-clik_planck2020-lollipop-lowlE_planck2020-hillipop-TTTEEE_planck-act-dr6-lensing':
        if model == 'base_mu_sigma':
            covmat = str(get_cobaya_output(model='base', theory='camb', dataset=['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE'], sampler='cobaya', add='', remove='', base_dir=base_dir, run='run1', suffix=False)) + '.covmat'
            return covmat
    
    if dataset == 'planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck2018-highl-plik-TTTEEE_planck-act-dr6-lensing':
        if model == 'base_mu_sigma':
            covmat = str(get_cobaya_output(model=model, theory=theory, dataset=dataset, sampler='cobaya', add='', remove='', base_dir=base_dir, run='run1', suffix=False)) + '.covmat'
            return covmat
            
        from y1_bao_cosmo_tools import get_cobaya_output as get_cobaya_bao_output
        covmat = str(get_cobaya_bao_output(model=model, theory=theory, dataset=dataset.replace('_planck-act-dr6-lensing', ''), sampler='cobaya', add='', remove='', base_dir=base_dir, run='run0', source='hanyuz' if 'mnu' in model else 'jiamingp', suffix=False)) + '.covmat'
        return covmat
    
    dataset = dataset.replace('-nolya', '')
    if 'desi-v1.5-bao' in dataset:
        dataset = dataset.replace('-v1.5', '')
        dataset = dataset.replace('_planck-act-dr6-lensing', '')
        from y1_bao_cosmo_tools import get_cobaya_output as get_cobaya_bao_output
        covmat = str(get_cobaya_bao_output(model=model, theory=theory, dataset=dataset, sampler='cobaya', add='', remove='', base_dir=base_dir, run='run1', suffix=False)) + '.covmat'
        return covmat
    if dataset.endswith('planck2018-ns'):
        dataset = dataset.replace('planck2018-ns', 'planck2018-ns10')
    if 'shapefit' in dataset:
        dataset = dataset.replace('shapefit-joint', 'reptvelocileptors-fs').replace('shapefit', 'reptvelocileptors-fs')
        covmat = str(get_cobaya_output(model=model, theory=theory, dataset=dataset, sampler='cobaya', add='', remove='', base_dir=base_dir, run='run3', suffix=False)) + '.covmat'
        assert os.path.exists(covmat), covmat
        return covmat
    if '-bao' not in dataset:
        dataset = dataset.replace('-fs', '-fs-bao')
    if 'mock-cmblens' in dataset:
        dataset = dataset.replace('mock-cmblens-', '')
    if 'desy5sn' in dataset and any(name in dataset for name in ['planck-NPIPE-highl-CamSpec-TTTEEE', 'planck2020-hillipop-TTTEEE']):
        dataset = dataset.replace('_desy5sn', '')
    if model == 'base_mu_sigma' and any(name in dataset for name in ['planck-NPIPE-highl-CamSpec-TTTEEE', 'planck2020-hillipop-TTTEEE']):
        theory, model = 'camb', 'base'
    if model == 'base_nnu':
        model = 'base_mnu'
    if model in ['base_mnu_w_wa', 'base_nnu_w_wa']:
        model = 'base_mnu'
        for sn in ['pantheonplus', 'union3', 'desy5sn']:
            dataset = dataset.replace('_' + sn, '')
    run = 'run1'
    covmat = str(get_cobaya_output(model=model, theory=theory, dataset=dataset, sampler='cobaya', add='', remove='', base_dir=base_dir, run=run, suffix=False)) + '.covmat'
    if 'planck-act-dr6-lensing' in dataset:
        if not os.path.exists(covmat):
            dataset = dataset.replace('_planck-act-dr6-lensing', '')
            covmat = str(get_cobaya_output(model=model, theory=theory, dataset=dataset, sampler='cobaya', add='planck-act-dr6-lensing', remove='', base_dir=base_dir, run=run, suffix=False)) + '.covmat'
            if not os.path.exists(covmat):
                covmat = str(get_cobaya_output(model=model, theory=theory, dataset=dataset, sampler='cobaya', add='', remove='', base_dir=base_dir, run=run, suffix=False)) + '.covmat'

    #activate the lines to run without passing covmat
    #if not os.path.exists(covmat):
    #    return None
    #return covmat


def get_cobaya_info(model='base', theory='camb', dataset='desi-bao-all', sampler='cobaya', seed=None, save_fn=None, get_cobaya_likelihoods=get_cobaya_likelihoods, get_cobaya_params=get_cobaya_params, get_cobaya_covmat=get_cobaya_covmat, get_cobaya_output=get_cobaya_output, get_parameterization=get_parameterization, add=(), remove=(), temperature=1, **kwargs):
    """Return cobaya info dictionary for sampling."""
    if isinstance(dataset, str): dataset = [dataset]
    else: dataset = list(dataset)
    if isinstance(add, str): add = [add]
    else: add = list(add)
    likelihoods = get_cobaya_likelihoods(dataset=dataset + add)
    likelihoods_remove = get_cobaya_likelihoods(dataset=remove)
    for likelihood in likelihoods_remove:
        likelihoods.pop(likelihood)
    params, extra_args = get_cobaya_params(model=model, theory=theory, dataset=dataset + add)

    info = {}
    output = get_cobaya_output(model=model, theory=theory, dataset=dataset, sampler=sampler, suffix='cobaya' in sampler, add=add, remove=remove, **kwargs)
    #info['stop_at_error'] = True  #TOREMOVE
    if 'evaluate' in sampler:
        sampler = {'evaluate': None}
        info['stop_at_error'] = True
        output = None
    elif sampler == 'cobaya':
        covmat = get_cobaya_covmat(model=model, theory=theory, dataset=dataset, run=kwargs.get('run', 'run3'))
        if covmat is not None:
            print('Using covmat: {}'.format(covmat))
        is_cmb = 'cmb' in get_parameterization(dataset)
        sampler = {'mcmc': {'drag': is_cmb,
                            'oversample_power': 0.4,
                            'proposal_scale': 1.9,
                            'covmat': covmat,
                            'temperature': temperature,
                            'Rminus1_stop': 0.01,
                            'Rminus1_cl_stop': 0.2 if is_cmb else 0.02,
                            #'Rminus1_stop': 0.1,
                            #'Rminus1_cl_stop': 1,
                            'learn_proposal_Rminus1_max': 30.,
                            'seed': seed,
                            'measure_speeds': True,
                            'max_tries': 1000}}
    elif sampler == 'polychord':
        is_cmb = 'cmb' in get_parameterization(dataset)
        sampler = {'polychord': {}}
    else:
        for minimizer in ['bobyqa', 'scipy', 'iminuit']:
            if minimizer in sampler:
                sampler = {'minimize': {'method': minimizer,
                                        'ignore_prior': False,
                                        'max_evals': int(1e6),
                                        'best_of': 4,
                                        'confidence_for_unbounded': 0.9999995,
                                        'seed': seed}}
                #if theory != 'classy':
                #    extra_args['lAccuracyBoost'] = 1
                #if minimizer == 'iminuit':
                #    # default tol is 0.1
                #    sampler['minimize']['override_iminuit'] = {} # distance to minimum is 0.002 * tol * errordef, https://github.com/scikit-hep/iminuit/blob/298afadc645999fbad32ca00be6c0ebfec7c7bb9/src/iminuit/minuit.py#L44
                break
    if not isinstance(sampler, dict):
        raise ValueError('unknown sampler {}'.format(sampler))
    info.update({'theory': {theory: {'extra_args': extra_args, 'speed': 2}}, 'likelihood': likelihoods, 'params': params, 'sampler': sampler, 'output': output})
    for likelihood in info['likelihood']:
        if 'desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods' in likelihood:
            info['theory']['desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods.reptvelocileptors'] = {'python_path': info['likelihood'][likelihood]['python_path']}
    if theory == 'isitgr':
        from pathlib import Path
        import desi_y1_files
        local_pythonpath = str(Path(desi_y1_files.__file__).parent.parent)
        theory = info['theory'].pop('isitgr')
        theory['python_path'] = local_pythonpath
        #theory['extra_args'] = {'MG_parameterization': 'muSigma', 'AccuracyBoost': 1.1, 'NonLinear': 'NonLinear_none'}
        theory['stop_at_error'] = False
        #theory['ignore_obsolete'] = True
        info['theory']['desi_y1_cosmo_bindings.cobaya_theories.isitgr.ISiTGR'] = theory
        if model == 'base_mu_sigma':
            info['prior'] = {'MG_prior': 'lambda mu0, Sigma0: (0 if mu0 < 2*Sigma0 + 1 else -np.inf)'}
    if save_fn is not None:
        import yaml
        import os
        os.makedirs(Path(save_fn).parent, exist_ok=True)
        with open(save_fn, 'w') as file:
            yaml.dump(info, file, sort_keys=False)
    return info


def print_margestats(output, limits=(1, 2), sigfigs=2, fn=None):
    """
    Print 1D marginalized posterior statistics, for given cobaya output path.
    If ``fn`` is ``True``, save to disk.
    """
    from cobaya.output import load_samples
    samples = load_samples(output, skip=0.3, thin=1, combined=False, to_getdist=True)
    if fn and fn is True:
        fn = str(output) + '.margestats'
    #table = samples.getTable(limit=1).tableTex()
    params = samples.paramNames.list()
    values = [samples.getLatex(params, limit=limit, err_sig_figs=sigfigs)[1] for limit in limits]
    fmt = '{:<40}' * (len(values) + 1)
    txt = fmt.format('parameter', *['{:.1f}%'.format(contour * 100) for contour in samples.contours]) + '\n'
    for iparam, param in enumerate(params):
        if any(name in param for name in ['chi2']): continue
        txt += fmt.format(param, *[value[iparam] for value in values]) + '\n'
    print(txt)
    if fn:
        with open(fn, 'w') as file:
            file.write(txt)


def sample_cobaya(resume=False, model='base', theory='camb', dataset='desi-fs-bao-all', sampler='cobaya', get_cobaya_info=get_cobaya_info, print_margestats=print_margestats, test=False, **kwargs):
    """Sample with cobaya."""
    import os
    import glob
    import traceback
    from mpi4py import MPI
    from cobaya import run
    from cobaya.log import LoggedError

    mpicomm = MPI.COMM_WORLD
    mpicomm.Barrier()
    info = get_cobaya_info(model=model, theory=theory, dataset=dataset, sampler=sampler, **kwargs)
    
    if False:
        if 'mcmc' in info['sampler']:
            blocking = [[1, []], [2, []]]
            for param in model.prior.params:
                if param in info['params']:  # only cosmo parameters
                    blocking[0][1].append(param)
                else:
                    blocking[1][1].append(param)
            info['sampler']['mcmc']['blocking'] = blocking

    import yaml
    print(yaml.dump(info, sort_keys=False))

    from cobaya.model import get_model
    model = get_model({name: value for name, value in info.items() if name != 'sampler'})
    output = info['output']

    ### Hack starts ###
    # Cobaya tries loading with dill first, here: https://github.com/CobayaSampler/cobaya/blob/f21ed21059a6c37aa141fb465cc7c7a99b053063/cobaya/output.py#L478
    # This works well, but yaml cannot cope with the dynamic likelihood in https://github.com/CobayaSampler/cobaya/blob/f21ed21059a6c37aa141fb465cc7c7a99b053063/cobaya/output.py#L488, therefore crashes.
    # Let's ignore all previous info at once
    def reload_updated_info(self, cache=False):
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

    from cobaya.sampler import Sampler
    from cobaya.output import OutputReadOnly as Output
    set_checkpoint_info_bak, reload_updated_info_bak = Sampler.set_checkpoint_info, Output.reload_updated_info
    Sampler.set_checkpoint_info, Output.reload_updated_info = set_checkpoint_info, reload_updated_info
    ### Hack ends ###

    mpicomm.Barrier()
    if output is not None:
        try:
            for fn in glob.glob(output + '*.lock*'): os.remove(fn)
        except FileNotFoundError:
            pass
    mpicomm.Barrier()

    success = False
    trace = ''
    try:
        updated_info, mcmc = run(info, force=not bool(resume), resume=bool(resume), test=test)
        success = True
    except LoggedError:
        trace = traceback.format_exc()
    # Did it work? (e.g. did not get stuck)
    success, trace = all(mpicomm.allgather(success)), mpicomm.allgather(trace)

    Sampler.set_checkpoint_info, Output.reload_updated_info = set_checkpoint_info_bak, reload_updated_info_bak

    if not success:
        raise RuntimeError('Sampling failed!:{}'.format('\n'.join(trace)))
    try: print_margestats(output, fn=True)
    except: pass
    return success


def profile_cobaya(model='base', theory='camb', dataset='desi-bao-all', sampler='iminuit', get_cobaya_info=get_cobaya_info, get_cobaya_output=get_cobaya_output, ignore_prior=False, add=(), remove=(), run='run0', **kwargs):
    info = get_cobaya_info(model=model, theory=theory, dataset=dataset, sampler=sampler, add=add, remove=remove, run=run, **kwargs)

    for likelihood, di in info['likelihood'].items():
        if 'fs_' in likelihood:
            di = di or {}
            di['solve'] = '.best'
            info['likelihood'][likelihood] = di

    sampler = info.pop('sampler')
    sampler['minimize']['ignore_prior'] = ignore_prior
    import os
    import yaml
    print(yaml.dump(info, sort_keys=False))
    import numpy as np
    from cobaya.model import get_model
    from cobaya.sampler import get_sampler
    from cobaya.output import get_output
    prefix = info.pop('output')
    output = get_output(prefix=prefix, resume=False, force=True)
    #add = ()
    input = get_cobaya_output(model=model.replace('mnu059', 'mnu').replace('mnu100', 'mnu'), theory=theory, dataset=dataset, add=add, remove=remove, run='auto', source='main', sampler='cobaya')
    if input is not None and os.path.exists(os.path.dirname(input)):
        input = get_output(prefix=input, resume=False, force=False)
        input.get_updated_info = lambda *args, **kwargs: {}
        input.check_and_dump_info = lambda *args, **kwargs: None  # to avoid overwriting updated.yaml
    else:
        input = None
    sampler = get_sampler(sampler, model=get_model(info), output=input)
    sampler._output = output
    sampler.run()
    from cobaya.samplers.minimize.minimize import getdist_ext_ignore_prior
    fn = prefix + getdist_ext_ignore_prior[ignore_prior] + '.npy'
    products = sampler.products()
    products["result_object"].minuit = None
    np.save(fn, products)


def get_cobaya_importance_info(model='base', theory='camb', dataset='desi-bao-all', add=(), remove=(), skip=0.3, thin=2, save_fn=None, get_cobaya_likelihoods=get_cobaya_likelihoods, get_cobaya_params=get_cobaya_params, get_cobaya_covmat=get_cobaya_covmat, get_cobaya_output=get_cobaya_output, run='run1', **kwargs):
    """Return cobaya info dictionary for importance sampling."""
    import os
    # Generate the output path for the chain based on the model and dataset
    output = get_cobaya_output(model=model, theory=theory, dataset=dataset, run='run3', **kwargs)
    output_importance = get_cobaya_output(model=model, theory=theory, dataset=dataset, add=add, remove=remove, suffix=False, run='test', **kwargs)

    likelihoods_add = get_cobaya_likelihoods(dataset=add)
    likelihoods_remove = get_cobaya_likelihoods(dataset=remove)

    info = {'output': output, 'post': {}}
    info['post']['output'] = output_importance
    info['post']['suffix'] = 'importance'
    info['post']['skip'] = skip
    #info['post']['thin'] = 4
    info['post']['thin'] = thin
    info['post']['add'] = {'likelihood': likelihoods_add}
    if 'lensing' in '_'.join(add):
        info['post']['add']['theory'] = {theory: {'extra_args': get_cobaya_params(model=model, theory=theory, dataset=add)[1]}}
    info['post']['remove'] = {'likelihood': likelihoods_remove}
    if 'reptvelocileptors' in '_'.join(dataset):
        info['post']['remove']['theory'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods.reptvelocileptors': {}}
    return info


def importance_cobaya(resume=False, model='base', theory='camb', dataset='desi-bao-all', add=(), remove=(), get_cobaya_importance_info=get_cobaya_importance_info, **kwargs):
    """Importance sample with cobaya."""
    import os
    import glob
    import traceback
    from mpi4py import MPI
    from cobaya import post
    from cobaya.log import LoggedError

    mpicomm = MPI.COMM_WORLD
    mpicomm.Barrier()
    info = get_cobaya_importance_info(model=model, theory=theory, dataset=dataset, add=add, remove=remove, **kwargs)
    import yaml
    print(yaml.dump(info, sort_keys=False))
    output = info['post']['output']

    mpicomm.Barrier()
    if output is not None:
        for fn in glob.glob(output + '*.lock*'):
            try: os.remove(fn)
            except FileNotFoundError: pass
    mpicomm.Barrier()

    success = False
    trace = ''
    try:
        updated_info, mcmc = post(info)
        success = True
    except LoggedError:
        trace = traceback.format_exc()
    # Did it work? (e.g. did not get stuck)
    success, trace = all(mpicomm.allgather(success)), mpicomm.allgather(trace)

    if not success:
        raise RuntimeError('Sampling failed!:{}'.format('\n'.join(trace)))


def plot_progress(output, fn=None):
    """
    Plot chain progress for given cobaya output path.
    To save the figure, provide the path ``fn``.
    """
    import logging
    from pathlib import Path
    from matplotlib import pyplot as plt
    from cobaya.samplers.mcmc import plot_progress
    
    logger = logging.getLogger('Progress')

    # Define the save path for the figure based on the chain output path
    if fn is None:
        fn = Path(output).parent / f'progress.png'

    # Plot the progress
    plot_progress(output, figure_kwargs={'figsize': (6, 4)})

    plt.tight_layout()
    # Save the figure to the dynamically generated path
    plt.savefig(fn)
    # Close the figure to free up memory
    plt.close()
    logger.info(f'Progress plot saved to: {fn}')


def getGelmanRubinEigenvalues(self, pars=None, chainlist=None):
    """
    Modified GetDist utility, to pass parameter names.
    Assess convergence using var(mean)/mean(var) in the orthogonalized parameters
    c.f. Brooks and Gelman 1997.

    :param pars: Parameter names
    :param chainlist: list of :class:`~.chains.WeightedSamples`, the samples to use.
                      Defaults to all the separate chains in this instance.
    :return: array of  var(mean)/mean(var) for orthogonalized parameters
    """
    import numpy as np
    if chainlist is None:
        chainlist = self.getSeparateChains()
    if pars is None:
        pars = list(range(self.paramNames.numNonDerived()))
    else:
        indices = self._getParamIndices()
        pars = [indices[name] for name in pars]
    nparam = len(pars)
    meanscov = np.zeros((nparam, nparam))
    means = self.getMeans(pars)
    meancov = np.zeros(meanscov.shape)
    for chain in chainlist:
        diff = chain.getMeans(pars=pars) - means
        meanscov += np.outer(diff, diff)
        meancov += chain.getCov(pars=pars)
    meanscov /= (len(chainlist) - 1)
    meancov /= len(chainlist)
    w, U = np.linalg.eigh(meancov)
    if np.min(w) > 0:
        U /= np.sqrt(w)
        D = np.linalg.eigvalsh(np.dot(U.T, meanscov).dot(U))
        return D
    else:
        return None


def print_convergence(*output, what=None, max_gr=0.01, min_corr=0.05, skip=0.3, thin=1, short=True, cosmo_only=False):
    """
    Print Gelman-Rubin convergence and Effective Sample Size in the terminal, for given cobaya output path(s).
    
    Parameters
    ----------
    min_corr : float
        Mnimum value of the auto-correlation to use for correlation length estimation. Default to 0.05.
    """
    # what = ('MeanVar', 'GelmanRubin', 'SplitTest', 'RafteryLewis', 'CorrLengths')
    import numpy as np
    from cobaya.output import load_samples
    for output in output:
        samples = load_samples(output, skip=skip, thin=thin, combined=False, to_getdist=True)
        if not samples: return
        if what is None:
            if cosmo_only:
                params = list(get_cobaya_params(model='base_nnu_mnu_w_wa', theory='camb')[0])
                pars = []
                for name in samples.paramNames.names:
                    if not name.isDerived and name.name in params:
                        pars.append(name.name)
                gr = getGelmanRubinEigenvalues(samples, pars=pars).max()
            else:
                gr = samples.getGelmanRubinEigenvalues().max()
            ok = gr < max_gr
            if ok:
                print(f'{TermColors.OKGREEN}Gelman-Rubin is {gr:.3f} < {max_gr:.3f}.{TermColors.ENDC}')
            else:
                print(f'{TermColors.FAIL}Gelman-Rubin is {gr:.3f} > {max_gr:.3f}.{TermColors.ENDC}')
            ess = {name.name: samples.getEffectiveSamples(samples.paramNames.numberOfName(name.name), min_corr) for name in samples.paramNames.names if not name.isDerived}
            txt = 'ESS is:'
            if not short: txt += '\n'.join(['{} {:.0f}'.format(n, e) for n, e in ess.items()])
            ess = list(ess.values())
            txt += '\nmean ESS = {:.0f}, min ESS = {:.0f}'.format(np.mean(ess), np.min(ess))
            print(txt)
            return ok
        else:
            print(samples.getConvergeTests(test_confidence=0.95, writeDataToFile=False,
                                           what=what, filename=None, feedback=False))


def add_derived_getdist(samples):
    from desi_y1_plotting import KP7StylePaper
    # Add aliases here
    def set_renames(samples, *list_renames):
        renames = {}
        for name in samples.paramNames.list():
            for lr in list_renames:
                if name in lr:
                    renames[name] = list(lr)
                    break
        samples.updateRenames(renames)

    set_renames(samples, *[('omegam', 'Om', 'Omega_m'),
                          ('ombh2', 'omega_b'),
                          ('omch2', 'omega_c'),
                          ('omegal', 'oml', 'Oml', 'Omega_Lambda'),
                          ('omegak', 'omk', 'Omk', 'Omega_k'),
                          #('H0rd', 'H0_rd', 'H0rdrag'),
                          ('ns', 'n_s'),
                          ('sigma8', 'sigma8_m'),
                          ('rd', 'rdrag'),
                          ('w', 'w0', 'w0_fld'),
                          ('wa', 'wa_fld')])
    samples.paramNames.setLabelsFromParamNames(KP7StylePaper().settings.param_names_for_labels)
    from cosmoprimo import constants
    try: samples.addDerived(samples['H0'] * samples['rdrag'], 'H0rdrag', r'H_0 r_\mathrm{d}')
    except: pass  # already exists
    try: samples.addDerived(samples['H0rdrag'] / (constants.c * 1e-3), 'H0rdragc', r'H_0 r_\mathrm{d}/c')
    except: pass  # already exists
    try: samples.addDerived(samples['H0rdrag'] * 1e-2, 'hrdrag', r'r_\mathrm{d}h')
    except: pass  # already exists
    try: samples.addDerived(samples['sigma8'] * (samples['omegam'] / 0.3)**0.5, 'S8', r'S_8')
    except: pass  # already exists
            

def load_cobaya_samples(skip=None, thin=1, combined=False, add=(), remove=(), label=None, source='auto', sampler='cobaya', convergence=False, bestfit=None, ichain=None, get_cobaya_output=get_cobaya_output, add_derived_getdist=add_derived_getdist, **config):
    """
    Load cobaya samples.

    Parameters
    ----------
    skip : float, default=None
        Fraction of samples to skip (burnin).
        Defaults to 0.3 if not importance sampling, else 0.

    thin : int, default=1
        Thin samples by this factor (helps decorrelate samples for nicer plots).

    label : str, default=None
        Label for GetDist.

    source : str, default='main'
        If 'auto', try to find available chains, from ['main', 'hanyuz', 'kushal', 'jiamingp', 'jshree'].
        Else, one source from this list.

    bestfit : str, default=None
        'iminuit' to set GetDist best fit.

    **config : dict
        See :func:`get_cobaya_output` for arguments.

    Returns
    -------
    samples : getdist.MCSamples
    """
    output = get_cobaya_output(add=add, remove=remove, source=source, sampler=sampler, **config)
    print('Loading {}'.format(output))
    if skip is None:
        skip = 0 if add or remove else 0.3
    if convergence:
        print_convergence(output, skip=skip, thin=thin)
    if 'bestfit' in output:
        import os
        suffix = '.minimum' if 'posterior' in bestfit else '.bestfit'
        fn_txt = '{}{}.txt'.format(output, suffix)
        if not os.path.isfile(fn_txt):
            di = {}
        else:
            with open(fn_txt, 'r') as file:
                lines = [line[:-1] for line in file]  # -1 to remove \n
            names = [name for name in lines[0].split(' ') if name][2:]
            values = [float(value) for value in lines[1].split(' ') if value][1:]
            di = dict(zip(names, values))
        import numpy as np
        return di, np.load('{}{}.npy'.format(output, suffix), allow_pickle=True)[()]
    from cobaya.output import load_samples
    collections = load_samples(output, skip=skip, thin=thin, combined=False, to_getdist=False)
    if ichain is None:
        collections[0].reset_temperature(with_batch=collections[1:])
        samples = collections[0].to_getdist(combine_with=collections[1:])
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            collections[ichain].reset_temperature()
            samples = collections[ichain].to_getdist()
    #samples = load_samples(output, skip=skip, thin=thin, combined=combined, to_getdist=True)
    add_derived_getdist(samples)
    if label is not None:
        samples.label = label
    if bestfit is not None:
        bf_root = get_cobaya_output(add=add, remove=remove, source='main', sampler=bestfit, check_exists=True, suffix=True, **config)
        samples.root = bf_root
        """
        def getBestFit(max_posterior=True):
            ext = '.minimum' if max_posterior else '.bestfit'
            bf_file = bf_root + ext
            from getdist import types
            from getdist.mcsamples import MCSamplesError
            if os.path.exists(bf_file):
                return types.BestFit(bf_file, max_posterior=max_posterior)
            else:
                raise MCSamplesError('Best fit can only be included if loaded from file and file_root%s exists '
                                     '(cannot be calculated from samples)' % ext)
        """
    return samples


def load_sdss_samples(model='base', dataset='sdss-fs-bao-all', skip=0.5, label=None):
    """
    Load SDSS samples.
    
    Taken from https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/mcmc/base
    """
    from pathlib import Path
    from getdist import loadMCSamples

    base_dir = Path('/global/cfs/cdirs/desicollab/science/cpe/y1kp7/fs/sdss_dr16/')
    if not isinstance(dataset, str):
        dataset = sort_dataset(dataset)
        dataset = '_'.join(dataset)
    
    def convert_dataset(dataset, convert):
        output = []
        for rename, ds in convert.items():
            if all(dds in dataset for dds in ds):
                output.append(rename)
                for dds in ds: dataset = dataset.replace(dds, '')
        return output

    convert = {'BAOonly': ['sdss-bao-all'], 'RSD_lenspriors': ['sdss-fs-all'], 'BAORSD_lenspriors': ['sdss-fs-bao-all']}
    dataset = '_'.join(convert_dataset(dataset, convert))

    output = str(base_dir / f'{model}/{dataset}/{model}_{dataset}')
    
    samples = loadMCSamples(output)
    add_derived_getdist(samples)
    if label is not None:
        samples.label = label
    return samples


def load_harvester_samples(dataset='des_6x2_desi', label=None):
    # Combination by Peter Taylor
    from pathlib import Path
    import pickle
    import numpy as np
    with open('/global/cfs/cdirs/desicollab/science/cpe/y1kp7/fs/pltaylor/joint_final/combine_scripts/combine_harvester_desi_weights.pkl', 'rb') as file:
        weight_dict = pickle.load(file)
    
    flow_dir = Path('/global/cfs/cdirs/desicollab/science/cpe/y1kp7/fs/pltaylor/joint_final/flow_data/')
    desi = np.load(flow_dir / 'desi_chain.npy')
    desi_planck_nl = np.load(flow_dir / 'desi_planck_nl_chain.npy')
    des_6x2 = np.load(flow_dir / 'des_6x2_chain.npy')
    des_6x2_weights = np.load(flow_dir / 'des_6x2_weights.npy')
    
    names = ['sigma8', 'omegam', 'ns', 'H0', 'ombh2']
    labels = [r'\sigma_8', r'\Omega_\mathrm{m}', r'n_\mathrm{s}', r'H_0', r'\omega_\mathrm{b}']

    from getdist import MCSamples
    if dataset == 'des_6x2_desi':
        samples = MCSamples(samples=desi, weights=weight_dict['des_6x2_desi'], names=names, labels=labels, label=label)
    if dataset == 'des_3x2_desi':
        samples = MCSamples(samples=desi, weights=weight_dict['des_3x2_desi'], names=names, labels=labels, label=label)
    if dataset == 'des_6x2':
        samples = MCSamples(samples=des_6x2, weights=des_6x2_weights, names=names, labels=labels, label=label)
    if dataset == 'des_6x2_desi_planck_nl':
        samples = MCSamples(samples=desi_planck_nl, weights=weight_dict['des_6x2_desi_planck_nl'], names=names, labels=labels, label=label)
    add_derived_getdist(samples)
    return samples


def load_des_samples(model='base', skip=0.5, label=None):
    """
    Load DES (Y3 3x2pt) samples.
    
    Taken from https://des.ncsa.illinois.edu/releases/y3a2/Y3key-products
    """

    import numpy as np
    from getdist import MCSamples
    from y1_fs_cosmo_tools import add_derived_getdist
    
    if model == 'base':
        names = ['omegam', 'H0', 'ns', 'sigma8']
        labels = ['\Omega_\mathrm{m}','H_0', 'n_\mathrm{s}', '\sigma_8']
        fn = '/global/cfs/cdirs/desicollab/science/cpe/y1kp7/fs/des_y3/base/chain_3x2pt_lcdm_SR_maglim.txt'
        samples = np.loadtxt(fn, usecols=[0, 1, 3, 31, -2, -1], comments='#')
        samples[:, 1] *= 100.
    elif model == 'base_mu_sigma':
        names = ['omegam', 'H0', 'ns', 'sigma8', 'Sigma0', 'mu0']
        labels = ['\Omega_\mathrm{m}','H_0', 'n_\mathrm{s}', '\sigma_8', '\Sigma_0', '\mu_0']
        fn = '/global/cfs/cdirs/desicollab/science/cpe/y1kp7/fs/des_y3/base_mu_sigma/chain_2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits.Y3_mg_scales-feb22_linonly_ml.ini.d3_sigmu_nla_realy3dat.txt'
        samples = np.loadtxt(fn, usecols=[0, 1, 3, 30, 28, 29, -2, -1], comments='#')
        samples[:, 1] *= 100.

    if skip is not None:
        if skip < 1.:
            burnin = int(skip * len(samples))
        else:
            burnin = int(skip)
        samples = samples[burnin:]
    samples = samples.T

    samples = MCSamples(samples=np.column_stack(samples[:len(names)]), weights=samples[-1], loglikes=samples[-2], names=names, labels=labels, label=label)
    add_derived_getdist(samples)
    return samples


def load_planck2018_samples(model='base', dataset='', add=(), remove=(), label=None):
    from pathlib import Path
    from getdist import loadMCSamples

    base_dir = Path('/global/cfs/cdirs/desi/science/cpe//perlmutter/cosmodesiconda/20240118-1.0.0/desilike/data/FullGridPlanck2018GaussianLikelihood')

    if not isinstance(dataset, str):
        dataset = sort_dataset(dataset)
        dataset = '_'.join(dataset)
    if not isinstance(add, str):
        add = '_'.join(add)
    if not isinstance(remove, str):
        remove = '_'.join(remove)
    
    def convert_dataset(dataset, convert):
        output = []
        for rename, ds in convert.items():
            if all(dds in dataset for dds in ds):
                output.append(rename)
                for dds in ds: dataset = dataset.replace(dds, '')
        if dataset.replace('_', ''):
            if 'planck-act-dr6-lensing' in dataset:
                raise ValueError('pass "planck-act-dr6-lensing" to add=')
            raise ValueError('remaining unknown dataset {}'.format(dataset))
        return output

    convert = {'plikHM_TTTEEE_lowl_lowE': ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']}
    convert_add = {'lensing': ['planck2018-lensing']}
    dataset = '_'.join(convert_dataset(dataset, convert))
    add = '_'.join(convert_dataset(add, convert_add))
    
    importance = f'_post_{add}' if add else ''
    output = str(base_dir / f'{model}/{dataset}/{model}_{dataset}{importance}')
    
    samples = loadMCSamples(output)
    add_derived_getdist(samples)
    if label is not None:
        samples.label = label
    return samples


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
                      'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001},
                      'proposal': 0.001,
                      'latex': r'\ln(10^{10} A_\mathrm{s})',
                      'delta': 0.07,
                      'drop': True}
    params['A_s'] = {'derived': '1e-10*jnp.exp({logA})',
                    'latex': r'A_\mathrm{s}'}
    params['n_s'] = {'prior': {'limits': [0.8, 1.2]}, 
                    'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004},
                    'proposal': 0.002,
                    'latex': r'n_\mathrm{s}',
                    'delta': 0.01}
    params['theta_MC_100'] = {'prior': {'limits': [0.5, 10.]},
                              'ref': {'dist': 'norm', 'loc': 1.04109, 'scale': 0.0004},
                              'proposal': 0.0002, 'latex': r'100\theta_\mathrm{MC}', 'delta': 0.001}
    params['H0'] = {'derived': True, 'latex': 'H_0'}
    params['omega_b'] = {'prior': {'limits': [0.005, 0.1]},
                        'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001},
                        'proposal': 0.0001,
                        'latex': r'\Omega_\mathrm{b} h^2',
                        'delta': 0.0015}

    params['omega_cdm'] = {'prior': {'limits': [0.001, 0.99]},
                           'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001},
                           'proposal': 0.0005,
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
                        'proposal': 0.05,
                        'latex': r'N_\mathrm{eff}',
                        'delta': 0.2}
    params['w0_fld'] = {'prior': {'limits': [-3., 1.]},
                        'ref': {'dist': 'norm', 'loc': -1., 'scale': 0.02},
                        'proposal': 0.02,
                        'latex': r'w_{0}',
                        'delta': 0.1}
    params['wa_fld'] = {'prior': {'limits': [-3., 2.]},
                        'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.05},
                        'proposal': 0.05,
                        'latex': r'w_{a}',
                        'delta': 0.2}
    params['Omega_k'] = {'prior': {'limits': [-0.3, 0.3]},
                         'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.01},
                         'proposal': 0.01,
                         'latex': r'\Omega_\mathrm{k}',
                         'delta': 0.05}
    #params['zrei'] = {'latex': r'z_\mathrm{reio}'}
    #params['YHe'] = {'latex': r'Y_\mathrm{P}'}
    params['sigma8_m'] = {'derived': True, 'latex': r'\sigma_8'}
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
    if 'pk' in get_parameterization(dataset):
        fix('tau_reio')
        params.pop('theta_MC_100')
        params['H0'] = {'prior': {'limits': [20, 100]},
                        'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
                        'latex': r'H_0',
                        'delta': 3.}
    if 'cmb' not in get_parameterization(dataset): # background only
        for name in ['logA', 'n_s', 'tau_reio']: fix(name)
        for name, di in list(params.items()):  # remove derived perturb quantities
            if any(n in name for n in ['sigma8', 's8']): params.pop(name)
        params.pop('theta_MC_100')
        params['H0'] = {'prior': {'limits': [20, 100]},
                        'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
                        'latex': r'H_0',
                        'delta': 3.}
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
        mean = np.array([0.02196, 3.034])
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

    from bindings_bao import DESICompressedBAOLikelihood
    from bindings_fs_bao import DESIFSLikelihood

    likelihood_renames, likelihood_mapping = {}, {}
    tracers = {'all': None, 'all_nolya': ['bgs_z0', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'elg_z1', 'qso_z0'], 'bgs': ['bgs_z0'], 'lrg': ['lrg_z0', 'lrg_z1', 'lrg_z2'], 'elg': ['elg_z1'], 'qso': ['qso_z0'], 'lya': ['lya_z0']}
    ## DESI BAO
    for tracer in ['all', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrgpluselg', 'elg', 'qso', 'lya']:
        #likelihood_mapping[f"desi-bao-{tracer.replace('_', '-')}"] = (DESICompressedBAOLikelihood, {'tracers': tracers.get(tracer, [tracer])})
        likelihood_mapping[f"desi-bao-{tracer}".replace('_', '-')] = (DESIFSLikelihood, {'tracers': tracers.get(tracer, [tracer]), 'observable_name': 'post-bao', 'data_name': ''})
    ## DESI FS
    for observable, observable_name in zip(['fs', 'fs_bao', 'fs_corr_recon', 'shapefit', 'shapefit_bao', 'shapefit_nodm_bao', 'shapefit_nodf_bao', 'shapefit_noqap_bao', 'standard_bao'],
                                           ['power', 'power+bao-recon', 'power+correlation-recon', 'shapefit', 'shapefit+bao-recon', 'shapefit-nodm+bao-recon', 'shapefit-nodf+bao-recon', 'shapefit-noqap+bao-recon', 'standard+bao-recon']):
        for data in ['mock-cmblens-', '']:
            for tracer in ['all', 'all_nolya', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'elg', 'qso', 'lya']:
                #likelihood_mapping[f"desi{data}-{observable}-{tracer}".replace('_', '-')] = (DESIFSLikelihood, {'tracers': tracers.get(tracer, [tracer]), 'observable_name': observable_name, 'data_name': 'data'})
                if 'power' in observable_name:
                    for theory in ['lptvelocileptors', 'reptvelocileptors', 'folpsax', 'folpsv2']:
                        likelihood_mapping[f"{data}desi-{theory}-{observable}-{tracer}".replace('_', '-')] = (DESIFSLikelihood, {'tracers': tracers.get(tracer, [tracer]), 'theory_name': theory, 'observable_name': observable_name, 'data_name': data[:-1], 'solve': solve})
                else:
                    likelihood_mapping[f"{data}desi-{observable}-{tracer}".replace('_', '-')] = (DESIFSLikelihood, {'tracers': tracers.get(tracer, [tracer]), 'observable_name': observable_name, 'data_name': data[:-1], 'solve': solve})


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
    
    from desilike.likelihoods.cmb import (TTHighlPlanck2018PlikLikelihood, TTHighlPlanck2018PlikLiteLikelihood, TTHighlPlanck2018PlikUnbinnedLikelihood,
                                      TTTEEEHighlPlanck2018PlikLikelihood, TTTEEEHighlPlanck2018PlikLiteLikelihood, TTTEEEHighlPlanck2018PlikUnbinnedLikelihood,
                                      LensingPlanck2018ClikLikelihood, TTLowlPlanck2018ClikLikelihood, EELowlPlanck2018ClikLikelihood)
    
    # the official 2018 clik likelihoods
    likelihood_mapping['planck2018-lowl-TT-clik'] = TTLowlPlanck2018ClikLikelihood
    likelihood_mapping['planck2018-lowl-EE-clik'] = EELowlPlanck2018ClikLikelihood
    # native python implementation
    #likelihood_mapping['planck2018-lowl-TT'] = 'planck_2018_lowl.TT'
    #likelihood_mapping['planck2018-lowl-EE'] = 'planck_2018_lowl.EE'
    # plikHM high-temperature
    likelihood_mapping['planck2018-highl-plik-TT'] = TTHighlPlanck2018PlikLikelihood
    # plikHM temperature+polarization
    likelihood_mapping['planck2018-highl-plik-TTTEEE'] = TTTEEEHighlPlanck2018PlikLikelihood
    # planck 2018 CamSpec likelihoods
    #likelihood_mapping['planck2018-highl-CamSpec-TT-clik'] = 'planck_2018_highl_CamSpec.TT'
    #likelihood_mapping['planck2018-highl-CamSpec-TTTEEE-clik'] = 'planck_2018_highl_CamSpec.TTTEEE'
    # native python implementation - planck 2018 CamSpec likelihoods
    #likelihood_mapping['planck2018-highl-CamSpec-TT'] = 'planck_2018_highl_CamSpec.TT_native'
    #likelihood_mapping['planck2018-highl-CamSpec-TTTEEE'] = 'planck_2018_highl_CamSpec.TTTEEE_native'
    # official clik code lensing
    likelihood_mapping['planck2018-lensing-clik'] = LensingPlanck2018ClikLikelihood
    # native python implementation
    #likelihood_mapping['planck2018-lensing'] = 'planck_2018_lensing.native'
    likelihood_mapping['planck2018-highl-plik-TTTEEE-lite'] = TTTEEEHighlPlanck2018PlikLiteLikelihood

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


def get_desilike_output(model='base', theory='class', dataset='desi-fs-bao-all', sampler='mcmc', add='', remove='', check_exists=False, base_dir=base_dir, run='auto', suffix=None, sort_dataset=sort_dataset, **kwargs):
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
        for run in ['run4', 'run3']:
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
    chain.set(chain['Omega_m'].clone(value=chain['sigma8_m'] * (chain['Omega_m'] / 0.3)**0.5, param={'basename': 'S8', 'derived': True, 'latex': r'S_8'}))

    if getdist:
        params = [str(param) for param in chain.params() if param.varied or (param.derived and not param.solved)]
        #params = [param for param in params if 'loglikelihood' not in param and 'logprior' not in param and 'logposterior' not in param]
        samples = chain.to_getdist(params=params, label=label)
        add_derived_getdist(samples)
        chain = samples

    return chain


def emulate_desilike(model='base', theory='class', dataset='desi-fs-bao-all', get_desilike_likelihoods=get_desilike_likelihoods, get_desilike_cosmo=get_desilike_cosmo, **kwargs):

    cosmo = get_desilike_cosmo(model=model, engine=theory, dataset=dataset)
    get_desilike_likelihoods(dataset=dataset, save_emulator=True, emulator_fn=model, cosmo=cosmo, **kwargs)


def measure_speed(model='base', theory='class', dataset='desi-fs-bao-all', niterations=5, get_desilike_likelihoods=get_desilike_likelihoods, get_desilike_cosmo=get_desilike_cosmo, **kwargs):
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


def sample_desilike(model='base', theory='class', dataset='desi-fs-bao-all', get_desilike_likelihoods=get_desilike_likelihoods, test=False, resume=False, sampler='mcmc', get_parameterization=get_parameterization, get_desilike_output=get_desilike_output, get_desilike_cosmo=get_desilike_cosmo, run='run0', **kwargs):
    import os
    from desilike.samplers import MCMCSampler, NUTSSampler, ZeusSampler, PocoMCSampler
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
    if sampler == 'cobaya':
        covariance = [get_desilike_output(model=model, theory=theory, dataset=['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10'], sampler='nuts', suffix=ichain, run='run2', **{**kwargs, 'emulator_fn': model}) for ichain in range(chains)]
        covariance = Chain.concatenate([Chain.load(fn).remove_burnin(0.5) for fn in covariance]).covariance(return_type=None)
        covmat_params = covariance.names()
        covmat = covariance.view(params=covmat_params)
        from desilike.bindings.cobaya import CobayaLikelihoodFactory
        likelihood_cobaya = CobayaLikelihoodFactory(lambda: likelihood, cosmo=None, params=True)
        sampler = {'mcmc': {'proposal_scale': 2.4, 'max_samples': 10000, 'Rminus1_stop': 0.02, 'Rminus1_cl_stop': 1., 'covmat': covmat, 'covmat_params': covmat_params}}
        info = {'likelihood': {'my_likelihood': likelihood_cobaya}, 'sampler': sampler, 'output': None}
        from cobaya.run import run
        updated_info, sampler = run(info)
        from desilike.samples import Chain
        samples = sampler.products()['sample']
    elif sampler == 'mcmc':
        covariance = [get_desilike_output(model=model, theory=theory, dataset=['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10'], sampler='nuts', suffix=ichain, run='run2', **{**kwargs, 'emulator_fn': model}) for ichain in range(chains)]
        covariance = Chain.concatenate([Chain.load(fn).remove_burnin(0.5) for fn in covariance])
        sampler = MCMCSampler(likelihood, chains=chains, covariance=covariance, oversample_power=0.4, drag=False, learn={'max_eigen_gr': 2., 'every': '40 * ndim'}, seed=42, save_fn=save_fn)
    elif sampler == 'nuts':
        sampler = NUTSSampler(likelihood, chains=chains, seed=42, save_fn=save_fn, ref_scale=0.01)
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


def profile_desilike(model='base', theory='camb', dataset='desi-fs-bao-all', start=None, get_desilike_likelihoods=get_desilike_likelihoods, get_desilike_cosmo=get_desilike_cosmo, get_desilike_output=get_desilike_output, run='run0', test=False, **kwargs):
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
    profiles = profiler.maximize(niterations=likelihood.mpicomm.size)
    #profiler.interval(params=cosmo.varied_params)
    #profiler.profile(params=cosmo.varied_params)
    #profiler.contour(params=cosmo.varied_params)
    #profiler.interval(['Mb'])
    print(profiles.to_stats(tablefmt='pretty'))
    for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
        profiles.to_stats(tablefmt=tablefmt, fn=os.path.splitext(output)[0] + '_stats.' + ext)


def prepare_pte(model='base', theory='camb', dataset='desi-fs-bao-all', get_desilike_likelihoods=get_desilike_likelihoods, get_desilike_output=get_desilike_output, get_desilike_cosmo=get_desilike_cosmo, load_cobaya_samples=load_cobaya_samples, run='run0', thin=10, skip=0.5, **kwargs):
    from desilike.samples import Chain
    from desilike import vmap
    
    cosmo = get_desilike_cosmo(model=model, engine=theory, dataset=dataset)
    likelihood = get_desilike_likelihoods(dataset=dataset, cosmo=cosmo, solve='.marg', covsyst='hod', **kwargs)[0]
    #likelihood = get_desilike_likelihoods(dataset=dataset, cosmo=cosmo, solve='.marg', jit=False, **kwargs)[0]
    likelihood()

    list_samples = load_cobaya_samples(model=model, theory=theory, dataset=dataset, run=run, thin=thin, skip=skip, **kwargs)

    list_samples = Chain.from_getdist(list_samples)
    mpicomm = likelihood.mpicomm
    for ichain, samples in enumerate(list_samples):
        for name in samples.names(name='*.likelihood'):
            samples['cobaya.{}'.format(name)] = samples.pop(name)
        samples['cobaya.logposterior'] = samples.pop('logposterior')
        output = get_desilike_output(model=model, theory=theory, dataset=dataset, sampler='reprocessing', run=run, suffix=ichain)
        for name, rename in {'ns': 'n_s', 'ombh2': 'omega_b', 'omch2': 'omega_cdm'}.items(): samples[rename] = samples.pop(name)
        params = [param for param in samples.params() if param in likelihood.varied_params]
        for param in params:
            samples[param] = samples[param].clone(param=likelihood.varied_params[param])

        vlikelihood = vmap(likelihood, backend='mpi', errors='return', return_derived=True)
        (_, derived), errors = vlikelihood(samples.to_dict(params=params) if mpicomm.rank == 0 else None)
        #derived['logposterior'] = derived['loglikelihood'][()] + derived['logprior'][()]

        if mpicomm.rank == 0:
            samples.update(derived)
            if likelihood.name:
                samples._logprior = likelihood.name + '.logprior'
                samples._loglikelihood = likelihood.name + '.loglikelihood'
            samples['logposterior'] = samples[samples._loglikelihood][()] + samples[samples._logprior][()]
            samples.attrs['size'] = likelihood.size
            samples.save(output)


def print_pte(model='base', theory='camb', dataset='desi-fs-bao-all', run='run0'):
 
    def pte(self, factor=1.):
        # self is samples
        from scipy import special
        params = self.params(solved=True)
        if self.params(solved=True):
            self = self.sample_solved()
        nd = self.attrs['size']
        integrand = 1. - special.gammainc(nd / 2., - self[self._loglikelihood][()] * factor)
        return integrand.mean()

    if not isinstance(dataset, str):
        dataset = sort_dataset(dataset)
        dataset = '_'.join(dataset)

    from desilike.samples import Chain
    samples = []
    for ichain in range(4):
        try:
            samples.append(load_desilike_samples(model=model, theory=theory, dataset=dataset, sampler='reprocessing', run=run, skip=0., getdist=False, ichain=ichain))
        except:
            pass
    samples = Chain.concatenate(samples)
    profiles = load_desilike_samples(model=model, theory=theory, run='run4', sampler='minuit', emulator_fn=False, dataset=dataset)

    def percival(nobs: int, nbins: int, nparams: int):
        A = 2 / (nobs - nbins - 1) / (nobs - nbins - 4)
        B = (nobs - nbins - 2) / (nobs - nbins - 1) / (nobs - nbins - 4)
        return (1 + B * (nbins - nparams)) / (1 + A + B * (nparams + 1))

    nd, nparams = profiles.bestfit.attrs['size'], profiles.bestfit.attrs['nvaried']
    if 'all' in dataset:
        nd = 72
        nparams += 2 * 6 + 2
    else:
        nparams += 2 + int('elg' in dataset) + int('qso' in dataset) # 2 parameters for rotation, 1 for photometric systematics
    factor = percival(1000, nd, 7)  # percival factor we applied
    nd = samples.attrs['size'] = profiles.bestfit.attrs['size']
    
    chi2 = -2. * profiles.bestfit.logposterior.max() * factor

    from scipy import stats
    import numpy as np

    print('=' * 40)
    print('For {}'.format(dataset))
    print('chi2 / ndof = {chi2:.0f} / ({nd:d} - {np:d}) = {rchi2:.2f}'.format(chi2=chi2, nd=nd, np=nparams, rchi2=chi2 / (nd - nparams)))

    pvalue = 1. - stats.chi2.cdf(chi2, nd - nparams)
    print('p-value: {:.2f}'.format(pvalue))
    size = samples.size
    nsamples = 4
    ptes = [pte(samples[i * size // nsamples:(i + 1) * size // nsamples], factor) for i in range(nsamples)]
    pte = pte(samples, factor)
    print('Bayesian PTE: {:.2f} \pm {:.2f}'.format(pte, np.std(ptes, ddof=1) / nsamples**0.5))
    print('=' * 40)


if __name__ == '__main__':

    #todo = ['runs']
    todo = ['test2']
   
    if 'runs' in todo:
        dataset = ['desi-abacus-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']
        config = dict(dataset=dataset, model='base_w_wa')
        sample_cobaya(**config, sampler='evaluate')
    
    if 'test' in todo:
        dataset = 'desi-reptvelocileptors-fs-bao-all'
        theory = 'camb'
        model = 'base'

        from cobaya.model import get_model
        from cobaya.sampler import get_sampler

        info = get_cobaya_info(model=model, theory=theory, dataset=[dataset], sampler='evaluate')
        sampler = info.pop('sampler')
        print(info)
        exit()
        model1 = get_model(info)
        info['likelihood']['desi_y1_cosmo_bindings.cobaya_bindings.desi_fs_bao_all'].update(jit=False)
        model2 = get_model(info)
        for p in [{'logA': 3., 'H0': 69., 'pre_ELG_z1.b1p': 0.6},
                  {'logA': 3.1, 'H0': 70., 'pre_LRG_z1.b1p': 0.7},
                  {'logA': 3.3, 'H0': 71., 'pre_LRG_z1.b1p': 0.8}]:
            params = {'logA':3.057147, 'ns':0.9657119, 'H0':70., 'ombh2':0.02246306, 'omch2':0.1184775, 'pre_QSO_z0.b1p':0.8900941, 'pre_QSO_z0.b2p':-0.05521521, 'pre_QSO_z0.bsp':0.1505953, 'pre_ELG_z1.b1p':0.599127, 'pre_ELG_z1.b2p':-0.07168614, 'pre_ELG_z1.bsp':-0.5048193, 'pre_LRG_z2.b1p':1.11637, 'pre_LRG_z2.b2p':-0.3405483, 'pre_LRG_z2.bsp':1.314247, 'pre_LRG_z1.b1p':1.150648, 'pre_LRG_z1.b2p':0.3174276, 'pre_LRG_z1.bsp':0.3973257, 'pre_LRG_z0.b1p':1.152034, 'pre_LRG_z0.b2p':-0.5777709, 'pre_LRG_z0.bsp':1.321482, 'pre_BGS_z0.b1p':1.080688, 'pre_BGS_z0.b2p':-0.0187345, 'pre_BGS_z0.bsp':0.8893648}
            logp1 = model1.logposterior({**params, **p})
            print(logp1)
            logp2 = model2.logposterior({**params, **p})
            print(logp2)
            print(logp1, logp2)

    if 'test2' in todo:
        tracer = 'all'
        dataset = 'desi-reptvelocileptors-fs-bao-{}'.format(tracer.lower().replace('_', '-'))
        theory = 'camb'
        model = 'base'

        from cobaya.model import get_model
        from cobaya.sampler import get_sampler

        info = get_cobaya_info(model=model, theory=theory, dataset=[dataset], sampler='evaluate')
        sampler = info.pop('sampler')
        #info['theory']['camb']['extra_args']['num_massive_neutrinos'] = 0
        #info['params']['mnu'] = 0.

        info['theory']['desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods2.reptvelocileptors'] = info['theory'].pop('desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods.reptvelocileptors')
        info['likelihood']['desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods2.desi_fs_bao_{}'.format(tracer.lower())] = info['likelihood'].pop('desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods.desi_fs_bao_{}'.format(tracer.lower()))
        info['likelihood']['desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods2.desi_fs_bao_{}'.format(tracer.lower())].update(solve='best')
        model1 = get_model(info)

        del info['theory']['desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods2.reptvelocileptors']
        info['likelihood']['desi_y1_cosmo_bindings.cobaya_bindings.desi_fs_bao_{}'.format(tracer.lower())] = info['likelihood'].pop('desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods2.desi_fs_bao_{}'.format(tracer.lower()))
        info['likelihood']['desi_y1_cosmo_bindings.cobaya_bindings.desi_fs_bao_{}'.format(tracer.lower())].pop('data_name')
        model2 = get_model(info)

        info['theory']['cosmoprimo.bindings.cobaya.cosmoprimo'] = info['theory'].pop('camb')
        info['theory']['cosmoprimo.bindings.cobaya.cosmoprimo']['engine'] = 'camb'
        for name in ['DHBBN', 'Y_p']:
            del info['params'][name]
        model3 = get_model(info)

        cosmo = get_desilike_cosmo(model=model, engine=theory, dataset=dataset)
        cosmo.init.update(extra_params={'bbn_predictor': 'PArthENoPE_880.2_standard.dat'})
        #cosmo.init.params.pop('m_ncdm')
        likelihood = get_desilike_likelihoods(dataset=dataset, cosmo=cosmo, solve='.best', jit=False)[0]
        likelihood()

        for p in [{}]:
            #params = {'logA':3.057147, 'ns':0.9657119, 'H0':70., 'ombh2':0.02246306, 'nnu':3.044, 'omch2':0.1184775, 'pre_{}.b1p'.format(tracer):1.152034, 'pre_{}.b2p'.format(tracer):-0.5777709, 'pre_{}.bsp'.format(tracer):1.321482}
            params = {'logA':3.057147, 'ns':0.9657119, 'H0':70., 'ombh2':0.02246306, 'omch2':0.1184775, 'pre_QSO_z0.b1p':0.8900941, 'pre_QSO_z0.b2p':-0.05521521, 'pre_QSO_z0.bsp':0.1505953, 'pre_ELG_z1.b1p':0.599127, 'pre_ELG_z1.b2p':-0.07168614, 'pre_ELG_z1.bsp':-0.5048193, 'pre_LRG_z2.b1p':1.11637, 'pre_LRG_z2.b2p':-0.3405483, 'pre_LRG_z2.bsp':1.314247, 'pre_LRG_z1.b1p':1.150648, 'pre_LRG_z1.b2p':0.3174276, 'pre_LRG_z1.bsp':0.3973257, 'pre_LRG_z0.b1p':1.152034, 'pre_LRG_z0.b2p':-0.5777709, 'pre_LRG_z0.bsp':1.321482, 'pre_BGS_z0.b1p':1.080688, 'pre_BGS_z0.b2p':-0.0187345, 'pre_BGS_z0.bsp':0.8893648}
            params = {**params, **p}
            logp1 = model1.logposterior(params)
            derived1 = dict(zip(model1.derived_params, logp1.derived))

            logp2 = model2.logposterior(params)
            derived2 = dict(zip(model2.derived_params, logp2.derived))

            logp3 = model3.logposterior(params)
            derived3 = dict(zip(model3.derived_params, logp3.derived))

            for name, rename in {'ns': 'n_s', 'ombh2': 'omega_b', 'omch2': 'omega_cdm', 'mnu': 'm_ncdm', 'nnu': 'N_eff'}.items():
                if name in params: params[rename] = params.pop(name)
            logp0, derived0 = likelihood(params, return_derived=True)
            
            #tmp = -0.5 * logp1['chi2__desi_y1_cosmo_bindings.cobaya_likelihoods.fs_bao_likelihoods.desi_fs_bao_{}'.format(dataset)]
            for name in ['omega_b', 'n_s']:
                logp0 -= likelihood.all_params[name].prior(params[name])

            print('logp', logp0, logp1.loglikes, logp2.loglikes, logp3.loglikes)
            derived0 = {name: float(derived0[name][()]) for name in derived0.names()}
            s0, s1, s2, s3 = 0., 0., 0., 0.
            for name in ['BGS_z0', 'LRG_z0', 'LRG_z1', 'LRG_z2', 'ELG_z1', 'QSO_z0']:
                print(name, derived0[name + '.loglikelihood'], derived1[name + '.loglikelihood'], derived2[name + '.loglikelihood'], derived3[name + '.loglikelihood'])
                s0 += derived0[name + '.loglikelihood']
                s1 += derived1[name + '.loglikelihood']
                s2 += derived2[name + '.loglikelihood']
                s3 += derived3[name + '.loglikelihood']
            print(s0, s1, s2, s3)
                