"""
Run cosmological inference with Y1 BAO.
Authors:
- Uendert Andrade
- Arnaud de Mattia
- Hanyu Zhang
- Kushal Lodha
- Nhat-Minh Nguyen

Configs (datasets, models) are listed in :func:`yield_configs`: these correspond to the cobaya samples available on disk.
To load cobaya samples, use :func:`load_cobaya_samples` in getdist format.
To add a source where the find samples, look at :func:`get_cobaya_output`. Just add the new source, following ['hanyuz', 'kushal', 'jiamingp'].
To add aliases for parameter names, look at ``list_renames`` in :func:`load_cobaya_samples`. There you can also set default derived parameters.

See ``y1_bao_cosmo.py`` for the script running chains with :mod:`desipipe`.
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


base_dir = Path('/global/cfs/cdirs/desicollab/science/cpe/y1kp7')


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
    for tracer in ['all', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrgpluselg', 'elg', 'qso', 'lya']:
        likelihood_mapping[f"mock-fiducial-desi-bao-{tracer.replace('_', '-')}"] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.bao_likelihoods_mock_fiducial.desi_bao_{tracer}': {'python_path': local_pythonpath, 'path': local_path}}
    likelihood_mapping[f"forecast-desiy5-bao-all"] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.bao_likelihoods.forecast_desiy5_bao_all': {'python_path': local_pythonpath, 'path': local_path}}
    
    ## SDSS BAO
    likelihood_mapping['sdss-bao-dr7-mgs'] = 'bao.sdss_dr7_mgs'
    likelihood_mapping['sdss-bao-dr12-lrg'] = 'bao.sdss_dr12_lrg_bao_dmdh'
    likelihood_mapping['sdss-bao-dr16-lrg'] = 'bao.sdss_dr16_lrg_bao_dmdh'

    ## DESI x eBOSS Lya BAO
    tracer = 'lya'
    likelihood_mapping[f'desi-eboss-bao-{tracer}'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.bao_likelihoods.desi_eboss_bao_{tracer}': {'python_path': local_pythonpath, 'path': local_path}}
    likelihood_mapping[f'fiducial-desi-eboss-bao-{tracer}'] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.bao_likelihoods_mock_fiducial.desi_eboss_bao_{tracer}': {'python_path': local_pythonpath, 'path': local_path}}
    
    ## Best BAO
    likelihood_renames['desi-sdss-bao-best'] = ['sdss-bao-dr7-mgs', 'sdss-bao-dr12-lrg', 'desi-bao-lrg-z1', 'desi-bao-lrgpluselg', 'desi-bao-elg', 'desi-bao-qso', 'desi-eboss-bao-lya']
    
    ## v1.5
    ## DESI BAO
    for tracer in ['all', 'bgs', 'lrg', 'lrg_z0', 'lrg_z1', 'lrgpluselg', 'elg', 'qso', 'lya']:
        likelihood_mapping[f"desi-v1.5-bao-{tracer.replace('_', '-')}"] = {f'desi_y1_cosmo_bindings.cobaya_likelihoods.bao_likelihoods_v1_5.desi_bao_{tracer}': {'python_path': local_pythonpath, 'path': local_path}}
    
    ## Best BAO
    likelihood_renames['desi-sdss-bao-best'] = ['sdss-bao-dr7-mgs', 'sdss-bao-dr12-lrg', 'desi-bao-lrg-z1', 'desi-bao-lrgpluselg', 'desi-bao-elg', 'desi-bao-qso', 'desi-eboss-bao-lya']
    
    ## BBN-Schoneberg
    likelihood_mapping['schoneberg2024-bbn'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.bbn_likelihoods.schoneberg2024': {'python_path': local_pythonpath}}
    likelihood_mapping['schoneberg2024-bbn-fixed-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.bbn_likelihoods.schoneberg2024_fixed_nnu': {'python_path': local_pythonpath}}

    ## Planck2018 CMB priors on theta_star and rdrag
    likelihood_mapping['planck2018-thetastar-fixed-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.thetastar_planck2018_fixed_nnu': {'python_path': local_pythonpath}}
    likelihood_mapping['planck2018-thetastar-varied-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.thetastar_planck2018_varied_nnu': {'python_path': local_pythonpath}}
    likelihood_mapping['planck2018-thetastar-fixed-marg-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.thetastar_planck2018_fixed_marg_nnu': {'python_path': local_pythonpath}}
    likelihood_mapping['planck2018-rdrag-fixed-nnu'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.rdrag_planck2018_fixed_nnu': {'python_path': local_pythonpath}}
    #likelihood_mapping['planck2018-rdrag-shifted'] = {'desi_y1_cosmo_bindings.cobaya_likelihoods.cmb_likelihoods.rdrag_planck2018_shifted': {'python_path': local_pythonpath}}

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
    likelihood_mapping['act-dr6-lensing'] = {'act_dr6_lenslike_v1_1.ACTDR6LensLike': {'lens_only': False, 'variant': 'act_baseline', 'lmax': 4000, 'version': 'v1.1'}}
    likelihood_mapping['planck-act-dr6-lensing'] = {'act_dr6_lenslike_v1_1.ACTDR6LensLike': {'lens_only': False, 'variant': 'actplanck_baseline', 'lmax': 4000, 'version': 'v1.1'}}
    likelihood_mapping['planck-act-dr6-lensing-v1.2'] = {'act_dr6_lenslike_v1_2.ACTDR6LensLike': {'lens_only': False, 'variant': 'actplanck_baseline', 'lmax': 4000, 'version': 'v1.2'}}
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
    return likelihoods


def get_parameterization(dataset):
    """Return parameterization type ('thermodynamics' when absolute BAO scale information, 'background' (no absolute BAO scale information), 'cmb', 'cmb-lensing')."""
    if not isinstance(dataset, str):
        dataset = '_'.join(dataset)
    if any(name in dataset for name in ['bbn', 'thetastar']):
        return 'thermodynamics'
    if 'planck2018-rdrag' in dataset:
        return 'background'
    if 'planck' in dataset:
        if 'lensing' in dataset:
            return 'cmb-lensing'
        return 'cmb'
    return 'background'


def get_cobaya_params(model='base', theory='camb', dataset='desi-bao-all', get_parameterization=get_parameterization):
    """Return cobaya ``params`` dictionary, given input model, theory and dataset."""
    params, extra, kw = {}, {}, {}
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
                                  'proposal': 0.0002, 'latex': r'100\theta_\mathrm{s}'}
        kw['renames'] = {'theta_s_100': '100*theta_s'}
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
        if 'ede' in model:
            #Rafaela: let's add some extra ede parameters
            params['fraction_axion_ac'] = {'prior': {'min': 0.0, 'max': 0.5},
                                  'ref': {'dist': 'norm', 'loc': 0.02, 'scale': 0.1},
                                  'proposal': 0.01,
                                  'latex': r'f_{\rm ede}(a_c)'}
            params['log10_axion_ac'] = {'prior': {'min': -4.5, 'max': -3.0},
                                  'ref': {'dist': 'norm', 'loc': -3.5, 'scale': 0.5},
                                  'proposal': 0.1,
                                  'latex': r'\log_{10}a_c'}
            params['log10_fraction_axion_ac'] = {'derived': 'lambda fraction_axion_ac: np.log10(fraction_axion_ac)'}
            params['f_ede'] = {'latex': r'f_{\rm ede}(a_{\rm peak})'}
            params['log10_z_c'] = {'latex': r'\log_{10}z_c'}
            params['log10_f_axion'] = {'latex': r'\log_{10}f_{\rm axion}'}
            params['log10_m_axion'] = {'latex': r'\log_{10}m_{\rm axion}'}
            params['scf_param_1'] = {'value': 2.719464130,
                            'latex': r'scf_1', 'drop': True}
            params['scf_parameters'] = {'value': 'lambda scf_param_1: str(scf_param_1)+", 0."', 'derived': False}

            extra['do_shooting'] = 'yes'
            extra['do_shooting_scf'] = 'yes'
            extra['scf_potential']='axion'
            extra['n_axion'] = 3
            extra['security_small_Omega_scf'] = 0.001
            extra['use_big_theta_scf'] = 'yes'
            extra['n_axion_security'] = 2.09
            extra['scf_has_perturbations'] = 'yes'
            extra['attractor_ic_scf'] = 'no'
            extra['scf_tuning_index'] = 0
            import os
            kw['ignore_obsolete'] = True
            #kw['path'] = os.path.join(os.environ['HOME'], 'cosmodesi/AxiCLASS/')

    else:  # camb

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
                params['smnu'] = {'value': 'lambda mnul, deltamnu21sq, deltamnu32sq: (mnul**2 - deltamnu32sq - deltamnu21sq)**0.5 + (mnul**2 - deltamnu32sq)**0.5 + mnul', 'drop': True, 'derived': False}
                value = 'lambda mnul, deltamnu21sq, deltamnu32sq, smnu: [2 * (mnul**2 - deltamnu32sq - deltamnu21sq)**0.5, smnu - 2 * (mnul**2 - deltamnu32sq - deltamnu21sq)**0.5]'
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
        #if False:  # TOREMOVE
        #    params.pop('theta_MC_100')
        #    params.pop('cosmomc_theta')
        #    params['H0'] = {'prior': {'min': 20, 'max': 100},
        #                    'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01},
        #                    'latex': r'H_0'}
        
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

    return params, extra, kw


def yield_configs(importance=False, models=None, **kwargs):
    """
    Yield configurations to be run, optionally those with importance sampling (``importance=True``).
    Optional ``kwargs`` are passed on to the returned config dictionary.

    See ``all_models`` below for all models considered.
    See ``datasets`` below for all datasets considered.
    """
    all_models = ['base', 'base_w', 'base_omegak', 'base_w_wa', 'base_omegak_w_wa', 'base_mnu', 'base_mnu_w', 'base_mnu_w_wa', 'base_nnu', 'base_nnu_w', 'base_nnu_w_wa', 'base_mnu059', 'base_mnu100', 'base_mnu_nh3', 'base_mnu_nhcamb', 'base_mnu_ih3', 'base_mnu_ihcamb', 'base_mnu_nh2_dmnu2fixed', 'base_mnu_ih2_dmnu2fixed']
    #all_models = ['base_nnu_w', 'base_nnu_w_wa']
        
    # Filter models based on interested_models argument
    if models is not None:
        # Check if all interested models are in the all_models list
        unrecognized_models = [model for model in models if model not in all_models]
        if unrecognized_models:
            raise ValueError(f"Unrecognized models: {', '.join(unrecognized_models)}. Please check the model names.")
        models = [model for model in all_models if model in models]
    else:
        models = all_models  # Use all models if no preference specified
    
    if importance:
        for config in yield_configs(importance=False, models=models, **kwargs):
            if 'TT' not in ''.join(config['dataset']): continue
            config['add'] = ['planck-act-dr6-lensing']
            yield config
    
    elif kwargs.get('sampler', 'cobaya') != 'cobaya':  # minimizer
        for model in models:
            for bao in ['desi-bao-all', 'desi-sdss-bao-best']:
                datasets = []
                if model in ['base']:
                    datasets += [[bao], [bao, 'schoneberg2024-bbn']]
                if model in ['base', 'base_mnu', 'base_mnu059', 'base_mnu100']:
                    datasets += [[bao, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                if model in ['base', 'base_w_wa']:
                    for sn in ['pantheonplus', 'union3', 'desy5sn']:
                        datasets += [[bao, sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                for dataset in datasets:
                    kwargs['add'] = ['planck-act-dr6-lensing'] if 'TT' in '_'.join(dataset) else []
                    yield dict(model=model, dataset=dataset, **kwargs)

    else:
        for model in models:
            #if model in ['base_mnu', 'base_mnu059', 'base_mnu100', 'base_mnu_nh3', 'base_mnu_nhcamb', 'base_mnu_ih3', 'base_mnu_ihcamb']:
            if model in ['base_mnu_nh2_dmnu2fixed', 'base_mnu_ih2_dmnu2fixed']:
                datasets = []
                datasets += [['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                datasets += [['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
                for dataset in datasets:
                    yield dict(model=model, dataset=dataset, **kwargs)
            else:
                nextensions = model.count('_')
                # SN datasets and lensing to be added as importance sampling
                datasets = []
                if model == 'base':
                    datasets += [['desi-bao-bgs'],
                                 ['desi-bao-lrg'],
                                 ['desi-bao-lrgpluselg'],
                                 ['desi-bao-elg'],
                                 ['desi-bao-qso'],
                                 ['desi-bao-lya'],
                                 ['desi-bao-bgs', 'schoneberg2024-bbn'],
                                 ['desi-bao-lrg', 'schoneberg2024-bbn'],
                                 ['desi-bao-lrgpluselg', 'schoneberg2024-bbn'],
                                 ['desi-bao-elg', 'schoneberg2024-bbn'],
                                 ['desi-bao-qso', 'schoneberg2024-bbn'],
                                 ['desi-bao-lya', 'schoneberg2024-bbn']]
                if model in ['base', 'base_w', 'base_omegak', 'base_w_wa', 'base_omegak_w_wa']:
                    datasets += [['desi-bao-all'],
                                 ['mock-fiducial-desi-bao-all'],
                                 ['desi-sdss-bao-best'],
                                 ['desi-bao-all', 'schoneberg2024-bbn'],
                                 ['desi-sdss-bao-best', 'schoneberg2024-bbn'],
                                 ['mock-fiducial-desi-bao-all', 'schoneberg2024-bbn'],
                                 ['desi-bao-all', 'schoneberg2024-bbn-fixed-nnu'],
                                 ['desi-sdss-bao-best', 'schoneberg2024-bbn-fixed-nnu'],
                                 ['mock-fiducial-desi-bao-all', 'schoneberg2024-bbn-fixed-nnu'],
                                 ['desi-bao-all', 'planck2018-thetastar-fixed-nnu'],
                                 ['desi-sdss-bao-best', 'planck2018-thetastar-fixed-nnu'],
                                 ['desi-bao-all', 'schoneberg2024-bbn', 'planck2018-thetastar-fixed-marg-nnu'],
                                 ['desi-sdss-bao-best', 'schoneberg2024-bbn', 'planck2018-thetastar-fixed-marg-nnu'],
                                 ['desi-bao-all', 'schoneberg2024-bbn-fixed-nnu', 'planck2018-thetastar-fixed-nnu'],
                                 ['desi-sdss-bao-best', 'schoneberg2024-bbn-fixed-nnu', 'planck2018-thetastar-fixed-nnu'],
                                 ['desi-bao-all', 'planck2018-rdrag-fixed-nnu'],
                                 ['desi-sdss-bao-best', 'planck2018-rdrag-fixed-nnu'],
                                 ['pantheonplus'],
                                 ['union3'],
                                 ['desy5sn'],
                                 ['desi-bao-all', 'pantheonplus'],
                                 ['desi-bao-all', 'union3'],
                                 ['desi-bao-all', 'desy5sn'],
                                 ['desi-bao-all', 'pantheonplus', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'],
                                 ['desi-bao-all', 'union3', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'],
                                 ['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                                 #['pantheonplus', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE'],
                                 #['union3', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']]
                    #datasets += [['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'],
                    #             ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']]
                if model in ['base_nnu']:
                    datasets += [['desi-bao-all'],
                                 ['desi-sdss-bao-best'],
                                 ['desi-bao-all', 'schoneberg2024-bbn'],
                                 ['desi-sdss-bao-best', 'schoneberg2024-bbn'],
                                 ['desi-bao-all', 'schoneberg2024-bbn', 'planck2018-thetastar-fixed-marg-nnu'],
                                 ['desi-sdss-bao-best', 'schoneberg2024-bbn', 'planck2018-thetastar-fixed-marg-nnu'],
                                 ['desi-bao-all', 'schoneberg2024-bbn', 'planck2018-thetastar-varied-nnu'],
                                 ['desi-sdss-bao-best', 'schoneberg2024-bbn', 'planck2018-thetastar-varied-nnu']]
                    

                datasets += [['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'],
                             #['desi-sdss-bao-best', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'],
                             ['desi-bao-all', 'pantheonplus', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'],
                             ['desi-bao-all', 'union3', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'],
                             ['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                # CamSpec
                if model in ['base', 'base_mnu', 'base_nnu', 'base_w_wa']:
                    datasets += [['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']]
                # BAO best
                if model in ['base', 'base_mnu']:
                    datasets += [['desi-sdss-bao-best', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                if model in ['base_w_wa']:
                    datasets += [['desi-sdss-bao-best', 'pantheonplus', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                    datasets += [['desi-sdss-bao-best', 'union3', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                    datasets += [['desi-sdss-bao-best', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                    datasets += [['pantheonplus', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'],
                                 ['union3', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'],
                                 ['desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                if model in ['base_mnu']:
                    datasets += [['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2020-hillipop-TTTEEE']]
                if model in ['base', 'base_w_wa']:
                    datasets += [['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2020-hillipop-TTTEEE']]
                    datasets += [['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']]
                    datasets += [['sdss-bao-dr7-mgs', 'sdss-bao-dr12-lrg', 'sdss-bao-dr16-lrg', 'desi-bao-lrgpluselg', 'desi-bao-elg', 'desi-bao-qso', 'desi-eboss-bao-lya', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
                for dataset in datasets:
                    yield dict(model=model, dataset=dataset, **kwargs)


def sort_dataset(dataset, yield_configs=yield_configs):
    """Sort input dataset."""
    for config in yield_configs(importance=False):
        if set(dataset) == set(config['dataset']):
            return config['dataset']
    return dataset


def get_cobaya_output(model='base', theory='camb', dataset='desi-bao-all', sampler='cobaya', add='', remove='', source=None, check_exists=False, base_dir=base_dir, run='run1', suffix=True, sort_dataset=sort_dataset):
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
        If 'auto', try to find available chains, from ['main', 'hanyuz', 'kushal', 'jiamingp', 'jshree', 'avijit', 'ndeiosso'].
        Else, one source from this list.

    run : str, default='run1'
        Runs in the Key Paper are 'run1' (v1.2 BAO). 'run3' for the planck-act-dr6-lensing likelihood.
        'v1.5_imp' for the v1.5 BAO.

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
    if run == 'run0':
        main_dir = base_dir / 'kp4'
        dataset = dataset.replace('desy5sn', 'desy5')
        if source is None: source = 'auto'
    else:
        main_dir = base_dir / 'bao'
        if source is None: source = 'main'
    
    bestfit = 'cobaya' not in sampler
    if sampler.endswith('-likelihood'):
        bestfit = 'likelihood'
        sampler = sampler.replace('-likelihood', '')
    if sampler.endswith('-posterior'):
        bestfit = 'posterior'
        sampler = sampler.replace('-posterior', '')

    output = str(main_dir / f'{"cobaya" if "cobaya" in sampler else sampler}/{theory}/{run}/{model}/{dataset}{importance}')
    if bestfit:
        output = output + '/bestfit'
    else:
        output = output + '/chain'
    if suffix:
        if bestfit:
            if bestfit is not True:
                output = output + ('.minimum' if bestfit == 'posterior' else '.bestfit')
        elif importance or '_add_' in output:
            output = output + '.post.importance'
    output_main = output
    
    def _get_cobaya_output_source(source='main', model=model, dataset=dataset, add=add, remove=remove):
        
        output = output_main

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

        # Add other sources here!
        if source == 'hanyuz':
            convert = {'BAO': ['desi-bao-all'], 'BestBAO': ['desi-sdss-bao-best'], 'Union3': ['union3'], 'PantheonPlus': ['pantheonplus'], 'plik_TTTEEE_lowE_lowl': ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], 'NPIPECamSpec': ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']}
            convert_add = {'actplanck_lensing': ['planck-act-dr6-lensing']}
            try: output = convert_dataset(dataset, convert) + convert_dataset(add, convert_add)
            except ValueError: output = []
            output = base_dir / f'hanyuz/cobaya_camb/{model}' / '_'.join(output)
            for output in glob.glob(str(output / 'chain*.txt')):
                output = output[:-6]
                break
        if source == 'kushal':
            convert = {'DESI': ['desi-bao-all'], 'DESI_SDSS_BAO_best': ['desi-sdss-bao-best'], 'planck2018': ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], 'planck_camspec': ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE'], 'union3': ['union3'], 'pantheonplus': ['pantheonplus']}
            convert_add = {'add_actplanck_lensing': ['planck-act-dr6-lensing']}
            try: output = convert_dataset(dataset, convert) + convert_dataset(add, convert_add)
            except ValueError: output = []
            output = base_dir / f'kushal/cobaya/{model}' / '_'.join(output)
            for output in glob.glob(str(output / '*.txt')):
                output = output[:-6]
                break
        if source == 'jiamingp':
            convert = {'desi-bao-gaussian': ['desi-bao-all'], 'CMBTP': ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], 'Union3': ['union3'], 'PantheonPlus': ['pantheonplus']}
            convert_add = {'CMBTPL': ['planck-act-dr6-lensing']}
            try:
                output = convert_dataset(dataset, convert)
                output = '_'.join(output)
                if convert_dataset(add, convert_add):
                    output = output.replace('CMBTP', 'CMBTPL')
            except ValueError:
                output = ''
            output = base_dir / f'jiamingp/cobaya/chains/{model}' / output
            for output in glob.glob(str(output / 'chain*.txt')):
                output = output[:-6]
                break
        if source == 'jshree':
            convert = {'DESI': ['desi-bao-all'], 'DESI_SDSS_BAO_best': ['desi-sdss-bao-best'], 'Planck18': ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], 'Union3': ['union3'], 'PantheonPlus': ['pantheonplus']}
            convert_add = {'Lensing': ['planck-act-dr6-lensing']}
            try: output = convert_dataset(dataset, convert) + convert_dataset(add, convert_add)
            except ValueError: output = []
            model = {'base': 'lcdm', 'base_omegak': 'base_k', 'base_mnu': 'mnucdm', 'base_w_wa': 'base_w0wa', 'base_nnu_mnu': 'Neff_mnucdm', 'base_omegak_w_wa': 'base_kw0wa'}.get(model, model)
            output = base_dir / f'jshree/cobaya/chains/{model}' / '_'.join(output)
            for output in glob.glob(str(output / '*.txt')):
                output = output[:-6]
                break
        if source == 'avijit':
            convert = {'DESI': ['desi-bao-all'], 'best_BAO': ['desi-sdss-bao-best'], 'Planck': ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], 'Union3': ['union3'], 'PantheonPlus': ['pantheonplus']}
            if model == 'base':
                convert['DESI_SDSS_BAO'] = convert.pop('best_BAO')
                convert['CMBTP'] = convert.pop('Planck')
            convert_add = {'CMBTPL': ['planck-act-dr6-lensing']}
            try:
                output = convert_dataset(dataset, convert)
                output = '_'.join(output)
                if convert_dataset(add, convert_add):
                    output = output.replace('CMBTP', 'CMBTPL')
            except ValueError:
                output = ''
            model = {'base': 'LCDM', 'base_omegak': 'kcdm', 'base_w_wa': 'w0waCDM'}.get(model, model)
            output = base_dir / f'avijit/cobaya/chains/{model}' / output
            for output in glob.glob(str(output / '*.txt')):
                output = output[:-6]
                break
        if source == 'ndeiosso':
            convert = {'Y1bao': ['desi-bao-all'], 'BestBao': ['desi-sdss-bao-best'], 'plik': ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], 'Union3': ['union3'], 'Pantheon': ['pantheonplus']}
            convert_add = {}
            try:
                output = convert_dataset(dataset, convert)
                output = '+'.join(output)
            except ValueError:
                output = ''
            model = {'base_mnu': 'base_mnu', 'base_mnu_nnu': 'base_mnu_Neff', 'base_mnu_w': 'base_w_mnu', 'base_nnu_w': 'base_Neff_w'}.get(model, model)
            output = base_dir / f'ndeiosso/cobaya/{model}' / output
            print(output)
            for output in glob.glob(str(output / '*.txt')):
                output = output[:-6]
                break   
        return str(output)
    
    if source != 'main' and run == 'run0':
        if source == 'auto':
            sources = ['main', 'hanyuz', 'kushal', 'jiamingp', 'jshree', 'avijit', 'ndeiosso']
        else:
            sources = [source]
        for source in sources:
            output = _get_cobaya_output_source(source)
            if bool(glob.glob(output + '*.txt')):
                return output
        raise ValueError('no samples found for model = {}, dataset = {}, add = {}!'.format(model, dataset, add))
    
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


def get_cobaya_covmat(model='base', theory='camb', dataset='desi-bao-all', base_dir=base_dir, get_parameterization=get_parameterization, get_cobaya_output=get_cobaya_output):
    """Return cobaya proposal matrix."""
    import os
    from pathlib import Path
    return None
    if 'cmb' not in get_parameterization(dataset):
        return None
    if not isinstance(dataset, str):
        dataset = '_'.join(dataset)
    #if 'desy5sn' in dataset:
    #    dataset = dataset.replace('desy5sn', 'union3')
    if 'desi-sdss-bao-best' in dataset:
        dataset = dataset.replace('desi-sdss-bao-best', 'desi-bao-all')
    if model == 'base_mnu_w' and any(dd in dataset for dd in ['pantheonplus', 'union3', 'desy5sn']):
        model = 'base_w'
    if model in ['base_mnu059', 'base_mnu100', 'base_mnu_nh3', 'base_mnu_nhcamb', 'base_mnu_ih3', 'base_mnu_ihcamb', 'base_mnu_nh3_dmnu2fixed', 'base_mnu_ih3_dmnu2fixed', 'base_mnu_nh2_dmnu2fixed', 'base_mnu_ih2_dmnu2fixed']:
        model = 'base_mnu'
    if '_planck-act-dr6-lensing-v1.2' in dataset:
        dataset = dataset.replace('_planck-act-dr6-lensing-v1.2', '_planck-act-dr6-lensing')
    #if '_planck-act-dr6-lensing' in dataset:
    #    dataset = dataset.replace('_planck-act-dr6-lensing', '')
    if 'forecast-desiy5-bao-all' in dataset:
        dataset = dataset.replace('forecast-desiy5-bao-all', 'desi-bao-all')
    if 'sdss-bao-dr7-mgs_sdss-bao-dr12-lrg_sdss-bao-dr16-lrg_desi-bao-lrgpluselg_desi-bao-elg_desi-bao-qso_desi-eboss-bao-lya' in dataset:
        dataset = dataset.replace('sdss-bao-dr7-mgs_sdss-bao-dr12-lrg_sdss-bao-dr16-lrg_desi-bao-lrgpluselg_desi-bao-elg_desi-bao-qso_desi-eboss-bao-lya', 'desi-bao-all')
    #if 'planck2020-hillipop-TTTEEE' in dataset:
    #    dataset = dataset.replace('planck2020-hillipop-TTTEEE', 'planck2018-highl-plik-TTTEEE')
    #if 'planck-NPIPE-highl-CamSpec-TTTEEE' in dataset:
    #    dataset = dataset.replace('planck-NPIPE-highl-CamSpec-TTTEEE', 'planck2018-highl-plik-TTTEEE')
    if 'planck2020-lollipop-lowlE' in dataset:
        dataset = dataset.replace('planck2020-lollipop-lowlE', 'planck2018-lowl-EE-clik')
    #run = 'run3'  # 'run0'
    run = 'run1'
    output = get_cobaya_output(model=model, theory=theory, dataset=dataset, sampler='cobaya', add='', remove='', source='auto', base_dir=base_dir, run=run, suffix=False)
    covmat = str(output) + '.covmat'
    assert os.path.exists(covmat), covmat
    return covmat
    """
    Shall be set to run0 paths for the next (final?) iteration.
    covmat = Path(base_dir) / 'covmats'
    if not isinstance(dataset, str):
        dataset = '_'.join(dataset)
    if 'cmb' not in get_parameterization(dataset):
        return None
    if theory == 'classy':
        covmat /= 'kushal/base_omegak_w_wa/desi-bao-gaussian_planck2018/_chains_DESIY1_cmb.covmat'
    else:  # camb
        covmat /= 'camb'
        if 'mnu' in model:
            covmat /= 'y1bao_plk_base_mnu.covmat'
        else:
            covmat /= 'y1bao_plk_base_w_wa.covmat'
    return str(covmat)
    """


def get_cobaya_info(model='base', theory='camb', dataset='desi-bao-all', sampler='cobaya', seed=None, save_fn=None, get_cobaya_likelihoods=get_cobaya_likelihoods, get_cobaya_params=get_cobaya_params, get_cobaya_covmat=get_cobaya_covmat, get_cobaya_output=get_cobaya_output, get_parameterization=get_parameterization, add=(), remove=(), temperature=1, debug=False, **kwargs):
    """Return cobaya info dictionary for sampling."""
    if isinstance(dataset, str): dataset = [dataset]
    else: dataset = list(dataset)
    if isinstance(add, str): add = [add]
    else: add = list(add)
    likelihoods = get_cobaya_likelihoods(dataset=dataset + add)
    likelihoods_remove = get_cobaya_likelihoods(dataset=remove)
    for likelihood in likelihoods_remove:
        likelihoods.pop(likelihood)
    params, extra_args, theory_kw = get_cobaya_params(model=model, theory=theory, dataset=dataset + add)

    info = {}
    output = get_cobaya_output(model=model, theory=theory, dataset=dataset, sampler=sampler, suffix='cobaya' in sampler, add=add, remove=remove, **kwargs)

    #info['stop_at_error'] = True  #TOREMOVE
    if 'evaluate' in sampler:
        sampler = {'evaluate': None}
        info['stop_at_error'] = True
        output = None
    elif sampler == 'cobaya':
        covmat = get_cobaya_covmat(model=model, theory=theory, dataset=dataset)
        if covmat is not None:
            print('Using covmat: {}'.format(covmat))
        is_cmb = 'cmb' in get_parameterization(dataset)
        sampler = {'mcmc': {'drag': is_cmb,
                            'oversample_power': 0.4,
                            'proposal_scale': 1.9,
                            'covmat': covmat,
                            'temperature': temperature,
                            'Rminus1_stop': 0.01, #0.01,
                            'Rminus1_cl_stop': 0.2 if is_cmb else 0.02,
                            #'learn_proposal_Rminus1_max': 30.,
                            #'Rminus1_stop': 0.1,
                            #'Rminus1_cl_stop': 1,
                            'seed': seed,
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
                if theory == 'camb':
                    extra_args['lAccuracyBoost'] = 1
                #if minimizer == 'iminuit':
                #    # default tol is 0.1
                #    sampler['minimize']['override_iminuit'] = {} # distance to minimum is 0.002 * tol * errordef, https://github.com/scikit-hep/iminuit/blob/298afadc645999fbad32ca00be6c0ebfec7c7bb9/src/iminuit/minuit.py#L44
                break
    if not isinstance(sampler, dict):
        raise ValueError('unknown sampler {}'.format(sampler))
    info.update({'theory': {theory: {'extra_args': extra_args} | theory_kw}, 'likelihood': likelihoods, 'params': params, 'sampler': sampler, 'output': output})
    if debug:
        info['debug'] = True
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


def sample_cobaya(resume=False, model='base', theory='camb', dataset='desi-bao-all', sampler='cobaya', get_cobaya_info=get_cobaya_info, print_margestats=print_margestats, **kwargs):
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

    import yaml
    print(yaml.dump(info, sort_keys=False))

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
    Output.reload_updated_info, Sampler.set_checkpoint_info = reload_updated_info, set_checkpoint_info
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
        updated_info, mcmc = run(info, force=not bool(resume), resume=bool(resume))
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


def profile_cobaya(model='base', theory='camb', dataset='desi-bao-all', sampler='iminuit', get_cobaya_info=get_cobaya_info, get_cobaya_output=get_cobaya_output, ignore_prior=True, add=(), remove=(), run='run1', **kwargs):
    info = get_cobaya_info(model=model, theory=theory, dataset=dataset, sampler=sampler, add=add, remove=remove, run=run, **kwargs)
    sampler = info.pop('sampler')
    sampler['minimize']['ignore_prior'] = ignore_prior
    import os
    import yaml
    print(yaml.dump({**info, 'sampler': sampler}, sort_keys=False))
    import numpy as np
    from cobaya.model import get_model
    from cobaya.sampler import get_sampler
    from cobaya.output import get_output
    prefix = info.pop('output')
    output = get_output(prefix=prefix, resume=False, force=True)

    dataset = [dd.replace('-zmin0.1', '') for dd in dataset]
    input = get_cobaya_output(model=model, theory=theory, dataset=dataset, add=add, remove=remove, source='main', sampler='cobaya')
    if input is not None and os.path.exists(os.path.dirname(input)):
        input = get_output(prefix=input, resume=False, force=False)
        input.get_updated_info = lambda *args, **kwargs: {}
        input.check_and_dump_info = lambda *args, **kwargs: None  # to avoid overwriting updated.yaml
    else:
        input = None
    #input = output
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
    output = get_cobaya_output(model=model, theory=theory, dataset=dataset, run='run1', **kwargs)
    output_importance = get_cobaya_output(model=model, theory=theory, dataset=dataset, add=add, remove=remove, suffix=False, run=run, **kwargs)

    likelihoods_add = get_cobaya_likelihoods(dataset=add)
    likelihoods_remove = get_cobaya_likelihoods(dataset=remove)

    info = {'output': output, 'post': {}}
    info['post']['output'] = output_importance
    info['post']['suffix'] = 'importance'
    info['post']['skip'] = skip
    #info['post']['thin'] = 4
    info['post']['thin'] = thin
    info['post']['add'] = {'likelihood': likelihoods_add}
    info['post']['remove'] = {'likelihood': likelihoods_remove}
    if 'lensing' in '_'.join(add):
        info['post']['add']['theory'] = {theory: {'extra_args': get_cobaya_params(model=model, theory=theory, dataset=add)[1]}}
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


def print_convergence(*output, what=None, max_gr=0.01, min_corr=0.05, skip=0.3, thin=1, short=True):
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
        if what is None:
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
            

def load_cobaya_samples(skip=None, thin=1, combined=False, add=(), remove=(), label=None, source='auto', sampler='cobaya', convergence=False, bestfit=None, **config):
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
        if any(name in output for name in ['hanyuz', 'kushal', 'jiamingp', 'jshree', 'avijit']): skip = 0.3
        else: skip = 0 if add or remove else 0.3
    if convergence:
        print_convergence(output, skip=skip, thin=thin)
    if 'bestfit' in output:
        import os
        suffix = '.minimum' if bestfit == 'posterior' else '.bestfit'
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
    collections[0].reset_temperature(with_batch=collections[1:])
    samples = collections[0].to_getdist(combine_with=collections[1:])
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

    base_dir = Path('/global/cfs/cdirs/desicollab/science/cpe/y1kp7/bao/sdss_dr16/')
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


def load_des_samples(skip=0.5, label=None):
    """
    Load DES (Y3 3x2pt) samples.
    
    Taken from https://des.ncsa.illinois.edu/releases/y3a2/Y3key-products
    """

    import numpy as np
    from getdist import MCSamples
    
    names = ['omegam', 'H0', 'sigma8']
    labels = ['$\\Omega_m$', '$H_0$', '$S_8$']
    fn = '/global/cfs/cdirs/desicollab/science/cpe/y1kp7/fs/des_y3/base/chain_3x2pt_lcdm_SR_maglim.txt'
    samples = np.loadtxt(fn, usecols=[0, 1, 31, -2, -1], comments='#')
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

def data_dir(version='v1.2'):
    return base_dir / 'bao' / 'data{}'.format('' if version == 'v1.2' else '_' + version)


def load_bao_chain(tracer, zrange, version='v1.2', return_profiles=False):
    """
    Load desilike bao chains for input tracer.

    Parameters
    ----------
    tracer : str
        Tracer name, e.g. 'ELG'.

    zrange : tuple, int
        Redshift range, or index or redshift range, e.g. for ``tracer='ELG'``, 1 for ``(1.1, 1.6)``.

    version : str, default='v1.2'
        Version.
    
    return_profiles : bool, default=False
        If ``True``, also return desilike.samples.Profiles

    Returns
    -------
    chain : desilike.samples.Chain
    """
    import os
    import numbers
    from desilike.samples import Chain, Profiles
    
    base_dir = data_dir(version=version)
    
    #di = {('BGS_BRIGHT-21.5', (0.1, 0.4)): 0.29523469494710347, ('LRG', (0.4, 0.6)): 0.5098040294582544, ('LRG', (0.6, 0.8)): 0.7055803366319268, ('LRG', (0.8, 1.1)): 0.9190285465077, ('LRG+ELG_LOPnotqso', (0.8, 1.1)): 0.9299393952106122, ('ELG_LOPnotqso', (0.8, 1.1)): 0.9550735465711971, ('ELG_LOPnotqso', (1.1, 1.6)): 1.3172765289974155, ('QSO', (0.8, 2.1)): 1.4909589720021144}

    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'LRG+ELG_LOPnotqso': [(0.8, 1.1)], 'QSO': [(0.8, 2.1)]}
    for tr, zr in list_zrange.items():
        if tracer == tr:
            # Try to match input tracer (which can be 'ELG') and zrange (which can be 1 for 2nd ELG redshift bin).
            if isinstance(zrange, numbers.Number):
                tracer, zrange = tr, zr[zrange]
                break
            else:
                tracer, zrange = tr, tuple(zrange)
                break

    def load_bao_chain(fi, burnin=0.5):
        tracer, zrange = fi[0].options['tracer'], fi[0].options['zrange']
        chains = [Chain.load(ff).remove_burnin(burnin)[::10].select(name=['qpar', 'qper', 'qiso', 'qap']) for ff in fi]
        chain = chains[0].concatenate(chains)
        eta = 1. / 3.
        if 'qpar' in chain and 'qper' in chain:
            chain.set((chain['qpar']**eta * chain['qper']**(1. - eta)).clone(param=dict(name='qiso', derived=True, latex=r'q_{\rm iso}')))
            chain.set((chain['qpar'] / chain['qper']).clone(param=dict(name='qap', derived=True, latex=r'q_{\rm ap}')))
        if 'qiso' in chain and 'qap' in chain:
            chain.set((chain['qiso'] * chain['qap']**(1. - eta)).clone(param=dict(name='qpar', derived=True, latex=r'q_{\parallel}')))
            chain.set((chain['qiso'] * chain['qap']**(-eta)).clone(param=dict(name='qper', derived=True, latex=r'q_{\perp}')))
        #chain.attrs['zeff'] = di[tracer, zrange]
        if version == 'v1.5': chain.attrs['zeff'] = float('{:.3f}'.format(chain.attrs['zeff']))
        
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

    chain_fn = base_dir / 'chain_bao_{}_GCcomb_z{:.1f}-{:.1f}.npy'.format(tracer, *zrange)
    profiles_fn = base_dir / 'profiles_bao_{}_GCcomb_z{:.1f}-{:.1f}.npy'.format(tracer, *zrange)
    if not os.path.exists(chain_fn) or not os.path.exists(profiles_fn):
        from desi_y1_files import get_data_file_manager, get_bao_baseline_fit_setup
        dfm = get_data_file_manager(conf='unblinded')
        options = dict(get_bao_baseline_fit_setup(tracer, zrange=zrange), version=version)
        fchains = list(dfm.select(id='chains_bao_recon_y1', **options, ignore=True))
        print(options, fchains[0])
        fprofiles = dfm.get(id='profiles_bao_recon_y1', **options, ignore=True)
        chain = load_bao_chain(fchains)
        if not os.path.exists(chain_fn): chain.save(chain_fn)
        profiles = fprofiles.load()
        if version == 'v1.5': profiles.attrs['zeff'] = float('{:.3f}'.format(profiles.attrs['zeff']))
        if not os.path.exists(profiles_fn): profiles.save(profiles_fn)
    chain = Chain.load(chain_fn)
    profiles = Profiles.load(profiles_fn)
    if return_profiles:
        return chain, profiles
    return chain


def load_bao_fisher(tracer, zrange, apmode=None, scale='distance', with_syst=True, version='v1.2', return_type='getdist'):
    import os
    import numpy as np
    from desilike import LikelihoodFisher
    
    base_dir = data_dir(version=version)
    
    fisher_fn = base_dir / 'gaussian_bao_{}_GCcomb_z{:.1f}-{:.1f}.npy'.format(tracer, *zrange)
    if with_syst and os.path.exists(fisher_fn):
        fisher = LikelihoodFisher.load(fisher_fn)
    else:
        if 'lya' in tracer.lower():
            dirname = '/global/cfs/cdirs/desicollab/science/lya/y1-kp6/iron-final-bao/final-result/'
            z, center = np.loadtxt(os.path.join(dirname, 'DESI-Y1-Lya.dat'), comments='#', usecols=[0, 1], unpack=True)
            z = z.flat[0]
            assert np.allclose(z, 2.33)
            assert center[0] < center[1]  # hacky way to check order: DH, DM
            cov = np.loadtxt(os.path.join(dirname, 'DESI-Y1-Lya.cov'), comments='#', usecols=[0, 1], unpack=True)
            quantities = ['qpar', 'qper']
            from cosmoprimo.fiducial import DESI
            from cosmoprimo import constants
            fiducial = DESI()
            DM_over_rd_fid = fiducial.comoving_angular_distance(z) / fiducial.rs_drag
            DH_over_rd_fid = (constants.c / 1e3) / (100. * fiducial.efunc(z)) / fiducial.rs_drag
            fid = np.array([DH_over_rd_fid, DM_over_rd_fid])
            center /= fid
            cov /= fid[:, None] * fid
            fisher = LikelihoodFisher(center=center, params=quantities, hessian=-np.linalg.inv(cov), attrs={'zeff': z})
            import pandas as pd
            #table = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'blinded-lya-correlations-y1-5-2-0-0-v2.csv'))
            table = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'combined-correlations-monopole-desi-y1.csv'))
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
            chain, profiles = load_bao_chain(tracer, zrange, version=version, return_profiles=True)
            iso = 'qpar' not in chain
            fisher = chain.to_fisher(params=['qiso'] if iso else ['qpar', 'qper'])
            if with_syst:
                covsyst_qisoqap = np.diag([0.245, 0.3])**2  # eq. 5.1 of https://fr.overleaf.com/project/645d2ce132ee6c4f6baa0ddd
                if iso:
                    covsyst = covsyst_qisoqap[:1, :1]
                else:
                    assert fisher.names() == ['qpar', 'qper']
                    eta = 1. / 3.
                    jac = np.array([[1., 1. - eta], [1., - eta]])  # ('qisoqap' -> 'qparqper')
                    #covsyst = np.linalg.inv(jac.dot(np.linalg.inv(covsyst_qisoqap)).dot(jac.T))
                    #print(jac)
                    covsyst = jac.dot(covsyst_qisoqap).dot(jac.T)
                    ref = np.array([[0.316**2, 0.478 * 0.264 * 0.316], [0.478 * 0.264 * 0.316, 0.264**2]])
                    #print(covsyst)
                    #print(ref)
                    #std = np.diag(covsyst)**0.5
                    #print(std, covsyst / (std[:, None] * std))
                covsyst *= 1e-4  # unit is percent
                fisher._hessian = np.linalg.inv(np.linalg.inv(fisher._hessian) - covsyst)
                if version == 'v1.5' and with_syst:
                    ref = LikelihoodFisher.load('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/gaussian_bao-recon_{}_GCcomb_z{:.1f}-{:.1f}.npy'.format(tracer, *zrange))
                    assert np.allclose(ref.mean(), fisher.mean())
                    assert np.allclose(ref.covariance(), fisher.covariance())
            fisher.attrs.update(chain.attrs)
            fisher.attrs.update(profiles.attrs)

        if with_syst:
            fisher.save(fisher_fn)

    from desi_y1_files.cosmo_tools import convert_bao_fisher;
    #print(1, fisher.mean())
    try:
        fisher = convert_bao_fisher(fisher, apmode=apmode, scale=scale)
    except ValueError as exc:
        if apmode == 'qisoqap':  # isotropic fit
            fisher = convert_bao_fisher(fisher, apmode='qiso', scale=scale)
        else:
            raise exc
    #print(2, 'f', fisher.mean(), fisher.attrs['zeff'])
    if return_type == 'getdist':
        fisher = fisher.to_getdist()
    return fisher


def get_veff():
    import numbers
    from desi_y1_files import get_data_file_manager, get_bao_baseline_fit_setup
    from y1_data_2pt_tools import get_footprint
    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8)], 'LRG+ELG_LOPnotqso': [(0.8, 1.1)], 'ELG_LOPnotqso': [(1.1, 1.6)], 'QSO': [(0.8, 2.1)]}
    
    def get_P0(tracer):
        if tracer.startswith('BGS'): P0 = 7000
        if tracer.startswith('LRG'): P0 = 10000
        if tracer.startswith('ELG'): P0 = 4000
        if tracer.startswith('QSO'): P0 = 6000
        return P0
    
    def get_veff(footprint):
        import numpy as np
        from cosmoprimo.fiducial import DESI
        P0 = get_P0(footprint.attrs['tracer'])
        cosmo = DESI()
        d = cosmo.comoving_radial_distance(footprint._zrange) / cosmo.h
        dv = footprint._area / (180. / np.pi)**2 / 3. * (d[1:]**3 - d[:-1]**3)
        return np.sum(((footprint._nbar * P0) / (1 + footprint._nbar * P0))**2 * dv) / 1e9  # in (Gpc/h)^(-3)
        #return np.sum(dv) / 1e9  # in (Gpc/h)^(-3)

    dfm = get_data_file_manager(conf='unblinded').select(version='v1.2')

    for tracer, zranges in list_zrange.items():
        data = dfm.select(id='catalog_data_y1', tracer=tracer)
        all_randoms = [dfm.select(id='catalog_randoms_y1', tracer=tracer, iran=iran) for iran in range(1)]
        for zrange in zranges:
            footprint_nside = get_footprint(data, all_randoms=all_randoms, tracer=tracer, region='GCcomb', nside=64, zrange=zrange)
            footprint = get_footprint(data, all_randoms=all_randoms, tracer=tracer, region='GCcomb', nside=None, zrange=zrange)
            print('In {}, {}: {:d} tracers, zeff = {:.2f}, Veff = {:.2f} - {:.2f}'.format(tracer, zrange, footprint.attrs['ndata'], footprint.attrs['zeff'], get_veff(footprint_nside), get_veff(footprint)))

    def get_cross(footprint1, footprint2):
        footprint12 = footprint1 & footprint2
        P1, P2 = get_P0(footprint1.attrs['tracer']), get_P0(footprint2.attrs['tracer'])
        V12 = footprint12.volume
        V1, V2 = footprint1.volume, footprint2.volume
        #print(footprint1.attrs['ndata'] / footprint1.size, footprint2.attrs['ndata'] / footprint2.size)
        nbar1, nbar2 = footprint1.size / V1, footprint2.size / V2
        #print(V12, V1, V2, V12 / (V1 * V2)**(0.5), P1 / (P1 + 1 / nbar1), P2 / (P2 + 1 / nbar2), nbar1, nbar2)
        return V12 * (P1 * P2) / ((V1 * V2)**(0.5) * (P1 + 1 / nbar1) * (P2 + 1 / nbar2))

    # Overlap
    zranges = [(0.8, 1.1), (1.1, 1.6)]
    tracers = [['LRG+ELG_LOPnotqso', 'QSO'], ['ELG_LOPnotqso', 'QSO']]
    for zrange, tracers in zip(zranges, tracers):
        footprints_nside, footprints = [], []
        for tracer in tracers:
            data = dfm.select(id='catalog_data_y1', tracer=tracer)
            all_randoms = [dfm.select(id='catalog_randoms_y1', tracer=tracer, iran=iran) for iran in range(1)]
            footprints_nside.append(get_footprint(data, all_randoms=all_randoms, tracer=tracer, region='GCcomb', nside=64, zrange=zrange))
            footprints.append(get_footprint(data, all_randoms=all_randoms, tracer=tracer, region='GCcomb', nside=None, zrange=zrange))
        print('For {}, in {}, correlation is {:.2f} - {:.2f}'.format(tracers, zrange, get_cross(*footprints_nside), get_cross(*footprints)))


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


def get_bao_forecasts(region='GCcomb', apmode='qparqper'):

    from desilike.theories.galaxy_clustering import (BAOPowerSpectrumTemplate, SimpleBAOWigglesTracerPowerSpectrumMultipoles, SimpleTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles, StandardPowerSpectrumTemplate)
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, ObservablesCovarianceMatrix
    from desilike.likelihoods.galaxy_clustering import SNWeightedPowerSpectrumLikelihood, ObservablesGaussianLikelihood
    from desilike import Fisher
    from desi_y1_files.file_manager import get_data_file_manager, list_zrange, get_fit_setup
    from y1_data_2pt_tools import get_footprint

    dfm = get_data_file_manager(conf='unblinded').select(version='v1.2')

    fishers_bao = []

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

    def get_bias(tracer='ELG'):
        if tracer.startswith('BGS'):
            bias = 1.5
        elif tracer.startswith('LRG+ELG'):
            bias = 1.6
        elif tracer.startswith('LRG'):
            bias = 2.0
        elif tracer.startswith('ELG'):
            bias = 1.2
        elif tracer.startswith('QSO'):
            bias = 2.1
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
        return bias

    def get_bias(tracer='ELG', zrange=None):
        if tracer.startswith('BGS'):
            bias = 1.5
        elif tracer.startswith('LRG+ELG'):
            bias = 1.6
        elif tracer.startswith('LRG'):
            #bias = 2.0
            bias = {(0.4, 0.6): 1.9, (0.6, 0.8): 2.1, (0.8, 1.1): 2.3}[zrange]
        elif tracer.startswith('ELG'):
            #bias = 1.2
            bias = {(0.8, 1.1): 1.02, (1.1, 1.6): 1.53}[zrange]
        elif tracer.startswith('QSO'):
            #bias = 2.1
            bias = 2.3
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
        return bias

    def get_sigmas(tracer='ELG', zrange=None):
        if tracer.startswith('BGS'):
            sigmapar, sigmaper = 8., 3.
        elif tracer.startswith('LRG+ELG'):
            sigmapar, sigmaper = 6., 3.
        elif tracer.startswith('LRG'):
            sigmapar, sigmaper = 6., 3.
        elif tracer.startswith('ELG'):
            sigmapar, sigmaper = 6., 3.
        elif tracer.startswith('QSO'):
            sigmapar, sigmaper = 6., 3.
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
        return sigmapar, sigmaper
    
    for tracer, zranges in list_zrange.items():
        if tracer != 'LRG': continue # not tracer.startswith('LRG'): continue
        data = dfm.select(id='catalog_data_y1', tracer=tracer)
        all_randoms = [dfm.select(id='catalog_randoms_y1', tracer=tracer, iran=iran) for iran in range(1)]
        for zrange in zranges:
            footprint = get_footprint(data, all_randoms=all_randoms, region='GCcomb', zrange=zrange, nside=32)
            cosmo, z = footprint.cosmo, footprint.attrs['zeff']
            fo = cosmo.get_fourier()
            template = StandardPowerSpectrumTemplate(z=z, fiducial=cosmo, apmode=apmode)
            b0 = get_fit_setup(tracer, return_list='b0')
            s, s0 = fo.sigma8_z(z, of='delta_cb'), fo.sigma8_z(0., of='delta_cb')
            b1 = b0 / (s / s0)  # prescription for linear bias
            b1 = get_bias(tracer=tracer, zrange=zrange)

            r = get_recon_factor(template, footprint.shotnoise, b1=b1)
            f = fo.sigma8_z(z, of='theta_cb') / s
            sigmaper = r * 9.4 * (s / 0.9)
            sigmapar = (1. + f) * sigmaper
            sigmapar, sigmaper = get_sigmas(tracer=tracer, zrange=zrange)
            #sigmaper, sigmapar = {'LRG': (2.6, 6.6)}.get(tracer, (sigmaper, sigmapar))
            # print(r, b1, b1 * fo.sigma8_z(z, of='delta_cb'), sigmaper, sigmapar, footprint.size, footprint.area, footprint.attrs['zeff'])
            params = {'b1': b1, 'sigmapar': sigmapar, 'sigmaper': sigmaper, 'sigmas': 2.}  # fiducial model parameters
            covariance_params = params
            
            template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI', apmode=apmode)
            theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template)
            lim = {0: (48., 148., 4.), 2: (48., 148., 4.)}
            observable = TracerCorrelationFunctionMultipolesObservable(data=params, slim=lim, theory=theory)
            covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)  # Gaussian covariance matrix
            covariance1 = covariance(**covariance_params)
            from desipipe.file_manager import BaseFile
            covariance = BaseFile('/global/cfs/cdirs/desi//users/mrash/RascalC/Y1/unblinded/v1.2/xi024_{}_IFFT_recsym_sm{}_GCcomb_{:.1f}_{:.1f}_default_FKP_lin4_s20-200_cov_RascalC_rescaled.txt'.format(tracer, 30 if tracer == 'QSO' else 15, *zrange), filetype='correlation_covariance', options=dict()).load(lim)
            import numpy as np
            print(np.diag(covariance1) / np.diag(covariance))
            observable.init.update(covariance=covariance)
            observable(**params)
            fig = observable.plot()
            from desilike.samples import Profiles
            profiles = Profiles.load('/global/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/altmtl/fits_2pt/mean_precscale1/fits_correlation_dampedbao_bao-qisoqap_fixed/recon_IFFT_recsym_sm{}/profiles_{}_GCcomb_z{:.1f}-{:.1f}_default_FKP_sigmas-2.0-2.0_sigmapar-{:.1f}-2.0_sigmaper-{:.1f}-1.0_lim_0-50-150_2-50-150.npy'.format(30 if tracer == 'QSO' else 15, tracer, *zrange, sigmapar, sigmaper))
            s = profiles.attrs['observable']['s']
            xi = profiles.attrs['observable']['data']
            fig.axes[0].plot(s[0], s[0]**2 * xi[0], linestyle='--')
            from matplotlib import pyplot as plt
            plt.savefig('tmp_{}_z{:.1f}-{:.1f}.png'.format(tracer, *zrange))
            likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=covariance)

            """
            theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
            observable = TracerPowerSpectrumMultipolesObservable(data=params,  # data can be a dictionary of parameters
                                                                 klim={0: [0.01, 0.3, 0.005], 2: [0.01, 0.3, 0.005]},
                                                                 theory=theory)
            covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)  # Gaussian covariance matrix
            likelihood = ObservablesGaussianLikelihood(observables=observable, covariance=covariance(**covariance_params))
            """
            """
            params = {'b1': b1, 'sigmapar': sigmapar, 'sigmaper': sigmaper, 'sigmas': 2.}  # fiducial model parameters
            covariance_params = params
            #covariance_params = {'b1': b1, 'sigmapar': 0., 'sigmaper': 0.}  # fiducial covariance parameters (simple Kaiser model)
            theory = SimpleBAOWigglesTracerPowerSpectrumMultipoles(template=template)  # this BAO model shifts wiggles only
            # For klim=(0.01, 0.5), we only use the information from the BAO feature in the power spectrum
            likelihood = SNWeightedPowerSpectrumLikelihood(theories=theory, data=params, covariance=covariance_params, footprints=footprint, klim=(0.01, 0.5))
            """
            for param in theory.params.select(basename='al*'):
                param.update(value=0., fixed=True)  # fixing broadband parameters

            fisher = Fisher(likelihood)  # initializing Fisher
            fisher_bao = fisher(**params).view(params=['qpar', 'qper'] if apmode == 'qparqper' else ['qiso', 'qap'])  # computing Fisher prediction at fiducial parameters
            attrs = {'zeff': z, 'tracer': tracer, 'zrange': zrange}
            fisher_bao.attrs = attrs
            fishers_bao.append(fisher_bao)

    return fishers_bao


def get_bao_y5_forecasts():

    import os
    import numpy as np
    
    forecasts_dir = 'y5_forecasts'

    class TracerData(object):

        """Class storing tracer's density, fiducial bias, and reference forecasts (for comparison purposes)."""

        data_dir = forecasts_dir

        def __init__(self, tracer):
            self.b0 = {'BGS': 1.34, 'LRG': 1.7, 'ELG': 0.84, 'QSO': 1.2}[tracer]
            zlim = {'BGS': [0., 0.4], 'LRG': [0.4, 1.1], 'ELG': [1.1, 1.6], 'QSO': [1.6, 2.1]}[tracer]
            self.read_nz(tracer=tracer, zlim=zlim)
            self.read_ref(tracer=tracer, zlim=zlim, source='gofish')

        def read_nz(self, tracer='ELG', zlim=None, fn=None):
            if fn is None:
                fn = os.path.join(self.data_dir, 'nz_table_DESI.asc')
            zmin, zmax, self.nbar = np.loadtxt(fn, usecols=[0, 1, 2 + ['BGS', 'LRG', 'ELG', 'QSO'].index(tracer)], unpack=True)
            if zlim is not None:
                mask = (zmin >= zlim[0]) & (zmax <= zlim[-1])
                zmin, zmax, self.nbar = zmin[mask], zmax[mask], self.nbar[mask]
            self.zranges = list(zip(zmin, zmax))

        def read_ref(self, tracer='ELG', zlim=None, source='gofish', fn=None):
            if source == 'gofish':
                # From GoFish directly, with Abacus fiducial cosmology
                if fn is None:
                    fn = 'gofish.txt'
                self.sigma = {}
                z, self.sigma['df'], self.sigma['qper'], self.sigma['qpar'] = np.loadtxt(os.path.join(self.data_dir, fn), usecols=[0, 3, 5, 7], unpack=True)
                mask = Ellipsis
                if zlim is not None:
                    mask = (z >= zlim[0]) & (z <= zlim[-1])
                for name in self.sigma:
                    self.sigma[name] = np.array(self.sigma[name])[mask] / 100.
            else:
                # Figures in SV paper, using A_s = 2.1e-9, instead of 2.0830e-9
                self.sigma = {}
                self.sigma['qper'] = {'BGS': [6.65, 2.57, 1.64, 1.37],
                                      'LRG': [1.25, 1.05, 0.92, 0.84, 0.78, 0.87, 1.25],
                                      'ELG': [1.24, 1.26, 1.30, 1.37, 1.87],
                                      'QSO': [3.39, 3.48, 3.67, 3.83, 4.22]}[tracer]
                self.sigma['qpar'] = {'BGS': [13.92, 5.40, 3.41, 2.70],
                                      'LRG': [2.38, 1.99, 1.74, 1.56, 1.44, 1.52, 2.04],
                                      'ELG': [1.80, 1.80, 1.82, 1.89, 2.46],
                                      'QSO': [4.76, 4.87, 5.14, 5.36, 5.90]}[tracer]
                self.sigma['df'] = {'BGS': [31.64, 12.04, 7.54, 5.76],
                                    'LRG': [5.96, 5.16, 4.67, 4.34, 4.14, 4.19, 4.77],
                                    'ELG': [2.58, 2.62, 2.69, 2.80, 3.34],
                                    'QSO': [7.30, 7.63, 8.17, 8.66, 9.58]}[tracer]
                for name in self.sigma:
                    self.sigma[name] = np.array(self.sigma[name]) / 100.
    
    from cosmoprimo.fiducial import DESI
    from desilike.theories.galaxy_clustering import (BAOPowerSpectrumTemplate, SimpleBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerPowerSpectrumMultipoles,
                                                     StandardPowerSpectrumTemplate, SimpleTracerPowerSpectrumMultipoles)
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, CutskyFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods.galaxy_clustering import ObservablesGaussianLikelihood, SNWeightedPowerSpectrumLikelihood
    from desilike import Fisher, setup_logging

    #setup_logging()

    cosmo = DESI()
    fo = cosmo.get_fourier()

    def get_recon_factor(template, shotnoise, **params):
        """The reconstruction damping factor: 1 at low density, 0.5 at high density."""
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
        return spline(np / 0.1734).item()

    tracer_data, fishers_bao, fishers_fs, fishers = {}, {}, {}, {}

    for tracer in ['BGS', 'LRG', 'ELG', 'QSO']:

        print('Running {}'.format(tracer))

        fishers_bao[tracer], fishers_fs[tracer], fishers[tracer] = [], [], []
        tracer_data[tracer] = data = TracerData(tracer)

        for zrange, nbar in zip(data.zranges, data.nbar):
            z = np.mean(zrange)
            # Footprint, to get volume and shot noise; provided nbar is in deg^(-2)
            footprint = CutskyFootprint(area=14000., zrange=zrange, nbar=nbar, cosmo=cosmo)
            b1 = data.b0 / cosmo.growth_factor(z)  # prescription for linear bias
            # Additionally, to calculate the BAO errors we assume degradation of the BAO damping scale
            # by a factor which emulates a standard reconstruction procedure that is used in typical BAO analyses.
            template = StandardPowerSpectrumTemplate(z=z, fiducial='DESI', apmode='qparqper')  # just to get nP
            r = get_recon_factor(template, footprint.shotnoise, b1=b1)
            sigmaper = 9.4 * (fo.sigma8_z(z, of='delta_m') / 0.9)
            f = cosmo.growth_rate(z)
            params = {'b1': b1, 'sigmapar': r * (1. + f) * sigmaper, 'sigmaper': r * sigmaper}  # fiducial model parameters
            covariance_params = {'b1': b1, 'sigmapar': 0., 'sigmaper': 0.}  # fiducial covariance parameters (simple Kaiser model)
            template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI', apmode='qparqper')
            #theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
            theory = SimpleBAOWigglesTracerPowerSpectrumMultipoles(template=template)  # this BAO model shifts wiggles only
            for param in theory.params.select(basename=['al*', 'sigmas', 'dbeta', 'b1']):
                param.update(fixed=True)  # fixing broadband parameters and b1, FoG, dbeta, only shift wiggles
            # For klim=(0.01, 0.5), we only use the information from the BAO feature in the power spectrum
            likelihood = SNWeightedPowerSpectrumLikelihood(theories=theory, data=params, covariance=covariance_params, footprints=footprint, klim=(0.01, 0.5))
            fisher = Fisher(likelihood)  # initializing Fisher
            fisher_bao = fisher(**params)  # computing Fisher prediction at fiducial parameters
            fisher_bao.attrs['zeff'] = z
            fishers_bao[tracer].append(fisher_bao.view(params=['qpar', 'qper']))

            template = StandardPowerSpectrumTemplate(z=z, fiducial='DESI', apmode='qparqper')  # here we use a standard power spectrum template, to vary f
            theory = SimpleTracerPowerSpectrumMultipoles(template=template)  # this is a damped Kaiser model
            # For klim=(0.01, 0.5), we only use the RSD signal (f, b varied)
            likelihood = SNWeightedPowerSpectrumLikelihood(theories=theory, data=params, covariance=covariance_params, footprints=footprint, klim=(0.01, 0.1))
            for param in likelihood.all_params:
                if param.basename not in ['df', 'b1']: param.update(fixed=True)  # fixing all parameters (including shot noise) except f and b1
            fisher = Fisher(likelihood)  # initializing Fisher
            fisher_fs = fisher(**params)  # computing Fisher prediction at fiducial parameters
            fisher_fs.attrs['zeff'] = z
            fishers_fs[tracer].append(fisher_fs)

            # Concatenating BAO information with RSD information on f
            fishers[tracer].append(fisher_bao.view(params=['qpar', 'qper']) + fisher_fs)

    gal_cov = np.loadtxt(os.path.join(forecasts_dir, 'desi_forecast_covariance.txt'))
    from desilike import LikelihoodFisher
    tracers = [tracer for tracer in fishers]
    fishers_gofish = []
    with open(os.path.join(forecasts_dir, 'desi_forecast_datavector.dat'), 'r') as file:
        lines = np.array([[float(item) for item in line.split(' ')[:2]] for line in file])
        for i in range(0, len(lines), 3):
            i += 1
            z = lines[i][0]
            assert z == lines[i + 1][0]
            cov = gal_cov[np.ix_([i + 1, i], [i + 1, i])]
            print('gal1', cov)
            fid2 = lines[[i + 1, i], 1][:, None] * lines[[i + 1, i], 1]
            cov = cov / fid2
            print('gal2', cov)
            cov[0, 1] = cov[1, 0] = -cov[0, 1]  # Hz_rs => DH_over_rs
            fisher = LikelihoodFisher(center=[1., 1.], params=['qpar', 'qper'], hessian=-np.linalg.inv(cov), attrs={'zeff': z})
            #if len(fishers_gofish[tracer]) == len(fishers_gofish):
            #    tracer = tracers[tracers.index(tracer) + 1]
            fishers_gofish.append(fisher)
    
    from matplotlib import pyplot as plt

    for name in ['qper', 'qpar', 'df']:
        ax = plt.gca()
        ax.plot([], [], linestyle='-', color='k', label='desilike')
        ax.plot([], [], linestyle=':', color='k', label='GoFish')
        for itracer, tracer in enumerate(fishers):
            z = np.mean(tracer_data[tracer].zranges, axis=-1)
            ax.plot(z, [fisher.std(params=name) for fisher in fishers[tracer]], color='C{:d}'.format(itracer), label=tracer)
            #ax.plot(z, tracer_data[tracer].sigma[name], color='C{:d}'.format(itracer), linestyle=':')
        ax.plot([fisher.attrs['zeff'] for fisher in fishers_gofish], [fisher.std(params=name) for fisher in fishers_gofish], color='k', linestyle=':')
        ax.set_xlabel('z')
        ax.set_ylabel(r'% fractional error on {}'.format(name))
        ax.legend()
        plt.savefig('forecast_{}.png'.format(name))
        plt.close(plt.gcf())
       
    from cosmoprimo.fiducial import DESI
    from cosmoprimo import constants

    desi = DESI()
    bestfit = load_cobaya_samples(model='base_w_wa', run='run1', dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
    fiducial = desi.clone(N_eff=3.044, omega_cdm=bestfit.mean('omch2'), omega_b=bestfit.mean('ombh2'), H0=bestfit.mean('H0'), w0_fld=bestfit.mean('w'), wa_fld=bestfit.mean('wa'), logA=bestfit.mean('logA'), n_s=bestfit.mean('ns'), tau_reio=bestfit.mean('tau'))
 
    def get_fid(z):
        DM_over_rd_fid = fiducial.comoving_angular_distance(z) / fiducial.rs_drag
        DH_over_rd_fid = (constants.c / 1e3) / (100. * fiducial.efunc(z)) / fiducial.rs_drag
        return np.array([DH_over_rd_fid, DM_over_rd_fid])
    
    # Save cobaya
    fishers = [fisher for fishers in fishers_bao.values() for fisher in fishers]
    lya_cov = np.loadtxt(os.path.join(forecasts_dir, 'desi_lyman_alpha_forecast_covariance.txt'))
    from desilike import LikelihoodFisher
    with open(os.path.join(forecasts_dir, 'desi_lyman_alpha_forecast_datavector.dat'), 'r') as file:
        lines = np.array([[float(item) for item in line.split(' ')[:2]] for line in file])
        for i in range(0, len(lines), 2):
            z = lines[i][0]
            assert z == lines[i + 1][0]
            cov = lya_cov[np.ix_([i, i + 1], [i, i + 1])]
            print('lya1', cov)
            fid2 = lines[[i, i + 1], 1][:, None] * lines[[i, i + 1], 1]
            cov = cov / fid2
            print('lya2', cov)
            cov[0, 1] = cov[1, 0] = -cov[0, 1]  # Hz_rs => DH_over_rs
            fisher = LikelihoodFisher(center=[1., 1.], params=['qpar', 'qper'], hessian=-np.linalg.inv(cov), attrs={'zeff': z})
            fishers.append(fisher)

    bao_dir = '../desi_y1_cosmo_bindings/cobaya_likelihoods/bao_data'
    mean_fn = os.path.join(bao_dir, 'forecast_desiy5_gaussian_bao_ALL_GCcomb_mean.txt')
    cov_fn = os.path.join(bao_dir, 'forecast_desiy5_gaussian_bao_ALL_GCcomb_cov.txt')
    with open(mean_fn, 'w') as file:
        for fisher in fishers:
            z = fisher.attrs['zeff']
            fid = get_fid(z)
            file.write('{:.6f} {:.6f} DH_over_rs\n'.format(z, fid[0]))
            file.write('{:.6f} {:.6f} DM_over_rs\n'.format(z, fid[1]))
    from scipy import linalg
    cov = []
    for fisher in fishers:
        z = fisher.attrs['zeff']
        fid = get_fid(z)
        cov.append(fisher.covariance(params=['qpar', 'qper']) * (fid[:, None] * fid))
    cov = linalg.block_diag(*cov)
    np.savetxt(cov_fn, cov)


if __name__ == '__main__':

    todo = ['runs', 'data', 'load', 'table', 'test', 'forecasts'][1:2]
    #todo = ['test2']
    #todo = ['data', 'data_chain'][:1]
    
    if 'runs' in todo:
        for config in yield_configs(theory='camb', run='run1'):
            # dataset = ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']
            # dataset = ['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']
            # dataset = ['desi-bao-all', 'planck-NPIPE-highl-CamSpec-TTTEEE']
            # get_cobaya_info(**config)
            # get_cobaya_info(**config, save_fn=f'test_{model}_{dataset}.yaml')
            sample_cobaya(**config, sampler='evaluate')
            # sample_cobaya(**config)

            # Make sure the arguments match those used in your sampling
            output = get_cobaya_output(**config, base_dir=base_dir)
            plot_progress(output)
            print_convergence(output)
    
    if 'data' in todo:
        #get_veff()
        version = 'v1.5'
        print(data_dir(version=version))
        tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1)), ('Lya', (1.8, 4.2))]
        #tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1)), ('Lya', (1.8, 4.2))]
        acc_low, acc_high = 0., 0.
        for tracer, zrange in tracers:
            if 'Lya' not in tracer:
                chain = load_bao_chain(tracer, zrange, version=version)
                meas = ' '.join(['{} = {:.4f} \pm {:.4f}'.format(param, chain.mean(param), chain.std(param)) for param in chain.params(name=['qiso', 'qap'])])
                zeff = chain.attrs['zeff']
                #print('In {}, {}, zeff = {:.3f}: {}'.format(tracer, zrange, zeff, meas))

            scale = 'distance'
            fisher = load_bao_fisher(tracer, zrange, scale=scale, with_syst=True, version=version, return_type=None)
            corr = 1.
            if len(fisher.names()) > 1: corr = fisher.corrcoef()[0, 1]
            #print(fisher.attrs.get('data', None))
            meas = ' '.join([r'{} = {:.2f} \pm {:.2f}'.format(param, fisher.mean(param), fisher.std(param)) for param in fisher.params()])
            zeff = fisher.attrs['zeff']
            print(r'In {}, {}, zeff = {:.3f}: {}, corr = {:.3f}'.format(tracer, zrange, zeff, meas, corr))
            fisher = load_bao_fisher(tracer, zrange, apmode='qisoqap', scale=None, with_syst=True, return_type=None)
            #print(tracer, zrange, fisher.std('qiso'))
            tmp = 1. / fisher.std('qiso')**2
            if zeff < 1.1:
                acc_low += tmp
            else:
                acc_high += tmp
        acc = (acc_low + acc_high)**(-0.5)
        print('Aggregate precision: {:.3f}%, low-z {:.3f}%, high-z {:.3f}%'.format(100 * (acc_low + acc_high)**(-0.5), 100 * acc_low**(-0.5), 100. * acc_high**(-0.5)))


    if 'data_chain' in todo:
        import numpy as np
        tracers = [('BGS_BRIGHT-21.5', (0.1, 0.4)), ('LRG', (0.4, 0.6)), ('LRG', (0.6, 0.8)), ('LRG', (0.8, 1.1)), ('LRG+ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (0.8, 1.1)), ('ELG_LOPnotqso', (1.1, 1.6)), ('QSO', (0.8, 2.1))]
        
        def convert(fisher):
            from desi_y1_files.cosmo_tools import predict_bao
            z = fisher.attrs['zeff']
            eta = 1./3.
            if fisher.names() == ['qpar', 'qper']:
                DH_over_rd_fid, DM_over_rd_fid = predict_bao(z, apmode='qparqper', scale='distance', eta=eta)
                fid = np.array([DH_over_rd_fid, DM_over_rd_fid])
            if fisher.names() == ['qiso', 'qap']:
                DV_over_rd_fid, FAP_fid = predict_bao(z, apmode='qisoqap', scale='distance', eta=eta)
                fid = np.array([DV_over_rd_fid, 1. / FAP_fid])
            if fisher.names() == ['qiso']:
                DV_over_rd_fid, FAP_fid = predict_bao(z, apmode='qisoqap', scale='distance', eta=eta)
                fid = np.array([DV_over_rd_fid])
            mean = fisher.mean()
            #print(1, mean)
            cov = fisher.covariance()
            mean = mean * fid
            #print(2, mean, fisher.attrs['zeff'])
            cov = cov * (fid[:, None] * fid)
            return fisher.clone(center=mean, hessian=-np.linalg.inv(cov), params=fisher.params())
        
        for tracer, zrange in tracers:
            chain = load_bao_chain(tracer, zrange)
            covsyst_qisoqap = np.diag([0.245, 0.3])**2 * 1e-4  # eq. 5.3 of https://fr.overleaf.com/project/645d2ce132ee6c4f6baa0ddd
            eta = 1. / 3.
            jac = np.array([[1., 1. - eta], [1., - eta]])  # ('qisoqap' -> 'qparqper')
            covsyst_qparqper = jac.dot(covsyst_qisoqap).dot(jac.T)
            
            iso = 'qpar' not in chain
            if iso:
                fisher = chain.to_fisher(params=['qiso'])
                fisher._hessian = np.linalg.inv(np.linalg.inv(fisher._hessian) - covsyst_qisoqap[:1, :1])
                fisher.attrs.update(chain.attrs)
                
                fisher = convert(fisher)
                zeff = fisher.attrs['zeff']
                meas = ' '.join(['{} = {:.2f} \pm {:.2f}'.format(param, fisher.mean(param), fisher.std(param)) for param in fisher.params()])
                print('In {}, {}, zeff = {:.2f}: {}'.format(tracer, zrange, zeff, meas))
            else:
                fisher = chain.to_fisher(params=['qiso', 'qap'])
                fisher._hessian = np.linalg.inv(np.linalg.inv(fisher._hessian) - covsyst_qisoqap)
                fisher.attrs.update(chain.attrs)
                
                fisher = convert(fisher)
                zeff = fisher.attrs['zeff']
                meas = ' '.join(['{} = {:.2f} \pm {:.2f}'.format(param, fisher.mean(param), fisher.std(param)) for param in fisher.params()[:1]] + ['{} = {:.3f} \pm {:.3f}'.format(param, fisher.mean(param), fisher.std(param)) for param in fisher.params()[1:]])
                print('In {}, {}, zeff = {:.2f}: {}'.format(tracer, zrange, zeff, meas))
                
                fisher = chain.to_fisher(params=['qpar', 'qper'])
                fisher._hessian = np.linalg.inv(np.linalg.inv(fisher._hessian) - covsyst_qparqper)
                fisher.attrs.update(chain.attrs)
                
                fisher = convert(fisher)
                zeff = fisher.attrs['zeff']
                meas = ' '.join(['{} = {:.2f} \pm {:.2f}'.format(param, fisher.mean(param), fisher.std(param)) for param in fisher.params()])
                print('In {}, {}, zeff = {:.2f}: {}, corr = {:.3f}'.format(tracer, zrange, zeff, meas, fisher.corrcoef()[0, 1]))

    if 'load' in todo:
        #print(load_cobaya_samples(model='base_mnu_w_wa', dataset=['desi-bao-all', 'union3', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']))
        #print(load_cobaya_samples(model='base_mnu_w_wa', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'union3']))
        #print(load_cobaya_samples(model='base', dataset=['desi-bao-all'], source='', sampler='iminuit'))
        #print(load_cobaya_samples(theory='camb', model='base', dataset=['desi-bao-all','planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing']))
        #print(load_cobaya_samples(model='base_w_wa', dataset=['desi-bao-all', 'pantheonplus', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing']))
        #print(load_cobaya_samples(model='base', dataset=['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing']))
        #print(load_cobaya_samples(model='base_omegak', dataset=['desi-sdss-bao-best', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'union3'], add=['planck-act-dr6-lensing']))
        #print(load_cobaya_samples(theory='camb', model='base', dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing']))
        #print(load_cobaya_samples(theory='camb', model='base', dataset=['desi-bao-all', 'schoneberg2024-bbn']).getLatex(params=['H0', 'H0rdrag'], limit=1, err_sig_figs=2))
        #print(load_cobaya_samples(theory='camb', model='base', dataset=['desi-bao-all', 'planck2018-rdrag']).getLatex(params=['H0', 'H0rdrag'], limit=1, err_sig_figs=2))
        #print(load_cobaya_samples(theory='camb', model='base', dataset=['desi-bao-all',  'schoneberg2024-bbn', 'planck2018-thetastar-marg-nnu']).getLatex(params=['H0', 'H0rdrag'], limit=1, err_sig_figs=2))
        #print(load_cobaya_samples(theory='camb', model='base_mnu', dataset=['desi-bao-all',  'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'], source='hanyuz').getLatex(params=['mnu'], limit=2, err_sig_figs=2))
        #planck = load_cobaya_samples(model='base', run='run0', dataset=['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
        #print(planck['hrdrag'])
        #planck = load_cobaya_samples(model='base', dataset=['pantheonsplus', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'])
        #planck = load_cobaya_samples(model='base_omegak', dataset=['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'])
        #config = dict(dataset=['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'], run='run1', bestfit='iminuit')
        #print(load_cobaya_samples(model='base', **config).getBestFit(max_posterior=False))
        #load_cobaya_samples(model='base_mnu', run='run0', dataset=['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'])
        #samples = load_planck2018_samples(model='base_nnu', dataset=['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck2018-lensing'])
        #print(samples.mean('thetastar'), samples.std('thetastar'), samples.mean('nnu'), samples.std('nnu'))
        #import numpy as np
        #print(np.cov(samples['thetastar'], samples['nnu'], ddof=1))
        config = dict(dataset=['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'], add=['planck-act-dr6-lensing'], sampler='iminuit')
        bestfit1 = load_cobaya_samples(model='base_mnu', **config)

    if 'test' in todo:
        theory = 'camb'
        dataset = ['desi-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']
        #model = 'base_mnu_nh2_dmnu2fixed'
        model = 'base_mnu_nhcamb'
        info = get_cobaya_info(model=model, theory=theory, dataset=dataset, sampler='evaluate')
        import yaml
        print(yaml.dump(info, sort_keys=False))
        sampler = info.pop('sampler')
        from cobaya.model import get_model
        from cobaya.sampler import get_sampler
        params = dict(logA=3.03361, ns=0.964165, theta_MC_100=1.04113, ombh2=0.0222388, omch2=0.119087, tau=0.0524148, mnul=0.1, A_planck=1.00056, calib_100T=1.00027, calib_217T=0.997781, A_cib_217=69.1731, xi_sz_cib=0.0512099, A_sz=4.87497, ksz_norm=0.885228, gal545_A_100=4.1864, gal545_A_143=9.24124, gal545_A_143_217=12.5247, gal545_A_217=89.5789, ps_A_100_100=205.127, ps_A_143_143=42.7511, ps_A_143_217=37.6634, ps_A_217_217=96.2765, galf_TE_A_100=0.191156, galf_TE_A_100_143=0.142553, galf_TE_A_100_217=0.482051, galf_TE_A_143=0.180519, galf_TE_A_143_217=0.607538, galf_TE_A_217=2.10085)
        #params['H0'] = params.pop('theta_MC_100')
        #params['H0'] = 67.597
        #deltamnu21sq = (7.42e-5, 0.21e-5)
        #deltamnu31sq = (2.510e-3, 0.027e-3)
        #deltamnu32sq = (-2.490e-3, 0.027e-3)
        
        deltamnu21sq = (7.54e-5, 0.)
        deltamnu31sq = (2.46e-3, 0.)
        deltamnu32sq = (- deltamnu31sq[0] - deltamnu21sq[0], 0.)

        if 'nhcamb' in model:
            mnu0 = params.pop('mnul')
            params['mnu'] = mnu0 + (mnu0**2 + deltamnu21sq[0])**0.5 + (mnu0**2 + deltamnu31sq[0])**0.5
            #params['mnu'] = 2 * mnu0 + (mnu0**2 + deltamnu31sq[0])**0.5
        elif 'ihcamb' in model:
            mnu0 = params.pop('mnu0')
            params['mnu'] = mnu0 + (mnu0**2 - deltamnu32sq[0] - deltamnu21sq[0])**0.5 + (mnu0**2 - deltamnu32sq[0])**0.5
        elif 'nh3' in model:
            params['deltamnu21sq'] = deltamnu21sq[0]
            params['deltamnu31sq'] = deltamnu31sq[0]
        elif 'ih3' in model:
            params['deltamnu21sq'] = deltamnu21sq[0]
            params['deltamnu32sq'] = deltamnu32sq[0]
        model = get_model(info)
        logp = model.logposterior(params)
        print(logp.loglike)

    if 'test2' in todo:
        dataset = ['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']
        theory = 'camb'
        model = 'base_w_wa'

        config = dict(dataset=dataset[:-1], add=['planck-act-dr6-lensing'], sampler='iminuit')
        bestfit1 = load_cobaya_samples(model=model, bestfit='posterior', **config)[0]
        bestfit2 = load_cobaya_samples(model='base', bestfit='posterior', **config)[0]
        bestfit2['w'] = -1.
        bestfit2['wa'] = 0.
        
        def select(model, values):
            p = model.prior
            return {name: values[name] for name in p.params}

        from cobaya.model import get_model
        from cobaya.sampler import get_sampler

        info = get_cobaya_info(model=model, theory=theory, dataset=dataset, sampler='evaluate')
        sampler = info.pop('sampler')
        model1 = get_model(info)
        logp1 = model1.logposterior(select(model1, bestfit1))
        logp12 = model1.logposterior(select(model1, bestfit2))
        info = get_cobaya_info(model='base', theory=theory, dataset=dataset, sampler='evaluate')
        sampler = info.pop('sampler')
        model2 = get_model(info)
        logp2 = model2.logposterior(select(model2, bestfit2))
        dchi2_new = 2. * (logp12.logpost - logp1.logpost)
        import numpy as np
        dp = np.log(4 * 5)
        dchi2_old = -2. * (bestfit2['minuslogpost'] - bestfit1['minuslogpost'] + dp)
        print(select(model1, bestfit1))
        print(logp1.logpost, bestfit1['minuslogpost'], logp2.logpost, bestfit2['minuslogpost'], dchi2_new, dchi2_old)

    if 'table' in todo:
        samples = {}
        samples['$w$CDM', 'DESI+SNIa+CMB'] = load_cobaya_samples(model='base_w', dataset=['desi-bao-all', 'union3', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'])
        samples['$w_{0}w_{a}$CDM', 'DESI+SNIa+CMB'] = load_cobaya_samples(model='base_mnu_w_wa', dataset=['desi-bao-all', 'union3', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE'])
        print(make_table(samples, params=('omegam', 'w0', 'wa', 'H0', 'H0rd'), add_rule=True))

    if 'forecasts' in todo:
        get_bao_y5_forecasts()
        """
        from desilike import LikelihoodFisher
        for fisher_forecast in get_bao_forecasts(region='GCcomb', apmode='qisoqap'):
            tracer, zrange = fisher_forecast.attrs['tracer'], fisher_forecast.attrs['zrange']
            fisher_y1 = load_bao_fisher(tracer, zrange, apmode='qisoqap', scale=None, with_syst=False, return_type=None)
            print('For tracer = {} in {}, forecast:\n{}'.format(tracer, zrange, fisher_forecast.to_stats(tablefmt='pretty')))
            #print('For tracer = {} in {}, data:\n{}'.format(tracer, zrange, fisher_y1.to_stats(tablefmt='pretty')))
            #print('For tracer = {} in {}; data / forecast errors: {}'.format(tracer, zrange, fisher_y1.std() / fisher_forecast.std(fisher_y1.params())))
        """
