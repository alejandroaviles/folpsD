prior_cov = 0.2**2


def plot_template_power(k, ells, diff, template, fn=None):
    from matplotlib import pyplot as plt
    ax = plt.gca()
    ax.plot([], [], color='k', linestyle='--', label='data')
    ax.plot([], [], color='k', linestyle='-', label='fit')
    for ill, ell in enumerate(ells):
        color = 'C{:d}'.format(ill)
        ax.plot(k, diff[ill], color=color, linestyle='--')
        ax.plot(k, template(ell, k), color=color, linestyle='-')
    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    ax.set_ylabel(r'$\Delta P(k)$ [$(\mathrm{Mpc}/h)^3$]')
    ax.legend()
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(plt.gcf())
        

def export_photo():
    import numpy as np
    from desi_y1_files import get_data_file_manager

    fm = get_data_file_manager()
    for ft in fm.select(id='template_syst_y1', syst=['aic', 'photo'], weighting='default_FKP', observable='power', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region='GCcomb', ignore=True).iter(intersection=False):
        options = dict(ft.options)
        tracer, zrange, region, syst = options['tracer'], options['zrange'], options['region'], options['syst']
        ttracer = None
        if 'ELG' in tracer: ttracer = 'ELG'
        if 'QSO' in tracer: ttracer = 'QSO'

        from desi_y1_files.systematic_template import PolynomialTemplate

        #template_fn = '/global/cfs/cdirs/desi/users/ruiyang/systemplate/baseline/power_systcorr_{syst}_{tracer}_{zrange[0]:.1f}_{zrange[1]:.1f}_{region}.npy'
        syst = {'photo': 'poly2', 'aic': 'mock'}[syst]
        template_fn = '/global/cfs/cdirs/desi/users/ruiyang/systemplate/v1.5/systcorr_{syst}_{tracer}_{zrange[0]:.1f}_{zrange[1]:.1f}_{region}_power_{weight}_thetacut0.05.npy'
        weight = 'RF' if ttracer == 'QSO' else 'SN'
        if ttracer is None: template_fn = template_fn.format(syst=syst, tracer='QSO', zrange=(0.8, 2.1), region=region, weight='RF')
        else: template_fn = template_fn.format(syst=syst, tracer=ttracer, zrange=zrange, region=region, weight=weight)
        template = PolynomialTemplate.load(template_fn)
        if not hasattr(template, 'attrs'):
            template.attrs = {}
        if ttracer is None:  # zero template
            template.popts = [np.zeros_like(popt) for popt in template.popts]
            template.orders = [np.zeros_like(order) for order in template.orders]
        if syst == 'poly2':
            template.attrs['prior_cov'] = prior_cov
        template.save(ft)


def fit_photo(of=None, rotated=False):
    import os
    from pathlib import Path
    import numpy as np
    from desi_y1_files import get_data_file_manager, get_ez_file_manager, get_abacus_file_manager
    from desi_y1_files.systematic_template import PolynomialTemplate
    from y1_data_2pt_tools import postprocess_rotate_wmatrix

    if of is None: of = ['data', 'abacus', 'ez'][:2]

    def run(fs, fm, of):
        if of == 'data': of = ''
        else: of = '_{}'.format(of)
        for ft in fs:
            options = dict(ft.options)
            tracer, zrange, region, syst = options['tracer'], options['zrange'], options['region'], options['syst']

            xlim = (0.02, 0.2, 0.005)
            #options.pop('version')
            ttracer = None
            if 'ELG' in tracer: ttracer = 'ELG_LOPnotqso'
            if 'QSO' in tracer: ttracer = 'QSO'
            nonzero = ttracer is not None and zrange != (0.8, 1.1)
            if not nonzero:
                tracer, zrange = options['tracer'], options['zrange'] = 'QSO', (0.8, 2.1)
                ttracer = tracer
            #fcov = fm.get(id='covariance{}_y1'.format(rotated), **{**options, 'source': 'ezmock', 'marg': False}, ignore=True).load()
            #print(options, {**options, 'source': 'ezmock', 'marg': False})
            fcov = cfm.get(id='covariance_y1', **{**options, 'source': 'ezmock', 'version': 'v1'}, ignore=True)
            if rotated:
                roptions = {**options, 'weighting': 'default_FKP'}
                roptions.pop('version')
                rotation = fm.get(id='rotation_wmatrix_power{}{}_y1'.format('_merged' if of else '', of), **roptions, ignore=True)
                fcov = postprocess_rotate_wmatrix(rotation, covariance=fcov, return_covariance=True)[0]
            else:
                fcov = fcov.load()
            observable = fcov.observables('power')

            from pypower import PowerSpectrumMultipoles
            imock = list(range(25))
            if 'ELG' in tracer:
                imock = [iimock for iimock in imock if iimock not in (2, 7)]
            mocks1 = [fma.get(id='power_abacus_y1', **{**options, 'version': 'v4_2', 'fa': 'altmtl'}, imock=iimock, ignore=True) for iimock in imock]
            base_dir = Path('/global/cfs/cdirs/desi/users/ruiyang/data/v4_1fixran/altmtl/2pt/')
            mocks2 = [base_dir / 'mock{imock}/pk/pkpoles_{tracer}_{region}_{zrange[0]:.1f}_{zrange[1]:.1f}_default_FKP_{sys}_lin_thetacut0.05.npy'.format(imock=iimock, tracer=ttracer, region=region, zrange=zrange, sys='SYSRF' if ttracer == 'QSO' else 'SYSSN') for iimock in imock]
            if rotated:
                mocks1 = postprocess_rotate_wmatrix(rotation, power=mocks1, return_power=True)[1]
                mocks2 = postprocess_rotate_wmatrix(rotation, power=mocks2, return_power=True)[1]
            else:
                mocks1 = [PowerSpectrumMultipoles.load(dd) for dd in mocks1]
                mocks2 = [PowerSpectrumMultipoles.load(dd) for dd in mocks2]
            mocks1 = [dd.select(xlim) for dd in mocks1]
            mocks2 = [dd.select(xlim) for dd in mocks2]
            k = np.mean([mock.k for mock in mocks1], axis=0)
            #ells = mocks1[0].ells
            ells = observable.projs
            assert ells == [0, 2, 4]
            yerrs = [fcov.view(observables=observable, xlim=xlim[:2], projs=ell, return_type=None).std() for ell in ells]
            orders = [[-5, -3, -2]] * len(yerrs)

            noaic = np.mean([mock(ell=ells, complex=False) for mock in mocks1], axis=0)
            aic = np.mean([mock(ell=ells, complex=False) for mock in mocks2], axis=0)
            diff = aic - noaic

            if syst == 'photo':
                doptions = dict(options)
                doptions.pop('version')
                fpower1 = dfm.get(id='power_y1', **{**doptions, 'weighting': 'default_SYS1_FKP'}, ignore=True)
                fpower2 = dfm.get(id='power_y1', **{**doptions, 'weighting': 'default_FKP'}, ignore=True)
                if rotated:
                    power1 = postprocess_rotate_wmatrix(rotation, power=[fpower1], return_power=True)[1][0]
                    power2 = postprocess_rotate_wmatrix(rotation, power=[fpower2], return_power=True)[1][0]
                else:
                    power1 = fpower1.load()
                    power2 = fpower2.load()
                power1 = power1.select(xlim)
                power2 = power2.select(xlim)
                k = power1.k
                diff = power2(ell=ells, complex=False) - power1(ell=ells, complex=False) - diff
            template = PolynomialTemplate.fit_multipole(k, diff, orders=orders, yerrs=yerrs)
            if syst == 'photo':
                template.attrs['prior_cov'] = prior_cov
            if not nonzero:
                template.popts = [np.zeros_like(popt) for popt in template.popts]
                template.orders = [np.zeros_like(order) for order in template.orders]
            template.save(ft)
            plot_template_power(k, ells, diff, template, fn=os.path.splitext(ft)[0] + '.png')

    cfm = get_data_file_manager(conf='unblinded')
    fma = get_abacus_file_manager()
    rotated = '_rotated' if rotated else ''
    cut = [('theta', 0.05)] if rotated else [None, ('theta', 0.05)]
    region = 'GCcomb' if rotated else ['NGC', 'SGC', 'GCcomb']

    dfm = get_data_file_manager(conf='unblinded').select(version='v1.5')
    if 'data' in of:
        fm = get_data_file_manager(conf='unblinded').select(version='v1.5')
        fs = fm.select(id='template_syst{}_y1'.format(rotated), syst=['aic', 'photo'], weighting='default_FKP', observable='power', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region=region, cut=cut, ignore=True).iter(intersection=False)
        run(fs, fm, of='data')
    if 'abacus' in of:
        fm = get_abacus_file_manager()
        fm = fm.select(tracer=['BGS_BRIGHT-21.5'], version='v1') + fm.select(tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v4_2')
        fs = fm.select(id='template_syst{}_abacus_y1'.format(rotated), syst=['aic', 'photo'], weighting='default_FKP', observable='power', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region=region, cut=cut, ignore=True).iter(intersection=False)
        run(fs, fm, of='abacus')
    if 'ez' in of:
        fm = get_ez_file_manager().select(version='v1')
        fs = fm.select(id='template_syst{}_ez_y1'.format(rotated), syst=['aic', 'photo'], weighting='default_FKP', observable='power', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region=region, cut=cut, ignore=True).iter(intersection=False)
        run(fs, fm, of='ez')


def fit_ric(of=None, rotated=False):
    import os
    import logging
    import numpy as np
    from desi_y1_files import get_data_file_manager, get_ez_file_manager, get_abacus_file_manager
    from desi_y1_files.systematic_template import PolynomialTemplate
    from y1_data_2pt_tools import postprocess_rotate_wmatrix
    
    logger = logging.getLogger('RIC')
    
    if of is None: of = ['data', 'abacus', 'ez']

    def run(fs, fm, of):
        if of == 'data': of = ''
        else: of = '_{}'.format(of)
        for ft in fs:
            options = dict(ft.options)
            #options.pop('version')

            xlim = (0.02, 0.2, 0.005)
            imock = list(range(1, 51))
            mocks1 = list(fm_ez.select(id='power_ez_y1', **{**options, 'fa': 'ffa', 'version': 'v1ric'}, imock=imock, ignore=True))
            mocks2 = list(fm_ez.select(id='power_ez_y1', **{**options, 'fa': 'ffa', 'version': 'v1noric'}, imock=imock, ignore=True))
            if (not mocks1) or (not mocks1[0].exists()): continue

            if rotated:
                rotation = fm.get(id='rotation_wmatrix_power{}{}_y1'.format('_merged' if of else '', of), **{**options, 'weighting': 'default_FKP'}, ignore=True)
                mocks1 = postprocess_rotate_wmatrix(rotation, power=mocks1, return_power=True)[1]
                mocks2 = postprocess_rotate_wmatrix(rotation, power=mocks2, return_power=True)[1]
            else:
                mocks1 = [dd.load() for dd in mocks1]
                mocks2 = [dd.load() for dd in mocks2]
            mocks1 = [dd.select(xlim) for dd in mocks1]
            mocks2 = [dd.select(xlim) for dd in mocks2]
            k = np.mean([mock.k for mock in mocks1], axis=0)
            ells = mocks1[0].ells
            ric = np.mean([mock(complex=False) for mock in mocks1], axis=0)
            noric = np.mean([mock(complex=False) for mock in mocks2], axis=0)
            diff = ric - noric
            yerrs = np.std([mock(complex=False) for mock in mocks1], axis=0, ddof=1)
            orders = [[-5, -3, -2]] * len(yerrs)
            template = PolynomialTemplate.fit_multipole(k, diff, orders=orders, yerrs=yerrs)
            template.save(ft)
            plot_template_power(k, ells, diff, template, fn=os.path.splitext(ft)[0] + '.png')
    
    rotated = '_rotated' if rotated else ''
    cut = [('theta', 0.05)] if rotated else [None, ('theta', 0.05)]
    region = 'GCcomb' if rotated else ['NGC', 'SGC', 'GCcomb']

    dfm = get_data_file_manager(conf='unblinded').select(version='v1.5')
    fm_ez = get_ez_file_manager()
    if 'data' in of:
        fm = get_data_file_manager(conf='unblinded').select(version='v1.5')
        fs = fm.select(id='template_syst{}_y1'.format(rotated), syst='ric', weighting='default_FKP', observable='power', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region=region, cut=cut, ignore=True).iter(intersection=False)
        run(fs, fm, of='data')
    if 'abacus' in of:
        fm = get_abacus_file_manager()
        fm = fm.select(tracer=['BGS_BRIGHT-21.5'], version='v1') + fm.select(tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v4_2')
        fs = fm.select(id='template_syst{}_abacus_y1'.format(rotated), syst='ric', weighting='default_FKP', observable='power', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region=region, cut=cut, ignore=True).iter(intersection=False)
        run(fs, fm, of='abacus')
    if 'ez' in of:
        fm = get_ez_file_manager().select(version='v1')
        fs = fm.select(id='template_syst{}_ez_y1'.format(rotated), syst='ric', weighting='default_FKP', observable='power', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region=region, cut=cut, ignore=True).iter(intersection=False)
        run(fs, fm, of='ez')


def correct(of=None, rotated=False):
    import numpy as np
    from desi_y1_files import get_data_file_manager, get_abacus_file_manager, get_ez_file_manager

    if of is None: of = ['data', 'abacus', 'ez']
    
    def run(fs, fm, rotated, of):
        if of == 'data': of = ''
        else: of = '_{}'.format(of)
        for fcorr in fs:
            options = dict(fcorr.options)
            power = fm.get(id='power{}{}_y1'.format(rotated, of), **options, ignore=True)
            if not power.exists(): continue
            power = power.load().select((0.02, 0.4, 0.005))
            offset = np.zeros_like(power.power_nonorm)
            options.pop('version')
            tric = fm.get(id='template_syst{}{}_y1'.format(rotated, of), syst='ric', **options, ignore=True).load()
            offset -= [tric(ell, power.k) for ell in power.ells]
            if not of:  # data only
                taic = fm.get(id='template_syst{}_y1'.format(rotated), syst='aic', **options, ignore=True).load()
                offset -= [taic(ell, power.k) for ell in power.ells]
            #print(options, power.k, np.sum(offset[:2]))
            power.power_nonorm[...] += offset * power.wnorm
            fcorr.save(power)

    rotated = '_rotated' if rotated else ''
    cut = [('theta', 0.05)] if rotated else [None, ('theta', 0.05)]
    region = 'GCcomb' if rotated else ['NGC', 'SGC', 'GCcomb']
    if 'data' in of:
        fm = get_data_file_manager().select(version='v1.5')
        fs = fm.select(id='power{}_corrected_y1'.format(rotated), weighting='default_FKP', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region=region, cut=cut, ignore=True).iter(intersection=False)
        run(fs, fm, rotated, of='data')
    if 'abacus' in of:
        fm = get_abacus_file_manager()
        fm = fm.select(tracer=['BGS_BRIGHT-21.5'], version='v1') + fm.select(tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v4_2')
        fs = fm.select(id='power{}_corrected_abacus_y1'.format(rotated), weighting='default_FKP', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region=region, cut=cut).iter(intersection=False)
        run(fs, fm, rotated, of='abacus')
    if 'ez' in of:
        fm = get_ez_file_manager().select(version='v1')
        fs = fm.select(id='power{}_corrected_ez_y1'.format(rotated), weighting='default_FKP', tracer=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], region=region, cut=cut, version='v1').iter(intersection=False)
        run(fs, fm, rotated, of='ez')


def export_systematic_covariance(of=None, rotated=True):
    import numpy as np
    from desilike.observables import ObservableCovariance
    from desi_y1_files import get_data_file_manager, get_ez_file_manager, get_abacus_file_manager

    if of is None: of = ['data', 'abacus', 'ez']

    def run(fm, of):
        if of == 'data': of = ''
        else: of = '_{}'.format(of)
        # Export Nathan's HOD covariance
        for fcov in fm.select(id='covariance_syst_rotated{}_y1'.format(of), tracer=tracer, region=region, cut=cut, syst='hod').iter(intersection=False):
            covariance_fn = '/global/cfs/cdirs/desi/users/ntbfin/HOD_tests/KP5_HOD_systematics/HOD_covariance/covariances/{}/C_HOD_window_rotated.npy'.format(fcov.options['tracer'][:3])
            covariance = ObservableCovariance.load(covariance_fn)
            covariance.save(fcov)
        # Export photometric systematic covariance from Ruiyang
        for fcov in fm.select(id='covariance_syst{}{}_y1'.format(rotated, of), tracer=tracer, region=region, cut=cut, syst='photo').iter(intersection=False):
            cov_options = fcov.options
            #if cov_options['tracer'] != 'ELG_LOPnotqso': continue
            temp = fm.get(id='template_syst{}{}_y1'.format(rotated, of), **{'syst': 'photo', **cov_options, 'observable': 'power'}, ignore=True).load()
            cov_options.pop('version')
            cov_options['source'] = 'ezmock'
            covariance = dfm.get(id='covariance{}_y1'.format(rotated), **cov_options, ignore=True).load()
            observable = covariance.observables('power')
            template = np.concatenate([temp(ell, k) for k, ell in zip(observable.x(), observable.projs)], axis=0)
            # Systematic covariance is prior * template.T.dot(template)
            syst = temp.attrs['prior_cov'] * template[..., None] * template
            syst = covariance.clone(value=syst)
            syst.attrs['photo'] = {'templates': [template], 'prior': [temp.attrs['prior_cov']]}
            syst.save(fcov)

    dfm = get_data_file_manager(conf='unblinded')
    rotated = '_rotated' if rotated else ''
    cut = [('theta', 0.05)] if rotated else [None, ('theta', 0.05)]
    region = 'GCcomb' if rotated else ['NGC', 'SGC', 'GCcomb']
    tracer = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']
    
    if 'data' in of:
        fm = get_data_file_manager().select(version='v1.5')
        run(fm, of='data')
    if 'abacus' in of:
        fm = get_abacus_file_manager()
        fm = fm.select(tracer=['BGS_BRIGHT-21.5'], version='v1') + fm.select(tracer=['LRG', 'ELG_LOPnotqso', 'QSO'], version='v4_2')
        run(fm, of='abacus')
    if 'ez' in of:
        fm = get_ez_file_manager().select(version='v1')
        run(fm, of='ez')


def compressed_fn(basename, tracer, zrange, version='v1.5', klim=None):
    from pathlib import Path
    from desi_y1_files import get_data_file_manager

    dfm = get_data_file_manager(conf='unblinded')
    for fcov in dfm.select(id='forfit_y1', version=version).iter(intersection=False):
        base_dir = Path(fcov.filepathro).parent

    if klim is None:
        klim = ''
    else:
        klim = '_klim_' + '_'.join(['{:d}-{:.2f}-{:.2f}'.format(key, *value) for key, value in klim.items()])
    return base_dir / '{}{}_{}_GCcomb_z{:.1f}-{:.1f}.npy'.format(basename, klim, tracer, *zrange)


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
    list_nran = {'BGS_BRIGHT-21.5': 1, 'LRG': 8, 'LRG+ELG_LOPnotqso': 10, 'ELG_LOPnotqso': 10, 'QSO': 4}
    if observable == 'correlation':
        toret.update({'split': 20., 'nran': list_nran[tracer], 'njack': 0})
    if recon:
        toret.update(get_baseline_recon_setup(tracer=tracer, zrange=zrange))
        toret.update({'cut': None})
    return toret


def get_bao_baseline_fit_setup(tracer, zrange=tuple(), observable=None, recon=None, iso=None):
    if iso is None:
        if 'BGS' in tracer or ('ELG' in tracer and 'LRG' not in tracer and tuple(zrange) == (0.8, 1.1)) or 'QSO' in tracer:
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
    return {'theory': 'reptvelocileptors', 'observable': observable, 'lim': lim, 'freedom': 'max', 'tracer': tracer, 'zrange': tuple(zrange), 'weighting': 'default_FKP', 'region': 'GCcomb', 'covmatrix': 'rascalc' if 'corr' in observable else 'ezmock', 'syst': 'rotation-hod-photo', 'wmatrix': 'rotated', 'cut': ('theta', 0.05)}


def prepare_bao(version='v1.5'):

    from pathlib import Path
    from desilike.observables import ObservableArray, ObservableCovariance

    def load_chain(tracer, zrange, version=version, return_profiles=False):
        import os
        from desilike.samples import Chain, Profiles

        def load_chain(fi, burnin=0.5):
            tracer, zrange = fi[0].options['tracer'], fi[0].options['zrange']
            fns = [Path(ff) for ff in fi]
            if 'v1.2' in version: fns = [fn.parent / fn.name.replace('_cov-rascalc', '') for fn in fns]
            chains = [Chain.load(ff).remove_burnin(burnin)[::10] for ff in fns]
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
            chain.attrs['DM_over_rd_fid'] = DM_over_rd_fid
            chain.attrs['DH_over_rd_fid'] = DH_over_rd_fid
            chain.attrs['DV_over_rd_fid'] = DV_over_rd_fid
            chain.attrs['FAP_fid'] = FAP_fid
            return chain

        chain_fn = compressed_fn('chain_bao-recon', tracer, zrange, version=version)
        profiles_fn = compressed_fn('profiles_bao-recon', tracer, zrange, version=version)
        if True: #not os.path.exists(chain_fn) or not os.path.exists(profiles_fn):
            from desi_y1_files import get_data_file_manager
            dfm = get_data_file_manager(conf='unblinded')
            options = dict(get_bao_baseline_fit_setup(tracer, zrange=zrange), version=version)
            fchains = list(dfm.select(id='chains_bao_recon_y1', **options, ignore=True))
            fprofiles = dfm.get(id='profiles_bao_recon_y1', **options, ignore=True)
            chain = load_chain(fchains)
            #chain.attrs['observable'].pop('nowindow', None)
            #if not os.path.exists(chain_fn):
            chain.save(chain_fn)
            fn = Path(fprofiles)
            if 'v1.2' in version: fn = fn.parent / fn.name.replace('_cov-rascalc', '')
            profiles = Profiles.load(fn)
            if version == 'v1.5': profiles.attrs['zeff'] = float('{:.3f}'.format(profiles.attrs['zeff']))
            #profiles.attrs['observable'].pop('nowindow', None)
            #if not os.path.exists(profiles_fn):
            profiles.save(profiles_fn)
        chain = Chain.load(chain_fn)
        profiles = Profiles.load(profiles_fn)
        if return_profiles:
            return chain, profiles
        return chain

    def load_fisher(tracer, zrange, version=version):
        import os
        import numpy as np
        from desilike import LikelihoodFisher

        fisher_fn = compressed_fn('gaussian_bao-recon', tracer, zrange, version=version)
        covstat_fn = compressed_fn('covariance_stat_bao-recon', tracer, zrange, version=version)
        covsyst_fn = compressed_fn('covariance_syst_bao-recon', tracer, zrange, version=version)

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
            fisher.attrs['DM_over_rd_fid'] = DM_over_rd_fid
            fisher.attrs['DH_over_rd_fid'] = DH_over_rd_fid
            fisher_stat = fisher.deepcopy()
            covsyst = np.zeros_like(fisher_stat._hessian)
        else:
            chain, profiles = load_chain(tracer, zrange, version=version, return_profiles=True)
            iso = 'qpar' not in chain
            fisher = chain.to_fisher(params=['qiso'] if iso else ['qpar', 'qper'])
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
            fisher.attrs.update(chain.attrs)
            fisher.attrs.update(profiles.attrs)
            fisher_stat = fisher.deepcopy()
            fisher._hessian = np.linalg.inv(np.linalg.inv(fisher._hessian) - covsyst)

        fisher.save(fisher_fn)
        projs = fisher_stat.params().basenames()
        mean = fisher_stat.mean(params=projs)
        cov = fisher_stat.covariance(params=projs)
        covstat = ObservableCovariance(cov, observables=[{'value': mean, 'projs': projs, 'name': 'bao-recon', 'attrs': {name: fisher_stat.attrs[name] for name in ['zeff']}}])
        covstat.save(covstat_fn)
        covsyst = covstat.clone(value=covsyst)
        covsyst.save(covsyst_fn)

    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'LRG+ELG_LOPnotqso': [(0.8, 1.1)], 'QSO': [(0.8, 2.1)], 'Lya': [(1.8, 4.2)]}
    for tracer, zranges in list_zrange.items():
        for iz, zrange in enumerate(zranges):
            load_fisher(tracer, zrange, version=version)


def prepare_shapefit(version='v1.5', klim=None):

    from desilike.observables import ObservableArray, ObservableCovariance

    def load_chain(tracer, zrange, klim=None, version=version, return_profiles=False):
        import os
        from desilike.samples import Chain, Profiles

        def load_chain(fi, burnin=0.5):
            tracer, zrange = fi[0].options['tracer'], fi[0].options['zrange']
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

        chain_fn = compressed_fn('chain_shapefit_power_syst-rotation-hod-photo', tracer, zrange, version=version, klim=klim)
        profiles_fn = compressed_fn('profiles_shapefit_power_syst-rotation-hod-photo', tracer, zrange, version=version, klim=klim)
        if True: #not os.path.exists(chain_fn) or not os.path.exists(profiles_fn):
            from desi_y1_files import get_data_file_manager
            dfm = get_data_file_manager(conf='unblinded')
            options = dict(get_fs_baseline_fit_setup(tracer, zrange=zrange), version=version, template='shapefit-qisoqap', emulator=False)
            if klim is not None: options['lim'] = klim
            # TODO
            options['syst'] = 'rotation-hod-photo'
            fchains = list(dfm.select(id='chains_full_shape_y1', **options, ignore=True))
            fprofiles = dfm.get(id='profiles_full_shape_y1', **options, ignore=True)
            chain = load_chain(fchains)
            #chain.attrs['observable'].pop('nowindow', None)
            chain.save(chain_fn)
            profiles = fprofiles.load()
            #profiles.attrs['observable'].pop('nowindow', None)
            profiles.save(profiles_fn)
        chain = Chain.load(chain_fn)
        profiles = Profiles.load(profiles_fn)
        if return_profiles:
            return chain, profiles
        return chain

    def load_fisher(tracer, zrange, klim):
        import os
        import numpy as np
        from desilike import LikelihoodFisher

        fisher_fn = compressed_fn('gaussian_shapefit_power_syst-rotation-hod-photo', tracer, zrange, klim=klim, version=version)
        covstat_fn = compressed_fn('covariance_shapefit_power_syst-rotation-hod-photo', tracer, zrange, version=version)
        forfit_fn = compressed_fn('forfit_shapefit_power_syst-rotation-hod-photo', tracer, zrange, version=version)
        chain, profiles = load_chain(tracer, zrange, klim=klim, version=version, return_profiles=True)
        fisher = chain.to_fisher(params=['qiso', 'qap', 'df', 'dm'])
        fisher = fisher.clone(center=profiles.bestfit.choice(index='argmax', params=fisher.params(), return_type='nparray'))  # to avoid projection effects
        fisher.attrs.update(chain.attrs)
        fisher.attrs.update(profiles.attrs)
        fisher.save(fisher_fn)
        fisher_stat = fisher
        projs = fisher_stat.params().basenames()
        mean = fisher_stat.mean(params=projs)
        cov = fisher_stat.covariance(params=projs)
        covstat = ObservableCovariance(cov, observables=[{'value': mean, 'projs': projs, 'name': 'shapefit', 'attrs': {name: fisher_stat.attrs[name] for name in ['zeff']}}])
        covstat.save(covstat_fn)
        covstat.save(forfit_fn)

    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(1.1, 1.6)], 'QSO': [(0.8, 2.1)]}
    for tracer, zranges in list_zrange.items():
        for iz, zrange in enumerate(zranges):
            #if not(tracer == 'LRG' and zrange == (0.4, 0.6)): continue
            for klim in [[0.02, 0.2, 0.005], [0.02, 0.12, 0.005]][:1]:
                load_fisher(tracer, zrange, klim={ell: klim for ell in [0, 2]})


def prepare_shapefit_bao(version='v1.5', klim=None):

    from desilike.observables import ObservableArray, ObservableCovariance

    def load_chain(tracer, zrange, klim=None, version=version, return_profiles=False):
        import os
        from desilike.samples import Chain, Profiles

        def load_chain(fi, burnin=0.5):
            tracer, zrange = fi[0].options['tracer'], fi[0].options['zrange']
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

        chain_fn = compressed_fn('chain_shapefit_power+bao-recon_syst-rotation-hod-photo', tracer, zrange, version=version, klim=klim)
        profiles_fn = compressed_fn('profiles_shapefit_power+bao-recon_syst-rotation-hod-photo', tracer, zrange, version=version, klim=klim)
        if True: #not os.path.exists(chain_fn) or not os.path.exists(profiles_fn):
            from desi_y1_files import get_data_file_manager
            dfm = get_data_file_manager(conf='unblinded')
            options = dict(get_fs_baseline_fit_setup(tracer, zrange=zrange), version=version, template='shapefit-qisoqap', emulator=False)
            options['observable'] = 'power+bao-recon'
            if klim is not None: options['lim'] = klim
            # TODO
            options['syst'] = 'rotation-hod-photo'
            fchains = list(dfm.select(id='chains_full_shape_y1', **options, ignore=True))
            fprofiles = dfm.get(id='profiles_full_shape_y1', **options, ignore=True)
            chain = load_chain(fchains)
            #chain.attrs['observable'].pop('nowindow', None)
            chain.save(chain_fn)
            profiles = fprofiles.load()
            #profiles.attrs['observable'].pop('nowindow', None)
            profiles.save(profiles_fn)
        chain = Chain.load(chain_fn)
        profiles = Profiles.load(profiles_fn)
        if return_profiles:
            return chain, profiles
        return chain

    def load_fisher(tracer, zrange, klim):
        import os
        import numpy as np
        from desilike import LikelihoodFisher

        fisher_fn = compressed_fn('gaussian_shapefit_power+bao-recon_syst-rotation-hod-photo', tracer, zrange, klim=klim, version=version)
        covstat_fn = compressed_fn('covariance_shapefit_power+bao-recon_syst-rotation-hod-photo', tracer, zrange, version=version)
        forfit_fn = compressed_fn('forfit_shapefit_power+bao-recon_syst-rotation-hod-photo', tracer, zrange, version=version)
        chain, profiles = load_chain(tracer, zrange, klim=klim, version=version, return_profiles=True)
        fisher = chain.to_fisher(params=['qiso', 'qap', 'df', 'dm'])
        fisher = fisher.clone(center=profiles.bestfit.choice(index='argmax', params=fisher.params(), return_type='nparray'))  # to avoid projection effects
        fisher.attrs.update(chain.attrs)
        fisher.attrs.update(profiles.attrs)
        fisher.save(fisher_fn)
        fisher_stat = fisher
        projs = fisher_stat.params().basenames()
        mean = fisher_stat.mean(params=projs)
        cov = fisher_stat.covariance(params=projs)
        covstat = ObservableCovariance(cov, observables=[{'value': mean, 'projs': projs, 'name': 'shapefit', 'attrs': {name: fisher_stat.attrs[name] for name in ['zeff']}}])
        covstat.save(covstat_fn)
        covstat.save(forfit_fn)

    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(1.1, 1.6)], 'QSO': [(0.8, 2.1)]}
    for tracer, zranges in list_zrange.items():
        for iz, zrange in enumerate(zranges):
            #if not(tracer == 'LRG' and zrange == (0.4, 0.6)): continue
            for klim in [[0.02, 0.2, 0.005]]:
                load_fisher(tracer, zrange, klim={ell: klim for ell in [0, 2]})


def get_lim(observable, tracer, zrange=None):
    if observable == 'power':
        return (0.02, 0.2), (0, 2)
    if observable == 'correlation':
        return (30., 150.), (0, 2)
    if observable == 'correlation-recon':
        ells = (0, 2)
        if tracer.startswith('BGS') or (tracer.startswith('ELG') and 'LRG' not in tracer and tuple(zrange) == (0.8, 1.1)) or tracer.startswith('QSO'):
            ells = (0,)
        return (80., 120.), ells


def prepare_forfit_data(only_2pt=False):
    from pathlib import Path
    import numpy as np

    from desilike import LikelihoodFisher
    from desilike.observables import ObservableArray, ObservableCovariance
    from desi_y1_files import get_data_file_manager

    dfm = get_data_file_manager(conf='unblinded')
    version = 'v1.5'

    if not only_2pt:
        fcov = dfm.get(id='forfit_y1', tracer='Lya', version=version, observable=['bao-recon'])
        tracer, zrange = fcov.options['tracer'], fcov.options['zrange']
        """
        fisher_bao = LikelihoodFisher.load(compressed_fn('gaussian_bao-recon', tracer, zrange, version=version))
        projs = fisher_bao.params().basenames()
        mean = fisher_bao.mean(params=projs)
        cov = fisher_bao.covariance(params=projs)
        covariance = ObservableCovariance(cov, observables=[{'value': mean, 'projs': projs, 'name': 'bao-recon', 'attrs': dict(fisher_bao.attrs)}])
        """
        covariance = ObservableCovariance.load(compressed_fn('covariance_stat_bao-recon', tracer, zrange, version=version))  # no syst. anyway
        covariance.save(fcov)

    observables = ['power', 'power+correlation-recon'] + (['power+bao-recon', 'shapefit+bao-recon'] if not only_2pt else [])
    tracer = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']

    for fcov in dfm.select(id='forfit_y1', tracer=tracer, version=version, observable=observables).iter(intersection=False):
        observable = fcov.options['observable']
        print(observable, fcov.options)
        tracer, zrange = fcov.options['tracer'], fcov.options['zrange']
        if 'shapefit' in observable and 'ELG' in tracer and zrange == (0.8, 1.1): continue
        cov_options = dict(fcov.options)
        cov_options.pop('version')
        if 'shapefit' not in observable: cov_options.pop('klim')
        cov_options['source'] = 'ezmock'
        if 'power' in observable:
            rotated = '_rotated'
            cov_options['cut'] = ('theta', 0.05)
        else:
            rotated = ''
            cov_options['cut'] = None
        covariance = dfm.get(id='covariance{}_y1'.format(rotated), **cov_options, ignore=True).load()
        observables = [observable.name for observable in covariance.observables()]

        for observable in ['correlation-recon', 'correlation']:
            xlim, ells = get_lim(observable, tracer, zrange=fcov.options['zrange'])
            covariance = covariance.select(observables=observable, xlim=xlim, projs=list(ells), select_projs=True)
        xlim, ells = get_lim('power', tracer)

        if 'power' in observables:
            klim = fcov.options['klim']
            ells = list(klim)
            klim = klim[0]
            kw_power = dict(observables='power', xlim=xlim, projs=list(ells), select_projs=True)
            covariance = covariance.select(**kw_power)

        factor = {('BGS_BRIGHT-21.5', (0.1, 0.4)): 1.39, ('LRG', (0.4, 0.6)): 1.15, ('LRG', (0.6, 0.8)): 1.15, ('LRG', (0.8, 1.1)): 1.22, ('ELG_LOPnotqso', (0.8, 1.1)): 1.25, ('ELG_LOPnotqso', (1.1, 1.6)): 1.29, ('QSO', (0.8, 2.1)): 1.11}[(tracer, zrange)]
        if 'power' in observables:
            factor *= covariance.percival2014_factor(7) / covariance.hartlap2007_factor()
        covariance.log_info('Scaling covariance by {:.3f}.'.format(factor))
        covariance = covariance.clone(value=covariance.view() * factor)
        covariance.nobs = None

        if 'bao-recon' in observables:
            # EZmock power+bao-recon or shapefit+bao-recon
            # Get post_bao alpha cov from posterior
            covariance_bao = ObservableCovariance.load(compressed_fn('covariance_stat_bao-recon', tracer, zrange, version=version))
            observable_bao = covariance.observables('bao-recon')
            value = covariance_bao.view(projs=observable_bao.projs)
            std_bao = np.diag(value)**0.5
            corr_bao = value / (std_bao[:, None] * std_bao)

            # Replace post_bao part in the covariance matrix
            value = covariance.view()
            std = np.diag(value)**0.5
            corr = value / (std[:, None] * std)
            index = covariance._index(observables=observable_bao)
            std[index] = std_bao
            corr[np.ix_(index, index)] = corr_bao
            covariance = covariance.clone(value=corr * (std[:, None] * std))
            assert covariance_bao.observables('bao-recon').projs == observable_bao.projs
            index = covariance._observable_index(observables=observable_bao.name)
            covariance._observables[index] = covariance_bao.observables('bao-recon')
        
        if 'shapefit' in observables:
            # shapefit+bao-recon
            """
            fisher_shapefit = LikelihoodFisher.load(compressed_fn('gaussian_shapefit', tracer, zrange, version=version, klim=fcov.options['klim']))
            observable_shapefit = covariance.observables('shapefit')
            """
            covariance_shapefit = ObservableCovariance.load(compressed_fn('covariance_shapefit_power_syst-rotation-hod-photo', tracer, zrange, version=version))
            observable_shapefit = covariance.observables('shapefit')
            value = covariance_shapefit.view(projs=observable_shapefit.projs)
            std_shapefit = np.diag(value)**0.5
            corr_shapefit = value / (std_shapefit[:, None] * std_shapefit)

            # Replace ShapeFit part in the covariance matrix
            value = covariance.view()
            std = np.diag(value)**0.5
            corr = value / (std[:, None] * std)
            index = covariance._index(observables=observable_shapefit)
            std[index] = std_shapefit
            corr[np.ix_(index, index)] = corr_shapefit
            covariance = covariance.clone(value=corr * (std[:, None] * std))
            """
            observable = ObservableArray(**{'value': fisher_shapefit.mean(params=observable_shapefit.projs), 'projs': observable_shapefit.projs, 'name': 'shapefit'})
            for name in ['zeff']:
                observable.attrs[name] = fisher_bao.attrs[name]
            index = covariance._observable_index(observables=observable.name)
            covariance._observables[index] = observable
            """
            assert covariance_shapefit.observables('shapefit').projs == observable_shapefit.projs
            index = covariance._observable_index(observables=observable_shapefit.name)
            covariance._observables[index] = covariance_shapefit.observables('shapefit')

        if 'power' in observables:
            # Covariance
            observable = covariance.observables('power')
            index = covariance._index(observables=observable, concatenate=True)
            value = covariance._value[np.ix_(index, index)]
            cov_options['source'] = 'syst'
            syst = dfm.get(id='covariance_syst{}_y1'.format(rotated), **{**cov_options, 'syst': 'rotation', 'version': version, 'observable': 'power'}, ignore=True).clone(ro=None).load()
            if 'rotation' in fcov.options['syst']:
                value += syst.select(**kw_power)
            covariance.attrs['rotation'] = syst.attrs['rotation']
            index_template = syst._index(**{name: kw_power[name] for name in ['xlim', 'projs']})
            covariance.attrs['rotation']['templates'] = [template[index_template] for template in covariance.attrs['rotation']['templates']]
            syst = dfm.get(id='covariance_syst{}_y1'.format(rotated), **{**cov_options, 'syst': 'photo', 'observable': 'power'}, ignore=True).load()
            if 'photo' in fcov.options['syst']:
                value += syst.select(**kw_power)
            covariance.attrs['photo'] = syst.attrs['photo']
            index_template = syst._index(**{name: kw_power[name] for name in ['xlim', 'projs']})
            covariance.attrs['photo']['templates'] = [template[index_template] for template in covariance.attrs['photo']['templates']]
            if 'hod' in fcov.options['syst']:
                syst = dfm.get(id='covariance_syst{}_y1'.format(rotated), **{**cov_options, 'syst': 'hod', 'version': version, 'observable': 'power'}, ignore=True).load()
                if kw_power['projs'] == [0, 2, 4]:  # monopole, quadrupole only in HOD syst
                    tmp = np.zeros_like(value)
                    syst = syst.select(**(kw_power | {'projs': [0, 2]})).view()
                    tmp[:syst.shape[0],:syst.shape[1]] = syst
                    syst = tmp
                else:
                    syst = syst.select(**kw_power)
                value += syst
            covariance._value[np.ix_(index, index)] = value

            options = {**fcov.options, **cov_options, **get_baseline_2pt_setup(tracer=tracer, zrange=zrange, observable='power', recon=False)}
            power = dfm.get(id='power_rotated_corrected_y1', **options, ignore=True).load()
            xlim, ells = kw_power['xlim'], kw_power['projs']
            power = power.select(xlim + (0.005,))
            observable = ObservableArray(**{'value': power(ells, complex=False), 'x': [power.k] * len(ells), 'edges': [power.edges[0]] * len(ells), 'weights': [power.nmodes] * len(ells), 'projs': ells, 'name': 'power'})
            wmatrix = dfm.get(id='wmatrix_power_rotated_y1', **options, ignore=True).load()
            wmatrix.select_x(xoutlim=xlim)
            wmatrix.select_proj(projsout=[(ell, None) for ell in ells])
            kin = np.arange(0.001, 0.35, 0.001)
            from scipy import linalg

            def matrix_lininterp(xin, xout):
                # Matrix for linear interpolation
                toret = np.zeros((len(xin), len(xout)), dtype='f8')
                for iout, xout in enumerate(xout):
                    iin = np.searchsorted(xin, xout, side='right') - 1
                    if 0 <= iin < len(xin) - 1:
                        frac = (xout - xin[iin]) / (xin[iin + 1] - xin[iin])
                        toret[iin, iout] = 1. - frac
                        toret[iin + 1, iout] = frac
                    elif np.isclose(xout, xin[-1]):
                        toret[iin, iout] = 1.
                return toret
            
            rebin = linalg.block_diag(*[matrix_lininterp(kin, xin) for xin in wmatrix.xin])
            matrix = wmatrix.value.T.dot(rebin.T)
            observable.attrs.update({'zeff': wmatrix.attrs['zeff'], 'klim': xlim, 'shotnoise': power.shotnoise, 'wmatrix': matrix, 'kin': kin, 'ellsin': [proj.ell for proj in wmatrix.projsin], 'wshotnoise': np.concatenate(wmatrix.vectorout, axis=0)})
            index = covariance._observable_index(observables=observable.name)
            covariance._observables[index] = observable

        if 'bao-recon' in observables:
            # EZmock power+bao-recon or shapefit+bao-recon
            # Get post_bao alpha cov from posterior
            syst = ObservableCovariance.load(compressed_fn('covariance_syst_bao-recon', tracer, zrange, version=version)).view(observables='bao-recon', projs=observable_bao.projs)
            index = covariance._index(observables='bao-recon', concatenate=True)
            value = covariance._value[np.ix_(index, index)]
            value += syst
            covariance._value[np.ix_(index, index)] = value

        for observable in [obs.name for obs in covariance.observables()]:
            if observable in ['correlation', 'correlation-recon']:
                options = {**fcov.options, **cov_options, **get_baseline_2pt_setup(tracer=tracer, zrange=cov_options['zrange'], observable='correlation', recon='recon' in observable)}
                corr = dfm.get(id='{}_y1'.format(observable.replace('-', '_')), **options, ignore=True).load()
                xlim, ells = get_lim(observable, tracer, zrange=options['zrange'])
                corr = corr.select(xlim + (4.,))
                weights = np.mean(corr.R1R2.normalized_wcounts(), axis=-1)  # mean over mu
                sep, xi = corr(ell=ells, ignore_nan=True, return_sep=True, return_std=False)
                observable = ObservableArray(**{'value': xi, 'x': [sep] * len(ells), 'edges': [corr.edges[0]] * len(ells), 'projs': ells, 'name': observable})
                observable.attrs.update({'zeff': corr.D1D2.attrs['zeff'], 'slim': xlim, 'R1R2': {'sedges': corr.R1R2.edges[0], 'muedges': corr.R1R2.edges[1], 'wcounts': corr.R1R2.wcounts}})
                index = covariance._observable_index(observables=observable.name)
                covariance._observables[index] = observable

        assert sum(observable.size for observable in covariance._observables) == covariance._value.shape[0]
        fcov.save(covariance)


def prepare_forfit_mock(of=None):
    from pathlib import Path
    import numpy as np

    from desilike.observables import ObservableArray
    from desilike import LikelihoodFisher
    from desi_y1_files import get_abacus_file_manager, get_ez_file_manager

    tracer = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']

    if of is None: of = ['abacus', 'ez']
    
    def run(fm, fs, of='ez'):
        if of:
            of = '_' + of
        for fcov in fs:
            cov_options = dict(fcov.options)
            cov_options.pop('version')
            cov_options.pop('klim')
            cov_options['source'] = 'ezmock'
            if 'power' in cov_options['observable']:
                rotated = '_rotated'
                cov_options['cut'] = ('theta', 0.05)
            else:
                rotated = ''
                cov_options['cut'] = None
            tracer, zrange = cov_options['tracer'], cov_options['zrange']
            #print('covariance{}{}_y1'.format(rotated, of), cov_options)
            covariance = fm.get(id='covariance{}{}_y1'.format(rotated, of), **cov_options, ignore=True).load()
            observables = [observable.name for observable in covariance.observables()]

            xlim, ells = get_lim('power', tracer)
            if 'power' in observables:
                klim = fcov.options['klim']
                ells = list(klim)
                klim = klim[0]
                kw_power = dict(observables='power', xlim=xlim, projs=list(ells), select_projs=True)
                covariance = covariance.select(**kw_power)

            factor = {('BGS_BRIGHT-21.5', (0.1, 0.4)): 1.39, ('LRG', (0.4, 0.6)): 1.15, ('LRG', (0.6, 0.8)): 1.15, ('LRG', (0.8, 1.1)): 1.22, ('ELG_LOPnotqso', (0.8, 1.1)): 1.25, ('ELG_LOPnotqso', (1.1, 1.6)): 1.29, ('QSO', (0.8, 2.1)): 1.11}[(tracer, zrange)]
            if 'power' in observables:
                factor *= covariance.percival2014_factor(7) / covariance.hartlap2007_factor()
            covariance.log_info('Scaling covariance by {:.3f}.'.format(factor))
            covariance = covariance.clone(value=covariance.view() * factor)
            covariance.nobs = None

            if 'power' in observables:
                # Covariance
                observable = covariance.observables('power')
                index = covariance._index(observables=['power'], concatenate=True)
                value = covariance._value[np.ix_(index, index)]
                cov_options['source'] = 'syst'
                syst = fm.get(id='covariance_syst{}{}_y1'.format(rotated, of), **{**cov_options, 'syst': 'rotation', 'observable': 'power'}, ignore=True).load()
                if 'rotation' in fcov.options['syst']:
                    value += syst.select(**kw_power)
                covariance.attrs['rotation'] = syst.attrs['rotation']
                index_template = syst._index(**{name: kw_power[name] for name in ['xlim', 'projs']})
                covariance.attrs['rotation']['templates'] = [template[index_template] for template in covariance.attrs['rotation']['templates']]
                syst = fm.get(id='covariance_syst{}{}_y1'.format(rotated, of), **{**cov_options, 'syst': 'photo', 'observable': 'power'}, ignore=True).load()
                if 'photo' in fcov.options['syst']:
                    value += syst.select(**kw_power)
                covariance.attrs['photo'] = syst.attrs['photo']
                index_template = syst._index(**{name: kw_power[name] for name in ['xlim', 'projs']})
                covariance.attrs['photo']['templates'] = [template[index_template] for template in covariance.attrs['photo']['templates']]
                if 'hod' in fcov.options['syst']:
                    syst = fm.get(id='covariance_syst{}{}_y1'.format(rotated, of), **{**cov_options, 'syst': 'hod', 'observable': 'power'}, ignore=True).load().select(**kw_power)
                    value += syst
                covariance._value[np.ix_(index, index)] = value
            assert sum(observable.size for observable in covariance._observables) == covariance._value.shape[0]
            fcov.save(covariance)

    if of == 'abacus':
        fm = get_abacus_file_manager()
        #fs = fm.select(id='forfit_abacus_y1', region='GCcomb',
        #               tracer=['BGS_BRIGHT-21.5'], version='v1', observable=['power']).iter(intersection=False)
        fs = fm.select(id='forfit_abacus_y1', region='GCcomb',
                       tracer=tracer, version='v1', observable=['power']).iter(intersection=False)
        run(fm, fs, of='abacus')
    if of == 'ez':
        fm = get_ez_file_manager()
        fs = fm.select(id='forfit_ez_y1', region='GCcomb',
                       tracer=tracer, version='v1', observable=['power']).iter(intersection=False)
        run(fm, fs, of='ez')
        

if __name__ == '__main__':

    """Look at y1_covariance.py for instructions."""

    from desipipe import setup_logging

    setup_logging()

    prepare_shapefit_bao()
    exit()

    only_2pt = False
    of = None
    if only_2pt:
        for rotated in [False, True]:
            ##export_photo()
            #fit_ric(of=of, rotated=rotated)
            #fit_photo(of=of, rotated=rotated)
            #correct(of=of, rotated=rotated)
            export_systematic_covariance(of=of, rotated=rotated)
        prepare_forfit_data(only_2pt=True)
        prepare_forfit_mock()
    else:
        prepare_bao()
        prepare_shapefit()
        prepare_shapefit_bao()
        prepare_forfit_data(only_2pt=False)
            
        

