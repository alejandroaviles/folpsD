import os
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
from y1_data_fits_tools import profile, sample, importance, get_footprint, get_observable_likelihood

setup_logging()

queue = Queue('y1_data_fits')
queue.clear(kill=False)

#environ = Environment('nersc-cosmodesi', command=['export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID'])
environ = Environment('nersc-cosmodesi')
output, error = '_sbatch/slurm-%j.out', '_sbatch/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ)
tm_footprint = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00', mpiprocs_per_worker=6, nodes_per_worker=0.2, output=output, error=error))
tm_profile = tm.clone(scheduler=dict(max_workers=16), provider=dict(provider='nersc', time='00:20:00', mpiprocs_per_worker=10, nodes_per_worker=0.15, output=output, error=error))
tm_sample = tm.clone(scheduler=dict(max_workers=50), provider=dict(provider='nersc', time='06:00:00', mpiprocs_per_worker=8, nodes_per_worker=0.15, output=output, error=error))
tm_emulate = tm.clone(scheduler=dict(max_workers=4), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=64, output=output, error=error))
tm_importance = tm.clone(scheduler=dict(max_workers=50), provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=8, nodes_per_worker=0.2, output=output, error=error))


@tm_footprint.python_app
def compute_footprint(all_data, output, get_footprint=get_footprint, **kwargs):
    """Return footprints (for all redshift slices), specifying redshift density nbar and area, using Y1 data."""
    get_footprint(all_data, output=output, **kwargs)
    return output

# Emulate theory for faster inference
#@tm_emulate.python_app
def emulate(emulator_fn=None, get_observable_likelihood=get_observable_likelihood, **kwargs):
    from desilike import setup_logging
    setup_logging()
    get_observable_likelihood(save_emulator=True, emulator_fn=emulator_fn, **kwargs)
    return emulator_fn

# Profile (maximize) posterior
#@tm_profile.python_app
def profile(output, profile=profile, **kwargs):
    from desilike import setup_logging
    setup_logging()
    profile(output, **kwargs)
    return output

# Sample posterior
@tm_sample.python_app
def sample(output, sample=sample, resume=False, **kwargs):
    from desilike import setup_logging
    setup_logging()
    sample(output, resume=resume, **kwargs)
    return output


def importance(chains, output, importance=importance, **kwargs):
    from desilike import setup_logging
    setup_logging()
    importance(chains, output, **kwargs)
    return output


if __name__ == '__main__':

    from desi_y1_files import get_data_file_manager

    todo = ['full_shape']

    version = 'v1.5'
    #tracers = ['BGS_BRIGHT-21.5', 'LRG', 'LRG+ELG_LOPnotqso', 'ELG_LOPnotqso', 'QSO']
    #version = 'v1'
    tracers = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']
    cut = ('theta', 0.05)

    fm = get_data_file_manager(conf='unblinded')
    fm.save('y1_data_files.yaml', replace_environ=False)  # exports files as a *.yaml file
    dfm = get_data_file_manager(conf='unblinded')
    
    def get_options(fit_type):
        if 'full_shape' in fit_type:
            list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(1.1, 1.6)], 'QSO': [(0.8, 2.1)]}
            #list_zrange = {'QSO': [(0.8, 2.1)]}
            #list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)]}
            toret = []
            cuts = [cut]
            for tracer in tracers:
                for zrange in list_zrange[tracer]:
                    #toret += list(fm.select(id='profiles_{}_y1'.format(fit_type), version=version, tracer=tracer, zrange=zrange, region=['GCcomb'], observable='power', weighting='default_FKP', template='shapefit-qisoqap', syst='rotation-hod-photo', theory='reptvelocileptors', covmatrix='ezmock', wmatrix='rotated', emulator=False, lim={0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]}, cut=cuts).iter_options(intersection=False))
                    toret += list(fm.select(id='profiles_{}_y1'.format(fit_type), version=version, tracer=tracer, zrange=zrange, region=['GCcomb'], observable='power+bao-recon', weighting='default_FKP', template='shapefit-qisoqap', syst='rotation-hod-photo', theory='reptvelocileptors', covmatrix='ezmock', wmatrix='rotated', emulator=False, lim={0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]}, cut=cuts).iter_options())
                    #toret += list(fm.select(id='profiles_{}_y1'.format(fit_type), version=version, tracer=tracer, zrange=zrange, region=['GCcomb'], observable='power', weighting='default_FKP', template='direct', syst='rotation-hod-photo', theory='reptvelocileptors', covmatrix='ezmock', wmatrix='rotated', emulator=True, lim={0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]}, cut=cuts).iter_options())
            return toret
            
        from desi_y1_files.file_manager import get_bao_baseline_fit_setup, list_zrange
        toret = []
        for tracer in tracers:
            for zrange in list_zrange[tracer]:
                list_options = {}

                ref_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=False)
                #list_options['baseline'] = ('bao_recon', {**ref_options, 'region': 'GCcomb'})
                #list_options['$d\\beta \\in [0.25, 1.75]$'] = ('bao_recon', {**ref_options, 'region': 'GCcomb', 'dbeta': (0.25, 1.75)})
                ref_iso_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=True)
                #list_options['1D fit'] = ('bao_recon', {**ref_iso_options, 'region': 'GCcomb'})

                list_options['now-qiso'] = ('bao_recon', {**ref_iso_options, 'region': 'GCcomb', 'template': 'bao-now-qiso'})
                """
                list_options['pre-recon'] = ('bao', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=False, iso=None), 'region': 'GCcomb'})
                list_options['power\nspectrum'] = ('bao_recon', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='power', recon=True, iso=None), 'region': 'GCcomb'})
                list_options['power\nspectrum 1D'] = ('bao_recon', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='power', recon=True, iso=True), 'region': 'GCcomb'})

                list_options['NGC'] = ('bao_recon', {**ref_options, 'region': 'NGC'})
                list_options['polynomial\nbroadband'] = ('bao_recon', {**ref_options, 'broadband': 'power3', 'region': 'GCcomb'})
                list_options['polynomial\nbroadband iso'] = ('bao_recon', {**ref_iso_options, 'broadband': 'power3', 'region': 'GCcomb'})
                list_options['fixed\nbroadband'] = ('bao_recon', {**ref_options, 'broadband': 'fixed', 'region': 'GCcomb'})
                list_options['fixed\nbroadband iso'] = ('bao_recon', {**ref_iso_options, 'broadband': 'fixed', 'region': 'GCcomb'})
                list_options['flat prior\non $\Sigma_{s}, \Sigma_{\parallel}, \Sigma_{\perp}$'] = ('bao_recon', {**ref_options, 'sigmas': {'sigmas': None, 'sigmapar': None, 'sigmaper': None}, 'region': 'GCcomb'})
                """
                #ref_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=None)
                #list_options['baseline'] = ('bao_recon', {**ref_options, 'region': 'GCcomb', 'covmatrix': 'rascalc'})
                #ref_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='power', recon=False, iso=None)
                #list_options['baseline'] = ('bao', {**ref_options, 'region': 'GCcomb', 'covmatrix': 'ezmock'})
                #ref_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=True)
                #list_options['baseline'] = ('bao_recon', {**ref_options, 'region': 'GCcomb'})
                #list_options = {name: list_options[name] for name in ['1D fit']}
                toret += [options for ft, options in list_options.values() if ft == fit_type]
        for options in toret: options['version'] = version
        return toret

    emulator_fns = {}
    for fit_type in todo:
        cuts = [cut] if 'full_shape' in fit_type else [None]
        for options in get_options(fit_type=fit_type):
            observable = options['observable']
            observables = observable.split('+')
            rotated = '_rotated' if 'rotated' in options.get('wmatrix', '') else ''
            syst = options.get('syst', '')
            #syst = False
            if version == 'v1': options['recon_zrange'] = None
            fdata = fm.get(id='{}{}{}{}_y1'.format(observables[0], '_recon' if 'recon' in fit_type else '', rotated, '_corrected' if 'photo' in syst else ''), **options, ignore=True)
            fdata = (fdata,)
            #print(fdata)
            #if not fdata[0].exists(): continue
            fwmatrix = None
            if observables[0] == 'power':
                fwmatrix = fm.get(id='wmatrix_power{}_y1'.format(rotated), **options, ignore=True)
                if not fwmatrix.exists(): continue
            # For now, for power spectrum let's just take pre- for post-
            cov_options = {**options, 'source': options['covmatrix'], 'observable': '{}-recon'.format(observable) if 'recon' in fit_type else observable}
            if rotated:
                cov_options.pop('source')
                cov_options['klim'] = cov_options.pop('lim')
                fcovariance = fm.get(id='forfit_y1', **cov_options, ignore=True)
                if len(observables) > 1 and observables[1] == 'bao-recon':
                    fdata = fdata + (fcovariance,)
            else:
                cov_options.pop('version')
                fcovariance = fm.get(id='covariance_y1', **cov_options, ignore=True)
            #print(fcovariance, fcovariance.exists())
            #print(fcovariance, fcovariance.exists())
            if not fcovariance.exists(): continue
            ffootprint = None
            kwargs = dict(data=fdata, covariance=fcovariance, footprint=ffootprint, observable_name=observables,
                          theory_name=options['theory'], template_name=options['template'], wmatrix=fwmatrix)
            femulator = None
            if 'full_shape' in fit_type and options.get('emulator', True) and options['template'] != 'fixed' and options['theory'] != 'folpsax':
                femulator = fm.get(id='emulator_{}_y1'.format(fit_type), **options, ignore=True)
                if femulator not in emulator_fns:  # wondering if I should not add this argument-based filter to desipipe
                    emulator_fns[femulator] = emulate(emulator_fn=femulator, **kwargs)
                    #emulator_fns[femulator] = femulator
                femulator = emulator_fns[femulator]
            #print(options)
            fo = fm.get(id='profiles_{}_y1'.format(fit_type), **options, ignore=True)
            #if fo.exists(): continue
            profile(fo, emulator_fn=femulator, **kwargs)
            fo = fm.select(id='chains_{}_y1'.format(fit_type), **options, ignore=True)
            #if not os.path.exists(tmp.filepath.replace('_0.npy', '_stats.txt')):
            #sample(fo, emulator_fn=femulator, resume=True, **kwargs)
            fimp = fm.select(id='chains_{}_importance_y1'.format(fit_type), **options, ignore=True)
            #importance(fo, fimp, **kwargs)

    #spawn(queue, spawn=True)