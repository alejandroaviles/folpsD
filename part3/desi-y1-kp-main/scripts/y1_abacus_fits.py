from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
from y1_data_fits_tools import profile, sample, get_footprint, get_observable_likelihood

setup_logging()

queue = Queue('y1_abacus_fits')
queue.clear(kill=False)

output, error = '_sbatch_abacus/slurm-%j.out', '_sbatch_abacus/slurm-%j.err'
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm_footprint = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00', mpiprocs_per_worker=6, nodes_per_worker=0.2, output=output, error=error))
tm_profile = tm.clone(scheduler=dict(max_workers=100), provider=dict(provider='nersc', time='00:20:00', mpiprocs_per_worker=6, nodes_per_worker=0.2, output=output, error=error))
tm_sample = tm.clone(scheduler=dict(max_workers=60), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=32, nodes_per_worker=0.5, output=output, error=error))
tm_emulate = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='00:45:00', mpiprocs_per_worker=64, output=output, error=error))


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
#@tm_sample.python_app
def sample(output, sample=sample, resume=False, **kwargs):
    from desilike import setup_logging
    setup_logging()
    sample(output, sample=sample, resume=resume, **kwargs)
    return output


if __name__ == '__main__':

    from desi_y1_files import get_data_file_manager, get_abacus_file_manager

    todo = ['bao', 'bao_recon', 'full_shape'][2:]

    regions = ['GCcomb']
    #version = 'v3_1_lrg+elg'
    #tracers = ['LRG+ELG_LOPnotqso']
    version = 'v4_2'
    tracers = ['LRG', 'ELG_LOPnotqso', 'QSO']
    #version = 'v1'
    #tracers = ['BGS_BRIGHT-21.5']
    observables = ['power', 'correlation'][:1]
    mean = True
    
    cut = ('theta', 0.05)
    #cut = None
    #fa = ['altmtl']
    fa = ['complete']

    fm = get_abacus_file_manager()
    dfm = get_data_file_manager()
    fm.save('y1_abacus_files.yaml', replace_environ=False)  # exports files as a *.yaml file

    def get_options(fit_type):
        if 'full_shape' in fit_type:
            cuts = [None] #[cut]
            list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'QSO': [(0.8, 2.1)]}
            toret = []
            cuts = [cut]
            for tracer in tracers:
                for zrange in list_zrange[tracer]:
                    toret += list(fm.select(id='profiles_{}_mean_abacus_y1'.format(fit_type), version=version, tracer=tracer, zrange=zrange, region=['GCcomb'], observable='power', weighting='default_FKP', template='shapefit-qisoqap', syst='rotation', theory='reptvelocileptors', covmatrix='ezmock', wmatrix='rotated', cut=cuts).iter_options(intersection=False))
    
        else:
            from desi_y1_files.file_manager import get_bao_baseline_fit_setup, list_zrange
            toret = []
            for observable in observables:
                for tracer in tracers:
                    for zrange in list_zrange[tracer]:
                        list_options = {}
                        ref_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable=observable, recon=True, iso=False)
                        #ref_options['broadband'] = 'fixed'
                        list_options['baseline'] = ('bao_recon', {**ref_options, 'region': 'GCcomb'})
                        toret += [options for ft, options in list_options.values() if ft == fit_type]
        for options in toret:
            options['version'] = version
            options['fa'] = fa
            options['precscale'] = [1]
        if not mean:
            toret = [{**options, 'imock': iimock} for options in toret for iimock in list(range(25))[1:]]
        return toret

    mean = 'mean_' if mean else ''
    emulator_fns = {}
    for fit_type in todo:
        for options in get_options(fit_type):
            observable = options['observable']
            rotated = '_rotated' if 'rotated' in options.get('wmatrix', '') else ''
            fdata = list(fm.select(id='{}{}{}_abacus_y1'.format(observable, '_recon' if 'recon' in fit_type else '', rotated), **options, ignore=True))
            if any(not fd.exists() for fd in fdata): continue
            fwmatrix = None
            if observable == 'power':
                fwmatrix = fm.get(id='wmatrix_power_merged{}_abacus_y1'.format(rotated), **options, ignore=True)
                if not fwmatrix.exists(): continue
            # For now, for power spectrum let's just take pre- for post-
            #rotated = ''
            syst = False
            cov_options = {**options, 'source': options['covmatrix'], 'marg': 'syst' if syst else False, 'observable': '{}-recon'.format(observable) if 'recon' in fit_type else observable}
            cov_options.pop('version')
            if rotated:
                cov_options.pop('source')
                cov_options['klim'] = cov_options.pop('lim')
                fcovariance = fm.get(id='forfit_abacus_y1', **cov_options, ignore=True)
            else:
                cov_options.pop('version')
                fcovariance = fm.get(id='covariance_abacus_y1', **cov_options, ignore=True)
            if not fcovariance.exists(): continue

            ffootprint = None
            kwargs = dict(data=fdata, covariance=fcovariance, footprint=ffootprint,
                          theory_name=options['theory'], template_name=options['template'], wmatrix=fwmatrix)
            femulator = None
            if 'full_shape' in fit_type and options['template'] != 'fixed':
                femulator = fm.get(id='emulator_full_shape_mean_abacus_y1', **{**options, 'precscale': 1}, ignore=True)
                if femulator not in emulator_fns:  # wondering if I should not add this argument-based filter to desipipe
                    #emulator_fns[femulator] = femulator
                    emulator_fns[femulator] = emulate(emulator_fn=femulator, **kwargs)
                femulator = emulator_fns[femulator]
            #print(options)
            fi = fm.get(id='profiles_{}_{}abacus_y1'.format(fit_type, mean), **options, ignore=True)
            profile(fi, emulator_fn=femulator, **kwargs)
            if mean: sample(fm.select(id='chains_{}_{}abacus_y1'.format(fit_type, mean), **options), emulator_fn=femulator, **kwargs)

    #spawn(queue, spawn=True)