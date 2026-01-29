from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
from y1_data_fits_tools import profile, sample, get_footprint, get_observable_likelihood

setup_logging()

queue = Queue('y1_kp4_unblinding_fits')
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi')
output, error = '_sbatch_kp4_unblinding_fits/slurm-%j.out', '_sbatch_kp4_unblinding_fits/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ)
tm_footprint = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00', mpiprocs_per_worker=6, nodes_per_worker=0.2, output=output, error=error))
tm_profile = tm.clone(scheduler=dict(max_workers=16), provider=dict(provider='nersc', time='00:20:00', mpiprocs_per_worker=6, nodes_per_worker=0.2, output=output, error=error))
tm_sample = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='00:20:00', mpiprocs_per_worker=32, nodes_per_worker=0.5, output=output, error=error, stop_after=1))
tm_emulate = tm.clone(scheduler=dict(max_workers=4), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=64, output=output, error=error))


@tm_footprint.python_app
def compute_footprint(all_data, output, get_footprint=get_footprint, **kwargs):
    """Return footprints (for all redshift slices), specifying redshift density nbar and area, using Y1 data."""
    get_footprint(all_data, output=output, **kwargs)
    return output

# Emulate theory for faster inference
@tm_emulate.python_app
def emulate(emulator_fn=None, get_observable_likelihood=get_observable_likelihood, **kwargs):
    from desilike import setup_logging
    setup_logging()
    get_observable_likelihood(save_emulator=True, emulator_fn=emulator_fn, **kwargs)
    return emulator_fn

# Profile (maximize) posterior
@tm_profile.python_app
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
    sample(output, sample=sample, resume=resume, **kwargs)
    return output


if __name__ == '__main__':

    from desi_y1_files import get_data_file_manager
    from desi_y1_files.file_manager import list_zrange

    version = 'v1'
    tracers = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']

    fm = get_data_file_manager(conf='wrong')
    fm.save('y1_data_files.yaml', replace_environ=False)  # exports files as a *.yaml file

    def get_list_options(tracer, zrange):
        from desi_y1_files.file_manager import get_bao_baseline_fit_setup
        list_options = {}
        ref_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=False)
        list_options['baseline'] = ('bao_recon', {**ref_options, 'region': 'GCcomb'})
        #list_options['$d\\beta \\in [0.25, 1.75]$'] = ('bao_recon', {**ref_options, 'region': 'GCcomb', 'dbeta': (0.25, 1.75)})
        ref_iso_options = get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=True, iso=True)
        list_options['1D fit'] = ('bao_recon', {**ref_iso_options, 'region': 'GCcomb'})
        list_options['now-qiso'] = ('bao_recon', {**ref_iso_options, 'region': 'GCcomb', 'template': 'bao-now-qiso'})
        list_options['pre-recon'] = ('bao', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='correlation', recon=False, iso=False), 'region': 'GCcomb'})
        list_options['power\nspectrum'] = ('bao_recon', {**get_bao_baseline_fit_setup(tracer, zrange=zrange, observable='power', recon=True, iso=False), 'region': 'GCcomb'})
        list_options['NGC'] = ('bao_recon', {**ref_options, 'region': 'NGC'})
        list_options['polynomial\nbroadband'] = ('bao_recon', {**ref_options, 'broadband': 'power3', 'region': 'GCcomb'})
        list_options['flat prior\non $\Sigma_{s}, \Sigma_{\parallel}, \Sigma_{\perp}$'] = ('bao_recon', {**ref_options, 'sigmas': {'sigmas': None, 'sigmapar': None, 'sigmaper': None}, 'region': 'GCcomb'})
        for fit_type, options in list_options.values(): options['version'] = version
        return list_options
    
    for tracer in tracers:
        for zrange in list_zrange[tracer]:
            for fit_type, options in get_list_options(tracer, zrange).values():
                options['version'] = version
                observable = options['observable']
                fdata = fm.get(id='{}{}_y1'.format(observable, '_recon' if 'recon' in fit_type else ''), **options, ignore=True)
                if not fdata.exists(): continue
                fwmatrix = None
                if observable == 'power':
                    fwmatrix = fm.get(id='wmatrix_power_y1', **options, ignore=True)
                    if not fwmatrix.exists(): continue
                # For now, for power spectrum let's just take pre- for post-
                fcovariance = fm.get(id='covariance_{}{}_y1'.format(observable, '_recon' if 'recon' in fit_type and 'correlation' in observable else ''), **{**options, 'cut': None, 'version': 'v0.6'}, ignore=True)
                if not fcovariance.exists(): continue

                ffootprint = None
                #ffootprint = fm.get(id='footprint_y1', **options, ignore=True)
                #fd = fm.select(id='catalog_data_y1', ignore=['region', 'zrange'], **ffootprint.options)
                #fr = [fm.select(id='catalog_randoms_y1', ignore=['region', 'zrange'], iran=iran, **ffootprint.options) for iran in range(4)]
                # This task automatically be added only once for bao, bao_recon and full_shape fits, as input arguments are the same in all cases
                # We request the covariance, so let's skip it
                #ffootprint = compute_footprint(fd, ffootprint, all_randoms=fr)

                kwargs = dict(data=fdata, covariance=fcovariance, footprint=ffootprint,
                              theory_name=options['theory'], template_name=options['template'], wmatrix=fwmatrix)
                femulator = None
                if 'full_shape' in fit_type and options['template'] != 'fixed':
                    femulator = fm.get(id='emulator_{}_y1'.format(fit_type), **options, ignore=True)
                    if femulator not in emulator_fns:  # wondering if I should not add this argument-based filter to desipipe
                        #emulator_fns[femulator] = emulate(emulator_fn=femulator, **kwargs)
                        emulator_fns[femulator] = femulator
                    femulator = emulator_fns[femulator]
                #print(fm.get(id='profiles_{}_y1'.format(fit_type), **options))
                #print(fm.get(id='profiles_{}_y1'.format(fit_type), **{key: options[key] for key in ['tracer', 'region', 'version', 'weighting', 'sigmas', 'template', 'broadband', 'cut']}))
                #print(options)
                profile(fm.get(id='profiles_{}_y1'.format(fit_type), **options), emulator_fn=femulator, **kwargs)
                sample(fm.select(id='chains_{}_y1'.format(fit_type), **options), emulator_fn=femulator, **kwargs)

    spawn(queue, spawn=True)