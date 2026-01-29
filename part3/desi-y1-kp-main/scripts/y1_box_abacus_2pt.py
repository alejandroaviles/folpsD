from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
import y1_box_2pt_tools

setup_logging()

queue = Queue('y1_box_abacus_2pt')
queue.clear(kill=False)

output, error = '_sbatch_box_abacus/slurm-%j.out', '_sbatch_box_abacus/slurm-%j.err'
environ = Environment('nersc-cosmodesi', command='module swap pyrecon/main pyrecon/mpi')
#environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm_power = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error))
tm_power_wmatrix = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error, stop_after=1))
gpu_nthreads = 4
cpu_nthreads = 64
gpu = True
if gpu:
    nthreads = gpu_nthreads
    tm_corr = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:40:00', mpiprocs_per_worker=1, output=output, error=error, constraint='gpu'))
else:
    nthreads = cpu_nthreads
    tm_corr = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:40:00', mpiprocs_per_worker=64, output=output, error=error))
tm_recon = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:45:00', mpiprocs_per_worker=64, output=output, error=error))
tm_combine = tm.clone(scheduler=dict(max_workers=2), provider=dict(provider='local'))
tm_merge = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=1, nodes_per_worker=0.1, output=output, error=error))


@tm_power.python_app
def compute_power_spectrum(output, data, compute_power_spectrum=y1_box_2pt_tools.compute_power_spectrum, output_wmatrix=None, **kwargs):
    from pypower import setup_logging
    setup_logging()
    compute_power_spectrum(data, output=output, output_wmatrix=output_wmatrix, **kwargs)
    return output


#@tm_power_wmatrix.python_app
def compute_power_spectrum_wmatrix(output, data, compute_power_spectrum=y1_box_2pt_tools.compute_power_spectrum, output_wmatrix=None, **kwargs):
    from pypower import setup_logging
    setup_logging()
    compute_power_spectrum(data, output=output, output_wmatrix=output_wmatrix, **kwargs)
    return output


@tm_corr.python_app
def compute_correlation_function(output, data, compute_correlation_function=y1_box_2pt_tools.compute_correlation_function, gpu=gpu, nthreads=nthreads, **kwargs):
    from pycorr import setup_logging
    setup_logging()
    compute_correlation_function(data, output=output, gpu=gpu, nthreads=nthreads, **kwargs)
    return output


@tm_recon.python_app
def compute_reconstruction(output_data, all_output_randoms, data, compute_reconstruction=y1_box_2pt_tools.compute_reconstruction):
    from pyrecon import setup_logging
    setup_logging()
    compute_reconstruction(data, output_data, all_output_randoms)
    return output_data[0]


@tm_combine.python_app
def postprocess_power_spectrum(power, postprocess_power_spectrum=y1_box_2pt_tools.postprocess_power_spectrum, **kwargs):
    from pypower import setup_logging
    setup_logging()
    postprocess_power_spectrum(power, **kwargs)
    return power


@tm_combine.python_app
def postprocess_correlation_function(correlation, postprocess_correlation_function=y1_box_2pt_tools.postprocess_correlation_function, **kwargs):
    from pypower import setup_logging
    setup_logging()
    postprocess_correlation_function(correlation, **kwargs)
    return correlation


if __name__ == '__main__':

    from desi_y1_files import get_box_abacus_file_manager

    #todo = ['power', 'correlation', 'power_recon', 'correlation_recon']  # reconstruction is run automatically when asking for power_recon / correlation_recon
    #todo = ['symlink']
    #todo = ['correlation']
    todo = ['power', 'correlation'][:1]
    #todo = ['merge']
    #version = 'v1.1'
    #tracers = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'][1:]
    version = 'v0.1'
    tracers = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'][:1]

    fm = get_box_abacus_file_manager()
    fm.save('y1_box_abacus_files.yaml', replace_environ=False)  # exports files as a *.yaml file
    
    imock = list(range(25))
    #imock = [0]
    #for i in range(4, 10): imock.remove(i)
    #imock = list(range(4, 10))

    def compute_merged_region(fi):
        stat_type = fi.filetype
        options, fid = fi.options, fi.id
        compute = compute_power_spectrum_wmatrix
        data = list(fm.select(id='catalog_data_box_abacus_y1', **options, imock=0, ignore=True))
        kwargs = {}
        kwargs['output_wmatrix'] = fm.get('wmatrix_power_merged_box_abacus_y1', **options, ignore=True)
        fi = compute(fi, data, **kwargs)
        return fi

    def compute_region(fi):
        # Compute power spectrum, correlation function, pre/post in a given region
        stat_type = fi.filetype
        recon = 'smoothing_radius' in fi.options
        options, fid = fi.options, fi.id
        compute = {'power': compute_power_spectrum, 'correlation': compute_correlation_function}[stat_type]
        data = list(fm.select(id='catalog_data_box_abacus_y1', **options, ignore=True))
        assert all(dd.exists() for dd in data)
        kwargs = {}
        if recon:
            data_recon = fm.get(id='catalog_data_recon_box_abacus_y1', **options, ignore=True)
            data_recon_modes = [fm.get(id='catalog_data_recon_box_abacus_y1', **{**options, 'mode': mode}, ignore=True) for mode in ['recsym', 'reciso']]
            all_randoms_recon_modes = [fm.select(id='catalog_randoms_recon_box_abacus_y1', **{**options, 'mode': mode}, ignore=True) for mode in ['recsym', 'reciso']]
            data_recon = compute_reconstruction(data_recon_modes, all_randoms_recon_modes, data)  # only run once for all z-ranges and power / correlation
            data, kwargs['all_shifted'] = data_recon, randoms_recon
        fi = compute(fi, data, **kwargs)
        return fi

    for stat_type in todo:
        if stat_type == 'symlink':
            fm.symlink(raise_error=False)

        if stat_type == 'merge':
            for fi in fm.select(id='power_merged_box_abacus_y1', tracer=tracers).iter(intersection=False):
                fid = fi.id
                options = fi.options
                compute_merged_region(fi)
                break

        recon = 'recon' in stat_type
        fid = '{}_box_abacus_y1'.format(stat_type)
        stat_type = stat_type.split('_')[0]
        for fi in fm.select(id=fid, tracer=tracers, version=version, imock=imock).iter(intersection=False):
            options = fi.options
            compute_region(fi)

    #spawn(queue, spawn=True)