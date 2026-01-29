from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
import y1_data_2pt_tools

setup_logging()

queue = Queue('y1_kp4_unblinding_2pt2')
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi', command='module swap pyrecon/main pyrecon/mpi')
#environ = Environment('nersc-cosmodesi')
output, error = '_sbatch_kp4_unblinding_2pt/slurm-%j.out', '_sbatch_kp4_unblinding_2pt/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ)
tm_power = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='03:00:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error))
gpu_nthreads = 4
cpu_nthreads = 64
gpu = True
if gpu:
    nthreads = gpu_nthreads
    tm_corr = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='00:10:00', mpiprocs_per_worker=1, output=output, error=error, constraint='gpu'))
else:
    nthreads = cpu_nthreads
    tm_corr = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=64, output=output, error=error))
tm_recon = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=64, output=output, error=error))
tm_combine = tm.clone(scheduler=dict(max_workers=2), provider=dict(provider='local'))
tm_wmatrix = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:20:00', mpiprocs_per_worker=1, nodes_per_worker=0.1, output=output, error=error))


@tm_power.python_app
def compute_power_spectrum(output, data, all_randoms, compute_power_spectrum=y1_data_2pt_tools.compute_power_spectrum, output_wmatrix=None, nthreads=cpu_nthreads, **kwargs):
    from pypower import setup_logging
    setup_logging()
    compute_power_spectrum(data, all_randoms, output=output, output_wmatrix=output_wmatrix, nthreads=nthreads, **kwargs)
    return output


@tm_corr.python_app
def compute_correlation_function(output, data, all_randoms, compute_correlation_function=y1_data_2pt_tools.compute_correlation_function, gpu=gpu, nthreads=nthreads, **kwargs):
    from pycorr import setup_logging
    setup_logging()
    compute_correlation_function(data, all_randoms, output=output, gpu=gpu, nthreads=nthreads, **kwargs)
    return output


@tm_recon.python_app
def compute_reconstruction(output_data, all_output_randoms, data, all_randoms, compute_reconstruction=y1_data_2pt_tools.compute_reconstruction):
    from pyrecon import setup_logging
    setup_logging()
    compute_reconstruction(data, all_randoms, output_data, all_output_randoms)
    return output_data[0]


@tm_combine.python_app
def postprocess_power_spectrum(power, output_power_nodirect=None, postprocess_power_spectrum=y1_data_2pt_tools.postprocess_power_spectrum, **kwargs):
    from pypower import setup_logging
    setup_logging()
    postprocess_power_spectrum(power, output_power_nodirect=output_power_nodirect, **kwargs)
    return power, output_power_nodirect


#@tm_combine.python_app
def postprocess_correlation_function(correlation, postprocess_correlation_function=y1_data_2pt_tools.postprocess_correlation_function, **kwargs):
    from pypower import setup_logging
    setup_logging()
    postprocess_correlation_function(correlation, **kwargs)
    return correlation


@tm_combine.python_app
def combine_regions_power_spectrum(output, inputs, inputs_wmatrix=None, output_wmatrix=None, combine_regions=y1_data_2pt_tools.combine_regions_power_spectrum, **kwargs):
    from pypower import setup_logging
    setup_logging()
    inputs = list(inputs)
    if isinstance(output, (tuple, list)):  # cut, nocut
        for iout, out in enumerate(output):
            if inputs_wmatrix is not None:
                kwargs['inputs_wmatrix'] = [tmp[iout] for tmp in inputs_wmatrix]
                kwargs['output_wmatrix'] = output_wmatrix[iout]
            combine_regions(out, [tmp[iout] for tmp in inputs], **kwargs)
    else:
        combine_regions(output, inputs, inputs_wmatrix=inputs_wmatrix, output_wmatrix=output_wmatrix, **kwargs)


#@tm_combine.python_app
def combine_regions_correlation_function(output, inputs, combine_regions=y1_data_2pt_tools.combine_regions_correlation_function):
    from pycorr import setup_logging
    setup_logging()
    combine_regions(output, inputs)


if __name__ == '__main__':

    from desi_y1_files import get_data_file_manager

    todo = ['power', 'correlation', 'power_recon', 'correlation_recon']  # reconstruction is run automatically when asking for power_recon / correlation_recon
    #todo = ['correlation', 'correlation_recon']
    todo = ['symlink']

    version = 'v1'
    weighting = 'default_FKP'
    regions = ['GCcomb']  # NGC, SGC are computed automatically when requiring GCcomb
    tracers = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']
    cut = ('theta', 0.05)

    fm = get_data_file_manager(conf='unblinded')
    fm.save('y1_data_files.yaml', replace_environ=False)  # exports files as a *.yaml file

    def compute_region(fi):
        # Compute power spectrum, correlation function, pre/post in a given region
        stat_type = fi.filetype
        recon = 'smoothing_radius' in fi.options
        options, fid = fi.options, fi.id
        compute = {'power': compute_power_spectrum, 'correlation': compute_correlation_function}[stat_type]
        data = fm.get(id='catalog_data_y1', **options, ignore=True)
        all_randoms = fm.select(id='catalog_randoms_y1', **options, ignore=True)
        kwargs = {}
        if recon:
            data_recon = fm.get(id='catalog_data_recon_y1', **options, ignore=True)
            randoms_recon = fm.select(id='catalog_randoms_recon_y1', **options, ignore=True)
            data_recon_modes = [fm.get(id='catalog_data_recon_y1', **{**options, 'mode': mode}, ignore=True) for mode in ['recsym', 'reciso']]
            all_randoms_recon_modes = [fm.select(id='catalog_randoms_recon_y1', **{**options, 'mode': mode}, ignore=True) for mode in ['recsym', 'reciso']]
            #data_recon = compute_reconstruction(data_recon_modes, all_randoms_recon_modes, data, all_randoms)  # only run once for all z-ranges and power / correlation
            data, kwargs['all_shifted'] = data_recon, randoms_recon
        if 'power' in stat_type and not recon:
            kwargs['output_wmatrix'] = fm.get('wmatrix_power_y1', **options, ignore=True)
            kwargs['output_window'] = fm.select('window_power_y1', **options, ignore=True)
        #fi = compute(fi, data, all_randoms, **kwargs)
        if 'power' in stat_type and not recon:
            kwargs = {'windows': kwargs.get('output_window', None), 'output_wmatrix': kwargs.get('output_wmatrix', None)}
            kwargs['output_power_nodirect'] = fm.get(fid, **{**options, 'cut': [None]}, ignore=True)
            kwargs['output_wmatrix_nodirect'] = fm.get('wmatrix_power_y1', **{**options, 'cut': [None]}, ignore=True)
            #return (fi, kwargs['output_power_nodirect'])
            fi = postprocess_power_spectrum(fi, **kwargs)
        if 'correlation' in stat_type:
            fi = postprocess_correlation_function(fi)
        return fi

    for stat_type in todo:
        import os, glob
        if stat_type == 'symlink':
            for fi in fm.select(filetype=['power', 'correlation'], weighting=weighting, cut=[None, cut], version=version).iter(intersection=False):
                if fi.exists():
                    fi.symlink(raise_error=False)
                    for fn in glob.glob(fi.filepath.replace('.npy', '*.txt')):
                        sympath = fi.sympath.replace('.npy', fn.replace(fi.filepath.replace('.npy', ''), ''))
                        try: os.symlink(fn, sympath)
                        except: pass
            continue
        recon = 'recon' in stat_type
        fid = '{}_y1'.format(stat_type)
        stat_type = stat_type.split('_')[0]
        if recon: cuts = [None]
        elif 'power' in stat_type: cuts = [cut]  # no-cut P(k) inferred from cut one
        else: cuts = [None, cut]
        #cuts = [None, cut]
        for fi in fm.select(id=fid, region=regions, tracer=tracers, version=version, weighting=weighting, cut=cuts).iter(intersection=False):
            options = fi.options
            if options['region'] == 'GCcomb':
                fis = [compute_region(fi) for fi in fm.select(id=fid, **{**options, 'region': ['NGC', 'SGC']})]
                inputs_wmatrix = [[fm.get(id='wmatrix_power_y1', **{**options, 'region': region, 'cut': [cut]}, ignore=True, raise_error=False) for cut in [cut, None]] for region in ['NGC', 'SGC']]
                output_wmatrix = [fm.get(id='wmatrix_power_y1', **{**options, 'region': 'GCcomb', 'cut': [cut]}, ignore=True, raise_error=False) for cut in [cut, None]]
                if stat_type == 'power':
                    if recon:
                        combine_regions_power_spectrum(fi, fis)
                    else:
                        combine_regions_power_spectrum((fi, fm.get(fid, **{**options, 'cut': [None]})), fis, inputs_wmatrix=inputs_wmatrix, output_wmatrix=output_wmatrix)
                else:
                    combine_regions_correlation_function(fi, fis)
            else:
                compute_region(fi)
    #spawn(queue, spawn=True)