from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
import y1_data_2pt_tools

setup_logging()

queue = Queue('y1_data_2pt')
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi', command='module swap pyrecon/main pyrecon/mpi')
#environ = Environment(environ='base', data=dict(DESICFS='/global/cfs/cdirs/desi'), command=['source /global/common/software/desi/users/adematti/cosmodesi_environment.sh new', 'module swap pyrecon/main pyrecon/mpi'])
#environ = Environment('nersc-cosmodesi')
output, error = '_sbatch/slurm-%j.out', '_sbatch/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ)
tm_power = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='03:00:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error))
gpu_nthreads = 4
cpu_nthreads = 64
gpu = False #True
if gpu:
    nthreads = gpu_nthreads
    tm_corr = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=1, output=output, error=error, constraint='gpu'))
else:
    nthreads = cpu_nthreads
    tm_corr = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=1, output=output, error=error))
tm_recon = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=64, output=output, error=error))
tm_combine = tm.clone(scheduler=dict(max_workers=6), provider=dict(provider='local'))
tm_wmatrix = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:20:00', mpiprocs_per_worker=1, nodes_per_worker=0.1, output=output, error=error))
tm_rotate = tm.clone(scheduler=dict(max_workers=40), provider=dict(provider='nersc', time='00:40:00', mpiprocs_per_worker=1, output=output, error=error, constraint='gpu'))


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


@tm_combine.python_app
def combine_regions_correlation_function(output, inputs, combine_regions=y1_data_2pt_tools.combine_regions_correlation_function):
    from pycorr import setup_logging
    setup_logging()
    combine_regions(output, inputs)


#@tm_rotate.python_app
def rotate_wmatrix(output, wmatrix, covariance, rotate_wmatrix=y1_data_2pt_tools.rotate_wmatrix, **kwargs):
    from pypower import setup_logging
    setup_logging()
    rotate_wmatrix(wmatrix, covariance, output=output, **kwargs)


def postprocess_rotate_wmatrix(rotation, postprocess_rotate_wmatrix=y1_data_2pt_tools.postprocess_rotate_wmatrix, **kwargs):
    from pypower import setup_logging
    setup_logging()
    postprocess_rotate_wmatrix(rotation, **kwargs)

    
if __name__ == '__main__':

    from desi_y1_files import get_data_file_manager, is_baseline_2pt_setup, get_zsnap_from_z

    todo = []
    #todo = ['power', 'correlation', 'power_recon', 'correlation_recon']  # reconstruction is run automatically when asking for power_recon / correlation_recon
    #todo = ['power']
    #todo = ['power', 'correlation']
    #todo = ['power_recon', 'correlation_recon']
    #todo = ['correlation', 'correlation_recon'][:1]
    todo = ['symlink']
    #todo = ['rotate_wmatrix']

    #version = 'v1.5'
    version = 'v1.2'
    #version = 'v1'
    #weighting = ['default_SYSSN_FKP', 'default_SYSRF_FKP', 'default_SYSIMLIN_FKP']  # for ELG, QSO
    #weighting = ['default_SYSRF_FKP']  # for ELG
    #weighting = ['default_SYS1_FKP']  # for BGS, LRG, ELG, QSO
    weighting = ['default_FKP']
    regions = ['GCcomb']  # NGC, SGC are computed automatically when requiring GCcomb
    #regions = ['N', 'S']
    tracers = ['BGS_BRIGHT-21.5', 'LRG', 'LRG+ELG_LOPnotqso', 'ELG_LOPnotqso', 'QSO']
    #tracers = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']
    #tracers = ['LRG+ELG_LOPnotqso']
    cut = ('theta', 0.05)

    fm = get_data_file_manager(conf='unblinded')
    #fm.save('y1_data_files.yaml', replace_environ=False)  # exports files as a *.yaml file

    def compute_region(fi):
        # Compute power spectrum, correlation function, pre/post in a given region
        stat_type = fi.filetype
        options, fid = dict(fi.options), fi.id
        recon = 'smoothing_radius' in options
        compute = {'power': compute_power_spectrum, 'correlation': compute_correlation_function}[stat_type]
        kwargs = {}
        region = options.get('region')
        caps = ['NGC', 'SGC']
        print(options)
        if region in caps:
            data = fm.get(id='catalog_data_y1', **options, ignore=True)
            all_randoms = fm.select(id='catalog_randoms_y1', **options, ignore=True)
        else:
            regions = caps
            data = [fm.get(id='catalog_data_y1', **{**options, 'region': region}, ignore=True) for region in regions]
            all_randoms = list(zip(*[list(fm.select(id='catalog_randoms_y1', **{**options, 'region': region}, region=region, ignore=True)) for region in regions]))

        if recon:
            data = fm.get(id='catalog_data_y1', **options, ignore=True)
            #if fi.exists(): return fi
            assert data.exists()
            all_randoms = fm.select(id='catalog_randoms_y1', **options, ignore=True)
            data_recon = fm.get(id='catalog_data_recon_y1', **options, ignore=True)
            randoms_recon = fm.select(id='catalog_randoms_recon_y1', **options, ignore=True)
            data_recon_modes = [fm.get(id='catalog_data_recon_y1', **{**options, 'mode': mode}, ignore=True) for mode in ['recsym', 'reciso']]
            all_randoms_recon_modes = [fm.select(id='catalog_randoms_recon_y1', **{**options, 'mode': mode}, ignore=True) for mode in ['recsym', 'reciso']]
            #run_recon = not data_recon.exists() or not any(ff.exists() for ff in randoms_recon)
            run_recon = False
            if run_recon:
                data_recon = compute_reconstruction(data_recon_modes, all_randoms_recon_modes, data, all_randoms)  # only run once for all z-ranges and power / correlation
            data, kwargs['all_shifted'] = data_recon, randoms_recon
        if 'power' in stat_type and not recon:
            kwargs['output_wmatrix'] = fm.get('wmatrix_power_y1', **options, ignore=True)
            kwargs['output_window'] = fm.select('window_power_y1', **options, ignore=True)
        fi = compute(fi, data, all_randoms, **kwargs)
        if 'power' in stat_type and not recon:
            options['region'] = region
            kwargs = {'windows': kwargs.get('output_window', None), 'output_wmatrix': kwargs.get('output_wmatrix', None)}
            kwargs['output_power_nodirect'] = fm.get(fid, **{**options, 'cut': [None]}, ignore=True)
            kwargs['output_wmatrix_nodirect'] = fm.get('wmatrix_power_y1', **{**options, 'cut': [None]}, ignore=True)
            #return (fi, kwargs['output_power_nodirect'])
            fi = postprocess_power_spectrum(fi, **kwargs)
        #else:
        #    fi = postprocess_correlation_funcion(fi, **kwargs)
        return fi

    for stat_type in todo:
        if stat_type == 'symlink':
            import os, glob
            
            def symlink(filepath, sympath):
                try:
                    if os.path.islink(sympath): os.remove(sympath)
                    os.symlink(filepath, sympath)
                except:
                    pass

            def symlink_special(fi):
                if not is_baseline_2pt_setup(**fi.options): return
                if 'recon_zrange' in fi.options:
                    sympath = fi.clone(foptions={**fi.foptions, 'zrange': fi.foptions['recon_zrange'], 'recon_zrange': ''}).sympath
                    if fi.exists():
                        symlink(fi.filepath, sympath)
            
            for id in ['correlation_y1', 'correlation_recon_y1', 'power_y1', 'power_recon_y1', 'wmatrix_power_y1', 'power_rotated_y1', 'power_corrected_y1', 'power_rotated_corrected_y1', 'wmatrix_power_rotated_y1']:
                for fi in fm.select(id=id, weighting=weighting, cut=[None, cut], tracer=tracers, version=version).iter(intersection=False):
                    options = dict(fi.options)
                    options.pop('mode', None)
                    if not is_baseline_2pt_setup(**options): continue
                    symlink_special(fi)
                    print(fi)
                    if fi.exists():
                        fi.symlink(raise_error=False)
                        # print(fi.options)
                        for fn in glob.glob(fi.filepath.replace('.npy', '*.txt')):
                            sympath = fi.sympath.replace('.npy', fn.replace(fi.filepath.replace('.npy', ''), ''))
                            symlink(fn, sympath)

            for id in ['catalog_data_recon_y1', 'catalog_randoms_recon_y1']:
                for fi in fm.select(id=id, tracer=tracers, version=version).iter(intersection=False):
                    if not is_baseline_2pt_setup(**fi.options): continue
                    symlink_special(fi)
                    if fi.exists():
                        fi.symlink(raise_error=False)
                        # print(fi.options)

        if stat_type == 'rotate_wmatrix':
            from desi_y1_files import get_abacus_file_manager, get_box_abacus_file_manager
            dfm = fm
            afm = get_abacus_file_manager()
            tfm = get_box_abacus_file_manager()
            imock = list(range(25))
            for fi in fm.select(id='rotation_wmatrix_power_y1', region=regions, tracer=tracers, version=version, cut=[None, cut][1:], weighting='default_FKP').iter(intersection=False):
                fid = fi.id
                options = fi.options
                tracer = options['tracer']
                cov_options = {**options, 'source': 'thecov', 'observable': 'power', 'cut': None, 'version': 'v1.2'}
                fcovariance = dfm.get(id='covariance_y1', **cov_options, ignore=True)
                mdata = [afm.get(id='power_abacus_y1', **{**options, 'version': 'v1' if 'BGS' in tracer else 'v4_2'}, fa='complete', imock=iimock, ignore=True) for iimock in imock]
                zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                theory = [tfm.get(id='power_box_abacus_y1', version='v0.1' if 'BGS' in tracer else 'v1.1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, ignore=True) for iimock in imock for los in ['x', 'y', 'z']]
                wmatrix = fm.get(id='wmatrix_power_y1', **options)
                output_wmatrix = fm.get(id='wmatrix_power_rotated_y1', **options)
                #tmp = output_wmatrix.load()
                #print(tmp.vectorout)
                rotate_wmatrix(fi, wmatrix, fcovariance, theory=theory, data=mdata, output_wmatrix=output_wmatrix)
                # Rotation is now performed in y1_covariance.py
                ## data = [fm.get(id='power_y1', **options, ignore=True)]
                ## data_rotated = [fm.get(id='power_rotated_y1', **options, ignore=True)]
                ## postprocess_rotate_wmatrix(fi, power=data, output_power=data_rotated)
                        
        recon = 'recon' in stat_type
        fid = '{}_y1'.format(stat_type)
        stat_type = stat_type.split('_')[0]
        if recon: cuts = [None]
        elif 'power' in stat_type: cuts = [cut]  # no-cut P(k) inferred from cut one
        else: cuts = [None, cut]
        for fi in fm.select(id=fid, region=regions, tracer=tracers, version=version, weighting=weighting, cut=cuts).iter(intersection=False):
            if not is_baseline_2pt_setup(**{name: value for name, value in fi.options.items() if name in ['tracer', 'recon_weighting', 'recon_zrange', 'smoothing_radius', 'mode']}, observable=stat_type): continue
            options = fi.options
            #if options['mode'] != 'reciso': continue
            #if options['njack'] != 60: continue
            #if options['recon_zrange'] is None: continue
            #if options['weighting'] == 'default_FKP': continue
            #if options['recon_weighting'] == 'default': continue
            if options['region'] == 'GCcomb':
                fis = [compute_region(fi) for fi in fm.select(id=fid, **{**options, 'region': ['NGC', 'SGC']})]
                inputs_wmatrix = [[fm.get(id='wmatrix_power_y1', **{**options, 'region': region, 'cut': [cut]}, ignore=True, raise_error=False) for cut in [cut, None]] for region in ['NGC', 'SGC']]
                output_wmatrix = [fm.get(id='wmatrix_power_y1', **{**options, 'region': 'GCcomb', 'cut': [cut]}, ignore=True, raise_error=False) for cut in [cut, None]]
                if stat_type == 'power':
                    if any(fi is None for fi in fis): continue
                    if recon:
                        combine_regions_power_spectrum(fi, fis)
                    else:
                        combine_regions_power_spectrum((fi, fm.get(fid, **{**options, 'cut': [None]})), fis, inputs_wmatrix=inputs_wmatrix, output_wmatrix=output_wmatrix)
                else:
                    combine_regions_correlation_function(fi, fis)
            else:
                compute_region(fi)

    #spawn(queue, spawn=True)