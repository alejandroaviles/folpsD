from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
import y1_data_2pt_tools

setup_logging()

output, error = '_sbatch_abacus/slurm-%j.out', '_sbatch_abacus/slurm-%j.err'
gpu_nthreads = 4
cpu_nthreads = 64
gpu = False
nthreads = cpu_nthreads

queue = Queue('y1_abacus_2pt')
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi', command='module swap pyrecon/main pyrecon/mpi')
#environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm_power = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error))
tm_power_wmatrix = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='03:00:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error, stop_after=1))
if gpu:
    nthreads = gpu_nthreads
    tm_corr = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:40:00', mpiprocs_per_worker=1, output=output, error=error, constraint='gpu'))
else:
    nthreads = cpu_nthreads
    tm_corr = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=1, output=output, error=error))
tm_recon = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='00:45:00', mpiprocs_per_worker=64, output=output, error=error))
tm_combine = tm.clone(scheduler=dict(max_workers=6), provider=dict(provider='local'))
tm_merge = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='00:15:00', mpiprocs_per_worker=1, nodes_per_worker=0.1, output=output, error=error))
tm_rotate = tm.clone(scheduler=dict(max_workers=40), provider=dict(provider='nersc', time='00:40:00', mpiprocs_per_worker=1, output=output, error=error, constraint='gpu'))


@tm_power.python_app
def compute_power_spectrum(output, data, all_randoms, compute_power_spectrum=y1_data_2pt_tools.compute_power_spectrum, output_wmatrix=None, nthreads=cpu_nthreads, **kwargs):
    from pypower import setup_logging
    setup_logging()
    compute_power_spectrum(data, all_randoms, output=output, output_wmatrix=output_wmatrix, nthreads=nthreads, **kwargs)
    return output


@tm_power_wmatrix.python_app
def compute_power_spectrum_wmatrix(output, data, all_randoms, compute_power_spectrum=y1_data_2pt_tools.compute_power_spectrum, output_wmatrix=None, nthreads=cpu_nthreads, **kwargs):
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


@tm_combine.python_app
def combine_regions_correlation_function(output, inputs, combine_regions=y1_data_2pt_tools.combine_regions_correlation_function):
    from pycorr import setup_logging
    setup_logging()
    combine_regions(output, inputs)


@tm_merge.python_app
def merge_catalogs(output, inputs, merge_catalogs=y1_data_2pt_tools.merge_catalogs, **kwargs):
    from mockfactory import setup_logging
    setup_logging()
    merge_catalogs(output, inputs, **kwargs)

    
@tm_merge.python_app
def merge_randoms_catalogs(output, inputs, merge_catalogs=y1_data_2pt_tools.merge_randoms_catalogs, **kwargs):
    from mockfactory import setup_logging
    setup_logging()
    merge_catalogs(output, inputs, **kwargs)

    
#@tm_rotate.python_app
def rotate_wmatrix(output, wmatrix, covariance, rotate_wmatrix=y1_data_2pt_tools.rotate_wmatrix, **kwargs):
    from pypower import setup_logging
    setup_logging()
    rotate_wmatrix(wmatrix, covariance, output=output, **kwargs)


def postprocess_rotate_wmatrix(rotation, postprocess_rotate_wmatrix=y1_data_2pt_tools.postprocess_rotate_wmatrix, **kwargs):
    from pypower import setup_logging
    setup_logging()
    postprocess_rotate_wmatrix(rotation, **kwargs)


def postprocess_ricwmatrix(ric, wmatrix, postprocess_ricwmatrix=y1_data_2pt_tools.postprocess_ricwmatrix, **kwargs):
    from pypower import setup_logging
    setup_logging()
    postprocess_ricwmatrix(ric, wmatrix, **kwargs)


if __name__ == '__main__':

    from desi_y1_files import get_abacus_file_manager, get_data_file_manager, get_box_abacus_file_manager, is_baseline_2pt_setup, get_zsnap_from_z

    #todo = ['power', 'correlation', 'power_recon', 'correlation_recon']  # reconstruction is run automatically when asking for power_recon / correlation_recon
    #todo = ['power', 'correlation']
    #todo = ['merge_catalogs']
    #todo = ['merge']
    #todo = ['power']
    #todo = ['correlation', 'correlation_recon'][1:]
    #todo = ['correlation']
    #todo = ['symlink']
    todo = ['rotate_wmatrix']

    #version = 'v3_1_lrg+elg'
    #version = 'v4_1fixran'
    version = 'v4_2'
    #version = 'v1'
    weighting = 'default_FKP'
    #weighting = 'default'
    regions = ['GCcomb']  # NGC, SGC are computed automatically when requiring GCcomb
    #regions = ['NGC']
    #tracers = ['BGS_BRIGHT-21.5', 'LRG', 'LRG+ELG_LOPnotqso', 'ELG_LOPnotqso', 'QSO']
    #tracers = ['BGS_BRIGHT-21.5']
    tracers = ['LRG', 'ELG_LOPnotqso', 'QSO']
    #tracers = ['ELG_LOPnotqso', 'QSO']
    #tracers = ['LRG+ELG_LOPnotqso']
    #tracers = ['ELG_LOPnotqso']
    #tracers = ['LRG']
    #cut = ('rp', 2.5)
    cut = ('theta', 0.05)
    fa = ['complete', 'ffa', 'altmtl']
    #fa = ['complete']
    #fa = ['altmtl']
    #fa = ['ffa']

    fm = get_abacus_file_manager()
    #fm.save('y1_abacus_files.yaml', replace_environ=False)  # exports files as a *.yaml file

    imock = list(range(25))

    def compute_merged_region(fi):
        stat_type = fi.filetype
        options, fid = fi.options, fi.id
        compute = compute_power_spectrum_wmatrix
        data = fm.get(id='catalog_data_merged_abacus_y1', **options, ignore=True)
        all_randoms = fm.select(id='catalog_randoms_merged_abacus_y1', **options, ignore=True)
        kwargs = {}
        kwargs['output_wmatrix'] = fm.get('wmatrix_power_merged_abacus_y1', **options, ignore=True)
        kwargs['output_window'] = fm.select('window_power_merged_abacus_y1', **options, ignore=True)
        fi = compute(fi, data, all_randoms, **kwargs)
        kwargs = {'windows': kwargs.get('output_window', None), 'output_wmatrix': kwargs.get('output_wmatrix', None)}
        kwargs['output_power_nodirect'] = fm.get(fid, **{**options, 'cut': [None]}, ignore=True)
        kwargs['output_wmatrix_nodirect'] = fm.get('wmatrix_power_merged_abacus_y1', **{**options, 'cut': [None]}, ignore=True)
        fi = postprocess_power_spectrum(fi, **kwargs)
        #return (fi, kwargs['output_power_nodirect'])
        return fi

    def compute_region(fi):
        # Compute power spectrum, correlation function, pre/post in a given region
        stat_type = fi.filetype
        recon = 'smoothing_radius' in fi.options
        options, fid = fi.options, fi.id
        compute = {'power': compute_power_spectrum, 'correlation': compute_correlation_function}[stat_type]
        data = fm.get(id='catalog_data_abacus_y1', **options, ignore=True)
        #assert data.exists(), data.filepath
        if not data.exists(): print(data.filepath)
        all_randoms = fm.select(id='catalog_randoms_abacus_y1', **options, ignore=True)
        #if fi.exists(): return fi
        kwargs = {}
        #recon = False
        if recon:
            data_recon = fm.get(id='catalog_data_recon_abacus_y1', **options, ignore=True)
            randoms_recon = fm.select(id='catalog_randoms_recon_abacus_y1', **options, ignore=True)
            data_recon_modes = [fm.get(id='catalog_data_recon_abacus_y1', **{**options, 'mode': mode}, ignore=True) for mode in ['recsym', 'reciso']]
            all_randoms_recon_modes = [fm.select(id='catalog_randoms_recon_abacus_y1', **{**options, 'mode': mode}, ignore=True) for mode in ['recsym', 'reciso']]
            data_recon = compute_reconstruction(data_recon_modes, all_randoms_recon_modes, data, all_randoms)  # only run once for all z-ranges and power / correlation
            data, kwargs['all_shifted'] = data_recon, randoms_recon
        print(data.filepath, fi, options)
        fi = compute(fi, data, all_randoms, **kwargs)
        if 'power' in stat_type:
            fi = postprocess_power_spectrum(fi, output_power_nodirect=fm.get(fid, **{**options, 'cut': [None]}, ignore=True))
        #if 'correlation' in stat_type:
        #    fi = postprocess_correlation_function(fi)
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

            for id in ['correlation_abacus_y1', 'correlation_recon_abacus_y1', 'power_abacus_y1', 'power_recon_abacus_y1', 'wmatrix_power_merged_abacus_y1', 'power_corrected_abacus_y1', 'power_rotated_abacus_y1', 'power_rotated_corrected_abacus_y1', 'wmatrix_power_merged_rotated_abacus_y1'][-4:]:
                for fi in fm.select(id=id, weighting=weighting, cut=[None, cut], version=version, tracer=tracers).iter(intersection=False):
                    if not is_baseline_2pt_setup(observable='correlation' if 'correlation' in id else 'power', **fi.options): continue
                    symlink_special(fi)
                    if fi.exists():
                        fi.symlink(raise_error=False)
                        # print(fi.options)
                        for fn in glob.glob(fi.filepath.replace('.npy', '*.txt')):
                            sympath = fi.sympath.replace('.npy', fn.replace(fi.filepath.replace('.npy', ''), ''))
                            symlink(fn, sympath)

            for id in ['catalog_data_recon_abacus_y1', 'catalog_randoms_recon_abacus_y1']:
                for fi in fm.select(id=id, fa=fa, version=version, tracer=tracers).iter(intersection=False):
                    if not is_baseline_2pt_setup(**fi.options): continue
                    symlink_special(fi)
                    if fi.exists():
                        print(fi)
                        fi.symlink(raise_error=True)

        if stat_type == 'merge_catalogs':
            for fo in fm.select(id='catalog_data_merged_abacus_y1', region=['NGC', 'SGC'], tracer=tracers, version=version, fa=fa).iter(intersection=False):
                fi = fm.select(id='catalog_data_abacus_y1', **fo.options, ignore=True, imock=imock)
                merge_catalogs(fo, fi, factor=100.)

            for fo in fm.select(id='catalog_randoms_merged_abacus_y1', region=['NGC', 'SGC'], tracer=tracers, version=version, fa=fa).iter(intersection=False):
                fi = fm.select(id='catalog_randoms_abacus_y1', **fo.options, ignore=True, imock=imock)
                merge_randoms_catalogs(fo, fi)

        if stat_type == 'merge':
            for fi in fm.select(id='power_merged_abacus_y1', region=regions, tracer=tracers, version=version, fa=fa, weighting=weighting, cut=cut).iter(intersection=False):
                fid = fi.id
                options = fi.options
                if options['region'] == 'GCcomb':
                    fis = [compute_merged_region(fi) for fi in fm.select(id=fid, **{**options, 'region': ['NGC', 'SGC']})]
                    inputs_wmatrix = [[fm.get(id='wmatrix_power_merged_abacus_y1', **{**options, 'region': region, 'cut': [cut]}, ignore=True, raise_error=False) for cut in [cut, None]] for region in ['NGC', 'SGC']]
                    output_wmatrix = [fm.get(id='wmatrix_power_merged_abacus_y1', **{**options, 'region': 'GCcomb', 'cut': [cut]}, ignore=True, raise_error=False) for cut in [cut, None]]
                    combine_regions_power_spectrum((fi, fm.get(fid, **{**options, 'cut': [None]})), fis, inputs_wmatrix=inputs_wmatrix, output_wmatrix=output_wmatrix)
                else:
                    compute_merged_region(fi)

        if stat_type == 'rotate_wmatrix':
            dfm = get_data_file_manager()
            tfm = get_box_abacus_file_manager()
            limock = list(range(25))
            for fi in fm.select(id='rotation_wmatrix_power_merged_abacus_y1', region=regions, tracer=tracers, version=version, cut=[None, cut][1:], fa=fa).iter(intersection=False):
                fid = fi.id
                options = fi.options
                fcovariance = dfm.get(id='covariance_y1', **{**options, 'source': 'thecov', 'observable': 'power', 'cut': None, 'version': 'v1.2'}, ignore=True)
                data = [fm.get(id='power_abacus_y1', **options, imock=iimock, ignore=True) for iimock in limock]
                #data_rotated = [fm.get(id='power_rotated_abacus_y1', **options, imock=iimock, ignore=True) for iimock in limock]
                tracer = options['tracer']
                zsnap = get_zsnap_from_z(tracer=tracer, z=options['zrange'])
                theory = [tfm.get(id='power_box_abacus_y1', version='v0.1' if 'BGS' in tracer else 'v1.1', tracer=tracer, zsnap=zsnap, imock=iimock, los=los, ignore=True) for iimock in imock for los in ['x', 'y', 'z']]
                wmatrix = fm.get(id='wmatrix_power_merged_abacus_y1', **options)
                #print(wmatrix)
                #exit()
                output_wmatrix = fm.get(id='wmatrix_power_merged_rotated_abacus_y1', **options)
                #fi = fm.get(id='wmatrix_power_merged_abacus_y1', tracer='QSO', zrange=(0.8, 2.1), version=version, region='GCcomb', cut=None)
                rotate_wmatrix(fi, wmatrix, fcovariance, theory=theory, data=data, output_wmatrix=output_wmatrix)
                #print(data, data_rotated)
                #postprocess_rotate_wmatrix(fi, power=data, output_power_marg=data_rotated)
    
        recon = 'recon' in stat_type
        fid = '{}_abacus_y1'.format(stat_type)
        stat_type = stat_type.split('_')[0]
        if recon: cuts = [None]
        elif 'power' in stat_type: cuts = [cut]  # no-cut P(k) inferred from cut one
        else: cuts = [None, cut]
        #cuts = [None]
        #cuts = [cut]
        for fi in fm.select(id=fid, region=regions, tracer=tracers, version=version, fa=fa, weighting=weighting, cut=cuts, imock=imock).iter(intersection=False):
            options = fi.options
            if not is_baseline_2pt_setup(**{name: value for name, value in fi.options.items() if name in ['tracer', 'weighting', 'recon_weighting', 'recon_zrange', 'mode', 'smoothing_radius', 'njack']}, observable=stat_type): continue
            #if is_baseline_2pt_setup(**{name: value for name, value in fi.options.items() if name in ['tracer', 'weighting', 'smoothing_radius']}, observable=stat_type): continue
            #if options['njack'] != 60: continue
            if options['region'] == 'GCcomb':
                fis = [compute_region(fi) for fi in fm.select(id=fid, **{**options, 'region': ['NGC', 'SGC']})]
                #if any(fi is None for fi in fis): continue
                if stat_type == 'power':
                    combine_regions_power_spectrum((fi, fm.get(fid, **{**options, 'cut': [None]})), fis)
                else:
                    combine_regions_correlation_function(fi, fis)
            else:
                compute_region(fi)

    #spawn(queue, spawn=True)