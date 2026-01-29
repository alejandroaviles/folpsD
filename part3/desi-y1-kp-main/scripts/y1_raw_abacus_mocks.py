from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
import y1_mock_tools

setup_logging()

queue = Queue('y1_raw_abacus_mocks')
queue.clear(kill=False)

output, error = '_sbatch_abacus/slurm-%j.out', '_sbatch_abacus/slurm-%j.err'
environ = Environment('nersc-cosmodesi')
#environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm_catalog = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=16, nodes_per_worker=0.5, output=output, error=error))
#tm_catalog = tm.clone(scheduler=dict(max_workers=20), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=16, nodes_per_worker=0.5, output=output, error=error, constraint='gpu'))
tm_dataz = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=1, nodes_per_worker=0.2, output=output, error=error))


@tm_catalog.python_app
def make_cutsky_data(output, make_cutsky_data=y1_mock_tools.make_cutsky_data, **kwargs):
    from mockfactory import setup_logging
    setup_logging()
    make_cutsky_data(output, **kwargs)
    return output


@tm_catalog.python_app
def make_cutsky_randoms(output, make_cutsky_randoms=y1_mock_tools.make_cutsky_randoms, **kwargs):
    from mockfactory import setup_logging
    setup_logging()
    make_cutsky_randoms(output, **kwargs)
    return output


#@tm_dataz.python_app
def randoms_with_data_z(output, data, randoms, randoms_with_data_z=y1_mock_tools.randoms_with_data_z, **kwargs):
    from mockfactory import setup_logging
    setup_logging()
    randoms_with_data_z(data, randoms, output, **kwargs)
    return output


#@tm_catalog.python_app
def make_mask_healpix(output, randoms, make_mask_healpix=y1_mock_tools.make_mask_healpix):
    from mockfactory import setup_logging
    setup_logging()
    make_mask_healpix(list(randoms), output)
    return output


if __name__ == '__main__':

    from desi_y1_files import get_raw_abacus_file_manager, get_abacus_file_manager

    todo = ['hpmap', 'data', 'randoms', 'dataz'][-1:]

    version = 'v4_1_complete'
    #tracers = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'][1:]
    tracers = ['ELG_LOPnotqso']

    fm = get_raw_abacus_file_manager().select(version=version)
    
    if 'hpmap' in todo:
        fa = get_abacus_file_manager()
        for fi in fm.select(id='mask_healpix', tracer=tracers).iter(intersection=False):
            options = dict(fi.options)
            randoms = list(str(fi) for fi in fa.select(id='catalog_randoms_abacus_y1', fa='complete', imock=0, **options, ignore=True).iter(intersection=False))
            make_mask_healpix(fi, randoms)

    if 'data' in todo:
        for fi in fm.select(id='catalog_data_raw_abacus_y1', tracer=tracers, ignore=True).iter(intersection=False):
            make_cutsky_data(fi, hpmask=fm.get(id='mask_healpix', **fi.options, ignore=True))

    if 'randoms' in todo:
        for fi in fm.select(id='catalog_randoms_raw_abacus_y1', zshuffled=False, tracer=tracers).iter(intersection=False):
            make_cutsky_randoms(fi, hpmask=fm.get(id='mask_healpix', **fi.options, ignore=True))
    
    if 'dataz' in todo:
        regions = ['NGC', 'SGC']
        for fd in fm.select(id='catalog_data_raw_abacus_y1', tracer=tracers, ignore=True).iter(intersection=False, exclude='region'):
            fr = list(zip(*[fm.select(id='catalog_randoms_raw_abacus_y1', zshuffled=False, **{**fd.options, 'region': region}, ignore=True) for region in regions]))
            fs = list(zip(*[fm.select(id='catalog_randoms_raw_abacus_y1', zshuffled=True, **{**fd.options, 'region': region}, ignore=True) for region in regions]))
            for iran, (fr, fs) in enumerate(zip(fr, fs)):
                randoms_with_data_z(fs, list(fd), fr, seed=iran + 42)