from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

setup_logging()

output, error = '_sbatch_ric/slurm-%j.out', '_sbatch_ric/slurm-%j.err'
queue = Queue('y1_ric')
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi', command='module swap pyrecon/main pyrecon/mpi')
#environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm_catalog = tm.clone(scheduler=dict(max_workers=40), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=1, nodes_per_worker=0.1, output=output, error=error))


def catalog_fn(base='randoms', imock=1, tracer='LRG', iran=0, regions=None, ric=False):
    cregions = regions
    if cregions is None:
        cregions = ['NGC', 'SGC']
    ric = 'ric' if ric else 'noric'
    if base == 'merged_data':
        return ['/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/desipipe/{}/ffa/ric/merged/{}_{}_clustering.dat.fits'.format('BGS_v1' if 'BGS' in tracer else 'v1', tracer, region) for region in cregions]
    if base == 'data':
        return ['/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/{}/mock{:d}/{}_ffa_{}_clustering.dat.fits'.format('FFA_BGS' if 'BGS' in tracer else 'FFA', imock, tracer, region) for region in cregions]
    if base == 'randoms':
        return ['/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/{}/mock{:d}/{}_ffa_{}_{:d}_clustering.ran.fits'.format('FFA_BGS' if 'BGS' in tracer else 'FFA', imock, tracer, region, iran) for region in cregions]
    if base == 'randoms_data':
        return '/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/{}_ffa_imaging_HPmapcut{:d}_full.ran.fits'.format({'BGS': 'BGS_BRIGHT'}.get(tracer, tracer), iran)
    if base == 'output_randoms':
        return ['/pscratch/sd/a/adematti/desipipe/EZmock/desipipe/{}/ffa/{}/mock{:d}/{}_ffa_{}_{:d}_clustering.ran.fits'.format('BGS_v1' if 'BGS' in tracer else 'v1', ric, imock, tracer, region, iran) for region in cregions]
    if base == 'map_ntile':
        return ['/pscratch/sd/a/adematti/desipipe/EZmock/desipipe/{}/ffa/noric/map_ntile_{}_ffa_{}_{:d}.ran.fits'.format('BGS_v1' if 'BGS' in tracer else 'v1', tracer, region, iran) for region in cregions]
        

@tm_catalog.python_app
def make_map_ntile(tracer='LRG', nran=18, catalog_fn=catalog_fn):
    # To be run with 1 process
    import logging

    import numpy as np
    from mockfactory import Catalog, setup_logging
    from mpytools import COMM_WORLD, COMM_SELF
    
    setup_logging()

    logger = logging.getLogger('MAPNTILE')
    
    # Too lazy to make it work with MPI
    if COMM_WORLD.rank != 0: exit()
    mpicomm = COMM_SELF

    imock = 1

    def index_in_array(a, b):
        sorter = np.argsort(b)
        return sorter[np.searchsorted(b, a, sorter=sorter)]

    for iran in range(nran):
        logger.info('Processing random {:d}.'.format(iran))
        data_randoms_fn = catalog_fn(base='randoms_data', imock=imock, tracer=tracer, iran=iran)
        randoms_data = Catalog.read(data_randoms_fn, mpicomm=mpicomm)
        randoms_data[randoms_data.columns()]
        for iregion, randoms_fn in enumerate(catalog_fn(base='randoms', imock=imock, tracer=tracer, iran=iran)):
            randoms = Catalog.read(randoms_fn, mpicomm=mpicomm)
            randoms[randoms.columns()]
            if 'NTILE' in randoms:
                randoms_ntile = randoms['NTILE']
            else:
                randoms_ntile = randoms_data['NTILE'][index_in_array(randoms['TARGETID'], randoms_data['TARGETID'])]
                #mask = np.abs(randoms_ntile - randoms['NTILE']) > 0
                #assert not mask.any()
            catalog = Catalog(mpicomm=mpicomm)
            catalog['NTILE'] = randoms_ntile
            names = ['RA', 'DEC']
            catalog[names] = randoms[names]
            catalog.write(catalog_fn(base='map_ntile', imock=imock, tracer=tracer, iran=iran)[iregion])

        
@tm_catalog.python_app
def reshuffle_ezmock_ffa(imock=1, tracer='LRG', ric=False, nran=18, catalog_fn=catalog_fn):
    # To be run with 1 process
    import logging

    import numpy as np
    from desi_y1_files import select_region
    from mockfactory import Catalog, setup_logging
    from mpytools import COMM_WORLD, COMM_SELF

    setup_logging()

    logger = logging.getLogger('FFANORIC')
    logger.info('Processing mock {:d}, tracer {}.'.format(imock, tracer))

    # Too lazy to make it work with MPI
    if COMM_WORLD.rank != 0: exit()
    mpicomm = COMM_SELF

    cregions = ['NGC', 'SGC']

    data_fn = catalog_fn(base='data', imock=imock, tracer=tracer)
    # This is to be a merged data catalog
    if ric:
        merged_data_fn = data_fn
    else:
        merged_data_fn = catalog_fn(base='merged_data', tracer=tracer)

    data = Catalog.read(data_fn, mpicomm=mpicomm)
    merged_data = Catalog.read(merged_data_fn, mpicomm=mpicomm)
    data[data.columns()]
    merged_data[merged_data.columns()]
    
    #data_wcomp = get_mean(data['NTILE'], data['WEIGHT_COMP'])[data['NTILE']]
    data_wtotp = data['WEIGHT_COMP'] * data['WEIGHT_SYS'] * data['WEIGHT_ZFAIL']
    data_wcomp = data_wtotp / data['WEIGHT']
    #print(np.std(get_mean(data['NTILE'], data['WEIGHT_COMP'])[data['NTILE']] / data_wcomp))

    merged_data_wtotp = merged_data['WEIGHT_COMP'] * merged_data['WEIGHT_SYS'] * merged_data['WEIGHT_ZFAIL']
    merged_data_wcomp = merged_data_wtotp / merged_data['WEIGHT']
    #data_ftile = get_mean(randoms['TILELOCID'], randoms['FRAC_TLOBS_TILES'])[data['TILELOCID']]  # all 1 for EZmock ffa
    #data_ftile = get_mean(data['NTILE'], data_ftile)[data['NTILE']]
    merged_data_ftile = data_ftile = 1.  # ok for FFA

    merged_data_ftile_wcomp = merged_data_ftile / merged_data_wcomp
    merged_data_nz = merged_data['NX'] / merged_data_ftile_wcomp

    P0 = np.rint(np.mean((1. / merged_data['WEIGHT_FKP'] - 1.) / merged_data['NX']))

    def get_mean(idx, weights):
        wsum = np.bincount(idx, weights=weights)
        ssum = np.bincount(idx)
        ssum[ssum == 0.] = 1.
        return wsum / ssum

    for iran in range(nran):
        logger.info('Processing random {:d}.'.format(iran))
        map_ntile_fn = catalog_fn(base='map_ntile', tracer=tracer, iran=iran)
        randoms_fn = catalog_fn(base='randoms', imock=imock, tracer=tracer, iran=iran)
        output_randoms_fn = catalog_fn(base='output_randoms', imock=imock, tracer=tracer, iran=iran, ric=ric)
        map_ntile = Catalog.read(map_ntile_fn, mpicomm=mpicomm)
        map_ntile[map_ntile.columns()]
        randoms = Catalog.read(randoms_fn, mpicomm=mpicomm)
        randoms[randoms.columns()]

        try: del randoms['TARGETID_DATA']
        except KeyError: pass

        for name in ['RA', 'DEC']: assert np.allclose(randoms[name], map_ntile[name])
        if 'NTILE' in randoms: assert np.allclose(randoms['NTILE'], map_ntile['NTILE'])
        randoms_ntile = map_ntile['NTILE']
        #print(randoms)
        randoms_wcomp = get_mean(data['NTILE'], data_wcomp)[randoms_ntile]

        if 0:
            print(P0)
            from matplotlib import pyplot as plt
            ax = plt.gca()
            ax.scatter(data['Z'], data['NX'], marker='.', s=1., color='b')
            ax.scatter(data['Z'], merged_data_nz, marker='.', s=1., color='r')
            plt.savefig('tmp.png')
            plt.close(plt.gcf())

        rng = np.random.RandomState(seed=100 * imock + iran)

        regions = ['N', 'S']
        if tracer == 'QSO':
            regions = ['N', 'SnoDES', 'DES']

        randoms['NZ'] = randoms.zeros()

        randoms_ftile = 1.  # ok for FFA
        sum_data_weights, sum_randoms_weights = [], []

        for region in regions:
            mask_data = select_region(data['RA'], data['DEC'], region=region)
            mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
            mask_merged_data = select_region(merged_data['RA'], merged_data['DEC'], region=region)

            # Shuffle z
            index = rng.choice(mask_merged_data.sum(), size=mask_randoms.sum())
            randoms['Z'][mask_randoms] = merged_data['Z'][mask_merged_data][index]
            randoms['WEIGHT'][mask_randoms] = (merged_data_wtotp * randoms_ftile)[mask_merged_data][index]
            randoms['NZ'][mask_randoms] = merged_data_nz[mask_merged_data][index]

            sum_data_weights.append(data_wtotp[mask_data].sum())
            sum_randoms_weights.append(randoms['WEIGHT'][mask_randoms].sum())

        # Renormalize randoms / data here
        sum_data_weights, sum_randoms_weights = np.array(sum_data_weights), np.array(sum_randoms_weights)
        alphas = sum_data_weights / sum_randoms_weights / (sum(sum_data_weights) / sum(sum_randoms_weights))
        logger.info('alpha before renormalization: {}'.format(alphas))

        for region, alpha in zip(regions, alphas):
            mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
            randoms['WEIGHT'][mask_randoms] *= alpha

        randoms['WEIGHT'] /= randoms_wcomp
        randoms['NX'] = randoms['NZ'] * (randoms_ftile / randoms_wcomp)
        randoms['WEIGHT_FKP'] = 1. / (1. + randoms['NX'] * P0)
        del randoms['NZ']

        alphas = []
        sum_data_weights, sum_randoms_weights = [], []
        for region in regions:
            mask_data = select_region(data['RA'], data['DEC'], region=region)
            mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
            sum_data_weights.append(data['WEIGHT'][mask_data].sum())
            sum_randoms_weights.append(randoms['WEIGHT'][mask_randoms].sum())

        sum_data_weights, sum_randoms_weights = np.array(sum_data_weights), np.array(sum_randoms_weights)
        alphas = sum_data_weights / sum_randoms_weights / (sum(sum_data_weights) / sum(sum_randoms_weights))
        logger.info('alpha after renormalization & reweighting: {}'.format(alphas))

        if 0:
            from matplotlib import pyplot as plt
            ax = plt.gca()
            ax.scatter(randoms['Z'], 1. / randoms['WEIGHT_FKP'] - 1., marker='.', s=1., color='b')
            plt.savefig('tmp2.png')
            plt.close(plt.gcf())

        for iregion, region in enumerate(cregions):
            randoms[select_region(randoms['RA'], randoms['DEC'], region=region)].write(output_randoms_fn[iregion])


def compute_ricwmatrix(wmatrix, power_ric, power_noric, covariance=None, output=None):
    
    import os
    import numpy as np
    from pypower import PowerSpectrumStatistics
    from desi_y1_files import WindowRIC, is_file_sequence, is_path

    wmatrix = wmatrix.load()
    kinlim = (1.7e-3, 0.3)
    koutlim = (3e-3, 0.1)
    kinrebin = 1
    koutrebin = 1
    wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)

    def get_power(data, k=None):
        if isinstance(data, np.ndarray):
            return np.ravel(data)
        if k is None:
            data = data[:(data.shape[0] // koutrebin) * koutrebin:koutrebin].select(koutlim)(ell=ells, return_k=False, complex=False, remove_shotnoise=False)
        else:
            data = data(ell=ells, k=k, return_k=False)
        return data.ravel()
    
    ells = tuple(proj.ell for proj in wmatrix.projsout)
    power_ric = [get_power(data.load()) for data in power_ric]
    power_noric = [get_power(data.load()) for data in power_noric]
    covariance_scaled = None

    if covariance is not None:
        covariance = covariance.load(koutlim)
        covariance_scaled = covariance / len(power_ric)
    ric = WindowRIC(wmatrix, power_ric=power_ric, power_noric=power_noric, covmatrix=covariance, attrs={'kinlim': kinlim, 'kinrebin': kinrebin, 'koutlim': koutlim, 'koutrebin': koutrebin})

    if output is not None:
        base_fn = os.path.splitext(output)[0]
    plot = output is not None
    plot = True
    base_fn = 'test_ric'
    if plot:
        #ric.plot_wmatrix(fn='{}_wmatrix.png'.format(base_fn), klim=(0., 0.2), ric=False)
        #ric.plot_wmatrix(fn='{}_rickernel.png'.format(base_fn), klim=(0., 0.2), ric='kernel')
        nobs = len(power_ric)
        ric.plot_validation(covmatrix=covariance_scaled, klim=(0., 0.1), fn='{}_validation_noric.png'.format(base_fn))
        ric.plot_validation_gic(fn='{}_gic_noric.png'.format(base_fn))
        ric.plot_power(fn='{}_power_ric_noric.png'.format(base_fn))
    ric.fit()
    print(ric.asmatrix)
    if plot:
        ric.plot_wmatrix(fn='{}_ricwmatrix.png'.format(base_fn), klim=(0., 0.1))
        ric.plot_wmatrix(fn='{}_rickernel.png'.format(base_fn), klim=(0., 0.2), ric='kernel')
        ric.plot_validation(covmatrix=covariance_scaled, klim=(0., 0.1), fn='{}_validation.png'.format(base_fn))
        ric.plot_validation_gic(fn='{}_gic.png'.format(base_fn))
    if output is not None:
        ric.save(output)
    return ric

            
if __name__ == '__main__':
    

    todo = ['ntile', 'catalog', 'ricmatrix'][:2]

    if 'ntile' in todo:
        for tracer in ['BGS', 'LRG', 'ELG_LOP', 'QSO']:
            make_map_ntile(tracer=tracer)

    if 'catalog' in todo:
        for tracer in ['BGS', 'LRG', 'ELG_LOP', 'QSO']:
            for imock in range(100, 101):
                for ric in [False, True]:
                    reshuffle_ezmock_ffa(imock=imock, tracer=tracer, ric=ric)

    if 'ricmatrix' in todo:
        from desi_y1_files import get_ez_file_manager, get_data_file_manager
        fm = get_ez_file_manager()
        dfm = get_data_file_manager()

        for options in fm.select(id='power_ez_y1', version='v1ric', tracer='QSO', region='GCcomb').iter_options(exclude=['imock'], intersection=False):
            options.pop('version')
            options.pop('nran')
            power_ric = fm.select(id='power_ez_y1', version='v1ric', **options)
            power_noric = fm.select(id='power_ez_y1', version='v1noric', **options)
            covariance = dfm.get(id='covariance_power_y1', **{**options, 'cut': None, 'version': 'v1.2'}, ignore=True)
            print(options)
            wmatrix = fm.get(id='wmatrix_power_merged_ez_y1', version='v1', **options, ignore=True)
            print(len([dd for dd in power_ric]))
            compute_ricwmatrix(wmatrix, power_ric, power_noric, covariance=None)
            