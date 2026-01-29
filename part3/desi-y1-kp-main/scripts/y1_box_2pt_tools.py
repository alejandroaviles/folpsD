def get_clustering_positions_weights(data, z=None, los='x', cosmo=None, option=None, dtype='f8'):
    import numpy as np
    from mockfactory import utils
    data.get(data.columns())
    option = option or ''
    if 'rmag21.5' in option:
        mask = data['R_MAG_ABS'] + 0.05 < -21.5
        data = data[mask]
    positions = np.column_stack([data[name] for name in ['x', 'y', 'z']])
    if los and 'vx' in data:
        from cosmoprimo.fiducial import DESI
        if cosmo is None: cosmo = DESI()
        a = 1. / (1. + z)
        E = cosmo.efunc(z)
        velocities = np.column_stack([data[name] for name in ['vx', 'vy', 'vz']]) / (100. * a * E)
        vlos = [1. * (los == axis) for axis in 'xyz']
        positions += utils.vector_projection(velocities, vlos)
    return list(positions.T.astype(dtype)), [np.ones(positions.shape[0], dtype=dtype)]


# Run reconstruction and write catalogs to disk
def compute_reconstruction(data, output_data, all_output_randoms, get_clustering_positions_weights=get_clustering_positions_weights, mpicomm=None):
    import logging
    import numpy as np
    import mpytools as mpy
    from mockfactory import Catalog, RandomBoxCatalog
    from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction
    from cosmoprimo.fiducial import DESI
    from desipipe.file_manager import FileEntryCollection
    from desi_y1_files import load, is_sequence

    logger = logging.getLogger('Reconstruction')
    if mpicomm is None: mpicomm = mpy.COMM_WORLD

    def get_bias(tracer='ELG'):
        if tracer.startswith('BGS'):
            bias = 1.5
        elif tracer.startswith('LRG'):
            bias = 2.0
        elif tracer.startswith('ELG'):
            bias = 1.2
        elif tracer.startswith('QSO'):
            bias = 2.1
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
        return bias

    cosmo = DESI()
    if not is_sequence(data): data = [data]
    data = list(data)
    if not is_sequence(output_data): output_data = [output_data]
    output_data = list(output_data)
    all_output_randoms = list(all_output_randoms)
    if not is_sequence(all_output_randoms[0]): all_output_randoms = [all_output_randoms]
    all_output_randoms = [list(output_randoms) for output_randoms in all_output_randoms]  # recsym, reciso

    default_options = {'los': 'x', 'cellsize': 4., 'boxsize': 2000., 'boxcenter': 1000.}
    options = {**default_options, **output_data[0].options}

    algorithm = options['algorithm']
    smoothing_radius = options['smoothing_radius']  # recsym or reciso
    los = options['los']
    cellsize = options['cellsize']
    boxsize = options['boxsize']
    boxcenter = options['boxcenter']
    dtype = 'f8'

    z = options['zsnap']
    data = Catalog.read(data, mpicomm=mpicomm, filetype='fits')
    f = cosmo.growth_rate(z)
    bias = get_bias(tracer=options['tracer'])
    catalog_attrs = {'z': z, 'los': los, 'cosmo': cosmo, 'option': options.get('catalog', None)}
    data_positions, data_weights = get_clustering_positions_weights(data, **catalog_attrs)
    data_weights = data_weights[-1]
    if mpicomm.rank == 0:
        logger.info('Using f = {:.3f} and bias = {:.3f} at z = {:.2f} with smoothing = {:.3f} and los = {}.'.format(f, bias, z, smoothing_radius, los))

    Reconstruction = {'IFFT': IterativeFFTReconstruction, 'IFFTP': IterativeFFTParticleReconstruction, 'MG': MultiGridReconstruction}[algorithm]
    recon = Reconstruction(f=f, bias=bias, los=los, cellsize=cellsize, boxsize=boxsize, boxcenter=boxcenter, position_type='xyz', dtype=dtype, mpicomm=mpicomm, mpiroot=None, wrap=True)
    recon.assign_data(data_positions, data_weights)
    recon.set_density_contrast(smoothing_radius=smoothing_radius)
    recon.run()
    # If using IterativeFFTParticleReconstruction, displacements are to be taken at the reconstructed data real-space positions;
    if mpicomm.rank == 0:
        logger.info('Shifting data.')
    if Reconstruction is IterativeFFTParticleReconstruction:
        data_positions_recon = recon.read_shifted_positions('data', dtype=dtype)
    else:
        data_positions_recon = recon.read_shifted_positions(data_positions, dtype=dtype)
    data['x'], data['y'], data['z'] = data_positions_recon
    data = data[['x', 'y', 'z']]
    for output_data in output_data:  # recysm, reciso
        output_data.save(data)

    nrandoms = min(len(r) for r in all_output_randoms)
    nrandoms_splits = min(6, mpicomm.size)  # maximum 6 output_randoms written in parallel
    nsplits = (nrandoms + nrandoms_splits - 1) // nrandoms_splits
    for isplit in range(nsplits):
        sl = slice(isplit * nrandoms // nsplits, (isplit + 1) * nrandoms // nsplits)
        if mpicomm.rank == 0:
            logger.info('Shifting randoms {}.'.format(sl))
        tosave = []
        for irank, output_randoms in enumerate(list(zip(*all_output_randoms))[sl]):
            randoms = RandomBoxCatalog(boxsize=boxsize, boxcenter=boxcenter, csize=data.csize, mpicomm=mpicomm)
            randoms_positions = randoms.get('Position').T
            del randoms['Position']
            for output_randoms in output_randoms:
                mode = output_randoms.options['mode']
                if mpicomm.rank == 0:
                    logger.info('Using mode = {}.'.format(mode))
                if mode == 'recsym':
                    # RecSym = remove large scale RSD from randoms
                    randoms_positions_recon = recon.read_shifted_positions(randoms_positions, dtype=dtype)
                else:
                    # or RecIso
                    randoms_positions_recon = recon.read_shifted_positions(randoms_positions, field='disp', dtype=dtype)
                randoms['x'], randoms['y'], randoms['z'] = randoms_positions_recon
                tosave.append((output_randoms, randoms.gather(mpiroot=irank)))
        for output_randoms, randoms in tosave:
            if randoms is not None:
                output_randoms.save(randoms)


def postprocess_correlation_function(correlation, rebinning_factors=(1, 4)):
    from matplotlib import pyplot as plt
    from pycorr import TwoPointCorrelationFunction

    fn = str(correlation)
    correlation = TwoPointCorrelationFunction.load(fn)
    for factor in rebinning_factors:
        rebinned = correlation[:(correlation.shape[0] // factor) * factor:factor]
        db = rebinned.edges[0][1] - rebinned.edges[0][0]
        ff = fn.replace('.npy', '_d{:.0f}'.format(db))
        rebinned.save_txt(ff + '.txt')
        rebinned.save_txt(ff + '_poles.txt', ell=(0, 2, 4))
        rebinned.plot(ells=(0, 2, 4), fn=ff + '_poles.png')
        plt.close()


def compute_correlation_function(data, output=None, all_shifted=None, data2=None, all_shifted2=None, nthreads=64, gpu=False, mpicomm=None,
                                 postprocess_correlation_function=postprocess_correlation_function, get_clustering_positions_weights=get_clustering_positions_weights):

    import logging
    import numpy as np
    from mpytools import Catalog
    from pycorr import TwoPointCorrelationFunction, BoxSubsampler, mpi
    from cosmoprimo.fiducial import DESI
    from desi_y1_files.catalog_tools import concatenate_data_randoms
    from desi_y1_files import load

    logger = logging.getLogger('CorrelationFunction')
    if mpicomm is None: mpicomm = mpi.COMM_WORLD

    cosmo = DESI()
    autocorr = data2 is None
    with_shifted = all_shifted is not None
    corr_type = 'smu'
    default_options = {'binning': 'lin', 'weighting': 'default', 'tracer': '', 'nran': 4, 'los': 'x', 'boxsize': 2000., 'boxcenter': 1000., 'split': None}
    options = {**default_options, **output.options}
    bin_type = options['binning']
    weight_type = options['weighting']
    tracer = options['tracer']
    nran = options['nran']
    los = options['los']
    split_randoms_above = options['split']
    if split_randoms_above is None: split_randoms_above = np.inf
    dtype = 'f8'
    kwargs = {'boxsize': options['boxsize'], 'los': los if los is not None else 'x', 'dtype': dtype}
    weight_attrs = {}
    z = options['zsnap']

    if mpicomm.rank == 0:
        logger.info('Using options = {}.'.format(options))

    def get_edges(corr_type='smu', bin_type='lin'):

        if bin_type == 'log':
            sedges = np.geomspace(0.01, 100., 49)
        elif bin_type == 'lin':
            sedges = np.linspace(0., 200, 201)
        else:
            raise ValueError('bin_type must be one of ["log", "lin"]')
        if corr_type == 'smu':
            edges = (sedges, np.linspace(-1., 1., 201))  # s is input edges and mu evenly spaced between -1 and 1
        elif corr_type == 'rppi':
            if bin_type == 'lin':
                edges = (sedges, np.linspace(-40., 40, 101))  # transverse and radial separations are coded to be the same here
            else:
                edges = (sedges, np.linspace(0., 40., 41))
        elif corr_type == 'theta':
            edges = (np.linspace(0., 4., 101),)
        else:
            raise ValueError('corr_type must be one of ["smu", "rppi", "theta"]')
        return edges

    edges = get_edges(corr_type, bin_type=bin_type)

    data_positions1 = data_weights1 = data_positions2 = data_weights2 = None
    shifted_positions1 = shifted_weights1 = shifted_positions2 = shifted_weights2 = None

    randoms_splits_size = None  # simply split according to input randoms
    catalog_attrs = {'z': z, 'los': los, 'cosmo': cosmo, 'option': options.get('catalog', None)}
    data_positions1, data_weights1 = get_clustering_positions_weights(Catalog.read(data, mpicomm=mpicomm, filetype='fits'), **catalog_attrs)
    if with_shifted:
        shifted = [[get_clustering_positions_weights(load(shifted, mpicomm=mpicomm), **catalog_attrs)] for shifted in all_shifted[:nran]]
        (shifted_positions1, shifted_weights1) = concatenate_data_randoms([(data_positions1, data_weights1)], *shifted, randoms_splits_size=randoms_splits_size, mpicomm=mpicomm)[1]

    if not autocorr:
        data_positions2, data_weights2 = get_clustering_positions_weights(Catalog.read(data, mpicomm=mpicomm, filetype='fits'), **catalog_attrs)
        if with_shifted:
            shifted2 = [[get_clustering_positions_weights(load(shifted, mpicomm=mpicomm), **catalog_attrs)] for shifted in all_shifted2[:nran]]
            (shifted_positions2, shifted_weights2) = concatenate_data_randoms([(data_positions2, data_weights2)], *shifted2, weight_attrs=weight_attrs, randoms_splits_size=randoms_splits_size, mpicomm=mpicomm)[1]

    randoms_kwargs = dict(shifted_positions1=shifted_positions1, shifted_weights1=shifted_weights1,
                          shifted_positions2=shifted_positions2, shifted_weights2=shifted_weights2)

    zedges = np.array(list(zip(edges[0][:-1], edges[0][1:])))
    mask = zedges[:, 0] >= split_randoms_above
    zedges = [zedges[~mask], zedges[mask]]
    split_edges, split_randoms = [], []
    for ii, zedge in enumerate(zedges):
        if zedge.size:
            split_edges.append([np.append(zedge[:, 0], zedge[-1, -1])] + list(edges[1:]))
            split_randoms.append(ii > 0)

    results = []
    nsplits = min([1] + [len(pw) if pw is not None else np.inf for pw in randoms_kwargs.values()])
    if mpicomm.rank == 0:
        logger.info('Using {:d} randoms splits for s > {:.0f}.'.format(nsplits, split_randoms_above))
    for i_split_randoms, edges in zip(split_randoms, split_edges):
        result = 0
        D1D2 = None
        for isplit in range(nsplits if i_split_randoms else 1):
            tmp_randoms_kwargs = dict(randoms_kwargs)
            if i_split_randoms:
                # On scales above split_randoms_above, sum correlation function over multiple randoms
                for name, arrays in randoms_kwargs.items():
                    if arrays is None: continue
                    tmp_randoms_kwargs[name] = arrays[isplit]
                if mpicomm.rank == 0:
                    logger.info('Running split {:d} / {:d} for edges = {:.1f} - {:.1f}.'.format(isplit + 1, nsplits, edges[0][0], edges[0][-1]))
            else:
                for name, arrays in randoms_kwargs.items():
                    if arrays is None: continue
                    tmp_randoms_kwargs[name] = [np.concatenate([array[iarr] for array in arrays]) for iarr in range(len(arrays[0]))]
            tmp = TwoPointCorrelationFunction(corr_type, edges, data_positions1=data_positions1, data_weights1=data_weights1,
                                              data_positions2=data_positions2, data_weights2=data_weights2,
                                              engine='corrfunc', position_type='xyz', nthreads=nthreads, gpu=gpu,
                                              mesh_refine_factors=(4, 4, 2), **tmp_randoms_kwargs, **kwargs,
                                              D1D2=D1D2, mpicomm=mpicomm, mpiroot=None)
            D1D2 = tmp.D1D2
            if mpicomm.rank == 0:
                logger.info('Adding correlation function with edges = {}'.format(tmp.D1D2.edges[0]))
            result += tmp
        results.append(result)
    corr = results[0].concatenate_x(*results)
    corr.D1D2.attrs['options'] = dict(options)
    corr.D1D2.attrs['nsplits'] = nsplits
    if 'zsnap' in options:
        corr.D1D2.attrs['zeff'] = options['zsnap']
    if tmp.mpicomm.rank == 0:
        if output is not None:
            output.save(corr)
        if output is not None:
            postprocess_correlation_function(output)

    return corr


def postprocess_power_spectrum(power, rebinning_factors=(1, 5)):
    import numpy as np
    from matplotlib import pyplot as plt
    from pypower import PowerSpectrumStatistics, PowerSpectrumSmoothWindow, PowerSpectrumOddWideAngleMatrix, PowerSpectrumSmoothWindowMatrix
    from desi_y1_files import load

    fn = str(power)
    power = PowerSpectrumStatistics.load(fn)
    for factor in rebinning_factors:
        rebinned = power[:(power.shape[0] // factor) * factor:factor]
        rebinned.save_txt(fn.replace('.npy', '_d{:.3f}.txt'.format(rebinned.edges[0][1] - rebinned.edges[0][0])))


def compute_power_spectrum(data, output=None, output_wmatrix=None, all_shifted=None, data2=None, all_shifted2=None,
                           mpicomm=None, nthreads=1, postprocess_power_spectrum=postprocess_power_spectrum, get_clustering_positions_weights=get_clustering_positions_weights):

    import os
    import logging
    import numpy as np
    from mpytools import Catalog
    from pypower import CatalogFFTPower, MeshFFTWindow, mpi
    from cosmoprimo.fiducial import DESI
    from desi_y1_files.catalog_tools import concatenate_data_randoms
    from desi_y1_files import load

    logger = logging.getLogger('PowerSpectrum')
    if mpicomm is None: mpicomm = mpi.COMM_WORLD

    cosmo = DESI()
    autocorr = data2 is None
    with_shifted = all_shifted is not None

    default_options = {'binning': 'lin', 'weighting': 'default', 'tracer': '', 'nran': 4, 'los': 'x', 'cellsize': 2., 'boxsize': 2000., 'boxcenter': 1000.}
    options = {**default_options, **output.options}
    bin_type = options['binning']  # not used
    weight_type = options['weighting']
    tracer = options['tracer']
    nran = options['nran']
    cellsize = options['cellsize']
    boxsize = options['boxsize']
    boxcenter = options['boxcenter']
    los = options['los']

    if mpicomm.rank == 0:
        logger.info('Using options = {}.'.format(options))

    dtype = 'f8'
    ells = [0, 2, 4]
    edges = {'min': 0., 'step': 0.001}
    kwargs = {'resampler': 'tsc', 'interlacing': 3, 'boxsize': boxsize, 'boxcenter': boxcenter, 'cellsize': cellsize, 'los': los if los is not None else 'x', 'wrap': True, 'dtype': dtype}
    weight_attrs = {}
    z = options['zsnap']

    data_positions1 = data_weights1 = data_positions2 = data_weights2 = None
    shifted_positions1 = shifted_weights1 = shifted_positions2 = shifted_weights2 = None

    randoms_splits_size = None  # simply split according to input randoms
    catalog_attrs = {'z': z, 'los': los, 'cosmo': cosmo, 'option': options.get('catalog', None)}
    data_positions1, data_weights1 = get_clustering_positions_weights(Catalog.read(data, mpicomm=mpicomm, filetype='fits'), **catalog_attrs)
    if with_shifted:
        shifted = [[get_clustering_positions_weights(load(shifted, mpicomm=mpicomm), **catalog_attrs)] for shifted in all_shifted[:nran]]
        (shifted_positions1, shifted_weights1) = concatenate_data_randoms([(data_positions1, data_weights1)], *shifted, randoms_splits_size=randoms_splits_size, concatenate=True, mpicomm=mpicomm)[1]

    if not autocorr:
        data_positions2, data_weights2 = get_clustering_positions_weights(Catalog.read(data, mpicomm=mpicomm, filetype='fits'), **catalog_attrs)
        if with_shifted:
            shifted2 = [[get_clustering_positions_weights(load(shifted, mpicomm=mpicomm), **catalog_attrs)] for shifted in all_shifted2[:nran]]
            (shifted_positions2, shifted_weights2) = concatenate_data_randoms([(data_positions2, data_weights2)], *shifted2, weight_attrs=weight_attrs, randoms_splits_size=randoms_splits_size, concatenate=True, mpicomm=mpicomm)[1]

    randoms_kwargs = dict(shifted_positions1=shifted_positions1, shifted_weights1=shifted_weights1,
                          shifted_positions2=shifted_positions2, shifted_weights2=shifted_weights2)

    power = CatalogFFTPower(data_positions1=data_positions1, data_weights1=data_weights1,
                            data_positions2=data_positions2, data_weights2=data_weights2,
                            edges=edges, ells=ells, position_type='xyz',
                            **randoms_kwargs, **kwargs, mpicomm=mpicomm, mpiroot=None).poles

    power.attrs['options'] = dict(options)
    if 'zsnap' in options:
        power.attrs['zsnap'] = options['zsnap']
    if mpicomm.rank == 0:
        if output is not None:
            output.save(power)

    if output_wmatrix is not None:
        edgesin = np.linspace(0., 0.5, 501)
        wmatrix = MeshFFTWindow(edgesin=edgesin, power_ref=power, periodic=True)
        if mpicomm.rank == 0:
            wmatrix.save(output_wmatrix)

    if mpicomm.rank == 0 and output is not None:
        postprocess_power_spectrum(output)

    return power


if __name__ == '__main__':

    todo = ['reconstruction', 'correlation', 'correlation_recon', 'power', 'power_recon']
    todo = ['correlation']

    from desipipe import setup_logging
    from desipipe.file_manager import BaseFile

    setup_logging()

    if 'reconstruction' in todo:
        data = [BaseFile('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/ELG/z0.950/AbacusSummit_base_c000_ph000/ELG_real_space.sub{:d}.fits.gz'.format(isub), filetype='catalog') for isub in range(64)]
        output_data = BaseFile('_tests/ELG_clustering.IFTrecsym.dat.fits', filetype='catalog', options={'algorithm': 'IFFT', 'mode': 'recsym', 'smoothing_radius': 10., 'cellsize': 20., 'tracer': 'LRG', 'los': 'x', 'zsnap': 0.950})
        all_output_randoms = [BaseFile('_tests/ELG_{iran:d}_clustering.IFTrecsym.ran.fits', filetype='catalog', options={'mode': 'recsym', 'iran': iran}) for iran in range(4)]
        compute_reconstruction(data, output_data, all_output_randoms)

    if 'correlation' in todo:
        data = [BaseFile('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/ELG/z0.950/AbacusSummit_base_c000_ph000/ELG_real_space.sub{:d}.fits.gz'.format(isub), filetype='catalog') for isub in range(64)]
        output = BaseFile('_tests/correlation.npy', options={'binning': 'lin', 'zsnap': 0.950, 'los': None})
        compute_correlation_function(data, output=output)

    if 'correlation_recon' in todo:
        data = BaseFile('_tests/ELG_clustering.IFTrecsym.dat.fits', filetype='catalog')
        all_shifted = [BaseFile('_tests/ELG_{iran:d}_clustering.IFTrecsym.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        output = BaseFile('_tests/correlation.npy', options={'binning': 'lin', 'nran': 1, 'zsnap': 0.950, 'split': 20.})
        compute_correlation_function(data, all_shifted=all_shifted, output=output)

    if 'power' in todo:
        data = [BaseFile('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/ELG/z0.950/AbacusSummit_base_c000_ph000/ELG_real_space.sub{:d}.fits.gz'.format(isub), filetype='catalog') for isub in range(64)]
        output = BaseFile('_tests/power.npy', filetype='power', options={'zsnap': 0.950, 'cellsize': 50})
        output_wmatrix = BaseFile('_tests/wmatrix.npy', filetype='wmatrix')
        compute_power_spectrum(data, output=output, output_wmatrix=output_wmatrix)

    if 'power_recon' in todo:
        data = BaseFile('_tests/ELG_clustering.IFTrecsym.dat.fits', filetype='catalog')
        all_shifted = [BaseFile('_tests/ELG_{iran:d}_clustering.IFTrecsym.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        output = BaseFile('_tests/power.npy', filetype='power', options={'zsnap': 0.950, 'cellsize': 50})
        compute_power_spectrum(data, all_shifted=all_shifted, output=output)
