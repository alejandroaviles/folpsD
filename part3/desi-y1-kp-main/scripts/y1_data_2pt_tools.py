# Run reconstruction and write catalogs to disk
def compute_reconstruction(data, all_randoms, output_data, all_output_randoms, mpicomm=None):
    import logging
    import numpy as np
    import mpytools as mpy
    from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction
    from cosmoprimo.utils import DistanceToRedshift
    from cosmoprimo.fiducial import DESI
    from desipipe.file_manager import FileEntryCollection
    from desi_y1_files import load
    from desi_y1_files.catalog_tools import get_clustering_positions_weights

    logger = logging.getLogger('Reconstruction')
    if mpicomm is None: mpicomm = mpy.COMM_WORLD

    def get_bias(tracer='ELG'):
        if tracer.startswith('BGS'):
            bias = 1.5
        elif tracer.startswith('LRG+ELG'):
            bias = 1.6
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
    if not isinstance(output_data, (tuple, list)):
        output_data = [output_data]
    all_output_randoms = list(all_output_randoms)
    if not isinstance(all_output_randoms[0], (tuple, list, FileEntryCollection)):
        all_output_randoms = [all_output_randoms]
    all_output_randoms = [list(output_randoms) for output_randoms in all_output_randoms]  # recsym, reciso

    options = output_data[0].options
    algorithm = options['algorithm']
    smoothing_radius = options['smoothing_radius']  # recsym or reciso
    cellsize = 4.  # options.get('cellsize', 4.)
    dtype = 'f8'
    zrange = options.get('recon_zrange', options.get('zrange', None))
    weight_type = options.get('recon_weighting', options.get('weighting', 'default'))

    catalog_attrs = {'cosmo': cosmo, 'zrange': None, 'weight_type': weight_type, 'tracer': options['tracer'], 'option': options.get('catalog', None)}
    data, all_randoms = load(data, mpicomm=mpicomm), [load(randoms, mpicomm=mpicomm) for randoms in all_randoms]
    if zrange is not None:

        def select_zrange(catalog):
            return catalog[(catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])]
        
        data = select_zrange(data)
        all_randoms = [select_zrange(randoms) for randoms in all_randoms]

    data_z, data_positions, data_weights = get_clustering_positions_weights(data, **catalog_attrs, return_z=True)  # default weights
    data_weights = data_weights[-1]
    zeff = (data_z * data_weights).csum() / data_weights.csum()
    f = cosmo.growth_rate(zeff)
    bias = get_bias(tracer=options['tracer'])
    czmin, czmax = data_z.cmin(), data_z.cmax()
    if mpicomm.rank == 0:
        logger.info('Using f = {:.3f} and bias = {:.3f} at zeff = {:.2f} ({:.3f} < z < {:.3f}) with smoothing = {:.3f}.'.format(f, bias, zeff, czmin, czmax, smoothing_radius))
        logger.info('With weighting = {}.'.format(weight_type))
    Reconstruction = {'IFFT': IterativeFFTReconstruction, 'IFFTP': IterativeFFTParticleReconstruction, 'MG': MultiGridReconstruction}[algorithm]
    recon = Reconstruction(f=f, bias=bias, positions=data_positions, los='local', cellsize=cellsize, boxpad=1.2, position_type='rdd', dtype=dtype, mpicomm=mpicomm, mpiroot=None)
    recon.assign_data(data_positions, data_weights)
    #csum = 0.
    for randoms in all_randoms:
        randoms_positions, randoms_weights = get_clustering_positions_weights(randoms, **catalog_attrs)
        recon.assign_randoms(randoms_positions, randoms_weights[-1])
        #csum += randoms_weights[-1].sum()
        #print(recon.mesh_randoms.csum(), csum)
    #exit()
    recon.set_density_contrast(smoothing_radius=smoothing_radius)
    recon.run()
    if mpicomm.rank == 0:
        logger.info('Shifting data.')
    # If using IterativeFFTParticleReconstruction, displacements are to be taken at the reconstructed data real-space positions;
    if Reconstruction is IterativeFFTParticleReconstruction:
        data_positions_recon = recon.read_shifted_positions('data', dtype=dtype)
    else:
        data_positions_recon = recon.read_shifted_positions(data_positions, dtype=dtype)
    distance_to_redshift = DistanceToRedshift(cosmo.comoving_radial_distance)
    data['RA'], data['DEC'], data['Z'] = data_positions_recon[:2] + [distance_to_redshift(data_positions_recon[2])]
    for output_data in output_data:  # recysm, reciso
        output_data.save(data)

    nrandoms = min(len(r) for r in [all_randoms] + all_output_randoms)
    nrandoms_splits = min(6, mpicomm.size)  # maximum 6 output_randoms written in parallel
    nsplits = (nrandoms + nrandoms_splits - 1) // nrandoms_splits
    for isplit in range(nsplits):
        sl = slice(isplit * nrandoms // nsplits, (isplit + 1) * nrandoms // nsplits)
        if mpicomm.rank == 0:
            logger.info('Shifting randoms {}.'.format(sl))
        tosave = []
        for irank, (randoms, output_randoms) in enumerate(zip(all_randoms[sl], list(zip(*all_output_randoms))[sl])):
            randoms_positions = get_clustering_positions_weights(randoms, **catalog_attrs)[0]
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
                randoms['RA'], randoms['DEC'], randoms['Z'] = randoms_positions_recon[:2] + [distance_to_redshift(randoms_positions_recon[2])]
                tosave.append((output_randoms, randoms[['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP']].gather(mpiroot=irank)))
        for output_randoms, randoms in tosave:
            if randoms is not None:
                output_randoms.save(randoms)


def compute_zeff(z1, weights1, z2=None, weights2=None, cosmo=None, zrange=None, mpicomm=None):
    import numpy as np
    import mpytools as mpy
    if cosmo is None:
        from cosmoprimo.fiducial import DESI
        cosmo = DESI()
    if zrange is None:
        zrange = min(mpy.cmin(z, mpicomm=mpicomm) if z is not None else np.inf for z in [z1, z2]), max(mpy.cmax(z, mpicomm=mpicomm) if z is not None else 0 for z in [z1, z2])
    zstep = 0.01
    zbins = np.arange(zrange[0], zrange[1] + zstep / 2., zstep)
    dbins = cosmo.comoving_radial_distance(zbins)
    hist2 = hist1 = mpicomm.allreduce(np.histogram(z1, weights=weights1, density=False, bins=zbins)[0])
    zhist2 = zhist1 = mpicomm.allreduce(np.histogram(z1, weights=z1 * weights1, density=False, bins=zbins)[0])
    if z2 is not None:
        hist2 = mpicomm.allreduce(np.histogram(z2, weights=weights2, density=False, bins=zbins)[0])
        zhist2 = mpicomm.allreduce(np.histogram(z2, weights=z2 * weights2, density=False, bins=zbins)[0])
    z = (zhist1 + zhist2) / (hist1 + hist2)
    z[np.isnan(z)] = 0.
    dv = dbins[1:]**3 - dbins[:-1]**3
    return np.sum(hist1 * hist2 / dv * z) / np.sum(hist1 * hist2 / dv)


def get_footprint(data, all_randoms=tuple(), output=None, compute_zeff=compute_zeff, nside=32, **kwargs):
    """Return footprint, specifying redshift density nbar and area, using Y1 data."""
    import numpy as np
    from mockfactory import RedshiftDensityInterpolator
    from desilike import mpi
    from desilike.observables.galaxy_clustering import CutskyFootprint
    from cosmoprimo.fiducial import DESI
    from desi_y1_files import load
    from desi_y1_files.catalog_tools import get_clustering_positions_weights, concatenate_data_randoms

    cosmo = DESI()

    mpicomm = mpi.COMM_WORLD
    tracer = kwargs.get('tracer', None)
    region = kwargs.get('region', 'ALL')
    zrange = kwargs.get('zrange', None)
    weight_type = kwargs.get('weight_type', 'default_FKP')
    if output is not None:
        tracer = output.options['tracer']
        region = output.options['region']
        zrange = output.options['zrange']
        weight_type = output.options.get('weighting', weight_type)
    regions = [region]
    catalog_attrs = {'cosmo': cosmo, 'zrange': zrange, 'regions': regions, 'weight_type': weight_type, 'tracer': tracer}
    data = get_clustering_positions_weights(load(data, mpicomm=mpicomm), return_z=True, **catalog_attrs)

    all_randoms = [load(randoms, mpicomm=mpicomm) for randoms in all_randoms]
    randoms = [get_clustering_positions_weights(randoms, return_z=True, **catalog_attrs) for randoms in all_randoms]
    if randoms:
        (data_z1, data_positions1, data_weights1), (randoms_z1, randoms_positions1, randoms_weights1) = concatenate_data_randoms(data, *randoms, weight_attrs=None, concatenate=True, mpicomm=mpicomm)
    else:
        data_z1, data_positions1, data_weights1 = concatenate_data_randoms(data, weight_attrs=None, concatenate=True, mpicomm=mpicomm)

    import mpytools as mpy
    data_z1 = mpy.array(data_z1, mpicomm=mpicomm)

    if nside is not None:
        import healpy as hp
        hpindex = hp.ang2pix(nside, *(randoms_positions1[:2] if randoms else data_positions1[:2]), lonlat=True)  # ra, dec
        hpindex = mpy.gather(np.unique(hpindex), mpicomm=mpicomm, mpiroot=0)
        fsky = mpicomm.bcast(np.unique(hpindex).size if mpicomm.rank == 0 else None, root=0) / hp.nside2npix(nside)
        area = fsky * 4. * np.pi * (180. / np.pi)**2
    else:
        if not all_randoms:
            raise ValueError('provide randoms!')
        area = sum(sum(randoms.csize for randoms in randoms) if isinstance(randoms, list) else randoms.csize for randoms in all_randoms) / len(all_randoms) / 2500.
        fsky = area / (4. * np.pi * (180. / np.pi)**2)
    cosmo = DESI()
    step = 0.01
    if zrange is None: zrange = (data_z1.cmin(), data_z1.cmax())
    bins = np.arange(zrange[0], zrange[1] + step / 2., step)
    density = RedshiftDensityInterpolator(z=data_z1, bins=bins, fsky=fsky, distance=cosmo.comoving_radial_distance, mpicomm=mpicomm)
    footprint = CutskyFootprint(area=area, zrange=density.edges, nbar=density.nbar, cosmo=cosmo)
    z, w = data_z1, data_weights1[-1]
    if randoms: z, w = randoms_z1, randoms_weights1[-1]
    footprint.attrs = {}
    footprint.attrs['ndata'] = data_z1.csize
    footprint.attrs['tracer'] = tracer
    footprint.attrs['zeff'] = compute_zeff(z, w, cosmo=cosmo, zrange=zrange, mpicomm=mpicomm)
    if output is not None: footprint.save(output)
    return footprint


def compute_angular_weights(data_full, all_randoms_full, data_full2=None, all_randoms_full2=None, weight_type='default', nthreads=64, gpu=False, regions=None, weight_attrs=None):

    import numpy as np
    from pycorr import TwoPointCorrelationFunction
    from desi_y1_files.catalog_tools import get_full_positions_weights, concatenate_data_randoms

    kwargs = {'dtype': 'f8'}

    autocorr = data_full2 is None
    mpicomm = data_full.mpicomm

    fibered_data_positions1 = fibered_data_weights1 = fibered_data_positions2 = fibered_data_weights2 = None
    parent_data_positions1 = parent_data_weights1 = parent_data_positions2 = parent_data_weights2 = None
    parent_randoms_positions1 = parent_randoms_weights1 = parent_randoms_positions2 = parent_randoms_weights2 = None

    fibered_data = get_full_positions_weights(data_full, weight_type=weight_type, fibered=True, regions=regions, weight_attrs=weight_attrs)
    parent_data = get_full_positions_weights(data_full, weight_type=weight_type, fibered=False, regions=regions, weight_attrs=weight_attrs)
    parent_randoms = get_full_positions_weights(all_randoms_full[0].concatenate(all_randoms_full), weight_type=weight_type, fibered=False, regions=regions, weight_attrs=weight_attrs)
    fibered_data_positions1, fibered_data_weights1 = concatenate_data_randoms(fibered_data, weight_attrs=weight_attrs, mpicomm=mpicomm)
    (parent_data_positions1, parent_data_weights1), ([parent_randoms_positions1], [parent_randoms_weights1]) = concatenate_data_randoms(parent_data, parent_randoms, mpicomm=mpicomm)
    if not autocorr:
        fibered_data = get_full_positions_weights(data_full2, weight_type=weight_type, fibered=True, regions=regions, weight_attrs=weight_attrs)
        parent_data = get_full_positions_weights(data_full2, weight_type=weight_type, fibered=False, regions=regions, weight_attrs=weight_attrs)
        parent_randoms = get_full_positions_weights(all_randoms_full2[0].concatenate(all_randoms_full2), weight_type=weight_type, fibered=False, regions=regions, weight_attrs=weight_attrs)
        fibered_data_positions2, fibered_data_weights2 = concatenate_data_randoms(fibered_data, weight_attrs=weight_attrs, mpicomm=mpicomm)
        (parent_data_positions2, parent_data_weights2), ([parent_randoms_positions2], [parent_randoms_weights2]) = concatenate_data_randoms(parent_data, parent_randoms, mpicomm=mpicomm)

    tedges = np.logspace(-4., 0.5, 41)
    # First D1D2_parent/D1D2_PIP angular weight
    wangD1D2 = TwoPointCorrelationFunction('theta', tedges, data_positions1=fibered_data_positions1, data_weights1=fibered_data_weights1,
                                            data_positions2=fibered_data_positions2, data_weights2=fibered_data_weights2,
                                            randoms_positions1=parent_data_positions1, randoms_weights1=parent_data_weights1,
                                            randoms_positions2=parent_data_positions2, randoms_weights2=parent_data_weights2,
                                            estimator='weight', engine='corrfunc', position_type='rd', nthreads=nthreads, gpu=gpu,
                                            mpicomm=mpicomm, mpiroot=None, **kwargs)

    # First D1R2_parent/D1R2_IIP angular weight
    # Input bitwise weights are automatically turned into IIP
    if autocorr:
         parent_randoms_positions2, parent_randoms_weights2 = parent_randoms_positions1, parent_randoms_weights1
    wangD1R2 = TwoPointCorrelationFunction('theta', tedges, data_positions1=fibered_data_positions1, data_weights1=fibered_data_weights1,
                                            data_positions2=parent_randoms_positions2, data_weights2=parent_randoms_weights2,
                                            randoms_positions1=parent_data_positions1, randoms_weights1=parent_data_weights1,
                                            randoms_positions2=parent_randoms_positions2, randoms_weights2=parent_randoms_weights2,
                                            estimator='weight', engine='corrfunc', position_type='rd', nthreads=nthreads, gpu=gpu,
                                            mpicomm=mpicomm, mpiroot=None, **kwargs)
    wangR1D2 = None
    if not autocorr:
        wangR1D2 = TwoPointCorrelationFunction('theta', tedges, data_positions1=parent_randoms_positions1, data_weights1=parent_randoms_weights1,
                                               data_positions2=fibered_data_positions2, data_weights2=fibered_data_weights2,
                                               randoms_positions1=parent_randoms_positions1, randoms_weights1=parent_randoms_weights1,
                                               randoms_positions2=parent_data_positions2, randoms_weights2=parent_data_weights2,
                                               estimator='weight', engine='corrfunc', position_type='rd', nthreads=nthreads, gpu=gpu,
                                               mpicomm=mpicomm, mpiroot=None, **kwargs)

    wang = {}
    wang['D1D2_twopoint_weights'] = wangD1D2
    wang['D1R2_twopoint_weights'] = wangD1R2
    wang['R1D2_twopoint_weights'] = wangR1D2

    return wang


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


def compute_correlation_function(data, all_randoms, output=None, all_shifted=None, data2=None, all_randoms2=None, all_shifted2=None,
                                 data_full=None, data_full2=None, all_randoms_full=None, all_randoms_full2=None,
                                 wang=None, output_wang=None, nthreads=64, gpu=False, mpicomm=None,
                                 compute_angular_weights=compute_angular_weights, compute_zeff=compute_zeff,
                                 postprocess_correlation_function=postprocess_correlation_function):

    import logging
    import numpy as np
    from pycorr import TwoPointCorrelationFunction, KMeansSubsampler, mpi
    from cosmoprimo.fiducial import DESI
    from desi_y1_files.catalog_tools import get_clustering_positions_weights, split_region, concatenate_data_randoms
    from desi_y1_files import load, is_file_sequence

    logger = logging.getLogger('CorrelationFunction')
    if mpicomm is None: mpicomm = mpi.COMM_WORLD

    cosmo = DESI()
    autocorr = data2 is None
    with_shifted = all_shifted is not None
    corr_type = 'smu'
    default_options = {'binning': 'lin', 'weighting': 'default', 'zrange': None, 'region': None, 'tracer': '', 'cut': None, 'njack': 0, 'nran': 4, 'split': None, 'catalog': None}
    options = {**default_options, **output.options}
    bin_type = options['binning']
    weight_type = options['weighting']
    zrange = zrange_sel = options['zrange']
    if zrange_sel is None: zrange_sel = options['recon_zrange']
    region = options['region']
    tracer = options['tracer']
    #regions = split_region(region, tracer=tracer)
    regions = [None]  # we assume input clustering catalogs are correctly normalized
    if (data[0] if is_file_sequence(data) else data).options.get('region', region) != region: regions = [region]
    cut = options['cut']
    njack = options['njack']
    nran = options['nran']
    split_randoms_above = options['split']
    if split_randoms_above is None: split_randoms_above = np.inf
    dtype = 'f8'
    kwargs = {'dtype': dtype}
    weight_attrs = {}

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

    if 'angular' in weight_type:
        if wang is None:
            wang = compute_angular_weights(data_full=load(data_full, mpicomm=mpicomm), all_randoms_full=[load(randoms, mpicomm=mpicomm) for randoms in all_randoms_full], data_full2=load(data_full2, mpicomm=mpicomm) if not autocorr else None, all_randoms_full2=[load(randoms, mpicomm=mpicomm) for randoms in all_randoms_full2] if not autocorr else None, weight_type=weight_type, regions=regions, weight_attrs=weight_attrs, nthreads=nthreads, gpu=gpu)
        else:
            wang = np.load(wang)[()]
            for name, value in wang.items():
                wang[name] = TwoPointCorrelationFunction.from_state(value)

    data_positions1 = data_weights1 = data_samples1 = data_positions2 = data_weights2 = data_samples2 = None
    randoms_z1 = randoms_positions1 = randoms_weights1 = randoms_samples1 = randoms_z2 = randoms_positions2 = randoms_weights2 = randoms_samples2 = None
    shifted_positions1 = shifted_weights1 = shifted_samples1 = shifted_positions2 = shifted_weights2 = shifted_samples2 = None
    jack_positions = None

    weight_type_randoms = weight_type.replace('bitwise', '')
    randoms_splits_size = None  # simply split according to input randoms
    catalog_attrs = {'cosmo': cosmo, 'zrange': zrange, 'regions': regions, 'tracer': tracer, 'option': options.get('catalog', None)}
    data = get_clustering_positions_weights(load(data, mpicomm=mpicomm), weight_type=weight_type, **catalog_attrs)
    randoms = [get_clustering_positions_weights(load(randoms, mpicomm=mpicomm), weight_type=weight_type_randoms, return_z=True, **{**catalog_attrs, 'zrange': zrange_sel}) for randoms in all_randoms[:nran]]

    (data_positions1, data_weights1), (randoms_z1, randoms_positions1, randoms_weights1) = concatenate_data_randoms(data, *randoms, weight_attrs=weight_attrs, randoms_splits_size=randoms_splits_size, mpicomm=mpicomm)
    if with_shifted:
        shifted = [get_clustering_positions_weights(load(shifted, mpicomm=mpicomm), weight_type=weight_type_randoms, **catalog_attrs) for shifted in all_shifted[:nran]]
        (shifted_positions1, shifted_weights1) = concatenate_data_randoms(data, *shifted, weight_attrs=weight_attrs, randoms_splits_size=randoms_splits_size, mpicomm=mpicomm)[1]
    jack_positions = data_positions1

    if not autocorr:
        data2 = get_clustering_positions_weights(load(data2, mpicomm=mpicomm), weight_type=weight_type, **catalog_attrs)
        randoms2 = [get_clustering_positions_weights(load(randoms, mpicomm=mpicomm), weight_type=weight_type_randoms, return_z=True, **{**catalog_attrs, 'zrange': zrange_sel}) for randoms in all_randoms2[:nran]]
        (data_positions2, data_weights2), (randoms_z2, randoms_positions2, randoms_weights2) = concatenate_data_randoms(data2, *randoms2, weight_attrs=weight_attrs, randoms_splits_size=randoms_splits_size, mpicomm=mpicomm)
        if with_shifted:
            shifted2 = [get_clustering_positions_weights(load(shifted, mpicomm=mpicomm), weight_type=weight_type_randoms, **catalog_attrs) for shifted in all_shifted2[:nran]]
            (shifted_positions2, shifted_weights2) = concatenate_data_randoms(data2, *shifted2, weight_attrs=weight_attrs, randoms_splits_size=randoms_splits_size, mpicomm=mpicomm)[1]
        jack_positions = [np.concatenate(p1, p2) for p1, p2 in zip(jack_positions, data_positions2)]
    zeff = compute_zeff(np.concatenate(randoms_z1),
                        weights1=np.concatenate([weights[-1] for weights in randoms_weights1]),
                        z2=np.concatenate(randoms_z2) if not autocorr else None,
                        weights2=np.concatenate([weights[-1] for weights in randoms_weights2]) if not autocorr else None,
                        cosmo=cosmo, zrange=zrange_sel, mpicomm=mpicomm)

    if njack > 1:
        subsampler = KMeansSubsampler('angular', positions=jack_positions, nsamples=njack, nside=512, random_state=42, position_type='rdd',
                                      dtype=dtype, mpicomm=mpicomm, mpiroot=None)

        data_samples1 = subsampler.label(data_positions1)
        randoms_samples1 = list(map(subsampler.label, randoms_positions1))
        if with_shifted:
            shifted_samples1 = list(map(subsampler.label, shifted_positions1))
        if not autocorr:
            data_samples2 = subsampler.label(data_positions2)
            randoms_samples2 = list(map(subsampler.label, randoms_positions2))
            if with_shifted:
                shifted_samples2 = list(map(subsampler.label, shifted_positions2))
                
    kwargs.update(wang or {})
    selection_attrs = None
    if cut is not None: selection_attrs = {cut[0]: (cut[1], np.inf)}
    randoms_kwargs = dict(randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1, randoms_samples1=randoms_samples1,
                          randoms_positions2=randoms_positions2, randoms_weights2=randoms_weights2, randoms_samples2=randoms_samples2,
                          shifted_positions1=shifted_positions1, shifted_weights1=shifted_weights1, shifted_samples1=shifted_samples1,
                          shifted_positions2=shifted_positions2, shifted_weights2=shifted_weights2, shifted_samples2=shifted_samples2)

    zedges = np.array(list(zip(edges[0][:-1], edges[0][1:])))
    mask = zedges[:, 0] >= split_randoms_above
    zedges = [zedges[~mask], zedges[mask]]
    split_edges, split_randoms = [], []
    for ii, zedge in enumerate(zedges):
        if zedge.size:
            split_edges.append([np.append(zedge[:, 0], zedge[-1, -1])] + list(edges[1:]))
            split_randoms.append(ii > 0)

    results = []
    nsplits = min(len(pw) if pw is not None else np.inf for pw in randoms_kwargs.values())
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
                    if arrays is None:
                        continue
                    elif 'samples' in name:
                        array = np.concatenate(arrays, axis=0)
                    else:  # e.g., list of positions / weights
                        array = [np.concatenate([arr[iarr] for arr in arrays], axis=0) for iarr in range(len(arrays[0]))]
                    tmp_randoms_kwargs[name] = array
            mesh_refine_factors = (4, 4, 2) if i_split_randoms else (2, 2, 1)
            tmp = TwoPointCorrelationFunction(corr_type, edges, data_positions1=data_positions1, data_weights1=data_weights1, data_samples1=data_samples1,
                                              data_positions2=data_positions2, data_weights2=data_weights2, data_samples2=data_samples2,
                                              engine='corrfunc', position_type='rdd', nthreads=nthreads, gpu=gpu, **tmp_randoms_kwargs, **kwargs,
                                              D1D2=D1D2, selection_attrs=selection_attrs, mpicomm=mpicomm, mpiroot=None, mesh_refine_factors=mesh_refine_factors)
            D1D2 = tmp.D1D2
            if mpicomm.rank == 0:
                logger.info('Adding correlation function with edges = {}'.format(tmp.D1D2.edges[0]))
            result += tmp
        results.append(result)
    corr = results[0].concatenate_x(*results)
    corr.D1D2.attrs['options'] = dict(options)
    corr.D1D2.attrs['nsplits'] = nsplits
    corr.D1D2.attrs['zeff'] = zeff
    if tmp.mpicomm.rank == 0:
        if output is not None:
            output.save(corr)
        if output_wang is not None:
            np.save(output_wang, {name: value.__getstate__() for name, value in wang.items()})
        if output is not None:
            postprocess_correlation_function(output)

    return corr


def postprocess_power_spectrum(power, rebinning_factors=(1, 5), output_power_nodirect=None, windows=None, output_wmatrix=None, output_wmatrix_nodirect=None):
    import numpy as np
    from matplotlib import pyplot as plt
    from pypower import PowerSpectrumStatistics, PowerSpectrumSmoothWindow, PowerSpectrumOddWideAngleMatrix, PowerSpectrumSmoothWindowMatrix
    from desi_y1_files import load

    fn = str(power)
    power = PowerSpectrumStatistics.load(fn)
    fn_nodirect = power_nodirect = None

    if output_power_nodirect is not None:
        power_nodirect = power.deepcopy()
        power_nodirect.power_direct_nonorm[...] = 0.
        for name in ['corr_direct_nonorm', 'sep_direct']: setattr(power_nodirect, name, None)
        power_nodirect.save(output_power_nodirect)
        fn_nodirect = str(output_power_nodirect)

    for output, fn in zip([power, power_nodirect], [fn, fn_nodirect]):

        if output is None: continue

        for factor in rebinning_factors:
            rebinned = output[:(output.shape[0] // factor) * factor:factor]
            db = rebinned.edges[0][1] - rebinned.edges[0][0]
            ff = fn.replace('.npy', '_d{:.3f}'.format(db))
            rebinned.save_txt(ff + '.txt')
            rebinned.plot(fn=ff + '.png')
            plt.close()

    for output, nodirect in zip([output_wmatrix, output_wmatrix_nodirect], [False, True]):

        if output is None: continue

        windows = load(windows, load=PowerSpectrumSmoothWindow.load)
        if isinstance(windows, (tuple, list)):
            argsort = np.argsort([np.max(window.attrs['boxsize']) for window in windows])[::-1]
            windows = [windows[ii] for ii in argsort]
            window = windows[0].concatenate_x(*windows, frac_nyq=0.9)
        else:
            window = windows
        if nodirect:
            window = window.deepcopy()
            window.power_direct_nonorm[...] = 0.
            for name in ['corr_direct_nonorm', 'sep_direct']: setattr(window, name, None)

        # Let us compute the wide-angle and window function matrix
        ellsin = (0, 2, 4)  # input (theory) multipoles
        wa_orders = 1 # wide-angle order
        sep = np.geomspace(1e-4, 1e5, 1024 * 16) # configuration space separation for FFTlog
        kin_rebin = 2 # rebin input theory to save memory
        #sep = np.geomspace(1e-4, 2e4, 1024 * 16) # configuration space separation for FFTlog, 2e4 > sqrt(3) * 8000
        #kin_rebin = 4 # rebin input theory to save memory
        kin_lim = (0, 2e1) # pre-cut input (theory) ks to save some memory
        # Input projections for window function matrix:
        # theory multipoles at wa_order = 0, and wide-angle terms at wa_order = 1
        projsin = tuple(ellsin) + tuple(PowerSpectrumOddWideAngleMatrix.propose_out(ellsin, wa_orders=wa_orders))
        # Window matrix
        wmatrix = PowerSpectrumSmoothWindowMatrix(power, projsin=projsin, window=window, sep=sep, kin_rebin=kin_rebin, kin_lim=kin_lim)
        # We resum over theory odd-wide angle
        wmatrix.resum_input_odd_wide_angle()
        wmatrix.attrs.update(power.attrs)
        output.save(wmatrix)


def compute_power_spectrum(data, all_randoms, output=None, output_wmatrix=None, output_window=None, all_shifted=None, data2=None, all_randoms2=None, all_shifted2=None,
                           data_full=None, data_full2=None, all_randoms_full=None, all_randoms_full2=None,
                           wang=None, output_wang=None, mpicomm=None, nthreads=1,
                           compute_angular_weights=compute_angular_weights, compute_zeff=compute_zeff,
                           postprocess_power_spectrum=postprocess_power_spectrum):

    import os
    import logging
    import numpy as np
    from pypower import CatalogFFTPower, CatalogSmoothWindow, mpi
    from cosmoprimo.fiducial import DESI
    from desi_y1_files.catalog_tools import get_clustering_positions_weights, split_region, concatenate_data_randoms
    from desi_y1_files import load, is_file_sequence

    logger = logging.getLogger('PowerSpectrum')
    if mpicomm is None: mpicomm = mpi.COMM_WORLD

    cosmo = DESI()
    autocorr = data2 is None
    with_shifted = all_shifted is not None

    default_options = {'binning': 'lin', 'weighting': 'default', 'zrange': None, 'region': None, 'tracer': '', 'cut': None, 'nran': 4, 'cellsize': 6., 'boxsize': None, 'catalog': None}
    options = {**default_options, **output.options}
    bin_type = options['binning']  # not used
    weight_type = options['weighting']
    zrange = zrange_sel = options['zrange']
    if zrange_sel is None: zrange_sel = options['recon_zrange']
    region = options['region']
    tracer = options['tracer']
    #regions = split_region(region, tracer=tracer)
    regions = [None]  # we assume input clustering catalogs are correctly normalized
    if (data[0] if is_file_sequence(data) else data).options.get('region', region) != region: regions = [region]
    cut = options['cut']
    nran = options['nran']
    cellsize = options['cellsize']
    boxsize = options['boxsize']

    if mpicomm.rank == 0:
        logger.info('Using options = {}.'.format(options))

    dtype = 'f8'
    ells = [0, 2, 4]
    edges = {'min': 0., 'step': 0.001}
    kwargs = {'dtype': dtype, 'resampler': 'tsc', 'interlacing': 3, 'boxsize': boxsize, 'cellsize': cellsize, 'los': 'firstpoint'}
    weight_attrs = {}
    direct_attrs = {'nthreads': nthreads}

    if 'angular' in weight_type:
        if wang is None:
            color = mpicomm.rank % nthreads == 0
            subcomm = mpicomm.Split(color, 0)
            if color:
                wang = compute_angular_weights(data_full=load(data_full, mpicomm=subcomm), all_randoms_full=load(all_randoms_full, mpicomm=subcomm),
                                               data_full2=load(data_full2, mpicomm=subcomm) if not autocorr else None, all_randoms_full2=load(all_randoms_full2, mpicomm=subcomm) if not autocorr else None,
                                               weight_type=weight_type, regions=regions,
                                               weight_attrs=weight_attrs, nthreads=nthreads, gpu=False)
            mpi.barrier_idle(mpicomm=mpicomm)
            wang = mpicomm.bcast(wang, root=0)
        else:
            wang = np.load(wang)[()]
            from pycorr import TwoPointCorrelationFunction
            for name, value in wang.items():
                wang[name] = TwoPointCorrelationFunction.from_state(value)

    data_positions1 = data_weights1 = data_positions2 = data_weights2 = None
    randoms_z1 = randoms_positions1 = randoms_weights1 = randoms_z2 = randoms_positions2 = randoms_weights2 = None
    shifted_positions1 = shifted_weights1 = shifted_positions2 = shifted_weights2 = None

    weight_type_randoms = weight_type.replace('bitwise', '')
    catalog_attrs = {'cosmo': cosmo, 'zrange': zrange, 'regions': regions, 'tracer': tracer, 'option': options.get('catalog', None)}
    data = get_clustering_positions_weights(load(data, mpicomm=mpicomm), weight_type=weight_type, **catalog_attrs)
    randoms = [get_clustering_positions_weights(load(randoms, mpicomm=mpicomm), weight_type=weight_type_randoms, return_z=True, **{**catalog_attrs, 'zrange': zrange_sel}) for randoms in all_randoms[:nran]]
    (data_positions1, data_weights1), (randoms_z1, randoms_positions1, randoms_weights1) = concatenate_data_randoms(data, *randoms, weight_attrs=weight_attrs, concatenate=True, mpicomm=mpicomm)
    if with_shifted:
        shifted = [get_clustering_positions_weights(load(shifted, mpicomm=mpicomm), weight_type=weight_type_randoms, **catalog_attrs) for shifted in all_shifted[:nran]]
        (shifted_positions1, shifted_weights1) = concatenate_data_randoms(data, *shifted, weight_attrs=weight_attrs, concatenate=True, mpicomm=mpicomm)[1]
    if not autocorr:
        data2 = get_clustering_positions_weights(load(data2, mpicomm=mpicomm), weight_type=weight_type, **catalog_attrs)
        randoms2 = [get_clustering_positions_weights(load(randoms, mpicomm=mpicomm), weight_type=weight_type_randoms, return_z=True, **{**catalog_attrs, 'zrange': zrange_sel}) for randoms in all_randoms2[:nran]]
        (data_positions2, data_weights2), (randoms_z2, randoms_positions2, randoms_weights2) = concatenate_data_randoms(data2, *randoms2, weight_attrs=weight_attrs, concatenate=True, mpicomm=mpicomm)
        if with_shifted:
            shifted2 = [get_clustering_positions_weights(load(shifted, mpicomm=mpicomm), weight_type=weight_type_randoms, **catalog_attrs) for shifted in all_shifted2[:nran]]
            (shifted_positions2, shifted_weights2) = concatenate_data_randoms(data2, *shifted2, weight_attrs=weight_attrs, concatenate=True, mpicomm=mpicomm)[1]

    zeff = compute_zeff(randoms_z1, weights1=randoms_weights1[-1], z2=randoms_z2, weights2=randoms_weights2[-1] if not autocorr else None, cosmo=cosmo, zrange=zrange_sel, mpicomm=mpicomm)

    randoms_kwargs = dict(randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                          randoms_positions2=randoms_positions2, randoms_weights2=randoms_weights2,
                          shifted_positions1=shifted_positions1, shifted_weights1=shifted_weights1,
                          shifted_positions2=shifted_positions2, shifted_weights2=shifted_weights2)

    kwargs.update(wang or {})
    direct_edges = win_direct_selection_attrs = direct_selection_attrs = None
    if wang or len(data_weights1) > 1:
        direct_selection_attrs = {'theta': (0., 1.)}
    elif cut is not None:
        win_direct_selection_attrs = direct_selection_attrs = {cut[0]: (0., cut[1])}
        #direct_edges = {'min': 0., 'step': 0.1, 'max': 100.}  # use this to reduce the computing time for direct pair counts to a few seconds
        direct_edges = {'min': 0., 'step': 0.1}
        #direct_edges = None

    power = CatalogFFTPower(data_positions1=data_positions1, data_weights1=data_weights1,
                            data_positions2=data_positions2, data_weights2=data_weights2,
                            edges=edges, ells=ells, position_type='rdd',
                            direct_selection_attrs=direct_selection_attrs, direct_edges=direct_edges,
                            direct_attrs=direct_attrs, **randoms_kwargs, **kwargs, mpicomm=mpicomm, mpiroot=None).poles

    power.attrs['zeff'] = zeff
    power.attrs['options'] = dict(options)
    if mpicomm.rank == 0:
        if output is not None:
            output.save(power)
        if output_wang is not None:
            np.save(output_wang, {name: value.__getstate__() for name, value in wang.items()})

    window = None
    if output_wmatrix is not None or output_window is not None:

        boxsize = power.attrs['boxsize']
        boxscales = [1., 5., 20.]
        windows = []
        if output_window is not None:
            islist_output_window = not isinstance(output_window, (str, os.PathLike))
            if islist_output_window:
                output_window = list(output_window)
                tmp = [getattr(output, 'options', {}).get('boxscale', None) for output in output_window]
                if not any(tt is None for tt in tmp):
                    boxscales = np.array(tmp)
                    iboxsizes = np.argsort(boxscales)
                    boxscales = boxscales[iboxsizes]
                assert len(output_window) >= len(boxscales), f'provide as many output_window as boxscales = {len(boxscales):d}'
        boxsizes = boxsize * np.array(boxscales)
        edges = {'step': 2. * np.pi / np.max(boxsizes)}
        for iboxsize, boxsize in enumerate(boxsizes):
            windows.append(CatalogSmoothWindow(**{name: array for name, array in randoms_kwargs.items() if 'randoms' in name},
                                               power_ref=power, edges=edges, boxsize=boxsize, position_type='rdd',
                                               direct_attrs=direct_attrs,
                                               direct_selection_attrs=win_direct_selection_attrs if iboxsize == 0 else None,
                                               direct_edges=direct_edges if iboxsize == 0 else None,
                                               mpicomm=mpicomm, mpiroot=None).poles)

        if windows[0].mpicomm.rank == 0:
            windows[0].log_info('Concatenating windows')
            window = windows[0].concatenate_x(*windows[::-1], frac_nyq=0.9)
            if output_window is not None:
                if islist_output_window:
                    for iwin, win in enumerate(windows):
                        output_window[iboxsizes[iwin]].save(win)
                else:
                    output_window.save(window)
    if mpicomm.rank == 0 and output is not None:
        postprocess_power_spectrum(output, windows=window, output_wmatrix=output_wmatrix)

    return power, wang


def combine_regions_correlation_function(output, inputs, postprocess_correlation_function=postprocess_correlation_function):
    import numpy as np
    from desi_y1_files import load
    filetype = output.filetype
    inputs = [load(input) for input in inputs]
    zeff = np.average([input.D1D2.attrs['zeff'] for input in inputs], weights=[np.sum(input.D1D2.wnorm) for input in inputs])
    comb = sum(input.normalize() for input in inputs)
    comb.D1D2.attrs['zeff'] = zeff
    output.save(comb)
    postprocess_correlation_function(output)
    return output


def combine_regions_power_spectrum(output, inputs, inputs_wmatrix=None, output_wmatrix=None, postprocess_power_spectrum=postprocess_power_spectrum):
    import numpy as np
    from desi_y1_files import load
    inputs = [load(input) for input in inputs]
    zeff = np.average([input.attrs['zeff'] for input in inputs], weights=[np.sum(input.wnorm) for input in inputs])
    comb_power = sum(inputs)
    comb_power.attrs['zeff'] = zeff
    output.save(comb_power)
    if inputs_wmatrix is not None and output_wmatrix is not None:
        comb_wmatrix = sum(load(tmp) for tmp in inputs_wmatrix)
        comb_wmatrix.attrs['zeff'] = zeff
        output_wmatrix.save(comb_wmatrix)
    postprocess_power_spectrum(output)


def merge_catalogs(output, inputs, factor=1., seed=42):
    import numpy as np
    from mockfactory import Catalog
    inputs = list(inputs)
    ncatalogs = len(inputs)
    catalogs = []
    columns = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP', 'MASK']
    columns += ['WEIGHT_COMP', 'WEIGHT_SYS', 'WEIGHT_ZFAIL', 'NX']
    rng = np.random.RandomState(seed=seed)
    from pyrecon.utils import MemoryMonitor
    with MemoryMonitor() as mem:
        for fn in inputs:
            catalog = Catalog.read(fn)
            mask = rng.uniform(0., 1., catalog.size) < factor / ncatalogs
            catalog.get(catalog.columns())
            columns = [col for col in columns if col in catalog.columns()]
            catalogs.append(catalog[columns][mask])
            mem()
    catalog = Catalog.concatenate(catalogs, intersection=True)
    catalog.write(output)


def merge_randoms_catalogs(output, inputs, factor=1., seed=42):
    import numpy as np
    from mockfactory import Catalog
    inputs = list(inputs)
    ncatalogs = len(inputs)
    catalogs = []
    rng = np.random.RandomState(seed=seed)
    from pyrecon.utils import MemoryMonitor
    concatenate = None
    columns = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP', 'MASK']
    
    def get_uid(ra, dec):
        factor = 1000000
        return np.rint(ra * factor) + 360 * factor * np.rint((dec + 90.) * factor)
    
    def get_uid(ra, dec):
        return ra + 1j * dec
    
    with MemoryMonitor() as mem:
        for fn in inputs:
            catalog = Catalog.read(fn)
            catalog.get(catalog.columns())
            columns = [col for col in columns if col in catalog.columns()]
            if concatenate is None:
                mask = rng.uniform(0., 1., catalog.size) < factor / ncatalogs
                concatenate = catalog[columns][mask]
            else:
                csize = catalog.size
                mask = np.isin(get_uid(catalog['RA'], catalog['DEC']), get_uid(concatenate['RA'], concatenate['DEC']))
                print(mask.sum(), mask.sum() / mask.size, factor / ncatalogs)
                catalog = catalog[~mask]
                if not catalog.csize: break
                print(factor * csize / catalog.size / ncatalogs)
                mask = rng.uniform(0., 1., catalog.size) < factor * csize / catalog.size / ncatalogs
                concatenate = Catalog.concatenate(concatenate, catalog[columns][mask])
            mem()
    concatenate.write(output)


def rotate_wmatrix_ref(power, wmatrix, covariance):
    import anotherpipe.powerestimation.rotatewindow as rw
    import anotherpipe.powerestimation.powerestimate as pe
    cut = bool(wmatrix.options['cut'])
    wmatrix = wmatrix.load()
    kinlim = (wmatrix.xout[0][0] / 2, 0.5)
    koutlim = (0., 0.4)
    kinrebin = 1
    koutrebin = 5
    wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)
    power = power.load()
    power = power[:len(power.k) // koutrebin * koutrebin:koutrebin].select(koutlim)
    covariance = covariance.load(koutlim)
    
    ells = tuple(proj.ell for proj in wmatrix.projsout)
    data = pe.make_from_obs(0., 1., power, wmatrix, covariance)
    mmatrix, state = rw.fit(data, ls=ells, momt=cut, capsig=5, difflfac=10)

    # Rotated data
    rotated_data = pe.rotate(data, mmatrix, ls=ells)

    rotated_power = rotated_data.data.P
    rotated_wmatrix = data['wmatrix'].copy()
    rotated_wmatrix.value = np.array(rotated_data.W(ls=ells)).T
    rotated_covmatrix = rotated_data.C(ls=ells)
    mo = rotated_data.mo
    if len(mmatrix) == 4:
        mt = mmatrix[2]
        m = mmatrix[3]
    else:
        mt = None
        m = None
    return rotated_wmatrix


def postprocess_rotate_wmatrix(rotation, power=None, output_power=None, output_power_marg=None, covariance=None, output_covariance=None, output_covariance_marg=None, output_covariance_syst=None, return_covariance=False, return_power=False):
    import numpy as np
    from desi_y1_files import WindowRotation, is_file_sequence
    rotation = rotation.load(WindowRotation.load)
    koutlim, koutrebin = rotation.attrs['koutlim'], rotation.attrs['koutrebin']
    kcutlim = (0., 0.2)
    ellscut = [0, 2, 4]
    koutlim = (0., 0.4, 0.005)

    toret_covariance = []
    if covariance is not None:
        covariance = covariance.load()
        observable = covariance.observables('power').name
        projs = list(rotation.mask_ellsout)
        covariance = covariance.select(observables=observable, xlim=koutlim[:2])
        index = covariance._index(observables=observable, projs=projs)
        #print(index, len(index), covariance.shape[0])
        #if rotation.with_momt: covmatrix_rotated, precmatrix_rotated = rotation.rotate(covmatrix=covariance.view(), mask_cov=index)[1:]
        #else: covmatrix_rotated = rotation.rotate(covmatrix=covariance.view(), mask_cov=index)[1:]
        # Just apply matrix M
        covmatrix_rotated = rotation.rotate(covmatrix=covariance.view(), mask_cov=index)[1]
        covariance._value[...] = covmatrix_rotated
        toret_covariance.append(covariance)
        if output_covariance is not None:
            output_covariance.save(covariance)
        mask_kout = covariance.observables(observable)._index(xlim=kcutlim, projs=ellscut)
        covariance = covariance.select(observables=observable, xlim=kcutlim, projs=ellscut, select_projs=True)
        if rotation.with_momt:
            #mask_kout = rotation._index_kout(kcutlim)
            ellsrot = list(rotation.kout)
            templates = [mo[mask_kout] for mo in rotation.mmatrix[1]]
            templates = [templates[ellsrot.index(ell)] for ell in ellscut]
            marg_prior = [rotation.marg_prior_mo[ellsrot.index(ell)]**2 for ell in ellscut]

            def marginalize_inv(self, templates, prior=1., **kwargs):
                # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
                from desilike import utils
                index = self._index(**kwargs, concatenate=True)
                templates = np.atleast_2d(np.asarray(templates, dtype='f8'))  # adds first dimension
                deriv = np.zeros(templates.shape[:1] + self.shape[:1], dtype='f8')
                deriv[..., index] = templates
                invcov = self.inv()
                fisher = deriv.dot(invcov).dot(deriv.T)
                derivp = deriv.dot(invcov)
                prior = np.array(prior)
                if prior.ndim == 2:
                    iprior = utils.inv(prior)
                else:
                    iprior = np.ones(templates.shape[:1], dtype='f8')
                    iprior[...] = prior
                    iprior = np.diag(1. / iprior)
                fisher += iprior
                invcov = invcov - derivp.T.dot(np.linalg.solve(fisher, derivp))
                indices = [self._index(observables=observable, concatenate=True) for observable in self._observables]
                value = utils.blockinv([[invcov[np.ix_(index1, index2)] for index2 in indices] for index1 in indices])
                return self.clone(value=value)

            input_covariance = covariance
            #covariance2 = marginalize_inv(covariance, templates, prior=marg_prior, observables=observable)
            covariance = covariance.marginalize(templates, prior=marg_prior, observables=observable)
            attrs = {'templates': templates, 'prior': marg_prior}
            covariance.attrs['rotation'] = attrs
            #print(np.diag(covariance._value) / np.diag(covariance2._value))
            #diff = covariance._value / covariance2._value - 1.
            #print(np.abs(diff).max())
            #assert np.allclose(covariance._value, covariance2._value, atol=3e-5, rtol=3e-5)
            if output_covariance_marg is not None:
                output_covariance_marg.save(covariance)
            if output_covariance_syst is not None:
                covariance_syst = input_covariance.clone(value=covariance._value - input_covariance._value)
                covariance_syst.attrs['rotation'] = attrs
                output_covariance_syst.save(covariance_syst)
            toret_covariance.append(covariance)
    
    if power is None: return toret_covariance
    toret_power, toret_power_marg = [], []
    
    for i, power in enumerate(power):
        from pypower import PowerSpectrumStatistics
        data = PowerSpectrumStatistics.load(power)
        data = data.select(koutlim)
        shotnoise = False #True #False
        ells = list(rotation.mask_ellsout)
        power_rotated = rotation.rotate(data=data(ell=ells, remove_shotnoise=not shotnoise).ravel())[2]
        power_rotated = power_rotated.reshape(len(data.ells), -1) * data.wnorm
        power_rotated[data.ells.index(0)] += data.shotnoise_nonorm * (not shotnoise)
        data.power_nonorm[...] = power_rotated
        data.power_direct_nonorm[...] = 0.
        for name in ['corr_direct_nonorm', 'sep_direct']: setattr(data, name, None)
        tmp = data.copy().select(kcutlim)
        toret_power.append(tmp)
        if output_power is not None:
            tmp.save(output_power[i])
        mo = rotation.mmatrix[1]
        data.power_nonorm.flat[...] -= np.dot(rotation.marg_prior_mo, mo) * data.wnorm
        tmp = data.copy().select(kcutlim)
        toret_power_marg.append(tmp)
        if rotation.with_momt and output_power_marg is not None:
            tmp.save(output_power_marg[i])
    if return_power:
        return toret_power, toret_power_marg


def rotate_wmatrix(wmatrix, covariance, theory=None, data=None, output=None, output_wmatrix=None):
    import os
    import logging
    import numpy as np
    from pypower import PowerSpectrumStatistics
    from desi_y1_files import WindowRotation, is_file_sequence, is_path
    #from desi_y1_files.window import WindowRotationMathilde
    
    logger = logging.getLogger('Rotation')

    cut = bool(wmatrix.options['cut'])
    wmatrix = wmatrix.load()
    kinlim = (wmatrix.xout[0][0] / 2, 0.5)
    #kinlim = (wmatrix.xout[0][0] / 2, 1.)
    #kinlim = (wmatrix.xout[0][0] / 2, np.inf)
    koutlim = (0., 0.4, 0.005)
    kcutlim = (0., 0.2)
    kinrebin = 1
    koutrebin = 5
    #covariance.path = '/global/cfs/cdirs/desi/users/mpinon/Y1/cov/pk/cov_gaussian_pre_ELG_LOPnotqso_GCcomb_1.1_1.6_default_FKP_lin.txt'
    covariance = covariance.load().view(xlim=koutlim[:2], projs=[0, 2, 4])
    wmatrix = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim[:2])
    
    covmatrix = covariance
    #covmatrix = np.eye(covariance.shape[0])
    
    ells = tuple(proj.ell for proj in wmatrix.projsout)
    rotation = WindowRotation(wmatrix, covmatrix, attrs={'kinlim': kinlim, 'kinrebin': kinrebin, 'koutlim': koutlim, 'koutrebin': koutrebin, 'kcutlim': kcutlim})

    def get_power(data, k=None):
        if is_file_sequence(data):
            return np.mean([get_power(dd, k=k) for dd in data], axis=0)
        if isinstance(data, np.ndarray):
            return np.ravel(data)
        if k is None:
            data = data.copy().select(koutlim)
            assert np.allclose(data.k, wmatrix.xout[0])
            data = data(ell=ells, return_k=False)
        else:
            data = data(ell=ells, k=k, complex=False, return_k=False)
        return data.ravel()
    
    def get_shotnoise(data):
        if is_file_sequence(data):
            return np.mean([get_shotnoise(dd) for dd in data])
        return data.shotnoise
    
    nobs, shotnoise = 1, 0.
    if data is not None:
        if is_file_sequence(data):
            data = [PowerSpectrumStatistics.load(dd) if is_path(dd) else dd for dd in data]
            nobs = len(list(data))
        else:
            data = PowerSpectrumStatistics.load(data) if is_path(data) else data
        try: shotnoise = get_shotnoise(data)
        except: pass
        data = get_power(data)
    if theory is not None:
        if is_file_sequence(theory):
            theory = [PowerSpectrumStatistics.load(dd) if is_path(dd) else dd for dd in theory]
        else:
            theory = PowerSpectrumStatistics.load(theory) if is_path(theory) else theory
        theory = np.nan_to_num(get_power(theory, k=rotation.kin[0]))
    marg = data is not None and theory is not None

    if output is not None:
        base_fn = os.path.splitext(output)[0]
    shotnoise = 0.
    plot = output is not None
    if plot:
        rotation.plot_covmatrix(fn='{}_covmatrix_norotation.png'.format(base_fn))
        if marg:
            rotation.plot_validation(data=data, theory=theory, covmatrix=covariance, shotnoise=shotnoise, marg_shotnoise=True, klim=kcutlim, nobs=nobs, fn='{}_validation_norotation.png'.format(base_fn))
    rotation.fit(Minit='momt' if cut else None, max_sigma_W=5, max_sigma_R=5, factor_diff_ell=10, csub=False)

    #rotation = WindowRotation.load(output)
    if plot:
        rotation.plot_covmatrix(fn='{}_covmatrix.png'.format(base_fn))
        rotation.plot_wmatrix(fn='{}_wmatrix.png'.format(base_fn))
        rotation.plot_compactness(fn='{}_compactness.png'.format(base_fn))
    if marg:
        if plot:
            rotation.plot_validation(data=data, theory=theory, covmatrix=covariance, shotnoise=shotnoise, marg_shotnoise=not rotation.with_momt, klim=kcutlim, nobs=nobs, fn='{}_validation.png'.format(base_fn))
        rotation.rotate(theory=theory, data=data, klim=kcutlim, covmatrix=covariance, shotnoise=shotnoise)  # to set up priors
        if rotation.with_momt: logger.info('mo priors: {}'.format(rotation.marg_prior_mo))

    if output is not None:
        rotation.save(output)
    if plot and data is not None:
        rotation.plot_rotated(data=data, shotnoise=shotnoise, fn='{}_power.png'.format(base_fn))
    wmatrix_rotated, covariance_rotated = rotation.rotate(covmatrix=covariance)[:2]

    if output_wmatrix is not None:
        wmatrix = wmatrix.deepcopy()
        wmatrix.value = wmatrix_rotated.T
        wmatrix = wmatrix.select_x(xoutlim=kcutlim)
        mask_kout = rotation._index_kout(kcutlim)
        attrs = {'covmatrix': rotation.covmatrix[np.ix_(mask_kout, mask_kout)], 'mmatrix': rotation.mmatrix[0][mask_kout, :]}
        if rotation.with_momt:
            attrs['mo'] = [mo[mask_kout] for mo in rotation.mmatrix[1]]
            attrs['marg_prior_mo'] = rotation.marg_prior_mo
        wmatrix.attrs['rotation'] = attrs
        wmatrix.save(output_wmatrix)
    return rotation


def postprocess_ricwmatrix(ric, wmatrix, wmatrix_cut=None, output_wmatrix=None):
    from desi_y1_files import WindowRIC
    if wmatrix_cut is None: wmatrix_cut = wmatrix
    wmatrix = wmatrix.load()
    wmatrix_cut = wmatrix_cut.load()
    ric = WindowRIC.load(ric)
    wmatrix = ric.export(wmatrix=wmatrix, wmatrix_cut=wmatrix_cut)
    if output_wmatrix is not None:
        output_wmatrix.save(wmatrix)


def compute_ricwmatrix(wmatrix, power_ric, power_noric, covariance=None, output=None):
    
    import os
    import numpy as np
    from pypower import PowerSpectrumStatistics
    from desi_y1_files import WindowRIC, is_file_sequence, is_path

    wmatrix = wmatrix.load()
    power_ric = [data.load() for data in power_ric]
    power_noric = [data.load() for data in power_noric]
    
    def get_meas(kinlim=(0., np.inf), koutlim=(0., np.inf), kinrebin=1, koutrebin=1):
        w = wmatrix[:len(wmatrix.xin[0]) // kinrebin * kinrebin:kinrebin, :len(wmatrix.xout[0]) // koutrebin * koutrebin:koutrebin].select_x(xinlim=kinlim, xoutlim=koutlim)

        def get_power(data, k=None):
            if isinstance(data, np.ndarray):
                return np.ravel(data)
            if k is None:
                data = data[:(data.shape[0] // koutrebin) * koutrebin:koutrebin].select(koutlim)(ell=ells, return_k=False, complex=False, remove_shotnoise=False)
            else:
                data = data(ell=ells, k=k, return_k=False, remove_shotnoise=False)
            return data.ravel()

        ells = tuple(proj.ell for proj in w.projsout)
        p_ric = [get_power(data) for data in power_ric]
        p_noric = [get_power(data) for data in power_noric]
        cov = None
        if covariance is not None:
            cov = covariance.load().view(xlim=koutlim, projs=[0, 2, 4]) / len(p_ric)
        return w, p_ric, p_noric, cov

    #kinlim = (0., 0.5)
    #koutlim = (0., 0.4)
    kinlim = (0., 0.1)
    koutlim = (0., 0.1)
    kinrebin = 2
    koutrebin = 1
    w, p_ric, p_noric, cov = get_meas(kinlim=kinlim, koutlim=koutlim, kinrebin=kinrebin, koutrebin=koutrebin)
    ric = WindowRIC(w, power_ric=p_ric, power_noric=p_noric, covmatrix=cov, attrs={'kinlim': kinlim, 'kinrebin': kinrebin, 'koutlim': koutlim, 'koutrebin': koutrebin})
    nobs = len(p_ric)
    covmatrix_scaled = ric.covmatrix / nobs

    if output is not None:
        base_fn = os.path.splitext(output)[0]
        base_fn = 'test'
    plot = output is not None
    if plot:
        ric.plot_wmatrix(fn='{}_wmatrix.png'.format(base_fn), klim=(0., 0.2), ric=False)
        #ric.plot_wmatrix(fn='{}_rickernel.png'.format(base_fn), klim=(0., 0.2), ric='kernel')
        nobs = len(power_ric)
        ric.plot_validation(covmatrix=covmatrix_scaled, klim=(0., 0.1), fn='{}_validation_noric.png'.format(base_fn))
        ric.plot_validation_gic(fn='{}_gic_noric.png'.format(base_fn))
        ric.plot_power(fn='{}_power_ric_noric.png'.format(base_fn))
    ric.fit()
    #ric = WindowRIC.load(output)
    #ric.set_power(p_noric, p_ric)
    if plot:
        ric.plot_wmatrix(fn='{}_ricwmatrix.png'.format(base_fn), klim=(0., 0.2))
        ric.plot_wmatrix(fn='{}_rickernel.png'.format(base_fn), klim=(0., 0.2), ric='kernel')
        ric.plot_validation(covmatrix=covmatrix_scaled, klim=(0., 0.1), fn='{}_validation.png'.format(base_fn))
        ric.plot_validation_gic(fn='{}_gic.png'.format(base_fn))
        #w, p_ric, p_noric, cov = get_meas(kinrebin=kinrebin)
        w, p_ric, p_noric, cov = get_meas(kinlim=kinlim, koutlim=koutlim, kinrebin=kinrebin, koutrebin=koutrebin)
        ric_full = WindowRIC(w, power_ric=p_ric, power_noric=p_noric, covmatrix=cov)
        ric_full.asmatrix = ric.asmatrix
        covmatrix_scaled = ric_full.covmatrix / nobs
        ric_full.plot_validation(covmatrix=covmatrix_scaled, klim=(0., 0.4), fn='{}_full_validation.png'.format(base_fn))
    if output is not None:
        ric.save(output)
    return ric


if __name__ == '__main__':

    todo = ['reconstruction', 'correlation', 'correlation_bitwise', 'correlation_recon', 'correlation_recon_jackknife', 'power', 'power_recon', 'rotate']
    todo = ['correlation_recon_jackknife']
    todo = ['test']
    todo = ['footprint']

    from desipipe import setup_logging
    from desipipe.file_manager import BaseFile

    setup_logging()

    if 'reconstruction' in todo:
        data = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_NGC_clustering.dat.fits', filetype='catalog')
        all_randoms = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_NGC_{iran:d}_clustering.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        output_data = BaseFile('_tests/LRG_NGC_clustering.IFTrecsym.dat.fits', filetype='catalog', options={'algorithm': 'IFFT', 'mode': 'recsym', 'smoothing_radius': 10., 'cellsize': 20., 'tracer': 'LRG'})
        all_output_randoms = [BaseFile('_tests/LRG_NGC_{iran:d}_clustering.IFTrecsym.ran.fits', filetype='catalog', options={'mode': 'recsym', 'iran': iran}) for iran in range(4)]
        compute_reconstruction(data, all_randoms, output_data, all_output_randoms)

    if 'correlation' in todo:
        data = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_NGC_clustering.dat.fits', filetype='catalog')
        all_randoms = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_NGC_{iran:d}_clustering.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        output = BaseFile('_tests/correlation.npy', options={'binning': 'lin', 'nran': 1, 'cut': ('rp', 2.5), 'zrange': (0.4, 0.6)})
        compute_correlation_function(data, all_randoms, output=output)

    if 'correlation_bitwise' in todo:
        data = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/edav1/sv3/LSScats/clustering/LRG_{region}_clustering.dat.fits', filetype='catalog', options={'region': region}) for region in ['N', 'S']]
        all_randoms = [[BaseFile('/global/cfs/cdirs/desi/survey/catalogs/edav1/sv3/LSScats/clustering/LRG_{region}_{iran:d}_clustering.ran.fits', filetype='catalog', options={'region': region, 'iran': iran}) for region in ['N', 'S']] for iran in range(4)]
        data_full = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/edav1/sv3/LSScats/full/LRG_full.dat.fits', filetype='catalog')
        all_randoms_full = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/edav1/sv3/LSScats/full/LRG_{iran:d}_full.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        output = BaseFile('_tests/correlation.npy', options={'binning': 'lin', 'nran': 2, 'cut': ('theta', 0.06), 'weighting': 'bitwise_angular', 'region': 'ALL', 'split': 20.})
        compute_correlation_function(data, all_randoms, data_full=data_full, all_randoms_full=all_randoms_full, output=output)

    if 'correlation_recon' in todo:
        data = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/recon_sm10/LRG_NGC_clustering.IFTrecsym.dat.fits', filetype='catalog')
        all_shifted = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/recon_sm10/LRG_NGC_{iran:d}_clustering.IFTrecsym.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        all_randoms = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_NGC_{iran:d}_clustering.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        output = BaseFile('_tests/correlation.npy', options={'binning': 'lin', 'nran': 1, 'cut': ('rp', 2.5)})
        compute_correlation_function(data, all_randoms, all_shifted=all_shifted, output=output)

    if 'correlation_recon_jackknife' in todo:
        data = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/recon_sm10/LRG_NGC_clustering.IFTrecsym.dat.fits', filetype='catalog')
        all_shifted = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/recon_sm10/LRG_NGC_{iran:d}_clustering.IFTrecsym.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        all_randoms = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_NGC_{iran:d}_clustering.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        output = BaseFile('_tests/correlation.npy', options={'binning': 'lin', 'nran': 1, 'cut': ('rp', 2.5), 'njack': 60, 'split': 20.})
        compute_correlation_function(data, all_randoms, all_shifted=all_shifted, output=output)

    if 'power' in todo:
        data = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_NGC_clustering.dat.fits', filetype='catalog')
        all_randoms = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_NGC_{iran:d}_clustering.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        output = BaseFile('_tests/power.npy', filetype='power', options={'cellsize': 60, 'cut': ('theta', 0.06)})
        output_wmatrix = BaseFile('_tests/wmatrix.npy', filetype='wmatrix')
        output_window = [BaseFile('_tests/window_{boxscale:.0f}.npy', options={'boxscale': boxscale}, filetype='wmatrix') for boxscale in [20., 5., 1.]]
        compute_power_spectrum(data, all_randoms, output=output, output_wmatrix=output_wmatrix, output_window=output_window, nthreads=32)

    if 'power_recon' in todo:
        data = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/recon_sm10/LRG_NGC_clustering.IFTrecsym.dat.fits', filetype='catalog')
        all_shifted = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/recon_sm10/LRG_NGC_{iran:d}_clustering.IFTrecsym.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        all_randoms = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_NGC_{iran:d}_clustering.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(4)]
        output = BaseFile('_tests/power.npy', filetype='power', options={'cellsize': 40})
        compute_power_spectrum(data, all_randoms, all_shifted=all_shifted, output=output)

    if 'footprint' in todo:

        def compute_zeff_nompi(*catalog_fns, cosmo=None, zrange=None):
            import fitsio
            import numpy as np
            if cosmo is None:
                from cosmoprimo.fiducial import DESI
                cosmo = DESI()

            def read(fn):
                catalog = fitsio.read(fn)
                z = catalog['Z']
                weights = catalog['WEIGHT'] * catalog['WEIGHT_FKP']
                mask = (z >= zrange[0]) & (z < zrange[1])
                return z[mask], weights[mask]

            z1, weights1 = [np.concatenate(tmp) for tmp in zip(*[read(fn) for fn in catalog_fns])]
            zstep = 0.01
            zbins = np.arange(zrange[0], zrange[1] + zstep / 2., zstep)
            dbins = cosmo.comoving_radial_distance(zbins)
            hist1 = np.histogram(z1, weights=weights1, density=False, bins=zbins)[0]
            zhist1 = np.histogram(z1, weights=z1 * weights1, density=False, bins=zbins)[0]
            z = zhist1 / hist1
            z[np.isnan(z)] = 0.
            dv = dbins[1:]**3 - dbins[:-1]**3
            # sum(dv * density^2 * z) / sum(dv * density^2), where density = hist1 / dv
            return np.sum(hist1**2 / dv * z) / np.sum(hist1**2 / dv)

        all_data = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_{region}_clustering.dat.fits', filetype='catalog', options={'region': region}) for region in ['NGC', 'SGC']]
        all_randoms = [[BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/LRG_{region}_{iran:d}_clustering.ran.fits', filetype='catalog', options={'region': region, 'iran': iran}) for region in ['NGC', 'SGC']] for iran in range(4)]
        for iregion, region in enumerate(['NGC', 'SGC', 'GCcomb']):
            zrange = [0.4, 0.6]
            footprint = BaseFile('_tests/footprint_LRG_{zrange[0]:.1f}_{zrange[1]:.1f}.npy', options={'tracer': 'LRG', 'region': region, 'zrange': zrange})
            footprint = get_footprint(all_data, all_randoms=all_randoms, output=footprint)
            if region != 'GCcomb':
                zeff = compute_zeff_nompi(*[file.filepath for file in [randoms[iregion] for randoms in all_randoms]], zrange=zrange)
                print(footprint.attrs['zeff'], zeff)

    if 'rotate' in todo:
        power = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v3/complete/2pt/mock0/pk/pkpoles_LRG_GCcomb_z0.8-1.1_default_FKP_lin_nran18_cellsize6_boxsize7000_thetacut0.05.npy', filetype='power', options={'cut': ('theta', 0.05)})
        wmatrix = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v3/complete/2pt/merged/pk/wmatrix_smooth_LRG_GCcomb_z0.8-1.1_default_FKP_lin_nran18_cellsize6_boxsize7000_thetacut0.05.npy', filetype='wmatrix', options={'cut': ('theta', 0.05)})
        #wmatrix = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v3/complete/2pt/merged/pk/wmatrix_smooth_ELG_LOPnotqso_GCcomb_z0.8-1.1_default_FKP_lin_nran18_cellsize6_boxsize9000_thetacut0.05.npy', filetype='wmatrix', options={'cut': ('theta', 0.05)})
        covariance = BaseFile('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v0.6/blinded/pk/covariances/cov_gaussian_pre_LRG_GCcomb_0.8_1.1_default_FKP_lin.txt', filetype='power_covariance')
        rotation = rotate_wmatrix(wmatrix, covariance, output='_tests/rotation.npy')
        ref = rotate_wmatrix_ref(power, wmatrix, covariance)
        import numpy as np
        diff = np.abs(ref.value.T - rotation.wmatrix).max()
        print(diff)

    if 'plot_rotate' in todo:
        from desi_y1_files import WindowRotation
        rotation = WindowRotation.load('_tests/rotation.npy')
        rotation.plot_wmatrix(fn='_tests/wmatrix.png')
        rotation.plot_covmatrix(fn='_tests/covmatrix.png')
        rotation.plot_compactness(fn='_tests/compactness.png')
        power = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v3/complete/2pt/mock0/pk/pkpoles_LRG_GCcomb_z0.8-1.1_default_FKP_lin_nran18_cellsize6_boxsize7000_thetacut0.05.npy', filetype='power', options={'cut': ('theta', 0.05)}).load()
        koutrebin, koutlim = 5, (0., 0.4)
        power = power[:len(power.k) // koutrebin * koutrebin:koutrebin].select(koutlim).power.ravel()
        rotation.plot_rotated(data=power, fn='_tests/rotated.png')
        theory = sum(BaseFile('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/desipipe/2pt/mock0_los-{los}/pk/pkpoles_LRG_z0.8000_lin_cellsize2_boxsize2000.npy'.format(los=los), filetype='power').load() for los in ['x', 'y', 'z'])
        theory = theory(k=rotation.kin[0]).ravel()
        rotation.plot_validation(data=power, theory=theory, fn='_tests/validation.png')

    if 'test' in todo:
        data = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/complete/2pt/merged2/LRG_NGC_0_clustering.ran.fits', filetype='catalog')
        all_randoms = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock{iran:d}/LRG_complete_NGC_{iran:d}_clustering.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(1, 10)]
        options = {'cellsize': 6., 'zrange': [0.8, 1.1], 'boxsize': 7000., 'weighting': 'default_FKP'}
        output = BaseFile('_tests/power_NGC_0_2.npy', filetype='power', options=options)
        compute_power_spectrum(data, all_randoms, output=output)
        """
        data = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock0/LRG_ffa_NGC_0_clustering.ran.fits', filetype='catalog')
        catalog = data.load()
        import numpy as np
        print(catalog.csize, catalog['RA'], np.unique(catalog['RA']).size)
        data2 = BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock1/LRG_ffa_NGC_0_clustering.ran.fits', filetype='catalog')
        catalog = data2.load()
        print(catalog.csize, catalog['RA'])
        all_randoms = [BaseFile('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock{iran:d}/LRG_complete_NGC_{iran:d}_clustering.ran.fits', filetype='catalog', options={'iran': iran}) for iran in range(1, 10)]
        options = {'cellsize': 6., 'zrange': [0.8, 1.1], 'boxsize': 7000., 'weighting': 'default_FKP'}
        output2 = BaseFile('_tests/power_NGC_mock0.npy', filetype='power', options=options)
        #compute_power_spectrum(data, all_randoms, output=output2)
        """
        power = output.load().select((0., 0.4, 0.01))
        #power2 = output2.load().select((0., 0.4, 0.01))
        ax = power.plot(fn='tmp.png')
        #power2.plot(ax=ax, fn='tmp.png')
        