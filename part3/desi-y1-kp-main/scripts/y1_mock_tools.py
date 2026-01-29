def get_box_fn(tracer, zsnap, imock):
    return ['/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/CubicBox/{tracer}/z{zsnap:.3f}/AbacusSummit_base_c000_ph{imock:03d}/{tracer}_real_space.sub{isub:d}.fits.gz'.format(tracer=tracer.upper()[:3], zsnap=zsnap, imock=imock, isub=isub) for isub in range(64)]


def get_randoms_fn(tracer, iran):
    itracer = {'LRG': 1, 'ELG': 10, 'QSO': 3}[tracer.upper()[:3]]
    return '/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-allsky-{itracer}-{iran:d}.fits'.format(itracer=itracer, iran=iran)


def get_nbar(tracer, region='N'):
    from pathlib import Path
    import numpy as np
    dirname = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v3_1/aux_data/')
    tracer = tracer.upper()
    downsample = {'LRG': 0.22, 'ELG_LOP': 0., 'QSO': 0.6}[tracer[:7]]
    downsample = 0.  # to get final densities
    tracer = tracer[:3]
    region = region.upper()[:1]
    if tracer == 'ELG':
        z, nz, frac = np.loadtxt(dirname / 'NZ_{}_{}_v3.txt'.format(tracer, 'NORTH' if region == 'N' else 'SOUTH'), unpack=True, usecols=(0, 1, 2))
    else:
        frac = 1.
        z, nz = np.loadtxt(dirname / 'NZ_{}_v3.txt'.format(tracer), unpack=True, usecols=(0, 1))
    nz = nz * frac / (1. - downsample)
    dz = z[1] - z[0]
    z, nz = np.append(z, z[-1] + dz), np.append(nz, nz[-1])
    return z, nz


def get_P0(tracer):
    if tracer.startswith('BGS'): P0 = 7000
    if tracer.startswith('LRG'): P0 = 10000
    if tracer.startswith('ELG'): P0 = 4000
    if tracer.startswith('QSO'): P0 = 6000
    return P0


def make_cutsky_data(output, tracer=None, imock=None, hpmask=None, get_box_fn=get_box_fn, get_nbar=get_nbar, get_P0=get_P0):
    import logging
    import itertools
    import numpy as np
    import fitsio
    from cosmoprimo import constants
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.utils import DistanceToRedshift
    import mpytools as mpy
    from mockfactory import Catalog, cartesian_to_sky, TabulatedRadialMask, utils
    from desimodel import footprint
    from desi_y1_files import get_zsnap_from_z, select_region
    
    logger = logging.getLogger('make_cutsky')

    if tracer is None: tracer = output.options['tracer']
    if imock is None: imock = output.options['imock']

    cosmo = DESI()
    z2d = cosmo.comoving_radial_distance
    d2z = DistanceToRedshift(z2d)

    tiles = fitsio.read('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits')

    boxsize = np.array([2000.] * 3)
    boxcenter = boxsize / 2.
    regions = ['N', 'S']

    tab_z, tab_nz = {}, {}
    for region in regions:
        tab_z[region], tab_nz[region] = get_nbar(tracer, region=region)
    
    P0 = get_P0(tracer)
    if hpmask is not None: hpmask = np.load(hpmask, allow_pickle=True)[()]

    def make_data_zrange(zrange):

        zsnap = get_zsnap_from_z(tracer, zrange)
        fn = get_box_fn(tracer, zsnap, imock)
        drange = z2d(zrange)
        nrepeats = np.ceil((drange[1] - boxsize / 2) / boxsize).astype('i4')

        data = Catalog.read(fn, filetype='fits')
        mpicomm = data.mpicomm
        mean_nz = data.csize / boxsize.prod()
        data.get(data.columns())
        box_position = np.column_stack([data[name] for name in ['x', 'y', 'z']]) - boxcenter
        a = 1. / (1. + zsnap)
        E = cosmo.efunc(zsnap)
        box_velocity = np.column_stack([data[name] for name in ['vx', 'vy', 'vz']])

        shifts = [np.arange(-nrepeat, 1 + nrepeat, 1) for nrepeat in nrepeats]
        cats = []

        for ishift, shift in enumerate(itertools.product(*shifts)):
            cat = Catalog()
            if mpicomm.rank == 0:
                logger.info('Processing shift {} ({:d} / {:d}) of zrange {} (zsnap = {:.3f}).'.format(shift, ishift, np.prod(2 * nrepeats + 1), zrange, zsnap))
            box_position_shift = box_position + np.array(shift) * boxsize
            dist = np.sum(box_position_shift**2, axis=-1)**0.5
            mask = (dist >= drange[0] - 100.) & (dist <= drange[1] + 100.)
            box_position_shift, dist = box_position_shift[mask], dist[mask]
            truez = cat['TRUEZ'] = d2z(dist)
            los = box_position_shift / dist[:, None]
            rsd_proj = np.sum(box_velocity[mask] * los, axis=-1)
            dz = rsd_proj * (1 + truez) / (constants.c / 1e3)
            rsdz = cat['RSDZ'] = truez + dz
            box_position_shift += rsd_proj[:, None] * los / (100. * a * E)
            dist, cat['RA'], cat['DEC'] = cartesian_to_sky(box_position_shift)
            z = cat['Z'] = d2z(dist)
            mask = (z >= zrange[0]) & (z <= zrange[1])
            cat = cat[mask]
            mask = footprint.is_point_in_desi(tiles, cat['RA'], cat['DEC'])
            cat = cat[mask]
            #rng = mpy.random.MPIRandomState(size=cat.size, seed=42 + imock)
            for region in regions:
                mask = TabulatedRadialMask(tab_z[region], tab_nz[region] / mean_nz, zrange=zrange, norm=1., interp_order=1)
                mask = mask(cat['Z'], seed=42 + imock)
                #mask &= np.interp(cat['Z'], tab_z[region], tab_nz[region] / mean_nz, left=0., right=0.) > rng.uniform(0., 1.,)
                mask &= select_region(cat['RA'], cat['DEC'], region=region)
                tmpcat = cat[mask]
                tmpcat['NZ'] = np.interp(tmpcat['Z'], tab_z[region], tab_nz[region], left=0., right=0.)
                tmpcat['WEIGHT_FKP'] = 1. / (1. + tmpcat['NZ'] * P0)
                tmpcat['WEIGHT'] = tmpcat.ones(dtype='f8')
                #tmpcat['REGION'] = np.full(tmpcat.size, region)
                cats.append(tmpcat)
        cat = Catalog.concatenate(cats)
        if hpmask is not None:
            import healpy as hp
            cat['MASK'] = hpmask['mask'][hp.ang2pix(hpmask['nside'], cat['RA'], cat['DEC'], lonlat=True)]
        return cat

    zranges = {'LRG': [(0.4, 0.6), (0.6, 1.1)], 'ELG': [(0.8, 1.1), (1.1, 1.6)], 'QSO': [(0.8, 2.1)]}[tracer.upper()[:3]]
    Catalog.concatenate([make_data_zrange(zrange) for zrange in zranges]).write(output)


def make_cutsky_randoms(output, tracer=None, iran=None, hpmask=None, get_randoms_fn=get_randoms_fn, get_nbar=get_nbar, get_P0=get_P0):

    import numpy as np
    import fitsio
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.utils import DistanceToRedshift
    from mockfactory import Catalog, TabulatedRadialMask
    import mpytools as mpy
    from desimodel import footprint
    from desi_y1_files import select_region

    if tracer is None: tracer = output.options['tracer']
    if iran is None: iran = output.options['iran']
    
    cosmo = DESI()
    z2d = cosmo.comoving_radial_distance
    d2z = DistanceToRedshift(z2d)

    zrange = {'LRG': (0.4, 1.1), 'ELG': (0.8, 1.6), 'QSO': (0.8, 2.1)}[tracer.upper()[:3]]
    drange = z2d(zrange)
    randoms_fn = get_randoms_fn(tracer=tracer, iran=iran)

    randoms = Catalog.read(randoms_fn)
    mpicomm = randoms.mpicomm
    cat = randoms

    regions = ['N', 'S']
    tab_z, tab_nz = {}, {}
    for region in regions:
        tab_z[region], tab_nz[region] = get_nbar(tracer, region=region)
    
    def normalize_factor(tab_z, tab_nz):
        index = np.flatnonzero((tab_z >= zrange[0]) & (tab_z <= zrange[1]))
        index = np.concatenate([[max(index[0] - 1, 0)], index, [min(index[-1] + 1, len(tab_z) - 1)]])
        return np.max(tab_nz[index])

    nfactor = max(normalize_factor(tab_z[region], tab_nz[region]) for region in regions)
    #for region in regions:
    #    tab_nz[region] /= nfactor

    tiles = fitsio.read('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits')
    mask = footprint.is_point_in_desi(tiles, cat['RA'], cat['DEC'])
    cat = cat[mask]

    def get_z(tab_z, tab_nz, size):
        ffactor = 5
        rng = mpy.random.MPIRandomState(ffactor * size, seed=42 + iran)
        d = rng.uniform(drange[0]**3, drange[1]**3)**(1. / 3.)  # distribution of distances from a box
        z = d2z(d)
        z = z[np.interp(z, tab_z, tab_nz, left=0, right=0) >= rng.uniform(0., 1.)]
        assert len(z) >= size, len(z) / size
        return z[:size], mpicomm.allreduce(len(z)) / (ffactor * mpicomm.allreduce(size))  # z[:size] does not guarantee rank-invariance

    #def get_z(tab_z, tab_nz, size):
    #    mask = TabulatedRadialMask(tab_z, tab_nz, zrange=zrange, norm=1., interp_order=1)
    #    return mask.sample(size, distance=z2d, seed=42 + iran), mask.integral()

    P0 = get_P0(tracer)
    if hpmask is not None: hpmask = np.load(hpmask, allow_pickle=True)[()]

    cats, factors = [], []
    for region in regions:
        mask = select_region(cat['RA'], cat['DEC'], region=region)
        z, factor = get_z(tab_z[region], tab_nz[region] / nfactor, mask.sum())
        factors.append(factor)
        tmpcat = cat[mask]
        tmpcat['Z'] = z
        tmpcat['NZ'] = np.interp(tmpcat['Z'], tab_z[region], tab_nz[region], left=0., right=0.)
        tmpcat['WEIGHT_FKP'] = 1. / (1. + tmpcat['NZ'] * P0)
        tmpcat['WEIGHT'] = tmpcat.ones(dtype='f8')
        #tmpcat['REGION'] = np.full(tmpcat.size, region)
        cats.append(tmpcat)
    factors = np.array(factors) / np.max(factors)
    for i, cat in enumerate(cats):
        rng = mpy.random.MPIRandomState(cat.size, seed=42 + iran)
        mask = factors[i] >= rng.uniform(0., 1.)
        cats[i] = cat[mask]
    cat = Catalog.concatenate(cats)
    if hpmask is not None:
        import healpy as hp
        cat['MASK'] = hpmask['mask'][hp.ang2pix(hpmask['nside'], cat['RA'], cat['DEC'], lonlat=True)]
    cat.write(output)

"""
def randoms_with_data_z(data, randoms, output=None):
    import numpy as np
    from mockfactory import Catalog, TabulatedRadialMask
    import mpytools as mpy
    from desi_y1_files import select_region
    if not isinstance(data, Catalog):
        data = Catalog.read(data)
    data.get(data.columns())
    if not isinstance(randoms, Catalog):
        randoms = Catalog.read(randoms)
    randoms.get(randoms.columns())
    regions = ['N', 'S']
    ratio = data['WEIGHT'].csum() / randoms['WEIGHT'].csum()
    for region in regions:
        if data.mpicomm.rank == 0: data.log_info('Processing region {}.'.format(region))
        datatmp = data[['Z', 'WEIGHT', 'WEIGHT_FKP']][select_region(data['RA'], data['DEC'], region)].gather(mpiroot=None)
        mask_region = select_region(randoms['RA'], randoms['DEC'], region)
        rng = mpy.random.MPIRandomState(mask_region.sum(), seed=42)
        idx = rng.choice(np.arange(datatmp.csize))
        for name in datatmp:
            randoms[name][mask_region] = datatmp[name][idx]
        randoms['WEIGHT'][mask_region] *= datatmp['WEIGHT'].csum() / randoms['WEIGHT'][mask_region].csum() / ratio
    if output is not None:
        randoms.write(output)
    if data.mpicomm.rank == 0: data.log_info('Written.')
    return randoms
"""

def randoms_with_data_z(list_data, list_randoms, list_output=None, seed=None):
    import numpy as np
    from mockfactory import Catalog
    import mpytools as mpy
    from desi_y1_files import select_region, is_file_sequence
    if not is_file_sequence(list_data): list_data = [list_data]
    if not is_file_sequence(list_randoms): list_randoms = [list_randoms]
    if list_output is not None and not is_file_sequence(list_output): list_output = [list_output]
    list_data = [Catalog.read(dd) if not isinstance(dd, Catalog) else dd for dd in list_data]
    for dd in list_data: dd.get(dd.columns())
    list_randoms = [Catalog.read(rr) if not isinstance(rr, Catalog) else rr for rr in list_randoms]
    for rr in list_randoms: rr.get(rr.columns())
    ratio = sum(dd['WEIGHT'].csum() for dd in list_data) / sum(rr['WEIGHT'].csum() for rr in list_randoms)
    for dd, rr in zip(list_data, list_randoms): rr['WEIGHT'] *= dd['WEIGHT'].csum() / rr['WEIGHT'].csum() / ratio
    regions = ['N', 'S']
    ratio = sum(dd['WEIGHT'].csum() for dd in list_data) / sum(rr['WEIGHT'].csum() for rr in list_randoms)
    for region in regions:
        if list_data[0].mpicomm.rank == 0: list_data[0].log_info('Processing region {}.'.format(region))
        datatmp = Catalog.concatenate([dd[['Z', 'WEIGHT', 'WEIGHT_FKP']][select_region(dd['RA'], dd['DEC'], region)] for dd in list_data]).gather(mpiroot=None)
        for data, randoms in zip(list_data, list_randoms):
            mask_region_randoms = select_region(randoms['RA'], randoms['DEC'], region)
            rng = mpy.random.MPIRandomState(mask_region_randoms.sum(), seed=seed)
            idx = rng.choice(np.arange(datatmp.csize))
            for name in datatmp:
                randoms[name][mask_region_randoms] = datatmp[name][idx]
            if mask_region_randoms.csum():
                randoms['WEIGHT'][mask_region_randoms] *= data['WEIGHT'][select_region(data['RA'], data['DEC'], region)].csum() / randoms['WEIGHT'][mask_region_randoms].csum() / ratio
    if list_output is not None:
        for randoms, output in zip(list_randoms, list_output):
            randoms.write(output)
    if list_data[0].mpicomm.rank == 0: list_data[0].log_info('Written.')
    return list_randoms


def make_mask_healpix(randoms, output, nside=4096):  # 4096 ~ 0.014 deg
    import os
    import numpy as np
    from mockfactory import Catalog, utils
    import mpytools as mpy
    import healpy as hp
    randoms = Catalog.read(randoms)
    mpicomm = randoms.mpicomm
    hpindex = hp.ang2pix(nside, randoms['RA'], randoms['DEC'], lonlat=True)  # ra, dec
    #counts, counts = np.unique(hpindex, return_counts=True)
    #print(np.mean(counts), np.std(counts))
    hpindex = mpy.gather(np.unique(hpindex), mpicomm=mpicomm, mpiroot=0)
    if mpicomm.rank == 0:
        mask = np.zeros(hp.nside2npix(nside), dtype='?')
        mask[hpindex] = True
        utils.mkdir(os.path.dirname(output))
        np.save(output, {'mask': mask, 'nside': nside})


def plot(data_fn, randoms_fn, fn='tmp.png'):
    import numpy as np
    from matplotlib import pyplot as plt
    from mockfactory import Catalog
    from desi_y1_files import select_region
    fig, lax = plt.subplots(1, 2, figsize=(10, 5), squeeze=True)
    ax = lax[0]
    data = Catalog.read(data_fn)
    randoms = Catalog.read(randoms_fn)
    zrange = (1.1, 1.6)
    mask_data = (data['Z'] > zrange[0]) & (data['Z'] < zrange[1])
    mask_randoms = (randoms['Z'] > zrange[0]) & (randoms['Z'] < zrange[1])
    data, randoms = data[mask_data], randoms[mask_randoms]
    size_data_n = select_region(data['RA'], data['DEC'], region='N').sum()
    size_data_s = select_region(data['RA'], data['DEC'], region='S').sum()
    size_randoms_n = select_region(randoms['RA'], randoms['DEC'], region='N').sum()
    size_randoms_s = select_region(randoms['RA'], randoms['DEC'], region='S').sum()
    print(size_randoms_n / size_data_n, size_randoms_s / size_data_s)
    print(randoms['MASK'].csum() / randoms.csize, data['MASK'].csum() / data.csize)
    randoms = randoms[np.random.uniform(0., 1., randoms.size) < 0.1]
    mask_data = data['MASK']
    mask_randoms = randoms['MASK']
    #mask = select_region(randoms['RA'], randoms['DEC'], region='N')
    #ax.scatter(randoms['RA'][mask], randoms['DEC'][mask], color='C1')
    ax.scatter(data['RA'][mask_data], data['DEC'][mask_data], marker='.', s=1, color='C0')
    ax.scatter(randoms['RA'][mask_randoms], randoms['DEC'][mask_randoms], marker='.', s=1, color='C1')
    ax = lax[1]
    bins = 100
    ax.hist(data['Z'][mask_data], bins=bins, histtype='step', color='C0', label='data', density=True)
    ax.hist(randoms['Z'][mask_randoms], bins=bins, histtype='step', color='C1', label='randoms', density=True)
    ax.legend()
    plt.savefig(fn)
    plt.close()


if __name__ == '__main__':

    import os
    from mockfactory import setup_logging

    setup_logging()

    tracer = 'ELG_LOP'
    todo = ['test', 'plot'][1:]
    todo = ['test_RR2']
    dirname = os.path.join(os.getenv('SCRATCH'), 'tests_cutsky')
    output_data_test = os.path.join(dirname, 'data_{}.fits'.format(tracer))
    output_randoms_test = os.path.join(dirname, 'randoms_{}.fits'.format(tracer))
    output_shuffled_test = os.path.join(dirname, 'shuffled_{}.fits'.format(tracer))
    output_hpmap = os.path.join(dirname, 'hpmap_{}.npy'.format(tracer))

    if 'test' in todo:
        #make_mask_healpix(['/global/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock0/{}_complete_{}_{:d}_clustering.ran.fits'.format(tracer, region, iran) for region in ['NGC', 'SGC'] for iran in range(18)], output_hpmap)
        #make_cutsky_data(output_data_test, tracer=tracer, hpmask=output_hpmap, imock=0)
        make_cutsky_randoms(output_randoms_test, tracer=tracer, hpmask=output_hpmap, iran=0)
        #randoms_with_data_z(output_data_test, output_randoms_test, output_shuffled_test)

    if 'plot' in todo:
        #plot(output_data_test, output_randoms_test, fn='check.png')
        output_data = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/raw/catalogs/mock0/ELG_LOP_raw_ALL_clustering.dat.fits'
        output_randoms = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/raw/catalogs/randoms/ELG_LOP_raw_ALL_0_clustering.ran.fits'
        #plot(output_data, output_randoms, fn='check.png')
        for i in [0]:
            fn = '/global/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/raw/2pt/mock{:d}/pk/pkpoles_ELG_LOP_NGC_z1.1-1.6_mask_default_FKP_lin_nran18_cellsize6_boxsize9000.npy'.format(i)
            from pypower import PowerSpectrumMultipoles
            poles = PowerSpectrumMultipoles.load(fn)
            poles.plot(fn='check_{:d}.png'.format(i))
    
    if 'test2' in todo:
        bins = 100
        tracer = 'ELG_LOP'
        for imock in [1]:
            import numpy as np
            from matplotlib import pyplot as plt
            from mockfactory import Catalog
            """
            data = Catalog.read('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock{:d}/{}_complete_SGC_clustering.dat.fits'.format(imock, tracer))
            randoms = Catalog.read(['/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock{:d}/{}_complete_SGC_{:d}_clustering.ran.fits'.format(imock, tracer, iran) for iran in range(18)])
            randoms2 = Catalog.read(['/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock{:d}/{}_complete_SGC_{:d}_clustering.ran.fits'.format(imock + 1, tracer, iran) for iran in range(18)])
            """
            data = Catalog.read('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/raw/catalogs/mock{:d}/{}_raw_ALL_clustering.dat.fits'.format(imock, tracer))
            randoms = Catalog.read(['/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/raw/catalogs/mock{:d}/{}_raw_ALL_{:d}_clustering.ran.fits'.format(imock, tracer, iran) for iran in range(8)])
            data[data.columns()], randoms[randoms.columns()]
            from desi_y1_files import select_region
            region = 'NGC'
            mask_data = select_region(data['RA'], data['DEC'], region=region)
            mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
            ax = plt.gca()
            ax.hist(data['Z'][mask_data], weights=data['WEIGHT'][mask_data], bins=bins, histtype='step', color='C0', label='data NGC', density=True)
            ax.hist(randoms['Z'][mask_randoms], weights=randoms['WEIGHT'][mask_randoms], bins=bins, histtype='step', color='C1', label='randoms NGC', density=True)
            region = 'SGC'
            mask_data = select_region(data['RA'], data['DEC'], region=region)
            mask_randoms = select_region(randoms['RA'], randoms['DEC'], region=region)
            ax.hist(data['Z'][mask_data], weights=data['WEIGHT'][mask_data], bins=bins, histtype='step', color='C0', label='data SGC', density=True)
            ax.hist(randoms['Z'][mask_randoms], weights=randoms['WEIGHT'][mask_randoms], bins=bins, histtype='step', color='C1', label='randoms SGC', density=True)
            #ax.hist(randoms['Z2'], weights=randoms['W2'], bins=bins, histtype='step', color='C2', label='randoms new shuffling', density=True)
            #ax.hist(randoms2['Z'], weights=randoms2['WEIGHT'] * randoms2['WEIGHT_FKP'], bins=bins, histtype='step', color='C3', label='randoms 2', density=True)
            ax.legend()
            ax.set_xlabel('$z$')
            plt.savefig('tmp_{:d}.png'.format(imock))
            plt.close()

    if 'test_RR' in todo:
        import numpy as np
        from mockfactory import Catalog
        from cosmoprimo.fiducial import DESI
        from pycorr import TwoPointCorrelationFunction
        if False:
            fiducial = DESI()
            zrange = (1.1, 1.6)
            #catalogs = [Catalog.read('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock1/ELG_LOP_complete_NGC_{:d}_clustering.ran.fits'.format(iran)) for iran in range(10)]
            #catalogs = [Catalog.read('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.3/ELG_LOPnotqso_NGC_{:d}_clustering.ran.fits'.format(iran)) for iran in range(10)]
            #catalogs = [Catalog.read('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/raw/catalogs/mock1/ELG_LOP_raw_ALL_{:d}_clustering.ran.fits'.format(iran)) for iran in range(8)]
            catalogs = [Catalog.read('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1_complete/raw/catalogs/mock1/ELG_LOP_raw_NGC_{:d}_clustering.ran.fits'.format(iran)) for iran in range(8)]

            def get_clustering_positions_weights(catalog, zrange):
                mask = (catalog['Z'] > zrange[0]) & (catalog['Z'] < zrange[1])
                mask_ngc = (catalog['RA'] > 100 - catalog['DEC'])
                mask_ngc &= (catalog['RA'] < 280 + catalog['DEC'])
                mask &= mask_ngc
                #print(catalog['Z'].dtype)
                return [catalog['RA'][mask], catalog['DEC'][mask], fiducial.comoving_radial_distance(catalog['Z'][mask])], catalog['WEIGHT'][mask]
            
            pw = [get_clustering_positions_weights(catalog, zrange=zrange) for catalog in catalogs]

            edges = (np.linspace(0., 30., 31), np.linspace(-1., 1., 201))
            result = sum(TwoPointCorrelationFunction(mode='smu', edges=edges,
                                                     data_positions1=pw[i][0], data_weights1=pw[i][1],
                                                     randoms_positions1=pw[i + 1][0], randoms_weights1=pw[i + 1][1],
                                                     position_type='rdd', dtype='f8') for i in range(2))
            result.save('counts.npy')
        if True:
            isbin = 28
            from matplotlib import pyplot as plt
            result = TwoPointCorrelationFunction.load('counts.npy')
            print(result.D1R2.dtype)
            ax = plt.gca()
            s, mu = result.sepavg(axis=0), result.sepavg(axis=1)
            ax.plot(mu, result.D1D2.wcounts[isbin], label='randoms 1 - randoms 1')
            ax.plot(mu, result.D1R2.wcounts[isbin], label='randoms 1 - randoms 2')
            ax.plot(mu, result.R1R2.wcounts[isbin], label='randoms 2 - randoms 2')
            ax.set_xlabel('$\mu$')
            ax.legend()
            plt.savefig('tmp.png')
            plt.close(plt.gcf())

    if 'test_RR2' in todo:
        import numpy as np
        from mockfactory import Catalog
        from cosmoprimo.fiducial import DESI
        from pycorr import TwoPointCorrelationFunction
        if True:
            fiducial = DESI()
            zrange = (1.1, 1.6)
            data = Catalog.read('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/raw/catalogs/mock1/ELG_LOP_raw_ALL_clustering.dat.fits')
            #data = Catalog.read(['/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock1/ELG_LOP_complete_NGC_clustering.dat.fits', '/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock1/ELG_LOP_complete_SGC_clustering.dat.fits'])
            #randoms = [Catalog.read('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/desipipe/v4_1/raw/catalogs/mock1/ELG_LOP_raw_ALL_{:d}_clustering.ran.fits'.format(iran)) for iran in range(4)]
            randoms = [Catalog.read(['/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock1/ELG_LOP_complete_NGC_{:d}_clustering.ran.fits'.format(iran), '/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_1/mock1/ELG_LOP_complete_SGC_{:d}_clustering.ran.fits'.format(iran)]) for iran in range(4)]
            #data['Z'] = data['Z'].astype('f4')

            randoms = [randoms_with_data_z(data, randoms, seed=iran + 42)[0] for iran, randoms in enumerate(randoms)]
            
            def get_clustering_positions_weights(catalog, zrange):
                mask = (catalog['Z'] > zrange[0]) & (catalog['Z'] < zrange[1])
                mask_ngc = (catalog['RA'] > 100 - catalog['DEC'])
                mask_ngc &= (catalog['RA'] < 280 + catalog['DEC'])
                mask &= mask_ngc
                #print(catalog['Z'].dtype)
                return [catalog['RA'][mask], catalog['DEC'][mask], fiducial.comoving_radial_distance(catalog['Z'][mask])], catalog['WEIGHT'][mask]
            
            pw = [get_clustering_positions_weights(catalog, zrange=zrange) for catalog in randoms]

            edges = (np.linspace(0., 30., 31), np.linspace(-1., 1., 201))
            result = sum(TwoPointCorrelationFunction(mode='smu', edges=edges,
                                                     data_positions1=pw[i][0], data_weights1=pw[i][1],
                                                     randoms_positions1=pw[i + 1][0], randoms_weights1=pw[i + 1][1],
                                                     position_type='rdd', dtype='f8') for i in range(1))
            result.save('counts.npy')
        if True:
            isbin = 28
            from matplotlib import pyplot as plt
            result = TwoPointCorrelationFunction.load('counts.npy')
            ax = plt.gca()
            s, mu = result.sepavg(axis=0), result.sepavg(axis=1)
            ax.plot(mu, result.D1D2.wcounts[isbin], label='randoms 1 - randoms 1')
            ax.plot(mu, result.D1R2.wcounts[isbin], label='randoms 1 - randoms 2')
            ax.plot(mu, result.R1R2.wcounts[isbin], label='randoms 2 - randoms 2')
            ax.set_xlabel('$\mu$')
            ax.legend()
            plt.savefig('tmp.png')
            plt.close(plt.gcf())
            