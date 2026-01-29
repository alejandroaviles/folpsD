import numpy as np
import healpy as hp


footprint = None
nside = 256


def load_footprint():
    global footprint
    from regressis import footprint
    footprint = footprint.DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False)


def split_region(region, tracer=''):
    if region == 'NGC':
        return ['N', 'SNGC']
    if region == 'SGC':
        if 'QSO' in tracer:
            return ['SSGCnoDES', 'DES']
        return ['SSGC']
    if region == 'N':
        return ['N']
    if region == 'S':
        if 'QSO' in tracer:
            return ['SnoDES', 'DES']
        return ['S']
    if region in [None, 'ALL', 'GCcomb']:
        return split_region('N', tracer=tracer) + split_region('S', tracer=tracer)
    raise ValueError('unknown region {}'.format(region))


def select_region(ra, dec, region=None):
    # print('select', region)
    import numpy as np
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    if footprint is None: load_footprint()
    north, south, des = footprint.get_imaging_surveys()
    mask_des = des[hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    raise ValueError('unknown region {}'.format(region))


def _format_bitweights(bitweights):
    if bitweights.ndim == 2: return list(bitweights.T)
    return [bitweights]


def sky_to_cartesian(rdd, degree=True):
    """
    Transform distance, RA, Dec into cartesian coordinates.

    Parameters
    ----------
    rdd : array of shape (3, N), list of 3 arrays
        Right ascension, declination and distance.

    degree : default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    Returns
    -------
    positions : list of 3 arrays
        Positions x, y, z in cartesian coordinates.
    """
    conversion = 1.
    if degree: conversion = np.pi / 180.
    ra, dec, dist = rdd
    cos_dec = np.cos(dec * conversion)
    x = dist * cos_dec * np.cos(ra * conversion)
    y = dist * cos_dec * np.sin(ra * conversion)
    z = dist * np.sin(dec * conversion)
    return [x, y, z]


def get_zsnap_from_z(tracer, z):
    """Return zsnapshot from redshift."""
    import numpy as np
    tracer = tracer.upper()[:3]
    zrange = {}
    if tracer == 'BGS':
        zrange[0.200] = (0.1, 0.4)
    elif tracer == 'LRG':
        zrange[0.500] = (0.4, 0.6)
        zrange[0.800] = (0.6, 1.1)
    elif tracer == 'ELG':
        zrange[0.950] = (0.8, 1.1)
        zrange[1.325] = (1.1, 1.6)
    elif tracer == 'QSO':
        zrange[1.400] = (0.8, 2.1)
    else:
        raise ValueError('unknown tracer {}'.format(tracer))
    z = np.array(z)
    for zsnap, zrange in zrange.items():
        if np.all((z >= zrange[0]) & (z <= zrange[1])):
            return zsnap
    raise ValueError('input z not found in any snapshot {}'.format(zsnap))

    
# Function returning positions, weights and effective redshift given input catalog and redshift range
def get_clustering_positions_weights(catalog, cosmo=None, zrange=None, weight_type='default', regions=None, position_type='rdd', return_z=False, tracer=None, option=None):
    import numpy as np
    
    weight_type = weight_type or ''
    option = option or ''

    if cosmo is None:
        from cosmoprimo.fiducial import DESI
        cosmo = DESI()
    z2d = cosmo.comoving_radial_distance
    from cosmoprimo.utils import DistanceToRedshift
    d2z = DistanceToRedshift(z2d)

    toret = []
    if isinstance(catalog, (tuple, list)): # list of catalogs, one for each region
        isscalar = False
        for cat in catalog:
            toret += get_clustering_positions_weights(cat, cosmo=cosmo, zrange=zrange, weight_type=weight_type, regions=regions, position_type=position_type, return_z=return_z, tracer=tracer, option=option)
    else:
        isscalar = not isinstance(regions, (tuple, list))
        if isscalar:
            regions = [regions]
        catalog.get(catalog.columns())  # fitsio much faster if all columns read at once, as long as there are not too many
        if 'WEIGHT_RESCALED' in catalog:
            catalog['WEIGHT'] = catalog['WEIGHT_RESCALED'] / catalog['WEIGHT_FKP']
        if 'OLD_RSDZ' in catalog:
            catalog['RSDZ'] = catalog['OLD_RSDZ']
        if 'rsd' in option and 'RSDZ' not in catalog:  # randoms
            option = ''
        if 'rsd' in option:
            catalog['Z'] = catalog['RSDZ']
        if zrange is not None:
            maskz = (catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])
        else:
            maskz = catalog.trues()
        if 'rsd-snapshot' in option:
            v = (catalog['RSDZ'] - catalog['TRUEZ']) / (1. + catalog['TRUEZ']) * 299792458. / 1000.
            zsnap = get_zsnap_from_z(tracer, zrange)
            a = 1. / (1. + zsnap)
            E = cosmo.efunc(zsnap)
            shift = v / (100. * a * E)
            dist = z2d(catalog['TRUEZ']) + shift
        elif 'rsd-no' in option:
            dist = z2d(catalog['TRUEZ'])
        else:
            dist = z2d(catalog['Z'])
        if 'mask' in option:
            maskz &= catalog['MASK']
            #print('MASK', maskz.sum() / maskz.size)
        for region in regions:
            mask = maskz & select_region(catalog['RA'], catalog['DEC'], region=region)
            #print('REGION', region, mask.sum() / mask.size)
            cat = catalog[mask]
            positions = [cat['RA'], cat['DEC'], dist[mask]]
            if position_type == 'xyz':
                positions = sky_to_cartesian(positions)
            indweights, bitweights = cat.ones(dtype='f8'), []
            if 'default' in weight_type:
                indweights *= cat['WEIGHT']
            if 'SYS' in weight_type:
                #weight_sys = {'AUTO': 'WEIGHT_SYS', 'BGS': 'WEIGHT_IMLIN', 'LRG': 'WEIGHT_IMLIN', 'ELG': 'WEIGHT_SN', 'QSO': 'WEIGHT_RF'}[tracer.upper()[:3]]
                weight_sys = 'WEIGHT_SYS'
                indweights /= cat[weight_sys]
                if 'SYS1' in weight_type:
                    pass
                elif 'SYSIMLIN' in weight_type:
                    indweights *= cat['WEIGHT_IMLIN']
                elif 'SYSSN' in weight_type:
                    indweights *= cat['WEIGHT_SN']
                elif 'SYSRF' in weight_type:
                    indweights *= cat['WEIGHT_RF']
                else:
                    raise ValueError('weight {} not known'.format(weight_type))
            if 'FKP' in weight_type:
                indweights *= cat.get('WEIGHT_FKP', 1.)
            if 'bitwise' in weight_type:
                if 'WEIGHT_COMP' in cat:
                    indweights /= cat['WEIGHT_COMP']
                else:
                    indweights *= cat['PROB_OBS']
                bitweights = _format_bitweights(cat['BITWEIGHTS'])
            weights = bitweights + [indweights]
            if return_z:
                toret.append((cat['Z'], positions, weights))
            else:
                toret.append((positions, weights))

    if isscalar:
        return toret[0]
    return toret


# Function returning positions, weights for full catalogs
def get_full_positions_weights(catalog, weight_type='default', fibered=False, regions=None, weight_attrs=None):

    import numpy as np
    from pycorr.twopoint_counter import get_inverse_probability_weight

    weight_attrs = weight_attrs or {}
    isscalar = not isinstance(regions, (tuple, list))
    if isscalar:
        regions = [regions]

    toret = []

    if isinstance(catalog, (tuple, list)):
        isscalar = False
        for cat in catalog:
            toret += get_full_positions_weights(cat, weight_type=weight_type, fibered=fibered, regions=regions, weight_attrs=weight_attrs)
    else:
        for region in regions:
            mask = select_region(catalog['RA'], catalog['DEC'], region=region)

            if fibered:
                mask &= catalog['LOCATION_ASSIGNED']

            cat = catalog[mask]
            positions = [cat['RA'], cat['DEC']]
            if fibered:
                if 'default' in weight_type or 'completeness' in weight_type:
                    weights = get_inverse_probability_weight(_format_bitweights(cat['BITWEIGHTS']), **weight_attrs)
                if 'bitwise' in weight_type:
                    weights = _format_bitweights(cat['BITWEIGHTS'])
            else:
                weights = [np.ones_like(positions[0])]
            toret.append((positions, weights))
    if isscalar:
        return toret[0]
    return toret


def concatenate_data_randoms(data_positions_weights, *randoms_positions_weights, weight_attrs=None, randoms_splits_size=None, concatenate=None, mpicomm=None):
    # data_positions_weights: list of (positions, weights) (for each region)
    # randoms_positions_weights: list of (positions, weights) (for each region)
    import numpy as np
    from pycorr import mpi
    if mpicomm is None:
        mpicomm = mpi.COMM_WORLD

    def _concatenate(array):
        if isinstance(array[0], (tuple, list)):
            array = [np.concatenate([arr[iarr] for arr in array], axis=0) if array[0][iarr] is not None else None for iarr in range(len(array[0]))]
        elif array is not None:
            array = np.concatenate(array)  # e.g. Z column
        return array

    if randoms_splits_size:
        
        if concatenate is None: concatenate = False

        def _map_arrays(func, arrays):
            toret = []
            for array in arrays:
                if isinstance(array, (tuple, list)): toret.append([func(arr) for arr in array])
                else: toret.append(func(array))
            return toret

        ndata = [mpicomm.allreduce(len(pw[-1][0])) for pw in data_positions_weights]  # weight
        nregions = len(randoms_positions_weights[0])
        assert len(ndata) == nregions, 'not the same number of regions in data = {:d} and randoms = {:d}'.format(len(ndata), nregions)
        # First, gather and concatenate
        randoms_positions_weights = [[_map_arrays(lambda array: mpi.gather(array, mpicomm=mpicomm, mpiroot=0), pw) for pw in rr] for rr in randoms_positions_weights]
        nrandoms, tmp_randoms_positions_weights = [], []  # number of regions
        for iregion in range(nregions):
            tmp = [_concatenate([rr[iregion][iarr] for rr in randoms_positions_weights]) if mpicomm.rank == 0 else None for iarr in range(len(randoms_positions_weights[0][iregion]))]
            tmp_randoms_positions_weights.append(tmp)
            nrandoms.append(len(tmp[-1][0] if mpicomm.rank == 0 else None))  # weight
        nrandoms = mpicomm.bcast(nrandoms, root=0)
        nsplits = max(min(nr // (nd * randoms_splits_size) for nd, nr in zip(ndata, nrandoms)), 1)
        randoms_positions_weights = [[None] * len(tmp_randoms_positions_weights) for isplit in range(nsplits)]
        # Then shuffle and split
        for iregion, pw in enumerate(tmp_randoms_positions_weights):
            pw_splits = [None] * len(pw)
            if mpicomm.rank == 0:
                rng = np.random.RandomState(seed=42)
                nr = ndata[iregion] * randoms_splits_size
                indices = rng.choice(nrandoms[iregion], nrandoms[iregion], replace=False)
                indices = [indices[isplit * nr:(isplit + 1) * nr] for isplit in range(nsplits)]
                pw_splits = [_map_arrays(lambda array: array[index], pw) for index in indices]                 
            for isplit in range(nsplits):
                randoms_positions_weights[isplit][iregion] = _map_arrays(lambda array: mpi.scatter(array, mpicomm=mpicomm, mpiroot=0), pw_splits[isplit])

    def normalize_data_randoms_weights(data_weights, randoms_weights, weight_attrs=None):
        # Renormalize randoms / data for each input catalogs
        # data_weights should be a list (for each N/S catalogs) of weights
        import inspect
        from pycorr.twopoint_counter import _format_weights, get_inverse_probability_weight
        if weight_attrs is None: weight_attrs = {}
        weight_attrs = {k: v for k, v in weight_attrs.items() if k in inspect.getargspec(get_inverse_probability_weight).args}

        def sum_weights(*weights):
            sum_weights, formatted_weights = [], []
            for weight in weights:
                weight, nbits = _format_weights(weight, copy=True)  # this will sort bitwise weights first, then single individual weight
                iip = (get_inverse_probability_weight(weight[:nbits], **weight_attrs) if nbits else 1.) * weight[nbits]
                sum_weights.append(mpicomm.allreduce(np.sum(iip)))
                formatted_weights.append(weight)
            return sum_weights, formatted_weights

        data_sum_weights, data_weights = sum_weights(*data_weights)
        randoms_sum_weights, randoms_weights = sum_weights(*randoms_weights)
        all_data_sum_weights, all_randoms_sum_weights = sum(data_sum_weights), sum(randoms_sum_weights)

        for icat, rw in enumerate(randoms_weights):
            if randoms_sum_weights[icat] != 0:
                factor = data_sum_weights[icat] / randoms_sum_weights[icat] * all_randoms_sum_weights / all_data_sum_weights
                rw[-1] *= factor
        return data_weights, randoms_weights

    data_positions, data_weights = tuple(_concatenate([pw[i] for pw in data_positions_weights]) for i in range(len(data_positions_weights[0]) - 1)), [pw[-1] for pw in data_positions_weights]
    if not randoms_positions_weights:
        data_weights = _concatenate(data_weights)
        return data_positions + (data_weights,)

    list_randoms_positions = tuple([_concatenate([pw[i] for pw in rr]) for rr in randoms_positions_weights] for i in range(len(randoms_positions_weights[0][0]) - 1))
    list_randoms_weights = [[pw[-1] for pw in rr] for rr in randoms_positions_weights]
    for iran, randoms_weights in enumerate(list_randoms_weights):
        list_randoms_weights[iran] = _concatenate(normalize_data_randoms_weights(data_weights, randoms_weights, weight_attrs=weight_attrs)[1])
    data_weights = _concatenate(data_weights)
    list_randoms_positions_weights = list_randoms_positions + (list_randoms_weights,)
    if concatenate:
        list_randoms_positions_weights = tuple(_concatenate(rr) for rr in list_randoms_positions_weights)
    return data_positions + (data_weights,), list_randoms_positions_weights