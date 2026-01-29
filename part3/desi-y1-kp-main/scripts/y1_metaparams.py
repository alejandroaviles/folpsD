def compute_nrandoms(data, randoms, zrange=None, simple=False, with_randoms_split=False, target=1.):
    # target: increase in std
    import numpy as np
    from desi_y1_files.catalog_tools import get_clustering_positions_weights
    data_positions, data_weights = get_clustering_positions_weights(data.load(), zrange=zrange, weight_type='default_FKP')
    randoms_positions, randoms_weights = get_clustering_positions_weights(randoms.load(), zrange=zrange, weight_type='default_FKP')
    data_weights, randoms_weights = data_weights[-1], randoms_weights[-1]
    if simple:
        # alpha^2 < target
        nrandoms = (data_weights.csize / randoms_weights.csize) / target
    else:
        target_var = (1. + target)**2 - 1.
        var_DD = (data_weights**2).csum()**2
        alpha = data_weights.csum() / randoms_weights.csum()
        var_RR = (alpha**2 * randoms_weights[-1]**2).csum()**2
        var_DR = 2. * var_DD**0.5 * var_RR**0.5
        if with_randoms_split:
            var_DR_RR = 3. / 2. * var_DR  # eg. eq. 30 of https://arxiv.org/pdf/1905.01133.pdf
            nrandoms = var_DR_RR / (target_var * var_DD)
        else:
            delta = var_DR**2 + 4. * target_var * var_DD * var_RR
            nrandoms = (2. * var_RR) / (-var_DR + delta**0.5)
    return nrandoms, data_weights.csize / (nrandoms * randoms_weights.csize)


def compute_boxsize(randoms, zrange=None, boxpad=1.2):
    import numpy as np
    from desi_y1_files.catalog_tools import get_clustering_positions_weights
    randoms_positions = get_clustering_positions_weights(randoms.load(), zrange=zrange, weight_type='default_FKP', position_type='xyz')[0]
    return boxpad * max(np.max(randoms_positions, axis=1) - np.min(randoms_positions, axis=1))


if __name__ == '__main__':

    import logging
    from desipipe import Environment, setup_logging
    from desi_y1_files import get_data_file_manager

    logger = logging.getLogger('EstimateRandoms')

    setup_logging()

    version = 'v0.6'
    regions = ['NGC', 'SGC']
    tracers = ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']

    fm = get_data_file_manager()
    fm = fm.select(version=version, region=regions, tracer=tracers)

    for fi in fm.select(id='power_y1', weighting='default_FKP', cut=[None]).iter(intersection=False):
        options = fi.options
        data = fm.get(id='catalog_data_y1', **options, ignore=True)
        all_randoms = list(fm.select(id='catalog_randoms_y1', **options, ignore=True))
        ##nrandoms, alpha = compute_nrandoms(data, all_randoms[0], zrange=options['zrange'], with_randoms_split=True, target=0.03)
        nrandoms, alpha = compute_nrandoms(data, all_randoms[0], zrange=options['zrange'], simple=True, target=0.02)
        logger.info('For {} {} in {}, {:.1f} randoms files are enough ({:.0f}x data)'.format(options['tracer'], options['region'], options['zrange'], nrandoms, 1. / alpha))
        #boxsize = compute_boxsize(all_randoms[0], zrange=options['zrange'], boxpad=1.5)
        #logger.info('For {} {} in {}, boxsize of {:.0f} is enough'.format(options['tracer'], options['region'], options['zrange'], boxsize)