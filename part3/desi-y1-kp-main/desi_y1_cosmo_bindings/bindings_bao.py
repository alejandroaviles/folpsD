import os
import shutil
from pathlib import Path


def data_dir(version):
    if version and not version.startswith('_'):
        version = '_' + version
    return Path(__file__).parent / ('bao_data' + version)


def get_tracer_label(tracer):
    return tracer.split('_')[0].replace('+', 'plus')


def dataset_fn(tracer, zrange, version=''):
    return data_dir(version) / f'gaussian_bao_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'


def DESICompressedBAOLikelihood(tracers=None, cosmo=None, version=''):
    from desilike import LikelihoodFisher
    from desilike.observables.galaxy_clustering import BAOCompressionObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood

    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'LRG+ELG_LOPnotqso': [(0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'QSO': [(0.8, 2.1)], 'Lya': [(1.8, 4.2)]}  # keep all samples for the z{:d} numbering
    if tracers:
        tracers = [tracer.lower() for tracer in tracers]

    if cosmo is None:
        from desilike.theories import Cosmoprimo
        cosmo = Cosmoprimo(fiducial='DESI')
        cosmo.init.params = {'Omega_m': {'prior': {'limits': [0.1, 0.9]}, 'ref': {'dist': 'norm', 'loc': 0.3, 'scale': 0.001}, 'latex': '\Omega_m'},
                            'omega_b': {'prior': {'dist': 'norm', 'loc': 0.02236, 'scale': 0.0005}, 'latex': '\omega_b'},
                            'H0':  {'prior': {'limits': [20., 100.]}, 'ref': {'dist': 'norm', 'loc': 70., 'scale': 1.}, 'latex': 'H_{0}'}}

    likelihoods = []
    for tracer, zranges in list_zrange.items():
        for iz, zrange in enumerate(zranges):
            tracer_label = get_tracer_label(tracer).lower()
            namespace = '{tracer}_z{iz}'.format(tracer=tracer_label, iz=iz)
            if tracers is not None and 'lrg_z2' in tracers:
                pass
            elif (tracer_label, zrange) == ('lrg', (0.8, 1.1)):
                continue
            if tracers is not None and 'elg_z0' in tracers:
                pass
            elif (tracer_label, zrange) == ('elg', (0.8, 1.1)):
                continue
            if tracers is not None and namespace not in tracers: continue
            fisher = LikelihoodFisher.load(dataset_fn(tracer, zrange, version=version))
            observable = BAOCompressionObservable(data=fisher, covariance=fisher, cosmo=cosmo, quantities=fisher.params(), fiducial='DESI', z=fisher.attrs['zeff'])
            likelihood = ObservablesGaussianLikelihood(observables=observable, name=f'desi_bao_{namespace}')
            likelihoods.append(likelihood)
    if not likelihoods:
        raise ValueError('provide at least one tracer in {}'.format(tracers))
    if len(likelihoods) > 1:
        likelihood = sum(likelihoods)
        namespace = 'all' if tracers is None else '_'.join(tracers)
        likelihood.init.update(name=f'desi_bao_{namespace}')
    else:
        likelihood = likelihoods[0]
    return likelihood


def SNLikelihood(name='pantheon', **kwargs):
    from desilike.likelihoods.supernovae import PantheonSNLikelihood, PantheonPlusSNLikelihood, PantheonPlusSHOESSNLikelihood, Union3SNLikelihood, DESY5SNLikelihood
    Likelihood = {'pantheon': PantheonSNLikelihood, 'pantheonplus': PantheonPlusSNLikelihood, 'pantheonplusshoes': PantheonPlusSHOESSNLikelihood, 'union3': Union3SNLikelihood, 'desy5': DESY5SNLikelihood}[name]
    likelihood = Likelihood(**kwargs)
    for param in likelihood.init.params.select(basename=['Mb', 'dM']):
        param.update(prior=None, derived='.prec')
    return likelihood


def export_compressed_bao_data(fmt='cobaya', version='', public=False, mock=False):
    import numpy as np
    from scipy import linalg
    from cosmoprimo.fiducial import DESI
    from cosmoprimo import constants
    from desilike import LikelihoodFisher, utils
    
    if version and not version.startswith('_'): version = '_' + version

    def get_fid(z):
        fiducial = DESI()
        DM_over_rd_fid = fiducial.comoving_angular_distance(z) / fiducial.rs_drag
        DH_over_rd_fid = (constants.c / 1e3) / (100. * fiducial.efunc(z)) / fiducial.rs_drag
        DV_over_rd_fid = DM_over_rd_fid**(2. / 3.) * DH_over_rd_fid**(1. / 3.) * z**(1. / 3.)
        FAP_fid = DM_over_rd_fid / DH_over_rd_fid
        return DM_over_rd_fid, DH_over_rd_fid, DV_over_rd_fid, FAP_fid

    list_zrange = {'BGS_BRIGHT-21.5': [(0.1, 0.4)], 'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'LRG+ELG_LOPnotqso': [(0.8, 1.1)], 'ELG_LOPnotqso': [(0.8, 1.1), (1.1, 1.6)], 'QSO': [(0.8, 2.1)], 'Lya': [(1.8, 4.2)]}  # keep all samples for the z{:d} numbering
    #list_zrange = {}
    if fmt == 'cobaya':
        config_template = """# DESI BAO results for {tracer_zrange}
path: null
measurements_file: {mean_fn}
cov_file: {cov_fn}
# Fiducial sound horizon with which data have been stored
rs_fid: 1  # Mpc
# Aliases for automatic covariance matrix
aliases: [BAO]
# Speed in evaluations/second
speed: 2000"""
        py_template = '''from cobaya.likelihoods.base_classes import BAO


class {name}(BAO):
    r"""
    DESI BAO likelihood for {tracer_zrange}.
    """
'''
        mean_txt_header = '# [z] [value at z] [quantity]'
        cobaya_data_basename = Path('bao_data' + ('_mock_fiducial' if mock else version))
        cobaya_dir = Path(__file__).parent / ('cobaya_public_likelihoods' if public else 'cobaya_likelihoods')
        cobaya_data_dir = cobaya_dir / cobaya_data_basename
        cobaya_likelihood_dir = cobaya_dir / ('bao_likelihoods' + ('_mock_fiducial' if mock else version.replace('.', '_')))
        utils.mkdir(cobaya_data_dir)
        utils.mkdir(cobaya_likelihood_dir)
        with open(cobaya_dir / '__init__.py', 'w') as file: pass
        with open(cobaya_likelihood_dir / '__init__.py', 'w') as file: pass
    
        basename = 'desi_2024' if public else 'desi'
        izoffset = 1 if public else 0
    
        def write(tracer, mean_txt=None, cov=None, iz=None, zrange=None, tracer_zrange_txt=None, basename=basename):
            tracer_label = get_tracer_label(tracer)
            base_data_fn = str(cobaya_data_basename / '{basename}_gaussian_bao_{tracer}_GCcomb'.format(basename=basename, tracer=tracer))
            if zrange is not None:
                assert iz is not None
                base_data_fn += '_z{zrange[0]:.1f}-{zrange[1]:.1f}'.format(zrange=zrange)
            base_like_name = '{basename}_bao_{tracer}'.format(basename=basename, tracer=tracer_label.lower())
            if iz is not None:
                assert zrange is not None
                base_like_name += '_z{:d}'.format(iz + izoffset)
            base_like_fn = str(cobaya_likelihood_dir / base_like_name)
            mean_fn, cov_fn, config_fn, py_fn = base_data_fn + '_mean.txt', base_data_fn + '_cov.txt', base_like_fn + '.yaml', base_like_fn + '.py'
            if tracer_zrange_txt is None:
                tracer_zrange_txt = tracer
                if zrange is not None:
                    tracer_zrange_txt = '{tracer} in {zrange[0]:.1f} < z < {zrange[1]:.1f}'.format(tracer=tracer, zrange=zrange)
            with open(config_fn, 'w') as file:
                file.write(config_template.format(tracer_zrange=tracer_zrange_txt, mean_fn=mean_fn, cov_fn=cov_fn))
            with open(py_fn, 'w') as file:
                file.write(py_template.format(name=base_like_name, tracer_zrange=tracer_zrange_txt))
            mean_fn, cov_fn = os.path.join(cobaya_dir, mean_fn), os.path.join(cobaya_dir, cov_fn)
            if isinstance(mean_txt, (str, Path)):  # filename
                shutil.copyfile(mean_txt, mean_fn)
            else:
                with open(mean_fn, 'w') as file:
                    file.write(mean_txt_header + '\n')
                    for mean in mean_txt: file.write(mean + '\n')
            if isinstance(cov, (str, Path)):  # filename
                shutil.copyfile(cov, cov_fn)
            else:
                np.savetxt(cov_fn, cov, fmt='%.8e')
        mean_template = '{:.3f} {:.8f} {}' if public else '{:.8f} {:.8f} {}'
        all_mean_txt, all_cov = [], []
        for tracer, zranges in list_zrange.items():
            tracer_mean_txt, tracer_cov = [], []
            for iz, zrange in enumerate(zranges):
                # Fist loop on each tracer, z-range
                tracer_label = get_tracer_label(tracer)
                fisher = LikelihoodFisher.load(dataset_fn(tracer, zrange, version=version))
                z = fisher.attrs['zeff']
                DM_over_rd_fid, DH_over_rd_fid, DV_over_rd_fid, FAP_fid = get_fid(z)
                # Remove dependency in fiducial cosmology
                if 'qiso' in fisher.params() and 'qap' in fisher.params():
                    mean = fisher.mean(['qiso', 'qap'])
                    if mock: mean[...] = 1.
                    cov = fisher.covariance(['qiso', 'qap'])
                    fid = np.array([DV_over_rd_fid, FAP_fid])  
                    mean = np.array([fid[0] * mean[0], fid[1] / mean[1]]) # F_AP ~ DM / DH ~ 1 / qap
                    jac = np.diag([DV_over_rd_fid, -FAP_fid])
                    label = ['DV_over_rs', 'F_AP']
                elif 'qiso' in fisher.params():
                    mean = fisher.mean(['qiso'])
                    if mock: mean[...] = 1.
                    cov = fisher.covariance(['qiso'])
                    fid = np.array([DV_over_rd_fid])
                    mean *= fid
                    jac = np.diag(fid)
                    label = ['DV_over_rs']
                else:  # qpar, qper
                    mean = fisher.mean(['qper', 'qpar'])
                    if mock: mean[...] = 1.
                    cov = fisher.covariance(['qper', 'qpar'])
                    fid = np.array([DM_over_rd_fid, DH_over_rd_fid])
                    mean *= fid
                    jac = np.diag(fid)
                    label = ['DM_over_rs', 'DH_over_rs']
                mean_txt = [mean_template.format(z, mean, label) for mean, label in zip(mean, label)]
                cov = jac.T.dot(cov).dot(jac)
                if 'lya' not in tracer.lower():
                    write(tracer, mean_txt=mean_txt, cov=cov, iz=iz, zrange=zrange)
                if (tracer_label.lower(), zrange) in [('lrg', (0.8, 1.1)), ('elg', (0.8, 1.1))]: continue
                tracer_mean_txt += mean_txt
                tracer_cov.append(cov)
            # Concatenate all z-ranges for one tracer
            cov = linalg.block_diag(*tracer_cov)
            if not public: write(tracer, mean_txt=tracer_mean_txt, cov=cov, iz=None, zrange=None)
            all_mean_txt += tracer_mean_txt
            all_cov += tracer_cov

        # All combined
        cov = linalg.block_diag(*all_cov)
        write('ALL', mean_txt=all_mean_txt, cov=cov, iz=None, zrange=None, tracer_zrange_txt='all tracers')
        
        # DESI-only Lya measurement
        dirname = Path('/global/cfs/cdirs/desicollab/science/lya/y1-kp6/iron-final-bao/final-result/')
        mean_txt = dirname / 'DESI-Y1-Lya.dat'
        if mock:
            z, center = np.loadtxt(mean_txt, comments='#', usecols=[0, 1], unpack=True)
            z = z.flat[0]
            assert np.allclose(z, 2.33)
            assert center[0] < center[1]  # hacky way to check order: DH, DM
            DM_over_rd_fid, DH_over_rd_fid, DV_over_rd_fid, FAP_fid = get_fid(z)
            mean = [DH_over_rd_fid, DM_over_rd_fid]
            label = ['DH_over_rs', 'DM_over_rs']
            mean_txt = [mean_template.format(z, mean, label) for mean, label in zip(mean, label)]
        dirname = Path('/global/cfs/cdirs/desicollab/science/lya/y1-kp6/iron-final-bao/final-result/')
        # Trick: this will overwrite the files written above obtained from Fisher, such that we use the exact ones
        write('Lya', mean_txt=mean_txt, cov=dirname / 'DESI-Y1-Lya.cov', iz=None, zrange=None, basename=basename)
        if not public:
            write('Lya', mean_txt=mean_txt, cov=dirname / 'DESI-Y1-Lya.cov', iz=0, zrange=(1.8, 4.2), basename=basename)

        # DESI-eBOSS Lya measurement
        dirname = Path('/global/cfs/cdirs/desicollab/science/lya/y1-kp6/iron-final-bao/final-result/')
        mean_txt = dirname / 'DESI-eBOSS-Lya.dat'
        if mock:
            z, center = np.loadtxt(mean_txt, comments='#', usecols=[0, 1], unpack=True)
            z = z.flat[0]
            assert np.allclose(z, 2.33)
            assert center[0] < center[1]  # hacky way to check order: DH, DM
            DM_over_rd_fid, DH_over_rd_fid, DV_over_rd_fid, FAP_fid = get_fid(z)
            mean = [DH_over_rd_fid, DM_over_rd_fid]
            label = ['DH_over_rs', 'DM_over_rs']
            mean_txt = [mean_template.format(z, mean, label) for mean, label in zip(mean, label)]
        write('Lya', mean_txt=mean_txt, cov=dirname / 'DESI-eBOSS-Lya.cov', iz=None, zrange=None, tracer_zrange_txt='Lya, in combination with eBOSS', basename=basename + '_eboss')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Y1 BAO cosmological likelihoods')
    parser.add_argument('--todo', type=str, nargs='*', required=False, default=['copy', 'export', 'bindings'], choices=['copy', 'export', 'bindings', 'sampling'])
    args = parser.parse_args()

    from desilike import setup_logging
    from desilike.bindings import CobayaLikelihoodGenerator, CosmoSISLikelihoodGenerator, MontePythonLikelihoodGenerator

    chains_dir = 'desilike'
    version = 'v1.5'

    if 'copy' in args.todo:
        from desilike import utils
        utils.mkdir(data_dir(version))

        import glob
        # what was used when running DESI chains
        #in_dir = Path('/global/cfs/cdirs/desicollab/science/cpe/y1kp7/bao/data_internal/')
        # updated versions
        #in_dir = Path('/global/cfs/cdirs/desicollab/science/cpe/y1kp7/bao/data' + ('_' + version if version else ''))
        # final repo, with BAO+FS measurements
        in_dir = Path('/global/cfs/cdirs/desi//survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/forfit_2pt/')
        for fn in glob.glob(str(in_dir / 'gaussian_bao*.npy')):
            #fisher = LikelihoodFisher.load(fn)
            #fisher.save(data_dir(version) / Path(fn).name)
            shutil.copyfile(fn, data_dir(version) / Path(fn).name.replace('-recon', ''))

    if 'export' in args.todo:
        #export_compressed_bao_data(fmt='cobaya', public=True)
        export_compressed_bao_data(fmt='cobaya', version=version)
        #export_compressed_bao_data(fmt='cobaya', mock=True)

    if 'bindings' in args.todo:
        setup_logging('info')
        name_likes, kw_likes = [], []
        for tracer in [None, 'bgs_z0', 'lrg_z0', 'lrg_z1', 'lrg_z2', 'lrgpluselg_z0', 'elg_z1', 'qso_z0', 'lya_z0']:
            name_like = 'desi_bao_all'
            if tracer: name_like = f'desi_bao_{tracer}'
            kw_like = {'cosmo': 'external', 'tracers': [tracer] if tracer else None, 'version': version}
            name_likes.append(name_like)
            kw_likes.append(kw_like)
        fn = 'bao.py'
        CobayaLikelihoodGenerator()([DESICompressedBAOLikelihood] * len(name_likes), name_likes, kw_like=kw_likes, fn=fn)
        CosmoSISLikelihoodGenerator()([DESICompressedBAOLikelihood] * len(name_likes), name_likes, kw_like=kw_likes, fn=fn)
        MontePythonLikelihoodGenerator()([DESICompressedBAOLikelihood] * len(name_likes), name_likes, kw_like=kw_likes, fn=fn)

        name_likes, kw_likes = [], []
        for name in ['pantheon', 'pantheonplus', 'pantheonplusshoes', 'union3', 'desy5']:
            name_like = name
            kw_like = {'cosmo': 'external', 'name': name}
            name_likes.append(name_like)
            kw_likes.append(kw_like)
        fn = 'supernovae.py'
        CobayaLikelihoodGenerator()([SNLikelihood] * len(name_likes), name_like=name_likes, kw_like=kw_likes, fn=fn)
        CosmoSISLikelihoodGenerator()([SNLikelihood] * len(name_likes), name_like=name_likes, kw_like=kw_likes, fn=fn)
        MontePythonLikelihoodGenerator()([SNLikelihood] * len(name_likes), name_like=name_likes, kw_like=kw_likes, fn=fn)

    if 'sampling' in args.todo:
        setup_logging('info')
        likelihood_names = ['bao', 'pantheonplus']

        from desilike.theories import Cosmoprimo
        from desilike.samplers import ZeusSampler, EmceeSampler

        cosmo = Cosmoprimo(fiducial='DESI')
        cosmo.init.params = {'Omega_m': {'prior': {'limits': [0.1, 0.9]}, 'ref': {'dist': 'norm', 'loc': 0.3, 'scale': 0.002}, 'latex': '\Omega_m'},
                            'omega_b': {'prior': {'dist': 'norm', 'loc': 0.02236, 'scale': 0.0005}, 'latex': '\omega_b'},
                            'H0':  {'prior': {'limits': [20., 100.]}, 'ref': {'dist': 'norm', 'loc': 70., 'scale': 1.}, 'latex': 'H_{0}'}}
        likelihoods = []

        if 'bao' in likelihood_names:
            likelihood = DESICompressedBAOLikelihood(cosmo=cosmo, tracers=['elg_z1', 'qso_z0'])
            likelihood()
            likelihoods.append(likelihood)

        if 'pantheonplus' in likelihood_names:
            from desilike.likelihoods.supernovae import PantheonPlusSNLikelihood
            likelihood = PantheonPlusSNLikelihood(cosmo=cosmo)
            from desilike.install import Installer
            installer = Installer()
            installer(likelihood)
            likelihoods.append(likelihood)

        chains_dir = os.path.join(os.path.dirname(__file__), 'chains_{}'.format('_'.join(likelihood_names)))
        sampler = EmceeSampler(sum(likelihoods), nwalkers=20, seed=42, save_fn=os.path.join(chains_dir, 'chain.1.npy'))
        sampler.run(check={'max_eigen_gr': 0.02})