nthreads = 2
environ_nthreads = {NAME: str(nthreads) for NAME in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}

import os
os.environ.update(dict(environ_nthreads))

from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
from y1_cosmo_tools import sample_cobaya, sample_desilike, emulate_desilike, profile_desilike, measure_speed, importance_planck, plot_bestfit
from y1_data_fits_tools import get_observable_likelihood as get_observable_likelihood_compression
from y1_data_fits_tools import profile as profile_compression
from y1_data_fits_tools import sample as sample_compression

setup_logging()


queue = Queue('y1_kp4_unblinding_cosmo')
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi')
environ.update(environ_nthreads)

tm = TaskManager(queue=queue, environ=environ)
output, error = '_sbatch_kp4_unblinding_cosmo/slurm-%j.out', '_sbatch_kp4_unblinding_cosmo/slurm-%j.err'
tm_sample = tm.clone(scheduler=dict(max_workers=40), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=8, nodes_per_worker=0.1, output=output, error=error, killed_at_timeout=False))
tm_importance = tm.clone(scheduler=dict(max_workers=40), provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=32, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
tm_sample_compression = tm.clone(scheduler=dict(max_workers=8), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=32, nodes_per_worker=0.5, output=output, error=error))
tm_profile = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='00:10:00', mpiprocs_per_worker=10, nodes_per_worker=0.2, output=output, error=error))
tm_emulate = tm.clone(scheduler=dict(max_workers=8), provider=dict(provider='nersc', time='00:45:00', mpiprocs_per_worker=64, output=output, error=error))


#@tm_profile.python_app
def profile(output, profile=profile_desilike, **kwargs):
    from desilike import setup_logging
    setup_logging()
    profile(output, **kwargs)
    return output


@tm_sample.python_app
def sample(output, sample_cobaya=sample_cobaya, sample_desilike=sample_desilike, **kwargs):
    from desilike import setup_logging
    setup_logging()
    if output.options['code'] == 'cobaya':
        sample_cobaya(output, **kwargs)
    else:
        sample_desilike(output, **kwargs)
    return output


def speed(output, measure_speed=measure_speed, **kwargs):
    measure_speed(output, **kwargs)


#@tm_importance.python_app
def importance(output, importance=importance_planck, **kwargs):
    from desilike import setup_logging
    setup_logging()
    importance(output, **kwargs)
    return output

    
# Emulate theory for faster inference
@tm_emulate.python_app
def emulate(output, emulate_desilike=emulate_desilike, **kwargs):
    from desilike import setup_logging
    setup_logging()
    emulate_desilike(output, **kwargs)
    return output


@tm_emulate.python_app
def emulate_compression(emulator_fn, get_observable_likelihood=get_observable_likelihood_compression, **kwargs):
    from desilike import setup_logging
    setup_logging()
    get_observable_likelihood(save_emulator=True, emulator_fn=emulator_fn, **kwargs)
    return emulator_fn


# Sample posterior
@tm_profile.python_app
def profile_compression(output, profile=profile_compression, **kwargs):
    from desilike import setup_logging
    setup_logging()
    profile(output, **kwargs)
    return output


# Sample posterior
@tm_sample_compression.python_app
def sample_compression(output, sample=sample_compression, resume=False, **kwargs):
    from desilike import setup_logging
    setup_logging()
    sample(output, sample=sample, resume=resume, **kwargs)
    return output


def fisher_compression(output, fprofiles, fchains):
    from desilike.samples import Profiles, Chain
    profiles = Profiles.load(fprofiles)
    chains = [Chain.load(fchain) for fchain in fchains]
    chain = Chain.concatenate([chain.remove_burnin(0.5)[::10] for chain in chains])
    params = chain.params(basename=['qpar', 'qper', 'qiso', 'qap', 'dm', 'df'], varied=True, derived=False)
    fisher = chain.to_fisher(params=params)
    fisher = fisher.clone(center=profiles.bestfit.choice(params=fisher.params(), index='argmax', return_type='nparray'))
    fisher.attrs.update(chain.attrs)
    fisher.save(output)


if __name__ == '__main__':

    from desi_y1_files import get_data_file_manager, get_cosmo_file_manager

    models = ['base', 'base_mnu', 'base_w', 'base_omegak'][:1]
    #models = ['base', 'base_w', 'base_omegak']
    list_datasets = []
    """
    list_datasets.append(['desi-bao-gaussian', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-bgs', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-lrg', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-elg', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-qso', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-lya', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-elg-qso-lya', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-lrg-elg-qso-lya', 'bbn-omega_b'])

    list_datasets.append(['desi-bao-gaussian'])
    list_datasets.append(['desi-bao-gaussian', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-elg-qso-lya', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg-elg-qso', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian', 'planck2018'])
    """
    """
    #list_datasets.append(['desi-bao-gaussian-bgs-lrg', 'bbn-omega_b'])
    #list_datasets.append(['desi-bao-gaussian-elg-qso-lya', 'bbn-omega_b'])
    #list_datasets.append(['desi-bao-gaussian', 'pantheon+', 'bbn-omega_b'])
    #list_datasets.append(['desi-bao-gaussian', 'pantheon+shoes', 'bbn-omega_b'])
    #list_datasets.append(['pantheon+'])
    #list_datasets.append(['pantheon+shoes'])
    #list_datasets.append(['pantheon'])
    """
    list_datasets.append(['desi-bao-gaussian'])
    list_datasets.append(['desi-bao-gaussian-bgs'])
    list_datasets.append(['desi-bao-gaussian-lrg'])
    list_datasets.append(['desi-bao-gaussian-elg'])
    list_datasets.append(['desi-bao-gaussian-qso'])
    list_datasets.append(['desi-bao-gaussian-lya'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg-elg-qso'])
    list_datasets.append(['desi-bao-gaussian-elg-qso-lya'])
    list_datasets.append(['desi-bao-gaussian-lrg-elg-qso-lya'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg'])

    models = ['base', 'base_mnu', 'base_w', 'base_w_wa', 'base_omegak']
    list_datasets = []
    list_datasets.append(['desi-bao-gaussian', 'planck2018'])

    models = ['base_w_wa']
    list_datasets = []
    list_datasets.append(['desi-bao-gaussian'])
    list_datasets.append(['desi-bao-gaussian-elg-qso-lya'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg-elg-qso'])
    list_datasets.append(['desi-bao-gaussian', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-elg-qso-lya', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg-elg-qso', 'bbn-omega_b'])

    models = ['base', 'base_w', 'base_omegak', 'base_w_wa']
    list_datasets = []
    list_datasets.append(['desi-bao-gaussian'])
    list_datasets.append(['desi-bao-gaussian-elg-qso-lya'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg-elg-qso'])
    list_datasets.append(['desi-bao-gaussian', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-elg-qso-lya', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian-bgs-lrg-elg-qso', 'bbn-omega_b'])

    models = ['base', 'base_w', 'base_omegak', 'base_w_wa']
    list_datasets = []
    list_datasets.append(['desi-bao-gaussian', 'pantheon'])
    list_datasets.append(['desi-bao-gaussian', 'pantheon+'])
    list_datasets.append(['pantheon', 'bbn-omega_b'])
    list_datasets.append(['pantheon+', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian', 'pantheon', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian', 'pantheon+', 'bbn-omega_b'])

    models = ['base', 'base_mnu', 'base_w', 'base_omegak'][:1]
    list_datasets = []
    list_datasets.append(['desi-bao-gaussian-lrg_0'])
    list_datasets.append(['desi-bao-gaussian-lrg_1'])
    list_datasets.append(['desi-bao-gaussian-lrg_2'])

    
    models = ['base', 'base_w', 'base_omegak', 'base_w_wa']
    list_datasets = []
    list_datasets.append(['desi-bao-gaussian', 'union3'])
    list_datasets.append(['pantheon'])
    list_datasets.append(['pantheon+'])
    list_datasets.append(['union3'])
    list_datasets.append(['union3', 'bbn-omega_b'])
    list_datasets.append(['desi-bao-gaussian', 'union3', 'bbn-omega_b'])

    version = 'v1'
    conf = 'wrong'
    fm = get_cosmo_file_manager(version=version, conf=conf)
    dfm = get_data_file_manager(conf=conf)
    for options in fm.select(id='compression_gaussian_y', observable='power').iter_options(intersection=False):
        observable = options['observable']
        fit_type = 'bao_recon' if 'bao' in options['theory'] else 'full_shape'
        fdata = fm.get(id='{}{}_y1'.format(observable, '_recon' if 'recon' in fit_type else ''), **options, ignore=True)
        if not fdata.exists(): continue
        fwmatrix = None
        if observable == 'power':
            fwmatrix = dfm.get(id='wmatrix_power_y1', **options, ignore=True)
            if not fwmatrix.exists(): continue
        # For now, for power spectrum let's just take pre- for post-
        fcovariance = dfm.get(id='covariance_{}{}_y1'.format(observable, '_recon' if 'recon' in fit_type and 'correlation' in observable else ''), **{**options, 'cut': None, 'version': 'v0.6'}, ignore=True)
        if not fcovariance.exists(): continue
        kwargs = dict(data=fdata, covariance=fcovariance,
                      theory_name=options['theory'], template_name=options['template'], wmatrix=fwmatrix)
        femulator = None
        if 'full_shape' in fit_type:
            femulator = fm.get(id='compression_emulator_y1', **options, ignore=True)
            #femulator = emulate_compression(femulator, **kwargs)
        fprofiles = fm.get(id='profiles_compression_y1', **options)
        fchains = fm.select(id='chains_compression_y1', **options)
        #profile_compression(fprofiles, emulator_fn=femulator, **kwargs)
        #sample_compression(fchains, emulator_fn=femulator, **kwargs)
        fisher_compression(fm.get(id='compression_gaussian_y1', **options), fprofiles, fchains)

    for fi in fm.select(id=['profiles_cosmological_inference_y1', 'chain_cosmological_inference_y1'],
                        code=['importance-planck', 'cobaya', 'desilike'][:2], model=models, datasets=list_datasets).iter(intersection=False):
        emulator_fn = {}
        for dataset, emulator in fi.options['emulators'].items():
            emulator_fn[dataset] = {}
            for femulator in fm.select(id='emulator_y1', model='base_omegak_w_wa_mnu', dataset=dataset).iter(intersection=False):
                namespace = femulator.options['namespace']
                #femulator = emulate(output=femulator)
                emulator_fn[dataset][namespace] = femulator
        #speed(fi, emulator_fn=emulator_fn, solve=True)
        if 'profile' in fi.id: profile(fi, emulator_fn=emulator_fn)
        #    plot_bestfit(fi, save_fn='test')
        if 'importance' in fi.options['code']: importance(fi, emulator_fn=emulator_fn)
        if 'chain' in fi.id and 'planck2018' not in fi.options['datasets']: sample(fi, emulator_fn=emulator_fn)

    spawn(queue, spawn=True)
