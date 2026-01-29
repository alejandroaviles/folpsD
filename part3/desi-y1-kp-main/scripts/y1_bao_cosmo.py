"""
Run chains on NERSC.

Look at https://desipipe.readthedocs.io/en/latest/user/getting_started.html#cheat-list.
To spawn a manager process that will send the tasks to NERSC's queue:

.. code-block:: bash

    desipipe spawn -q y1_bao_cosmo --spawn

To see the current state of the queue:

.. code-block:: bash

    desipipe queues -q y1_bao_cosmo

To see "live output"  of the running tasks:

.. code-block:: bash

    desipipe tasks -q y1_bao_cosmo --state RUNNING

To put the chains that haven't converged back in the queue (you may need to spawn a new manager process):

.. code-block:: bash

    desipipe retry -q y1_bao_cosmo --state KILLED

"""

nthreads = 8
environ_nthreads = {NAME: str(nthreads) for NAME in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}

import os
os.environ.update(dict(environ_nthreads))

from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
from y1_bao_cosmo_tools import profile_cobaya, sample_cobaya, importance_cobaya, yield_configs, get_cobaya_output, exists_cobaya_output, print_convergence, print_margestats

setup_logging()

queue = Queue('y1_bao_cosmo4')
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi')
environ.update(environ_nthreads)

output, error = '_sbatch_bao_cosmo/slurm-%j.out', '_sbatch_bao_cosmo/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ)
#tm_sample = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='08:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
#tm_sample = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
tm_sample = tm.clone(scheduler=dict(max_workers=100), provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.2, output=output, error=error, killed_at_timeout=True))
tm_profile = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='05:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.5, output=output, error=error, killed_at_timeout=True))
#tm_importance = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='03:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
tm_importance = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='10:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))


#@tm_profile.python_app
def profile(profile_cobaya=profile_cobaya, **kwargs):
    from desipipe import setup_logging
    setup_logging()
    profile_cobaya(**kwargs)


@tm_sample.python_app
def sample(sample_cobaya=sample_cobaya, **kwargs):
    from desipipe import setup_logging
    setup_logging()
    sample_cobaya(**kwargs)


#@tm_importance.python_app
def importance(importance_cobaya=importance_cobaya, **kwargs):
    from desipipe import setup_logging
    setup_logging()
    importance_cobaya(**kwargs)


if __name__ == '__main__':

    todo = []
    #todo += ['profile']
    todo += ['sample']
    #todo += ['importance']
    #todo += ['convergence']

    theory = 'camb'
    if 'profile' in todo:
        for model in ['base', 'base_w_wa']:
            datasets = []
            for sn in ['pantheonplus', 'union3', 'desy5sn']:
                datasets += [['desi-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing-v1.2']]
                datasets += [['desi-v1.5-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
            for dataset in datasets:
                profile(model=model, theory=theory, dataset=dataset, run='run3', ignore_prior=False)

    if 'sample' in todo:
        for sn in ['pantheonplus', 'union3', 'desy5sn']:
            sample(model='base', theory='classy', dataset=['desi-bao-all', 'planck2018-highl-plik-TTTEEE', sn], run='test', debug=True, resume=False)

    if 'sample2' in todo:
        sample(model='base_mnu_ih2_dmnu2fixed', theory=theory, dataset=['desi-bao-all', 'desy5sn'], run='test', resume=False)

    if 'convergence' in todo:
        for config in yield_configs(theory=theory):
            if not exists_cobaya_output(**config): continue
            print_convergence(get_cobaya_output(**config))
            print_margestats(get_cobaya_output(**config), fn=True)

    if 'importance' in todo:
        for config in yield_configs(importance=True, models=['base_omegak']):
            #if exists_cobaya_output(**config): continue
            if not any(name in get_cobaya_output(**config) for name in ['pantheonplus', 'union3', 'desy5']):
                continue
            print(get_cobaya_output(**config))
            importance(**config, theory=theory, resume=False)

    #spawn(queue, spawn=True)
