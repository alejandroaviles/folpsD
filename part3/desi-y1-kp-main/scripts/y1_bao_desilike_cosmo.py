"""
Run chains on NERSC.

Look at https://desipipe.readthedocs.io/en/latest/user/getting_started.html#cheat-list.
To spawn a manager process that will send the tasks to NERSC's queue:

.. code-block:: bash

    desipipe spawn -q y1_bao_desilike_cosmo --spawn

To see the current state of the queue:

.. code-block:: bash

    desipipe queues -q y1_bao_desilike_cosmo

To see "live output"  of the running tasks:

.. code-block:: bash

    desipipe tasks -q y1_bao_desilike_cosmo --state RUNNING

To put the chains that haven't converged back in the queue (you may need to spawn a new manager process):

.. code-block:: bash

    desipipe retry -q y1_bao_desilike_cosmo --state KILLED

"""

nthreads = 32
environ_nthreads = {NAME: str(nthreads) for NAME in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}

import os
os.environ.update(dict(environ_nthreads))

from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
from y1_bao_desilike_cosmo_tools import emulate_desilike, profile_desilike, sample_desilike, yield_configs, get_desilike_output

setup_logging()

queue = Queue('y1_bao_desilike')
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi', command='module unload cosmoprimo desilike')
environ.update(environ_nthreads)

output, error = '_sbatch_bao_cosmo/slurm-%j.out', '_sbatch_bao_cosmo/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ)
tm_emulate = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=64, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
#tm_sample = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='08:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
tm_sample = tm.clone(scheduler=dict(max_workers=9), provider=dict(provider='nersc', time='00:05:00', mpiprocs_per_worker=4, nodes_per_worker=0.25, output=output, error=error, killed_at_timeout=True))
#tm_sample = tm.clone(scheduler=dict(max_workers=100), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.2, output=output, error=error, killed_at_timeout=False))
tm_profile = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='05:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.5, output=output, error=error, killed_at_timeout=True))
#tm_importance = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='03:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
tm_importance = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='03:10:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
#tm_importance = tm.clone(scheduler=dict(max_workers=55), provider=dict(provider='nersc', time='20:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True, stop_after=1))


#@tm_emulate.python_app
def emulate(emulate_desilike=emulate_desilike, **kwargs):
    from desipipe import setup_logging
    setup_logging()
    emulate_desilike(**kwargs)


#@tm_profile.python_app
def profile(profile_desilike=profile_desilike, **kwargs):
    from desipipe import setup_logging
    setup_logging()
    profile_desilike(**kwargs)


#@tm_sample.python_app
def sample(sample_desilike=sample_desilike, **kwargs):
    from desipipe import setup_logging
    setup_logging()
    sample_desilike(**kwargs)


#@tm_importance.python_app
#def importance(importance_desilike=importance_desilike, **kwargs):
#    from desipipe import setup_logging
#    setup_logging()
#    importance_desilike(**kwargs)


if __name__ == '__main__':

    todo = []
    #todo = ['test']
    #todo += ['profile']
    todo += ['sample']

    theory = 'capse'
    run = 'test'
    model = 'base'
    #model = 'base_w_wa'

    if 'profile' in todo:
        model = 'base_w_wa'
        datasets = []
        #datasets.append(['desi-bao-all', 'schoneberg2024-bbn'])
        #datasets.append(['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT', 'planck2018-lowl-EE', 'planck-NPIPE-highl-CamSpec-TTTEEE'])
        #datasets.append(['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT', 'planck2018-lowl-EE', 'planck-NPIPE-highl-CamSpec-TTTEEE']) #, 'planck-act-dr6-lensing'])
        #datasets.append(['desi-bao-all', 'desy5sn', 'schoneberg2024-bbn'])
        #datasets.append(['desi-bao-all', 'desy5sn', 'planck2018-lowl-TT', 'planck2018-lowl-EE', 'planck2018-highl-TTTEEE-lite', 'planck-act-dr6-lensing'])
        datasets.append(['desi-dr2-bao-all', 'desy5sn', 'planck2018-lowl-TT', 'planck2018-lowl-EE', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing'])
        for dataset in datasets:
            profile(model=model, theory=theory, dataset=dataset, run=run, profile_params=['w0_fld', 'wa_fld'])

    if 'sample' in todo:
        model = 'base_w_wa'
        datasets = []
        #datasets.append(['desi-dr2-bao-all', 'desy5sn', 'planck2018-lowl-TT', 'planck2018-lowl-EE', 'planck2018-highl-TTTEEE-lite', 'planck-act-dr6-lensing'])
        datasets.append(['desi-dr2-bao-all', 'desy5sn', 'planck2018-lowl-TT', 'planck2018-lowl-EE', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing'])
        #sampler = 'mcmc'
        sampler = 'nuts'
        for dataset in datasets:
            sample(model=model, theory=theory, dataset=dataset, run=run, sampler=sampler, resume=True)
