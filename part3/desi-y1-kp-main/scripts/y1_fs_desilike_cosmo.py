"""
Run chains on NERSC.

Look at https://desipipe.readthedocs.io/en/latest/user/getting_started.html#cheat-list.
To spawn a manager process that will send the tasks to NERSC's queue:

.. code-block:: bash

    desipipe spawn -q y1_fs_desilike_cosmo --spawn

To see the current state of the queue:

.. code-block:: bash

    desipipe queues -q y1_fs_desilike_cosmo

To see "live output"  of the running tasks:

.. code-block:: bash

    desipipe tasks -q y1_fs_desilike_cosmo --state RUNNING

To put the chains that haven't converged back in the queue (you may need to spawn a new manager process):

.. code-block:: bash

    desipipe retry -q y1_fs_desilike_cosmo --state KILLED

"""

nthreads = 32
environ_nthreads = {NAME: str(nthreads) for NAME in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}

import os
os.environ.update(dict(environ_nthreads))

from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
from y1_fs_cosmo_tools import emulate_desilike, profile_desilike, sample_desilike, yield_configs, get_desilike_output, print_convergence, print_margestats

setup_logging()

queue = Queue('y1_fs_bao_desilike')
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi', command='module swap velocileptors/1.0.0 velocileptors/master')
environ.update(environ_nthreads)

output, error = '_sbatch_fs_cosmo/slurm-%j.out', '_sbatch_fs_cosmo/slurm-%j.err'
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
    #todo += ['profile2']
    todo += ['emulate']
    #todo += ['sample4']
    #todo += ['importance']
    #todo += ['convergence']

    theory = 'camb'
    #pt = '-synthetic-lptvelocileptors'
    pt = '-reptvelocileptors'
    #pt = '-folpsax'
    #run = 'proposal'
    #run = 'dragging'
    #run = 'synthetic'
    #run = 'run4'
    run = 'test'
    #model = 'base'

    if 'test' in todo:
        model = 'base'
        theory = 'camb'
        nchains = 4
        output = [get_desilike_output(model=model, theory=theory, dataset=['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10'], run='test', sampler='mcmc', emulator_fn=False, suffix=ichain) for ichain in range(nchains)]
        gr = diagnostics.gelman_rubin(samples, samples[0].params(varied=True, derived=False), method='eigen', check_valid='ignore').max()
        print(gr - 1.)
        pars = samples[0].names(varied=True, derived=False)
        samples = [sample.to_getdist() for sample in samples]
        from y1_fs_cosmo_tools import getGelmanRubinEigenvalues
        gr = getGelmanRubinEigenvalues(samples[0], pars=pars, chainlist=samples).max()
        print(gr)
        
        """
        observable = 'fs'
        tracer = 'qso'
        sample(model=model, theory=theory, dataset=['desi{}-{}-{}'.format(pt, observable, tracer), 'schoneberg2024-bbn'], run=run, sampler='nuts', emulator_fn=model, resume=False)
        """

    if 'profile0' in todo:
        model = 'base'
        from y1_fs_cosmo_tools import load_desilike_samples
        for observable in ['fs', 'fs-bao']:
            for tracer in ['bgs', 'lrg-z0', 'lrg-z1', 'lrg-z2', 'elg', 'qso', 'all', 'all-nolya'][-2:]:
                if observable == 'fs' and tracer == 'all-nolya': continue
                dataset = ['desi{}-{}-{}'.format(pt, observable, tracer), 'schoneberg2024-bbn', 'planck2018-ns10']
                start = load_desilike_samples(model=model, theory=theory, run='run3', sampler='minuit', emulator_fn=model, dataset=dataset).bestfit.choice(input=True, index='argmax')
                start = {name: float(value) for name, value in start.items()}
                profile(model=model, theory=theory, dataset=dataset, start=start, run=run, emulator_fn=None)

    if 'profile1' in todo:
        model = 'base'
        for observable in ['shapefit', 'shapefit-bao']:
            for tracer in ['bgs', 'lrg-z0', 'lrg-z1', 'lrg-z2', 'elg', 'qso', 'all', 'all-nolya'][-2:]:
                if observable == 'shapefit' and tracer == 'all-nolya': continue
                profile(model=model, theory=theory, dataset=['desi-{}-{}'.format(observable, tracer), 'schoneberg2024-bbn', 'planck2018-ns10'], run=run)
    
    if 'profile2' in todo:
        model = 'base'
        for observable in ['fs', 'fs-bao'][1:]:
            for tracer in ['all']:
                dataset = ['desi{}-{}-{}'.format(pt, observable, tracer), 'schoneberg2024-bbn', 'planck2018-ns10']
                #emulate(model=model, theory=theory, dataset=dataset)
                profile(model=model, theory=theory, dataset=dataset, run=run, emulator_fn=model, custom_prior=False)

    if 'profile4' in todo:
        model = 'base_w_wa'
        for observable in ['fs', 'fs-bao', 'fs-corr-recon', 'shapefit-bao'][1:2]:
            for tracer in ['all', 'bgs', 'lrg-z0', 'lrg-z1', 'lrg-z2', 'elg', 'qso'][:1]:
                print(observable, tracer)
                profile(model=model, theory=theory, dataset=['mock-cmblens-desi{}-{}-{}'.format(pt, observable, tracer), 'schoneberg2024-bbn', 'planck2018-ns10'], run=run) #, emulator_fn=model)

    if 'emulate' in todo:
        model = 'base'
        for tracer in ['all', 'bgs', 'lrg-z0', 'lrg-z1', 'lrg-z2', 'elg', 'qso'][:1]:
            emulate(model=model, theory=theory, dataset=['desi{}-fs-bao-{}'.format(pt, tracer), 'schoneberg2024-bbn', 'planck2018-ns10'])
            #emulate(model=model, theory=theory, dataset=['desi{}-fs-corr-recon-{}'.format(pt, tracer), 'schoneberg2024-bbn', 'planck2018-ns10'])
            #emulate(model=model, theory=theory, dataset=['desi{}-shapefit-bao-{}'.format(pt, tracer), 'schoneberg2024-bbn', 'planck2018-ns10'])

    if 'sample' in todo:
        model = 'base'
        for observable in ['fs', 'fs-bao', 'fs-corr-recon', 'shapefit-bao'][1:2]:
            for tracer in ['all', 'bgs', 'lrg-z0', 'lrg-z1', 'lrg-z2', 'elg', 'qso']:
                print(observable, tracer)
                sample(model=model, theory=theory, dataset=['mock-cmblens-desi{}-{}-{}'.format(pt, observable, tracer), 'schoneberg2024-bbn', 'planck2018-ns10'], run=run, sampler='nuts', emulator_fn=model, resume=False)

    if 'sample4' in todo:
        model = 'base_w_wa'
        for observable in ['fs', 'fs-bao', 'fs-corr-recon', 'shapefit-bao'][1:2]:
            for tracer in ['all', 'bgs', 'lrg-z0', 'lrg-z1', 'lrg-z2', 'elg', 'qso'][:1]:
                #emulate(model=model, theory=theory, dataset=['desi{}-fs-bao-{}'.format(pt, tracer), 'schoneberg2024-bbn', 'planck2018-ns10'])
                sample(model=model, theory=theory, dataset=['desi{}-{}-{}'.format(pt, observable, tracer), 'schoneberg2024-bbn', 'planck2018-ns10'], run=run, sampler='nuts', emulator_fn=model, resume=False)
    #spawn(queue, spawn=True)

    if 'sample5' in todo:
        model = 'base'
        for observable in ['reptvelocileptors-fs-bao', 'shapefit-bao', 'standard-bao', 'shapefit-bao', 'shapefit-noqap-bao', 'shapefit-nodf-bao'][-1:]:
            tracer = 'all'
            model = 'base'
            dataset = ['desi-{}-{}'.format(observable, tracer), 'schoneberg2024-bbn', 'planck2018-ns10']
            emulate(model=model, theory=theory, dataset=dataset)
            sample(model=model, theory=theory, dataset=dataset, run=run, sampler='nuts', emulator_fn=model, resume=False)