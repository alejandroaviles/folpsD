"""
Run chains on NERSC.

Look at https://desipipe.readthedocs.io/en/latest/user/getting_started.html#cheat-list.
To spawn a manager process that will send the tasks to NERSC's queue:

.. code-block:: bash

    desipipe spawn -q y1_fs_cosmo --spawn

To see the current state of the queue:

.. code-block:: bash

    desipipe queues -q y1_fs_cosmo

To see "live output"  of the running tasks:

.. code-block:: bash

    desipipe tasks -q y1_fs_cosmo --state RUNNING

To put the chains that haven't converged back in the queue (you may need to spawn a new manager process):

.. code-block:: bash

    desipipe retry -q y1_fs_cosmo --state KILLED

"""

nthreads = 8
#nthreads = 16
environ_nthreads = {NAME: str(nthreads) for NAME in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}

import os
os.environ.update(dict(environ_nthreads))

# Force FOLPS to use JAX backend (required for FOLPSv2 with desilike's JAX-based differentiation)
os.environ['FOLPS_BACKEND'] = 'jax'

# ============================================================================
# Force local desilike checkout (with FOLPSv2) BEFORE any desilike import
# ============================================================================
import sys as _sys
import importlib as _importlib
_LOCAL_DESILIKE = "/global/cfs/cdirs/desicollab/users/hernannb/__FOLPS_tutorial/desilike"

# Set PYTHONPATH environment variable for any subprocesses
_old_pythonpath = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = _LOCAL_DESILIKE + (':' + _old_pythonpath if _old_pythonpath else '')

# Ensure local desilike is FIRST in sys.path (remove if present, then insert at 0)
while _LOCAL_DESILIKE in _sys.path:
    _sys.path.remove(_LOCAL_DESILIKE)
_sys.path.insert(0, _LOCAL_DESILIKE)

# Remove any cached desilike modules so they reload from the local path
for _mod in list(_sys.modules.keys()):
    if _mod == 'desilike' or _mod.startswith('desilike.'):
        del _sys.modules[_mod]

# Invalidate import caches
_importlib.invalidate_caches()
# ============================================================================

from desipipe import Queue, Environment, TaskManager, spawn, setup_logging
from y1_fs_cosmo_tools import profile_cobaya, sample_cobaya, importance_cobaya, yield_configs, get_cobaya_output, exists_cobaya_output, print_convergence, print_margestats

setup_logging()

# best fits for > 20
i = 4
#i = 40
queue = Queue('y1_fs_bao_cosmo{:d}'.format(i))
queue.clear(kill=False)

environ = Environment('nersc-cosmodesi')
environ.update(environ_nthreads)

output, error = '_sbatch_fs_cosmo/slurm-%j.out', '_sbatch_fs_cosmo/slurm-%j.err'
tm = TaskManager(queue=queue, environ=environ)
#tm_sample = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='08:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
tm_sample = tm.clone(scheduler=dict(max_workers=50), provider=dict(provider='nersc', time='08:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.5, output=output, error=error, killed_at_timeout=False))
#tm_sample = tm.clone(scheduler=dict(max_workers=50), provider=dict(provider='nersc', time='14:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.5, output=output, error=error, killed_at_timeout=False, kwargs={'reservation': 'kp7b_3'}))
#tm_sample = tm.clone(scheduler=dict(max_workers=50), provider=dict(provider='nersc', time='06:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
#tm_sample = tm.clone(scheduler=dict(max_workers=100), provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.2, output=output, error=error, killed_at_timeout=False))
tm_profile = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='05:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.5, output=output, error=error, killed_at_timeout=True))
#tm_importance = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='03:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True))
tm_importance = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time='05:00:00', mpiprocs_per_worker=4, nodes_per_worker=0.5, output=output, error=error, killed_at_timeout=True))
#tm_importance = tm.clone(scheduler=dict(max_workers=55), provider=dict(provider='nersc', time='20:00:00', mpiprocs_per_worker=4, nodes_per_worker=1, output=output, error=error, killed_at_timeout=True, stop_after=1))


@tm_profile.python_app
def profile(profile_cobaya=profile_cobaya, **kwargs):
    from desipipe import setup_logging
    setup_logging()
    profile_cobaya(**kwargs)


#@tm_sample.python_app                          ### Comment
def sample(sample_cobaya=sample_cobaya, **kwargs):
    from desipipe import setup_logging
    setup_logging()
    sample_cobaya(**kwargs)


@tm_importance.python_app
def importance(importance_cobaya=importance_cobaya, **kwargs):
    from desipipe import setup_logging
    setup_logging()
    importance_cobaya(**kwargs)


if __name__ == '__main__':

    todo = []
    #todo += ['profile']
    #todo = ['sample1']
    todo = ['test']
    #todo += ['sample0', 'sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8'][i:i+1]
    #todo += ['profile0']
    #todo += ['profile1']
    #todo += ['importance']
    #todo += ['convergence']

    theory = 'camb'
    run = 'run'

    if 'test' in todo:
        model = 'base_mnu'
        dataset = ['desi-folpsv2-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10']
        sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

    if 'sample0' in todo:  # finished
        # run new BAO
        theory = 'camb'
        model = 'base'
        """
        datasets = []
        for tracer in ['bgs', 'lrg-z0', 'lrg-z1', 'lrgpluselg', 'elg', 'qso', 'all']:
            datasets.append(['desi-v1.5-bao-{}'.format(tracer), 'schoneberg2024-bbn'])
        for dataset in datasets:
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
        """
        dataset = ['desi-v1.5-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']
        for model in ['base', 'base_mnu', 'base_nnu', 'base_w_wa']:
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
        model = 'base_w_wa'
        for sn in ['pantheonplus', 'union3', 'desy5sn']:
            dataset = ['desi-v1.5-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
        #tracer = 'all-nolya'
        #dataset = ['desi-v1.5-bao-{}'.format(tracer), 'schoneberg2024-bbn']
        #sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
    
    if 'sample1' in todo:  # finished
        # base runs, DESI-only
        theory = 'camb'
        model = 'base'
        datasets = []
        """
        for tracer in ['bgs', 'lrg-z0', 'lrg-z1', 'lrg-z2', 'elg', 'qso']:
            datasets += [['desi-reptvelocileptors-fs-{}'.format(tracer), 'schoneberg2024-bbn', 'planck2018-ns10']]
            datasets += [['desi-reptvelocileptors-fs-bao-{}'.format(tracer), 'schoneberg2024-bbn', 'planck2018-ns10']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10']]
        datasets += [['desi-reptvelocileptors-fs-bao-all-nolya', 'schoneberg2024-bbn', 'planck2018-ns10']]
        datasets += [['desi-reptvelocileptors-fs-all', 'schoneberg2024-bbn', 'planck2018-ns10']]
        """
        """
        tracer = 'all-nolya'
        dataset = ['desi-v1.5-bao-{}'.format(tracer), 'schoneberg2024-bbn']
        sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

        tracer = 'lrg-z2'
        dataset = ['desi-v1.5-bao-{}'.format(tracer), 'schoneberg2024-bbn']
        sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

        for tracer in ['bgs', 'lrg-z0', 'lrg-z1', 'lrg-z2', 'elg', 'qso', 'all', 'all-nolya']:
            if tracer != 'all_nolya':
                datasets += [['desi-shapefit-{}'.format(tracer), 'schoneberg2024-bbn', 'planck2018-ns10']]
            datasets += [['desi-shapefit-bao-{}'.format(tracer), 'schoneberg2024-bbn', 'planck2018-ns10']]
        for dataset in datasets:
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
        """
        """
        tracer = 'all-nolya'
        datasets += [['desi-reptvelocileptors-fs-bao-{}'.format(tracer), 'schoneberg2024-bbn', 'planck2018-ns']]
        datasets += [['desi-shapefit-bao-{}'.format(tracer), 'schoneberg2024-bbn', 'planck2018-ns']]
        for dataset in datasets:
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
        """
        for tracer in ['bgs', 'lrg-z0', 'lrg-z1', 'lrg-z2', 'elg', 'qso', 'all', 'all-nolya']:
            datasets += [['desi-shapefit-bao-{}'.format(tracer), 'schoneberg2024-bbn', 'planck2018-ns10']]
            datasets += [['desi-shapefit-joint-{}'.format(tracer), 'schoneberg2024-bbn', 'planck2018-ns10']]
        for dataset in datasets:
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

    if 'sample2' in todo:
        # base runs, DESI alone
        theory = 'camb'
        model = 'base'
        datasets = []
        # base runs, DESI + CMB
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE']]
        for dataset in datasets:
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

    if 'sample3' in todo:
        # base and w0wa runs, with SN
        theory = 'camb'
        for model in ['base', 'base_w_wa']:
            datasets = []
            #  DESI + SN
            for sn in ['pantheonplus', 'union3', 'desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'schoneberg2024-bbn', 'planck2018-ns10']]
            for dataset in datasets:
                sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
            datasets = []
            #  DESI + CMB + SN
            for sn in ['pantheonplus', 'union3', 'desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
            for sn in ['desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']]
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE']]
            for dataset in datasets:
                sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

    if 'sample4' in todo:
        # MG
        theory = 'isitgr'
        model = 'base_mu_sigma'

        datasets = []
        #datasets += [['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
        datasets += [['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE', 'planck-act-dr6-lensing']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'desy3joint', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
        for dataset in datasets:
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

    if 'sample4b' in todo:
        # MG
        theory = 'isitgr'
        model = 'base_mu_sigma'
        datasets = []
        datasets += [['desi-reptvelocileptors-prior3-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10']]
        datasets += [['desi-reptvelocileptors-prior3-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
        for dataset in datasets:
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
    
    if 'sample5' in todo:
        # base_mnu, base_nnu
        theory = 'camb'
        for model in ['base_mnu', 'base_nnu']:
            datasets = []
            #  DESI + CMB
            datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
            datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing']]
            datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE', 'planck-act-dr6-lensing']]
            for dataset in datasets:
                sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
        for model in ['base_mnu_w_wa', 'base_nnu_w_wa']:
            datasets = []
            #  DESI + SN + CMB
            for sn in ['pantheonplus', 'union3', 'desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
            for sn in ['desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing']]
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE', 'planck-act-dr6-lensing']]
            for dataset in datasets:
                sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

    if 'sample6' in todo:
        model = 'base'
        dataset = ['mock-cmblens-desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']
        sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

        #model = 'base_mnu'
        #dataset = ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']
        #sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
        
        #dataset = ['planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']
        #sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)

    if 'sample7' in todo:  # include lensing
        # base runs, DESI + CMB
        model = 'base'
        datasets = []
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE', 'planck-act-dr6-lensing']]
        for dataset in datasets:
            sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
        for model in ['base', 'base_w_wa']:
            datasets = []
            #  DESI + CMB + SN
            for sn in ['pantheonplus', 'union3', 'desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
            for sn in ['desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing']]
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE', 'planck-act-dr6-lensing']]
            for dataset in datasets:
                sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
        for model in ['base_mnu', 'base_nnu']: # remove lensing
            datasets = []
            datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
            for dataset in datasets:
                sample(model=model, theory=theory, dataset=dataset, run=run, resume=True)
                
    if 'importance' in todo:
        model = 'base'
        datasets = []
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE']]
        for dataset in datasets:
            importance(model=model, theory=theory, dataset=dataset, add=['planck-act-dr6-lensing'], run=run, skip=0.3, thin=40, resume=True)
        for model in ['base', 'base_w_wa']:
            datasets = []
            #  DESI + CMB + SN
            for sn in ['pantheonplus', 'union3', 'desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
            for sn in ['desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']]
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE']]
            for dataset in datasets:
                importance(model=model, theory=theory, dataset=dataset, add=['planck-act-dr6-lensing'], run=run, skip=0.3, thin=40, resume=True)

        for model in ['base', 'base_w_wa']:
            datasets = []
            #  DESI + CMB + SN
            for sn in ['pantheonplus']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
            for dataset in datasets:
                importance(model=model, theory=theory, dataset=dataset, add=['planck-act-dr6-lensing'], run=run, skip=0.3, thin=40, resume=True)
                
    if 'convergence' in todo:
        from pathlib import Path
        import glob
        setup_logging('warning')
        for model in ['base', 'base_w_wa', 'base_mnu', 'base_nnu', 'base_mnu_w_wa', 'base_nnu_w_wa', 'base_mu_sigma']:
            dirname = Path(get_cobaya_output(run=run, model=model, theory='isitgr' if model == 'base_mu_sigma' else 'camb')).parent.parent / '*'
            for dirname in glob.glob(str(dirname)):
                if 'bak' in dirname: continue
                if 'reptvelocileptors-fs' not in dirname: continue
                #if 'planck2018-lowl' not in dirname: continue
                if 'add_' in dirname: continue
                suffix = '.post.importance' if 'add_' in dirname else ''
                dirname = Path(dirname)
                print('{}/{}'.format(model, dirname.name))
                #print_convergence(dirname / ('chain' + suffix), cosmo_only=True, max_gr=0.01)
                print_convergence(dirname / ('chain' + suffix), cosmo_only=False, max_gr=0.02)

    if 'profile0' in todo:
        """
        for model in ['base', 'base_w_wa']:
            datasets = []
            datasets += [['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10']]
            #  DESI + SN
            for sn in ['pantheonplus', 'union3', 'desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'schoneberg2024-bbn', 'planck2018-ns10']]
            for dataset in datasets:
                profile(model=model, theory=theory, dataset=dataset, run=run, ignore_prior=False)
            datasets = []
            #  DESI + CMB + SN
            for sn in ['pantheonplus', 'union3', 'desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
            for sn in ['desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE']]
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE']]
            for dataset in datasets:
                profile(model=model, theory=theory, dataset=dataset, add=['planck-act-dr6-lensing'], run=run, ignore_prior=False)
        """
        """
        for model in ['base', 'base_w_wa']:
            datasets = []
            #  DESI + SN
            for sn in ['pantheonplus']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'schoneberg2024-bbn', 'planck2018-ns10']]
            for dataset in datasets:
                profile(model=model, theory=theory, dataset=dataset, run=run, ignore_prior=False)
            datasets = []
            #  DESI + CMB + SN
            for sn in ['pantheonplus']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
            for dataset in datasets:
                profile(model=model, theory=theory, dataset=dataset, run=run, ignore_prior=False)

        for model in ['base_mnu', 'base_mnu059', 'base_mnu100']:
            datasets = []
            #  DESI + CMB
            datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
            datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing']]
            datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE', 'planck-act-dr6-lensing']]
            for dataset in datasets:
                profile(model=model, theory=theory, dataset=dataset, run=run, ignore_prior=False)
        """
        """
        for model in ['base', 'base_w_wa']:
            datasets = []
            #  DESI + CMB + SN
            for sn in ['pantheonplus', 'union3', 'desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]
            for sn in ['desy5sn']:
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing']]
                datasets += [['desi-reptvelocileptors-fs-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE', 'planck-act-dr6-lensing']]
            for dataset in datasets:
                profile(model=model, theory=theory, dataset=dataset, run=run, ignore_prior=False)
        """
        model = 'base_w_wa'
        for sn in ['pantheonplus', 'union3', 'desy5sn']:
            dataset = ['desi-v1.5-bao-all', sn, 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']
            profile(model=model, theory=theory, dataset=dataset, run=run, ignore_prior=False)

    if 'profile1' in todo:
        theory = 'isitgr'
        model = 'base_mu_sigma'
        datasets = []
        #datasets += [['desi-reptvelocileptors-fs-bao-all', 'schoneberg2024-bbn', 'planck2018-ns10']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing']]

        #datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planck-act-dr6-lensing']]
        #datasets += [['desi-reptvelocileptors-fs-bao-all', 'planck2018-lowl-TT-clik', 'planck2020-lollipop-lowlE', 'planck2020-hillipop-TTTEEE', 'planck-act-dr6-lensing']]
        datasets += [['desi-reptvelocileptors-fs-bao-all', 'desy3joint', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]

        datasets += [['desi-reptvelocileptors-fs-bao-all', 'desy3joint', 'desy5sn', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE']]
        for dataset in datasets:
            profile(model=model, theory=theory, dataset=dataset, run=run, ignore_prior=False)