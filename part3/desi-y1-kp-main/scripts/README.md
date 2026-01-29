# Scripts

- y1_data_2pt.py: run reconstruction and 2pt measurements of Y1 data. Call ```python y1_data_2pt.py``` to create a queue of tasks (reconstruction, power spectrum and correlation function estimation). ```desipipe spawn -q y1_data_2pt --spawn``` to spawn a manager process that distributes these tasks on NERSC. See the `desipipe` cheat list to know how to monitor the tasks, stop / resume the queue, etc.: https://desipipe.readthedocs.io/en/latest/user/getting_started.html. For a simple example to run interactively, see ```y1_data_2pt_tools.py```. See https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/test/blinded/ for an example result.

- y1_abacus_2pt.py: same as y1_data_2pt, but for SecondGen Abacus mocks.

- y1_ez_2pt.py: same as y1_data_2pt, but for EZmocks.

- y1_covariance.py, y1_systematic_templates.py: build covariance matrices / systematic corrections and covariance matrices.

- y1_data_fits: run fits to Y1 data. Call ```python y1_data_fits.py``` to create a queue of tasks (computing emulators, and profiling). ```desipipe spawn -q y1_data_fits --spawn``` to spawn a manager process that distributes these tasks on NERSC. See the `desipipe` cheat list to know how to monitor the tasks, stop / resume the queue, etc.: https://desipipe.readthedocs.io/en/latest/user/getting_started.html. For a simple example to run interactively, see ```y1_data_fits_tools.py```. See https://data.desi.lbl.gov/desi/survey/catalogs/Y1/LSS/iron/LSScats/test/blinded/fits/ for an example result.

- y1_bao_cosmo.py: cosmological inference using Y1 BAO data. First create cosmological likelihoods; go to ```../desi_y1_cosmo_bindings``` and follow the README. Call ```python y1_bao_cosmo.py``` to create a queue of tasks (cosmological chains). ```desipipe spawn -q y1_bao_cosmo --spawn``` to spawn a manager process that distributes these tasks on NERSC. See the `desipipe` cheat list to know how to monitor the tasks, stop / resume the queue, etc.: https://desipipe.readthedocs.io/en/latest/user/getting_started.html. For a simple example to run interactively, see ```y1_bao_cosmo_tools.py```. See https://data.desi.lbl.gov/desi/science/cpe/y1kp7/bao/ for an example result.

- y1_fs_cosmo.py: same as y1_bao_cosmo.py, using Y1 Full Shape data.

- y1kp7_key_paper_1/: plots for the KP7 Key Paper 1.
- y1kp7_key_paper_2/: plots for the KP7 Key Paper 2.


# Papers

- KP3: https://fr.overleaf.com/read/kmrtxnmpwzvm#9ccf6a
- KP4: https://www.overleaf.com/4858529853vxvcdhxpzzgv
- KP7 paper 1: https://www.overleaf.com/4321587124vksstvcvkwpf#58fc52
