# desi-y1-kp

This repository aims at gathering scripts to run the end-to-end standard analysis using DESI Y1 data, from the catalogs to the cosmological constraints.
This covers KP3 (2-pt measurements), KP4 (BAO), KP5 (full shape) and KP7 (cosmological constraints, using KP6 BAO compressed measurements).
See ``y1_bao_cosmo.py`` for cosmological inference for the Y1KP7 paper 1.
See ``y1_fs_cosmo.py`` for cosmological inference for the Y1KP7 paper 2.

Note: If you want to run cosmological inference using Y1 BAO and/or FS data, go to desi_y1_cosmo_bindings.

## Environment

```
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main  # source the environment
# You may want to have it in the jupyter-kernel for plots
${COSMODESIMODULES}/install_jupyter_kernel.sh main  # this to be done once
```
You may already have the above kernel (corresponding to the standard GQC environment) installed.
In this case, you can delete it:
```
rm -rf $HOME/.local/share/jupyter/kernels/cosmodesi-main
```
and rerun:
```
${COSMODESIMODULES}/install_jupyter_kernel.sh main
```
Note that you may need to restart (close and reopen) your notebooks for the changes to propagate.

## Structure

- desi_y1_files: file manager
- desi_y1_plotting: plotting routines
- desi_y1_cosmo_bindings: bindings for cosmological inference with DESI Y1 data
- scripts: run the standard analysis

## Installation

First:
```
git clone https://github.com/cosmodesi/desi-y1-kp.git
```
To install the code:
```
python -m pip install . --user
```
Or in development mode (any change to Python code will take place immediately):
```
python -m pip install -e . --user
```