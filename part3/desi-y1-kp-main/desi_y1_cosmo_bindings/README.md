# DESI Y1 cosmological bindings

First, source the cosmodesi environment at NERSC:
```
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main  # source the environment
```
# BAO

There are two options: using desilike for bindings to Cobaya / CosmoSIS / MontePython, or export the measurements to .txt files, matching Cobaya conventions.

## first, copy files locally

```
python bindings_bao.py --todo copy
```
Copies BAO chains and Gaussian likelihoods to bao_data/.

## desilike bindings

```
python bindings_bao.py --todo bindings
```
Read the instruction below about inference at NERSC. If you rather wish to run inference within your own environment (Cobaya / CosmoSIS / MontePython and desilike, cosmoprimo required), you can also just download desi-y1-kp, and add it to your PYTHONPATH.

Then go to the cobaya / cosmosis / montepython directory (e.g. cobaya_bindings/) and follow instructions in the README.

## Cobaya-only likelihoods (no desilike dependency)
```
python bindings_bao.py --todo export
```
This will create a new directory with mean and covariance *.txt files in cobaya_likelihoods/bao_data, and likelihoods *.py files in cobaya_likelihoods/bao_likelihoods.

Read the instruction below about inference at NERSC. If you rather wish to run inference within your own environment (Cobaya required, of course), you can also just download desi-y1-kp, and add it to your PYTHONPATH.

Then go to cobaya_likelihoods/ and follow instructions in the README.

# Full-shape (+ BAO)

## first, copy files locally
```
python bindings_fs_bao.py --todo copy
```
Copies data vectors, covariances to data_fs_bao/.

## desilike bindings

```
python bindings_fs_bao.py --todo bindings
```
Read the instruction below about inference at NERSC. If you rather wish to run inference within your own environment (Cobaya / CosmoSIS / MontePython and desilike, cosmoprimo required), you can also just download desi-y1-kp, and add it to your PYTHONPATH.

Then go to the cobaya / cosmosis / montepython directory (e.g. cobaya_bindings/) and follow instructions in the README.

# Inference at NERSC

To run inference at NERSC:
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

## External cosmological inference codes

CosmoSIS and Cobaya are already installed in this environment.

If you wish to use MontePython, you should:
```
git clone https://github.com/brinckmann/montepython_public.git
```
Then, in the directory montepython_public, copy-paste default.conf.template to default.conf and update path['cosmo'], path['clik'] with:
```
import os
path['cosmo'] = os.getenv('CLASS_STD_DIR')
path['clik'] = os.path.join(os.getenv('PLANCK_SRC_DIR'), 'code', 'plc_3.0', 'plc-3.1')
```
### /!\ Important /!\

To use desilike bindings, export the path to the desi-y1-kp directory:
```
cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH
```
Otherwise, with Cobaya, you will get:
```
cobaya.component.ComponentNotFoundError: 'desi_y1_cosmo_bindings.cobaya_bindings.DESICompressedBAOALLLikelihood' could not be found.
```
with CosmoSIS:
```
Error was No module named 'desi_y1_cosmo_bindings'
```
with MontePython:
```
ModuleNotFoundError: No module named 'desi_y1_cosmo_bindings'
```

## Acknowledgments

- Kushal Lodha for early debugging and feedback, debugging cobaya-only likelihoods (cobaya itself and scripts)
- William Matthewson and Rodrigo Calderon for the implementation of SN likelihoods in cobaya
- Lanyang Yi and Gongbo-Zhao for DESY3 3x2pt likelihood