# DESI DR1 Full Shape and BAO clustering products

## Overview

This repository contains clustering measurements (power spectrum, correlation function, window matrix, covariance matrix) for the DESI DR1 Full Shape and BAO analyses.
These measurements are referenced in:
* [DESI 2024 II: Sample Definitions, Characteristics, and Two-point Clustering Statistics](https://ui.adsabs.harvard.edu/abs/2025JCAP...07..017A/abstract)
* [DESI 2024 III: Baryon Acoustic Oscillations from Galaxies and Quasars](https://ui.adsabs.harvard.edu/abs/2025JCAP...04..012A/abstract)
* [DESI 2024 V: Full-Shape Galaxy Clustering from Galaxies and Quasars](https://ui.adsabs.harvard.edu/abs/2024arXiv241112021D/abstract)

## Data Access

**Data URL:** https://data.desi.lbl.gov/public/dr1/vac/dr1/dr1-fs-bao-clustering-measurements

NERSC access:
```
/global/cfs/cdirs/desi/public/dr1/vac/dr1/dr1-fs-bao-clustering-measurements
```

## Documentation

The primary directory contains four folders:
- ```data/``` contains power spectrum and correlation function measurements, their window matrix and covariance matrix, and the packaged (data, window, covariance) to be used for cosmological inference, used in the Key Project Full Shape (combined with BAO) analysis.
- ```data_v1.2/``` contains the correlation function measurements, their window matrix and covariance matrix for the version v1.2 of the clustering catalogs, used in the Key Project BAO analysis. See Appendix B of [DESI 2024 II](https://ui.adsabs.harvard.edu/abs/2025JCAP...07..017A/abstract).
- ```EZmock/``` contains power spectrum, correlation function and BAO measurements for the 1000 EZ mocks.
- ```AbacusSummit/``` contains power spectrum, correlation function and BAO measurements for the 25 Abacus cutsky mocks.

We recommend to read the files with [lsstypes](https://github.com/adematti/lsstypes), though files can be read with any hdf5 reader. See section [File reading](#file-reading) below.

In this folder we provide a Python script ```create_fiducial_likelihood.py``` to create the final likelihoods (including scale cuts and systematic contributions) from the raw data, window matrix and covariance matrix.

### data

For the fiducial likelihood (including data, window matrix, covariance matrix), go directly to **data/likelihood**.

**data/spectrum**

```data/spectrum``` contains the power spectrum multipoles.
The power spectrum multipoles obtained with the FKP estimator read: ```spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```, with:
* tracer ```tracer```: 'BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'LRG+ELG_LOPnotqso' (for post-reconstruction correlation functions only), 'QSO'
* region ```region```: 'NGC', 'SGC', or 'GCcomb'. Combined power spectrum measurements 'GCcomb' are the average of 'NGC' and 'SGC' power spectra, weighted by their normalization factor.
* redshift range ```zrange```: (0.1, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.1), (1.1, 1.6), (0.8, 2.1)
* $\theta$-cut: ```thetacut```: '' (no $\theta$-cut) or '_thetacut0.05'. $\theta$-cut removes all pairs with angular separation < 0.05°, to mitigate fiber assignment effects. It requires a modified window matrix (file name ending with '_thetacut0.05'), which in its raw form contains large high-$k$ theory tails. Therefore, we also provide "rotated" measurements (data, window, covariance), for which the window matrix is more compact, at the price of marginalizing over some templates, called "rotation systematics" in the following.
The file naming convention for the corresponding window matrices is: ```window_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```.

The power spectrum multipoles corrected for the radial integral constraint (RIC) and angular mode removal (AMR) read ```spectrum-poles-corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```. The RIC is the result of the so-called 'shuffling' technique to assign data redshifts to the random catalogs. The AMR is caused by the overfitting of imaging systematic weights. Both RIC and AMR impact low $k$-modes; they were estimated from mocks (RIC from EZmocks, AMR from Abacus mocks) and compensated for in the power spectrum measurements. The window matrices are left unchanged.

The 'rotated' power spectrum multipoles read ```spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5```. Rotation is designed to make the window matrix related to the power spectrum measurements more compact. This rotation is only applied to measurements with $\theta$-cut, for which the original window matrix has large high-$k$ theory tails. The technique is described in [Pinon et al. 2024](https://ui.adsabs.harvard.edu/abs/2025JCAP...01..131P/abstract). The file format is similar to raw power spectrum measurements ```spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5```. Corresponding rotated window matrices are provided as ```window_spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5```, and the file format is similar to that of raw window matrices.

The 'rotated' and 'corrected' (for RIC, AMR) power spectrum multipoles read ```spectrum-poles-rotated-corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5```. They represent the fiducial power spectrum measurements.

**data/templates_spectrum**

```data/templates_spectrum``` contains power spectrum templates for systematics.
Templates for the radial integral constraint (RIC) read ```template_ric_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5``` and ```template_ric_spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5``` (without and with rotation, respectively). These templates were obtained by fiiting $\sum_{n \in \lbrace -5, -3, -2 \rbrace} a_n k^n$ to the difference of EZmock power spectra (with and without RIC).
Templates for the angular mode removal (AMR) read ```template_amr_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5``` and ```template_amr_spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5``` (without and with rotation, respectively).
The corrected power spectrum multipoles ```spectrum-poles-corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5``` and ```spectrum-poles-rotated-corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5``` are obtained by subtracting these RIC and AMR templates from ```spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5``` and ```spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5```, respectively (see the function ``get_observable`` in ```create_fiducial_likelihood.py```).

Templates for photometric systematics (non-zero for ELGs and QSOs only) are ```template_photo_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5``` and ```template_photo_spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5``` (without and with rotation, respectively). The corresponding systematic covariance matrix is computed with a $0$-centered Gaussian prior of standard deviation $0.2$.


**data/rotation**

```data/rotation``` contains the rotation matrices, ```rotation_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5```, to obtain the rotated power spectrum measurements, window matrix, and covariance matrix from the raw ones.
See the script ```create_fiducial_likelihood.py```.

**data/recsym**

```data/recsym/correlation``` contains post-reconstruction correlation functions. File naming convention is ```counts-recsym-smu_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5``` for the pair counts (DD, DS, SD, SS, RR, with the Landy-Szalay estimator). Correlation function multipoles are named ```correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5```. Corresponding (binning) window matrices are named ```window_correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5```.

**data/covariance**

```data/covariance/EZmock/ffa``` contains the covariance as estimated from the raw power spectra and post-reconstruction correlation function of EZmocks (see See section [EZmock](#ezmock)).
The raw power spectrum covariance reads ```covariance_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```.
The joint (raw power spectrum, post-reconstruction correlation) covariance reads ```covariance_spectrum-poles+correlation-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```
The joint (raw power spectrum, post-reconstruction BAO) covariance reads ```covariance_spectrum-poles+bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```.

File names with **spectrum-poles-rotated** (e.g. ```covariance_spectrum-poles-rotated+bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```) correspond to covariance matrices with the rotated power spectrum.

```covariance/RascalC``` contains semi-analytic covariance matrices for the post-reconstruction correlation function measurements. File naming convention is ```covariance_correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5```.

```covariance/syst``` contains the systematic covariance matrix accounting for the systematic shifts due to galaxy-halo connexion (modeled as HOD): ```covariance_hod_spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```, see [Findlay et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv241112021D/abstract) for details.


**data/likelihood**

```data/likelihood``` contains the set of (observable, window, covariance), with systematic contributions, with the fiducial Key Project scale cuts, including the post-reconstruction BAO part. We recommend these files for cosmological inference.

The $\theta$-cut power-spectrum-only likelihood reads ```likelihood_spectrum-poles_syst-rotation-hod-photo_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5```.
The joint power spectrum - post-reconstruction BAO likelihood reads ```likelihood_spectrum-poles+bao-recon_syst-rotation-hod-photo_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5```.
These likelihoods, which we used for the cosmological inference, include correcting factors in the covariance matrix (Table 6 of [DESI 2024 II](https://ui.adsabs.harvard.edu/abs/2025JCAP...07..017A/abstract) and Sections 5.7 and 5.8 of [DESI 2024 V](https://ui.adsabs.harvard.edu/abs/2024arXiv241112021D/abstract)), and systematic contributions for:
* galaxy-halo connexion (```hod```)
* residual photometric systematics (```photo```)
* rotation of the window matrix (```rotation```): analytic marginalization over the parameter $s$ of Eq. 5.4 of [Pinon et al. 2024](https://ui.adsabs.harvard.edu/abs/2025JCAP...01..131P/abstract).
See the script ```create_fiducial_likelihood.py``` for instructions to reproduce these likelihood files given the raw power spectrum (and post-reconstruction BAO) measurements.
The last two contributions are mostly off-diagonal, and increase the size of the diagonal of the covariance, which results in "odd-looking" error bars. Therefore, we also provide likelihoods without analytic marginalization for residual photometric systematics (```photo```) and rotation of the window matrix (```rotation```). The corresponding files are named ```likelihood_spectrum-poles_syst-hod_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5``` and ```likelihood_spectrum-poles+bao-recon_syst-hod_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5``` for the power spectrum-only and joint power spectrum - post-reconstruction BAO likelihoods, respectively. These files however include two new theory components: 'photo' and 'rotation', to marginalize over in the inference, with prior (diagonal) **covariance** given by 'prior_variance'.

Post-reconstruction BAO-only likelihoods are provided as ```likelihood_bao-recon_syst_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5``` (files with 'stat-only' instead of 'syst' contain no systematic uncertainties). The BAO Ly-$\alpha$ likelihood (see [DESI 2024 IV](https://ui.adsabs.harvard.edu/abs/2025JCAP...01..124A)) is also provided for completeness.

ShapeFit likelihoods (see e.g. [Brieden et al. 2021](https://ui.adsabs.harvard.edu/abs/2021JCAP...12..054B)) obtained by fitting the joint (rotated) power spectrum and post-reconstruction BAO are named ```likelihood_shapefit_spectrum-poles-rotated+bao-recon_syst-rotation-hod-photo_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5```.

Likelihoods for the post-reconstruction correlation function are named ```likelihood_correlation-recon-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5```.

```data/likelihood/cobaya_likelihoods``` contains a Cobaya implementation of the full shape likelihoods; see README.md in that directory.

### data_v1.2

These files were used for the BAO cosmological inference. We provide them for completeness, though we recommend using the v1.5 (default) version of the files.

```data/recsym/correlation``` same structure as above.
```data/covariance/RascalC``` same structure as above.
```data/likelihood``` same structure as above.


### EZmock

```EZmock/ffa/spectrum``` contains power spectrum measurements ```spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}_{imock:d}.h5``` for cutsky EZmocks with fast fiber assignment (FFA). They can directly be used to compute the EZmock-based covariance matrices ```data/covariance/EZmock/ffa/covariance_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```. Format is the same as for the data files. Corresponding raw window matrices are provided, ```window_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```.
```EZmock/ffa/recsym/correlation``` contains post-reconstruction correlation function measurements ```correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{imock:d}.h5```. Format is the same as for the data files.
```EZmock/ffa/recsym/bao``` contains post-reconstruction BAO measurements (obtained from the correlation function) ```bao-recsym_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{imock:d}.h5```. They can directly be used to compute the EZmock-based covariance matrices ```covariance/EZmock/ffa/covariance_spectrum-poles+bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```.

Note: EZmocks corresponding to the BGS_BRIGHT-21.5 and ELG_LOPnotqso samples are named BGS_BRIGHT and ELG_LOP, respectively.


### AbacusSummit

```AbacusSummit/complete/spectrum``` contains power spectrum measurements ```spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}_{imock:d}.h5``` for Abacus SecondGen complete cutsky mocks. Format is the same as for the data files. Corresponding raw window matrices are provided, ```window_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5```.
Similar files are provided for mocks with fast fiber assignment (FFA) ```AbacusSummit/ffa/spectrum``` and alt-MTL ```AbacusSummit/altmtl/spectrum```.
```AbacusSummit/complete/recsym/correlation``` contains post-reconstruction correlation function measurements ```correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{imock:d}.h5```. Format is the same as for the data files. Similar files are provided for mocks with fast fiber assignment (FFA) ```AbacusSummit/ffa/spectrum``` and alt-MTL ```AbacusSummit/altmtl/spectrum```.

Note: "complete" and "ffa" AbacusSummit mocks corresponding to the ELG_LOPnotqso sample are named ELG_LOP.


## File reading

### With **lsstypes**

[lsstypes](https://github.com/adematti/lsstypes) version used is 1.0.0.

**spectrum**

```python
import lsstypes as types

data = types.read('data/spectrum/spectrum-poles_LRG_GCcomb_z0.8-1.1.h5')
# poles
print(data.ells)
# k-coordinates [h/Mpc]
data.get(ells=0).coords('k')
# k-edges
data.get(ells=0).edges('k')  # dk = 0.001 h/Mpc in these files
# Shotnoise-subtracted power spectrum multipoles [(Mpc/h)^3]
poles = [data.get(ells=ell).value() for ell in data.ells]
# Other quantities are:
data.get(0).values()
# Shot noise is:
data.get(0).values('shotnoise')
# Effective measurement redshift is:
data.attrs['zeff']

# "corrected" power spectra (RIC and AMR)
data = types.read('data/spectrum/spectrum-poles-corrected_LRG_GCcomb_z0.8-1.1.h5')
data.get(0).edges('k')  # 0.02 < k [h/Mpc] < 0.4, dk = 0.005 h/Mpc in these files

# "rotated" and "corrected" theta-cut power spectra
data = types.read('data/spectrum/spectrum-poles-rotated-corrected_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5')
data.get(0).edges('k')  # 0.02 < k [h/Mpc] < 0.2, dk = 0.005 h/Mpc in these files

# Window matrix, to relate the theory galaxy power spectrum to the observed power spectrum
window = types.read('data/spectrum/window_spectrum-poles-rotated_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5')
# For the following observable
print(window.observable)
window.observable.get(0).edges('k')  # 0.02 < k [h/Mpc] < 0.2, dk = 0.005 h/Mpc
# For the following theory: power spectrum multipoles [0, 2, 4]
print(window.theory)
# Theory multipoles
window.theory.ells  # [0, 2, 4]
# Theory k
window.theory.get(0).coords('k')
# Window matrix value
window.value()  # numpy array
assert window.value().shape == (window.observable.size, window.theory.size)
```

**templates_spectrum**

Power spectrum templates are in the same format as power spectrum measurements.
```python
template = types.read('data/templates_spectrum/template_ric_spectrum-poles_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5')
# poles
print(data.ells)
# k-coordinates [h/Mpc]
data.get(ells=0).coords('k')
# k-edges
data.get(ells=0).edges('k')  # dk = 0.005 h/Mpc in these files
```

**rotation**

```python
rotation = types.read('data/rotation/rotation_spectrum-poles_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5')
# Dictionary with keys "M", "mo", "mt", "s", which match eq. 5.4 - 5.6 of [Pinon et al. 2024](https://ui.adsabs.harvard.edu/abs/2025JCAP...01..131P/abstract)
# See the script create_fiducial_likelihood.py to see how to apply the rotation to power spectrum measurements, window matrix, covariance matrix
print(list(rotation.keys()))
```

**correlation**

```python
# Pair counts
data = types.read('data/recsym/correlation/counts-recsym-smu_LRG_GCcomb_z0.8-1.1.h5')
print(data)
# Normalized DD counts
data.get('DD').value()
# s-edges [Mpc/h], mu-edges
data.edges('s'), data.edges('mu')    # 0 < s [Mpc/h] < 200, ds = 1 Mpc/h

# Correlation function multipoles
data = types.read('data/recsym/correlation/correlation-recsym-poles_LRG_GCcomb_z0.8-1.1.h5')
print(data)
# s-coordinates [Mpc/h]
data.get(0).coords('s')  # 0 < s [Mpc/h] < 200, ds = 4 Mpc/h
# Monopole
data.get(0).value()

# (rebinning) window matrix
window = types.read('data/recsym/correlation/window_correlation-recsym-poles_LRG_GCcomb_z0.8-1.1.h5')
# For the following observable
print(window.observable)
window.observable.get(0).edges('s')  # 0 < s [Mpc/h] < 200, ds = 4 Mpc/h
# For the following theory: power spectrum multipoles [0, 2, 4]
print(window.theory)
# Theory multipoles
window.theory.ells  # [0, 2, 4]
# Theory s
window.theory.get(0).coords('s')
# Window matrix value
window.value()  # numpy array
assert window.value().shape == (window.observable.size, window.theory.size)
```

**covariance**

```python
# EZmock-based covariance matrix for rotated theta-cut power spectrum
covariance = types.read('data/covariance/EZmock/covariance_spectrum-poles-rotated_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5')
# For the observable
print(covariance.observable)
# With binning
covariance.observable.get(ells=0).edges('k')  # 0 < k [h/Mpc] < 0.4, dk = 0.005 h/Mpc
# Covariance matrix value
covariance.value()  # numpy array
assert covariance.value().shape == (covariance.observable.size,) * 2
covariance_spectrum = covariance

# Joint [rotated theta-cut power spectrum, BAO parameters] covariance matrix
covariance = types.read('data/covariance/EZmock/covariance_spectrum-poles-rotated+bao-recon_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5')
print(covariance.observable)
# With binning
covariance.observable.get(observables='spectrum', ells=0).edges('k')
# Covariance matrix value
covariance.value()  # numpy array
assert covariance.value().shape == (covariance.observable.size,) * 2
# To restrict to one observable
subcov = covariance.at.observable.get('spectrum')
assert np.allclose(subcov.value(), covariance_spectrum.value())

# Post-reconstruction correlation covariance matrix
covariance = types.read('data/covariance/RascalC/covariance_correlation-recsym-poles_LRG_GCcomb_z0.8-1.1.h5')
print(covariance.observable)
# With binning
covariance.observable.get(ells=0).edges('s')  # 0 < s [Mpc/h] < 200, ds = 4 Mpc/h
```


**likelihood**

```python
likelihood = types.read('data/likelihood/likelihood_spectrum-poles-rotated+bao-recon_syst-hod_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5')
likelihood.observable.value()  # data array
# For the following observables
print(likelihood.observable)
likelihood.covariance.value()  # covariance matrix array
likelihood.window.value()  # window matrix array
window = likelihood.window
# For the following input theory
print(window.theory)
# qpar, qper: BAO (DH/rd) / (DH/rd)_fid, (DM/rd) / (DM/rd)_fid using the DESI fiducial cosmology (Planck2018)
baorecon = window.theory.get('baorecon')
print(baorecon)
# Fiducial values are stored here:
DM_over_rd_fid = baorecon.get('qper').attrs['DM_over_rd_fid']
DH_over_rd_fid = baorecon.get('qpar').attrs['DH_over_rd_fid']
# And can be obtained easily with:
from cosmoprimo.fiducial import DESI
from cosmoprimo import constants, Cosmology
fiducial = DESI()
zeff = baorecon.get('qpar').attrs['zeff']
DM_over_rd_fid = fiducial.comoving_angular_distance(zeff) / fiducial.rs_drag
DH_over_rd_fid = (constants.c / 1e3) / (100. * fiducial.efunc(zeff)) / fiducial.rs_drag
DV_over_rd_fid = DM_over_rd_fid**(2. / 3.) * DH_over_rd_fid**(1. / 3.) * zeff**(1. / 3.)

# Likelihood ($\chi^2$) can then be computed as:

def get_theory_spectrum(k, ells, zeff, shotnoise):
    # Assuming (obviously incorrect) input galaxy power spectrum = 0
    spectrum = []
    for ell in ells:  # loop over multipoles
        spectrum.append(np.zeros_like(k))
    return spectrum

def get_theory_bao(zeff):
    cosmo = Cosmology(Omega_m=0.3, engine='camb')  # and all other cosmological parameters
    DM_over_rd = cosmo.comoving_angular_distance(zeff) / cosmo.rs_drag
    DH_over_rd = (constants.c / 1e3) / (100. * cosmo.efunc(zeff)) / cosmo.rs_drag
    qpar = DH_over_rd / DH_over_rd_fid
    qper = DM_over_rd / DM_over_rd_fid
    return np.atleast_1d(qpar), np.atleast_1d(qper)

spectrum = get_theory_spectrum(k=window.theory.get('spectrum').get(0).coords('k'), # here, same for all multipoles
                               ells=window.theory.get('spectrum').ells, # here, [0, 2, 4]
                               zeff=likelihood.observable.get('spectrum').attrs['zeff'],
                               shotnoise=np.mean(likelihood.observable.get('spectrum').get(0).values('shotnoise')))
photo = np.array([0.])  # to sample with prior window.theory.get('photo').attrs['prior_variance']
rotation = np.array([0., 0.])  # to sample with prior window.theory.get('photo').attrs['prior_variance']
qpar, qper = get_theory_bao(zeff)
theory = np.concatenate(spectrum + [photo, rotation] + [qpar, qper])
likelihood.chi2(theory)
# Which is just:
# delta.dot(invcov).dot(delta) with invcov the inverse of likelihood.covariance.value()
# and delta = likelihood.observable.value() - window.value().dot(theory)

# Using the analytically-marginalized version, you can forget about nuisance parameters
# for "photo" and "rotation"
likelihood = types.read('data/likelihood/likelihood_spectrum-poles-rotated+bao-recon_syst-rotation-hod-photo_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5')
theory = np.concatenate(spectrum + [qpar, qper])
likelihood.chi2(theory)

# For BGS and QSO, measured BAO is isotropic
likelihood = types.read('data/likelihood/likelihood_spectrum-poles-rotated+bao-recon_syst-rotation-hod-photo_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_thetacut0.05.h5')

def get_theory_bao_iso(zeff):
    qpar, qper = get_theory_bao(zeff)
    return qpar**(1. / 3.) * qper**(2. / 3.)

qiso = get_theory_bao_iso(likelihood.observable.get('baorecon').get('qiso').attrs['zeff'])
theory = np.concatenate(spectrum + [qiso])
likelihood.chi2(theory)

# Likelihood without BAO
likelihood = types.read('data/likelihood/likelihood_spectrum-poles-rotated_syst-rotation-hod-photo_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_thetacut0.05.h5')
print(likelihood.window.observable)
print(likelihood.window.theory)
print(likelihood.covariance.observable)
theory = np.concatenate(spectrum)
likelihood.chi2(theory)

# ShapeFit likelihood
likelihood = types.read('data/likelihood/likelihood_shapefit_spectrum-poles-rotated+bao-recon_syst-rotation-hod-photo_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5')
# ['qiso', 'qap', 'df', 'dm']
# Where 'qiso', 'qap' are BAO parameters, 'df' is the growth rate relative to the fiducial value and 'dm' the tilt of the power spectrum
print(likelihood.observable.get('shapefit').parameters)

# Post-reconstruction correlation function likelihood
likelihood = types.read('data/likelihood/likelihood_correlation-recon-poles_LRG_GCcomb_z0.8-1.1.h5')
print(likelihood.observable)
```


### With h5py (no other dependency)


```python
# Data is organized hierarchically
# Let's explore some files!
import h5py

def print_type(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"dataset: {name}")

# Power spectrum multipoles
with h5py.File('data/spectrum/spectrum-poles_LRG_GCcomb_z0.8-1.1.h5', 'r') as data:
    data.visititems(print_type)
    # Multipoles
    ells = [int(ell) for ell in data['labels_values']]
    # To access the (shotnoise-subtracted) power spectrum monopole, k-coordinates, k-edges
    data['0/value'], data['0/k'], data['0/k_edges']
    # Similarly for the quadrupole
    data['2/value'], data['2/k'], data['4/k_edges']

# The power spectrum window matrix
with h5py.File('data/spectrum/window_spectrum-poles-rotated_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5', 'r') as data:
    data.visititems(print_type)
    # To access the window matrix value
    data['value']
    # The corresponding theory k-values
    ells = [int(ell) for ell in data['theory/labels_values']]
    data['theory/0/k'], data['theory/2/k'], data['theory/4/k']
    # And the observed k-values
    data['observable/0/k'], data['observable/2/k'], data['observable/4/k']

# Correlation function multipoles are very similar
with h5py.File('data/recsym/correlation/correlation-recsym-poles_LRG_GCcomb_z0.8-1.1.h5', 'r') as data:
    data.visititems(print_type)
    # To access the correlation function monopole, s-coordinates, s-edges
    data['0/value'], data['0/s'], data['0/s_edges']
    # Similarly for the quadrupole
    data['2/value'], data['2/s'], data['4/s_edges']

# The correlation function window matrix follows the same scheme
with h5py.File('data/recsym/correlation/window_correlation-recsym-poles_LRG_GCcomb_z0.8-1.1.h5', 'r') as data:
    data.visititems(print_type)
    # To access the window matrix value
    data['value']
    # The corresponding theory s-values
    ells = [int(ell) for ell in data['theory/labels_values']]
    data['theory/0/s'], data['theory/2/s'], data['theory/4/s']
    # And the observed s-values
    data['observable/0/s'], data['observable/2/s'], data['observable/4/s']

# Covariance
with h5py.File('data/covariance/EZmock/covariance_spectrum-poles-rotated+bao-recon_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5', 'r') as data:
    data.visititems(print_type)
    # To access the covariance matrix value
    data['value']
    # The corresponding observables
    # observable_names is 'spectrum', 'baorecon' *IN THIS ORDER IN THE COVARIANCE*
    observable_names = list(data['observable/labels_values'])
    # Monpole, quadrupole, hecadecapole k's
    data['observable/spectrum/0/k'], data['observable/spectrum/2/k'], data['observable/spectrum/4/k']
    # BAO parameters
    data['observable/baorecon/qpar'], data['observable/baorecon/qper']

# Likelihood includes data, window, covariance (including systematics) in the same files, with fiducial scale cuts
with h5py.File('data/likelihood/likelihood_spectrum-poles-rotated+bao-recon_syst-hod_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5') as data:
    # observable_names is 'spectrum', 'baorecon' *IN THIS ORDER*
    observable_names = list(data['observable/labels_values'])
    # To access data power spectrum multipoles
    data['observable/spectrum/0/value'], data['observable/spectrum/2/value']
    # Mean shot noise
    np.mean(data['observable/spectrum/0/num_shotnoise'][...] / data['observable/spectrum/0/norm'][...])
    # Effective redshift
    data['observable/spectrum'].attrs['zeff']
    # And post-reconstruction BAO
    data['observable/baorecon/qpar'], data['observable/baorecon/qper']
    # Joint covariance matrix
    data['covariance/value']
    # And *joint* window matrix
    data['window/value']
    # For theory *IN THIS ORDER* (obtained with data['window/theory/labels_values']):
    data['window/theory/spectrum/0/k'], data['window/theory/spectrum/2/k'], data['window/theory/spectrum/4/k'], data['window/theory/photo'], data['window/theory/rotation'], data['window/theory/baorecon/qpar'], data['window/theory/baorecon/qper']
```


## Contact
Contact [Ashley J. Ross](mailto:ashley.jacob.ross@gmail.com) and [Arnaud de Mattia](mailto:arnaud.de-mattia@cea.fr) for questions about this catalog.