# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cosmodesi-main
#     language: python
#     name: cosmodesi-main
# ---

# +
##mpiexec -n 12 python simple_script_test.py --mpi
#jupyter nbconvert --to python like.ipynb


import sys, os, shutil, re
sys.path.append('../')

import time
import emcee
import numpy as np

from schwimmbad import MPIPool
from datetime import datetime
from mike_data_tools import *
from cosmo import *
from likelihood_tools import *


# +
def get_run_config():
    return dict(

        kmin=dict(
            P0=0.02,P2=0.02,P4=0.02,
            B0=0.02,B2=0.02,
        ),

        kmax=dict(
            P0=0.301,
            P2=0.301,
            P4=0.02,
            B0=0.12,
            B2=0.08,
        ),

        use_poles=dict(
            P0=True,
            P2=True,
            P4=False,
            B0=True,
            B2=True,
        ),

        tracer="LRG2",
        z=0.8,
        z_str="z0.800",

        # Model choices
        model="FOLPSD",           # FOLPSD, TNS, EFT
        bias_scheme="folps",    # folps, classpt, priordoc
        damping="lor",            # None, exp, lor
        kernels="fk",             # fk, EdS
        BispBase="Sugiyama",


        # Parameter toggles
        b2coev=False,
        bscoev=False,
        b3coev=True,
        ns_free=False,
        mnu_free=False,
        N_eff_free=False,
        N_ur_free=False,
        w0_free=False,
        wa_free=False,

        # Runtime / IO
        backend_reset=False,
        output_dir="./chains_LRG_bias-folps/",
        log_file="./chains/_logs.txt",
        chains_ext=".h5",
        DATA_DIR='./data/',
        
        A_full=True,
        Nfftlog=128,
    )


cfg = derive_run_flags(get_run_config())
params_fiducial_values=fiducial_values(cfg['tracer'],bias_scheme=cfg['bias_scheme'])


parameters = {
    
    "h": {
        "free": True,
        "prior": ("flat", 0.5, 0.9),
        "latex": r"$h$",
    },

    "omega_cdm": {
        "free": True,
        "prior": ("flat", 0.05, 0.2),
        "latex": r"$\omega_{cdm}$",
    },

    "omega_b": {
        "free": True,
        "prior": ("flat+gauss", 0.017, 0.027, params_fiducial_values['omega_b'], BBN_prior(N_eff_free=cfg['N_eff_free'])['stddev']),
        "latex": r"$\omega_b$",
    },

    "logAs": {
        "free": True,
        "prior": ("flat", 2.0, 4.0),
        "latex": r"$\ln(10^{10}A_s)$",
    },

    "ns": {
        "free": cfg['ns_free'],
        "prior": ("flat+gauss", 1e-5, 10, params_fiducial_values['ns'], 0.042),  #ns10_sigma=0.042
        "latex": r"$n_s$",
    },

    "omega_ncdm": {
        "free": cfg['mnu_free'],
        "prior": ("flat", 0, 0.02),
        "latex": r"$\omega_{ncdm}$",
    },

    "N_eff": {
        "free": cfg['N_eff_free'],
        "prior": ("flat", 1e-4, 8),
        "latex": r"$N_{\mathrm{eff}}$",
    },

    "N_ur": {
        "free": cfg['N_ur_free'],
        "prior": ("flat", 1e-4, 8),
        "latex": r"$N_{\mathrm{ur}}$",
    },

    "w0": {
        "free": cfg['w0_free'],
        "prior": ("flat", -3, 1),
        "latex": r"$w_0$",
    },

    "wa": {
        "free": cfg['wa_free'],
        "prior": ("flat", -3, 1),
        "latex": r"$w_a$",
    },

    "b1": {
        "free": True,
        "prior": ("flat", 1e-5, 10),
        "latex": r"$b_1$",
    },

    "b2": {
        "free": not cfg['b2coev'],
        "prior": ("flat", -50, 50),
        "latex": r"$b_2$",
    },

    "bs": {
        "free": not cfg['bscoev'],
        "prior": ("flat", -50, 50),
        "latex": r"$b_s$",
    },

    "b3": {
        "free": not cfg['b3coev'],
        "prior": ("flat", -50, 50),
        "latex": r"$b_3$",
    },

    "c1": {
        "free": cfg['BispBool'],
        "prior": ("flat+gauss", -2000, 2000, 66.6, 66.6 * 4.0),
        "latex": r"$c_1$",
    },

    "c2": {
        "free": cfg['BispBool'],
        "prior": ("flat+gauss", -2000, 2000, 0.0, 4.0),
        "latex": r"$c_2$",
    },

    "Pshot": {
        "free": cfg['BispBool'],
        "prior": ("flat+gauss", -50000, 50000, 0.0, n_bar(cfg['tracer']) * 4.0),
        "latex": r"$P_{shot}$",
    },

    "Bshot": {
        "free": cfg['BispBool'],
        "prior": ("flat+gauss", -50000, 50000, 0.0, n_bar(cfg['tracer']) * 4.0),
        "latex": r"$B_{shot}$",
    },


    "X_FoG_pk": {
        "free": cfg['model'] != "EFT",
        "prior": ("flat", 0.0, 15.0),
        "latex": r"$X_p$",
    },

    "X_FoG_bk": {
        "free": (cfg['BispBool'] and cfg['model'] != "EFT"),
        "prior": ("flat", 0.0, 15.0),
        "latex": r"$X_b$",
    },
}

params_derived = ['sigma8', 'Omega_m']
params_derived_latex = [r'$\sigma_8$', r'$\Omega_m$']

# -



# +
params_sampled=params_sampled_fn(parameters,chatty=True)
mcmc_file = os.path.realpath(__file__) if is_python_script() else "___.ipynb"
chains_filename = logging(cfg,parameters,params_sampled,
                                     params_derived,params_derived_latex,
                                     mcmc_file)

data_d=load_and_prepare_abacus2ndgen_data(cfg,showplot=False)

fiducial_dist=distances(params_fiducial_values,cfg['z'],chatty=True)

calculate_M_matrices(configuration = cfg)

out=FolpsD(fiducial=params_fiducial_values, 
           configuration=cfg, 
           data_dictionary=data_d,
           chatty=True)


# -



# # %%time 
start0 = np.array([params_fiducial_values[p] for p in params_sampled])
out=log_probability(start0,parameters=parameters,
                   configuration=cfg,
                   derived_params=params_derived,
                   data_dictionary=data_d,
                   fiducial=params_fiducial_values,
                  )
print(out)


# +
def log_probability_short(theta,parameters=parameters,
                   configuration=cfg,
                   derived_params=params_derived,
                   data_dictionary=data_d,
                   fiducial=params_fiducial_values):
    
    return log_probability(theta,
                   parameters=parameters,
                   configuration=configuration,
                   derived_params=derived_params,
                   data_dictionary=data_dictionary,
                   fiducial=fiducial,
                  )

start0 = np.array([params_fiducial_values[p] for p in params_sampled])
import time
start = time.perf_counter()
for i in range(10):
    out = log_probability_short(start0)
end = time.perf_counter()
print(f"Ten likehood evaluations take {end - start:.3f} seconds")
print(out)    

# +
ndim = len(start0) # Number of parameters/dimensions (e.g. m and c)
nwalkers = 3 * ndim # Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = 150000 # Number of steps/iterations. (max number)

start = np.array([start0 + 1e-3*np.random.rand(ndim) for i in range(nwalkers)])

 
backend = emcee.backends.HDFBackend(chains_filename)

##backend.reset(nwalkers, ndim)
###sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_short, backend=backend)


max_n = nsteps

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf



with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
        
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_short, backend=backend, 
                                    pool=pool)
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(start, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
            
            
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        
        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.005)
        if converged:
            break
        old_tau = tau



np.savetxt('autocorr_index'+str(index)+'.dat', np.transpose([autocorr[:index]]), 
           header = 'index ='+str(index)+',  mean_autocorr')


print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)
    )
)

# +

# import emcee

# def run_emcee(
#     log_prob_fn,
#     start0,
#     chains_filename,
#     nsteps=150_000,
#     nwalkers_factor=3,
#     min_steps_factor=50,
#     check_every=100,
#     tau_rel_tol=0.005,
#     progress=True,
# ):
#     """
#     Run an emcee EnsembleSampler with MPI, HDF backend, and autocorrelation
#     convergence stopping.

#     Parameters
#     ----------
#     log_prob_fn : callable
#         Log-probability function compatible with emcee.
#     start0 : array_like
#         Fiducial starting point (ndim,).
#     chains_filename : str
#         Path to HDF backend file.
#     nsteps : int
#         Maximum number of MCMC steps.
#     nwalkers_factor : int
#         Number of walkers = nwalkers_factor * ndim.
#         Recommended: 3 or 4 for FS likelihoods.
#     min_steps_factor : int
#         Minimum steps = min_steps_factor * ndim before convergence is checked.
#     check_every : int
#         Check convergence every this many steps.
#     tau_rel_tol : float
#         Relative tolerance for autocorr time convergence.
#     progress : bool
#         Show progress bar.

#     Returns
#     -------
#     sampler : emcee.EnsembleSampler
#         The sampler object (on master rank).
#     """

#     start0 = np.asarray(start0)
#     ndim = len(start0)

#     nwalkers = nwalkers_factor * ndim
#     min_steps = min_steps_factor * ndim

#     # Initial walker positions
#     start = start0 + 1e-3 * np.random.randn(nwalkers, ndim)

#     backend = HDFBackend(chains_filename)
#     # backend.reset(nwalkers, ndim)  # uncomment to overwrite existing chains

#     max_n = nsteps
#     autocorr = np.empty(max_n // check_every)
#     index = 0
#     old_tau = np.inf

#     with MPIPool() as pool:
#         if not pool.is_master():
#             pool.wait()
#             sys.exit(0)

#         sampler = emcee.EnsembleSampler(
#             nwalkers,
#             ndim,
#             log_prob_fn,
#             backend=backend,
#             pool=pool,
#         )

#         for sample in sampler.sample(start, iterations=max_n, progress=progress):

#             it = sampler.iteration
#             if it % check_every != 0:
#                 continue

#             # Do not trust tau too early
#             if it < min_steps:
#                 continue

#             try:
#                 tau = sampler.get_autocorr_time(tol=0)
#             except emcee.autocorr.AutocorrError:
#                 continue

#             # Use the slowest mode
#             tau_max = np.max(tau)
#             autocorr[index] = tau_max
#             index += 1

#             converged = np.all(tau * 100 < it)
#             converged &= np.all(np.abs(old_tau - tau) / tau < tau_rel_tol)

#             if converged:
#                 break

#             old_tau = tau

#     # Save autocorrelation history (master only)
#     if index > 0:
#         np.savetxt(
#             f"autocorr_index{index}.dat",
#             autocorr[:index],
#             header=f"index={index}, max_autocorr",
#         )

#     # Final diagnostics
#     print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

#     return sampler

# sampler = run_emcee(
#     log_prob_fn=log_probability_short,
#     start0=start0,
#     chains_filename=chains_filename,
#     nsteps=150_000,
# )

# -




