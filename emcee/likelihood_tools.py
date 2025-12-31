import numpy as np
import sys, os, shutil, re
import time
#from mike_data_tools import *
import numpy as np
sys.path.append('../')
import folps as folpsD


##configuration
    
def derive_run_flags(cfg):
    cfg = cfg.copy()

    cfg["BispBool"] = (
        cfg["use_poles"]["B0"] or cfg["use_poles"]["B2"]
    )

    cfg["use_TNS_model"] = (cfg["model"] == "TNS")

    if cfg["model"] == "EFT":
        cfg["damping"] = None

    if cfg["model"] not in ["FOLPSD", "TNS", "EFT"]:
        raise ValueError("model should be 'FOLPSD', 'TNS' or 'EFT'")

    if cfg["damping"] not in ["lor", "eft", "vdg", None]:
        raise ValueError("damping should be 'lor', 'eft', 'vdg' or None")

    if cfg["bias_scheme"]==["priorsdoc","DR2"]:
        cfg["bias_scheme"]="priordoc"

    if cfg["bias_scheme"] not in ["folps", "classpt", "priordoc"]:
        raise ValueError("bias scheme should be 'folps', 'classpt', 'priordoc'")

    if cfg["tracer"] not in ["LRG2","LRG1","QSO"]:
        raise ValueError("Tracer should be 'LRG1','LRG2','QSO'")

    return cfg


                
# likelihood functions    
    
def params_sampled_fn(parameters,chatty=False):
    params_sampled= [k for k, v in parameters.items() if v["free"]]
    if chatty:
        print(f'{len(params_sampled)} params_sampled:', params_sampled)
    return params_sampled

    
def params_sampled_latex(parameters):
    # Use list comprehension to preserve order
    return [
        v["latex"]
        for k, v in parameters.items()
        if v.get("free", False) and "latex" in v
    ]    

def params_sampled_and_priors_fn(parameters):
    """
    Extract prior information for parameters that are free.
    
    Args:
        parameters: Dictionary of parameters with 'free' and 'prior' keys
        
    Returns:
        Dictionary with only free parameters and their priors
    """
    free_priors = {}
    
    for param_name, param_dict in parameters.items():
        # Check if parameter is free
        if param_dict.get("free", False):
            # Extract just the prior information
            free_priors[param_name] = param_dict["prior"]
    
    return free_priors



def build_params(theta, params_sampled):
    """
    Build a parameter dictionary with only active parameters.
    
    Args:
        theta: Array of parameter values for active parameters
        params_sampled_fn: List of parameter names that are active/free
        fiducial_values: Full dictionary of all parameter values
        
    Returns:
        Dictionary with only active parameters (with their new values from theta)
    """
    # Create a new dictionary with only active parameters
    params = {}
    
    # Update with values from theta for active parameters
    for param_name, param_value in zip(params_sampled, theta):
        params[param_name] = param_value
    
    return params


def log_likelihood(theta,
                   parameters=None,
                   configuration=None,
                   derived_params=None,
                   data_dictionary=None,
                   fiducial=None,
                  ):
    
    from cosmo import FolpsD
    
    if fiducial is None:
        fiducial = {}
    if configuration is None:
        configuration = {}
    if parameters is None:
        parameters = {}
    if derived_params is None:
        derived_params = {}
    if data_dictionary is None:
        data_dictionary = {}
    
    params_sampled=params_sampled_fn(parameters)
    
    params = build_params(theta,params_sampled)

    md = FolpsD(cosmo_params=params,
                fiducial=fiducial, 
                configuration=configuration,
                data_dictionary=data_dictionary)

    md_const = md["pl02_const"]
    md_i = np.delete(md["pl02_i"], 2,  axis=0) if not configuration['use_poles']['P4'] else md["pl02_i"]

    data=data_dictionary['data']
    cov_inv=data_dictionary['cov_inv']

    L0 = folpsD.compute_L0(md_const, data, cov_inv)
    L1i = folpsD.compute_L1i(md_i, md_const, data, cov_inv)
    L2ij = folpsD.compute_L2ij(md_i, cov_inv)


    invL2ij = np.linalg.inv(L2ij)
    detL2ij = np.linalg.det(L2ij)

    # def startProduct(A, B, invCov):
    #     return A @ invCov @ B.T

    # term1 = FOLPS.startProduct(L1i, L1i, invL2ij)
    term1 = L1i @ invL2ij @ L1i
    term2 = np.log(abs(detL2ij))
    
    #ChatGPT says the following is better if L2ij is close to singular
    # sign, logdet = np.linalg.slogdet(L2ij)
    # if sign <= 0:
    #     return -np.inf
    # term2 = logdet

    L_marginalized = L0 + 0.5 * term1 - 0.5 * term2

    # 'f_growth': f_growth,
    # 'sigma8': sigma8_global,
    # 'sigma8_z': sigma8_zev_global,
    # 'cosmo': cosmo, 
    # 'A_AP': A_AP,

    derived = []
    if "sigma8" in derived_params:
        derived.append(md["sigma8"])
    if "Omega_m" in derived_params:
        derived.append(md["Omega_m"])
    # if "chi^2" in derived_params:
    #     derived.append(md["Omega_m"])

    dic={"sigma8_z": md["sigma8_z"],
        "f_growth": md["f_growth"],
        "A_AP": md["A_AP"]}    

    return L_marginalized, np.array(derived),  dic 
 
    
    
def log_prior(theta,parameters_sampled_and_priors):
    lp = 0.0

    for value, name in zip(theta, parameters_sampled_and_priors.keys()):
        prior = parameters_sampled_and_priors[name]

        if prior[0] == "flat":
            _, xmin, xmax = prior
            if not (xmin <= value <= xmax):
                return -np.inf

        elif prior[0] == "flat+gauss":
            _, xmin, xmax, mu, sigma = prior
            if not (xmin <= value <= xmax):
                return -np.inf
            lp -= 0.5 * ((value - mu) / sigma) ** 2

    return lp

# def log_prior_cosmo_dependent(theta,parameters_sampled_and_priors,dic):
    
#     A_AP = dic['A_AP']
#     sigma8_z = dic['sigma8_z']
#     f_growth = dic['f_growth']
    
#     return 0

def log_probability(theta,
                   parameters=None,
                   configuration=None,
                   derived_params=None,
                   data_dictionary=None,
                   fiducial=None):

    if fiducial is None:
        fiducial = {}
    if configuration is None:
        configuration = {}
    if parameters is None:
        parameters = {}
    if derived_params is None:
        derived_params = {}
    if data_dictionary is None:
        derived_params = {}  
        
    params_sampled_and_priors = params_sampled_and_priors_fn(parameters)    
    
    lp = log_prior(theta,params_sampled_and_priors)
    if not np.isfinite(lp):
        return -np.inf, np.full(len(derived_params), np.nan)

    ll, derived, dic = log_likelihood(theta,
                   parameters=parameters,
                   configuration=configuration,
                   derived_params=derived_params,
                   data_dictionary=data_dictionary,
                   fiducial=fiducial,
                  )
    
#     lpc = log_prior_cosmo_dependent(theta,parameters_sampled_and_priors,dic)
    
    return lp + ll, derived


# configuration IO functions  

def build_run_name(cfg):

    name = f"c_{cfg['model']}"

    if cfg["damping"] in ("exp", "vdg"):
        name += f"_Damp-{cfg['damping']}"

    name += f"_{cfg['tracer']}"

    namePk = ""
    
    use = cfg["use_poles"]
    kmax = cfg["kmax"]

    if use["P2"]:
        if kmax["P0"] == kmax["P2"]:
            namePk += f"_P02kmax{kmax['P0']:.3f}"
        else:
            namePk += (
                f"_P0kmax{kmax['P0']:.3f}"
                f"_P2kmax{kmax['P2']:.3f}"
            )

    if use["P4"]:
        namePk += f"_P4kmax{kmax['P4']:.3f}"

    if use["B0"] and cfg["BispBase"]=="Sugiyama":
        namePk += f"_B000kmax{kmax['B0']:.3f}"

    if use["B2"] and cfg["BispBase"]=="Sugiyama":
        namePk += f"_B202kmax{kmax['B2']:.3f}"

    name += namePk

    name += f"_bias-{cfg['bias_scheme']}"

    if not cfg["b3coev"]:
        name += "_b3free"

    if cfg["bscoev"]:
        name += "_b2coev"

    if cfg["ns_free"]:
        name += "_nsfree"

    return name

def build_shortname(cfg):
    """
    Construct a short model name
    used for plots, legends, and logs.
    """

    model_name = f"{cfg['model']}-Pk"

    if cfg["BispBool"]:
        model_name += "+Bk"

    model_name += f"_{cfg['tracer']}"
    model_name += f"_bias-{cfg['bias_scheme']}"

    if cfg["damping"] in ("exp", "vdg"):
        model_name += f"_Damp-{cfg['damping']}"

    if cfg["bscoev"]:
        model_name += "_bsCoev"

    if not cfg["b3coev"]:
        model_name += "_b3Free"

    return model_name


def h5_already_logged(h5_filename, log_file):
    if not os.path.exists(log_file):
        return False

    with open(log_file, "r") as f:
        for line in f:
            if line.strip() == h5_filename:
                return True
    return False


def chains_filename_fn(cfg, h5_filename):
     
    chains_filename = os.path.join(cfg["output_dir"], h5_filename)
    
    if os.path.exists(chains_filename):
        print(f"file '{chains_filename}' already exists.")
        if not cfg['backend_reset']:
            raise ValueError("Exit with 'backend_reset=False'")
        else:
            print("resume with 'backend_reset=True'")
            
    return chains_filename
            
            

# def is_master_mpi():
#     try:
#         from mpi4py import MPI
#         return MPI.COMM_WORLD.Get_rank() == 0
#     except Exception:
#         return True  # non-MPI execution
    

def list_to_string(lst):
    return ",".join(lst)


def write_log_block(
    h5_filename,
    param_names,
    param_names_latex,
    model_name,
    log_file="_logs.txt",
    accept_duplicated_entries=True
):
    
    # If already logged, do nothing
    if not accept_duplicated_entries:
        if h5_already_logged(h5_filename, log_file):
            # raise ValueError("f'{h5_filename} exists in {logfile}'")
            return 

    else:
        # timestamp = datetime.utcnow().isoformat(timespec="seconds")
        with open(log_file, "a") as f:
            f.write(h5_filename + "\n")
            f.write(list_to_string(param_names) + "\n")
            f.write(list_to_string(param_names_latex) + "\n")
            f.write(model_name + "\n")
            f.write('-' + "\n")
        return 


    
    
def is_python_script():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Running in a Jupyter notebook or qtconsole
            return False
        elif shell == 'TerminalInteractiveShell':
            # Running in IPython terminal
            return True
        else:
            # Other types of shells
            return True
    except NameError:
        # Probably running as a standard Python script
        return True

    
def copy_and_rename_if_py(new_filename,mcmc_file):

    current_file = os.path.realpath(__file__)
    target_directory = os.path.dirname(mcmc_file)
    print('target_directory',target_directory)
    target_directory = os.path.dirname(current_file)
    print('target_directory',target_directory)

    if mcmc_file.endswith('.py'):
        os.makedirs(target_directory, exist_ok=True)
        destination = os.path.join(target_directory, new_filename)
        shutil.copy2(mcmc_file, destination)
        print(f"Copied and renamed: {mcmc_file} → {destination}")
    elif mcmc_file.endswith('.ipynb'):
        print("Current file is a Jupyter notebook (.ipynb). Skipping copy.")
    else:
        print("Unsupported file type.")        
        
        
def logging(cfg,params_dictionary,params_sampled,params_derived,params_derived_latex,mcmc_file):
    
    run_name = build_run_name(cfg)
    shortname = build_shortname(cfg)
    h5_filename = run_name + cfg['chains_ext']
    copy_filename = cfg['output_dir']+run_name + ".py"
    
    params_latex=params_sampled_latex(params_dictionary)+params_derived_latex
    
    if is_python_script():
        copy_and_rename_if_py(copy_filename,mcmc_file)
        # if not os.path.exists(chains_filename):
        write_log_block(h5_filename, params_sampled, params_latex, 
                        shortname, log_file=cfg['log_file'])
    else:
        write_log_block(h5_filename, params_sampled, 
                        params_latex,shortname,
                             "_logs_h.txt",
                              accept_duplicated_entries=True)
        
    chains_filename = chains_filename_fn(cfg, h5_filename)
    
    if os.path.exists(chains_filename): 
        if not cfg['backend_reset']:
            raise ValueError(f'Exit: file {chains_filename} already exists')
        else:
            print(f'{chains_filename} exists. Running with back_end')

    print(f'chains: {chains_filename}') 
    print(f'shortname: {shortname}') 
                
    return chains_filename    



#. data functions

def load_and_prepare_abacus2ndgen_data(
    cfg,
    totsim=2000,
    chatty=True,
    showplot=False,
    showbinning=False,
    use_mike_data_tools=False,
    kfin_bk=0.28,
    total_bk_points=60,
):
    """
    Load Abacus2ndGen 2nd-generation power spectrum and bispectrum data,
    apply pole and k-range selections, construct covariance matrices,
    and build calibrated Abacus binning matrices.

    This implementation is numerically identical to the original pipeline,
    including the calibrated (N-1)/N binning factor and center-based shells.
    """

    # ================================================================
    # Configuration
    # ================================================================
    DATA_DIR = cfg["DATA_DIR"]
    tracer = cfg["tracer"]

    kmin = cfg["kmin"]
    kmax = cfg["kmax"]
    poles = cfg["use_poles"]

    # ================================================================
    # Load data
    # ================================================================
    loaded = np.load(DATA_DIR + f"Abacus2ndGen_{tracer}_kP0P2P4B00B202.npy")

    k_data_all = loaded[0]
    P0_data_all = loaded[1]
    P2_data_all = loaded[2]
    P4_data_all = loaded[3]
    B000_data_all = loaded[4]
    B202_data_all = loaded[5]

    cov_array_all = np.load(DATA_DIR + f"EZmocks_{tracer}_cov_array_all.npy")
    k_cov_all = np.load(DATA_DIR + f"EZmocks_{tracer}_k_cov_all.npy")
    mean_ezmocks_all = np.load(DATA_DIR + f"EZmocks_{tracer}_mean_ezmocks_all.npy")

    # ================================================================
    # Pole selection
    # ================================================================
    pole_selection = [
        poles["P0"],
        poles["P2"],
        poles["P4"],
        poles["B0"],
        poles["B2"],
    ]

    if chatty:
        print(f"pole selection [P0,P2,P4,B0,B2] = {pole_selection}")

    ranges = [
        [kmin["P0"], kmax["P0"]],
        [kmin["P2"], kmax["P2"]],
        [kmin["P4"], kmax["P4"]],
        [kmin["B0"], kmax["B0"]],
        [kmin["B2"], kmax["B2"]],
    ]

    def pole_k_mask(kvec, pole_flags, kranges):
        """
        Mask for concatenated [P0,P2,P4,B0,B2] vectors with equal k grids.
        """
        n_poles = len(pole_flags)
        n_per = len(kvec) // n_poles

        fit_sel = np.repeat(pole_flags, n_per)
        k_base = kvec[:n_per]

        range_sel = np.concatenate([
            (kr[0] < k_base) & (k_base < kr[1]) for kr in kranges
        ])

        return fit_sel & range_sel

    mask = pole_k_mask(k_cov_all, pole_selection, ranges)
    k_cov = k_cov_all[mask]

    # ================================================================
    # Data vector construction
    # ================================================================
    def select(k, d, kr, enabled):
        if not enabled:
            return np.array([]), np.array([]),np.array([])
        sel = (kr[0] < k) & (k < kr[1])
        return k[sel], d[sel], sel

    kr_p0, P0, selP0 = select(k_data_all, P0_data_all, ranges[0], poles["P0"])
    kr_p2, P2, selP2 = select(k_data_all, P2_data_all, ranges[1], poles["P2"])
    kr_p4, P4, selP4 = select(k_data_all, P4_data_all, ranges[2], poles["P4"])
    kr_b0, B0, selB0 = select(k_data_all, B000_data_all, ranges[3], poles["B0"])
    kr_b2, B2, selB2 = select(k_data_all, B202_data_all, ranges[4], poles["B2"])

    k_points_p0 = np.where((ranges[0][0] < k_data_all) & (k_data_all < ranges[0][1])  & poles['P0'])
    k_points_p2 = np.where((ranges[1][0] < k_data_all) & (k_data_all < ranges[1][1])  & poles['P2'])
    k_points_p4 = np.where((ranges[2][0] < k_data_all) & (k_data_all < ranges[2][1])  & poles['P4'])
    k_points_b0 = np.where((ranges[3][0] < k_data_all) & (k_data_all < ranges[3][1])  & poles['B0'])
    k_points_b2 = np.where((ranges[4][0] < k_data_all) & (k_data_all < ranges[4][1])  & poles['B2'])

    data = np.concatenate([P0, P2, P4, B0, B2])

    n_points = {
        "P0": len(P0),
        "P2": len(P2),
        "P4": len(P4),
        "B0": len(B0),
        "B2": len(B2),
    }

    if chatty:
        for i, (k, v) in enumerate(n_points.items()):
            print(f"{k:2s} points: {v}" + (f', from {ranges[i][0]} to {ranges[i][1]}' if v > 0 else ''))

    # ================================================================
    # Covariance
    # ================================================================
    cov_array = cov_array_all[np.ix_(mask, mask)]
    n_data = len(data)

    hartlap = (totsim - 1.0) / (totsim - n_data - 2.0)
    cov = cov_array * hartlap
    cov_inv = np.linalg.inv(cov)

    # ================================================================
    # Optional plots (original behavior preserved)
    # ================================================================
    if showplot:
        import matplotlib.pyplot as plt

        plt.plot(k_data_all, k_data_all * P0_data_all)
        plt.plot(k_data_all, k_data_all * P2_data_all)
        plt.plot(k_data_all, k_data_all * P4_data_all)

        plt.plot(kr_p0, kr_p0 * P0, lw=3, ls="--", color="k")
        plt.plot(kr_p2, kr_p2 * P2, lw=3, ls="--", color="k")
        plt.plot(kr_p4, kr_p4 * P4, lw=3, ls="--", color="k")
        plt.show()

        plt.plot(k_data_all, k_data_all**2 * B000_data_all)
        plt.plot(k_data_all, k_data_all**2 * B202_data_all)

        plt.plot(kr_b0, kr_b0**2 * B0, lw=3, ls="--", color="k")
        plt.plot(kr_b2, kr_b2**2 * B2, lw=3, ls="--", color="k")
        plt.show()

    # ================================================================
    # Binning
    # ================================================================
    N_SUB = 5
    DK_FINE = 0.001
    shell_factor = (N_SUB - 1) / N_SUB

    k_max = max(kmax["P0"], kmax["P2"], kmax["P4"])
    N_ck = max(int(k_max * 100) + 2, 25)
    
    if chatty:
        print("binning: points per bin =", N_SUB)

    k_thy = (
        np.linspace(0.0, 0.01 * N_ck, 2 * N_ck * N_SUB, endpoint=False)
        + 0.0025
        + 0.0005
    )
    
    k_data_coarse = k_data_all[: 2 * N_ck]

    m_bin = np.zeros((len(k_data_coarse), len(k_thy)))
    m_bin_k = np.zeros_like(m_bin)

    for i in range(len(k_data_coarse)):
        idx = slice(N_SUB * i, N_SUB * (i + 1))
        k_sub = k_thy[idx]

        # IMPORTANT: center-based shell limits (original behavior)
        norm = (k_sub[-1] ** 3 - k_sub[0] ** 3) / 3.0

        weights = (k_sub**2 * DK_FINE) / norm
        weights *= shell_factor

        m_bin[i, idx] = weights
        m_bin_k[i, idx] = k_sub

    # ================================================================
    # Debug block (restored & generalized)
    # ================================================================
    if showbinning:
        import matplotlib.pyplot as plt
        print(" ")
        print("plots for binning:")

        def debug_abacus_binning(indices, shift=0.0):
            ones = np.ones(N_SUB)

            for n in indices:
                idx = slice(N_SUB * n, N_SUB * (n + 1))
                print(f"bin {n}:")
                print("weights:", m_bin[n, idx])
                print("k_sub:", m_bin_k[n, idx])
                print("mean k_sub:", np.mean(m_bin_k[n, idx]))
                print("k_data:", k_data_coarse[n])
                print("k_thy_center:", k_thy[N_SUB * n + N_SUB // 2])

                plt.ylim(0.8, 1.3)
                plt.plot(m_bin_k[n, idx] - shift, ones, "o")
                plt.plot(k_data_coarse[n] - shift, 1.2, "o")
                plt.plot(k_thy[N_SUB * n + N_SUB // 2] - shift, 1.22, "o")
                plt.show()
                print()

        debug_abacus_binning([0, 2 * N_ck - 1])

    # ================================================================
    # Bispectrum k grid
    # ================================================================
    kb_all = np.linspace(0.5 * k_thy[0], kfin_bk, total_bk_points)
    k_ev_bk = np.vstack([kb_all, kb_all]).T


    return {
        "data": data,
        "cov": cov,
        "cov_inv": cov_inv,
        "k_cov": k_cov,
        "k_thy": k_thy,
        "m_bin": m_bin,
        "k_ev_bk": k_ev_bk,
        "kr": {
            "P0": kr_p0,
            "P2": kr_p2,
            "P4": kr_p4,
            "B0": kr_b0,
            "B2": kr_b2,
        },
        "sel": {
            "P0": selP0,
            "P2": selP2,
            "P4": selP4,
            "B0": selB0,
            "B2": selB2,
        },
        "k_points": {
            "P0": k_points_p0,
            "P2": k_points_p2,
            "P4": k_points_p4,
            "B0": k_points_b0,
            "B2": k_points_b2,
        },
        "n_points": n_points,
    }



    
def plotmultipoles(loaded_data_list, error_alpha=0.2):
    """
    Plot multipoles from multiple datasets with distinct error bar styles.
    
    Parameters:
    -----------
    loaded_data_list : list of dict
        List of data dictionaries
    error_alpha : float
        Transparency for error bars (0-1)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_datasets = len(loaded_data_list)
    
    # Generate distinct color palette
    colors = plt.cm.tab20(np.linspace(0, 1, n_datasets * 2))[::2]
    
    # Different combinations of line and error styles
    line_styles = ['-', '--', '-.', ':'][:min(4, n_datasets)]
    error_styles = ['solid', 'dashed', 'dashdot', 'dotted'][:min(4, n_datasets)]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    ps_labels = ['P0', 'P2', 'P4']
    bs_labels = ['B0', 'B2']
    
    # Store plotted lines for custom legend
    ps_legend_elements = []
    bs_legend_elements = []
    
    for idx, loaded_data in enumerate(loaded_data_list):
        kr = loaded_data['kr']
        n_points = loaded_data['n_points']
        data = loaded_data['data']
        errors = np.sqrt(np.diag(loaded_data['cov']))
        
        color = colors[idx]
        ls = line_styles[idx % len(line_styles)]
        error_ls = error_styles[idx % len(error_styles)]
        
        # Plot Power Spectrum
        dmin = 0
        for label in ps_labels:
            dmax = dmin + n_points[label]
            x, y, e = kr[label], data[dmin:dmax], errors[dmin:dmax]
            scale = 2 if label == 'P4' else 1
            
            # Plot line
            line = ax1.plot(x, x * (scale * y), 
                           color=color, ls=ls, linewidth=2,
                           label=f'Set {idx+1}: {label}')[0]
            
            # Plot error bars with different style
            ax1.fill_between(x, 
                            x * (scale * (y - e)), 
                            x * (scale * (y + e)),
                            color=color, alpha=error_alpha,
                            linestyle=error_ls, linewidth=0.5)
            
            if idx == 0:  # Store for legend
                ps_legend_elements.append(line)
            
            dmin = dmax
        
        # Plot Bispectrum
        for label in bs_labels:
            dmax = dmin + n_points[label]
            x, y, e = kr[label], data[dmin:dmax], errors[dmin:dmax]
            
            # Plot line
            line = ax2.plot(x, x**2 * y,
                           color=color, ls=ls, linewidth=2,
                           label=f'Set {idx+1}: {label}')[0]
            
            # Plot error bands
            ax2.fill_between(x,
                            x**2 * (y - e),
                            x**2 * (y + e),
                            color=color, alpha=error_alpha,
                            linestyle=error_ls, linewidth=0.5)
            
            if idx == 0:  # Store for legend
                bs_legend_elements.append(line)
            
            dmin = dmax
    
    # Custom legends
    dataset_names = [f'Dataset {i+1}' for i in range(n_datasets)]
    multipole_names = ['P0', 'P2', 'P4', 'B0', 'B2']
    
    # Create combined legend for datasets
    from matplotlib.lines import Line2D
    dataset_legend_elements = [
        Line2D([0], [0], color=colors[i], lw=3, ls=line_styles[i % len(line_styles)],
              label=f'Dataset {i+1}')
        for i in range(n_datasets)
    ]
    
    # Add dataset legend
    ax1.legend(handles=dataset_legend_elements, loc='upper right', fontsize=10)
    ax2.legend(handles=dataset_legend_elements, loc='upper right', fontsize=10)
    
    # Add multipole type indicators
    ax1.text(0.02, 0.98, 'Multipoles:\n• P0 (solid fill)\n• P2 (dashed fill)\n• P4 (dashdot fill)',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('k [h/Mpc]', fontsize=12)
    ax1.set_ylabel('P(k) × k', fontsize=12)
    ax1.set_title(f'Power Spectrum ({n_datasets} datasets)', fontsize=14)
    ax1.grid(True, alpha=0.2)
    
    ax2.set_xlabel('k [h/Mpc]', fontsize=12)
    ax2.set_ylabel('B(k) × k²', fontsize=12)
    ax2.set_title(f'Bispectrum ({n_datasets} datasets)', fontsize=14)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)    

