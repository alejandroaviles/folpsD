import numpy as np
from classy import Class
import sys, os, shutil, re
sys.path.append('../')
import folps as folpsD
from likelihood_tools import *
# from cosmo_class import *


def fiducial_values(tracer,bias_scheme='folps'):

    #root = abacus_cosm000.
    #Baseline LCDM, Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing mean    
    #https://github.com/abacusorg/AbacusSummit/blob/main/Cosmologies/abacus_cosm000/CLASS.ini
    abacus_cosmo = {'h':0.6736,
           'omega_cdm':0.12,
           'omega_b':0.02237, 
           'Omega_m': 0.315191849, #derived
           'logAs': np.log(10**10 * 2.083e-9),
           'ns': 0.9649,
           'sigma8':0.811355,
           'omega_ncdm':0.00064420,
           'N_eff':3.045, #derived
           'w0':-1,
           'wa':0,
           'omega_k':0.0,
           'N_ncdm':1,
           'N_ur': 2.0328,
           'alpha_s': 0,
                   }
    

    logAs_fid=abacus_cosmo['logAs']
    ns_fid=abacus_cosmo['ns']
    h_fid=abacus_cosmo['h']
    omega_cdm_fid=abacus_cosmo['omega_cdm']
    omega_ncdm_fid=abacus_cosmo['omega_ncdm']
    omega_b_fid=abacus_cosmo['omega_b']
    As_fid = np.exp(logAs_fid)/(10**10)
    N_eff_fid =abacus_cosmo['N_eff']
    N_ncdm_fid =abacus_cosmo['N_ncdm']
    w0_fid =abacus_cosmo['w0']
    wa_fid =abacus_cosmo['wa']
    omega_k_fid =abacus_cosmo['omega_k']
    N_ur_fid =abacus_cosmo['N_ur']
    alpha_s_fid =abacus_cosmo['alpha_s']
    
    if tracer=='LRG2':
        b1_fid=2.1
    elif tracer=='LRG1':
        b1_fid=2.0
    elif tracer=='QSO':
        b1_fid=2.0
    else:
        raise ValueError(
            f"Invalid tracer='{tracer}'. Only 'LRG1', 'LRG2', 'QSO' "
        )

    b2_fid, bs_fid, b3_fid = coev_fn_b2bsb3(b1_fid, 0, True, 0, True, 0, True, bias_scheme)  
    
    fiducial_values = {
    "h": h_fid,
    "omega_cdm": omega_cdm_fid,
    "omega_b": omega_b_fid,
    "logAs": logAs_fid,
    "ns": ns_fid,
    "omega_ncdm": omega_ncdm_fid,
    "N_eff": N_eff_fid,
    "N_ur": N_ur_fid,
    "N_ncdm": N_ncdm_fid,
    "w0": w0_fid,
    "wa": wa_fid,
    "omega_k": omega_k_fid,
    "b1": b1_fid,
    "b2": b2_fid,
    "bs": bs_fid,
    "b3": b3_fid,
    "c1": 66.6,
    "c2": 0.0,
    "Pshot": 0.0,
    "Bshot": 0.0,
    "X_FoG_pk": 0.0,
    "X_FoG_bk": 0.0,
    "is_w0wa": False,
    "N_ur": N_ur_fid,
    "alpha_s": alpha_s_fid,
    }
    
    return fiducial_values
        

def distances(fids,z,chatty=True):
    cosmo_fiducial = run_class_(h = fids['h'], ombh2 = fids['omega_b'], 
                               omch2 = fids['omega_cdm'], omnuh2 = fids['omega_ncdm'],As = np.exp(fids['logAs'])/(10**10), 
                               ns = fids['ns'],z=z)
    cosmo_fid = cosmo_fiducial['cosmo']   
    DA_fid    = cosmo_fid.angular_distance(z)
    H_fid     = cosmo_fid.Hubble(z)
    H0_fid    = cosmo_fid.Hubble(0.0)
    
    fids["DA"] = DA_fid
    fids["H"] = H_fid
    fids["H0"] = H0_fid
    fids["cosmo"] = cosmo_fid
    
    if chatty:
        print(f"DA_fid(z={z})= {fids['DA']},")
        print(f"H_fid(z={z})= {fids['H']},")
        print(f"H0_fid= {fids['H0']}")

    return {'DA':DA_fid,'H':H_fid,'H0':H0_fid,'cosmo':cosmo_fid}




def BBN_prior(N_eff_free=False,ref='schoneberg'):
    
    if ref=='schoneberg':  #2401.15054
        if N_eff_free:
            return {'mean': 0.02196, 'stddev': 0.00063}  #There is a covariance between N_eff and omega_b
        else:
            return {'mean': 0.02218, 'stddev': 0.00055}
    else:
        raise ValueError("only accepts ref='schoneberg'") 

    
def n_bar(tracer):
    nbars = {
        'LRG1': 1e-4,
        'LRG2': 1e-4,
        'QSO': 1e-4,
        'ELG': 1e-4,
    }
    
    try:
        return nbars[tracer]
    except KeyError:
        raise ValueError(f"tracer must be one of {list(nbars.keys())}, got '{tracer}'")


def b2coev_fn(b1,bias_scheme='folps'):
    
    if bias_scheme=='folps' or bias_scheme=='priordoc':
        return 8./21*(b1-1)
    elif bias_scheme=='classpt':
        return 0
    else:
        raise ValueError("bias scheme should be 'folps', 'classpt', 'priordoc'")

def bscoev_fn(b1,bias_scheme='folps'):
    
    if bias_scheme=='folps':
        return -4/7*(b1-1)
    elif bias_scheme=='classpt' or bias_scheme=='priordoc':
        return -2/7*(b1-1)
    else:
        raise ValueError("bias scheme should be 'folps', 'classpt', 'priordoc'")
        
        
def b3coev_fn(b1,bias_scheme='folps'):
    
    if bias_scheme=='folps':
        return 32/315*(b1-1)
    elif bias_scheme=='classpt' or bias_scheme=='priordoc':
        return 23/42*(b1-1)
    else:
        raise ValueError("bias scheme should be 'folps', 'classpt', 'priordoc'")


def coev_fn_b2bsb3(b1, b2, b2coev, bs, bscoev, b3, b3coev, bias_scheme='folps'):
    """
    Compute coevolution values for 
        b2 if bscoev, for 
        bs if b2coev, and for 
        b3 if b3coev.
    """
    b2_new = b2coev_fn(b1,bias_scheme) if b2coev else b2
    bs_new = bscoev_fn(b1,bias_scheme) if bscoev else bs
    b3_new = b3coev_fn(b1,bias_scheme) if b3coev else b3

    return b2_new, bs_new, b3_new


def transform_from_folpsbiases(b1folps, b2folps, bsfolps, b3folps, bias_scheme):
    """
    Apply transformations 
    b1classpt = b1folps
    b2classpt = b2folps + 2/3 *bsfolps
    bsclasspt = bsfolps/2
    b3classpt = -5/4 bsfolps - 105/64*b3folps

    b1priordoc = b2folps
    b2priordoc = b2folps
    bspriordoc = bsfolps/2
    b3priordoc = -5/4 bsfolps - 105/64*b3folps
    ----------

    bias_scheme : 'folps', 'classpt' or 'priordoc'

        
    """

    if bias_scheme == 'folps':
        return (b1folps, b2folps, bsfolps, b3folps)


    elif bias_scheme == "classpt":
        b1classpt = b1folps
        b2classpt = b2folps + (2.0 / 3.0) * bsfolps
        bsclasspt = bsfolps / 2.0
        b3classpt = -(5.0 / 4.0) * bsfolps - (105.0 / 64.0) * b3folps
        return (b1classpt, b2classpt, bsclasspt, b3classpt)

    elif bias_scheme == "priordoc":
        b1priordoc = b2folps
        b2priordoc = b2folps
        bspriordoc = bsfolps / 2.0
        b3priordoc = -(5.0 / 4.0) * bsfolps - (105.0 / 64.0) * b3folps

        return (b1priordoc, b2priordoc, bspriordoc, b3priordoc)

    else:
        return "Error: Invalid bias_scheme. Choose 'priordoc' or 'classpt'."

    
            

def transform_from_classpt_to_folpsbiases(b1_classpt, b2_classpt, bs_classpt, b3_classpt):
    """
    Apply transformations 
    b1_folps = b1_classpt
    b2_folps = b2_classpt - 4/3 * bs_classpt
    bs_folps = 2 * bs_classPT
    b3_folps = -32/21 * (bs_classPT + 2/5 * b3_classpt)
    """

    b1_folps = b1_classpt
    b2_folps = b2_classpt - 4/3 * bs_classpt
    bs_folps = 2 * bs_classPT
    b3_folps = -32/21 * (bs_classPT + 2/5 * b3_classpt)
    
    return (b1_folps, b2_folps, bs_folps, b3_folps)


def transform_from_priordoc_to_folpsbiases(b1_priordoc, b2_priordoc, bs_priordoc, b3_priordoc):
    """
    Apply transformations 
    b1_folps = b1_priordoc
    b2_folps = b2_priordoc
    bs_folps = 2 * bs_priordoc
    b3_folps = -32/21 * (bs_priordoc + 2/5 * b3_priordoc)
    """

    b1_folps = b1_priordoc
    b2_folps = b2_priordoc - 4/3 * bs_priordoc
    bs_folps = 2 * bs_priordoc
    b3_folps = -32/21 * (bs_priordoc + 2/5 * b3_priordoc)
    
    return (b1_folps, b2_folps, bs_folps, b3_folps)













def run_class_(h=0.6711, ombh2=0.022, omch2=0.122, omnuh2=0.0006442, 
              As=2e-9, ns=0.965, z=0.97, z_scale=None, N_ur=2.0328, N_ncdm=1,
              khmin=0.0001, khmax=2.0, nbk=700, is_w0wa=False,
              w0=-1, wa=0, Omkh2=0, deg_ncdm=None, spectra='cb'):
    """
    Generates the linear power spectrum (cb) using CLASS.

    Args:
        h (float): Reduced Hubble constant, H0/100, where H0 is the Hubble constant at z=0.
        ombh2 (float): Physical baryon density, Ω_b h².
        omch2 (float): Physical cold dark matter density, Ω_c h².
        omnuh2 (float): Physical massive neutrino density, Ω_ν h².
        As (float): Amplitude of primordial curvature fluctuations.
        ns (float): Spectral index of the primordial power spectrum.
        z (float): Redshift at which the power spectrum is evaluated.
        z_scale (float or list, optional): Single redshift value or list of redshift values for 
                                           additional scaling (default: None).
        N_ur (float): Number of ultra-relativistic (massless) neutrino species.
        khmin (float): Minimum wave number in units of h/Mpc.
        khmax (float): Maximum wave number in units of h/Mpc.
        nbk (int): Number of k points between khmin and khmax.
        w0_fld (float, optional): Dark energy equation of state parameter w0 (default: None).
        wa_fld (float, optional): Dark energy equation of state evolution parameter wa (default: None). 
                                  If w0_fld is provided but wa_fld is not, wa_fld is set to 0.
        Omkh2 (float, optional): Physical curvature density, Ω_k h² (default: None).
        deg_ncdm (float, optional): Degeneracy of massive neutrino species (default: None).
        spectra (str): Specifies which components to include in the power spectrum (e.g., 'cb' for 
                       cold dark matter + baryons, 'total' for total matter) (default: 'cb').

    Returns:
        kh (numpy.ndarray): Array of wave numbers.
        pk (numpy.ndarray): Array of the linear power spectrum values corresponding to the wave numbers.
        pk_z_scale_list (list, optional): List of arrays of power spectra for each redshift in z_scale, if provided.

    Notes:
        - If w0_fld is specified, the cosmological constant (Omega_Lambda) is set to 0, as we are 
          modeling a dark energy component with a time-varying equation of state.
        - If wa_fld is not specified but w0_fld is, wa_fld defaults to 0, representing no evolution
          in the dark energy equation of state.
    """
    
    
    params = {
        'output': 'mPk',
        'omega_b': ombh2,
        'omega_cdm': omch2,
        'omega_ncdm': omnuh2, 
        'h': h,
        'A_s': As,
        'n_s': ns,
        'P_k_max_1/Mpc': khmax,
        'z_max_pk':3,             # Default value is 10 
        'N_ur': N_ur,             # Massless neutrinos 
        'N_ncdm': N_ncdm          # Massive neutrinos species
    }
    
    if is_w0wa:
        # params['Omega_Lambda'] = 0
        # params['w0_fld'] = w0_fld
        # params['wa_fld'] = 0 if wa_fld is None else wa_fld
        params['Omega_Lambda'] = 0
        params['w0_fld'] = w0
        params['wa_fld'] = wa        
        
    if Omkh2 is not None:
        params['Omega_k'] = Omkh2
        
    if deg_ncdm is not None:
        params['deg_ncdm'] = deg_ncdm
        
    try:
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize CLASS with error: {e}")
    
    # Specify k [h/Mpc]
    k = np.logspace(np.log10(khmin), np.log10(khmax), num=nbk)
    
    # Compute the linear power spectrum for the provided redshift 'z'
    try:
        if spectra in ['m', 'matter', 'total']:
            Plin = np.array([cosmo.pk_lin(ki * h, z) * h**3 for ki in k])
        else:
            Plin = np.array([cosmo.pk_cb(ki * h, z) * h**3 for ki in k])
    except Exception as e:
        raise RuntimeError(f"Failed to compute power spectrum at z={z} with error: {e}")
        
    #Computing growths: f and D, and sigma8
    fz = cosmo.scale_independent_growth_factor_f(z)
    Dz = cosmo.scale_independent_growth_factor(z)
    s8 = cosmo.sigma(R = 8.0/h, z = z)
    
    fz_scale_values = []
    Dz_scale_values = []
    s8_scale_values = []
    
    # Check if z_scale is a single value or a list
    if z_scale is not None:
        if not isinstance(z_scale, list):
            z_scale = [z_scale]  # Convert to a list if it's a single value
        
        for z_scale_value in z_scale:
            fz_scale_value = cosmo.scale_independent_growth_factor_f(z_scale_value)
            Dz_scale_value = cosmo.scale_independent_growth_factor(z_scale_value)
            s8_scale_value = cosmo.sigma(R=8.0/h, z=z_scale_value)

            fz_scale_values.append(fz_scale_value)
            Dz_scale_values.append(Dz_scale_value)
            s8_scale_values.append(s8_scale_value)
    
    rdrag = cosmo.rs_drag()
    
    return {'k': k, 
            'pk': Plin,
            'fz': fz,
            'Dz': Dz,
            's8': s8,
            'fz_scale': fz_scale_values,
            'Dz_scale': Dz_scale_values,
            's8_scale': s8_scale_values,
            'rbao': rdrag,
            'cosmo': cosmo
            }






def calculate_M_matrices(configuration = None):
    
    if configuration is None:
        configuration= {'Nfftlog': 128,
                        'A_full': True,
                        'use_TNS_model': False,
                       }

    
    matrix = folpsD.MatrixCalculator(nfftlog=configuration['Nfftlog'], 
                                     A_full=configuration['A_full'], 
                                     use_TNS_model= configuration['use_TNS_model'])
    mmatrices = matrix.get_mmatrices()
    configuration["M_matrices"]=mmatrices
    
    return None





def FolpsD(cosmo_params=None, 
           fiducial=None, 
           configuration=None, 
           data_dictionary=None,
           chatty=False,
           engine='CLASS'):
    """
    FolpsD(cosmo_params=None,fiducial=fids,configuration=cfg,data_dictionary=data_d,engine='CLASS')
    input: e.g. cosmo_params = {'omega_cdm': 0.13, 'h': 0.65, 'b2': 0.2,...}. 
    params: h,omega_cdm,omega_b,logAs,ns,N_eff,N_ur,N_ncdm,omega_ncdm,omega_k,is_w0wa,
            w0,wa,b1,b2,bs,b3,c1,c2,Pshot,X_FoG_pk,X_FoG_bk
    Pk params alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2 are set to zero. For analytical marginalization
    
    """
    # global sigma8_global, cosmo, sigma8_zev_global, A_AP

    if fiducial is None:
        fiducial = {}
    if configuration is None:
        configuration= {}
    if data_dictionary is None:
        data_dictionary = {}
    
    
    if cosmo_params is None:
        cosmo_params = {}
        
    h = cosmo_params.get('h', fiducial['h'])
    omega_cdm = cosmo_params.get('omega_cdm', fiducial['omega_cdm'])
    omega_b = cosmo_params.get('omega_b', fiducial['omega_b'])
    logAs = cosmo_params.get('logAs', fiducial['logAs'])
    ns = cosmo_params.get('ns', fiducial['ns'])
    N_eff = cosmo_params.get('N_eff', fiducial['N_eff'])
    N_ur = cosmo_params.get('N_ur', fiducial['N_ur'])
    N_ncdm = cosmo_params.get('N_ncdm', fiducial['N_ncdm'])
    omega_ncdm = cosmo_params.get('omega_ncdm', fiducial['omega_ncdm'])
    omega_k = cosmo_params.get('omega_k', fiducial['omega_k'])
    is_w0wa = cosmo_params.get('is_w0wa', fiducial['is_w0wa']) #False,True
    w0 = cosmo_params.get('w0', fiducial['w0'])
    wa = cosmo_params.get('wa', fiducial['wa'])
    
    Omega_m = (omega_cdm+omega_b+omega_ncdm)/h**2
    
    # Bias parameters
    b1 = cosmo_params.get('b1', fiducial['b1'])
    b2 = cosmo_params.get('b2', fiducial['b2'])
    bs = cosmo_params.get('bs', fiducial['bs'])
    b3 = cosmo_params.get('b3', fiducial['b3'])
    
    c1 = cosmo_params.get('c1', fiducial['c1'])
    c2 = cosmo_params.get('c2', fiducial['c2'])
    Pshot = cosmo_params.get('Pshot', fiducial['Pshot'])
    Bshot = cosmo_params.get('Bshot', fiducial['Bshot'])
    
    X_FoG_pk = cosmo_params.get('X_FoG_pk', fiducial['X_FoG_pk'])
    X_FoG_bk = cosmo_params.get('X_FoG_bk', fiducial['X_FoG_bk'])                             
                                         
    # Configuration parameters
    data = data_dictionary['data']
    k_thy = data_dictionary['k_thy']
    k_ev_bk = data_dictionary['k_ev_bk']
    m_bin = data_dictionary['m_bin']
    kr = data_dictionary['kr']
    sel = data_dictionary['sel']
    k_points = data_dictionary['k_points']
                     
    model = configuration['model']                                         
    zval = configuration['z']
    b2coev = configuration['b2coev']
    bscoev = configuration['bscoev']
    b3coev = configuration['b3coev']
    use_poles = configuration['use_poles']
                       
    BispBase = configuration['BispBase']
    bias_scheme = configuration['bias_scheme']
    damping = configuration['damping']
    use_TNS_model = configuration['use_TNS_model']
    kernels = configuration['kernels']
    
    
    if b2coev:
        b2 = b2coev_fn(b1,bias_scheme=configuration['bias_scheme'])
    if bscoev:
        bs = bscoev_fn(b1,bias_scheme=configuration['bias_scheme'])
    if b3coev:
        b3 = b3coev_fn(b1,bias_scheme=configuration['bias_scheme'])
    
    
    alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP = 0, 0, 0, 0, 0, 0, 1e-4
    NuisanParams = [b1, b2, bs, b3, alpha0, alpha2, alpha4, ctilde,alphashot0, alphashot2, PshotP, X_FoG_pk]
    BispParams = [b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_bk]        

    As = np.exp(logAs) / (10**10)
    z_ev = zval
    Omega_m = (omega_cdm + omega_b + omega_ncdm) / h**2
    fnu = omega_ncdm / (omega_cdm + omega_b + omega_ncdm)
    
    if engine=='CLASS':
        if configuration['N_eff_free']:
            if not configuration['N_ur_free']:
                print('warning: CLASS engine uses N_ur instead of N_eff, we use N_ur=N_eff-1.0132. Or change to N_ur.')
                N_ur=N_eff-1.0132
            if configuration['N_ur_free']:
                print('warning: using N_eff and N_ur free. N_eff is not considered.')
        ps = run_class_(h = h, ombh2 = omega_b, omch2 = omega_cdm, omnuh2 = omega_ncdm,As = As, 
                       ns = ns, z = z_ev, z_scale = None, N_ur = N_ur, N_ncdm=1,
                       khmin = 0.0001, khmax = 2.0, nbk = 1000, 
                       is_w0wa=is_w0wa,w0=w0, wa=wa, Omkh2=omega_k, deg_ncdm=None, spectra='cb')
    else:
        raise ValueError(f"engine={engine} is not supported. Use engine='CLASS'")
            
    cosmo = ps['cosmo']   
    inputpkT = ps['k'], ps['pk']

    DA = cosmo.angular_distance(z_ev)
    H = cosmo.Hubble(z_ev)
    H0 = cosmo.Hubble(0.0)
    D_growth = cosmo.scale_independent_growth_factor(z_ev)
    # print(D_growth,cosmo.scale_independent_growth_factor(0.0))
    f_growth = cosmo.scale_independent_growth_factor_f(z_ev)
    sigma8_global = cosmo.sigma8()
    sigma8_zev_global = D_growth*sigma8_global
    
    H_fid = fiducial['H']
    H0_fid = fiducial['H0']
    DA_fid = fiducial['DA']
    
    q_par = H_fid / H
    q_perp = DA / DA_fid
    
    A_AP = (H0_fid/H0)**3 /( q_par * q_perp**2 ) 
    
    # qpar, qperp = folpsD.qpar_qperp(Omega_fid=abacus_LRG1_cosmo['Omega_m'], Omega_m=Omega_m, z_pk=kwargs['z'], cosmo=None) 
    # f0=folpsD.f0_function(z_pkl,kwargs['Omega_m']) 
    # print('qs class:', q_par,q_perp)
    # print('qs FOLPS:', qpar,qperp)
    # print('sigma8',sigma8_global)
    # print('fz class',f_growth)
    # print('fz folps',f0)

    kwargs_folps = {
        'z': z_ev,
        'h': h,
        'Omega_m': Omega_m,
        'f0': f_growth,
        'fnu': fnu,
    }

    nonlinear = folpsD.NonLinearPowerSpectrumCalculator(mmatrices=configuration['M_matrices'],
                                                        kernels=kernels,**kwargs_folps)
    table, table_nonwiggles = nonlinear.calculate_loop_table(k=inputpkT[0], pklin=inputpkT[1],
                                                             cosmo=cosmo,**kwargs_folps)
    multipoles = folpsD.RSDMultipolesPowerSpectrumCalculator(model=model) 
    
    
    
    pkl0, pkl2, pkl4 = multipoles.get_rsd_pkell(kobs=k_thy,qpar=q_par,qper=
                                                q_perp,pars=NuisanParams,
                                                table=table,table_now=table_nonwiggles,
                                                bias_scheme=bias_scheme,damping=damping)
                                        
    pkl0_const, pkl2_const, pkl4_const = folpsD.get_rsd_pkell_marg_const(kobs=k_thy,qpar=q_par,
                                                               qper=q_perp,pars=NuisanParams,
                                                               table=table,table_now=table_nonwiggles,
                                                            bias_scheme=bias_scheme,damping=damping,
                                                            nmu=6,ells=(0,2,4),model=model)
    
    pkl0_i, pkl2_i, pkl4_i =   folpsD.get_rsd_pkell_marg_derivatives(kobs=k_thy,qpar=q_par,
                                                               qper=q_perp,pars=NuisanParams,
                                                               table=table,table_now=table_nonwiggles,
                                                            bias_scheme=bias_scheme,damping=damping,
                                                            nmu=6,ells=(0,2,4),model=model)

    pkl0_const_mbin = m_bin @ pkl0_const 
    pkl2_const_mbin = m_bin @ pkl2_const 
    pkl4_const_mbin = m_bin @ pkl4_const 

    pl02_const_binning = np.concatenate((pkl0_const_mbin[k_points['P0']], 
                                         pkl2_const_mbin[k_points['P2']], 
                                         pkl4_const_mbin[k_points['P4']]))  

    pkl0_i_mbin = np.zeros((len(pkl0_i), len(m_bin)))
    pkl2_i_mbin = np.zeros((len(pkl2_i), len(m_bin)))
    pkl4_i_mbin = np.zeros((len(pkl4_i), len(m_bin)))

    for ii in range(len(pkl0_i)):
        pkl0_i_mbin[ii, :] = m_bin @ pkl0_i[ii]
        pkl2_i_mbin[ii, :] = m_bin @ pkl2_i[ii]
        pkl4_i_mbin[ii, :] = m_bin @ pkl4_i[ii]

    pl0_i_binning = np.array([pkl0_i_mbin[ii][k_points['P0']] for ii in range(len(pkl0_i))])
    pl2_i_binning = np.array([pkl2_i_mbin[ii][k_points['P2']] for ii in range(len(pkl2_i))])
    pl4_i_binning = np.array([pkl4_i_mbin[ii][k_points['P4']] for ii in range(len(pkl4_i))])
    
    pl02_i_binning_ = np.concatenate((pl0_i_binning, pl2_i_binning, pl4_i_binning), axis=1)
    pl02_i_binning = np.zeros((len(pkl0_i), len(data)))
    
    Npkp = len(k_points['P0'][0]) + len(k_points['P2'][0])+len(k_points['P4'][0])

    pl02_i_binning[:, 0:Npkp] = pl02_i_binning_

    
    if BispBase=='Sugiyama':
        if use_poles['B0'] or use_poles['B2']:
            linear = nonlinear.get_linear(ps['k'], ps['pk'],
                                          pknow=None,cosmo=None,
                                          **kwargs_folps)

            k_pkl_pklnw = np.array([linear['k'], linear['pk_l'], linear['pk_l_NW']])

            bispectrum = folpsD.BispectrumCalculator(model=model)

            B000__, B110, B220, B202__, B022, B112 = bispectrum.Sugiyama_Bl1l2L(
                k1k2pairs=k_ev_bk,
                f=kwargs_folps['f0'],
                bpars=BispParams,
                qpar=q_par, 
                qper=q_perp,
                k_pkl_pklnw=k_pkl_pklnw,
                precision=[10, 15, 15],
                renormalize=True,
                damping=damping,
                interpolation_method='cubic',
                bias_scheme=bias_scheme
            )

            B000_ = folpsD.interp(k_thy, k_ev_bk[:, 0], B000__)
            B202_ = folpsD.interp(k_thy, k_ev_bk[:, 0], B202__)

            B000_const_mbin = m_bin @ B000_ 
            B202_const_mbin = m_bin @ B202_

            pl02_const_binning = np.concatenate(
                (pkl0_const_mbin[k_points['P0']], 
                 pkl2_const_mbin[k_points['P2']], 
                 pkl4_const_mbin[k_points['P4']],
                 B000_const_mbin[k_points['B0']], 
                 B202_const_mbin[k_points['B2']])
            ) 
            
    Npoints = {'Np0': len(k_points['P0'][0]),
                 'Np2': len(k_points['P2'][0]),
                 'Np4': len(k_points['P4'][0]),
                 'Nb0': len(k_points['B0'][0]),
                 'Nb2': len(k_points['B2'][0])}

    output = {
        'pl02_const': pl02_const_binning, 
        'pl02_i': pl02_i_binning,
        'f_growth': f_growth,
        'sigma8': sigma8_global,
        'sigma8_z': sigma8_zev_global,
        'cosmo': cosmo, 
        'A_AP': A_AP,
        'Omega_m': Omega_m,
        'q_par': q_par,
        'q_per': q_perp,
        'Plin': inputpkT,
        'Npoints': Npoints,
    }

    if chatty:
        print(f"model = '{model}', damping = '{damping}'")
        print(f"bsCoev = '{bscoev}', b3Coev = '{b3coev}', BispBase ='{BispBase}'")
        print(f"bias_scheme = '{bias_scheme}', kernels ='{kernels}', use_TNS_model = '{use_TNS_model}'")
        print(f"Npoints = {Npoints}")

    return output
