
#updated desilike path
import sys
sys.path.insert(0, "/global/cfs/cdirs/desicollab/users/hernannb/__FOLPS_tutorial/desilike")

#import necessary packages
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BAOCompressionObservable 
from desilike.observables import ObservableCovariance
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import DESI


######### Settings ######### 
kernels = "eds"          #"eds" or "fk"
prior_basis = 'standard' #'physical_prior_doc' 
k_max = 0.20             #max. wavenumber 
pt_model = "EFT"         # 'EFT', 'TNS' or 'FolpsD'
set_emulator = True      #True or False
GR_criteria = 0.01       # R - 1 < GR_criteria 




# List of tracers you want to include in the analysis
tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']  # Add more tracers as needed


# derived parameters to be included in the chain
cosmo = Cosmoprimo(engine='class')
cosmo.init.params['H0'] = dict(derived=True)
cosmo.init.params['Omega_m'] = dict(derived=True)
cosmo.init.params['sigma8_m'] = dict(derived=True) 
fiducial = DESI() #fiducial cosmologyy


#Update cosmo priors
for param in ['n_s', 'h', 'omega_cdm', 'omega_b', 'logA', 'tau_reio','m_ncdm']:
    cosmo.params[param].update(fixed = False)
    if param == 'tau_reio':
        cosmo.params[param].update(fixed = True)
    if param == 'n_s':
            #cosmo.params[param].update(fixed = True)
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042}) #ns10 planck
    if param == 'omega_b': 
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02218, 'scale': 0.00055}) #bbn prior
    if param == 'h':
            cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.2, 1.]})
    if param == 'omega_cdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.01, 0.99]})
    if param == 'm_ncdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.0, 5]})
    if param == 'logA':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [1.61, 3.91]})




#template and theory
template = DirectPowerSpectrumTemplate(fiducial = fiducial, cosmo = cosmo, z=z) #cosmology and fiducial cosmology defined above
theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template, prior_basis=prior_basis, kernels=kernels) #Add the prior_basis='physical' argument to use physically motivated priors

#Update bias and EFT priors
if prior_basis == 'physical':
    theory.params['b1p'].update(prior = {'dist':'uniform','limits': [0, 3]})
    theory.params['b2p'].update(prior = {'dist':'norm','loc': 0, 'scale': 5 })
    theory.params['bsp'].update(prior = {'dist': 'norm', 'loc': 0, 'scale': 5})
    theory.params['alpha0p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
    theory.params['alpha2p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
    theory.params['alpha4p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_EFT})
    theory.params['sn0p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN0})
    theory.params['sn2p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': width_SN2})
    if pt_model=='EFT':
        theory.params['X_FoG_pp'].update(fixed=True)
    else:
        theory.params['X_FoG_pp'].update(prior = {'dist':'uniform','limits': [0, 10]})
  


# In[ ]:


#Define a function to create an observable
def create_observable(data_fn, theory, tracer_index):
    #Load and process covariance
    covariance = ObservableCovariance.load(data_fn)
    #covariance = covariance.select(xlim=(0.02, k_max), projs=[0, 2])
    
    #power spectrum
    data = covariance.observables('power')

    indices = np.where((data.flatx > 0.02) & (data.flatx < k_max))[0]  #new-line
    
    #Create and return the observable: following  https://desi.lbl.gov/trac/wiki/keyprojects/y1kp3/clusteringproducts#a101:Iwanttowritemychi2
    return TracerPowerSpectrumMultipolesObservable(
        data=data,
        covariance=covariance.select(observables=data, select_observables=True),
        klim={ell: [0.02, k_max, 0.005] for ell in [0, 2]},
        theory=theories[tracer_index],
        kin=data.attrs['kin'],
        wmatrix=data.attrs['wmatrix'][indices, :],
        ellsin=data.attrs['ellsin'],
        wshotnoise=data.attrs['wshotnoise'][indices]
    )

#Create observables for each tracer
observables = [create_observable(params['data_fn'], theories, i) 
                for i, params in tracer_params.items()]


# In[ ]:


if set_emulator:
    #Create an emulated theory for each tracer
    for i in range(len(theories)):
        emulator_filename = f'Emulators/emu-fs_klim_0-0.02-{k_max}_2-0.02-{k_max}-{tracers_str}-GCcomb_schoneberg2024-bbn_planck2018-ns10_{prior_basis}-prior_{model}_{pt_model}_{str(tracers[i])}_{kernels}.npy'
        
        if os.path.exists(emulator_filename):
            print(f"FS emulator {i} already exists, loading it now")
            emulator = EmulatedCalculator.load(emulator_filename)  
            theories[i].init.update(pt=emulator)
        else:
            print(f" Computing FS emulator {i}")
            theories[i] = observables[i].wmatrix.theory
            emulator = Emulator(theories[i].pt, engine=TaylorEmulatorEngine(method='finite', order=4))
            emulator.set_samples()
            emulator.fit()
            emulator.save(emulator_filename)
            theories[i].init.update(pt=emulator.to_calculator())
print('FS theories have been emulated successfully' if set_emulator else 'EMULATOR NOT ACTIVATED for FS; proceeding without emulation')


# In[ ]:


#Analytic marginalization over eft and nuisance parameters
for i in range(len(theories)): 
    if prior_basis == 'physical':
        params_list = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
    else:
        params_list = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']

    for param in params_list:    
        theories[i].params[param].update(derived = '.marg')
        
#Rename the eft and nuisance parameters to get a parameter for each tracer (i.e. QSO_alpha0, QSO_alpha2, BGS_alpha0,...)        
for i in range(len(theories)):    
    for param in theories[i].init.params:
        # Update latex just to have better labels
        param.update(namespace='{}'.format(tracers[i]))   


# In[ ]:


flatdatas=[]

def create_bao_observable(data_fn, tracer_index):
    # Create and return the observable
    forfit = ObservableCovariance.load(data_fn)
    covariance, data = forfit.view(observables='bao-recon'), forfit.observables(observables='bao-recon')   
    observable = BAOCompressionObservable(data=data.view(), covariance=covariance, cosmo=cosmo, quantities=data.projs, fiducial=fiducial, z=data.attrs['zeff'])
    flatdata = observable.flatdata
    flatdatas.append(flatdata)
    #observable.init.update(cosmo=cosmo)
    return observable
    
bao_observables = [create_bao_observable(params['data_fn'], i) 
                for i, params in tracer_params.items()]


# In[ ]:


if set_emulator:
    for i in range(len(bao_observables)):
        emulator_filename = f'Emulators/emu-bao_klim_0-0.02-{k_max}_2-0.02-{k_max}-{tracers_str}-GCcomb_schoneberg2024-bbn_planck2018-ns10_{prior_basis}-prior_{model}_{pt_model}_{str(tracers[i])}_{kernels}.npy'
        
        if os.path.exists(emulator_filename):
            print(f"BAO emulator {i} already exists, loading it now")
            emulator = EmulatedCalculator.load(emulator_filename)  
            bao_observables[i] = emulator
        else:
            print(f" Computing BAO emulator {i}")
            emulator = Emulator(bao_observables[i], engine=TaylorEmulatorEngine(method='finite', order=4))
            emulator.set_samples()
            emulator.fit()
            emulator.save(emulator_filename)
            bao_observables[i] = emulator.to_calculator()   
print('BAO has been emulated successfully' if set_emulator else 'EMULATOR NOT ACTIVATED for BAO; proceeding without emulation')


# In[ ]:

setup_logging()
Likelihoods = []
for i in range(len(observables)):
    forfit = ObservableCovariance.load(tracer_params[i]['data_fn'])
    
    data = forfit.observables('power')
    
    indices = np.where((data.flatx > 0.02) & (data.flatx < k_max))[0]
    size_pk = 72
    size_baoxbao = forfit.view().shape[0] - size_pk
    new_indices = indices.tolist() + list(range(size_pk, size_pk + size_baoxbao))
    
    covariance = forfit.view()
    cov = covariance[np.ix_(new_indices, new_indices)]

    #print(size_pk)
    #print(cov.shape)
    likely = ObservablesGaussianLikelihood(observables = [observables[i],bao_observables[i]], covariance = cov)
    likely()
    Likelihoods.append(likely)

    
#Create a likelihood per theory object
#setup_logging()
#Likelihoods = []
#for i in range(len(observables)):
#    forfit = ObservableCovariance.load(tracer_params[i]['data_fn'])
#    covariance = forfit.view()
#    likely = ObservablesGaussianLikelihood(observables = [observables[i],bao_observables[i]], covariance = covariance)
#    likely()
#    Likelihoods.append(likely)


#Run the sampler and save the chain
from desilike.samplers import EmceeSampler, MCMCSampler

if sampler == 'cobaya':
    if restart_chain is False:
        sampler = MCMCSampler(sum(Likelihoods), save_fn = chain_name)
        sampler.run(check={'max_eigen_gr': GR_criteria})
    else:
        sampler = MCMCSampler(sum(Likelihoods) ,save_fn = chain_name, 
                              chains=f'{chain_name}.npy')
        sampler.run(check={'max_eigen_gr': GR_criteria})
    
else:
    if restart_chain is False:
        sampler = EmceeSampler(sum(Likelihoods) ,save_fn = chain_name)
        sampler.run(check={'max_eigen_gr': GR_criteria})
    else:
        sampler = EmceeSampler(sum(Likelihoods) ,save_fn = chain_name, 
                               chains=f'{chain_name}.npy')
        sampler.run(check={'max_eigen_gr': GR_criteria})

