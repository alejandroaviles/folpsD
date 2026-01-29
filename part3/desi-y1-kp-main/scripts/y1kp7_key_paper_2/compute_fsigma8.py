import sys
from pathlib import Path
import numpy as np
from cosmoprimo.fiducial import DESI
from cosmoprimo import utils

sys.path.insert(1, '../')
from y1_fs_cosmo_tools import load_cobaya_samples
dirname = Path('base_mnu_bao_cmb_with_fsigma8')

zeff = 0.992
model = 'base_mnu'
bao_cmb = load_cobaya_samples(model=model, dataset=['desi-v1.5-bao-all', 'planck2018-lowl-TT-clik', 'planck2018-lowl-EE-clik', 'planck2018-highl-plik-TTTEEE', 'planck-act-dr6-lensing'],
                             label=r'BAO + CMB', thin=200)
indices = range(bao_cmb.weights.size)
fiducial = DESI(engine='camb', neutrino_hierarchy='degenerate')
fsigma8 = []
for index in indices:
    if index % 10 == 0: print(index)
    params = {param: bao_cmb[param][index] for param in ['ombh2', 'omch2', 'H0', 'logA', 'ns', 'mnu']}
    #fsigma8.append(0.)
    cosmo = fiducial.clone(**params)
    fsigma8.append(cosmo.get_fourier().sigma8_z(zeff, of='theta_cb'))
fsigma8 = np.array(fsigma8)
bao_cmb.addDerived(fsigma8, 'fsigma8', 'f\sigma_8')
utils.mkdir(dirname)
bao_cmb.saveAsText(str(dirname / 'chain'))