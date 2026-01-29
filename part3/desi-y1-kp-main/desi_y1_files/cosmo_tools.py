import numpy as np


from desilike import ParameterCollection
_bao_params = {'qiso': ParameterCollection({'qiso': {'latex': 'q_{\mathrm{iso}}'}}),
               'qisoqap': ParameterCollection({'qiso': {'latex': 'q_{\mathrm{iso}}'}, 'qap': {'latex': 'q_{\mathrm{ap}}'}}),
               'qparqper': ParameterCollection({'qpar': {'latex': 'q_{\parallel}'}, 'qper': {'latex': 'q_{\perp}'}})}


def get_bao_params(apmode):
    return _bao_params[apmode].deepcopy()


def get_bao_apmode(params):
    apmode = None
    for mode in _bao_params:
        if all(param in params for param in _bao_params[mode]):
            apmode = mode
    if apmode is None:
        raise ValueError('could not find apmode of {}'.format(params))
    return apmode


def convert_bao_fisher(fisher, apmode=None, scale=None, eta=1. / 3.):
    jac = {('qiso', 'qparqper'): np.array([[eta, 1. - eta]]),
           ('qiso', 'qisoqap'): np.array([[1., 0.]]),
           ('qisoqap', 'qparqper'): np.array([[eta, 1. - eta], [1., -1.]]),
           ('qparqper', 'qisoqap'): np.array([[1., 1. - eta], [1., - eta]])}
    for name in _bao_params:
        jac[name, name] = np.eye(len(_bao_params[name]))

    def convert(jac, mean, cov):
        return np.prod(np.array(mean)**jac, axis=-1), jac.dot(cov).dot(jac.T)

    current_apmode = get_bao_apmode(fisher.params())
    if apmode is None:
        apmode = current_apmode
    current_params = get_bao_params(current_apmode)
    params = get_bao_params(apmode)
    try:
        jac = jac[apmode, current_apmode]
    except KeyError:
        raise ValueError('cannot convert from apmode {} to {}'.format(current_apmode, apmode))
    mean, cov = convert(jac, fisher.mean(current_params), fisher.covariance(current_params))
    if scale == 'distance':
        z = fisher.attrs['zeff']
        scale = np.atleast_1d(predict_bao(z, apmode=apmode, scale='distance', eta=eta))
        if apmode == 'qisoqap':  # WARNING: FAP ~ 1 / qap
            mean[1] = 1 / mean[1]
            cov[0, 1] = cov[1, 0] = -cov[0, 1]
        mean = mean * scale
        cov = cov * scale[:, None] * scale
    return fisher.clone(center=mean, hessian=-np.linalg.inv(cov), params=params)
    

def predict_bao(z, apmode='qparqper', cosmo=None, scale=None, eta=1. / 3.):
    # If scale == 'distance', return DM/rd, DH/rd or DV/rd, FAP
    from cosmoprimo.fiducial import DESI
    from cosmoprimo import constants
    fiducial = DESI()
    if cosmo is None: cosmo = DESI()
    
    def predict(cosmo):
        DM_over_rd = cosmo.comoving_angular_distance(z) / cosmo.rs_drag
        DH_over_rd = (constants.c / 1e3) / (100. * cosmo.efunc(z)) / cosmo.rs_drag
        return DH_over_rd, DM_over_rd
    
    DH_over_rd, DM_over_rd = predict(cosmo)
    if scale == 'distance':
        qpar, qper = DH_over_rd, DM_over_rd
        qiso = qpar ** eta * qper ** (1. - eta) * z ** eta
        qap = qper / qpar  # WARNING: FAP ~ 1 / qap
    else:
        DH_over_rd_fid, DM_over_rd_fid = predict(fiducial)
        qpar = DH_over_rd / DH_over_rd_fid
        qper = np.full_like(qpar, fiducial.rs_drag / cosmo.rs_drag)
        mask_z = z > 0.
        qper[mask_z] = DM_over_rd[mask_z] / DM_over_rd_fid[mask_z]
        qiso = qpar ** eta * qper ** (1. - eta)
        qap = qpar / qper
    if apmode == 'qparqper':
        return qpar, qper
    if apmode == 'qisoqap':
        return qiso, qap
    if apmode == 'qiso':
        return qiso,
    if apmode == 'qap':
        return qap,
    raise ValueError('unknown apmode {}'.format(apmode))
    

    
        
