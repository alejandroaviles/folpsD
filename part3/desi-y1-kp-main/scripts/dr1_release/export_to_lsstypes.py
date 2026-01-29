from pathlib import Path
import shutil
import numpy as np

from pycorr import TwoPointCorrelationFunction
from pypower import PowerSpectrumMultipoles, BaseMatrix
from desilike.samples import Profiles
from desilike.observables import ObservableCovariance

from lsstypes import CovarianceMatrix, WindowMatrix, GaussianLikelihood, Mesh2SpectrumPole, Mesh2SpectrumPoles, Count2CorrelationPole, Count2CorrelationPoles, ObservableLeaf, ObservableTree, read, write
from lsstypes.external import from_pypower, from_pycorr


dr_base_dir = Path('/global/cfs/cdirs/desi/users/adematti/dr1_release/dr1-fs-bao-clustering-measurements')

desipipe_base_dir = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe')
desipipe_v12_dir = Path('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/desipipe')


def convert_spectrum(dp_fn, dr_fn):
    spectrum = PowerSpectrumMultipoles.load(dp_fn)
    attrs = {name: np.float64(spectrum.attrs[name]) for name in ['zeff']}
    spectrum = from_pypower(spectrum, complex=False)
    spectrum.attrs.update(attrs)
    spectrum.write(dr_fn)
    if with_txt: spectrum.write(str(dr_fn).replace('.h5', '.txt'))
    if check:
        assert read(dr_fn) == spectrum


def convert_correlation(dp_fn, dr_fn, dr_poles_fn, dr_window_fn=None):
    counts = TwoPointCorrelationFunction.load(dp_fn)
    attrs = {name: np.float64(counts.D1D2.attrs[name]) for name in ['zeff']}
    counts = from_pycorr(counts)
    counts.attrs.update(attrs)
    counts.write(dr_fn)
    if with_txt: counts.write(str(dr_fn).replace('.h5', '.txt'))
    if check:
        assert read(dr_fn) == counts

    correlation, window = counts.select(s=slice(0, None, 4)).project(ells=[0, 2, 4], ignore_nan=True, kw_window=dict(RR=counts.get('RR'), resolution=1))
    correlation.attrs.update(attrs)
    window.attrs.update(correlation.attrs)
    correlation.write(dr_poles_fn)
    if with_txt: correlation.write(str(dr_poles_fn).replace('.h5', '.txt'))
    if check:
        assert read(dr_poles_fn) == correlation
    if dr_window_fn is not None:
        window.write(dr_window_fn)
        if with_txt: window.write(str(dr_window_fn).replace('.h5', '.txt'))
        if check:
            assert read(dr_window_fn) == window


def convert_window(spectrum_fn, dp_fn, dr_fn, rebin=True):
    window = BaseMatrix.load(dp_fn)
    spectrum = read(spectrum_fn)
    norm = window.weight
    observable = []
    for ell, k, weight in zip(window.projsout, window.xout, window.weightsout):
        pole = spectrum.get(ell.ell)
        assert np.allclose(pole.coords('k'), k, equal_nan=True)
        observable.append(Mesh2SpectrumPole(k=k, num_raw=np.zeros_like(k), nmodes=weight, norm=norm * np.ones_like(k), k_edges=pole.edges('k'), ell=ell.ell))
    observable = Mesh2SpectrumPoles(observable)

    if rebin:
        kin = np.arange(0.001, 0.35, 0.001)
        from scipy import linalg

        def matrix_lininterp(xin, xout):
            # Matrix for linear interpolation
            toret = np.zeros((len(xin), len(xout)), dtype='f8')
            for iout, xout in enumerate(xout):
                iin = np.searchsorted(xin, xout, side='right') - 1
                if 0 <= iin < len(xin) - 1:
                    frac = (xout - xin[iin]) / (xin[iin + 1] - xin[iin])
                    toret[iin, iout] = 1. - frac
                    toret[iin + 1, iout] = frac
                elif np.isclose(xout, xin[-1]):
                    toret[iin, iout] = 1.
            return toret

        rebin = linalg.block_diag(*[matrix_lininterp(kin, xin) for xin in window.xin])
        value = window.value.T.dot(rebin.T)  # rebinned window matrix

        theory = []
        for ell in window.projsin:
            theory.append(Mesh2SpectrumPole(k=kin, num_raw=np.zeros_like(kin), ell=ell.ell))
        theory = Mesh2SpectrumPoles(theory)

    else:

        value = window.value.T
        theory = []
        for ell, k in zip(window.projsin, window.xin):
            theory.append(Mesh2SpectrumPole(k=k, num_raw=np.zeros_like(k), ell=ell.ell))
        theory = Mesh2SpectrumPoles(theory)

    attrs = {name: np.float64(window.attrs[name]) for name in ['zeff']}
    window = WindowMatrix(value=value, observable=observable, theory=theory)
    window.attrs.update(attrs)
    window.write(dr_fn)
    if with_txt: window.write(str(dr_fn).replace('.h5', '.txt'))
    if check:
        assert window.value().shape == (window.observable.size, window.theory.size)
        assert read(dr_fn) == window


def _convert_covariance(covariance, with_attrs=False):

    from cosmoprimo.fiducial import DESI
    from cosmoprimo import constants
    fiducial = DESI()

    def _get_attrs(observable):
        if with_attrs:
            return {name: np.float64(observable.attrs[name]) for name in ['zeff']}
        return {}

    def _get_fiducial(parameter, zeff=None):
        toret = {}
        if zeff is not None:
            DM_over_rd_fid = fiducial.comoving_angular_distance(zeff) / fiducial.rs_drag
            DH_over_rd_fid = (constants.c / 1e3) / (100. * fiducial.efunc(zeff)) / fiducial.rs_drag
            DV_over_rd_fid = DM_over_rd_fid**(2. / 3.) * DH_over_rd_fid**(1. / 3.) * zeff**(1. / 3.)
            FAP_fid = DM_over_rd_fid / DH_over_rd_fid
            if parameter == 'qpar':
                toret['DH_over_rd_fid'] = DH_over_rd_fid
            if parameter == 'qper':
                toret['DM_over_rd_fid'] = DM_over_rd_fid
            if parameter == 'qiso':
                toret['DV_over_rd_fid'] = DV_over_rd_fid
        return toret

    def convert_spectrum(observable):
        spectrum = []
        attrs = _get_attrs(observable)
        for ell, k, edges, nmodes, value in zip(observable.projs, observable._x, observable._edges, observable._weights, observable._value):
            edges = np.column_stack([edges[:-1], edges[1:]])
            spectrum.append(Mesh2SpectrumPole(k=k, k_edges=edges, nmodes=nmodes, num_raw=value, ell=ell, attrs=attrs))
        return Mesh2SpectrumPoles(spectrum, attrs=attrs)

    def convert_correlation(observable):
        correlation = []
        attrs = _get_attrs(observable)
        for ell, s, edges, weights, value in zip(observable.projs, observable._x, observable._edges, observable._weights, observable._value):
            edges = np.column_stack([edges[:-1], edges[1:]])
            correlation.append(Count2CorrelationPole(s=s, s_edges=edges, value=value, ell=ell, attrs=attrs))
        return Count2CorrelationPoles(correlation, attrs=attrs)

    def convert_compressed(observable):
        leaves, names = [], []
        attrs = _get_attrs(observable)
        for proj, value in zip(observable.projs, observable._value):
            names.append(proj)
            assert proj in ['qiso', 'qap', 'qpar', 'qper', 'df', 'dm']
            leaf = ObservableLeaf(value=np.atleast_1d(value), attrs=attrs | _get_fiducial(proj, **attrs))
            leaves.append(leaf)
        return ObservableTree(leaves, parameters=names, attrs=attrs)

    observables, names = [], []
    for observable in covariance.observables():
        if observable.name == 'power':
            names.append('spectrum')
            observables.append(convert_spectrum(observable))
        elif observable.name == 'correlation':
            names.append('correlation')
            observables.append(convert_correlation(observable))
        elif observable.name == 'correlation-recon':
            names.append(observable.name.replace('-', ''))
            observables.append(convert_correlation(observable))
        elif observable.name in ['shapefit', 'bao-recon']:
            names.append(observable.name.replace('-', ''))
            observables.append(convert_compressed(observable))
        elif observable.name in ['shapefit+bao-recon']:
            names.append('shapefit')
            observables.append(convert_compressed(observable))
        else:
            raise NotImplementedError(observable.name)
    if len(observables) > 1:
        observable = ObservableTree(observables, observables=names)
    else:
        observable = observables[0]
    value = covariance.view()
    covariance = CovarianceMatrix(value=value, observable=observable)
    return covariance


def convert_covariance(dp_fn, dr_fn):
    covariance = ObservableCovariance.load(dp_fn)
    covariance = _convert_covariance(covariance)
    covariance.write(dr_fn)
    if with_txt: covariance.write(str(dr_fn).replace('.h5', '.txt'))
    if check:
        assert covariance.value().shape == (covariance.observable.size,) * 2
        assert read(dr_fn) == covariance


def convert_likelihood(dp_fn, dr_fn, observables=('spectrum',), spectrum_fn=None, add_templates=False):
    from scipy import linalg

    likelihood = ObservableCovariance.load(dp_fn)
    covariance = _convert_covariance(likelihood, with_attrs=True)
    observable = covariance.observable.copy()
    templates_attrs = dict(likelihood.attrs)
    window_attrs = likelihood.observables()[0].attrs

    if not hasattr(observable, 'observables'):
        observable = ObservableTree([observable], observables=list(observables))
    covariance = covariance.clone(observable=observable.clone(value=np.zeros_like(observable.value())))

    if 'spectrum' in observable.observables:
        k = window_attrs['kin']
        spectrum = _spectrum = observable.get('spectrum')
        if spectrum_fn is not None:
            spectrum = read(spectrum_fn).match(spectrum)
            assert np.allclose(spectrum.value(), _spectrum.value())
            observable = observable.map(lambda branch, label: spectrum if label['observables'] == 'spectrum' else branch, level=1, input_label=True)
        theory = []
        for ell in [0, 2, 4]:  # theory ells
            theory.append(Mesh2SpectrumPole(k=k, num_raw=np.zeros_like(k), ell=ell))
        theory_spectrum = Mesh2SpectrumPoles(theory)
        theory = ObservableTree([theory_spectrum], observables=['spectrum'])
        value = window_attrs['wmatrix']

        if add_templates:
            for syst in ['photo', 'rotation']:
                value = np.concatenate([value, np.array(templates_attrs[syst]['templates']).T], axis=-1)
                leaf = ObservableLeaf(value=np.zeros(len(templates_attrs[syst]['templates'])))
                leaf._attrs['prior_variance'] = list(templates_attrs[syst]['prior'])
                theory = ObservableTree.join([theory, ObservableTree([leaf], observables=[f'{syst}'])])

        if len(observable.observables) > 1:
            assert observable.observables[0] == 'spectrum'  # make sure it is the first element
            pad_width = covariance.shape[0] - observable.get('spectrum').size
            value = linalg.block_diag(value, np.eye(pad_width, dtype=value.dtype))
            if theory.labels(return_type='keys') == ['ells']:
                theory = ObservableTree([theory_spectrum], observables=['spectrum'])
            theory = ObservableTree.join([theory, observable.get(observable.observables[1:])])
            theory = theory.clone(value=np.zeros_like(theory.value()))

        window = WindowMatrix(value, observable=observable.clone(value=np.zeros_like(observable.value())), theory=theory)

    else:
        value = np.eye(observable.size, dtype=observable.value().dtype)
        window = WindowMatrix(value, observable=observable.clone(value=np.zeros_like(observable.value())), theory=observable.clone(value=np.zeros_like(observable.value())))

    likelihood = GaussianLikelihood(observable=observable, window=window, covariance=covariance)
    likelihood.write(dr_fn)
    if with_txt: likelihood.write(str(dr_fn).replace('.h5', '.txt'))
    if check:
        for observable in likelihood.observable:
            observable.attrs['zeff']
        assert likelihood.covariance.value().shape == (likelihood.covariance.observable.size,) * 2 ==  (likelihood.window.observable.size,) * 2 == (likelihood.observable.size,) * 2
        assert likelihood.window.value().shape == (likelihood.observable.size, likelihood.window.theory.size)
        assert read(dr_fn) == likelihood


def convert_template(spectrum_fn, dp_fn, dr_fn, rebin=1, klim=(0., np.inf)):
    from desi_y1_files.systematic_template import PolynomialTemplate
    template = PolynomialTemplate.load(dp_fn)
    template.ells = np.array([ell for ell in template.ells])
    #template.save(dp_fn)
    spectrum = read(spectrum_fn).select(k=slice(0, None, rebin)).select(k=klim)
    poles = []
    for label, pole in spectrum.items(level=1):
        value = template(label['ells'], pole.coords('k'))
        value = Mesh2SpectrumPole(num_raw=value, k=pole.coords('k'), k_edges=pole.edges('k'), ell=label['ells'])
        poles.append(value)
    poles = Mesh2SpectrumPoles(poles, ells=spectrum.ells)
    if 'prior_cov' in template.attrs:
        poles.attrs['prior_variance'] = template.attrs['prior_cov']
    poles.write(dr_fn)

"""
def convert_rotation(spectrum_fn, dp_fn, dr_fn, rebin=1, klim=(0., np.inf)):
    from desi_y1_files import WindowRotation
    rotation = WindowRotation.load(dp_fn)

    spectrum = read(spectrum_fn).select(k=slice(0, None, rebin)).select(k=klim)
    for ell, k in rotation.kout.items():
        assert np.allclose(spectrum.get(ell).coords('k'), k)
    
    observable = spectrum
    prior = rotation.marg_prior_mo
    mo = ObservableTree([ObservableLeaf(value=-np.atleast_1d(value)) for value in prior], ells=list(rotation.kout))
    theory = ObservableTree([observable, mo], observables=['spectrum', 'mo']) 

    value = np.concatenate([rotation.mmatrix[0], np.array(rotation.mmatrix[1]).T], axis=1)
    matrix = WindowMatrix(theory=theory, observable=observable, value=value)
    matrix.write(dr_fn)
"""

def convert_rotation(window_fn, dp_fn, dr_fn, rebin=1, klim=(0., np.inf)):
    from desi_y1_files import WindowRotation
    rotation = WindowRotation.load(dp_fn)

    window = read(window_fn)
    observable = window.observable
    kinlim = (observable.get(0).coords('k')[0] / 2, 0.5)
    observable = observable.select(k=slice(0, None, rebin)).select(k=klim)
    window = window.at.observable.match(observable)
    for ell, k in rotation.kout.items():
        assert np.allclose(observable.get(ell).coords('k'), k)

    state = {}
    state['M'] = WindowMatrix(theory=observable, observable=observable, value=rotation.mmatrix[0])
    state['observable'] = observable
    state['theory'] = window.theory.select(k=kinlim)
    state['mo'] = ObservableTree([ObservableLeaf(value=np.atleast_1d(value)) for value in rotation.mmatrix[1]], ells=list(observable.ells))
    state['mt'] = ObservableTree([ObservableLeaf(value=np.atleast_1d(value)) for value in rotation.mmatrix[2]], ells=list(observable.ells))
    state['s'] = ObservableTree([ObservableLeaf(value=np.atleast_1d(value)) for value in rotation.marg_prior_mo], ells=list(observable.ells))
    write(dr_fn, state)


def convert_bao_profiles(dp_fn, dr_fn):
    profiles = Profiles.load(dp_fn)
    iso_bao = 'qap' not in profiles.bestfit.names(varied=True)
    params = profiles.bestfit.params(varied=True).select(basename=['qiso'] if iso_bao else ['qiso', 'qap'])
    values = profiles.bestfit.choice(index='argmax', params=params, return_type='nparray')
    assert len(profiles.bestfit) == 9
    parameters = params.basenames()
    attrs = {'zeff': profiles.attrs['zeff']}
    if not iso_bao:
        qiso, qap = values
        values = [qiso * qap**(2. / 3.), qiso * qap**(-1. / 3.)]
        parameters = ['qpar', 'qper']
    leaves = []
    for value in values:
        leaf = ObservableLeaf(value=np.atleast_1d(value), attrs=attrs)
        leaves.append(leaf)
    tree = ObservableTree(leaves, parameters=parameters, attrs=attrs)
    tree.write(dr_fn)
    


if __name__ == '__main__':

    check = True
    with_txt = False

    #export = ['data']
    #export = ['ezmock']
    #export = ['abacus']
    export = ['readme', 'cobaya']
    

    list_zrange = [('BGS_BRIGHT-21.5', (0.1, 0.4)),
                    ('LRG', (0.4, 0.6)),
                    ('LRG', (0.6, 0.8)),
                    ('LRG', (0.8, 1.1)),
                    ('ELG_LOPnotqso', (1.1, 1.6)),
                    ('QSO', (0.8, 2.1))]
    list_zrange_bao = [('BGS_BRIGHT-21.5', (0.1, 0.4)),
                        ('LRG', (0.4, 0.6)),
                        ('LRG', (0.6, 0.8)),
                        ('LRG', (0.8, 1.1)),
                        ('LRG+ELG_LOPnotqso', (0.8, 1.1)),
                        ('ELG_LOPnotqso', (1.1, 1.6)),
                        ('QSO', (0.8, 2.1))]

    regions = ['NGC', 'SGC', 'GCcomb']

    if 'readme' in export:
        dr_base_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy('README.md', dr_base_dir)
        shutil.copy('create_fiducial_likelihood.py', dr_base_dir)

    if 'cobaya' in export:
        src = '/global/u2/a/adematti/cosmodesi/desi-y1-kp/desi_y1_cosmo_bindings/cobaya_public_likelihoods/fs_bao_likelihoods/'
        shutil.copytree(src, dr_base_dir / 'data/likelihood/cobaya_likelihoods', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))

    if 'data' in export:

        for tracer, zrange in list_zrange:
            for region in regions:
                # Power spectrum
                for thetacut in ['', '_thetacut0.05']:
                    # raw
                    dp_spectrum_fn = f'pkpoles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                    dr_spectrum_fn = f'spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'

                    dp_dir = desipipe_base_dir / 'baseline_2pt/pk'
                    dr_dir = dr_base_dir / 'data/spectrum'
                    convert_spectrum(dp_dir / dp_spectrum_fn, dr_dir / dr_spectrum_fn)
                    raw_spectrum_fn = dr_dir / dr_spectrum_fn

                    dp_window_fn = f'wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                    dr_window_fn = f'window_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                    convert_window(raw_spectrum_fn, dp_dir / dp_window_fn, dr_dir / dr_window_fn, rebin=False)
                    raw_window_fn = dr_dir / dr_window_fn

                    dp_dir = desipipe_base_dir / 'templates_2pt'
                    dr_dir = dr_base_dir / 'data/templates_spectrum'

                    for syst in ['photo', 'ric', 'aic']:
                        drsyst = {'aic': 'amr'}.get(syst, syst)
                        dp_template_fn = f'template_{syst}_power_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin{thetacut}.npy'
                        dr_template_fn = f'template_{drsyst}_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                        convert_template(raw_spectrum_fn, dp_dir / dp_template_fn, dr_dir / dr_template_fn, rebin=5, klim=(0.02, 0.4))

                    if region in ['GCcomb']:
                        # corrected
                        dp_spectrum_fn = f'pkpoles_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                        dr_spectrum_fn = f'spectrum-poles-corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                        dp_dir = desipipe_base_dir / 'baseline_2pt/pk/corrected'
                        dr_dir = dr_base_dir / 'data/spectrum'
                        convert_spectrum(dp_dir / dp_spectrum_fn, dr_dir / dr_spectrum_fn)

                        if thetacut:
                            dp_spectrum_fn = f'pkpoles_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                            dr_spectrum_fn = f'spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                            dp_dir = desipipe_base_dir / 'baseline_2pt/pk/rotated'
                            dr_dir = dr_base_dir / 'data/spectrum'
                            convert_spectrum(dp_dir / dp_spectrum_fn, dr_dir / dr_spectrum_fn)

                            dp_window_fn = f'wmatrix_smooth_rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                            dr_window_fn = f'window_spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                            convert_window(dr_dir / dr_spectrum_fn, dp_dir / dp_window_fn, dr_dir / dr_window_fn, rebin=False)
    
                            dp_spectrum_fn = f'pkpoles_rotated_corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                            dr_spectrum_fn = f'spectrum-poles-rotated-corrected_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                            dp_dir = desipipe_base_dir / 'baseline_2pt/pk/rotated_corrected'
                            dr_dir = dr_base_dir / 'data/spectrum'
                            convert_spectrum(dp_dir / dp_spectrum_fn, dr_dir / dr_spectrum_fn)

                            dp_dir = desipipe_base_dir / 'templates_2pt/rotated'
                            dr_dir = dr_base_dir / 'data/templates_spectrum'
        
                            for syst in ['photo', 'ric', 'aic']:
                                drsyst = {'aic': 'amr'}.get(syst, syst)
                                dp_template_fn = f'template_rotated_{syst}_power_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin{thetacut}.npy'
                                dr_template_fn = f'template_{drsyst}_spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                                convert_template(raw_spectrum_fn, dp_dir / dp_template_fn, dr_dir / dr_template_fn, rebin=5, klim=(0.02, 0.4))

                            dp_dir = desipipe_base_dir / '2pt/pk/rotated'
                            dr_dir = dr_base_dir / 'data/rotation'
                            dp_rotation_fn = list(dp_dir.glob(f'rotation_wmatrix_smooth_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin_nran*_cellsize*_boxsize*{thetacut}.npy'))
                            assert len(dp_rotation_fn) == 1
                            dp_rotation_fn = dp_rotation_fn[0]
                            dr_rotation_fn = f'rotation_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                            convert_rotation(raw_window_fn, dp_rotation_fn, dr_dir / dr_rotation_fn, rebin=5, klim=(0., 0.4))

        for tracer, zrange in list_zrange:
            for region in ['GCcomb']:
                # Power spectrum
                for thetacut in ['', '_thetacut0.05']:
                    dp_dir = desipipe_base_dir / 'cov_2pt' / 'ezmock/v1'
                    dr_dir = dr_base_dir / 'data/covariance/EZmock'
                    dp_covariance_fn = f'covariance_power_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin{thetacut}.npy'
                    dr_covariance_fn = f'covariance_spectrum-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                    convert_covariance(dp_dir / dp_covariance_fn, dr_dir / dr_covariance_fn)

                    dp_covariance_fn = f'covariance_power+bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin{thetacut}.npy'
                    dr_covariance_fn = f'covariance_spectrum-poles+bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                    convert_covariance(dp_dir / dp_covariance_fn, dr_dir / dr_covariance_fn)

                    dp_covariance_fn = f'covariance_power+correlation-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin{thetacut}.npy'
                    dr_covariance_fn = f'covariance_spectrum-poles+correlation-poles-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                    convert_covariance(dp_dir / dp_covariance_fn, dr_dir / dr_covariance_fn)

                    if thetacut:
                        dp_dir = desipipe_base_dir / 'cov_2pt' / 'ezmock/v1/rotated'
                        dr_dir = dr_base_dir / 'data/covariance/EZmock'
                        dp_covariance_fn = f'covariance_rotated_marg-no_power_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin{thetacut}.npy'
                        dr_covariance_fn = f'covariance_spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                        convert_covariance(dp_dir / dp_covariance_fn, dr_dir / dr_covariance_fn)
                        covariance_ref = read(dr_dir / dr_covariance_fn)

                        dp_covariance_fn = f'covariance_rotated_marg-no_power+bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin{thetacut}.npy'
                        dr_covariance_fn = f'covariance_spectrum-poles-rotated+bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                        convert_covariance(dp_dir / dp_covariance_fn, dr_dir / dr_covariance_fn)

                        dp_dir = desipipe_base_dir / 'cov_2pt' / 'syst/v1.5/rotated'
                        dr_dir = dr_base_dir / 'data/covariance/syst'
                        dp_covariance_fn = f'covariance_syst-hod_rotated_power_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin{thetacut}.npy'
                        dr_covariance_fn = f'covariance_hod_spectrum-poles-rotated_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                        covariance = _convert_covariance(ObservableCovariance.load(dp_dir / dp_covariance_fn))
                        covariance = covariance_ref.at.observable.match(covariance_ref.observable.get(ells=[0, 2]).select(k=(0.02, 0.2))).clone(value=covariance.value())
                        covariance.write(dr_dir / dr_covariance_fn)

        for tracer, zrange in list_zrange:
            for region in ['GCcomb']:
                # likelihood
                dp_dir = desipipe_base_dir / 'forfit_2pt'
                dr_dir = dr_base_dir / 'data/likelihood'

                dr_spectrum_fn = dr_base_dir / 'data/spectrum' / f'spectrum-poles-rotated-corrected_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'

                dp_fn = f'forfit_power_syst-no_klim_0-0.02-0.20_2-0.02-0.20_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                dr_fn = f'likelihood_spectrum-poles-rotated_stat-only_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
                #convert_likelihood(dp_dir / dp_fn, dr_dir / dr_fn, add_templates=True, spectrum_fn=dr_spectrum_fn)

                dp_fn = f'forfit_power_syst-hod_klim_0-0.02-0.20_2-0.02-0.20_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                dr_fn = f'likelihood_spectrum-poles-rotated_syst-hod_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
                convert_likelihood(dp_dir / dp_fn, dr_dir / dr_fn, add_templates=True, spectrum_fn=dr_spectrum_fn)

                dp_fn = f'forfit_power_syst-rotation_klim_0-0.02-0.20_2-0.02-0.20_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                dr_fn = f'likelihood_spectrum-poles-rotated_syst-rotation_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
                #convert_likelihood(dp_dir / dp_fn, dr_dir / dr_fn, add_templates=True, spectrum_fn=dr_spectrum_fn)

                dp_fn = f'forfit_power_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                dr_fn = f'likelihood_spectrum-poles-rotated_syst-rotation-hod-photo_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
                convert_likelihood(dp_dir / dp_fn, dr_dir / dr_fn, spectrum_fn=dr_spectrum_fn)
                
                dp_fn = f'forfit_power+bao-recon_syst-hod_klim_0-0.02-0.20_2-0.02-0.20_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                dr_fn = f'likelihood_spectrum-poles-rotated+bao-recon_syst-hod_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
                convert_likelihood(dp_dir / dp_fn, dr_dir / dr_fn, add_templates=True, spectrum_fn=dr_spectrum_fn)

                dp_fn = f'forfit_power+bao-recon_syst-rotation-hod-photo_klim_0-0.02-0.20_2-0.02-0.20_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                dr_fn = f'likelihood_spectrum-poles-rotated+bao-recon_syst-rotation-hod-photo_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
                convert_likelihood(dp_dir / dp_fn, dr_dir / dr_fn, spectrum_fn=dr_spectrum_fn)

                # ShapeFit
                dp_fn = f'forfit_shapefit_power_syst-rotation-hod-photo_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                dr_fn = f'likelihood_shapefit_spectrum-poles-rotated_syst-rotation-hod-photo_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
                #convert_likelihood(dp_dir / dp_fn, dr_dir / dr_fn, observables=['shapefit'])

                dp_fn = f'forfit_shapefit_power+bao-recon_syst-rotation-hod-photo_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                dr_fn = f'likelihood_shapefit_spectrum-poles-rotated+bao-recon_syst-rotation-hod-photo_{tracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_thetacut0.05.h5'
                convert_likelihood(dp_dir / dp_fn, dr_dir / dr_fn, observables=['shapefit'])

        for (_desipipe_base_dir, _dr_base_dir) in [(desipipe_base_dir, dr_base_dir / 'data'), (desipipe_v12_dir, dr_base_dir / 'data_v1.2')]:
            for tracer, zrange in list_zrange_bao:
                for region in regions:
                    # BAO reconstruction
                    recon = 'recon_recsym'
                    if 'LRG+ELG' in tracer: recon = 'recon_recsym_z0.8-1.1'
                    if 'QSO' in tracer: recon = 'recon_recsym_z0.8-2.1'
                    dp_dir = _desipipe_base_dir / 'baseline_2pt' / recon / 'xi/smu'
                    dr_dir = _dr_base_dir / 'recsym/correlation'
                    dp_counts_fn = f'allcounts_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                    dr_counts_fn = f'counts-recsym-smu_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
                    dr_correlation_fn = f'correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
                    dr_window_fn = f'window_correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
                    observable_fn = dr_dir / dr_correlation_fn
                    window_fn = dr_dir / dr_window_fn
                    convert_correlation(dp_dir / dp_counts_fn, dr_dir / dr_counts_fn, dr_poles_fn=observable_fn, dr_window_fn=window_fn)
    
                    dp_dir = _desipipe_base_dir / 'cov_2pt' / 'rascalc' / ('v1.2' if 'v1.2' in str(_desipipe_base_dir) else 'v1.5')
                    dr_dir = _dr_base_dir / 'covariance/RascalC'
    
                    #dp_covariance_fn = 'covariance_correlation_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin.npy
                    #dr_covariance_fn = f'covariance_correlation-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
                    #convert_covariance(dp_dir / dp_covariance_fn, dr_dir / dr_covariance_fn)
    
                    dp_covariance_fn = f'covariance_correlation-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin.npy'
                    dr_covariance_fn = f'covariance_correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
                    covariance_fn = dr_dir / dr_covariance_fn
                    convert_covariance(dp_dir / dp_covariance_fn, covariance_fn)
    
                    dr_dir = _dr_base_dir / 'likelihood'
                    dr_fn = f'likelihood_correlation-recon-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
                    likelihood = GaussianLikelihood(observable=read(observable_fn), covariance=read(covariance_fn), window=read(window_fn))
                    likelihood = likelihood.at.observable.get(ells=[0] if tracer in ('BGS_BRIGHT-21.5', 'QSO') else [0, 2])
                    likelihood = likelihood.at.observable.select(s=(50., 150.))
                    likelihood.write(dr_dir / dr_fn)

                for region in ['GCcomb']:
                    dp_dir = _desipipe_base_dir / 'forfit_2pt'
                    dr_dir = _dr_base_dir / 'likelihood'
                    dp_covariance_fn = f'covariance_stat_bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                    dr_covariance_fn = f'likelihood_bao-recon_stat-only_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
                    convert_likelihood(dp_dir / dp_covariance_fn, dr_dir / dr_covariance_fn, observables=['baorecon'])
                    likelihood_bao_stat = read(dr_dir / dr_covariance_fn)

                    dp_covariance_fn = f'covariance_syst_bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                    dr_covariance_fn = f'likelihood_bao-recon_syst_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
                    convert_likelihood(dp_dir / dp_covariance_fn, dr_dir / dr_covariance_fn, observables=['baorecon'])
                    likelihood_bao_syst = read(dr_dir / dr_covariance_fn)
                    # Add stat + syst
                    likelihood_bao_syst = likelihood_bao_syst.clone(covariance=likelihood_bao_syst.covariance.clone(value=likelihood_bao_stat.covariance.value() + likelihood_bao_syst.covariance.value()))
                    likelihood_bao_syst.write(dr_dir / dr_covariance_fn)

            for tracer, zrange in [('Lya', (1.8, 4.2))]:
                for region in ['GCcomb']:
                    dp_dir = _desipipe_base_dir / 'forfit_2pt'
                    dr_dir = _dr_base_dir / 'likelihood'
                    dp_covariance_fn = f'covariance_stat_bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                    #dp_covariance_fn = f'forfit_bao-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                    dr_covariance_fn = f'likelihood_bao_syst_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.h5'
                    convert_likelihood(dp_dir / dp_covariance_fn, dr_dir / dr_covariance_fn, observables=['baorecon'])
                    likelihood = read(dr_dir / dr_covariance_fn)
                    print(likelihood.observable.value())
                    

    if 'ezmock' in export:
        imocks = list(range(1, 1001))

        def get_ezmock_dir(tracer, imock, base='baseline_2pt'):
            base_dir =  Path(f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/EZmock/desipipe/{"BGS_v1" if "BGS" in tracer else "v1"}/ffa') / base
            if imock == 'merged':
                return base_dir / 'merged'
            return base_dir / f'mock{imock:d}'

        for tracer, zrange in list_zrange:
            for region in regions:
                dptracer = {'BGS_BRIGHT-21.5': 'BGS', 'ELG_LOPnotqso': 'ELG_LOP'}.get(tracer, tracer)

                # Power spectrum
                for imock in imocks:
                    desipipe_ezmock_dir = get_ezmock_dir(tracer, imock, base='baseline_2pt')
                    for thetacut in ['', '_thetacut0.05']:
                        # raw
                        dp_spectrum_fn = f'pkpoles_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                        dr_spectrum_fn = f'spectrum-poles_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}_{imock:d}.h5'
    
                        dp_dir = desipipe_ezmock_dir / 'pk'
                        dr_dir = dr_base_dir / 'EZmock/ffa/spectrum'
                        #convert_spectrum(dp_dir / dp_spectrum_fn, dr_dir / dr_spectrum_fn)
                        if imock == 1:
                            # Window matrix
                            dp_merged_dir = get_ezmock_dir(tracer, 'merged', base='baseline_2pt') / 'pk'
                            raw_spectrum_fn = dr_dir / dr_spectrum_fn
                            dp_window_fn = f'wmatrix_smooth_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                            dr_window_fn = f'window_spectrum-poles_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                            convert_window(raw_spectrum_fn, dp_merged_dir / dp_window_fn, dr_dir / dr_window_fn, rebin=False)

                    # Post-recon correlation function
                    recon = 'recon_recsym'
                    dp_dir = desipipe_ezmock_dir / recon / 'xi/smu'
                    dr_dir = dr_base_dir / 'EZmock/ffa/recsym/correlation'

                    dptracer = {'BGS_BRIGHT-21.5': 'BGS', 'ELG_LOPnotqso': 'ELG_LOP'}.get(tracer, tracer)
                    dp_counts_fn = f'allcounts_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                    dr_counts_fn = f'counts-recsym-smu_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{imock:d}.h5'
                    dr_correlation_fn = f'correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{imock:d}.h5'
                    observable_fn = dr_dir / dr_correlation_fn
                    #convert_correlation(dp_dir / dp_counts_fn, dr_dir / dr_counts_fn, dr_poles_fn=observable_fn)

                    # BAO measurement
                    bao_name = 'fits_correlation_dampedbao_bao-qisoqap_pcs2/recon_IFFT_recsym_sm15'
                    if 'BGS' in tracer: bao_name = 'fits_correlation_dampedbao_bao-qiso_pcs2/recon_IFFT_recsym_sm15'
                    if 'QSO' in tracer: bao_name = 'fits_correlation_dampedbao_bao-qiso_pcs2/recon_IFFT_recsym_sm30'
                    fn = list((get_ezmock_dir(tracer, imock, base='fits_2pt') / bao_name).glob(f'profiles_{dptracer}_GCcomb_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_cov-rascalc_sigma*_lim_0-50-150*.npy'))
                    assert len(fn) == 1
                    fn = fn[0]
                    dr_dir = dr_base_dir / 'EZmock/ffa/recsym/bao'
                    dr_bao_fn = f'bao-recsym_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{imock:d}.h5'
                    convert_bao_profiles(fn, dr_dir / dr_bao_fn)

    if 'abacus' in export:

        imocks = list(range(25))

        def get_abacus_dir(tracer, fa, imock, base='baseline_2pt'):
            base_dir = Path(f'/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/{"AbacusSummitBGS"  if "BGS" in tracer else "AbacusSummit"}/desipipe/{"v1" if "BGS" in tracer else "v4_2"}') / fa / base
            if imock == 'merged':
                return base_dir / 'merged'
            return base_dir / f'mock{imock:d}'

        for tracer, zrange in list_zrange:
            for region in regions:
                # Power spectrum
                for imock in imocks:
                    for fa in ['complete', 'ffa', 'altmtl']:
                        if fa == 'altmtl':
                            dptracer = tracer
                        else:
                            dptracer = {'ELG_LOPnotqso': 'ELG_LOP'}.get(tracer, tracer)
                        for thetacut in ['', '_thetacut0.05']:
                            desipipe_abacus_dir = get_abacus_dir(tracer, fa, imock, base='baseline_2pt')
                            
                            # raw
                            dp_spectrum_fn = f'pkpoles_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                            dr_spectrum_fn = f'spectrum-poles_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}_{imock:d}.h5'
        
                            dp_dir = desipipe_abacus_dir / 'pk'
                            dr_dir = dr_base_dir / f'AbacusSummit/{fa}/spectrum'
                            convert_spectrum(dp_dir / dp_spectrum_fn, dr_dir / dr_spectrum_fn)
    
                            if imock == 0:
                                # Window matrix
                                dp_merged_dir = get_abacus_dir(tracer, fa, 'merged', base='baseline_2pt') / 'pk'
                                raw_spectrum_fn = dr_dir / dr_spectrum_fn
                                dp_window_fn = f'wmatrix_smooth_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.npy'
                                dr_window_fn = f'window_spectrum-poles_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}{thetacut}.h5'
                                convert_window(raw_spectrum_fn, dp_merged_dir / dp_window_fn, dr_dir / dr_window_fn, rebin=False)

                        # Post-recon correlation function
                        recon = 'recon_recsym'
                        dp_dir = desipipe_abacus_dir / recon / 'xi/smu'
                        dr_dir = dr_base_dir / f'AbacusSummit/{fa}/recsym/correlation'
    
                        dp_counts_fn = f'allcounts_{dptracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.npy'
                        dr_counts_fn = f'counts-recsym-smu_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{imock:d}.h5'
                        dr_correlation_fn = f'correlation-recsym-poles_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{imock:d}.h5'
                        observable_fn = dr_dir / dr_correlation_fn
                        convert_correlation(dp_dir / dp_counts_fn, dr_dir / dr_counts_fn, dr_poles_fn=observable_fn)
    
