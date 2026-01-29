# NOTE: This code mirrors other automatically generated bindings and
# wraps DESIFSLikelihood in a CobayaLikelihoodFactory for FS+BAO.

# Force use of the local desilike checkout (with FOLPSv2) BEFORE any
# desilike imports.  Remove pre-cached modules so the local version
# takes precedence.
import sys as _sys
_LOCAL_DESILIKE = "/global/cfs/cdirs/desicollab/users/hernannb/__FOLPS_tutorial/desilike"
if _LOCAL_DESILIKE not in _sys.path:
    _sys.path.insert(0, _LOCAL_DESILIKE)
for _mod in list(_sys.modules):
    if _mod == 'desilike' or _mod.startswith('desilike.'):
        del _sys.modules[_mod]

from desilike.bindings.cobaya.factory import CobayaLikelihoodFactory
from desilike.bindings.base import load_from_file

# Load the DESIFSLikelihood defined in bindings_fs_bao.py
DESIFSLikelihood = load_from_file(
    '/global/cfs/cdirs/desicollab/users/hernannb/__FOLPS_tutorial/part3/desi-y1-kp-main/desi_y1_cosmo_bindings/bindings_fs_bao.py',
    'DESIFSLikelihood',
)

# Full-shape + BAO, all tracers.
# We define separate Cobaya likelihoods for different PT theories,
# so that the theory is fixed inside the binding and no extra
# "theory_name" option is required in the Cobaya config.

# Default REPTVelocileptors binding (not used for folpsv2 tests,
# but kept for completeness if ever needed).
desi_fs_bao_all = CobayaLikelihoodFactory(
    DESIFSLikelihood,
    'desi_fs_bao_all',
    kw_like={
        'cosmo': 'external',
        'tracers': None,
        'theory_name': 'reptvelocileptors',
        'data_name': '',
        'observable_name': 'power+bao-recon',
    },
    module=__name__,
    kw_cobaya=None,
)

# FOLPS-AX binding (all tracers)
desi_fs_bao_all_folpsax = CobayaLikelihoodFactory(
    DESIFSLikelihood,
    'desi_fs_bao_all_folpsax',
    kw_like={
        'cosmo': 'external',
        'tracers': None,
        'theory_name': 'folpsax',
        'data_name': '',
        'observable_name': 'power+bao-recon',
    },
    module=__name__,
    kw_cobaya=None,
)

# FOLPSv2 binding (all tracers), FS+BAO
desi_fs_bao_all_folpsv2 = CobayaLikelihoodFactory(
    DESIFSLikelihood,
    'desi_fs_bao_all_folpsv2',
    kw_like={
        'cosmo': 'external',
        'tracers': None,
        'theory_name': 'folpsv2',
        'data_name': '',
        'observable_name': 'power+bao-recon',
    },
    module=__name__,
    kw_cobaya=None,
)

# FOLPSv2 binding (all tracers), FS-only (no BAO part)
desi_fs_all_folpsv2 = CobayaLikelihoodFactory(
    DESIFSLikelihood,
    'desi_fs_all_folpsv2',
    kw_like={
        'cosmo': 'external',
        'tracers': None,
        'theory_name': 'folpsv2',
        'data_name': '',
        'observable_name': 'power',
    },
    module=__name__,
    kw_cobaya=None,
)
