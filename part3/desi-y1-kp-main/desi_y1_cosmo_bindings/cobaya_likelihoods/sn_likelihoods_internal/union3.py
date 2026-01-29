import os
import numpy as np
from .pantheonplus import pantheonplus


class union3(pantheonplus):
    """
    Likelihood for the Union3 & UNITY1.5 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/pdf/2311.12098.pdf
    """
    def init_params(self, ini):
        self.twoscriptmfit = False
        #self.pecz = 0.
        #self.has_third_var = False
        data_file = os.path.normpath(os.path.join(self.path, ini.string("data_file")))
        self._read_data_file(data_file)
        self.covs = {}
        for name in ['mag']:
            self.log.debug('Reading covmat for: %s ' % name)
            self.covs[name] = self._read_covmat(os.path.join(self.path, ini.string('%s_covmat_file' % name)))
        self.alphabeta_covmat = False
        zmask = self.zcmb > self.zmin
        for col in self.cols + ['zhel']:
            setattr(self, col, getattr(self, col)[zmask])
        for name, cov in self.covs.items():
            self.covs[name] = cov[np.ix_(zmask, zmask)]
        self.pre_vars = 0.  # diagonal component
        self.inverse_covariance_matrix()
        if not self.use_abs_mag:
            self._marginalize_abs_mag()
        self.marginalize = False

    def _read_data_file(self, data_file):
        self.log.debug('Reading %s' % data_file)
        oldcols = ['zcmb', 'mu']
        self.cols = ['zcmb', 'mag']
        with open(data_file, 'r') as f:
            lines = f.readlines()
            line = lines[0]
            cols = [col.strip().lower() for col in line[1:].split()]
            indices = [cols.index(col) for col in oldcols]
            zeros = np.zeros(len(lines) - 1)
            for col in self.cols:
                setattr(self, col, zeros.astype(dtype='f8'))
            for ix, line in enumerate(lines[1:]):
                vals = [val.strip() for val in line.split()]
                vals = [vals[i] for i in indices]
                for i, (col, val) in enumerate(zip(self.cols, vals)):
                    tmp = getattr(self, col)
                    tmp[ix] = np.asarray(val, dtype=tmp.dtype)
        self.nsn = ix + 1
        self.log.debug('Number of SN read: %s ' % self.nsn)
        self.zhel = self.zcmb