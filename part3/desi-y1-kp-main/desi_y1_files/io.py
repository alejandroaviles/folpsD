import os

import numpy as np

from desipipe import utils
from desipipe.io import BaseFile


def load_covariance_matrix(covariance_fn, binning, select=None, ells=None):

    import numpy as np

    def cut_matrix(cov, xcov, ellscov, xlim):
        """Cut a matrix based on specified indices and returns the resulting submatrix."""
        import numpy as np

        if isinstance(xcov, tuple):
            xmin, xmax, xstep = xcov
            if xmax is None:
                xmax = xmin + xstep * (len(cov) // len(ellscov))
            xcov = np.arange(xmin + xstep / 2., xmax + xstep / 2., xstep)

        assert len(cov) == len(xcov) * len(ellscov), 'Input matrix {} has size {}, different than {} x {}'.format(covariance_fn, len(cov), len(xcov), len(ellscov))
        indices = []
        for ell, xlim in xlim.items():
            index = ellscov.index(ell) * len(xcov) + np.arange(len(xcov))
            if xlim is not None:
                index = index[(xcov >= xlim[0]) & (xcov <= xlim[1])]
            indices.append(index)
        indices = np.concatenate(indices, axis=0)
        return cov[np.ix_(indices, indices)]

    covariance = np.loadtxt(covariance_fn)
    ellscov = (0, 2, 4)

    if ells is None:
        ells = ellscov

    if select is None:
        select = dict.fromkeys(ells)
    elif not isinstance(select, dict):
        select = {ell: select for ell in ells}

    return cut_matrix(covariance, binning, ellscov, select)


class PowerSpectrumCovarianceFile(BaseFile):

    """Power spectrum covariance."""
    name = 'power_covariance'

    def load(self, *args, **kwargs):
        """Load power spectrum covariance."""
        return load_covariance_matrix(self.path, (0., None, 0.005), *args, **kwargs)

    def save(self, covariance):
        """Save file."""
        utils.mkdir(os.path.dirname(self.path))
        np.savetxt(self.path, covariance)


class CorrelationFunctionCovarianceFile(BaseFile):

    """Correlation function covariance."""
    name = 'correlation_covariance'

    def load(self, *args, **kwargs):
        """Load correlation function covariance."""
        return load_covariance_matrix(self.path, (20., 200., 4.), *args, **kwargs)

    def save(self, covariance):
        """Save file."""
        utils.mkdir(os.path.dirname(self.path))
        np.savetxt(self.path, covariance)


class CovarianceFile(BaseFile):

    """Covariance."""
    name = 'covariance'

    def load(self, select=None):
        """Load covariance."""
        from desilike.utils import LoggingContext
        with LoggingContext(level='warning'):
            from desilike.observables import ObservableCovariance
            toret = ObservableCovariance.load(self.path)
            if select:
                toret = toret.select(xlim=select[:2])
        return toret

    def save(self, covariance):
        """Save file."""
        covariance.save(self.path)


class ObservableFile(BaseFile):

    """Observable."""
    name = 'observable'

    def load(self, *args, **kwargs):
        """Load observable."""
        from desilike.utils import LoggingContext
        with LoggingContext(level='warning'):
            from desilike.observables import ObservableArray
            return ObservableArray.load(self.path)

    def save(self, covariance):
        """Save file."""
        covariance.save(self.path)


class TemplateFile(BaseFile):

    """Systematic template."""
    name = 'template'

    def load(self, *args, **kwargs):
        """Load template."""
        from .systematic_template import PolynomialTemplate
        return PolynomialTemplate.load(self.path)

    def save(self, template):
        """Save file."""
        template.save(self.path)

        
def is_file_sequence(item):
    from desipipe.file_manager import FileEntryCollection
    return isinstance(item, (tuple, list, FileEntryCollection))


def is_path(item):
    return isinstance(item, (str, os.PathLike))


def load(file, **kwargs):
    from desipipe.file_manager import BaseFile
    if is_file_sequence(file):
        return [load(fi, **kwargs) for fi in list(file)]
    if isinstance(file, BaseFile): return file.load(**kwargs)
    elif isinstance(file, (str, os.PathLike)) and 'load' in kwargs: return kwargs.pop('load')(file, **kwargs)
    return file