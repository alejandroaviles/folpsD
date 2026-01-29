import numpy as np
from cobaya.likelihood import Likelihood


class thetastar(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "thetastar"

    def initialize(self):
        self.mean = np.ravel(self.mean)
        self.invcov = np.linalg.inv(np.atleast_2d(self.cov))

    def get_requirements(self):
        convert = {'thetastar100': 'thetastar'}
        return {convert.get(name, name): None for name in self.quantities}

    def logp(self, **params_values):
        theory = []
        for name in self.quantities:
            if name == 'thetastar100':
                value = self.provider.get_param('thetastar') * 100.
            else:
                value = self.provider.get_param(name)
            theory.append(value)
        diff = self.mean - np.array(theory)
        return - 0.5 * diff.dot(self.invcov).dot(diff)