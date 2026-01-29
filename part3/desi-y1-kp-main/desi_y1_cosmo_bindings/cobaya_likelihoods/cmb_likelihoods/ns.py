import numpy as np
from cobaya.likelihood import Likelihood


class ns(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "ns"

    # variables from yaml
    ns_mean: float
    ns_std: float

    def initialize(self):
        self.minus_half_invvar = - 0.5 / self.ns_std ** 2

    def get_requirements(self):
        return {"ns": None}

    def logp(self, **params_values):
        ns_theory = self.provider.get_param("ns")
        return self.minus_half_invvar * (ns_theory - self.ns_mean) ** 2