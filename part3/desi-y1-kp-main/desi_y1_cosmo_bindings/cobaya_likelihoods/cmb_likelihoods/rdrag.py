import numpy as np
from cobaya.likelihood import Likelihood


class rdrag(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "rdrag"

    # variables from yaml
    rdrag_mean: float
    rdrag_std: float

    def initialize(self):
        self.minus_half_invvar = - 0.5 / self.rdrag_std ** 2

    def get_requirements(self):
        return {"rdrag": None}

    def logp(self, **params_values):
        rdrag_theory = self.provider.get_param("rdrag")
        return self.minus_half_invvar * (rdrag_theory - self.rdrag_mean) ** 2