"""Class to fit power spectrum systematic templates, by Ruiyang Zhao."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from numpy import newaxis
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from typing import Literal, Sequence

from desilike.base import BaseClass
from desilike.jax import numpy as jnp


def polyfit(x, y, order: int | list[int], sigma=None, p0=None):
    if isinstance(order, int):
        order = [order]
    order = sorted(order)
    if p0 is None:
        p0 = [1] * len(order)

    def func(x, *args):
        poly = np.array([x**n for n in order])
        return np.einsum("ix,i->x", poly, args)

    popt, pcov = curve_fit(func, x, y, p0=p0, sigma=sigma)
    yfit = func(x, *popt)
    res = yfit - y
    ndata = x.shape[0]
    nvaried = len(order)
    if sigma is None:
        sigma = np.ones(x.shape[0])
    cov = sigma if sigma.ndim == 2 else np.diag(sigma)
    invcov = np.linalg.inv(cov)
    chi2 = res @ invcov @ res
    return PolyfitResult(order=order, popt=popt, pcov=pcov, yfit=yfit, chi2=chi2, ndata=ndata, nvaried=nvaried)


def order_string(order, observable, math=False):
    if isinstance(order, int):
        order = [order]
    order = sorted(order)
    xs = "k" if observable == "power" else "s"
    terms = []
    for n in order:
        if n == 0:
            val = "c_0"
        elif n == 1:
            val = f"c_1{xs}"
        else:
            val = Rf"c_{{{n}}}{xs}^{{{n}}}"

        terms.append(val)
    retval = "+".join(terms)
    if math:
        return "$" + retval + "$"
    return retval


@dataclass
class PolyfitResult:
    order: list[int]
    popt: NDArray[np.floating]
    pcov: NDArray[np.floating]
    yfit: NDArray[np.floating]
    chi2: float
    ndata: int
    nvaried: int

    def chi2_per_dof(self):
        return self.chi2 / (self.ndata - self.nvaried)#

    def order_string(self, observable: Literal["corr", "power"], math=False):
        return order_string(self.order, observable, math=math)


class PolynomialTemplate(BaseClass):
    def __init__(
        self,
        ells: Sequence[int],
        orders: Sequence[Sequence[int]],
        popts: Sequence[Sequence[float]],
        x=None,
        ys=None,
        yerrs=None,
        attrs=None
    ):
        if len(ells) != len(orders) and len(ells) != len(popts):
            raise ValueError("Inconsistent lenth of ells")
        if any(len(order) != len(popt) for order, popt in zip(orders, popts)):
            raise ValueError("Inconsistent length of orders and popts")
        self.ells = np.asarray(ells)
        self.orders = [np.asarray(order) for order in orders]
        self.popts = [np.asarray(popt) for popt in popts]
        # raw data point
        self.x = x
        self.ys = ys
        self.yerrs = yerrs
        self.attrs = dict(attrs or {})

    def __call__(self, ell, x):
        if ell not in self.ells:
            return jnp.zeros_like(x)
        idx = jnp.where(self.ells == ell)[0].item()
        order = self.orders[idx]
        popt = self.popts[idx]
        return popt @ (x[newaxis, :] ** order[:, newaxis])

    def __getstate__(self):
        state = {name: getattr(self, name) for name in ['ells', 'orders', 'popts', 'x', 'ys', 'yerrs', 'attrs']}
        # Convert to numpy arrays; jax arrays are less stable!
        state['ells'] = np.asarray(state['ells'])
        for name in ['orders', 'popts']:
            state[name] = [np.asarray(v) for v in state[name]]
        return state

    @classmethod
    def fit_multipole(cls, x, ys, orders: list[list[int]], yerrs=None, p0=None):
        if len(ys) != len(orders) or (yerrs is not None and len(ys) != len(yerrs)):
            raise TypeError("Inconsistent length of inputs")
        ells = [2 * i for i in range(len(ys))]
        it = zip(orders, ys, yerrs if yerrs is not None else [None] * len(ys))
        popts = [polyfit(x, y, order, sigma=yerr, p0=p0).popt for order, y, yerr in it]
        return cls(ells=ells, orders=orders, popts=popts, x=x, ys=ys, yerrs=yerrs)  # type: ignore
