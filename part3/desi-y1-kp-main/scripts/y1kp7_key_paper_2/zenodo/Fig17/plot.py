import numpy as np

from matplotlib import pyplot as plt


def plot(tracer):
    
    ells = (0, 2)
    color = {'BGS': 'yellowgreen', 'LRG1': 'orange', 'LRG2': 'orangered',
              'LRG3': 'firebrick', 'ELG': 'blue', 'QSO': 'seagreen'}[tracer]
    data = np.loadtxt(f'bestfit_fs_{tracer}.txt', unpack=True)
    k, data, std, theory = data[0], data[1:1 + len(ells)], data[1 + len(ells):1 + 2 * len(ells)], data[1 + 2 * len(ells):] 
    
    height_ratios = [3] + [1] * len(ells)
    figsize = (6, 1.5 * sum(height_ratios))
    fig, lax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
    fig.subplots_adjust(hspace=0.1)

    k_exp = 1
    for ill, ell in enumerate(ells):
        lax[0].errorbar(k, k**k_exp * data[ill], yerr=k**k_exp * std[ill], color=color, linestyle='none', marker='o', label=r'$\ell = {:d}$'.format(ell))
        lax[0].plot(k, k**k_exp * theory[ill], color=color)
    for ill, ell in enumerate(ells):
        lax[ill + 1].plot(k, (data[ill] - theory[ill]) / std[ill], color=color)
        lax[ill + 1].set_ylim(-4, 4)
        for offset in [-2., 2.]: lax[ill + 1].axhline(offset, color='k', linestyle='--')
        lax[ill + 1].set_ylabel(r'$\Delta P_{{{0:d}}} / \sigma_{{ P_{{{0:d}}} }}$'.format(ell))
    for ax in lax[1:]: ax.grid(True)
    lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    ax = lax[0]
    ax.text(0.9, 0.95, tracer,
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=16, color=color)

    plt.savefig(f'pk_fs_{tracer}.pdf', bbox_inches="tight")


for tracer in ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']:
    fig = plot(tracer)