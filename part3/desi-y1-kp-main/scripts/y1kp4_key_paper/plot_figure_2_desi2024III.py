import sys, os
import jax
jax.config.update('jax_platform_name', 'cpu')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import proplot as pplt
import scipy.stats as stats
from tqdm import tqdm
from desilike.profilers import MinuitProfiler
from desilike.samples.profiles import Profiles
from uncertainties import unumpy as unp, ufloat
import scienceplots
plt.style.use(['science'])

def read_table_minuit(filename):
    df = pd.read_csv(filename, skiprows=3, sep = '|', names = ['bestfit', 'error'], skipfooter = 1, usecols = (3,4))
    return df#[['bestfit', 'error']]
def gather_results(dir, fit):
    print(dir, flush=True)
    res = []
    ids = range(0, 1000)
    percival_fac = None
    invalid_number = 0
    for i in tqdm(ids):
        fname = f"{dir}/minuit_prof_{i:03d}.txt.npy"
        prof = Profiles.load(fname)
        prof = prof.choice()
        invalid_number += (not prof.bestfit.attrs['is_valid'])
        if not prof.bestfit.attrs['is_valid']:
            print(f"ID {i} has an invalid fit")
            print(prof.bestfit.attrs, flush = True)
        chisq = prof.bestfit.chi2min
        if percival_fac is None:
            try:
                percival_fac =  prof.bestfit.attrs['percival2014_factor']
                if percival_fac is None: percival_fac = 1.
            except KeyError:
                percival_fac = 1.
        
        chisq *= percival_fac
        
        if fit == 'bao':
            res_ = np.array([ufloat(prof.bestfit['qiso'], prof.error['qiso']),
                                 ufloat(prof.bestfit['b1'], prof.error['b1']),
                                 ufloat(prof.bestfit['sigmas'], prof.error['sigmas']),
                                 ufloat(chisq, 0.),])
        elif fit == 'bao2':
            res_ = np.array([ufloat(prof.bestfit['qiso'], prof.error['qiso']), 
                                 ufloat(prof.bestfit['qap'], prof.error['qap']),
                                 ufloat(chisq, 0.)])
        elif fit == 'shapefit':
            res_ = np.array([ufloat(prof.bestfit['qiso'], prof.error['qiso']), 
                                 ufloat(prof.bestfit['qap'], prof.error['qap']),
                                 ufloat(prof.bestfit['dm'] + 1, prof.error['dm']),
                                 ufloat(prof.bestfit['df'], prof.error['df']),
                                 ufloat(chisq, 0.),
                                 ])
        elif fit == 'direct':
            #titles = ['h', 'Omega_m', 'omega_b', 'logA', 'chisq']
            res_ = np.array([ufloat(prof.bestfit['h'], prof.error['h']), 
                                 ufloat(prof.bestfit['omega_cdm'], prof.error['omega_cdm']),
                                 ufloat(prof.bestfit['omega_b'], prof.error['omega_b']),
                                 ufloat(prof.bestfit['logA'], prof.error['logA']),
                                 ufloat(chisq, 0.),
                                 ])
        res.append(res_)
    print(f"Found {100 * invalid_number / 1000}% of invalid fits", flush = True)
    return np.array(res), prof.bestfit.attrs['ndof']
def single_hist(results_a, results_b, ax, pax_x, pax_y, label_a, label_b, title, do_contour = False, plot_x = True, plot_y = True, write_stats = True, **kwargs):
    bins = np.linspace(min(results_a.min(), results_b.min()), max(results_a.max(), results_b.max()), 25+1)
    if plot_y:
        pax_y.histh(results_b, bins, histtype='step', density=True, **kwargs)
    if plot_x:
        _, bins, _ = pax_x.hist(results_a, histtype='step', bins=bins, density=True, **kwargs)
    #pax_y.format(xreverse=not ('chi' in title) if plot_x and plot_y else True, xlocator=[])
    pax_y.format(xreverse=True, xlocator=[])
    pax_x.format(xreverse=False, ylocator=[])
    if not do_contour:
        ax.hist2d(
                results_a, results_b, bins, vmin=None, vmax=None, levels=50,
                #cmap='reds', 
                )
    else:
        #ax.scatter(results_a, results_b, markersize = 0.01, c = kwargs['color'])
        levels, xedges, yedges = np.histogram2d(results_a, results_b, bins = bins)
        cmap = pplt.Colormap(kwargs['color'], l=100, name='Pacific', space='hsl', alpha = (0,1))
        ax.contourf(xedges, yedges, levels, cmap = cmap, lw = 0, N = 15, robust = True, transpose = True)
        #ax.contour(xedges, yedges, levels, color = kwargs['color'], lw = 0.5, N = 5, robust = True)
    ax.format(xlabel = label_a, ylabel = label_b)
    if title is not None:
        ax.set_title(f"${title}$",)
    mean_a = results_a.mean()
    mean_b = results_b.mean()
    ax.axline((mean_a, mean_a), slope=1, ls = ':', c = 'k', lw=1)
    
    pct_diff = 100 * (results_a / results_b - 1.).mean()
    pct_diff_std = 100 * (results_a / results_b - 1.).std()
    reg_res = stats.linregress(results_a, results_b)
    
    if write_stats:
        ax.text(0.05, 0.85, rf'$\langle\frac{{x}}{{y}}-1\rangle$={pct_diff:.3f}\%', transform='axes')#, fontsize=15)
        ax.text(0.05, 0.7, f'$r$={reg_res.rvalue:.3f}', transform='axes')#, fontsize=15)
        ax.text(0.55, 0.2, f'$\zeta$={reg_res.slope:.3f}', transform='axes')#, fontsize=15)
        ax.text(0.55, 0.05, f'$\\vartheta$={reg_res.intercept:.3f}', transform='axes')#, fontsize=15)
    
    #return np.array((pct_diff, reg_res.rvalue, reg_res.slope, reg_res.intercept))
    return np.array((pct_diff, pct_diff_std, reg_res.rvalue, reg_res.slope, reg_res.intercept, stats.pearsonr(results_a, results_b).statistic))
   
def filter_outliers(arr, m):
    diff = np.abs(arr - np.median(arr))
    mdev = np.median(diff)
    s = diff / mdev if mdev else np.zeros_like(arr)
    return s < m
    
    
 
def plot_results(results_a, results_b, label_a, label_b, fig = None, ax = None, titles = None, ndof = None, do_contour = False, pax = (None, None), plot_x = True, plot_y = True, write_stats = True, add_theory = True, add_theory_chisq = False, **kwargs):

    if fig is None or ax is None:
        fig, ax = pplt.subplots(nrows = 3, ncols = 4, share = 0)
    pax_x, pax_y = pax
    if pax_x is None and pax_y is None:    
        pax_x = ax.panel('top', space=0, width = '3em')
        pax_y = ax.panel('left', space=0, width='3em')
        
    results_a = np.atleast_2d(results_a)
    results_b = np.atleast_2d(results_b)
    
    full_stats = []
    for i in range(results_a.shape[1] - 1):
        this_results_a_nominal = unp.nominal_values(results_a[:,i])
        this_results_b_nominal = unp.nominal_values(results_b[:,i])
        
        this_results_a_sigma = unp.std_devs(results_a[:,i])
        this_results_b_sigma = unp.std_devs(results_b[:,i])
        
        results_a_mask = filter_outliers(this_results_a_sigma, 4.5)
        results_b_mask = filter_outliers(this_results_b_sigma, 4.5)
        
        results_mask = results_a_mask & results_b_mask
        stats = single_hist(this_results_a_nominal[results_mask], this_results_b_nominal[results_mask], ax[i,0], pax_x[i,0], pax_y[i,0], label_a, label_b, titles[i], do_contour = do_contour, plot_x = plot_x, plot_y = plot_y, write_stats = write_stats, **kwargs)
        full_stats.append(stats)
        if 'chi' not in titles[i]:
            stats = single_hist(this_results_a_sigma[results_mask], this_results_b_sigma[results_mask], ax[i,1], pax_x[i,1], pax_y[i,1], label_a, label_b, f"\sigma_{{{titles[i]}}}", do_contour = do_contour, plot_x = plot_x, plot_y = plot_y, write_stats = write_stats, **kwargs)
            full_stats.append(stats)
        else:
            ax[1,i].axis('off')
            pax_x[1,i].axis('off')
            pax_y[1,i].axis('off')
            if ndof is not None:
                if plot_x and (add_theory or add_theory_chisq):
                    add_chisq(ndof, this_results_a_nominal[results_mask], fig, pax_x[0,i], color = "darkred", ls = '--')
                if plot_y and (add_theory or add_theory_chisq):
                    add_chisq(ndof, this_results_b_nominal[results_mask], fig, pax_y[0,i], vert = True, color = "darkred", ls = '--')
        if plot_x and add_theory:
            add_gaussian(this_results_a_nominal[results_mask], this_results_a_sigma[results_mask], fig, pax_x[0,i], color = "darkred", ls = '--')
        if plot_y and plot_x and add_theory:
            add_gaussian(this_results_b_nominal[results_mask], this_results_b_sigma[results_mask], fig, pax_y[0,i], vert=True, color = "darkred", ls = '--')
            
            
    
    if not write_stats:
        return fig, ax, pax_x, pax_y, full_stats    
    return fig, ax, pax_x, pax_y



def add_gaussian(par, sigma_par, fig, ax, vert = False, **kwargs):
    import scipy.stats as stats
    mean_par = par.mean()
    mean_sig = sigma_par.mean()
    
    x = np.linspace(mean_par - 3 * mean_sig, mean_par + 3 * mean_sig, 100)
    if not vert:
        ax.plot(x, stats.norm(loc = mean_par, scale = mean_sig).pdf(x), **kwargs)
    else:
        ax.plot(stats.norm(loc = mean_par, scale = mean_sig).pdf(x), x, **kwargs)
    
def add_chisq(ndof, par, fig, ax, vert = False, **kwargs):
    import scipy.stats as stats
    mean_par = par.mean()
    x = np.linspace(par.min(), par.max(), 100)
    if not vert:
        ax.plot(x, stats.chi2.pdf(x, df = ndof), **kwargs)
    else:
        ax.plot(stats.chi2.pdf(x, df = ndof), x, **kwargs)
    
def stats_to_table(stats, max_rows, row_names, col_names, stats_names):
    from tabulate import tabulate
    index = []
    new_stats_names = []
    for sn in stats_names:
        new_stats_names.append(sn)
        if not 'chi' in sn:
            new_stats_names.append(f"\sigma_{{{sn}}}")
    print(stats_names)
    
    for sn in new_stats_names:
        for rn in row_names[:max_rows]:
            index.append(f"${rn}_{{{sn}}}$")
    print(index)        
    #if stats_mean.ndim > 2:
    print(stats[:,:max_rows,...].shape)
    print(tabulate(stats[:,:max_rows,...].reshape((len(index),-1)), showindex = index, floatfmt=".3f",colalign=["center"] * len(col_names), headers = col_names).replace("+/-", ' \pm '))
    print(tabulate(stats[:,:max_rows,...].reshape((len(index),-1)), showindex = index, tablefmt='latex_raw', floatfmt=".3f", colalign=["center"] * len(col_names), headers = col_names).replace("+/-", ' \pm '))
    
        
    

    
    
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    FIT = 'bao2'
    STAT = 'xi'
    CONV = 'sym'
    TRACER = ['QSO', 'ELG_LOP', 'LRG']
    CAP = 'GCcomb'
    PLOT_DATA_FN = "figure_2_desi2024II.pkl.npy"
    REDSHIFTS = dict(LRG = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 
                 ELG_LOP = [(0.8, 1.1), (1.1, 1.6)],
                 ELG = [(0.8, 1.1), (1.1, 1.6)],
                 ELG_LOPnotqso = [(0.8, 1.1), (1.1, 1.6)],
                 QSO = [(0.8, 2.1)])
    
    titles = ['\\alpha_{iso}', '\\alpha_{ap}']
    
    fig, ax = pplt.subplots(nrows = 2, ncols=len(titles), share = 1, spanx = False, refwidth = "2cm")
    pax_x, pax_y = None, None
    full_full_stats = []
    cid = 0
    tracer_labels = {'QSO':{0.8:"QSO"},
                     "ELG_LOP":{0.8:"ELG1", 1.1:"ELG2"},
                     "LRG":{0.4:"LRG1", 0.6:"LRG2", 0.8:"LRG3"}}

    plot_data = np.load(PLOT_DATA_FN, allow_pickle = True).item()
    for tracer in TRACER:
        for zmin, zmax in REDSHIFTS[tracer]:
            
            mock_res = plot_data[tracer][f'{zmin:.3f}-{zmax:.3f}']['mock']
            analytic_res = plot_data[tracer][f'{zmin:.3f}-{zmax:.3f}']['analytic']
            ndof = plot_data[tracer][f'{zmin:.3f}-{zmax:.3f}']['ndof']
            fig, ax, pax_x, pax_y, full_stats = plot_results(mock_res, analytic_res, 'mock', 'analytic', fig = fig, ax = ax, pax= (pax_x,pax_y), titles = titles, ndof = ndof, do_contour = True, write_stats = False, add_theory = False, color = f'C{cid}', add_theory_chisq=cid == 0)
            full_full_stats.append(np.array(full_stats))
            for a, p in enumerate(ax):
                p.plot([], [], color = f'C{cid}', label = f"{tracer_labels[tracer][zmin]}: $\Delta = {np.array(full_stats)[a, 0]:.1f}\pm{np.array(full_stats)[a, 1]:.1f}$")
            fig.savefig(f"figure_2_desi2024III.pdf", dpi=300)
            cid += 1
    
    full_full_stats = np.array(full_full_stats)

    for p in ax:
        p.legend(loc='right', ncols = 1, frame = True)
    fig.savefig(f"figure_2_desi2024III.pdf", dpi=300)
    
        
