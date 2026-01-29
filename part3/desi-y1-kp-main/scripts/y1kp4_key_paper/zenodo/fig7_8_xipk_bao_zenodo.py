#!/usr/bin/env python
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

#########################################
# Tracer colors and legend labels
#########################################
tracer_colors = {
    "LRG 0.4-0.6":    "orange",
    "LRG 0.6-0.8":    "orangered",
    "LRG 0.8-1.1":    "firebrick",
    "BGS 0.1-0.4":    "yellowgreen",
    "QSO 0.8-2.1":    "seagreen",
    "ELG 0.8-1.1":    "lightskyblue",
    "ELG 1.1-1.6":    "steelblue",
    "LRG+ELG 0.8-1.1": "slateblue"
}

labels = {
    "LRG 0.4-0.6":    "LRG1",
    "LRG 0.6-0.8":    "LRG2",
    "LRG 0.8-1.1":    "LRG3",
    "BGS 0.1-0.4":    "BGS",
    "QSO 0.8-2.1":    "QSO",
    "ELG 0.8-1.1":    "ELG1",
    "ELG 1.1-1.6":    "ELG2",
    "LRG+ELG 0.8-1.1": "LRG3+ELG1"
}

#########################################
# Plotting Functions for xi and pk
#########################################

def plot_xi(tracer, save_dir):
    profiles_pre = np.load('xi_bao_prerecon.npy', allow_pickle=True).item()
    profiles_post = np.load('xi_bao_postrecon.npy', allow_pickle=True).item()
    
    #Post-recon
    ells = profiles_post[tracer]["ells"]
    data = profiles_post[tracer]["xi"]
    fit = profiles_post[tracer]["xi_fit"]
    std = profiles_post[tracer]["std"]
    s = profiles_post[tracer]["s"]
    
    #Pre-recon
    data_pre = profiles_pre[tracer]["xi"]
    fit_pre = profiles_pre[tracer]["xi_fit"]
    std_pre = profiles_pre[tracer]["std"]
    s_pre = profiles_pre[tracer]["s"]
    
    height_ratios = [1] * len(ells)
    fig, axes = plt.subplots(len(ells), sharex=True,
                               gridspec_kw={"height_ratios": height_ratios},
                               figsize=(4, 3 * len(ells)))
    if len(ells) == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0)
    for i, ell in enumerate(ells):
        axes[i].errorbar(s_pre[i], data_pre[i],
                         yerr=std_pre[i], marker="o", linestyle="none",
                         label="Pre", markerfacecolor="none", color=tracer_colors[tracer], alpha=0.7)
        axes[i].plot(s_pre[i], fit_pre[i], linestyle="--", color=tracer_colors[tracer])
        axes[i].errorbar(s[i], data[i],
                         yerr=std[i], marker="o", linestyle="none", label="Post", color=tracer_colors[tracer])
        axes[i].plot(s[i], fit[i], color=tracer_colors[tracer])
        axes[i].set_ylabel(r"$\Delta \xi_{%d}(s)$" % ell, fontsize=15)
        axes[i].tick_params(axis='both', which='major', labelsize=13)
        axes[i].grid(True)
    axes[0].legend(title=fr"$\texttt{{{labels[tracer]}}}$", loc="best",
                   fontsize=12, title_fontsize=14, ncol=3, columnspacing=1.0)
    axes[-1].set_xlabel(r"$s$ [$h^{-1}\mathrm{Mpc}$]", fontsize=15)
    filename = tracer.replace(" ", "_")
    fig.savefig(f"{save_dir}/xi_pre-post_{filename}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_pk(tracer, save_dir):
    profiles_pre = np.load('pk_bao_prerecon.npy', allow_pickle=True).item()
    profiles_post = np.load('pk_bao_postrecon.npy', allow_pickle=True).item()
    
    #Post-recon
    ells = profiles_post[tracer]["ells"]
    data = profiles_post[tracer]["pk"]
    fit = profiles_post[tracer]["pk_fit"]
    std = profiles_post[tracer]["std"]
    k = profiles_post[tracer]["k"]
    
    #Pre-recon
    data_pre = profiles_pre[tracer]["pk"]
    fit_pre = profiles_pre[tracer]["pk_fit"]
    std_pre = profiles_pre[tracer]["std"]
    k_pre = profiles_pre[tracer]["k"]
    
    height_ratios = [1] * len(ells)
    fig, axes = plt.subplots(len(ells), sharex=True,
                               gridspec_kw={"height_ratios": height_ratios},
                               figsize=(6, 2 * len(ells)))
    if len(ells) == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0)
    for i, ell in enumerate(ells):
        axes[i].errorbar(k_pre[i], k_pre[i]*data_pre[i],
                         yerr=k_pre[i]*std_pre[i], marker="o", linestyle="none",
                         label="Pre", markerfacecolor="none", color=tracer_colors[tracer],alpha=0.7)
        axes[i].plot(k_pre[i], k_pre[i]*fit_pre[i], linestyle="--", color=tracer_colors[tracer],alpha=0.7)
        axes[i].errorbar(k[i], k[i]*data[i],
                         yerr=k[i]*std[i], marker="o", linestyle="none", label="Post", color=tracer_colors[tracer])
        axes[i].plot(k[i], k[i]*fit[i], color=tracer_colors[tracer])
        axes[i].set_ylabel(r"$k\Delta P_{%d}(k)$" % ell, fontsize=15)
        axes[i].tick_params(axis='both', which='major', labelsize=13)
        axes[i].grid(True)
    axes[0].legend(title=fr"$\texttt{{{labels[tracer]}}}$", loc="best",
                   fontsize=12, title_fontsize=14, ncol=3, columnspacing=1.0)
    axes[-1].set_xlabel(r"$k$ [$h{\rm Mpc}^{-1}$]", fontsize=15)
    filename = tracer.replace(" ", "_")
    fig.savefig(f"{save_dir}/pk_bao_pre_post_{filename}.pdf", bbox_inches="tight")
    plt.close(fig)

#########################################
# Main: Argument Parsing and Execution
#########################################
def main():
    parser = argparse.ArgumentParser(
        description="Generate xi and pk pre/post plots using a simplified generic file naming scheme."
    )
    parser.add_argument("plot_type", choices=["xi", "pk"],
                        help="Plot type: 'xi' for correlation, 'pk' for power spectrum")
    parser.add_argument("tracer", type=str,
                        help="Tracer name (e.g., 'LRG 0.4-0.6', or 'all' for all tracers)")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Directory to save the generated plots")
    args = parser.parse_args()
    
    # If the user specifies "all", run for every tracer in tracer_info.
    if args.tracer.lower() == "all":
        for tracer in labels.keys():
            print(f"\nGenerating {args.plot_type} plots for {tracer}...")
            if args.plot_type == "xi":
                try:
                    plot_xi(tracer, args.save_dir)
                except Exception as e:
                    print(f"Error generating xi plots for {tracer}: {e}")
            elif args.plot_type == "pk":
                try:
                    plot_pk(tracer, args.save_dir)
                except Exception as e:
                    print(f"Error generating pk plots for {tracer}: {e}")
    else:
        if args.plot_type == "xi":
            try:
                plot_xi(args.tracer, args.save_dir)
            except Exception as e:
                print("Error generating xi plots:", e)
                sys.exit(1)
        elif args.plot_type == "pk":
            try:
                plot_pk(args.tracer, args.save_dir)
            except Exception as e:
                print("Error generating pk plots:", e)
                sys.exit(1)

if __name__ == "__main__":
    main()
