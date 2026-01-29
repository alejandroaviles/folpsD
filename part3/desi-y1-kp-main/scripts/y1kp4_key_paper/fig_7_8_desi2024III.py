#!/usr/bin/env python
import argparse
import sys
import matplotlib.pyplot as plt
from desilike.samples import Profiles

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
# Consolidated tracer info
#########################################
# Each key is the tracer label (as you pass it) and the value is a dict with:
#  - "zrange": redshift range string,
#  - "dim": "1d" or "2d" (determines mode: qiso for 1d, qisoqap for 2d),
tracer_info = {
    "LRG 0.4-0.6":    {"zrange": "0.4-0.6", "dim": "2d"},
    "LRG 0.6-0.8":    {"zrange": "0.6-0.8", "dim": "2d"},
    "LRG 0.8-1.1":    {"zrange": "0.8-1.1", "dim": "2d"},
    "BGS 0.1-0.4":    {"zrange": "0.1-0.4", "dim": "1d"},
    "QSO 0.8-2.1":    {"zrange": "0.8-2.1", "dim": "1d"},
    "ELG 0.8-1.1":    {"zrange": "0.8-1.1", "dim": "1d"},
    "ELG 1.1-1.6":    {"zrange": "1.1-1.6", "dim": "2d"},
    "LRG+ELG 0.8-1.1": {"zrange": "0.8-1.1", "dim": "2d"}
}

#########################################
# File name dictionaries for xi and pk
#########################################
xi_post_files = {
    "BGS 0.1-0.4":    "profiles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_default_FKP_sigmas-2.0-2.0_sigmapar-8.0-2.0_sigmaper-3.0-1.0_lim_0-50-150.npy",
    "ELG 0.8-1.1":    "profiles_ELG_LOPnotqso_GCcomb_z0.8-1.1_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150.npy",
    "ELG 1.1-1.6":    "profiles_ELG_LOPnotqso_GCcomb_z1.1-1.6_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150_2-50-150.npy",
    "LRG 0.4-0.6":    "profiles_LRG_GCcomb_z0.4-0.6_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150_2-50-150.npy",
    "LRG 0.6-0.8":    "profiles_LRG_GCcomb_z0.6-0.8_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150_2-50-150.npy",
    "LRG 0.8-1.1":    "profiles_LRG_GCcomb_z0.8-1.1_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150_2-50-150.npy",
    "QSO 0.8-2.1":    "profiles_QSO_GCcomb_z0.8-2.1_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150.npy",
    "LRG+ELG 0.8-1.1": "profiles_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_default_FKP_sigmas-2.0-2.0_sigmapar-6.0-2.0_sigmaper-3.0-1.0_lim_0-50-150_2-50-150.npy"
}

xi_pre_files = {
    "BGS 0.1-0.4":    "profiles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_default_FKP_sigmas-2.0-2.0_sigmapar-10.0-2.0_sigmaper-6.5-1.0_lim_0-50-150.npy",
    "ELG 0.8-1.1":    "profiles_ELG_LOPnotqso_GCcomb_z0.8-1.1_default_FKP_sigmas-2.0-2.0_sigmapar-8.5-2.0_sigmaper-4.5-1.0_lim_0-50-150.npy",
    "ELG 1.1-1.6":    "profiles_ELG_LOPnotqso_GCcomb_z1.1-1.6_default_FKP_sigmas-2.0-2.0_sigmapar-8.5-2.0_sigmaper-4.5-1.0_lim_0-50-150_2-50-150.npy",
    "LRG 0.4-0.6":    "profiles_LRG_GCcomb_z0.4-0.6_default_FKP_sigmas-2.0-2.0_sigmapar-9.0-2.0_sigmaper-4.5-1.0_lim_0-50-150_2-50-150.npy",
    "LRG 0.6-0.8":    "profiles_LRG_GCcomb_z0.6-0.8_default_FKP_sigmas-2.0-2.0_sigmapar-9.0-2.0_sigmaper-4.5-1.0_lim_0-50-150_2-50-150.npy",
    "LRG 0.8-1.1":    "profiles_LRG_GCcomb_z0.8-1.1_default_FKP_sigmas-2.0-2.0_sigmapar-9.0-2.0_sigmaper-4.5-1.0_lim_0-50-150_2-50-150.npy",
    "QSO 0.8-2.1":    "profiles_QSO_GCcomb_z0.8-2.1_default_FKP_sigmas-2.0-2.0_sigmapar-9.0-2.0_sigmaper-3.5-1.0_lim_0-50-150.npy",
    "LRG+ELG 0.8-1.1": "profiles_LRG+ELG_LOPnotqso_GCcomb_0.8_1.1_sigmas2.0_sigmapar9.0_sigmaper4.5.npy"
}

pk_post_files = {
    "ELG 0.8-1.1":    "profiles_ELG_LOPnotqso_GCcomb_0.8_1.1_qiso_pcs_sigmas2.0_sigmapar6.0_sigmaper3.0.npy",
    "ELG 1.1-1.6":    "profiles_ELG_LOPnotqso_GCcomb_1.1_1.6_qisoqap_pcs_sigmas2.0_sigmapar6.0_sigmaper3.0.npy",
    "LRG 0.4-0.6":    "profiles_LRG_GCcomb_0.4_0.6_qisoqap_pcs_sigmas2.0_sigmapar6.0_sigmaper3.0.npy",
    "LRG 0.6-0.8":    "profiles_LRG_GCcomb_0.6_0.8_qisoqap_pcs_sigmas2.0_sigmapar6.0_sigmaper3.0.npy",
    "LRG 0.8-1.1":    "profiles_LRG_GCcomb_0.8_1.1_qisoqap_pcs_sigmas2.0_sigmapar6.0_sigmaper3.0.npy",
    "BGS 0.1-0.4":    "profiles_BGS_BRIGHT-21.5_GCcomb_0.1_0.4_qiso_pcs_sigmas2.0_sigmapar8.0_sigmaper3.0.npy",
    "QSO 0.8-2.1":    "profiles_QSO_GCcomb_0.8_2.1_sigmas2.0_sigmapar6.0_sigmaper3.0.npy"
}

pk_pre_files = {
    "ELG 0.8-1.1":    "profiles_ELG_LOPnotqso_GCcomb_0.8_1.1_sigmas2.0_sigmapar8.5_sigmaper4.5.npy",
    "ELG 1.1-1.6":    "profiles_ELG_LOPnotqso_GCcomb_1.1_1.6_sigmas2.0_sigmapar8.5_sigmaper4.5.npy",
    "LRG 0.4-0.6":    "profiles_LRG_GCcomb_0.4_0.6_sigmas2.0_sigmapar9.0_sigmaper4.5.npy",
    "LRG 0.6-0.8":    "profiles_LRG_GCcomb_0.6_0.8_sigmas2.0_sigmapar9.0_sigmaper4.5.npy",
    "LRG 0.8-1.1":    "profiles_LRG_GCcomb_0.8_1.1_sigmas2.0_sigmapar9.0_sigmaper4.5.npy",
    "BGS 0.1-0.4":    "profiles_BGS_BRIGHT-21.5_GCcomb_0.1_0.4_sigmas2.0_sigmapar10.0_sigmaper6.5.npy",
    "QSO 0.8-2.1":    "profiles_QSO_GCcomb_0.8_2.1_sigmas2.0_sigmapar9.0_sigmaper3.5.npy"
}

#########################################
# Full Path Functions for xi and pk
#########################################
def rec_folder(tracer, mode):
    # mode is "qiso" or "qisoqap" but we ignore it here:
    if tracer.startswith("QSO"):
        return "recon_IFFT_recsym_sm30"
    else:
        return "recon_IFFT_recsym_sm15"

def get_full_xi_paths(tracer):
    info = tracer_info.get(tracer)
    if info is None:
        raise ValueError(f"Unknown tracer: {tracer}")
    dim = info["dim"]
    mode = "qiso" if dim=="1d" else "qisoqap"
    rec = rec_folder(tracer, mode)
    # Base directories for xi
    base_post = (f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/desipipe/fits_2pt/"
                 f"fits_correlation_dampedbao_bao-{mode}_pcs2/{rec}/")
    
    if tracer.startswith("LRG+ELG"): 
        base_pre=(f'/global/cfs/cdirs/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/fits_correlation_{mode}_pcs2/')
    
    else:
        base_pre = (f"/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/desipipe/fits_2pt/"
                f"fits_correlation_dampedbao_bao-{mode}_pcs2/")
    # Look up the file names from the dictionaries
    post_file = xi_post_files.get(tracer)
    pre_file = xi_pre_files.get(tracer)
    if post_file is None or pre_file is None:
        raise ValueError(f"xi file names for tracer '{tracer}' not found in the dictionaries.")
    return base_post + post_file, base_pre + pre_file

def get_full_pk_paths(tracer):
    info = tracer_info.get(tracer)
    if info is None:
        raise ValueError(f"Unknown tracer: {tracer}")
    dim = info["dim"]
    mode = "qiso" if dim=="1d" else "qisoqap"
    rec = rec_folder(tracer, mode)
    base_post = (f"/global/cfs/cdirs/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/"
                 f"fits_power_{mode}_pcs/{rec}/")
    base_pre = (f"/global/cfs/cdirs/desi/users/epaillas/Y1/iron/v1.2/unblinded/fits_bao/"
                f"fits_power_{mode}_pcs/")
    post_file = pk_post_files.get(tracer)
    pre_file = pk_pre_files.get(tracer)
    if post_file is None or pre_file is None:
        raise ValueError(f"pk file names for tracer '{tracer}' not found in the dictionaries.")
    return base_post + post_file, base_pre + pre_file

#########################################
# Plotting Functions for xi and pk
#########################################

def plot_xi(tracer, save_dir):
    post_path, pre_path = get_full_xi_paths(tracer)
    print("xi post path:", post_path)
    print("xi pre path:", pre_path)
    profiles = Profiles.load(post_path)
    profiles_pre = Profiles.load(pre_path)
    ells = profiles.attrs["observable"]["ells"]
    data = profiles.attrs["observable"]["data"]
    theory = profiles.attrs["observable"]["theory"]
    std = profiles.attrs["observable"]["std"]
    nobao = profiles.attrs["observable"]["theory_nobao"]
    s = profiles.attrs["observable"]["s"]
    data_pre = profiles_pre.attrs["observable"]["data"]
    theory_pre = profiles_pre.attrs["observable"]["theory"]
    std_pre = profiles_pre.attrs["observable"]["std"]
    nobao_pre = profiles_pre.attrs["observable"]["theory_nobao"]
    s_pre = profiles_pre.attrs["observable"]["s"]
    height_ratios = [1] * len(ells)
    fig, axes = plt.subplots(len(ells), sharex=True,
                               gridspec_kw={"height_ratios": height_ratios},
                               figsize=(4, 3 * len(ells)))
    if len(ells) == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0)
    for i, ell in enumerate(ells):
        axes[i].errorbar(s_pre[i], (data_pre[i] - nobao_pre[i]),
                         yerr=std_pre[i], marker="o", linestyle="none",
                         label="Pre", markerfacecolor="none", color=tracer_colors[tracer], alpha=0.7)
        axes[i].plot(s_pre[i], (theory_pre[i] - nobao_pre[i]), linestyle="--", color=tracer_colors[tracer])
        axes[i].errorbar(s[i], (data[i] - nobao[i]),
                         yerr=std[i], marker="o", linestyle="none", label="Post", color=tracer_colors[tracer])
        axes[i].plot(s[i], (theory[i] - nobao[i]), color=tracer_colors[tracer])
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
    post_path, pre_path = get_full_pk_paths(tracer)
    print("pk post path:", post_path)
    print("pk pre path:", pre_path)
    profiles = Profiles.load(post_path)
    profiles_pre = Profiles.load(pre_path)
    ells = profiles.attrs["observable"]["ells"]
    k = profiles.attrs["observable"]["k"]
    data = profiles.attrs["observable"]["data"]
    theory = profiles.attrs["observable"]["theory"]
    std = profiles.attrs["observable"]["std"]
    nobao = profiles.attrs["observable"]["theory_nobao"]
    k_pre = profiles_pre.attrs["observable"]["k"]
    data_pre = profiles_pre.attrs["observable"]["data"]
    theory_pre = profiles_pre.attrs["observable"]["theory"]
    std_pre = profiles_pre.attrs["observable"]["std"]
    nobao_pre = profiles_pre.attrs["observable"]["theory_nobao"]
    height_ratios = [1] * len(ells)
    fig, axes = plt.subplots(len(ells), sharex=True,
                               gridspec_kw={"height_ratios": height_ratios},
                               figsize=(6, 2 * len(ells)))
    if len(ells) == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0)
    for i, ell in enumerate(ells):
        axes[i].errorbar(k_pre[i], k_pre[i]*(data_pre[i]- nobao_pre[i]),
                         yerr=k_pre[i]*std_pre[i], marker="o", linestyle="none",
                         label="Pre", markerfacecolor="none", color=tracer_colors[tracer],alpha=0.7)
        axes[i].plot(k_pre[i], k_pre[i]*(theory_pre[i]- nobao_pre[i]), linestyle="--", color=tracer_colors[tracer],alpha=0.7)
        axes[i].errorbar(k[i], k[i]*(data[i]- nobao[i]),
                         yerr=k[i]*std[i], marker="o", linestyle="none", label="Post", color=tracer_colors[tracer])
        axes[i].plot(k[i], k[i]*(theory[i]- nobao[i]), color=tracer_colors[tracer])
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
        for tracer in tracer_info.keys():
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
