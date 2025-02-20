import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(
    description="Generates plots to identify where SOAP spent its time."
)
parser.add_argument(
    "soap_catalogue", type=str, help="Filepath to SOAP catalogue to analyse"
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="timings",
    help="Name of output directory to place plots within",
)
args = parser.parse_args()

filename = args.soap_catalogue
output_dir = args.output
print(f"Output directory for plots: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

with h5py.File(filename, "r") as file:
    # Extract some subhalo properties
    n_bound = file["InputHalos/NumberOfBoundParticles"][:]
    is_central = file["InputHalos/IsCentral"][:] == 1
    host_idx = file["SOAP/HostHaloIndex"][:]
    host_n_bound = n_bound.copy()
    host_n_bound[~is_central] = n_bound[host_idx[~is_central]]

    # Load the number of loops required to process this halo
    n_loop = file["InputHalos/n_loop"][:]
    n_process = file["InputHalos/n_process"][:]
    # Load the total processing time for each subhalo
    process_time = file["InputHalos/process_time"][:]

    # Load the processing time for each halo type
    # (e.g. time spend on SO/200_crit)
    calc_time = 0 * process_time
    halo_props = file["Parameters"].attrs["calculations"].tolist()
    subhalo_types = file["Header"].attrs["SubhaloTypes"].tolist()
    halo_prop_times = {}
    for k in file["InputHalos"].keys():
        if "_time" in k:
            halo_prop_times[k] = file[f"InputHalos/{k}"][:]
        if "_final_time" in k:
            calc_time += file[f"InputHalos/{k}"][:]

# Defining limits used by multiple plots
xlim_lo = np.min(0.9 * n_bound)
xlim_hi = np.max(1.1 * n_bound)
ylim_lo = np.min(0.9 * process_time)
ylim_hi = np.max(1.1 * process_time)

#######################

print("Plotting total time to process each subhalo")


def plot_time_vs_nbound(cprop, cbar_label, plot_name):
    """
    Plot the total time taken to process a subhalo
    against its Nbound. Color the points according
    to cprop.
    """
    fig, ax = plt.subplots()
    sc = ax.scatter(n_bound, process_time, s=1, c=cprop)
    fig.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel("Subhalo $N_{bound}$")
    ax.set_ylabel("Time to process [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)
    plt.savefig(f"{output_dir}/ProcessTime_{plot_name}.png", dpi=200)
    plt.close()


plot_time_vs_nbound(
    calc_time / process_time, "Fraction of time spent doing calculation", "FracCalc"
)
plot_time_vs_nbound(is_central, "Is central?", "IsCentral")
plot_time_vs_nbound(n_loop, "Number of density loops", "Nloop")
plot_time_vs_nbound(n_process, "Number of times the subhalo was processed", "Nprocess")
plot_time_vs_nbound(np.log10(host_n_bound), "log(Host $N_{bound}$)", "HostNbound")

#######################

print("Plotting total processing time taken by each time-mass bin")

fig, ax = plt.subplots(1)
weights = process_time / np.sum(process_time)
n_bound_bins = 10 ** np.linspace(np.log10(xlim_lo), np.log10(xlim_hi), 15)
process_time_bins = 10 ** np.linspace(np.log10(ylim_lo), np.log10(ylim_hi), 15)
h = ax.hist2d(
    n_bound,
    process_time,
    weights=weights,
    bins=[n_bound_bins, process_time_bins],
    cmap="Reds",
    cmin=1e-5,
)
fig.colorbar(h[3], ax=ax, label="Fraction of total time")
ax.set_xlabel("$N_{bound}$")
ax.set_ylabel("Time to process [s]")
ax.set_xscale("log")
ax.set_yscale("log")
plt.savefig(f"{output_dir}/SubhaloMassTimeFraction.png", dpi=200)
plt.close()

#######################

print("Plotting time taken for each halo type, splitting results into mass bins")

n_bin = 4

n_bound_bins = 10 ** np.linspace(np.log10(xlim_lo), np.log10(xlim_hi), n_bin + 1)
fig, axs = plt.subplots(n_bin, sharex=True)
fig.subplots_adjust(hspace=0)
for i in range(n_bin):
    mask = (n_bound_bins[i] < n_bound) & (n_bound <= n_bound_bins[i + 1])

    final_fracs, total_fracs = [], []
    for halo_prop in halo_props:
        final_time = halo_prop_times[halo_prop + "_final_time"][mask]
        final_frac = np.mean(final_time / process_time[mask])
        final_fracs.append(final_frac)

        total_time = halo_prop_times[halo_prop + "_total_time"][mask]
        total_frac = np.mean(total_time / process_time[mask])
        total_fracs.append(total_frac)

    x = np.arange(len(halo_props))
    axs[i].bar(x, total_fracs)
    axs[i].bar(x, final_fracs)
    label = f"$10^{{{np.log10(n_bound_bins[i]):.2g}}} < N < 10^{{{np.log10(n_bound_bins[i+1]):.2g}}}$"
    axs[i].set_ylabel(label, fontsize=8)
axs[n_bin - 1].set_xticks(x)
axs[n_bin - 1].set_xticklabels(labels=halo_props, rotation=45, ha="right")
fig.text(
    -0.05,
    0.5,
    "Fraction of time spent on halo type",
    va="center",
    rotation="vertical",
    fontsize=14,
)
plt.savefig(f"{output_dir}/HaloTypeTimeFraction.png", dpi=400, bbox_inches="tight")
plt.close()

#######################

print(
    "Plotting the time taken on each property for each halo type, splitting results into mass bins"
)
# This essentially loads the entire SOAP file, so takes a while to run

n_bin = 4


# This loop also saves the total time spent on calculations,
# which is needed for the next plot
def get_internal_name(subhalo_type):
    """
    Helper function to convert HDF5 group names to internal SOAP names
    """
    if subhalo_type == "BoundSubhalo":
        return "bound_subhalo"
    if "ExclusiveSphere" in subhalo_type:
        r = subhalo_type.split("/")[1].replace("kpc", "")
        return f"exclusive_sphere_{r}kpc"
    if "InclusiveSphere" in subhalo_type:
        r = subhalo_type.split("/")[1].replace("kpc", "")
        return f"inclusive_sphere_{r}kpc"
    if "ProjectedAperture" in subhalo_type:
        r = subhalo_type.split("/")[1].replace("kpc", "")
        return f"projected_aperture_{r}kpc"
    if "SO/" in subhalo_type:
        return f"SO_{subhalo_type.split('/')[1]}"
    return None


total_prop_calculation_time = {}

n_bound_bins = 10 ** np.linspace(np.log10(xlim_lo), np.log10(xlim_hi), n_bin + 1)
with h5py.File(filename, "r") as file:
    for group_name in subhalo_types:
        # We don't store times for individual projections
        if ("proj" in group_name) or (group_name == "InputHalos"):
            continue

        prop_times = {i: [] for i in range(n_bin)}
        prop_names = []

        prop_calculation_time = 0 * halo_prop_times["bound_subhalo_final_time"]
        for k in file[group_name]:
            if "_time" not in k:
                continue

            prop_names.append(k.replace("_time", ""))
            arr = file[f"{group_name}/{k}"][:]
            # TODO: Remove, was previously outputting times as shape (n, 1)
            arr = arr.reshape(-1)
            for i_bin in range(n_bin):
                mask = (n_bound_bins[i_bin] < n_bound) & (
                    n_bound <= n_bound_bins[i_bin + 1]
                )
                prop_times[i_bin].append(np.sum(arr[mask]))
                prop_calculation_time[mask] += arr[mask]

        if len(prop_names) == 0:
            continue
        print(f"Plotting {group_name}")
        internal_name = get_internal_name(group_name)
        total_prop_calculation_time[internal_name] = prop_calculation_time

        # ProjectedAperture times are store in the top level group, but the properties
        # themselves are stored in the individual projections. This means we have
        # less keys, and so need to take that into account when setting the figsize.
        if "ProjectedAperture" in group_name:
            figsize = (0.4 * len(file[group_name].keys()), 4.8)
        else:
            figsize = (0.2 * len(file[group_name].keys()), 4.8)
        fig, axs = plt.subplots(n_bin, sharex=True, figsize=figsize)
        fig.subplots_adjust(hspace=0)

        x = np.arange(len(prop_names))
        for i_bin in range(n_bin):
            axs[i_bin].bar(x, prop_times[i_bin])
            label = f"$10^{{{np.log10(n_bound_bins[i_bin]):.2g}}} < N < 10^{{{np.log10(n_bound_bins[i_bin+1]):.2g}}}$"
            axs[i_bin].set_ylabel(label, fontsize=8)
        axs[n_bin - 1].set_xticks(x)
        axs[n_bin - 1].set_xticklabels(labels=prop_names, rotation=45, ha="right")
        # fig.text(0, 0.5, 'Time spent on property', va='center', rotation='vertical', fontsize=14)
        plot_name = f"PropertyTime_{group_name.replace('/', '_')}"
        plt.savefig(f"{output_dir}/{plot_name}.png", dpi=400, bbox_inches="tight")
        plt.close()

#######################

print("Plotting the time taken to set up for each halo type")

n_bin = 4

n_bound_bins = 10 ** np.linspace(np.log10(xlim_lo), np.log10(xlim_hi), n_bin + 1)
fig, axs = plt.subplots(n_bin, sharex=True)
fig.subplots_adjust(hspace=0)
for i in range(n_bin):
    mask = (n_bound_bins[i] < n_bound) & (n_bound <= n_bound_bins[i + 1])

    fracs = []
    for halo_prop in halo_props:
        final_time = halo_prop_times[halo_prop + "_final_time"][mask]
        sum_calc_time = total_prop_calculation_time[halo_prop][mask]
        frac = 1 - np.mean(sum_calc_time / final_time)
        fracs.append(frac)

    x = np.arange(len(halo_props))
    axs[i].bar(x, fracs)
    label = f"$10^{{{np.log10(n_bound_bins[i]):.2g}}} < N < 10^{{{np.log10(n_bound_bins[i+1]):.2g}}}$"
    axs[i].set_ylabel(label, fontsize=8)
axs[n_bin - 1].set_xticks(x)
axs[n_bin - 1].set_xticklabels(labels=halo_props, rotation=45, ha="right")
fig.text(
    -0.05,
    0.5,
    "Fraction of time spent setting up",
    va="center",
    rotation="vertical",
    fontsize=14,
)
plt.savefig(f"{output_dir}/HaloTypeSetupTime.png", dpi=400, bbox_inches="tight")
plt.close()

#######################

print("Done!")
