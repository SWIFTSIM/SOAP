import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

filename = '/snap8/scratch/dp004/dc-mcgi1/soap_runtime/L0100N0752_no_filter/Thermal_non_equilibrium/SOAP_uncompressed/halo_properties_0127.hdf5'
output_dir = 'z0_L0100N0752_no_filter'
filename = '/snap8/scratch/dp004/dc-mcgi1/soap_runtime/L0100N0752/Thermal_non_equilibrium/SOAP_uncompressed/halo_properties_0127.hdf5'
output_dir = 'z0_L0100N0752'
filename = '/snap8/scratch/dp004/dc-mcgi1/soap_runtime/L0100N0752/Thermal_non_equilibrium/SOAP_uncompressed/halo_properties_0064.hdf5'
output_dir = 'z3_L0100N0752'

print(output_dir)
os.makedirs(output_dir, exist_ok=True)

with h5py.File(filename, 'r') as file:
    n_bound = file['BoundSubhalo/NumberOfGasParticles'][:]
    n_bound += file['BoundSubhalo/NumberOfDarkMatterParticles'][:]
    n_bound += file['BoundSubhalo/NumberOfStarParticles'][:]
    n_bound += file['BoundSubhalo/NumberOfBlackHoleParticles'][:]
    is_central = file['InputHalos/IsCentral'][:] == 1
    host_idx = file['SOAP/HostHaloIndex'][:]
    host_n_bound = n_bound.copy()
    host_n_bound[~is_central] = n_bound[host_idx[~is_central]]

    n_loop = file['InputHalos/n_loop'][:]
    n_process = file['InputHalos/n_process'][:]
    process_time = file['InputHalos/process_time'][:]
    calc_time = 0 * process_time

    halo_props = file['Parameters'].attrs['calculations'].tolist()
    subhalo_types = file['Header'].attrs['SubhaloTypes'].tolist()
    halo_prop_times = {}
    for k in file['InputHalos'].keys():
        if k == 'total_time':
            continue
        if '_time' in k:
            halo_prop_times[k] = file[f'InputHalos/{k}'][:]
        if '_final_time' in k:
            calc_time += file[f'InputHalos/{k}'][:]

wallclock = np.sum(process_time) / 128
print(f'Estimated wallclock time (assuming 128 ranks): {wallclock}s')
print(f'Max processing time: {np.max(process_time)}s')

xlim_lo = np.min(0.9*n_bound)
xlim_hi = np.max(1.1*n_bound)
ylim_lo = np.min(0.9*process_time)
ylim_hi = np.max(1.1*process_time)

def plot_time_vs_mass(cprop, cbar_label, plot_name):
    fig, ax = plt.subplots()
    sc = ax.scatter(n_bound, process_time, s=1, c=cprop)
    fig.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel('$N_{bound}$')
    ax.set_ylabel('Time to process [s]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)
    plt.savefig(f'{output_dir}/{plot_name}.png', dpi=200)
    plt.close()

plot_time_vs_mass(
    calc_time / process_time,
    'Fraction of time spent doing calculation',
    'frac_calc'
)
plot_time_vs_mass(
    is_central,
    'Is central?',
    'is_central'
)
plot_time_vs_mass(
    n_loop,
    'Number of density loops',
    'n_loop'
)
plot_time_vs_mass(
    n_process,
    'Number of times halos was processed',
    'n_process'
)
plot_time_vs_mass(
    np.log10(host_n_bound),
    'log(Host $N_{bound}$)',
    'host_n_bound'
)

# Plotting total time taken by each time-mass bin
fig, ax = plt.subplots(1)
weights = process_time / np.sum(process_time)
n_bound_bins = 10**np.linspace(np.log10(xlim_lo), np.log10(xlim_hi), 15)
process_time_bins = 10**np.linspace(np.log10(ylim_lo), np.log10(ylim_hi), 15)
h = ax.hist2d(n_bound, process_time, weights=weights, bins=[n_bound_bins, process_time_bins], cmap='Reds', cmin=1e-5)
fig.colorbar(h[3], ax=ax, label='Fraction of total time')
ax.set_xlabel('$N_{bound}$')
ax.set_ylabel('Time to process [s]')
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig(f'{output_dir}/frac_time.png', dpi=200)
plt.close()

# Plot time taken for each halo type for different halo mass bins
n_bin = 4
n_bound_bins = 10**np.linspace(np.log10(xlim_lo), np.log10(xlim_hi), n_bin+1)
fig, axs = plt.subplots(n_bin, sharex=True)
fig.subplots_adjust(hspace=0)
for i in range(n_bin):
    mask = (n_bound_bins[i] < n_bound) & (n_bound <= n_bound_bins[i+1])

    # TODO: Mask
    # mask &= is_central
    # mask &= np.logical_not(is_central)
    # mask &= host_n_bound > 10**5

    final_fracs, total_fracs = [], []
    for halo_prop in halo_props:
        final_time = halo_prop_times[halo_prop+'_final_time'][mask]
        final_frac = np.mean(final_time / process_time[mask])
        final_fracs.append(final_frac)

        total_time = halo_prop_times[halo_prop+'_total_time'][mask]
        total_frac = np.mean(total_time / process_time[mask])
        total_fracs.append(total_frac)

    x = np.arange(len(halo_props))
    axs[i].bar(x, total_fracs)
    axs[i].bar(x, final_fracs)
    label = f'$10^{{{np.log10(n_bound_bins[i]):.2g}}} < N < 10^{{{np.log10(n_bound_bins[i+1]):.2g}}}$'
    axs[i].set_ylabel(label, fontsize=8)
axs[n_bin-1].set_xticks(x)
axs[n_bin-1].set_xticklabels(labels=halo_props, rotation=45, ha='right')
fig.text(-0.05, 0.5, 'Fraction of time spent on halo type', va='center', rotation='vertical', fontsize=14)
plt.savefig(f'{output_dir}/halo_prop_frac.png', dpi=400, bbox_inches='tight')
plt.close()

# Plot time taken for each property for different halo mass bins
# This essentially loads the entire SOAP file, so takes a while to run
n_bin = 4
n_bound_bins = 10**np.linspace(np.log10(xlim_lo), np.log10(xlim_hi), n_bin+1)
with h5py.File(filename, 'r') as file:
    for group_name in subhalo_types:
    # for group_name in ['BoundSubhalo']:
        if 'proj' in group_name:
            # We don't store times for individual projections
            continue

        prop_times = {i: [] for i in range(n_bin)}
        prop_names = []

        for k in file[group_name]:
            if '_time' not in k:
                continue

            prop_names.append(k.replace('_time', ''))
            arr = file[f'{group_name}/{k}'][:]
            for i_bin in range(n_bin):
                mask = (n_bound_bins[i_bin] < n_bound) & (n_bound <= n_bound_bins[i_bin+1])
                prop_times[i_bin].append(np.sum(arr[mask]))

        if len(prop_names) == 0:
            continue

        print(f'Plotting {group_name}')
        figsize = (0.2 * len(file[group_name].keys()), 4.8)
        fig, axs = plt.subplots(n_bin, sharex=True, figsize=figsize)
        fig.subplots_adjust(hspace=0)

        x = np.arange(len(prop_names))
        for i_bin in range(n_bin):
            axs[i_bin].bar(x, prop_times[i_bin])
            label = f'$10^{{{np.log10(n_bound_bins[i_bin]):.2g}}} < N < 10^{{{np.log10(n_bound_bins[i_bin+1]):.2g}}}$'
            axs[i_bin].set_ylabel(label, fontsize=8)
        axs[n_bin-1].set_xticks(x)
        axs[n_bin-1].set_xticklabels(labels=prop_names, rotation=45, ha='right')
        # fig.text(0, 0.5, 'Time spent on property', va='center', rotation='vertical', fontsize=14)
        plot_name = group_name.replace('/', '_')
        plt.savefig(f'{output_dir}/{plot_name}.png', dpi=400, bbox_inches='tight')
        plt.close()























