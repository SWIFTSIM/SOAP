#!/bin/bash -l
#
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/eagle_%a.%j.out
#SBATCH -J soap_eagle
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 02:00:00
#

sim_name='L0100N1504'
snap_nr="028"
z_suffix="z000p000"

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

######## Link files to snap (to remove awful z suffix)

sim_dir="/cosma7/data/Eagle/ScienceRuns/Planck1/${sim_name}/PE/REFERENCE/data"
output_dir="/snap7/scratch/dp004/dc-mcgi1/SOAP_EAGLE/${sim_name}"

sim_snap_dir="${sim_dir}/particledata_${snap_nr}_${z_suffix}"
output_snap_dir="${output_dir}/gadget_snapshots/snapshot_${snap_nr}"
mkdir -p $output_snap_dir
i=0
while [[ -e "${sim_snap_dir}/eagle_subfind_particles_${snap_nr}_${z_suffix}.${i}.hdf5" ]]; do
    old_name="${sim_snap_dir}/eagle_subfind_particles_${snap_nr}_${z_suffix}.${i}.hdf5"
    new_name="${output_snap_dir}/snap_${snap_nr}.${i}.hdf5"
    ln -s $old_name $new_name
    ((i++))
done

sim_group_dir="${sim_dir}/groups_${snap_nr}_${z_suffix}"
output_group_dir="${output_dir}/subfind/groups_${snap_nr}"
mkdir -p $output_group_dir
i=0
while [[ -e "${sim_group_dir}/eagle_subfind_tab_${snap_nr}_${z_suffix}.${i}.hdf5" ]]; do
    old_name="${sim_group_dir}/eagle_subfind_tab_${snap_nr}_${z_suffix}.${i}.hdf5"
    new_name="${output_group_dir}/subfind_tab_${snap_nr}.${i}.hdf5"
    ln -s $old_name $new_name
    ((i++))
done

######### Create SWIFT snapshot

set -e

mpirun -- python -u misc/convert_eagle.py \
    --snap-basename "${output_dir}/gadget_snapshots/snapshot_${snap_nr}/snap_${snap_nr}" \
    --output-basename "${output_dir}/swift_snapshots/swift_${snap_nr}/snap_${snap_nr}" \
    --membership-basename "${output_dir}/SOAP_uncompressed/membership_${snap_nr}/membership_${snap_nr}"

######### Run SOAP

chunks=10

mpirun -- python3 -u -m mpi4py SOAP/compute_halo_properties.py \
       parameter_files/EAGLE.yml \
       --sim-name=${sim_name} --snap-nr=${snap_nr} --chunks=${chunks}

##############

echo "Job complete!"
