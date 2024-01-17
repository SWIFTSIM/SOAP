#!/bin/bash -l
#
# Compute halo properties for a snapshot. 
# Must run the group_membership script first.
# Specify input and output files in this script.
# Array job index is the snapshot number to do.
#
# Submit with (for example):
#
# cd SOAP
# mkdir logs
# sbatch --array=0-3 ./scripts/COLIBRE/halo_properties.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/halo_properties_colibre.%a.out
#SBATCH -J halo_properties_colibre
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#TODO: Set runtime
#SBATCH -t 00:30:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

#TODO: Set these variables
snapshot_dir=""
vr_dir=""
scratch_dir=""
output_dir=""

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Generate input and output file names
snapshot_filename="${snapshot_dir}colibre_${snapnum}.hdf5"
vr_basename="${vr_dir}halo_${snapnum}"
chunkdir="${scratch_dir}/SOAP-tmp/"
outbase="${output_dir}/SOAP_uncompressed/"
membership_filename="${outbase}/membership_%(snap_nr)04d/membership_%(snap_nr)04d.%(file_nr)d.hdf5"
outfile="${outbase}/halo_properties_%(snap_nr)04d.hdf5"

# TODO: Set dmo_flag if needed
dmo_flag=""

# TODO: How many chunks are needed?
nr_chunks=1

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M ${outdir}

# Run the code
mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
    ${snapshot_filename} ${chunkdir} ${vr_basename} ${outfile} ${SLURM_ARRAY_TASK_ID} \
    --chunks=${nr_chunks} ${dmo_flag} \
    --extra-input=${membership_filename} \
    --parameters parameter_files/colibre_SOAP_params.yml

