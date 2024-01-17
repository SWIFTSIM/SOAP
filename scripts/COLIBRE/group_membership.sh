#!/bin/bash -l
#
# Compute VR group membership for each particle in a snapshot.
# Specify input and output files in this script.
# Array job index is the snapshot number to do.
#
# Submit with (for example):
#
# cd SOAP
# mkdir logs
# sbatch --array=0-3 ./scripts/COLIBRE/group_membership.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/group_membership_colibre.%a.out
#SBATCH -J group_membership_colibre
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 0:30:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

#TODO: Set these variables
snapshot_dir=""
vr_dir=""
output_dir=""

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Generate input and output file names
snapshot_filename="${snapshot_dir}colibre_${snapnum}.hdf5"
vr_basename="${vr_dir}halo_${snapnum}"
outbase="${output_dir}/SOAP_uncompressed/"
outfile="${outbase}/membership_${snapnum}/membership_${snapnum}.%(file_nr)d.hdf5"

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M ${outdir}

# Run the code
mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${snapshot_filename} ${vr_basename} ${outfile}
