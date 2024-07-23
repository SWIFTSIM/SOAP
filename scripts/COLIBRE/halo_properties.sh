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
# sbatch --array=0-3 -J SIM_NAME  ./scripts/COLIBRE/halo_properties.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/colibre_properties_%a.%j.out
#SBATCH -J halo_properties_colibre
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 04:00:00
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="${SLURM_JOB_NAME}"

# TODO: Set dmo_flag if needed
dmo_flag=""

#TODO: Set nodes and chunks
mpirun -- python3 -u -m mpi4py ./compute_halo_properties.py \
       parameter_files/COLIBRE.yml \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1 ${dmo_flag}

echo "Job complete!"
