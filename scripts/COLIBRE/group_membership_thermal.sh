#!/bin/bash -l
#
# Compute group membership for each particle in a snapshot.
# Specify input and output files in this script.
# Array job index is the snapshot number to do.
#
# Submit with (for example):
#
# cd SOAP
# mkdir logs
# ./scripts/cosma_python_env.sh
# sbatch --array=0-127 -J SIM_NAME ./scripts/COLIBRE/group_membership_thermal.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/colibre_membership.%a.%j.out
#SBATCH -J group_membership_colibre
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 0:30:00
#

set -e

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="${SLURM_JOB_NAME}"

# Run the code
mpirun -- python3 -u -m mpi4py ./group_membership.py \
       parameter_files/COLIBRE_THERMAL.yml \
       --sim-name=${sim} --snap-nr=${snapnum}

echo "Job complete!"
