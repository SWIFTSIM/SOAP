#!/bin/bash -l
#
# Compute HBT group membership for each particle in a snapshot.
#
# Job name determines which of the L2800N10080 runs we process.
# Array job index is the snapshot number to do.
#
# Submit with (for example):
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L2800N10080/hbt_membership_L2800N10080.sh
#
#SBATCH --nodes=98
#SBATCH --cpus-per-task=3
##SBATCH --tasks-per-node=64
#SBATCH -o ./logs/hbt_membership_L2800N10080_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 5:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L2800N10080/${SLURM_JOB_NAME}"

# Run the code
mpirun -np 4096 python3 -u -m mpi4py ./group_membership.py \
       ./scripts/FLAMINGO/L2800N10080/hbt_parameters.yml --sim-name=${sim} --snap-nr=${snapnum}
