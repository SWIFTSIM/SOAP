#!/bin/bash -l
#
# Compute halo properties for a snapshot. Must run the group_membership
# script first.
#
# Job name determines which of the L2800N10080 runs we process.
# Array job index is the snapshot number to do. Submit with (for example):
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L2800N10080/hbt_properties_L2800N10080_HYDRO.sh
#
#SBATCH --nodes=32
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/hbt_properties_L2800N10080_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 5:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which snapshot to do
snapnum=${SLURM_ARRAY_TASK_ID}

# Which simulation to do
sim="L2800N10080/${SLURM_JOB_NAME}"

# Check for DMO run
dmo_flag=""
if [[ $sim == DMO_* ]] ; then
  dmo_flag="--dmo"
fi

mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
       ./scripts/FLAMINGO/L2800N10080/hbt_parameters.yml \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=256 ${dmo_flag}
