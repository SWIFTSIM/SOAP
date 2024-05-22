#!/bin/bash -l
#
# Compute halo properties for a snapshot. Must run the group_membership
# script first.
#
# Job name determines which of the L1000N0900 runs we process.
# Array job index is the snapshot number to do. Submit with (for example):
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L1000N0900/halo_properties_L1000N0900.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/halo_properties_L1000N0900_%x.%a.%A.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 04:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

set -e

# Which snapshot to do
snapnum=${SLURM_ARRAY_TASK_ID}

# Which simulation to do
sim="L1000N0900/${SLURM_JOB_NAME}"

# Check for DMO run
dmo_flag=""
if [[ $sim == *DMO_* ]] ; then
  dmo_flag="--dmo"
fi

mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1 ${dmo_flag} \
       parameter_files/FLAMINGO.yml

echo "Job complete!"
