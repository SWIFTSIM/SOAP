#!/bin/bash -l
#
# Compute group membership for each particle in a snapshot.
#
# Job name determines which of the L1000N1800 runs we process.
# Array job index is the snapshot number to do.
#
# Submit with (for example):
#
# cd SOAP
# mkdir logs
# ./scripts/cosma_python_env.sh
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L1000N1800/group_membership_L1000N1800.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=128
#SBATCH -o ./logs/group_membership_L1000N1800_%x.%a.%A.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 0:30:00
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

set -e

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L1000N1800/${SLURM_JOB_NAME}"

# Run the code
mpirun -- python3 -u -m mpi4py SOAP/group_membership.py \
       --sim-name=${sim} --snap-nr=${snapnum} \
       parameter_files/FLAMINGO.yml

echo "Job complete!"
