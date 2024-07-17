#!/bin/bash -l
#
# Compute group membership for each particle in a snapshot.
#
# Job name determines which of the L1000N3600 runs we process.
# Array job index is the snapshot number to do.
#
# Submit with (for example):
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=0-78%4 ./scripts/FLAMINGO/L1000N3600/group_membership_L1000N3600.sh
#
#SBATCH --nodes=8
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=32
#SBATCH -o ./logs/group_membership_L1000N3600_%x.%a.%A.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 1:00:00
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

set -e

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L1000N3600/${SLURM_JOB_NAME}"

# Run the code
mpirun -- python3 -u -m mpi4py ./group_membership.py \
       --sim-name=${sim} --snap-nr=${snapnum} \
       parameter_files/FLAMINGO.yml

echo "Job complete!"
