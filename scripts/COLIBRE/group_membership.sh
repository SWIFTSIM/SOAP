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
# sbatch --array=0-3 ./scripts/COLIBRE/group_membership.sh
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

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Run the code
mpirun python3 -u -m mpi4py ./group_membership.py \
       parameter_files/COLIBRE.yml \
       --snap-nr=${snapnum}

echo "Job complete!"
