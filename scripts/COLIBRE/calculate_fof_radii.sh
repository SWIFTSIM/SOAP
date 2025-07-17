#!/bin/bash -l
#
# Create FOF catalogues containing radii. Note this links to the old
# catalogues for the other FOF properties, THEY ARE NOT COPIED.
#
# Run by passing the snapshots to process as the job array, and the 
# simulation as the job name. E.g.
#
# sbatch -J L0025N0188/Thermal --array=0-127%4 ./scripts/COLIBRE/calculate_fof_radii.sh
#
# N3008 runs need 8 nodes for z>1, 16 for z<1
# N1504 runs need 1 node for z>1, 2 for z<1
# Other runs need 1 node for all snapshots
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/fof_radii_%a.%A.out
#SBATCH -J fof_radii
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 01:00:00
#

set -e

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Which snapshot to do
snapnum=$(printf '%04d' ${SLURM_ARRAY_TASK_ID})

# Which simulation to do
sim="${SLURM_JOB_NAME}"

colibre_dir="/cosma8/data/dp004/dc-mcgi1/COLIBRE/fof_radii"
snap_basename="${colibre_dir}/${sim}/snapshots/colibre_${snapnum}/colibre_${snapnum}"
fof_basename="${colibre_dir}/${sim}/fof/fof_output_${snapnum}/fof_output_${snapnum}"
output_basename="${colibre_dir}/${sim}/fof_radii/fof_output_${snapnum}/fof_output_${snapnum}"

mpirun -- python -u misc/calculate_fof_radii.py \
  --snap-basename "${snap_basename}" \
  --fof-basename "${fof_basename}" \
  --output-basename "${output_basename}" \

echo "Setting files to read-only"
chmod a=r "${output_basename}"*

echo "Job complete!"
