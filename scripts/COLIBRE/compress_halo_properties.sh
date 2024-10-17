#!/bin/bash
#
# Compress SOAP catalogues.
#
# Output locations are specified by enviroment variables. E.g.
#
# export COLIBRE_SCRATCH_DIR=/snap8/scratch/dp004/${USER}/COLIBRE/ScienceRuns/
# export COLIBRE_OUTPUT_DIR=/cosma8/data/dp004/${USER}/COLIBRE/ScienceRuns/
#
# To run:
#
# cd SOAP
# mkdir logs
# sbatch -J L0025N0376/Thermal_fiducial --array=0-127%4 ./scripts/COLIBRE/compress_halo_properties.sh
#
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/compress_properties.%a.%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 01:00:00
#SBATCH --reservation=colibre
#

set -e

module purge
module load python/3.12.4

# Get location for temporary output
if [[ "${COLIBRE_SCRATCH_DIR}" ]] ; then
  scratch_dir="${COLIBRE_SCRATCH_DIR}"
else
  echo Please set COLIBRE_SCRATCH_DIR
  exit 1
fi

# Get location for final output
if [[ "${COLIBRE_OUTPUT_DIR}" ]] ; then
  output_dir="${COLIBRE_OUTPUT_DIR}"
else
  echo Please set COLIBRE_OUTPUT_DIR
  exit 1
fi

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="${SLURM_JOB_NAME}"

# compression script
script="./compression/compress_fast_metadata.py"

# Location of the input to compress
inbase="${scratch_dir}/${sim}/SOAP_uncompressed/"

# Location of the compressed output
outbase="${output_dir}/${sim}/SOAP/"
mkdir -p $outbase

# Name of the input SOAP catalogue
input_filename="${inbase}/halo_properties_${snapnum}.hdf5"

# name of the output SOAP catalogue
output_filename="${outbase}/halo_properties_${snapnum}.hdf5"

# directory used to store temporary files (preferably a /snap8 directory for
# faster writing and reading)
scratch_dir="${scratch_dir}/${sim}/SOAP_compression_tmp/"

# run the script using all available threads on the node
python3 -u ${script} --nproc 128 ${input_filename} ${output_filename} ${scratch_dir}

chmod a=r ${output_filename}

echo "Job complete!"
