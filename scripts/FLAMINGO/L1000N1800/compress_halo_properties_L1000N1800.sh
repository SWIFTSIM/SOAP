#!/bin/bash
#
# Compress SOAP catalogues.
#
# Output locations are specified by enviroment variables. E.g.
#
# export FLAMINGO_SCRATCH_DIR=/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/
# export FLAMINGO_OUTPUT_DIR=/cosma8/data/dp004/${USER}/FLAMINGO/ScienceRuns/
# export HALO_FINDER=HBTplus
#
# To run:
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L1000N1800/compress_halo_properties_L1000N1800.sh
#
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/compress_properties_L1000N1800_%x.%a.%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 01:00:00
#

set -e

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Get location for temporary output
if [[ "${FLAMINGO_SCRATCH_DIR}" ]] ; then
  scratch_dir="${FLAMINGO_SCRATCH_DIR}"
else
  echo Please set FLAMINGO_SCRATCH_DIR
  exit 1
fi

# Get location for final output
if [[ "${FLAMINGO_OUTPUT_DIR}" ]] ; then
  output_dir="${FLAMINGO_OUTPUT_DIR}"
else
  echo Please set FLAMINGO_OUTPUT_DIR
  exit 1
fi

# Get halo finder used
if [[ "${HALO_FINDER}" ]] ; then
  halo_finder="${HALO_FINDER}"
else
  echo Please set HALO_FINDER
  exit 1
fi

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L1000N1800/${SLURM_JOB_NAME}"

# compression script
script="./compression/compress_fast_metadata.py"

# Location of the input to compress
inbase="${scratch_dir}/${sim}/SOAP_uncompressed/${halo_finder}/"

# Location of the compressed output
outbase="${output_dir}/${sim}/SOAP/${halo_finder}/"
mkdir -p $outbase

# Name of the input SOAP catalogue
input_filename="${inbase}/halo_properties_${snapnum}.hdf5"

# name of the output SOAP catalogue
output_filename="${outbase}/halo_properties_${snapnum}.hdf5"

# directory used to store temporary files (preferably a /snap8 directory for
# faster writing and reading)
scratch_dir="${scratch_dir}/${sim}/SOAP_compression_tmp/"

# run the script using all available threads on the node
python3 ${script} --nproc 128 ${input_filename} ${output_filename} ${scratch_dir}

chmod a=r ${output_filename}

echo "Job complete!"
