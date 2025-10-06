#!/bin/bash
#
# Compress the membership files using h5repack, applying GZIP compression
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
# sbatch -J L0025N0376/Thermal_fiducial --array=0-127%4 ./scripts/COLIBRE/compress_group_membership.sh
#
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/compress_membership.%a.%A.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 00:30:00
#

set -e

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

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

# Location of the input to compress
inbase="${scratch_dir}/${sim}/SOAP_uncompressed/"

# Location of the compressed output
outbase="${output_dir}/${sim}/SOAP/"

# Create the output folder if it does not exist
outdir="${outbase}/membership_${snapnum}"
mkdir -p "${outdir}"

# Uncompressed membership file basename
input_filename="${inbase}/membership_${snapnum}/membership_${snapnum}"

# Compressed membership file basename
output_filename="${outbase}/membership_${snapnum}/membership_${snapnum}"

# Determine how many files we have
nr_files=`ls -1 ${input_filename}.*.hdf5 | wc -l`
nr_files_minus_one=$(( ${nr_files} - 1 ))

# run h5repack in parallel using 32 processes on files 0 to 63
# we could use more processes, but that causes a larger strain for the file
# system and is therefore not really more efficient
# make sure to update the 'seq' arguments when there are more/less membership
# files
echo Compressing ${nr_files} group membership files
echo Source     : ${input_filename}
echo Destination: ${output_filename}

seq 0 ${nr_files_minus_one} | xargs -I {} -P 32 bash -c \
  "h5repack -i ${input_filename}.{}.hdf5 -o ${output_filename}.{}.hdf5 -l CHUNK=10000 -f GZIP=4"

echo "Setting files to be read-only"
chmod a=r "${output_filename}"*

echo "Creating virtual snapshot"
snapshot="${output_dir}/${sim}/snapshots/colibre_${snapnum}/colibre_${snapnum}.hdf5"
membership="${output_filename}.{file_nr}.hdf5"
virtual="${outbase}/colibre_with_SOAP_membership_${snapnum}.hdf5"
python compression/make_virtual_snapshot.py $snapshot $membership $virtual

echo "Setting virtual file to be read-only"
chmod a=r "${virtual}"

echo "Job complete!"

