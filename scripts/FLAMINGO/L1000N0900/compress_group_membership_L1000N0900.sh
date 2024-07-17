#!/bin/bash
#
# Compress the membership files using h5repack, applying GZIP compression
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
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L1000N0900/compress_group_membership_L1000N0900.sh
#
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/compress_membership_L1000N0900_%x.%a.%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 02:00:00
#

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
sim="L1000N0900/${SLURM_JOB_NAME}"

# Location of the input to compress
inbase="${scratch_dir}/${sim}/SOAP_uncompressed/${halo_finder}/"

# Location of the compressed output
outbase="${output_dir}/${sim}/SOAP/${halo_finder}/"

# Create the output folder if it does not exist
outdir="${outbase}/membership_${snapnum}"
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M "${outdir}"

# Uncompressed membership file basename
input_filename="${inbase}/membership_${snapnum}/membership_${snapnum}"

# Compressed membership file basename
output_filename="${outbase}/membership_${snapnum}/membership_${snapnum}"

# Determine how many files we have
nr_files=`ls -1 ${input_filename}.*.hdf5 | wc -l`
nr_files_minus_one=$(( ${nr_files} - 1 ))

# run h5repack in parallel using 32 processes
# we could use more processes, but that causes a larger strain for the file
# system and is therefore not really more efficient
# make sure to update the 'seq' arguments when there are more/less membership
# files
echo Compressing ${nr_files} group membership files
echo Source     : ${input_filename}
echo Destination: ${output_filename}

seq 0 ${nr_files_minus_one} | parallel -j 32 \
  h5repack \
    -i "${input_filename}.{}.hdf5" \
    -o "${output_filename}.{}.hdf5" \
    -l CHUNK=10000 -f GZIP=9

echo Done.

