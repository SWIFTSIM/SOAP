#!/bin/bash
#
# Compress SOAP catalogues.
#
# Before running set the location of the input/output/scratch directories.
# Pass the simulation variation and snapshots to process when running, e.g.
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L1000N1800/compress_halo_properties_L1000N1800.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/compress_properties_%x.%a.%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 01:00:00

set -e

module purge
module load python/3.12.4

# TODO: Set these locations
input_dir="/snap8/scratch/dp004/dc-mcgi1/FLAMINGO/Runs"
output_dir="/cosma8/data/dp004/dc-mcgi1/FLAMINGO/Runs"
scratch_dir="/snap8/scratch/dp004/dc-mcgi1/FLAMINGO/Runs"

# compression script
script="./compression/compress_soap_catalogue.py"

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L1000N1800/${SLURM_JOB_NAME}"

# Name of the input SOAP catalogue
input_filename="${input_dir}/${sim}/SOAP_uncompressed/halo_properties_${snapnum}.hdf5"

# Location and name of the output SOAP catalogue
outbase="${output_dir}/${sim}/SOAP"
mkdir -p $outbase
output_filename="${outbase}/halo_properties_${snapnum}.hdf5"

# directory used to store temporary files (preferably a /snap8 directory for
# faster writing and reading)
tmp_dir="${scratch_dir}/${sim}/SOAP_compression_tmp/"

# run the script using all available threads on the node
mpirun -- python -u ${script} ${input_filename} ${output_filename} ${tmp_dir}

echo "Job complete!"
