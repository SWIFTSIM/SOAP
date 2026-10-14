#!/bin/bash -l

#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/descendant_info_%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -J DescendantInfo
#SBATCH --nodes=1
#SBATCH -t 00:30:00

set -e

# TODO: Set these values
input_dir="/cosma8/data/dp004/colibre/Runs"
output_dir="/cosma8/data/dp004/dc-mcgi1/COLIBRE"
sim="L0025N0188/Thermal"
final_snapnum=127

hbt_basename="${input_dir}/${sim}/HBT-HERONS/sorted_catalogues/OrderedSubSnap_{snap_nr:03d}.hdf5"
snapshot_basename="${input_dir}/${sim}/snapshots/colibre_{snap_nr:04d}/colibre_{snap_nr:04d}.{file_nr}.hdf5"
soap_basename="${input_dir}/${sim}/SOAP-HBT/halo_properties_{snap_nr:04d}"
output_basename="${output_dir}/${sim}/SOAP-DescendantInfo/descendant_info_{snap_nr:04d}"

# An output catalogue is written for every snapshot, even if there is no
# SOAP catalogue. Where a SOAP catalogue does a cross check.
# Remove --soap-basename to skip the comparison.
mpirun -- python -u misc/compute_descendant_info.py \
  --hbt-basename ${hbt_basename} \
  --snapshot-basename ${snapshot_basename} \
  --soap-basename ${soap_basename} \
  --output-basename ${output_basename} \
  --final-snap-nr ${final_snapnum}

chmod a=r "${output_dir}/${sim}/SOAP-DescendantInfo/"descendant_info_*.hdf5

echo "Job complete!"
