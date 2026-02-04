#!/bin/bash -l

#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/birth_track_id_%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -J BirthHaloCatalogueIndex
#SBATCH --nodes=4
#SBATCH -t 24:00:00
# N0752: 1 node, 2 hours
# N1504: 1 node, 12 hours
# N3008: 4 nodes, 24 hours

set -e

# TODO: Set these values
base_dir="/cosma8/data/dp004/dc-mcgi1/COLIBRE/BirthHaloCatalogueIndex"
output_dir="/cosma8/data/dp004/dc-mcgi1/COLIBRE/BirthHaloCatalogueIndex"
sim="L0400N3008/Thermal"
snapnum="0127"

snap_basename="${base_dir}/${sim}/snapshots/colibre_{snap_nr:04d}/colibre_{snap_nr:04d}"
membership_basename="${base_dir}/${sim}/SOAP-HBT/membership_{snap_nr:04d}/membership_{snap_nr:04d}"
output_basename="${output_dir}/${sim}/SOAP-ExSitu/birth_${snapnum}/birth_${snapnum}"

mpirun -- python misc/compute_BirthHaloCatalogueIndex.py \
  --snap-basename ${snap_basename} \
  --membership-basename ${membership_basename} \
  --output-basename ${output_basename} \
  --final-snap-nr ${snapnum} \
  --calculate-PreBirthHaloCatalogueIndex

chmod a=r "${output_basename}"*

snapshot="${snap_basename}.hdf5"
membership="${membership_basename}.{file_nr}.hdf5"
output="${output_basename}.{file_nr}.hdf5"
virtual="${output_dir}/${sim}/SOAP-ExSitu/birth_${snapnum}.hdf5"
python compression/make_virtual_snapshot.py \
  --virtual-snapshot "$snapshot" \
  --auxiliary-snapshots "$membership" "$output" \
  --output-file "$virtual" \
  --snap-nr "$snapnum" \

chmod a=r "${virtual}"

echo "Job complete!"
