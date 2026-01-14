#!/bin/bash -l

#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/birth_track_id_%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -J BirthHaloCatalogueIndex
#SBATCH --nodes=1
#SBATCH -t 12:00:00
# N0752: 1 node, 2 hours
# N1504: 1 node, 12 hours
# N3008: 4 nodes, 24 hours

set -e

base_dir="/cosma8/data/dp004/jlvc76/COLIBRE/ScienceRuns"
sim="L0400N3008/Thermal"
output_dir="/cosma8/data/dp004/dc-mcgi1/COLIBRE/BirthHaloCatalogueIndex"
snapnum="0127"

snap_basename="${base_dir}/${sim}/snapshots/colibre_{snap_nr:04d}/colibre_{snap_nr:04d}"
membership_basename="${base_dir}/${sim}/SOAP-HBT/membership_{snap_nr:04d}/membership_{snap_nr:04d}"
output_basename="${output_dir}/${sim}/SOAP_BirthTrackId/birth_${snapnum}/birth_${snapnum}"

mpirun -- python misc/compute_BirthHaloCatalogueIndex.py \
  --snap-basename ${snap_basename} \
  --membership-basename ${membership_basename} \
  --output-basename ${output_basename} \
  --final-snap-nr ${snapnum} \
  --calculate-PreBirthHaloCatalogueIndex

