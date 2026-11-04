#!/bin/bash -l

#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/history_properties_%j.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -J HistoryProperties
#SBATCH --nodes=1
#SBATCH -t 4:00:00
# Timings for computing all properties on a single node with 128 ranks.
# DMO runs take around half as long.
# L025m7: 3 minutes
# L100m6: 20 minutes
# L200m6: 1.5 hours, ~500GB peak memory with 4 --target-snap-nrs values

set -e

# TODO: Set these values
input_dir="/cosma8/data/dp004/colibre/Runs"
output_dir="/cosma8/data/dp004/dc-mcgi1/COLIBRE"
sim="L0025N0188/Thermal"
# Snapshots for z=0, 1, 2, 3
snapnums="127 92 76 64"

soap_basename="${input_dir}/${sim}/SOAP-HBT/halo_properties_{snap_nr:04d}"
output_basename="${output_dir}/${sim}/SOAP-History/history_{snap_nr:04d}"

# By default all property types are computed for their default halo types.
# To compute a subset, pass the arguments for the property types you
# want, each with a comma separated list of halo types, e.g.
#   --bh-maximum BoundSubhalo,ExclusiveSphere
#   --half-mass-formation BoundSubhalo,SO
#   --half-stellar-mass-formation BoundSubhalo,ExclusiveSphere
#   --stellar-mass-maximum BoundSubhalo,ExclusiveSphere
# Running with -m mpi4py means all ranks are aborted if one raises an error
mpirun -- python -u -m mpi4py misc/compute_history_properties.py \
  --soap-basename ${soap_basename} \
  --output-basename ${output_basename} \
  --target-snap-nrs ${snapnums}

chmod a=r "${output_dir}/${sim}/SOAP-History/"history_*.hdf5

echo "Job complete!"
