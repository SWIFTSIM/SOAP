#!/bin/bash
#
# This runs SOAP on a small DMO box
#
set -e

# Load the correct modules if we're running on cosma
if [[ $(hostname) == *cosma* ]] ; then
  module purge
  module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
  source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate
fi

# Download the required data
python tests/helpers.py

# Set "oversubscribe" for MPI so we don't fail on machines with <8 cores
export OMPI_MCA_rmaps_base_oversubscribe=1
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe

# Run SOAP. This computes the group membership and halo properties, and then
# creates a virtual snapshot linking the particle data and membership files
mpirun -np 8 python -u SOAP/run_soap.py \
    --sim-name=DM_test \
    --snap-nr=18 \
    --chunks=1 \
    --dmo \
    tests/small_volume.yml

# Generate documentation
echo
echo "######## Generating documentation"
python SOAP/property_table.py \
    tests/small_volume.yml \
    test_data/snap_0018.hdf5 \
    --dmo
# Build the html documentation if sphinx is installed
if command -v sphinx-build > /dev/null ; then
  sphinx-build -q -W -b html documentation documentation/build
  echo "Done"
else
  echo "sphinx-build not found, skipping the documentation build."
  echo "Install it with pip install \".[docs]\""
fi
