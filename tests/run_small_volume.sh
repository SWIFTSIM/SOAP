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

# Run the group membership script
mpirun -np 8 python -u SOAP/group_membership.py \
    --sim-name=DM_test \
    --snap-nr=18 \
    tests/small_volume.yml

# Calculate halo properties
mpirun -np 8 python -u SOAP/compute_halo_properties.py \
    --sim-name=DM_test \
    --snap-nr=18 \
    --chunks=1 \
    --dmo \
    tests/small_volume.yml

# Generate documentation
python SOAP/property_table.py \
    tests/small_volume.yml \
    test_data/snap_0018.hdf5
cd documentation
pdflatex -halt-on-error SOAP.tex
