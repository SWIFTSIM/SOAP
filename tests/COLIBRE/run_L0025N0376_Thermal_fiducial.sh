#!/bin/bash
#
# This runs SOAP on a few halos in the L0025N0376/Thermal_fiducial
# box on Cosma8. It can be used as a quick test of new halo property
# code. Takes ~2 minutes to run.
#
# Should be run from the SOAP source directory. E.g.:
#
# cd SOAP
# ./tests/FLAMINGO/run_L0025N0376_Thermal_fiducial.sh
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Which simulation to do
sim="L0025N0376/Thermal_fiducial"

# Snapshot number to do
snapnum=0123

# Halo indices to do: all halos with x<1, y<1, and z<1 cMpc in snap 123
halo_indices="1938 1850 1907 1150 1947 1234 436621 1056 1900 1858 1988 1212 1564 1948 2574 1567 1331 1940"

# Run SOAP on eight cores processing the selected halos. Use 'python3 -m pdb' to start in the debugger.
mpirun -np 8 python3 -u -m mpi4py ./compute_halo_properties.py \
       ./tests/COLIBRE/parameters.yml \
       --halo-indices ${halo_indices} \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1

