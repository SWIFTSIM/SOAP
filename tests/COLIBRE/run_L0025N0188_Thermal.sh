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
sim="L0025N0188/Thermal"

# Snapshot number to do
snapnum=127

# Halo indices to do: all halos with x<1, y<1, and z<1 cMpc in snap 123
halo_indices="1474 1668 1819 1995 2087 2114 2174 2314 2332 2386 2389 2390 2392 2439 2441 2444 2496 2498 2528 14937 78314"

# Create parameters files
python tests/COLIBRE/create_parameters_file.py

# Remove tmp directory (so we don't load chunks if they already exist)
rm -r output/SOAP-tmp

# Run SOAP on eight cores processing the selected halos. Use 'python3 -m pdb' to start in the debugger.
mpirun -np 8 python3 -u -m mpi4py SOAP/compute_halo_properties.py \
       ./tests/COLIBRE/test_parameters.yml \
       --halo-indices ${halo_indices} \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1

