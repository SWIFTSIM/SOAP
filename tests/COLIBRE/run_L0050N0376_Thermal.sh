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
sim="L0050N0376/Thermal"

# Snapshot number to do
snapnum=0127

# Halo indices to do: all halos with x<1, y<1, and z<1 cMpc in snap 123
halo_indices="2257 2664 3070 4086 4131 4307 5166 6418 6662 8644 9353 9480 10148 11102 11320 11902 12009 12123 13023 102778 312594 3251 4486 5352 5822 6682 8674 10975 12699 13133 64 10641 102411 1252 10511 1833 2524 5195 5756 5829 8224 10544 12046 195631 488 2576 4916 7732 7994 8008 8676 9320 10424 10506 10711 484989 10637"

# Create parameters files
python tests/COLIBRE/create_parameters_file.py

# Remove tmp directory (so we don't load chunks if they already exist)
rm -r output/SOAP-tmp

# Run SOAP on eight cores processing the selected halos. Use 'python3 -m pdb' to start in the debugger.
mpirun -np 8 python3 -u -m mpi4py SOAP/compute_halo_properties.py \
       ./tests/COLIBRE/test_parameters.yml \
       --halo-indices ${halo_indices} \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1

