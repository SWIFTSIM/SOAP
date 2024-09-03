#!/bin/bash
#
# This runs SOAP on a few halos in the L1000N1800/DMO_FIDUCIAL
# box on Cosma8. It can be used as a quick test of new halo property
# code. Takes ~2 minutes to run.
#
# Should be run from the SOAP source directory. E.g.:
#
# cd SOAP
# ./tests/FLAMINGO/run_L1000N1800_DMO.sh
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Which simulation to do
sim="L1000N1800/DMO_FIDUCIAL"

# Snapshot number to do
snapnum=0057

# Halo indices to do: all halos with x<10, y<10, and z<10Mpc in snap 57
halo_indices="188656 187627 38142 14580600 159179 182418 214329 226243 187624 172666 227086 214370 190343 19553 193246 125904 67585 193243 31517 127948 187623 37502 214331"

# Create parameters files
python tests/FLAMINGO/create_parameters_file.py

# Run SOAP on eight cores processing the selected halos. Use 'python3 -m pdb' to start in the debugger.
mpirun -np 8 python3 -u -m mpi4py ./compute_halo_properties.py \
       ./tests/FLAMINGO/test_parameters.yml \
       --halo-indices ${halo_indices} \
       --dmo \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1
