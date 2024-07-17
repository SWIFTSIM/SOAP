#!/bin/bash
#
# This runs SOAP on a few halos in the L1000N1800/HYDRO_FIDUCIAL
# box on Cosma8. It can be used as a quick test of new halo property
# code. Takes ~2 minutes to run.
#
# Should be run from the SOAP source directory. E.g.:
#
# cd SOAP
# ./tests/FLAMINGO/run_L1000N1800_HYDRO.sh
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Which simulation to do
sim="L1000N1800/HYDRO_FIDUCIAL"

# Snapshot number to do
snapnum=0057

# Halo indices to do: all halos with x<10, y<10, and z<10Mpc in snap 57
halo_indices="2358 2881 3633 9537 9732 14097 14639 18078 34414 38651 43121 43643 44887 45957 49461 51256 67007 68228 71121 72153 79479 80461 82525 83482 83955 89427 93408 96685 98640 117518 123068"

# Run SOAP on eight cores processing the selected halos. Use 'python3 -m pdb' to start in the debugger.
mpirun -np 8 python3 -u -m mpi4py ./compute_halo_properties.py \
       ./tests/FLAMINGO/parameters.yml \
       --halo-indices ${halo_indices} \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1

