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
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which simulation to do
sim="L1000N1800/HYDRO_FIDUCIAL"

# Snapshot number to do
snapnum=0050

# Halo indices to do: all halos with x<10, y<10, and z<10Mpc in snap 50
halo_indices="2208 3360 7167 12861 15349 33465 40199 41557 44559 73863 74544 77349 81088 87230 88604 99175 99725 111048 118709 118710"

# Run SOAP on eight cores processing the selected halos. Use 'python3 -m pdb' to start in the debugger.
mpirun -np 8 python3 -u -m mpi4py ./compute_halo_properties.py \
       ./tests/FLAMINGO/parameters.yml \
       --halo-indices ${halo_indices} \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1

