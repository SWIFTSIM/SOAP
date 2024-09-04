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
halo_indices="17254 18469 22841 42946 42950 63135 76467 93390 95879 109700 110158 134085 151292 151750 155367 156637 164846 165513 170042 183254 183257 183260 183262 186690 186692 190490 190494 190495 192175 194739 194752 204951 204954 204960 212597 13387915 17588428"

# Create parameters files
python tests/FLAMINGO/create_parameters_file.py tests/FLAMINGO/parameters_HYDRO.yml

# Run SOAP on eight cores processing the selected halos. Use 'python3 -m pdb' to start in the debugger.
mpirun -np 8 python3 -u -m mpi4py ./compute_halo_properties.py \
       ./tests/FLAMINGO/test_parameters.yml \
       --halo-indices ${halo_indices} \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1

