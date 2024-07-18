#!/bin/bash
#
# Set up a virtual env suitable for running mpi4py code using openmpi and
# parallel HDF5.
#

set -e

hdf5_version=1.12.3

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/${hdf5_version}
python -m pip cache purge

# Name of the new venv to create
venv_name="openmpi-5.0.3-hdf5-${hdf5_version}-env"

# Create an empty venv
python -m venv "${venv_name}"

# Activate the venv
source "${venv_name}"/bin/activate

# Install mpi4py and h5py using precompiled wheels
wheel_dir="/cosma/local/python-wheels/3.12.4/openmpi-5.0.3-hdf5-${hdf5_version}/"
pip install "${wheel_dir}mpi4py-3.1.6-cp312-cp312-linux_x86_64.whl"
pip install "${wheel_dir}h5py-3.11.0-cp312-cp312-linux_x86_64.whl"

# Add symlink to the right mpirun in the venv's bin directory
mpirun=`which mpirun`
ln -s "${mpirun}" "${venv_name}"/bin/mpirun

# Install other modules
pip install -r ../requirements.txt
git clone https://github.com/jchelly/VirgoDC.git "${venv_name}/VirgoDC"
pip install "${venv_name}/VirgoDC/python"
