# Creates a virtual environment for running SOAP on cosma
# To use when running jobs replace the "module load ..." lines 
# in the sbatch scripts with "source scripts/python_env.sh"
# To delete/reinstall the environment delete the venv with $ rm -rf venv
# Environment can be deactivated using the command $ deactivate

module purge
module load gnu_comp/13.1.0 openmpi/4.1.4 parallel_hdf5/1.12.2 python/3.10.1

if [ ! -d "venv" ]; then
    echo "Creating venv"
    python3 -m venv "venv"
    venv/bin/pip install --upgrade pip
    echo "Installing h5py"
    export CC="mpicc"
    export HDF5_MPI="ON"
    export HDF5_DIR=${HDF5_HOME}
    venv/bin/pip install --no-binary h5py h5py==3.10
    echo "Installing VirgoDC"
    # https for people without ssh setup
    git clone https://github.com/jchelly/VirgoDC.git venv/VirgoDC
    venv/bin/pip install venv/VirgoDC/python
    echo "Installing other modules"
    venv/bin/pip install -r requirements.txt
fi 

source "venv/bin/activate"
