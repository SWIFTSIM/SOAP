module purge
module load localhosts
module load HDF5
module load Python/3.11.5-GCCcore-13.2.0

if [ -d "venv" ]; then
    source venv/bin/activate
else
    python -m venv venv
    source venv/bin/activate

    python -m pip cache purge
    pip install mpi4py
    export CC=mpicc
    export HDF5_MPI="ON"
    pip install --no-binary=h5py h5py

    pip install -r requirements.txt

    pip install -e .
fi

