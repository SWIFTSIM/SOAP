#!/bin/bash

module purge
module load gnu_comp/13.1.0 hdf5/1.12.2
module load gsl
# module load gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3

git clone https://github.com/ICRAR/VELOCIraptor-STF.git
cd VELOCIraptor-STF
git submodule update --init --recursive
mkdir build
cd build
# cmake ../ -DVR_USE_GAS=ON -DVR_USE_STAR=ON -DV_USE_BH=ON
cmake ../ -DVR_USE_HYDRO=FALSE -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release
make -j 4
cd ../..

export OMP_NUM_THREADS=8
./VELOCIraptor-STF/build/stf -i snap_0018 -o vr_018 -I 2 -C VR_config.cfg

# Pretend we ran with mpi
mv vr_018.properties vr_018.properties.0
mv vr_018.catalog_particles.unbound vr_018.catalog_particles.unbound.0
mv vr_018.catalog_particles vr_018.catalog_particles.0
mv vr_018.catalog_groups vr_018.catalog_groups.0
