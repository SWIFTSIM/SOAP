#!/bin/bash -l
#
# Compute halo properties for a snapshot. Must run the group_membership
# script first.
#
# Job name determines which of the L1000N3600 runs we process.
# Array job index is the snapshot number to do. Submit with (for example):
#
# cd SOAP
# mkdir logs
# ./scripts/cosma_python_env.sh
# sbatch -J HYDRO_FIDUCIAL --array=0-78%4 ./scripts/FLAMINGO/L1000N3600/halo_properties_L1000N3600.sh
#
#SBATCH --nodes=40
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/halo_properties_L1000N3600_%x.%a.%A.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 72:00:00
#

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

set -e

# Which snapshot to do
snapnum=${SLURM_ARRAY_TASK_ID}

# Which simulation to do
sim="L1000N3600/${SLURM_JOB_NAME}"

# Check for DMO run
dmo_flag=""
if [[ $sim == *DMO_* ]] ; then
  dmo_flag="--dmo"
else
  echo "Running recalculate_xrays.py"
  swift_filename="/cosma8/data/dp004/flamingo/Runs/${sim}/snapshots/flamingo_{snap_nr:04}/flamingo_{snap_nr:04}.{file_nr}.hdf5"
  xray_filename="/snap8/scratch/dp004/${USER}/flamingo/Runs/${sim}/xray/xray_{snap_nr:04}/xray_{snap_nr:04}.{file_nr}.hdf5"
  xray_table_path='/cosma8/data/dp004/flamingo/Tables/Xray/X_Ray_table_metals_full.hdf5'
  mpirun -- python misc/recalculate_xrays.py $swift_filename $xray_filename $xray_table_path --snap-nr=$snapnum
fi

mpirun -- python3 -u -m mpi4py SOAP/compute_halo_properties.py \
       --sim-name=${sim} --snap-nr=${snapnum} --reference-snapshot=78 \
       --chunks=80 ${dmo_flag} parameter_files/FLAMINGO.yml

echo "Job complete!"
