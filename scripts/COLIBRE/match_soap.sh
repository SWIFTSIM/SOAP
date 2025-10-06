#!/bin/bash -l
#
# Match halos between different colibre runs with the
# same box size and resolution. Pass the snapshots to
# run when calling this script, e.g.
#
# cd SOAP
# mkdir logs
# sbatch -array=92,127%1 ./scripts/COLIBRE/match_colibre.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J match_colibre
#SBATCH -o ./logs/%x_%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 1:00:00
#

set -e

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Where to put the output files
outdir="/snap8/scratch/dp004/${USER}/COLIBRE/matching/"

# Sims to match
sims=(
  "L0025N0188/Thermal L0025N0188/DMO"
)
for sim in "${sims[@]}"; do
    set -- $sim
    sim1=$1
    sim2=$2

  # Location of the input files
  basedir="/cosma8/data/dp004/colibre/Runs"
  snap_basename1="${basedir}/${sim1}/snapshots/colibre_${snapnum}/colibre_${snapnum}"
  snap_basename2="${basedir}/${sim2}/snapshots/colibre_${snapnum}/colibre_${snapnum}"
  membership_basename1="${basedir}/${sim1}/SOAP-HBT/membership_${snapnum}/membership_${snapnum}"
  membership_basename2="${basedir}/${sim2}/SOAP-HBT/membership_${snapnum}/membership_${snapnum}"
  soap_filename1="${basedir}/${sim1}/SOAP-HBT/halo_properties_${snapnum}.hdf5"
  soap_filename2="${basedir}/${sim2}/SOAP-HBT/halo_properties_${snapnum}.hdf5"

  # Matching parameters
  nr_particles=50

  # Name of output file
  mkdir -p ${outdir}
  sim1=$(echo $sim1 | tr '/' '_')
  sim2=$(echo $sim2 | tr '/' '_')
  output_filename="${outdir}/match_${sim1}_${sim2}_${snapnum}.${nr_particles}.hdf5"

  echo
  echo Matching $sim1 to $sim2, snapshot ${snapnum}
  mpirun -- python -u misc/match_group_membership.py \
      --snap-basename1 ${snap_basename1}\
      --snap-basename2 ${snap_basename2}\
      --membership-basename1 ${membership_basename1}\
      --membership-basename2 ${membership_basename2}\
      --catalogue-filename1 ${soap_filename1} \
      --catalogue-filename2 ${soap_filename2} \
      --output-filename ${output_filename}\
      --nr-particles ${nr_particles} \

done

echo "Job complete!"
