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
snapnum=`printf '%03d' ${SLURM_ARRAY_TASK_ID}`

# Where to put the output files
outdir="/snap8/scratch/dp004/${USER}/COLIBRE/matching/"

# Sims to match
sims=(
  "L100_m7/THERMAL_AGN_m7 L100_m7/HYBRID_AGN_m7"
  "L100_m7/THERMAL_AGN_m7 L100_m7/DMO"
)
for sim in "${sims[@]}"; do
    set -- $sim
    sim1=$1
    sim2=$2

  # Location of the HBT catalogues
  basedir="/cosma8/data/dp004/colibre/Runs/"
  hbt_basename1="${basedir}/${sim1}/HBTplus/${snapnum}/SubSnap_${snapnum}"
  hbt_basename2="${basedir}/${sim2}/HBTplus/${snapnum}/SubSnap_${snapnum}"
  nr_particles=50

  # Name of output file
  mkdir -p ${outdir}
  sim1=$(echo $sim1 | tr '/' '_')
  sim2=$(echo $sim2 | tr '/' '_')
  outfile="${outdir}/match_${sim1}_${sim2}_${snapnum}.${nr_particles}.hdf5"

  echo
  echo Matching $sim1 to $sim2, snapshot ${snapnum}
  mpirun -- python -u \
      misc/match_hbt_halos.py ${hbt_basename1} ${hbt_basename2} ${nr_particles} ${outfile}

done

echo "Job complete!"
