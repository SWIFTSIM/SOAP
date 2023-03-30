#!/bin/bash
#
# This runs SOAP on a few halos in the L1000N1800/HYDRO_FIDUCIAL
# box on Cosma8. It can be used as a quick test of new halo property
# code. Takes ~2 minutes to run.
#
# Should be run from the SOAP source directory. E.g.:
#
# cd SOAP
# ./scripts/FLAMINGO/interactive_test/run_L1000N1800_HYDRO.sh
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which simulation to do
sim="L1000N1800/HYDRO_FIDUCIAL"

# Location of the simulation
basedir="/cosma8/data/dp004/flamingo/Runs/${sim}/"
swift_filename="${basedir}/snapshots/flamingo_%(snap_nr)04d/flamingo_%(snap_nr)04d.%(file_nr)d.hdf5"
extra_filename="${basedir}/SOAP/membership_%(snap_nr)04d/membership_%(snap_nr)04d.%(file_nr)d.hdf5"
vr_basename="${basedir}/VR/catalogue_%(snap_nr)04d/vr_catalogue_%(snap_nr)04d"

# Snapshot number to do
snap_nr=0077

# Halo IDs to do: all halos with x<10, y<10, and z<10Mpc in snap 77
halo_ids="975 1581 3735 6363 32039 34038 35393 35660 35692 53125 54033 58655 59282 59412 61735 64257 72073 74013 78514 91642 91643 91644 91645 91646 91647 91648 93219 93220"

# Location for temporary chunk output
scratchdir="./tmp/"

# Where to write the output
outfile="./output/halo_properties.${snap_nr}.hdf5"

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"

# Run SOAP on one core processing the selected halos. Use 'python3 -m pdb' to start in the debugger.
python3 ./compute_halo_properties.py \
    ${swift_filename} ${scratchdir} ${vr_basename} ${outfile} ${snap_nr} \
    --chunks=1 --extra-input=${extra_filename} --halo-ids ${halo_ids}
