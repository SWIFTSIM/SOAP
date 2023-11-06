# SOAP: Spherical Overdensity and Aperture Processor

This repository contains programs which can be used to compute
properties of halos in spherical apertures in SWIFT snapshots and to
match halos between simulations using the particle IDs.

The code is written in python and uses mpi4py for parallelism.

## Computing halo membership for particles in the snapshot

The first program, vr_group_membership.py, can compute bound and unbound 
VELOCIraptor halo indexes for all particles in a snapshot. The output
consists of the same number of files as the snapshot with particle halo
indexes written out in the same order as the snapshot.

## Computing halo properties

The second program, compute_halo_properties.py, reads the simulation
snapshot and the output from vr_group_membership.py and uses it to
calculate halo properties. It works as follows:

The simulation volume is split into chunks. Each compute node reads
in the particles in one chunk at a time and calculates the properties
of all halos in that chunk.

Within a compute node there is one MPI process per core. The particle
data and halo catalogue for the chunk are stored in shared memory.
Each core claims a halo to process, locates the particles in a sphere
around the halo, and calculates the required properties. When all halos
in the chunk have been done the compute node will move on to the next
chunk.

## Usage on COSMA

### Required modules

The same MPI module which was used to compile mpi4py must be loaded:
```
module load python/3.10.1 gnu_comp/11.1.0 openmpi/4.1.1
```

### Calculating particle group membership

The group membership program needs the location of the snapshot file(s),
the location of the VELOCIraptor catalogue and the name of the output file(s)
to generate. For example:
```
snapnum=0077
swift_filename="./snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
vr_basename="./VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
outfile="./group_membership/vr_membership_${snapnum}.%(file_nr)d.hdf5"

mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename} ${vr_basename} ${outfile}
```

See scripts/FLAMINGO/L1000N1800/group_membership_L1000N1800.sh for an example
batch script.

The code can optionally also write group membership to a single file
virtual snapshot specified with the `--update-virtual-file` flag. This
can be used to create a single file snapshot with group membership
included that can be read with swiftsimio or gadgetviewer.

The `--output-prefix` flag can be used to specify a prefix used to name the
datasets written to the virtual file. This may be useful if writing group
membership from several different VR runs to a single file.

### Calculating halo properties

To calculate halo properties:

```
swift_filename="./snapshots/flamingo_${snapnum}/flamingo_%(snap_nr)04d.%(file_nr)d.hdf5"
extra_filename="./group_membership/vr_membership_%(snap_nr)04d.%(file_nr)d.hdf5"
vr_basename="./VR/catalogue_${snapnum}/vr_catalogue_%(snap_nr)04d"
outfile="./halo_properties/halo_properties_%(snap_nr)04d.hdf5"
nr_chunks=4
snapnum=77

mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
    ${swift_filename} ${vr_basename} ${outfile} ${snapnum} \
    --chunks=${nr_chunks} \
    --extra-input=${extra_filename} \
    --calculations so_masses centre_of_mass \
    --max-ranks-reading=16
```

Here, `--chunks` determines how many chunks the simulation box is
split into. Ideally it should be set such that one chunk fills a compute node.
The `--calculations` flag specifies which calculations should be carried out.
The possible calculation names are defined in halo_properties.py.

The `--max-ranks-reading` flag determines how many MPI ranks per node read the
snapshot. This can be used to avoid overloading the file system. The default
value is 32.

### Batch scripts for running on FLAMINGO simulations on Cosma-8

There are slurm scripts to run on FLAMINGO in `./scripts/FLAMINGO/`. These
are intended to be run as array jobs where the job array indexes determine
which snapshots to process.

In order to reduce duplication only one script is provided per simulation
box size and resolution. The simulation to process is specified by setting
the job name with the slurm sbatch -J flag.

Output locations are specified using the environment variables
FLAMINGO_SCRATCH_DIR and FLAMINGO_OUTPUT_DIR. To write scratch files
on /snap8 and compressed output to /cosma8:
```
export FLAMINGO_OUTPUT_DIR=/cosma8/data/dp004/${USER}/FLAMINGO/ScienceRuns/
export FLAMINGO_SCRATCH_DIR=/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/
```
Then to run the group membership code on all snapshots of the 
L1000N1800/HYDRO_FIDUCIAL simulation:
```
cd SOAP
mkdir logs
sbatch --array=0-77%4 -J HYDRO_FIDUCIAL ./scripts/FLAMINGO/L1000N1800/group_membership_L1000N1800.sh
```

And to run the halo properties code:
```
cd SOAP
mkdir logs
sbatch --array=0-77%4 -J HYDRO_FIDUCIAL ./scripts/FLAMINGO/L1000N1800/halo_properties_L1000N1800.sh
```

### Script for submitting multiple batch jobs

There is also a script which can run the group membership code and the
halo properties code and them compress the output from both. To use it
on Cosma-8, starting from a fresh SOAP checkout:
```
git clone git@github.com:SWIFTSIM/SOAP.git
cd SOAP/scripts/FLAMINGO
./submit_jobs.sh --run=L1000N1800/HYDRO_FIDUCIAL --snapshots=0-77%4
```
This will submit jobs with dependencies set so that they're run in the
right order (e.g. the group membership files must be created before
SOAP can run).

Snapshots are specified using the syntax of the sbatch `--array` flag.
To do one snapshot you could do `--snapshots=77` or to do a range and
limit how many run at once you could use  `--snapshots=0-77%4`.

The output locations are set using environment variables. The defaults are
```
export FLAMINGO_OUTPUT_DIR=/cosma8/data/dp004/${USER}/FLAMINGO/SOAP-Output/
export FLAMINGO_SCRATCH_DIR=/snap8/scratch/dp004/${USER}/FLAMINGO/SOAP-Output/
```
Uncompressed output is all written to the scratch directory and the
final, compressed files are written to the output directory.

By default the script runs the group membership program, runs SOAP,
compresses the group membership output and compresses the SOAP
output. If some parts of the calculation have already been done you
can specify which parts to run with the following flags:
```
--membership: run the group membership calculation
--soap: run SOAP
--compress-membership: compress the group membership output
--compress-soap: compress the SOAP output
```

## Adding quantities

The property calculations are defined in these files:

  * Properties of particles in halos `subhalo_properties.py`
  * Properties of particles in spherical apertures `aperture_properties.py`
  * Properties of particles in projected apertures `projected_aperture_properties.py`
  * Properties of particles in spheres of a specified overdensity `SO_properties.py`

Adding new quantities to already defined SOAP apertures is a relatively easy business. There are four steps.

  * Start by adding an entry to the property table (https://github.com/SWIFTSIM/SOAP/blob/master/property_table.py). Here we store all the properties of the quantities (name, type, unit etc.) All entries in this table are checked with unit tests and added to the documentation. Adding your quantity here will make sure the code and the documentation are in line with each other.
  * Next you have to add the quantity to the type of aperture you want it to be calculated for (aperture_properties.py, SO_properties.py, subhalo_properties.py or projected_aperture_properties.py). In all these files there is a class named `property_list` which defines the subset of all properties that are calculated for this specific aperture.
  * To calculate your quantity you have to define a `@lazy_property` with the same name in the `XXParticleData` class in the same file. There should be a lot of examples of different quantities that are already calculated. An important thing to note is that fields that are used for multiple calculations should have their own `@lazy_property` to avoid loading things multiple times, so check if the things that you need are already there.
  * At this point everything should now work. To test the newly added quantities you can run a unit test using `python3 -W error -m pytest NAME_OF_FILE`. This checks whether the code crashes, and whether there are problems with units and overflows. This should make sure that SOAP never crashes while calculating the new properties.

If SOAP does crash while evaluating your new property it will try to
output the ID of the halo it was processing when it crashed. Then you
can re-run that halo on a single MPI rank in the python debugger as
described in the debugging section below.

## Units

All particle data are stored in unyt arrays internally. On opening the snapshot
a unyt UnitSystem is defined which corresponds to the simulation units. When
particles are read in unyt arrays are created with units based on the
attributes in the snapshot. These units are propagated through the halo
property calculations and used to write the unit attributes in the output.

Comoving quantities are handled by defining a dimensionless unit corresponding
to the expansion factor a.

## Debugging

For debugging it might be helpful to run on one MPI rank in the python debugger
and reduce the run time by limiting the number of halo to process with the
`--max-halos` flag:
```
python3 -Werror -m pdb ./compute_halo_properties.py --max-halos=10 ...
```
This works with OpenMPI at least, which will run single rank jobs without using
mpirun.

The `-Werror` flag is useful for making pdb stop on warnings. E.g. division by
zero in the halo property calculations will be caught.

It is also possible to select individual halos to process with the `--halo-ids`
flag. This specifies the VELOCIraptor IDs of the required halos. E.g.
```
python3 -Werror -m pdb ./compute_halo_properties.py --halo-ids 1 2 3 ...
```

See the scripts in `scripts/FLAMINGO/small_test` for examples showing how to
run SOAP on a few halos from the FLAMINGO simulations.

## Profiling

The code can be profiled by running with the `--profile` flag, which uses the
python cProfile module. Use `--profile=1` to profile MPI rank zero only or
`--profile=2` to generate profiles for all ranks. This will generate files
profile.N.txt with a text summary and profile.N.dat with data which can be
loaded into profiling tools.

The profile can be visualized with snakeviz, for example. Usage on Cosma with
x2go or VNC:
```
pip install snakeviz --user
snakeviz -b "firefox -no-remote %s" ./profile.0.dat
```

## Matching halos between VR outputs

Note that this requires the latest version of https://github.com/jchelly/VirgoDC 

This repository also contains a program to find halos which contain the same
particle IDs between two outputs. It can be used to find the same halos between
different snapshots or between hydro and dark matter only simulations.

For each halo in the first output we find the N most bound particle IDs and
determine which halo in the second output contains the largest number of these
IDs. This matching process is then repeated in the opposite direction and we
check for cases were we have consistent matches in both directions.

### Running the program

It can be run as follows:
```
vr_basename1="./vr/catalogue_0012/vr_catalogue_0012"
vr_basename2="./vr/catalogue_0013/vr_catalogue_0013"

outfile="halo_matching_0012_to_0013.hdf5"
nr_particles=10

mpirun python3 -u -m mpi4py \
    ./match_vr_halos.py ${vr_basename1} ${vr_basename2} \
    ${nr_particles} ${outfile} --use-types 0 1 2 3 4 5
```
Here `nr_particles` is the number of most bound particles to use for matching.

### Matching using only specified particle types

The `--use-types` flag specifies which particle types to use for matching using
the type numbering scheme from Swift. Only the specified types are included in
the most bound particles used to match halos between snapshots. For example,
`--use-types 1` will cause the code to track the `nr_particles` most bound dark
matter particles from each halo.

### Matching to field halos only

The `--to-field-halos-only` flag can be used to match field halos (those with
hostHaloID=-1 in the VR output) between outputs. If it is set we follow the
first `nr_particles` most bound particles from each halo as usual, but when
locating them in the other output any particles in halos with hostHaloID>=0
are treated as belonging to the host halo.

In this mode field halos in one catalogue will only ever be matched to field
halos in the other catalogue. 

Output is still generated for non-field halos. These halos will be matched to
the field halo which contains the largest number of their `nr_particles` most
bound particles. These matches will never be consistent in both directions
because matches to non-field halos are not possible.

### Output

The output is a HDF5 file with the following datasets:

  * `BoundParticleNr1` - number of bound particles in each halo in the first catalogue
  * `MatchIndex1to2` - for each halo in the first catalogue, index of the matching halo in the second
  * `MatchCount1to2` - how many of the most bound particles from the halo in the first catalogue are in the matched halo in the second
  * `Consistent1to2` - whether the match from first to second catalogue is consistent with second to first (1) or not (0)

There are corresponding datasets with `1` and `2` reversed with information about matching in the opposite direction.

## Documentation

Most of the files containing halo property calculations have been extensively documented
using docstrings. To generate documentation, you can for example use
```
python3 -m pydoc aperture_properties
```
This will display the documentation for the file `aperture_properties.py`.
```
python3 -m pydoc -b
```
will display an interactive web page in your browser with a lot of documentation, including all documented
files and functionality of SOAP.

Please have a look at this documentation before making any significant changes!
