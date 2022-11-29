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

See scripts/FLAMINGO/L1000N1800/halo_properties_L1000N1800.sh for an example
batch script.

## Adding quantities

Property calculations are defined as classes in halo_properties.py. See
halo_properties.SOMasses for an example. Each class should have the following
attributes:

  * particle_properties - specifies which particle properties should be read in. This is a dict with one entry per particle type named "PartType0", "PartType1" etc. Each entry is a list of the names of the properties needed for that particle type.
  * mean_density_multiple - specifies that particles must be read in a sphere of mean density no greater than this multiple of the mean density
  * critical_density_multiple - specifies that particles must be read in a sphere of mean density no greater than this multiple of the critical density
  * physical_radius_mpc - minimum physical radius to read in, in Mpc
  * name - a string which is used to select this calculation with the --calculations command line flag

There should also be a `calculate` method which implements the calculation
and updates the halo_results dict with the calculated properties. The returned
values must be unyt_arrays or unyt_quantities.

New classes must be added to halo_prop_list in compute_halo_properties.py.

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
