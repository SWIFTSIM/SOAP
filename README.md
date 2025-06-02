# SOAP: Spherical Overdensity and Aperture Processor

This repository contains programs which can be used to compute
properties of halos in spherical apertures in SWIFT[https://swift.strw.leidenuniv.nl/] snapshots.
The resulting output halo catalogues can be read using the
[swiftsimio](https://swiftsimio.readthedocs.io/en/latest/)
python package.

## Installation

The code is written in python and uses mpi4py for parallelism.
IO is carried out in parallel, and so [parallel h5py](https://docs.h5py.org/en/stable/mpi.html) is required. SOAP and it's dependencies can also be 
installed directly using the command
`pip install git+https://github.com/SWIFTSIM/SOAP.git@soap_runtime`
but this may install a serial version of h5py. Therefore the following
steps are recommended for install
```
pip install mpi4py
export HDF5_MPI="ON"; export CC=mpicc; pip install --no-binary=h5py h5py
pip install git+https://github.com/SWIFTSIM/SOAP.git@soap_runtime
```

### Installation on COSMA

If you are using the [COSMA system](https://cosma.readthedocs.io/en/latest/),
you can install an SOAP virtual environment by running
`./scripts/cosma_python_env.sh`

## Running SOAP

The command `./tests/run_small_volume.sh` will download a small example
simulation, run the group membership and halo properties scripts on it (the
resulting catalogue is placed in the `output` directory), and generate the 
pdf documentation to describe the output file (which is written to
`documentation/SOAP.pdf`).

### Computing halo membership for particles in the snapshot

The first step is to extract the subhalo index for all particles in a
snapshot using an input halo-finder catalogue. The output of this step
consists of the same number of files as the snapshot, with particle halo
indexes written out in the same order as the snapshot.

To run the group membership program you must pass the name of the simulation,
the snapshot number, and a parameter file. For example:

```
snapnum=0077
sim=L1000N0900/DMO_FIDUCIAL
mpirun python soap-group-membership \
    --sim-name=${sim} --snap-nr=${snapnum} parameter_files/FLAMINGO.yml
```

### Computing halo properties

The second program, `SOAP/compute_halo_properties.py`, reads the simulation
snapshot and the output from `SOAP/group_membership.py` and uses it to
calculate halo properties. It works as follows:

The simulation volume is split into chunks. Each compute node reads
in the particles in one chunk at a time and calculates the properties
of all halos in that chunk. Therefore when running SOAP the number of
chunks should always be greater than or equal to the number of nodes.

Within a compute node there is one MPI process per core. The particle
data and halo catalogue for the chunk are stored in shared memory.
Each core claims a halo to process, locates the particles in a sphere
around the halo, and calculates the required properties. When all halos
in the chunk have been done the compute node will move on to the next
chunk.

All particle data is stored in unyt arrays internally. On opening the snapshot
a unyt UnitSystem is defined which corresponds to the simulation units. When
particles are read in unyt arrays are created with units based on the
attributes in the snapshot. These units are propagated through the halo
property calculations and used to write the unit attributes in the output.
Comoving quantities are handled by defining a dimensionless unit corresponding
to the expansion factor a.

To calculate halo properties you must pass the same information as for
group membership, and also specify the number of chunks.
If the run is dark matter only the flag `--dmo` should
be passed. For example:

```
snapnum=0077
sim=L1000N0900/DMO_FIDUCIAL
mpirun python -u soap-compute-halo-properties \
       --sim-name=${sim} --snap-nr=${snapnum} --chunks=1 --dmo \
       parameter_files/FLAMINGO.yml
```

Here, `--chunks` determines how many chunks the simulation box is
split into. Ideally it should be set such that one chunk fills a compute node.

The optional `--max-ranks-reading` flag determines how many MPI ranks per node
read the snapshot. This can be used to avoid overloading the file system. The
default value is 32.

### Parameter files

To run either of the programs a parameters file must be passed. This
contains information including the input and output directories,
the halo finder to use, which halo definitions to use, and
which properties to calculate for each halo definition. A description
of all possible fields, and a number of example parameter files
can be found in the `parameters_files` directory.

### Compression

Two types of compression are useful for reducting the size of SOAP output.
The first is lossless compression via GZIP, the second is lossy compression.
For the group membership files we only apply lossless compression. However,
each property in the final SOAP catalogue has a lossy compression filter
associated with it, which are set in `SOAP/property_table.py`. The script 
`compression/compress_soap_catalogue.py` will apply both lossy and
lossless compression to SOAP catalogues.

### Documentation

A pdf describing the SOAP output can be generated. First run `SOAP/property_table.py` passing the parameter file used to run SOAP (to get the properties and halo types to include) and a snapshot (to get the units), e.g. `python SOAP/property_table.py parameter_files/COLIBRE_THERMAL.yml /cosma8/data/dp004/colibre/Runs/L0100N0752/Thermal/snapshots/colibre_0127/colibre_0127.hdf5`. This will generate a table containing all the properties which are enabled in the parameter file. To create the pdf run `cd documentation; pdflatex SOAP.tex; pdflatex SOAP.tex`.

### Slurm scripts for running on COSMA

The files in the `scripts` directory are made for running on cosma.
All scripts should be run from the base SOAP directory.

The scripts are intended to be run as array jobs where the job array indexes determine
which snapshots to process. In order to reduce duplication only one script is provided per simulation
box size and resolution. The simulation to process is specified by setting
the job name with the slurm sbatch -J flag.

## Modifying the code

You can install an editable version of SOAP by cloning this repository and running:
```
pip install mpi4py
export HDF5_MPI="ON"; export CC=mpicc; pip install --no-binary=h5py h5py
pip install -e .
```

The property calculations are defined in the following files in the `SOAP/particle_selection` directory:

  * Properties of particles in subhalos `subhalo_properties.py`
  * Properties of particles in spherical apertures `aperture_properties.py`
  * Properties of particles in projected apertures `projected_aperture_properties.py`
  * Properties of particles in spheres of a specified overdensity `SO_properties.py`

Adding new quantities to already defined SOAP apertures is relatively easy. There are five steps.

  * Start by adding an entry to `SOAP/property_table.py`. Here we store all the properties of the quantities (name, type, unit etc.) All entries in this table are checked with unit tests and added to the documentation. Adding your quantity here will make sure the code and the documentation are in line with each other.
  * Next you have to add the quantity to the type of aperture you want it to be calculated for (`aperture_properties.py`, `SO_properties.py`, `subhalo_properties.py`, or `projected_aperture_properties.py`). In all these files there is a class named `property_list` which defines the subset of all properties that are calculated for this specific aperture.
  * To calculate your quantity you have to define a `@lazy_property` with the same name in the `XXParticleData` class in the same file. There should be a lot of examples of different quantities that are already calculated. An important thing to note is that fields that are used for multiple calculations should have their own `@lazy_property` to avoid loading things multiple times, so check if the things that you need are already there.
  * Add the property to the parameter file.
  * At this point everything should now work. To test the newly added quantities you can run a unit test using `pytest -W error -m pytest tests/test_{NAME_OF_FILE}`. This checks whether the code crashes, and whether there are problems with units and overflows. This should make sure that SOAP never crashes while calculating the new properties.

If SOAP does crash while evaluating your new property it will try to
output the ID of the halo it was processing when it crashed. Then you
can re-run that halo on a single MPI rank in the python debugger as
described in the debugging section below.

### Tests

The directory `tests` contains a number of unit tests for SOAP, some of which
require MPI. They require the optional dependency `pytest-mpi`, and can be run with

```
cd tests
pytest -m "not mpi" -W error; mpirun -np 4 pytest -m mpi --with-mpi -W error
```

The scripts in `tests/FLAMINGO` and `tests/COLIBRE` run SOAP on a few
halos from the FLAMINGO/COLIBRE simulations. These tests rely on data
stored on COSMA, and therefore cannot be run from other systems.

### Debugging

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

It is also possible to select individual halos to process with the `--halo-indices`
flag. This specifies the index of the required halos in the halo catalogue. E.g.
```
python3 -Werror -m pdb ./compute_halo_properties.py --halo-indices 1 2 3 ...
```

### Timing

The flag `--record-halo-timings` can be passed to record the total amount of time
spent calculating properties for subhalos of different masses, and can be useful for
identifying what objects/apertures are dominating the SOAP runtime. The flag
`--record-property-timings` can be passed to record the amount of time spent
calculating each property for each subhalo. Note that this will double the size
of the final output catalogue. The timings can be analysed with the script
`misc/plot_times.py`.

### Profiling

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

