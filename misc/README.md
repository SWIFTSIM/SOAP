## Retrieving the evolution of subhaloes for selected properties

This repository contains a script (`get_evolution_HBT_tracks.py`) used to generate 
a HDF5 file that contains the property evolution for a subset of selected subhalos.
Using the script is recommended when looking at the evolution of more than ~1000
subhaloes at a time, as above this limit h5py fancy indexing is much slower than 
loading the whole catalogue directly.

The script works by opening in parallel each of the SOAP catalogue files available in
the provided  base directory, finding the target `TrackIDs` and getting the values of each
property of interest. After opening every available SOAP catalogue, the collected
data is saved to an HDF5 file.

### Running

The MPI routines used to load the SOAP catalogues in parallel use the [VirgoDC](https://github.com/jchelly/VirgoDC) 
package. If using COSMA, you can run `./scripts/cosma_python_env.sh` from the SOAP basefolder
to install the required packages in a virtual enviroment.

The following options need to be provided when running the scripts:

  - `SOAP_basedir`: Path to the directory containing the SOAP catalogues for the run, which are assumed to be named as `SOAP_basedir/halo_properties_*.hdf5`.
  - `output_file`: Name of the HDF5 file where the property evolution will be saved for each subhalo.
  - `-tracks` or `-t`: Path to a text file containing the `TrackId` of the subhaloes of interest. Each row should only contain a single `TrackId` value.
  - `-properties` or `-p`: Path to a text file containing the SOAP properties to save. Each row should only specify a single property, which should be the absolute path to the dataset within the SOAP catalogue (e.g. `ExclusiveSphere/50kpc/StellarMass`).

To run in MPI (recommended), you can do:

```bash
mpirun -np <NUMBER_MPI_RANKS> -- python3 -m mpi4py get_evolution_HBT_tracks.py <SOAP_basedir> <output_file> -tracks <PATH_TO_TRACKS.TXT> -properties <PATH_TO_PROPERTIES.TXT>
```
### Output

The HDF5 file saved by the script has a similar structure to SOAP catalogues,
but with a few important differences. Here are the main highlights:

  - There is no unit information. Additionally, quantities can be in comoving or physical units depending on what SOAP decided to use and save. Triple check what SOAP does.
  - The name of each property dataset matches the name in the original SOAP catalogues.
  - Each row in a given property dataset corresponds to a different TrackId, with the column being the value of the corresponding property at different output times.
  - An additional dataset indicating the output time of SOAP catalogues (`Redshift`) is always provided.

## Matching halos between outputs

This repository also contains a script to find halos which contain the same
particle IDs between two outputs. It can be used to find the same halos between
different snapshots or between hydro and dark matter only simulations.

For each halo in the first output we find the N most bound particle IDs and
determine which halo in the second output contains the largest number of these
IDs. This matching process is then repeated in the opposite direction and we
check for cases were we have consistent matches in both directions.

### Matching to field halos only

The `--to-field-halos-only` flag can be used to match central halos
between outputs. If it is set we follow the
first `nr_particles` most bound particles from each halo as usual, but when
locating them in the other output any particles in satellite subhalos
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


