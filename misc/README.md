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


