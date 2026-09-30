# Halo finders

SOAP reads the output of a halo finder to decide which particles belong to
each subhalo (when creating the membership files) and where each subhalo is
centred (when computing halo properties). The halo finder is set in the
`HaloFinder` section of the parameter file. Every halo finder needs two keys:

- **type**: Which halo finder produced the catalogue. One of the section
  names below.
- **filename**: Template for the halo catalogue files. What this should point to
  depends on the halo finder, as described below. Use `{snap_nr}` for the snapshot
  number, e.g. `{snap_nr:03d}`.

Some halo finders support extra keys, which are listed in their section. If a
key is set which the chosen halo finder does not support then SOAP prints a
warning and ignores it.

The value of `type` can be used as `{halo_finder}` in the filepath templates in
other sections of the parameter file.

### HBTplus

Catalogues from [HBT-HERONS](https://github.com/SWIFTSIM/HBT-HERONS).
Both the unsorted catalogues which HBT outputs directly, and catalogues
which have been sorted by TrackId into a single file, are supported.

- **filename**:
  - Unsorted catalogues: The path to the `SubSnap` files without the `.N.hdf5`
    suffix, e.g. `"{sim_dir}/{sim_name}/HBT/{snap_nr:03d}/SubSnap_{snap_nr:03d}"`
  - Sorted catalogues: The full path to the sorted file, e.g.
    `"{sim_dir}/{sim_name}/HBT/OrderedSubSnap_{snap_nr:03d}.hdf5"`

  SOAP uses the unsorted catalogues if `{filename}.0.hdf5` exists,
  otherwise it assumes `filename` is a sorted catalogue.
- **fof_filename**: Optional. Template for the FOF catalogue files, e.g.
  `"{sim_dir}/{sim_name}/fof/fof_output_{snap_nr:04d}.hdf5"`.
  If this is set, the centre, mass and size of the host FOF group are
  written to the output for central subhalos.
- **fof_radius_filename**: Optional. Template for FOF catalogue files which
  contain the `Groups/Radii` dataset, which is missing from the original FOF
  catalogues. These files can be created with `misc/calculate_fof_radii.py`.
  Requires `fof_filename` to be set.
- **read_potential_energies**: Optional, defaults to `false`. Whether to read the
  potential energies of the bound particles from the HBT catalogues and write
  them to the membership files.
- **index_by_track_id**: Optional, defaults to `false`. By default the index of
  each subhalo (`GroupNr_bound` in the membership files, and
  `InputHalos/HaloCatalogueIndex` in the SOAP catalogue) is its position in the
  HBT catalogue. If this is `true` then the TrackId of the subhalo is used
  as its index. This only affects unsorted catalogues, since in the sorted
  catalogues the position of each subhalo is already equal to its TrackId. The
  same value must be used when creating the membership files and when running
  SOAP.

  This is useful because creating the membership files requires the IDs of the
  particles in each subhalo. The unsorted catalogues contain these, and sorted
  catalogues can too, but storing them in both duplicates a lot of data. The
  suggested workflow is to create the membership files and run SOAP using the
  unsorted catalogues with `index_by_track_id: true`, and then generate sorted
  catalogues without particle IDs. SOAP can later be rerun using the sorted
  catalogues and the existing membership files, since both use the TrackId
  as the subhalo index.

HBT catalogues contain every subhalo which has been identified in
the simulation, so every TrackId is present at each snapshot.
This includes orphan subhalos,
which have been disrupted and have no bound particles. SOAP only computes
properties for resolved subhalos (those with bound particles), so in general
the SOAP catalogue has fewer entries than the input HBT catalogue.

If the catalogues for the previous and next snapshots exist then
SOAP also computes `SOAP/ProgenitorIndex` and `SOAP/DescendantIndex`.

```
HaloFinder:
  type: HBTplus
  filename: "{sim_dir}/{sim_name}/HBT/{snap_nr:03d}/SubSnap_{snap_nr:03d}"
  fof_filename: "{sim_dir}/{sim_name}/fof/fof_output_{snap_nr:04d}/fof_output_{snap_nr:04d}.{file_nr}.hdf5"
  read_potential_energies: true
```

### VR

Catalogues from [VELOCIraptor](https://github.com/ICRAR/VELOCIraptor-STF).

- **filename**: The path to the catalogue files without the `.properties`,
  `.catalog_groups`, etc. suffixes, e.g.
  `"{sim_dir}/{sim_name}/VR/catalogue_{snap_nr:04d}/vr_catalogue_{snap_nr:04d}"`.
  Both single file and multi-file (`.properties.N`) output are supported.

```
HaloFinder:
  type: VR
  filename: "{sim_dir}/{sim_name}/VR/catalogue_{snap_nr:04d}/vr_catalogue_{snap_nr:04d}"
```

### Subfind

Catalogues from the Gadget-4 version of Subfind.

- **filename**: The path to the Gadget-4 group sorted snapshot files, without the
  `.N.hdf5` suffix, e.g. `"{sim_dir}/{sim_name}/snapdir_{snap_nr:03d}/snapshot_{snap_nr:03d}"`.
  The filename must end with the three digit snapshot number. The group
  catalogues are then expected to be in
  `groups_{snap_nr:03d}/fof_subhalo_tab_{snap_nr:03d}.N.hdf5`, in the same
  directory as `snapdir_{snap_nr:03d}`.

```
HaloFinder:
  type: Subfind
  filename: "{sim_dir}/{sim_name}/snapdir_{snap_nr:03d}/snapshot_{snap_nr:03d}"
```

### SubfindEagle

Subfind catalogues from the original EAGLE simulations. The `group_membership.py`
script does not support these catalogues. Instead, `misc/convert_eagle.py`
converts the EAGLE snapshots to SWIFT format and creates the membership
files at the same time. See the documentation at the top of that script.

- **filename**: The path to the `subfind_tab` files without the `.N.hdf5` suffix, e.g.
  `"{sim_dir}/{sim_name}/subfind/groups_{snap_nr:03d}/subfind_tab_{snap_nr:03d}"`

```
HaloFinder:
  type: SubfindEagle
  filename: "{sim_dir}/{sim_name}/subfind/groups_{snap_nr:03d}/subfind_tab_{snap_nr:03d}"
```

### Rockstar

Catalogues from [Rockstar](https://bitbucket.org/gfcstanford/rockstar).

- **filename**: The path to the Rockstar binary files without the `.N.bin`
  suffix, e.g. `"{sim_dir}/{sim_name}/Rockstar/snapshot_{snap_nr:04d}/halos_{snap_nr:04d}"`.
  The filename must end with the four digit snapshot number. The files
  `merger_tree/snapshot_{snap_nr:04d}/parents_{snap_nr:04d}.N.list` must also
  exist, where `merger_tree` is in the same directory as `snapshot_{snap_nr:04d}`.

```
HaloFinder:
  type: Rockstar
  filename: "{sim_dir}/{sim_name}/Rockstar/snapshot_{snap_nr:04d}/halos_{snap_nr:04d}"
```

**Caveat:** Rockstar does not work well with SOAP, as SOAP assumes that each
particle is bound to at most one subhalo. Rockstar assigns mass inclusively,
meaning a particle can be bound to a subhalo and to its parent at the same time.
Its output files also don't record which particles are bound to each halo, only
the particles associated with it. As a result, properties which depend on bound
membership are not reliable for Rockstar, whereas properties which use all
particles (e.g. spherical overdensities) are unaffected. See
[Forouhar Moreno et al. 2025](https://arxiv.org/abs/2502.06932), §3.4 and §3.6.

### Adding a new halo finder

To add support for a new halo finder:

- Write a reader in `SOAP/catalogue_readers/`.
- Call it from `SOAP/group_membership.py` and `SOAP/core/halo_centres.py`.
- Add the halo finder, along with any extra keys it supports, to
  `_HALO_FINDER_KEYS` in `SOAP/core/parameter_file.py`.
- Add a section to this file.
