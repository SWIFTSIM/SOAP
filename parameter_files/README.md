# SOAP parameter files

The parameter files are a YAML dictionary which define the parameters and settings for running SOAP.
This file describes the structure of a parameter file, including all possible fields which can be specified.
This file does not detail what the differences are between the various aperture types, for that
see the main pdf documenation.

### DMO runs

SOAP does not require separate parameter files for DMO & HYDRO runs. Instead you
must pass the `--dmo` flag when running on a DMO simulation, and in that case
any hydro-only properties will be skipped.

### Parameters

The section is used to set variables which are subtituted into the filepath templates in the other sections of the parameter file.
Examples of filepath templates which use the variables from this section are given below.

- **sim_dir**: Base directory where the simulation data is stored.
- **output_dir**: Directory where output files will be written.
- **scratch_dir**: Temporary directory for intermediate files during processing.

Alongside these values you must pass `--sim-name` as an argument when running SOAP, and so `{sim_name}` can be used in filepath templates.
`{halo_finder}` can also be used in filepath templates.

### Snapshots

This section defines the format and location of the SWIFT simulation snapshots. **If there are chunk files you must pass `{file_nr}` in the filepath template**, SOAP does not currently read virtual hdf5 files.

- **filename**: Template for the snapshot files. Use `{snap_nr:04d}` for the snapshot number and `{file_nr}` for the file number, e.g. `"{sim_dir}/{sim_name}/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5"`
- **fof_filename**: Optional. Path to snapshots with FOF (Friends-of-Friends) IDs if FOF has been re-run. This is only used by the `group_membership.py` script as the values are then stored in the membership files.

### Extra input

This section is optional. If you wish to run SOAP with datasets not present in the original snapshots, you can pass them as extra input files.
If a dataset is present in both the snapshot and the extra input files, the values from the extra input will be used. Multiple extra input files can be passed.

- **extra_input**: Template for extra input file paths, e.g. `"{sim_dir}/{sim_name}/recalculated_xray/xray_{snap_nr:04d}/xray_{snap_nr:04d}.{file_nr}.hdf5"`

### Halo Finder

Settings for the halo finding algorithm and output file locations.

- **type**: The subhalo finder being used. Possible options are `HBTplus`, `VR`, `Subfind`, and `Rockstar`.
- **filename**: Template for input halo catalogue files. The format of this depends on the halo finder as they have different output structure. HBTplus example: `"{sim_dir}/{sim_name}/HBT/{snap_nr:03d}/SubSnap_{snap_nr:03d}"`
- **fof_filename**: Template for FOF catalog files. Used for storing host FOF information for central subhalos. Only supported for HBTplus
- **fof_radius_filename**: Template for FOF catalog files which contain the "Groups/Radii" dataset. These were produced by a post-processing script, and are missing from the main FOFs
- **read_potential_energies**: Optional boolean value, defaults to False. Whether to read potential energies and place them in the membership files. Only supported for HBTplus


### Group Membership

Configuration for writing group membership files.

- **filename**: Template for group membership file paths, e.g. `"{output_dir}/{sim_name}/SOAP_uncompressed/membership_{snap_nr:04d}/membership_{snap_nr:04d}.{file_nr}.hdf5"`

### Halo Properties

Settings for writing the output SOAP catalogues properties, and for handling temporary files.

- **filename**: Template for the halo properties file paths, e.g. `"{output_dir}/{sim_name}/SOAP_uncompressed/{halo_finder}/halo_properties_{snap_nr:04d}.hdf5"`
- **chunk_dir**: Directory for temporary chunk output files. e.g. `"{scratch_dir}/{sim_name}/SOAP-tmp/{halo_finder}/"`

### SubhaloProperties

Define which properties to compute for each subhalo.

- **properties**: A list of properties to compute. The key for each property should be either a boolean, a string, or a dictionary. Passing a boolean simply flags whether to enable or disable that calculation (for all subhalos). If the property should only be computed for subhalos that meet a certain criteria then the name of the filter should be passed as a string (see the Filters section below). Alternatively a dictionary can be passed with two entries: `snapshot` and `snipshot`. In this case the behaviour will be determined based on whether the `--snipshot` flag is passed when running SOAP.

An example is as follows
```
SubhaloProperties:
  properties:
    TotalMass: true
    StellarMass: general
    GasMass:
      snapshot: true
      snipshot: false
```

### ApertureProperties

Define which fixed spherical apertures to compute, and what properties to compute within them.

- **properties**: The same format as used for SubhaloProperties.
- **variations**: A list of which apertures to compute. Each aperture must have a name, a boolean key `inclusive` to indicate whether to include unbound particles. The aperture can be specified in two different ways:
  - Passing a value of `radius_in_kpc` to indicate the **physical size** of the aperture. 
  - Passing a string as `property`, which is the name of another property which has been computed by SOAP. In this case each subhalo will use it's own value of that property as the aperture radius. `radius_muliple` can optionally be passed, in which case the aperture radius will be equal to the multiplier times the property value.
Each aperture can optionally be passed the `filter` key. If the filter key is set then the aperture will only be computed for subhalos that fulfill the filter criteria. If no `filter` key is passed then the aperture will be computed for all subhalos. For inclusive apertures the boolean flag `skip_gt_enclose_radius` can be set (defaults to False). If it is set then properties will not be calculated for any subhalos where all bound particles are within the aperture radius.

An example is as follows
```
ApertureProperties:
  properties:
    TotalMass: true
  variations:
    exclusive_50kpc:
      inclusive: false
      radius_in_kpc: 50.0
    inclusive_50kpc:
      inclusive: true
      radius_in_kpc: 50.0
      filter: general
      skip_gt_enclose_radius: false
    exclusive_half_mass:
      inclusive: false
      property: BoundSubhalo/HalfMassRadiusTotal
    exclusive_twice_half_mass:
      inclusive: false
      property: BoundSubhalo/HalfMassRadiusTotal
      radius_multiple: 2.0
```

If you do not wish to calculate any apertures then pass any empty dict to both the properties and the variations, e.g.

```
ApertureProperties:
  properties:
    {}
  variations:
    {}
```

### ProjectedApertureProperties

Define which fixed projected apertures to compute, and what properties to compute within them. The structure is exactly the same as for ApertureProperties, with the exception that the keys `inclusive` and `skip_gt_enclose_radius` can be not set for the variations. This is because ProjectedApertures will always computed using only bound particles.

An example is as follows
```
ProjectedApertureProperties:
  properties:
    TotalMass: true
    GasMass:
      snapshot: true
      snipshot: false
  variations:
    50_kpc:
      radius_in_kpc: 50.0
    100_kpc:
      radius_in_kpc: 100.0
      filter: general
    twice_half_mass:
      property: BoundSubhalo/HalfMassRadiusTotal
      radius_multiple: 2.0
```

### SOProperties

Define which spherical overdensity aperture to compute, and what properties to compute within them.

- **properties**: The same as for SubhaloProperties.
- **variations**: A list of which apertures to compute. Each aperture must have a name, a `type` (options: `crit`, `mean`, `BN98`), and a `value` which indicates what multiple of the crit/mean density to use. As with ApertureProperties, a `filter` can optionally be passed. `core_excision_fraction` can optionally be passed to set the size of the core when calculating core_excised properties (no core_excised properties will be calculated if this is not passed). `radius_multiple` can optionally be passed for calculating an aperture which is a multiple of one of the previous aperture.

An example is as follows
```
SOProperties:
  properties:
    TotalMass: true
    GasMass:
      snapshot: true
      snipshot: false
  variations:
    200_crit:
      type: crit
      value: 200.0
    200_mean:
      type: mean
      value: 200.0
      filter: general
    500_crit:
      type: crit
      value: 500.0
      core_excision_fraction: 0.15
    5xR500_crit:
      type: crit
      value: 500.0
      radius_multiple: 5.0 
      filter: general
    BN98:
      type: BN98
      value: 0.0
      filter: general
```

### Aliases

Optional. Used if field names in the snapshots do not agree with what SOAP expects. If the `snipshot` section is passed then those aliases will be used when running in snipshot mode (the snipshot values will not be combined with the snapshot aliases present. If no `snipshot` section is present then the snapshot aliases will be used when running in snipshot mode). E.g.
```
aliases:
  PartType0/ElementMassFractions: PartType0/SmoothedElementMassFractions
  PartType4/ElementMassFractions: PartType4/SmoothedElementMassFractions
  snipshot:
    PartType0/ElementMassFractions: PartType0/ReducedElementMassFractions
    PartType4/ElementMassFractions: PartType4/SmoothedElementMassFractions

```
For each alias the key is the name of the property that SOAP expects, and the value is the name of the property in the snapshot being passed.

### Filters

SOAP uses filters to determine whether to skip the calculation of an aperture or
property based on the number of bound particles. This section of the parameter file
defines the particle limits for each filter. New filters can be added if required.

```
filters:
  general:
    limit: 100
    properties:
      - BoundSubhalo/NumberOfGasParticles
      - BoundSubhalo/NumberOfDarkMatterParticles
      - BoundSubhalo/NumberOfStarParticles
      - BoundSubhalo/NumberOfBlackHoleParticles
    combine_properties: sum
  baryon:
    limit: 100
    properties:
      - BoundSubhalo/NumberOfGasParticles
      - BoundSubhalo/NumberOfStarParticles
    combine_properties: sum
  dm:
    limit: 100
    properties:
      - BoundSubhalo/NumberOfDarkMatterParticles
  gas:
    limit: 100
    properties:
      - BoundSubhalo/NumberOfGasParticles
  star:
    limit: 100
    properties:
      - BoundSubhalo/NumberOfStarParticles
```

### Defined constants

Optional. Constants used when running SOAP

```
defined_constants:
  O_H_sun: 4.9e-4
  Fe_H_sun: 2.82e-5
```

### Calculations

Contains information about how to run SOAP

- **min_read_radius_cmpc**: Optional. Using the input halo catalogues SOAP makes an initial guess of the radius around each halo to read in. This value can be set so SOAP will read a minimum radius by default, which can be useful if large SOs are being calculated.
- **calculate_missing_properties**: Optional, default True. If set to true then SOAP will calculate any properties which are not listed in the parameter file. If set to false then SOAP will ignore these properties 
- **reduced_snapshots**: Optional. We create reduced snapshots where we keep the particles within the virial radius of certain objects. The values here determine which halos to keep.
  - **min_halo_mass**: The minimumum M200 halo mass to keep
  - **halo_bin_size_dex**: The size of the halo mass bins
  - **halos_per_bin**: The number of objects we want in each halo mass bin.
- **recently_heated_gas_filter**: Optional. How to determine which gas particles count as being recently heated.
  - **delta_time_myr**: For particles to be recently heated they must have experienced an AGN event the past x Myr.
  - **use_AGN_delta_T**: If set to true then for particles to be considered recently heated they must have a temperature close to AGN_delta_T
- **cold_dense_gas_filter**: Optional. How to determine which gas particles count as being cold & dense
  - **maximum_temperature_K**: Value above which gas is not considered to be cold
  - **minimum_hydrogen_number_density_cm3**: Value below which gas gas is not considered to be dense
- **strict_halo_copy**: Optional, default False. When a halo has multiple ExclusiveSphere/ProjectedAperture halo types which encompass all the bound particles then we just copy across the values rather than recomputing them. There are a small number of properties for which this is not correct. If this flag is set then these properties are set to zero for the larger apertures instead of being copied across.
