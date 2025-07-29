---
title: 'SOAP: A Python Package for Calculating the Properties of Galaxies and Halos Formed in Cosmological Simulations'
tags:
  - Python
  - astronomy
authors:
  - name: Robert McGibbon
    affiliation: 1
    orcid: 0000-0003-0651-0776
  - name: John C. Helly
    affiliation: 2
    orcid: 0000-0002-0647-4755
  - name: Joop Schaye
    affiliation: 1
    orcid: 0000-0002-0668-5560
  - name: Matthieu Schaller
    affiliation: "1, 3"
    orcid: 0000-0002-2395-4902
  - name: Bert Vandenbroucke
    affiliation: 1
    orcid: 0000-0001-7241-1704
affiliations:
 - index: 1
   name: Leiden Observatory, Leiden University, PO Box 9513, 2300 RA Leiden, The Netherlands
 - index: 2
   name: Institute for Computational Cosmology, Department of Physics, University of Durham, South Road, Durham, DH1 3LE, UK
 - index: 3
   name: Lorentz Institute for Theoretical Physics, Leiden University, PO box 9506, 2300 RA Leiden, The Netherlands
date: 24 January 2025
bibliography: paper.bib

---

# Summary

Cosmological simulations model the evolution of dark matter and baryons under
the influence of gravitational and hydrodynamic forces. Beginning at
high redshift, they capture the hierarchical formation of structures, where smaller
structures form first and later merge into larger ones. These simulations incorporate
hydrodynamics to evolve the gas and include a number of subgrid prescriptions
for modelling important physical processes, such as star formation. Once the simulation has concluded, a halo-finding algorithm is used to identify bound
structures (subhalos) within the resulting particle distribution.

Here, we introduce `SOAP` (Spherical Overdensity & Aperture Processor), a Python package designed to compute halo and galaxy properties from simulations that
have been post-processed with a halo finder. `SOAP` takes a subhalo catalogue as
input and calculates a wide array of properties for each object. Its output
is compatible with the swiftsimio package [@swiftsimio], enabling seamless unit
handling. `SOAP` has already been used to generate halo catalogues for the
FLAMINGO simulation suite [@flamingo; @flamingoCalibration], which includes the largest cosmological
hydrodynamic simulation to date. These catalogues have been used in more than
20 publications to date.[^1]

`SOAP` is hosted on GitHub. We strongly encourage
contributions to `SOAP`, such as opening issues and submitting pull requests.

# Statement of Need

Modern galaxy simulations are often analyzed by a large number of researchers. However,
due to the large volume of data generated, it is often impractical for individual users
to compute the specific properties they require independently. `SOAP` addresses this challenge
by producing comprehensive catalogues containing a wide range of properties that can be
shared across the community.

Given the substantial volume of data, it is essential for the output to be processed in parallel.
`SOAP` achieves this using the mpi4py library
[@mpi4py]. This enables `SOAP` to scale efficiently across multiple compute nodes.
`SOAP` is also designed to handle subvolumes of the simulation independently, allowing
for large simulations to be processed sequentially if required. This approach reduces the need
for high-memory resources.
The ability to efficiently process subhalos in parallel is a unique feature of `SOAP` when compared
with other packages for computing galaxy properties [e.g. @cosmic-profiles; @tangos; @caesar].

A large number of halo finders are used by the community
to identify bound structures within simulation outputs. These employ a variety of methods
which can result in subhalo catalogues with significant differences [@hbt-herons].
Therefore, it is important to be able to compare the results of various halo finders to
help quantify the uncertainty associated with structure finding.
However, along with the different structure identification methods, the halo finder codes often
vary in their implementation of halo/galaxy property calculations and may even have different definitions
(e.g. using inclusive/exclusive bound mass) for the same property. This can
lead to further differences in the resulting catalogues, although in this case it is
not due to the halo finding method itself. `SOAP` can take input from multiple halo
finders and calculate properties consistently, thereby enabling
actual differences between structure-finding algorithms to be identified. Currently `SOAP`
supports HBT-HERONS [@hbt; @hbt-herons], SubFind [@subfind], VELOCIraptor [@velociraptor], and
ROCKSTAR [@rockstar]. Adding a new halo finder requires a script to convert the subhalo catalogue
into the standard format used by SOAP; no other code changes are necessary.

The most common definition of a halo is based on spherical overdensities (SO): regions of
the universe which have a much larger density than the average. The overdensity of a
region is based on all the particles within it, whether bound or unbound, and
is therefore not always output by halo finders. `SOAP` determines spherical overdensity
radii by constructing expanding spheres until the target density limit is reached.
It then calculates the properties of each halo using all the particles within its SO radius.
`SOAP` also calculates properties for several other definitions of a halo:
subhalo properties (using all particles bound to a subhalo),
fixed physical projected apertures (using all bound particles within a projected radius),
and two types of fixed physical apertures (using all/bound particles within a sphere of the same
radius for all objects). These various
types give users the freedom to select the most appropriate definition for
their scientific use case e.g. the type of observational data they are comparing with.

# Overview of Features

- `SOAP` can currently calculate over 250 halo and galaxy properties.
Users can easily add new properties to tailor the tool to their specific scientific needs.
When combined with the four different halo definitions, this makes `SOAP` exceptionally versatile.
- `SOAP` is compatible with both dark matter-only (DMO) and full hydrodynamic simulations. For DMO runs, any
properties which are irrelevant (e.g. gas mass) are automatically excluded, requiring no changes
to the parameter file.
- `SOAP` makes it easy to enable or disable specific halo definitions and properties
using the `SOAP` parameter file. This is possible because all properties are lazily defined within the code
and are only computed if required. Additionally, if certain objects require further analysis,
`SOAP` can be run on a subset of subhalos.
- Properties can be assigned filters so that they are only calculated for objects that
meet certain criteria (e.g. only calculate the halo concentration if a subhalo has a
minimum number of bound particles of a particular type). This improves the runtime of `SOAP` and also reduces
the data volume of the final output catalogues.
- `SOAP` was originally written to run on Swift simulation snapshots [@swift], utilizing their metadata for
unit handling and spatial sorting to enable efficient loading of the data. However, it has also been
used to create halo catalogues from the EAGLE simulation [@eagle] snapshots [which use a modified GADGET format, @gadget].
Supporting additional snapshot formats requires a conversion script to be written.
- The output is saved as an HDF5 file which is spatially sorted, enabling quick loading of
simulation subvolumes for analysis without requiring the entire dataset.
- The catalogues can be read with the swiftsimio package [@swiftsimio], which provides
unit conversion (including handling comoving versus physical coordinates) and a number of
visualization tools. All datasets are output in units that are *h*-free.
- When provided with a parameter file, `SOAP` can automatically generate a corresponding PDF document with
a detailed description of all the output properties. This ensures that the documentation of
the generated catalogues (e.g., the property names, units, compression level, etc.)
always reflects the specific setup of the current `SOAP` run.

# Acknowledgements

We gratefully acknowledge contributions to the code from Joey Braspenning, Jeger Broxterman,
Evgenii Chaikin, Camila Correa, Victor Forouhar Moreno, and Roi Kugel. `SOAP` relies heavily on
the following packages: mpi4py [@mpi4py], NumPy [@numpy], h5py [@h5py], and unyt [@unyt].

# References

[^1]: For a complete list, see https://flamingo.strw.leidenuniv.nl/papers.html
