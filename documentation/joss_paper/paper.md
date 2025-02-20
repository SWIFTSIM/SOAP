---
title: 'SOAP: A Python package for calculating the properties of galaxies and halos formed in cosmological simulations'
tags:
  - Python
  - astronomy
authors:
  - name: Robert McGibbon
    affiliation: 1
  - name: John C. Helly
    affiliation: 2
  - name: Joop Schaye 
    affiliation: 1
  - name: Matthieu Schaller
    affiliation: [ 1, 2 ]
  - name: Bert Vandenbroucke
    affiliation: 1
affiliations:
 - index: 1
   name: Leiden Observatory, Leiden University, PO Box 9513, 2300 RA Leiden, the Netherlands
 - index: 2
   name: Institute for Computational Cosmology, Department of Physics, University of Durham, South Road, Durham, DH1 3LE, UK
 - index: 3
   name: Lorentz Institute for Theoretical Physics, Leiden University, PO box 9506, 2300 RA Leiden, the Netherlands
date: 24 January 2025
bibliography: paper.bib

---

# Summary

Cosmological simulations model the evolution of dark matter and baryons under 
the influence of gravitational and hydrodynamic forces. Beginning at 
high redshift, they capture the hierarchical formation of structures, where smaller 
structures form first and later merge into larger ones. These simulations incorporate
hydrodynamics to evolve the gas, and contain a number of subgrid prescriptions
for modelling other important physical processes such as star formation. Once a 
simulation has concluded, a halo-finding algorithm is used to identify bound 
structures (subhalos) within the resulting particle distribution.

Here we introduce SOAP (Spherical Overdensity & Aperture Processor), 
a Python package designed to compute halo and galaxy properties from simulations that
have been post-processed with a halo finder. SOAP takes a subhalo catalogue as 
input and calculates a wide array of properties for each object. Its output 
is compatible with the swiftsimio package [@swiftsimio], enabling seamless unit
handling. SOAP has already been employed to generate halo catalogues for the 
FLAMINGO simulation suite [@flamingo; @flamingoCalibration], which includes the largest cosmological 
hydrodynamical simulation run to date, and these have been used in more than
20 publications so far[^1].

SOAP is hosted on GitHub at https://github.com/SWIFTSIM/SOAP. We strongly encourage
contributions to SOAP by opening issues and submitting pull requests.

# Statement of need

Modern galaxy simulations are often analysed by a large number of people. However,
due to the enormous volume of data generated, it is often impractical for individuals
to compute the specific properties they need by themselves. SOAP addresses this challenge
by producing comprehensive catalogues with a wide range of properties that can be
shared across the community. Given the substantial data volume, it is essential
for the output to be processed in parallel. SOAP achieves this through the mpi4py library
[@mpi4py], enabling it to scale efficiently across multiple compute nodes.
Additionally, SOAP is designed to handle subvolumes of the simulation independently,
allowing large simulations to be processed sequentially if required. This approach reduces the need
for high-memory compute nodes, making it feasible to process
simulation outputs without a large number of high-memory resources.

A large number of halo finders are used by the community
to identify bound structures within simulation outputs. These employ a range of different methods, 
which can result in subhalo catalogues with significant differences [@hbt-herons].
It is therefore important to be able to compare the results of various halo finders to 
help quantify the uncertainty associated with structure finding.
However, along with the different structure identification methods, the halo finder codes often 
vary in their implementation of halo/galaxy property calculations, or may even have different definitions 
for the same property (e.g. using inclusive/exclusive bound mass). This can
lead to further differences in the resulting catalogues, although in this case it is
not due to the halo finding method itself. SOAP can take the input from multiple halo
finders and calculate properties in a consistent manner, therefore allowing the 
actual differences between structure finding algorithms to be pinned down. Currently SOAP
supports HBT-HERONS [@hbt; @hbt-herons], SubFind [@subfind], VELOCIraptor [@velociraptor], and ROCKSTAR
[@rockstar]. Adding a new halo finder requires a script to convert the subhalo catalogue
into a common form used by SOAP, but no other changes to the code are needed.

The most common definition of a halo is based on spherical overdensities, regions of
the universe which have a much larger density than the average. The overdensity of a
region is based on all the particles within it, whether they are bound or unbound, and
is therefore not always output by halo finders. SOAP determines spherical overdensity
radii by constructing expanding spheres until the target density limit has been reached,
and then calculates properties using all particles within the SO radius of each halo.
SOAP also calculates halo properties for a number of other "definitions" of a halo:
subhalo properties (using all particles bound to a subhalo), 
fixed physical projected apertures (using all bound particles within a projected radius), and
two kinds of fixed physical apertures (using all/bound particles within a sphere of the same 
radius for all objects). These various 
types give the user the freedom to pick the definition that is most appropriate for 
their scientific use case (such as the type of observational data they are comparing to).

# Overview of features

- SOAP can currently calculate more than 250 halo/galaxy properties.
Users can easily add new properties, tailoring the tool to their specific scientific needs.
Combined with the four different halo definitions, this makes SOAP exceptionally versatile.
- SOAP works with both dark matter-only (DMO) and full hydrodynamical simulations. For DMO runs any
properties which are irrelevant (e.g. gas mass) are automatically excluded, requiring no changes
to the parameter file.
- When running SOAP it is easy to enable or disable specific halo definitions and properties 
using the SOAP parameter file. This is possible since all properties are defined lazily within the code, 
and so are only computed if required.  Additionally, if certain objects require extra analysis, 
SOAP can be run on a subset of subhalos.
- Properties can be given filters such that they will only be calculated for objects which
meet certain criteria (e.g. only calculate the halo concentration if a subhalo has a
minimum number of bound particles of a particular type). This improves the runtime of SOAP, as well as reducing 
the data volume of the final output catalogues.
- SOAP was originally written to run on Swift [@swift] simulation snapshots, utilising their metadata for 
unit handling and spatial sorting for efficient loading of the data. However, it has also been
used to create halo catalogues from the EAGLE simulation [@eagle] snapshots (which use a modified GADGET [@gadget] format). Supporting
additional snapshot formats will require a conversion script to be written.
- The output is saved as an HDF5 file which is spatially sorted, enabling quick loading of 
simulation subvolumes for analysis without requiring the entire dataset.
- The catalogues can be read with the swiftsimio package [@swiftsimio], which provides 
unit conversion (including handling comoving vs. physical coordinates) and a number of 
visualisation tools. All datasets are output in units that are *h*-free.
- When provided with a parameter file, SOAP can automatically generate a corresponding PDF document with 
a detailed description of all the output properties. This ensures that the documentation of 
the generated catalogues (such as the property names, units, compression level, etc.) 
always reflects the specific setup of the current SOAP run.

# Acknowledgements

We gratefully acknowledge contributions to the code from Joey Braspenning, Jeger Broxterman,
Evgenii Chaikin, Camila Correa, Victor Forouhar Moreno, and Roi Kugel.

# References

[^1]: For a list see https://flamingo.strw.leidenuniv.nl/papers.html
