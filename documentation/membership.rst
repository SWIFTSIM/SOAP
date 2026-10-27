Group membership files
======================

Before SOAP can be run we generate a set of files which contain halo
membership information for each particle in the SWIFT snapshot. The
datasets in these files are stored in the same order and with the same
partitioning between files as the datasets in the snapshots. This
allows SOAP to read halo membership information for sub-regions of the
simulation volume without reading the full halo-finder output. These files may
also be useful for visualising the input halo catalogue.

The group membership files are HDF5 files with one group for each
particle type, named ``PartType0``, ``PartType1``, ... as in the
snapshots. The following datasets can be contained in the membership files:

* ``GroupNr_bound``: for each particle in the corresponding snapshot
  file this contains the index in the original halo finder catalogue of the object
  which the particle is bound to. Note this is not the same as the index of the object
  in the SOAP catalogue, it instead corresponds to the value of ``HaloCatalogueIndex``
  stored in the SOAP catalogues. If a particle is not bound to any subhalo it
  will have ``GroupNr_bound=-1``.
* ``Rank_bound``: the ranking by total energy of this particle within
  the subhalo it belongs to, or -1 if the particle is not bound to
  any subhalo. The particle with the most negative total energy has
  ``Rank_bound=0``.
* ``FOFGroupIDs``: the 3D FOF group the particle is part of.
  This field is only present if a FOF snapshot is listed in the
  parameter file. This field is present in the snapshots themselves,
  but for FLAMINGO hydro simulations the FOF was regenerated. If this
  field is present it will overwrite the value from the snapshots when
  SOAP is run.
* ``SpecificPotentialEnergies``: the specific potential energy of
  each particle as calculated by the subhalo finder. Note that for HBT
  these values may not have the same order as ``Rank_bound``. This is because
  HBT has a centre refinement step which can reorder the particles in the centre
  of the subhalo, but binding energies are not updated as part of this step.

The script ``make_virtual_snapshot.py`` will combine snapshot and group membership files
into a single virtual snapshot file. This virtual file can be read by swiftsimio and
gadgetviewer to provide halo membership information alongside other particle properties.
Using the virtual file along with the spatial masking functionality within swiftsimio
means it is possible to quickly load all the particles bound to a given subhalo.
The `SWIFTGalaxy package <https://swiftgalaxy.readthedocs.io/en/latest/index.html>`__
makes use of the membership files, and is helpful for analysing individual galaxies when
manipulating their particles (recentering, rotations, cylindrical coordinates, etc.).
