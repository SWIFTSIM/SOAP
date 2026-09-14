#!/bin/env python


class SearchRadiusTooSmallError(Exception):
    pass


class HaloProperty:
    def __init__(self, cellgrid):

        # Store parameters needed for halo property calculations
        self.unit_registry = (
            cellgrid.snap_unit_registry
        )  # unyt registry with snapshot units
        self.critical_density = (
            cellgrid.critical_density
        )  # critical density as unyt_quantity
        self.mean_density = cellgrid.mean_density  # mean density as unyt_quantity
        self.a = cellgrid.a  # expansion factor of this snapshot
        self.a_unit = (
            cellgrid.a_unit
        )  # Dimensionless unit used to define comoving quantities
        self.z = cellgrid.z  # redshift of this snapshot
        self.boxsize = cellgrid.boxsize  # boxsize as unyt_quantity
        self.softening_of_parttype = {
            "PartType0": cellgrid.baryon_softening,
            "PartType1": cellgrid.dark_matter_softening,
            "PartType4": cellgrid.baryon_softening,
            "PartType5": cellgrid.baryon_softening,
            "PartType6": cellgrid.nu_softening,
        }  # Softening length of each particle type

        # No density criterion by default
        self.mean_density_multiple = None
        self.critical_density_multiple = None

    # Whether the particles this calculation uses are all of them (True) or only
    # those bound to the halo (False). Calculations which share a
    # SharedHaloParticleData object must agree on this. None means the
    # calculation does not use one.
    shared_inclusive = None

    def shared_key(self, data):
        """
        Return the key identifying the SharedHaloParticleData object this
        calculation would use for the given particle data, or None if it does
        not use one.

        The particle types are part of the key because they determine the order
        in which the arrays are concatenated. Neutrinos are excluded because
        they are never part of those arrays, which is what allows the SO
        calculations to share an object with the inclusive apertures.

        Parameters:
         - data: Dict
           Dictionary containing particle data.
        """
        if self.shared_inclusive is None:
            return None
        types_present = tuple(
            ptype
            for ptype in self.particle_properties
            if ptype in data and ptype != "PartType6"
        )
        return ("SharedHaloParticleData", self.shared_inclusive, types_present)

    def get_shared_particle_data(self, input_halo, data, cache):
        """
        Return the SharedHaloParticleData object this calculation should use,
        taking it from the cache if another calculation has already built the
        same one.

        The object is built from the key, so which calculation happens to
        create it cannot change what it contains. That matters because several
        calculations share a key: the bound subhalo, the exclusive apertures
        and the projected apertures all use one object, and the inclusive
        apertures and the SO calculations another.

        Parameters:
         - input_halo: Dict
           Dictionary containing properties of the halo read from the halo
           catalogue.
         - data: Dict
           Dictionary containing particle data.
         - cache: ParticleDataCache or None
           Cache shared with the other calculations for this halo. If None, the
           object is built for this calculation's own use.
        """
        # imported here rather than at module scope because
        # SnapshotDatasets imports the property table, which imports this module
        from SOAP.particle_selection.shared_halo_particle_data import (
            SharedHaloParticleData,
        )

        key = self.shared_key(data)

        def build():
            _, inclusive, types_present = key
            return SharedHaloParticleData(
                input_halo,
                data,
                list(types_present),
                inclusive,
                self.snapshot_datasets,
                self.softening_of_parttype,
            )

        if cache is None:
            return build()
        return cache.get(key, build)

    def expected_dataset_names(self):
        """
        Return the set of HDF5 dataset names that this calculation will add
        to halo_result for the current parameter file and run configuration.
        """
        names = set()
        for prop in self.property_list.values():
            # Skip properties disabled in the parameter file
            if not self.property_filters[prop.name]:
                continue
            # Skip non-DMO properties for DMO runs
            if self.category_filter.dmo and not prop.dmo_property:
                continue
            names.add(f"{self.group_name}/{prop.name}")
            if getattr(self, "record_timings", False):
                names.add(f"{self.group_name}/{prop.name}_time")
        return names
