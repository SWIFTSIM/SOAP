#!/bin/env python

"""
shared_halo_particle_data.py

Particle arrays shared between the aperture, projected aperture, subhalo and
spherical overdensity property calculations of a single halo.

All of those calculations start by concatenating the same per particle
quantities (masses, positions, velocities and particle types) for either the
gravitationally bound particles of the halo (exclusive apertures, projected
apertures and the bound subhalo) or for every particle in the search radius
(inclusive apertures). Doing that once per halo instead of once per
calculation avoids repeating the work for every aperture.

Quantities derived from those arrays are implemented as lazy properties, so a
calculation which does not need one never pays for it: the projected radii are
only computed if a projected aperture asks for them, and the softening and the
(3D) radius only if an aperture or the bound subhalo does.
"""

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
import unyt

from SOAP.core.dataset_names import mass_dataset
from SOAP.core.lazy_properties import lazy_property
from SOAP.core.snapshot_datasets import SnapshotDatasets


class SharedHaloParticleData:
    """
    Concatenated particle arrays for a single halo.

    One of these objects covers every calculation which sees the same set of
    particles, which is determined by "inclusive": exclusive apertures,
    projected apertures and the bound subhalo all use the gravitationally
    bound particles of the halo, while inclusive apertures use every particle
    that was read in.

    Note that the order of the concatenated arrays is set by types_present, so
    calculations may only share one of these objects if they use the same
    particle types in the same order. That is what the cache key in
    SharedParticleData accounts for.
    """

    def __init__(
        self,
        input_halo: Dict,
        data: Dict,
        types_present: List[str],
        inclusive: bool,
        snapshot_datasets: SnapshotDatasets,
        softening_of_parttype: unyt.unyt_array,
    ):
        """
        Constructor.

        Parameters:
         - input_halo: Dict
           Dictionary containing properties of the halo read from the halo
           catalogue.
         - data: Dict
           Dictionary containing particle data.
         - types_present: List
           List of all particle types (e.g. 'PartType0') that are present in the
           data dictionary.
         - inclusive: bool
           Whether or not to include particles that are not gravitationally
           bound to the subhalo.
         - snapshot_datasets: SnapshotDatasets
           Object containing metadata about the datasets in the snapshot, like
           appropriate aliases and column names.
         - softening_of_parttype: unyt.unyt_array
           Softening length of each particle types
        """
        self.input_halo = input_halo
        self.data = data
        # Neutrinos are not part of the concatenated arrays: they only
        # contribute to the spherical overdensity radius and to neutrino
        # specific properties, and are handled separately below.
        self.types_present = [t for t in types_present if t != "PartType6"]
        self.has_neutrinos = "PartType6" in data
        self.inclusive = inclusive
        self.snapshot_datasets = snapshot_datasets
        self.softening_of_parttype = softening_of_parttype
        self.centre = input_halo["cofp"]
        self.index = input_halo["index"]
        self.in_halo_masks = {}
        self.compute_basics()

    def get_dataset(self, name: str) -> unyt.unyt_array:
        """
        Local wrapper for SnapshotDatasets.get_dataset().
        """
        return self.snapshot_datasets.get_dataset(name, self.data)

    def in_halo_mask(self, ptype: str) -> NDArray[bool]:
        """
        Mask which selects the particles of ptype that are included in the
        calculations: only the particles bound to this halo for exclusive
        calculations, all of them for inclusive ones.

        Parameters:
         - ptype: str
           Particle type, e.g. 'PartType0'.
        """
        if ptype not in self.in_halo_masks:
            groupnr_bound = self.get_dataset(f"{ptype}/GroupNr_bound")
            if self.inclusive:
                mask = np.ones(groupnr_bound.shape, dtype=bool)
            else:
                mask = groupnr_bound == self.index
            self.in_halo_masks[ptype] = mask
        return self.in_halo_masks[ptype]

    def compute_basics(self):
        """
        Concatenate the quantities which every calculation sharing this object
        needs, over all particle types that are present.
        """
        mass = []
        position = []
        velocity = []
        types = []
        self.nr_part_of_type = []
        for ptype in self.types_present:
            in_halo = self.in_halo_mask(ptype)
            mass.append(self.get_dataset(f"{ptype}/{mass_dataset(ptype)}")[in_halo])
            pos = (
                self.get_dataset(f"{ptype}/Coordinates")[in_halo, :]
                - self.centre[None, :]
            )
            position.append(pos)
            velocity.append(self.get_dataset(f"{ptype}/Velocities")[in_halo, :])
            nr_part = pos.shape[0]
            types.append(int(ptype[-1]) * np.ones(nr_part, dtype=np.int32))
            self.nr_part_of_type.append((ptype, nr_part))

        self.mass = np.concatenate(mass)
        self.position = np.concatenate(position)
        self.velocity = np.concatenate(velocity)
        self.types = np.concatenate(types)

    @lazy_property
    def radius(self) -> unyt.unyt_array:
        """
        Distance of each particle from the halo centre.
        """
        pos = self.position
        return np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)

    @lazy_property
    def softening(self) -> unyt.unyt_array:
        """
        Softening length of each particle.
        """
        softening = []
        for ptype, nr_part in self.nr_part_of_type:
            softening.append(
                np.ones(nr_part, dtype=np.float64) * self.softening_of_parttype[ptype]
            )
        return np.concatenate(softening)

    @lazy_property
    def radius_projx(self) -> unyt.unyt_array:
        """
        Distance of each particle from the halo centre, projected along the
        x axis.
        """
        pos = self.position
        return np.sqrt(pos[:, 1] ** 2 + pos[:, 2] ** 2)

    @lazy_property
    def radius_projy(self) -> unyt.unyt_array:
        """
        Distance of each particle from the halo centre, projected along the
        y axis.
        """
        pos = self.position
        return np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)

    @lazy_property
    def radius_projz(self) -> unyt.unyt_array:
        """
        Distance of each particle from the halo centre, projected along the
        z axis.
        """
        pos = self.position
        return np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)

    def fofid_of_particle(self, index: int) -> int:
        """
        FOF group ID of a single particle in the concatenated arrays.

        Only the particle type that particle belongs to is read, so the full
        length array of FOF IDs never has to be built just to look up one value.

        Parameters:
         - index: int
           Position of the particle in the concatenated arrays.
        """
        index = int(index)
        offset = 0
        for ptype, nr_part in self.nr_part_of_type:
            if index < offset + nr_part:
                fofid = self.get_dataset(f"{ptype}/FOFGroupIDs")
                if self.inclusive:
                    # every particle is included, so the position in the
                    # concatenated array is also the position in the raw array
                    return fofid[index - offset]
                return fofid[self.in_halo_mask(ptype)][index - offset]
            offset += nr_part
        raise IndexError(f"particle {index} is not in the concatenated arrays")

    def compute_bound_masks(self, fofid_central: int):
        """
        Flag the particles which are bound to a halo other than this one,
        separating those in the same FOF group from those in another one.

        Parameters:
         - fofid_central: int
           FOF group ID of this halo.
        """
        satellite = []
        external = []
        for ptype, _ in self.nr_part_of_type:
            in_halo = self.in_halo_mask(ptype)
            groupnr = self.get_dataset(f"{ptype}/GroupNr_bound")[in_halo]
            bound_elsewhere = (groupnr >= 0) & (groupnr != self.index)
            del groupnr
            fofid = self.get_dataset(f"{ptype}/FOFGroupIDs")[in_halo]
            same_fof = fofid == fofid_central
            del fofid
            satellite.append(bound_elsewhere & same_fof)
            external.append(bound_elsewhere & ~same_fof)
        self.is_bound_to_satellite = np.concatenate(satellite)
        self.is_bound_to_external = np.concatenate(external)

    def compute_mass_profile(self, cosmology: Dict):
        """
        Compute the cumulative mass profile used to determine the SO radius.

        Adds the contribution from neutrinos (if present) to the masses and
        radii, sorts the particles by radius, and computes the cumulative mass
        profile and the mean density within the radius of each particle. Also
        determines the FOF ID of this object from its central particle, and
        uses that to flag the particles which are bound to another halo.

        None of this depends on the density threshold of an individual SO
        variation, so it is computed once and used by all of them. It is a
        method rather than a lazy property because it needs the cosmology,
        which the aperture calculations that also use this object do not have.
        Repeated calls after the first are no-ops.

        Parameters:
         - cosmology: dict
           Cosmological parameters required for the SO calculation.
        """
        if getattr(self, "have_mass_profile", False):
            return
        # add neutrinos
        if self.has_neutrinos:
            numass = self.get_dataset("PartType6/Masses") * self.get_dataset(
                "PartType6/Weights"
            )
            pos = self.get_dataset("PartType6/Coordinates") - self.centre[None, :]
            nur = np.sqrt(np.sum(pos**2, axis=1))
            self.nu_mass = numass
            self.nu_radius = nur
            self.nu_softening = (
                np.ones_like(nur) * self.softening_of_parttype["PartType6"]
            )
            all_mass = np.concatenate([self.mass, numass / unyt.dimensionless])
            all_r = np.concatenate([self.radius, nur])
        else:
            all_mass = self.mass
            all_r = self.radius

        # Sort by radius
        order = np.argsort(all_r)
        ordered_radius = all_r[order]
        cumulative_mass = np.cumsum(all_mass[order], dtype=np.float64).astype(
            self.mass.dtype
        )
        del all_mass, all_r
        # add mean neutrino mass
        cumulative_mass += (
            cosmology["nu_density"] * 4.0 / 3.0 * np.pi * ordered_radius**3
        )
        # Determine FOF ID of object using the central non-neutrino particle
        non_neutrino_order = order[order < self.radius.shape[0]]
        fofid_central = self.fofid_of_particle(non_neutrino_order[0])
        del order, non_neutrino_order

        # Compute density within radius of each particle.
        # Will need to skip any at zero radius.
        # Note that because of the definition of the halo centre, the first
        # particle *should* be at r=0. We need to manually exclude it, in case round
        # off error places it at a very small non-zero radius.
        nskip = max(1, np.argmax(ordered_radius > 0.0 * ordered_radius.units))
        self.ordered_radius = ordered_radius[nskip:]
        self.cumulative_mass = cumulative_mass[nskip:]
        self.nr_parts = len(self.ordered_radius)
        self.density = self.cumulative_mass / (
            4.0 / 3.0 * np.pi * self.ordered_radius**3
        )

        # figure out which particles in the list are bound to a halo that is not the
        # central halo
        self.compute_bound_masks(fofid_central)
        self.have_mass_profile = True
