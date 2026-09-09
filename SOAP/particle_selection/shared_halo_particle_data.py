#!/bin/env python

"""
shared_halo_particle_data.py

Particle arrays shared between the aperture, projected aperture and subhalo
property calculations of a single halo.

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
        self.types_present = types_present
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
        calculations, all of them for inclusive ones. This mask needs to be
        applied _first_ to raw "PartTypeX" datasets.

        The mask is computed once per particle type and then reused, since
        every calculation sharing this object needs the same one.

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

        Also records the number of particles of each type, which the lazy
        softening below uses to rebuild a per type quantity in the same order.
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
