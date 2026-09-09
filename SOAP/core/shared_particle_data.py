#!/bin/env python

"""
shared_particle_data.py

Cache of particle quantities that are shared between the property
calculations of a single halo.

process_single_halo() in halo_tasks.py hands the same set of particles to
every property calculation it runs for a halo. Several of those calculations
begin by deriving the same quantities from that set (concatenated masses and
radii, sorted radial profiles, ...), which is wasted work when it is repeated
once per calculation.

A SharedParticleData object lets those calculations look up quantities that
have already been derived from the same particles. It is created inside the
search radius loop of process_single_halo(), so a new (empty) cache is used
whenever the set of particles changes.
"""

from typing import Any, Callable, Hashable


class SharedParticleData:
    """
    Cache of quantities derived from the particles of a single halo.

    Entries are created on first use, so nothing is computed for a halo unless
    a property calculation actually asks for it.
    """

    def __init__(self):
        """
        Constructor. Creates an empty cache.
        """
        self.cache = {}

    def get(self, key: Hashable, factory: Callable[[], Any]) -> Any:
        """
        Return the cached entry for key, creating it with factory() if this is
        the first time it has been requested.

        Parameters:
         - key: Hashable
           Identifies the quantity being requested. Calculations that want to
           share an entry have to agree on the key, so it needs to include
           everything the entry depends on (e.g. the particle types that were
           used to compute it).
         - factory: Callable
           Function taking no arguments which computes the entry. It is only
           called if the key is not already in the cache.
        """
        if key not in self.cache:
            self.cache[key] = factory()
        return self.cache[key]
