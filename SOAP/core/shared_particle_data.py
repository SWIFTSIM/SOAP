#!/bin/env python

"""
shared_particle_data.py

Cache of particle quantities that are shared between the property
calculations of a single halo.
"""

from typing import Any, Callable, Hashable, Iterable


class ParticleDataCache:
    """
    Entries are created on first use, so nothing is computed for a halo unless
    a property calculation actually asks for it, and discarded once the
    calculations which need them have all run.
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
           Identifies the quantity being requested. See HaloProperty.shared_key().
         - factory: Callable
           Function taking no arguments which computes the entry. It is only
           called if the key is not already in the cache.
        """
        if key not in self.cache:
            self.cache[key] = factory()
        return self.cache[key]

    def keep_only(self, keys: Iterable[Hashable]):
        """
        Drop every entry whose key is not in keys.

        Parameters:
         - keys: Iterable
           Keys to keep. Anything else is discarded.
        """
        keys = set(keys)
        for key in [k for k in self.cache if k not in keys]:
            del self.cache[key]
