.. _spherical_overdensity_description:

Spherical overdensity
---------------------

Spherical overdensities are fundamentally different from the other
halo variations in that their
aperture radius is determined from the density profile and so has a different value
for each halo. They always include all particles within a sphere
centred on the halo centre, regardless of particle subhalo membership. The
radius is either the radius at which the density reaches a certain
target value (e.g. 200 crit, 500 crit, 200 mean,
`Bryan & Norman (1998) <https://ui.adsabs.harvard.edu/abs/1998ApJ...495...80B>`__)
or a multiple of such a radius (e.g. 5xR 500 crit).
See `Spherical overdensity calculations`_ for details about how the radii are
calculated.
Spherical overdensities are only computed for centrals,
i.e. field halos. The spherical overdensity quantities are stored in a group
``SO/XXX``, where ``XXX`` can be either ``XXX_mean`` for density multiples of the
mean density, ``XXX_crit`` for density multiples of the critical density, ``BN98``
for the overdensity definition of Bryan & Norman (1998), or ``YxR_XXX_ZZZ`` for
multiples of some other radius (e.g. ``5xR_500_crit``).
