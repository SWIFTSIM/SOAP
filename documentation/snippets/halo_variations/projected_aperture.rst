.. _projected_aperture_description:

Exclusive projected aperture
----------------------------

Exclusive projected quantities are similar to exclusive sphere quantities, except that their
aperture cut is applied in projection. For each radius there are three
independent projections: along the
x-, y-, and z-axes. Along the projection axis, we do not apply any radial cut, meaning the depth corresponds to all particles
bound to the subhalo. Projected aperture quantities are stored in a group named
``ProjectedAperture/XXX/projP``, where ``XXX`` is the corresponding aperture cut, and ``P``
corresponds to a particular projection direction (``x``, ``y`` or ``z``).
