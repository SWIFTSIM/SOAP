.. _inclusive_sphere_description:

Inclusive sphere
----------------

Inclusive sphere properties apply an aperture radius cut,
the same as the exclusive sphere
quantities. However for the inclusive sphere we include all particles within the radius, regardless of their
membership status.
The quantities are stored within the group ``InclusiveSphere``.

``InclusiveSphere`` properties are computed for all aperture radii smaller than the ``EncloseRadius``
(the maximum distance between any bound particle and the halo centre),
plus the first aperture that exceeds it. For example, if a subhalo has an
``EncloseRadius`` of 29 kpc, and there are apertures with radii of 20, 30 and 50 kpc,
then the 20 kpc and 30 kpc apertures are computed, but the 50 kpc aperture is not.
