**Core excised quantities** exclude the inner region of the halo when computing the quantity.
They are calculated for the following spherical overdensities: $CORE_EXCISION_VARIATIONS$.
Any core excised calculation only uses the particles for which

.. math::

   f R_{\rm{}SO} \leq \mathbf{r} \leq R_{\rm{}SO},

where :math:`R_{\rm{}SO}` is the radius of the spherical overdensity and $CORE_EXCISION_FRACTION$.
