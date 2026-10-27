.. _exclusive_sphere_description:

Exclusive sphere
----------------

Exclusive sphere properties are similar to subhalo quantities as they include only the
particles that are bound to the subhalo, but they apply an additional radial cut (aperture). Exclusive sphere
properties are contained within the group ``ExclusiveSphere``. Groups with the label ``XXXkpc`` have an
aperture radius of ``XXX`` **physical** kpc. Groups with a property name have a cut using the value
of the property calculated for the bound subhalo (e.g. the aperture cut for ``ExclusiveSphere/HalfMassRadiusTotal``
is given by the value of ``BoundSubhalo/HalfMassRadiusTotal``), meaning a different radial cut is used for each subhalo.
