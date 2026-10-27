Property filters
================

For certain properties it only makes sense to compute them if a subhalo
contains sufficient particles. Halo finders are often run with a
configuration that requires at least 20 particles for a subhalo. However,
even for those particle numbers, a lot of the properties computed by SOAP will be zero
(e.g. the gas mass within a 10 kpc aperture), or have values that are
outliers compared to the full halo population because of undersampling.
To save disk space and reduce the computation time, we filter these
out by applying appropriate cuts. Filtering means setting the
value of the property to
zero, and HDF5 file compression then very effectively reduces the data
storage required to store these properties, while the size of the arrays
that the end user sees remains unchanged.

Since different properties can have very different requirements,
filtering is done in categories, where each category corresponds to a
set of quantities that are filtered using the same criterion. The filter for
a property is the same for all halo types, e.g. the properties for an inclusive
sphere with a 10 kpc aperture radius use the same filters as the properties of
an inclusive sphere with a 3000 kpc aperture radius, even though the latter by
construction has many more particles.
