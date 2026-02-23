SOAP properties table
=====================

The tables below list the (sub)halo properties available within the SOAP catalogues. The first table contains the properties within the ``input_halos`` group.
The second table contains the properties which are calculated for both DMO and HYDRO simulations.
The third table contains the properties which are only calculated for the HYDRO simulations.
The final table contains the datasets copied over from the HBT-HERONS and FoF catalogues.
Within each table the properties are sorted based on their filters.

The first column gives the name of the property when opened using the `swiftsimio library <https://swiftsimio.readthedocs.io/en/latest/soap/index.html>`_. Clicking on each property name will open a dropdown box, which contains information about the dataset within the HDF5 file. The second column gives the filter applied to that property, as descibred in :doc:`property_filters`. The third column indicates the halo variations for which this property is available (green if the property is computed for a certain variation, red if not). The variations are as follows:

* ``BS`` - :ref:`bound_subhalo_description`
* ``ES`` - :ref:`exclusive_sphere_description`
* ``IS`` - :ref:`inclusive_sphere_description`
* ``EP`` - :ref:`projected_aperture_description`
* ``SO`` - :ref:`spherical_overdensity_description`

The final column gives a description of the property. Certain properties also contain a link to a footnote at the bottom of this page which gives a full description of how they were calculated.


