Property table
==============

The tables below list the (sub)halo properties available within the SOAP catalogues.
The properties are split into separate tables for:

$PROPERTY_TABLES$

Within each table the properties are sorted based on their filters.
The bottom of the page contains a number of footnotes, which give further explanation as to how different properties are computed.

The first column gives the name of the property when opened using the `swiftsimio library <https://swiftsimio.readthedocs.io/en/latest/soap/index.html>`_. Clicking on each property name will open a dropdown box, which contains information about the dataset within the HDF5 file. The second column gives the filter applied to that property, as described in :doc:`filters`. The third column indicates the halo variations for which this property is available (:avail:`green` if the property is computed for a certain variation, :snaponly:`blue` if the property is available for snapshots but not snipshots, :unavail:`red` if not computed for that variation). The variations are as follows:

$HALO_VARIATIONS$

The final column gives a description of the property. Certain properties also contain a link to a footnote at the bottom of this page which gives a full description of how they were calculated.

Properties are computed using the value of ``HaloCentre`` as the centre of the subhalo, unless otherwise stated.
**All quantities are h-free.** The scale factor :math:`a` is explicitly included in the units of comoving properties
(e.g. the units of ``HaloCentre`` are :math:`a \cdot \rm{Mpc}`).
We recommend that swiftsimio is used to load the catalogues as it has unit handling built in.
However, the attributes of each SOAP dataset contain all the relevant metadata to convert between physical
and comoving units, i.e. information about how the quantity depends on the scale factor, and what the
conversion factor to and from CGS units is.
