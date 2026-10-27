SOAP catalogues
===============

SOAP (Spherical Overdensity and Aperture Processor) computes properties for
different types of halos, which differ in how they decide which particles to
include (by radius, in projection...). SOAP does not identify halos itself,
instead it uses the halo centre and the boundedness of particles as determined
by an input subhalo catalogue (such as HBT-HERONS, SubFind, etc...).

The purpose of these pages is to document the catalogues output by SOAP. For
information about running the code you should refer to the
`README <https://github.com/SWIFTSIM/SOAP>`__.
This documentation is generated using the SOAP parameter file, and so the halo
types and properties listed reflect those present in the current run of SOAP,
rather than all possible properties.

SOAP catalogues are output as HDF5 files, with the properties being grouped by
halo type. For loading the catalogues we recommend using the
`swiftsimio <https://swiftsimio.readthedocs.io/en/latest/soap/index.html>`__
package. A simple example is:

.. code-block:: python

   import swiftsimio as sw
   base_dir = '/cosma8/data/dp004/flamingo/Runs'
   run = 'L1000N1800/HYDRO_FIDUCIAL'
   data = sw.load(f'{base_dir}/{run}/SOAP-HBT/halo_properties_0077.hdf5')
   # Load a dataset
   m200 = data.spherical_overdensity_200_crit.total_mass
   # Specify units
   m200.convert_to_units('Msun')
   # Specify physical/comoving
   m200.convert_to_physical()
   # Get metadata from file
   z = data.metadata.redshift

The following pages describe the halo types computed by SOAP, the filters
applied to each property, the full list of properties in the catalogues, and
the group membership files:

.. toctree::
   :maxdepth: 1

   halo_variations
   filters
   property_table
   membership

.. include:: version.rst
