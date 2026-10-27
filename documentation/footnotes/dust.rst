**Dust quantities assume a six element dust model** with one type of graphite
grains and two types of silicate grains, each of which has two size bins. The dust mass fractions are
then based on the ``DustMassFractions`` dataset in the snapshot, assuming the following order of the
six columns:

#. Large graphite grains
#. Large silicate grains of type 1
#. Large silicate grains of type 2
#. Small graphite grains
#. Small silicate grains of type 1
#. Small silicate grains of type 2

The total graphite mass, ``DustGraphiteMass``, is then for example calculated by summing the masses
contained in columns 1 and 3.

The mass of dust in molecular gas is defined as

.. math::

   \sum_i m_d \frac{m_{\rm{H}_2,i}}{m_{\rm{H}, i}}

where :math:`m_d` is the dust mass of particle :math:`i`, :math:`m_{\rm{H}_2,i}` is the molecular hydrogen mass of the particle,
and :math:`m_{\rm{H},i}` is the total hydrogen mass of the particle. A similar expression is used for the mass of dust in atomic gas.
