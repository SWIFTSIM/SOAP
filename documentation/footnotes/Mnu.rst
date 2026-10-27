**The neutrino masses** exist in two flavours. ``RawNeutrinoMass`` is
obtained by simply summing the neutrino particle masses, while the noise suppressed version,
``NoiseSuppressedNeutrinoMass`` is defined as

.. math::

   M_{\nu{},{\rm{}NS}} = \sum_i m_i w_i + \frac{4\pi{}}{3} \rho{}_{\nu{}} R_{\rm{}SO}^3,

where :math:`w_i` are the neutrino weights (which can be negative), and :math:`\rho_{\nu}` is the background density
of neutrinos that is also used in the SO radius calculation. The latter is obtained from the snapshot header.
