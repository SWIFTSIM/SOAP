**The Compton y parameter** is computed as in `McCarthy et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.2936M>`__:

.. math::

   y \, {d_A}^2(z) = \sum_i \frac{\sigma{}_T}{m_e c^2} n_{e,i} k_B T_{e,i} V_i,

where :math:`d_A(z)` is the angular diameter distance, :math:`\sigma_T` is the Thomson cross section, :math:`m_e` the electron mass, :math:`c` the speed of light and :math:`k_B` the
Boltzmann constant. :math:`n_{e,i}` and :math:`T_{e,i}` are the electron number density and electron temperature for gas
particle :math:`i`, while :math:`V_i=m_i/\rho_i` is the SPH volume element that turns the sum over all particles :math:`i`
within the inclusive sphere into a volume integral. Note that the snapshot already contains the individual
:math:`y_i` values for the SPH particles.
