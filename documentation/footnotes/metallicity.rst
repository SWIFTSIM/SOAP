**Metallicity** values are calculated using two different ways to average over the particles being considered.

LinearMassWeighted properties, :math:`Z_{\mathrm{lin}}`, are calculated as (using O/H as an example)

.. math::

   Z_{\mathrm{lin,O/H}} = \frac{1}{\sum_i m_i} \sum_i m_i \frac{n_{\mathrm{O},i}}{n_{\mathrm{H},i}}

where :math:`m_i` is the particle mass, :math:`n_{O,i}` is the number density of oxygen for particle :math:`i`, and :math:`n_{H,i}` is the number density of hydrogen for particle :math:`i`.
When calculating gas metallicity we only sum over particles which are cold (:math:`T_i < 10^{$LOG_COLD_GAS_TEMP$}`\ K) and dense (:math:`\frac{\rho_i}{m_{\mathrm{H}}} > 10^{$LOG_COLD_GAS_DENSITY$} \mathrm{cm}^{-3}`).

LogarithmicMassWeighted properties, :math:`Z_{\mathrm{log}}`, are calculated as

.. math::

   \log_{10}Z_{\mathrm{log,O/H}} = \frac{1}{\sum_i m_i} \sum_i m_i \: \log_{10} \left[ \max \left( \frac{n_{O,i}}{n_{H,i}}, \frac{f n_{O,\odot}}{n_{H,\odot}} \right) \right]

where :math:`\frac{n_{O,\odot}}{n_{H,\odot}}` is the solar ratio of oxygen to hydrogen, and :math:`f` is either :math:`10^{-3}` (high limit) or :math:`10^{-4}` (low limit). The max function is required to stop us getting infinities for particles with no oxygen.
