**The mass-weighted temperature** is computed as

.. math::

   T = \frac{1}{\sum_i m_i} \sum_i m_i T_i,

and the ``GasTemperatureWithoutRecentAGNHeating`` variant uses the same definition, but excludes particles
that satisfy

.. math::

   0.1 \Delta{}T_{\rm{}AGN} \leq{} T_i \leq{} 10^{0.3} \Delta{}T_{\rm{}AGN},

and

.. math::

   \texttt{LastAGNFeedbackScaleFactors}_i \geq{} a(t - 15{\rm{}Myr})

where :math:`\Delta T_{\rm{}AGN}` is the same value as used internally by SWIFT, :math:`t` is the cosmic time, and :math:`a()`
gives the scale factor as a function of time.
