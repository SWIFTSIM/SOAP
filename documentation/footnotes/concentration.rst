**The concentration** is computed using the
method described in `Wang et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.52710760W>`__, but using a fifth order polynomial fit to
the R1-concentration relation for :math:`1<c<1000`. Therefore, we set a floor of 1 and
a ceiling of 1000 for the values calculated by SOAP. This method assumes halos have
an NFW profile, and is only calculated for the
following SO variations: 200 crit, 200 mean, and BN98.
Neutrinos are included in the calculation of total concentration.
The first moment of the density distribution, :math:`R1`, can be estimated from
the concentration. From :math:`R1` the Einasto concentration can be calculated. It
also possible to estimate other properties, such as :math:`v_{\rm{}max}`, by using the :math:`R1`
value and assuming an NFW profile.
