**The stellar asymmetry** is computed following
`arXiv:1805.03210 <https://arxiv.org/abs/1805.03210>`__. The directions around the centre are divided into
:math:`N_{\rm{}pix}` equal-area HEALPix pixels, and the asymmetry is

.. math::

   A = \frac{1}{2 M_*} \sum_{i=1}^{N_{\rm{}pix}} \left| M_i - M_{\bar{i}} \right|,

where :math:`M_i` is the stellar mass in pixel :math:`i`, :math:`M_{\bar{i}}` is the stellar mass in the pixel in the opposite
direction, and :math:`M_*` is the total stellar mass. Each pair of opposite pixels appears twice in the sum, so :math:`A`
ranges from 0 for a perfectly symmetric distribution to 1 if all the stellar mass is on one side of the centre.
Star particles within 0.1 kpc of the centre are excluded, since their direction from the centre is poorly
defined. The asymmetry is not calculated if there are fewer than 3 star particles.

``StellarAsymmetry`` uses :math:`N_{\rm{}pix}=12` and is centred on the halo centre. The other variants differ
as follows: ``StellarAsymmetryShrink`` is centred on ``ShrinkingSphereCentre``,
``StellarAsymmetrySubsample`` uses a random subsample of 1/8 of the star particles, and
``StellarAsymmetry48`` and ``StellarAsymmetry192`` use :math:`N_{\rm{}pix}=48` and :math:`N_{\rm{}pix}=192`.
