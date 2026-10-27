**The Doppler B parameter** is computed as in `Roncarelli et al. (2018) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.467..985R>`__:

.. math::

   b = \frac{\sigma{}_T}{c} \sum_i n_{e,i} v_{r,{\rm{}obs},i} \frac{m_i}{\rho{}_i A_{\rm{}obs}},

where :math:`\sigma_T` is the Thomson cross section, :math:`c` the speed of light, :math:`n_{e,i}` the electron number density
for gas particle :math:`i`, with :math:`V_i=m_i/\rho_i` the corresponding SPH particle volume. The relative
*peculiar* velocity is taken relative to the box and along a line of sight towards a particular observer,
so

.. math::

   v_{r,{\rm{}obs},i} = \vec{v}_{i} \cdot{}
   \frac{\left(\vec{x}_i - \vec{x}_{\rm{}obs}\right)}{\left|\vec{x}_i - \vec{x}_{\rm{}obs}\right|},

with :math:`\vec{x}_i` and :math:`\vec{v}_i` the physical position and velocity of particle :math:`i`, and :math:`\vec{x}_{\rm{}obs}`
the arbitrary observer position.

The surface area :math:`A_{\rm{}obs}` that turns the volume integral into a line integral is that of the aperture
for which :math:`b` is computed, i.e. :math:`A_{\rm{}obs}=\pi{} R_{\rm{}SO}^2`.

As the observer position we use the position of the observer for the first lightcone in the simulation, or the
centre of the box if no lightcone was present. This choice is arbitrary and can be adapted. Since
:math:`\vec{x}_{\rm{}obs}` can in principle coincide with :math:`\vec{x}_i`, we make sure :math:`v_{r,{\rm{}obs},i}` is set to
zero in this case to avoid division by zero.
