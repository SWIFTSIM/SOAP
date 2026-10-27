**Kappa-corot** is computed as in `Correa et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.472L..45C>`__:

.. math::

   \kappa_{\rm{}corot,comp} = \frac{K_{\rm{}corot,comp}}{K_{\rm{}comp}},

with the kinetic energy given by

.. math::

   K_{\rm{}comp} = \frac{1}{2} \sum_{i={\rm{}comp}} m_i |\vec{v}_{{\rm{}comp},r,i}|^2,

the corotational kinetic energy given by

.. math::

   K_{\rm{}corot,comp} = \sum_{i={\rm{}comp}} \begin{cases}
   K_{{\rm{}rot,comp},i}, &L_{{\rm{}comp},p,i} > 0, \\
   0, &L_{{\rm{}comp},p,i} \leq{} 0, \\
   \end{cases}

the rotational kinetic energy given by

.. math::

   K_{{\rm{}rot,comp},i} = \frac{1}{2} \frac{L_{{\rm{}comp},p,i}^2}{m_i R_i^2},

the projected angular momentum along the angular momentum direction given by

.. math::

   L_{{\rm{}comp},p,i} = \vec{L}_i \frac{\vec{L}_{\rm{}comp}}{|\vec{L}_{\rm{}comp}|},

and the orthogonal distance to the angular momentum vector given by

.. math::

   R_i^2 = |\vec{x}_{r,i}|^2 - \left(\vec{x}_{r,i} \frac{\vec{L}_{\rm{}comp}}{|\vec{L}_{\rm{}comp}|}\right)^2,

where the angular momentum vector and the relative position and velocity are the same as above for
consistency.
