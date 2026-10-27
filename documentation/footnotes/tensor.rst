**The inertia tensor** for a set of particles is computed as

.. math::

   I_{ij} = \frac{1}{\sum_k m_k} \sum_k m_k \; r_{k,i} \; r_{k, j}

where the index :math:`k` loops over all particles, :math:`m_k` is the mass of particle :math:`k`, and :math:`r_{k, i}` is the :math:`i`-component of the position vector of particle :math:`k` relative to the halo centre. We first compute the inertia tensor using all particles within a sphere (with radius equal to the aperture size, except for subhalos where we use the half mass radius of the particles). This is the tensor we output in the non-iterative case. In the iterative case we construct an ellipsoid with a volume equal to the initial sphere, but whose shape is given by the inertia tensor. We then recalculate the inertia tensor using only the particles within the ellipsoid. This process is repeated until the value of the :math:`q` parameter converges, or we reach 20 iterations. If at any point during the iterations there is only a single particle within the ellipsoid, we return zero. For projected apertures the process is similar, except we use circles and ellipses in the projected plane to determine which particles to include.

The reduced inertia tensor is calculated as

.. math::

   I_{ij} = \frac{1}{\sum_k m_k} \sum_k m_k \; r_{k,i} \; r_{k, j} \; r_{k}^{-2}

where :math:`r_k` is the radial distance of the particle.

We do not calculate the inertia tensor if there are fewer than 20 particles within the initial sphere.

When calculating the inertia tensor for a bound subhalo we use a sphere with a radius equal to 10 times the half mass radius of the particles being considered.
