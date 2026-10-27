Spherical overdensity calculations
----------------------------------

The radius at which the density reaches a certain threshold value is
found by linear interpolation of the cumulative mass profile obtained
after sorting the particles by radius. The process is as follows:

#. Discard any particles at zero radius, since we cannot compute a
   density for those. The mass of these particles is used as an
   :math:`r=0` offset for the cumulative mass profile. Since the halo centre
   is the position of the most bound particle, there should
   always be at least one such particle.

#. Sort remaining particles according to radius and construct the cumulative mass
   profile. :math:`R_{i}` gives the radius of the particles, and
   :math:`M_{i}` the cumulative mass up to and including that radius.

#. Construct the halo density profile,
   :math:`\rho_{i}`,
   by dividing the cumulative mass at
   every radius by the volume of the sphere with that radius.

#. Find intersection points between the density profile and the target
   density, i.e. the consecutive indices :math:`i_{low}`, :math:`i_{high}`
   where :math:`\rho_{low} > \rho_{target}` and
   :math:`\rho_{high} < \rho_{target}`

#. To determine :math:`R_{\rm{}SO}` we solve

   .. math::

      \frac{4\pi{}}{3} \rho{}_{\rm{}target} R_{\rm{}SO}^3 = M_{\rm{}low} + \left( \frac{M_{\rm{}high}-M_{\rm{}low}}{R_{\rm{}high} - R_{\rm{}low}} \right) \left(R_{\rm{}SO} - R_{\rm{}low}\right),

   This third degree polynomial equation has no unique solution, although
   in practice only one of the three possible complex solutions is real. We
   find this solution by using a root finding algorithm within the
   intersecting bin (we use Brent’s method for this).

#. From :math:`R_{\rm{}SO}`, we determine :math:`M_{\rm{}SO}` using

   .. math::

      M_{\rm{}SO} = \frac{4\pi{}}{3} R_{\rm{}SO}^{3} \rho{}_{\rm{}target}

The SOAP process is different to that used by some halo finders, where
both the mass and the radius are obtained from independent interpolations
on the mass and density profiles (the latter uses the logarithm of the
density in the interpolation).

Neutrinos, if present in the model, are included in the inclusive sphere
calculation (and only here, since neutrino particles cannot be bound to a halo)
by adding both their weighted masses (which can be negative), as well as the
contribution from the background neutrino density. The latter is achieved by
explicitly adding the cumulative mass profile at constant neutrino density to the
total cumulative mass profile before computing the density profile. This is the
only place where neutrinos explicitly enter the algorithm, except for the neutrino
masses computed for the SOs. Neutrinos are not included in the calculation of the
centre of mass and centre of mass velocity.
