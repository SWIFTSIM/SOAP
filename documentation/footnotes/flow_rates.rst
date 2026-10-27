**The flow rates** are computed for three spherical
shells: :math:`R` = :math:`f R_{SO}` with :math:`f = 0.1, 0.3, 1`. In all cases the width of the
spherical shell is given by :math:`dR = 0.1 R`.

For each particle :math:`i` within the shell we calculate their radial velocity as

.. math::

   v_{r,i} = (\underline{v_i} - \underline{v_{COM}}) \cdot \underline{\hat{r}} - \dot{R}

where :math:`\underline{v_i}` is the velocity of the particle, :math:`\underline{v_{COM}}` is the
centre of mass velocity of all particles within :math:`R` (therefore we use a different value
for each spherical shell). The final term accounts for the "pseudo-evolution" of the halo
radius and is given by

.. math::

   \dot{R} = -\frac{1}{3} f R_{SO} H \frac{\mathrm{d} \ln \rho_{\rm ref}}{\mathrm{d} \ln a},

where :math:`\rho_{\rm ref}` is the reference density used to define the SO radius. Since
this depends on the SO definition, we can write :math:`\dot{R} = c f R_{SO} H` with

.. math::

   c = \begin{cases}
   1 & \Delta_{m} \; ({\rm e.g.} \; 200_{m}) \\[4pt]
   \frac{2}{3} \left( 2 \Omega_r + \frac{3}{2} \Omega_m \right) & \Delta_{c} \; ({\rm e.g.} \; 200_{c}, 500_{c}) \\[4pt]
   \frac{1}{3} \left[ 2 \left( 2 \Omega_r + \frac{3}{2} \Omega_m \right) - \frac{\mathrm{d} \ln \Delta_{BN}}{\mathrm{d} \ln a} \right] & BN98
   \end{cases}

where :math:`\Delta_{BN}` is the Bryan & Norman (1998) critical density multiple. The density
parameters are evaluated at the redshift of the snapshot, not at :math:`z=0`.

This is required since accretion rates are often measured by subtracting
the halo mass between consecutive snapshots and dividing by the time interval.
To be consistent with this method we must consider that the virial radius is
defined w.r.t background density (which decreases in time). Hence, the virial radius actually moves
outward with a velocity that we can compute analytically. This means that static
particles at the virial radius actually become inflowing. The expression is derived by taking the partial
differential of the analytic expression for :math:`R_{SO}` w.r.t. time at fixed :math:`M_{SO}`.
Note that :math:`v_{r,i}` does not include the Hubble flow relative to the halo centre.

To calculate the mass inflow (outflow) rate
we compute the following sum over particles within the spherical shell which satisfy
:math:`v_{r,i} < 0` (:math:`v_{r,i} > 0`),

.. math::

   \frac{1}{dR} \sum_{i} m_i v_{r, i},

where :math:`m_i` is the mass of particle :math:`i`. For energy flow rates the sum is

.. math::

   \frac{1}{dR} \sum_{i} m_i v_{r, i} \left(\frac{v_i^2}{2} + u_i\right),

where :math:`u_i` is the internal energy per unit mass and :math:`v_i` is the total 3D velocity
relative to the center of mass velocity. For momentum flow rates the sum is

.. math::

   \frac{1}{dR} \sum_{i} m_i \left(v_{r,i}^2 + \frac{c_s^2}{\gamma}\right),

where :math:`c_s` is the sound speed and :math:`\gamma` = 5/3 (the second term accounts for pressure).
For the gas phases we also calculate "fast outflow" rates.
These are calculated by using the equations above, but only for particles that satisfy :math:`v_{r,i} > V_{max} / 4`,
where :math:`V_{max}` is the maximum circular velocity of the halo. The flow rates are always positive,
so to compute the net rate you must subtract the inflow rate from the outflow rate.
Flow rates are calculated for all SO definitions except those with a radius that is a multiple of another SO radius.
To calculate the total gas flow rate the individual phases should be summed together.
