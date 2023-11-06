#! /usr/bin/env python

"""
kinematic_properties.py

Some utility functions to compute kinematic properies for particle
distributions.

We put them in a separate file to facilitate unit testing.
"""

import numpy as np
import unyt
from typing import Union, Tuple


def get_velocity_dispersion_matrix(
    mass_fraction: unyt.unyt_array,
    velocity: unyt.unyt_array,
    ref_velocity: unyt.unyt_array,
) -> unyt.unyt_array:
    """
    Compute the velocity dispersion matrix for the particles with the given
    fractional mass (particle mass divided by total mass) and velocity, using
    the given reference velocity as the centre of mass velocity.

    The result is a 6 element vector containing the unique components XX, YY,
    ZZ, XY, XZ and YZ of the velocity dispersion matrix.

    Parameters:
     - mass_fraction: unyt.unyt_array
       Fractional mass of the particles (mass/mass.sum()).
     - velocity: unyt.unyt_array
       Velocity of the particles.
     - ref_velocity: unyt.unyt_array
       Reference point in velocity space. velocity and ref_velocity are assumed
       to use the same reference point upon entry into this function.

    Returns an array with 6 elements: the XX, YY, ZZ, XY, XZ and YZ components
    of the velocity dispersion matrix.
    """

    result = unyt.unyt_array(np.zeros(6), dtype=np.float32, units=velocity.units ** 2)

    vrel = velocity - ref_velocity[None, :]
    result[0] += (mass_fraction * vrel[:, 0] * vrel[:, 0]).sum()
    result[1] += (mass_fraction * vrel[:, 1] * vrel[:, 1]).sum()
    result[2] += (mass_fraction * vrel[:, 2] * vrel[:, 2]).sum()
    result[3] += (mass_fraction * vrel[:, 0] * vrel[:, 1]).sum()
    result[4] += (mass_fraction * vrel[:, 0] * vrel[:, 2]).sum()
    result[5] += (mass_fraction * vrel[:, 1] * vrel[:, 2]).sum()

    return result


def get_angular_momentum(
    mass: unyt.unyt_array,
    position: unyt.unyt_array,
    velocity: unyt.unyt_array,
    ref_position: Union[None, unyt.unyt_array] = None,
    ref_velocity: Union[None, unyt.unyt_array] = None,
) -> unyt.unyt_array:
    """
    Compute the total angular momentum vector for the particles with the given
    masses, positions and velocities, and using the given reference position
    and velocity as the centre of mass (velocity).

    Parameters:
     - mass: unyt.unyt_array
       Masses of the particles.
     - position: unyt.unyt_array
       Position of the particles.
     - velocity: unyt.unyt_array
       Velocities of the particles.
     - ref_position: unyt.unyt_array or None
       Reference position used as centre for the angular momentum calculation.
       position and ref_position are assumed to use the same reference point upon
       entry into this function. If None, position is assumed to be already using
       the desired referece point.
     - ref_velocity: unyt.unyt_array or None
       Reference point in velocity space for the angular momentum calculation.
       velocity and ref_velocity are assumed to use the same reference point upon
       entry into this function. If None, velocity is assumed to be already using
       the desired reference point.

    Returns the total angular momentum vector.
    """

    if ref_position is None:
        prel = position
    else:
        prel = position - ref_position[None, :]
    if ref_velocity is None:
        vrel = velocity
    else:
        vrel = velocity - ref_velocity[None, :]
    return (mass[:, None] * unyt.array.ucross(prel, vrel)).sum(axis=0)


def get_angular_momentum_and_kappa_corot(
    mass: unyt.unyt_array,
    position: unyt.unyt_array,
    velocity: unyt.unyt_array,
    ref_position: Union[None, unyt.unyt_array] = None,
    ref_velocity: Union[None, unyt.unyt_array] = None,
    do_counterrot_mass: bool = False,
) -> Union[
    Tuple[unyt.unyt_array, unyt.unyt_quantity],
    Tuple[unyt.unyt_array, unyt.unyt_quantity, unyt.unyt_quantity],
]:
    """
    Get the total angular momentum vector (as in get_angular_momentum()) and
    kappa_corot (Correa et al., 2017) for the particles with the given masses,
    positions and velocities, and using the given reference position and
    velocity as centre of mass (velocity).

    If both kappa_corot and the angular momentum vector are desired, it is more
    efficient to use this function that calling get_angular_momentum() (and
    get_kappa_corot(), if that would ever exist).

    Parameters:
     - mass: unyt.unyt_array
       Masses of the particles.
     - position: unyt.unyt_array
       Position of the particles.
     - velocity: unyt.unyt_array
       Velocities of the particles.
     - ref_position: unyt.unyt_array or None
       Reference position used as centre for the angular momentum calculation.
       position and ref_position are assumed to use the same reference point upon
       entry into this function. If None, position is assumed to be already using
       the desired referece point.
     - ref_velocity: unyt.unyt_array or None
       Reference point in velocity space for the angular momentum calculation.
       velocity and ref_velocity are assumed to use the same reference point upon
       entry into this function. If None, velocity is assumed to be already using
       the desired reference point.
     - do_counterrot_mass: bool
       Also compute the counterrotating mass?

    Returns:
     - The total angular momentum vector.
     - The ratio of the kinetic energy in counterrotating movement and the total
       kinetic energy, kappa_corot.
     - The total mass of counterrotating particles (if do_counterrot_mass == True).
    """

    kappa_corot = unyt.unyt_array(
        0.0, dtype=np.float32, units="dimensionless", registry=mass.units.registry
    )

    if ref_position is None:
        prel = position
    else:
        prel = position - ref_position[None, :]
    if ref_velocity is None:
        vrel = velocity
    else:
        vrel = velocity - ref_velocity[None, :]

    Lpart = mass[:, None] * unyt.array.ucross(prel, vrel)
    Ltot = Lpart.sum(axis=0)
    Lnrm = unyt.array.unorm(Ltot)

    if do_counterrot_mass:
        M_counterrot = unyt.unyt_array(
            0.0, dtype=np.float32, units=mass.units, registry=mass.units.registry
        )

    if Lnrm > 0.0 * Lnrm.units:
        K = 0.5 * (mass[:, None] * vrel ** 2).sum()
        if K > 0.0 * K.units or do_counterrot_mass:
            Ldir = Ltot / Lnrm
            Li = (Lpart * Ldir[None, :]).sum(axis=1)
        if K > 0.0 * K.units:
            r2 = prel[:, 0] ** 2 + prel[:, 1] ** 2 + prel[:, 2] ** 2
            rdotL = (prel * Ldir[None, :]).sum(axis=1)
            Ri2 = r2 - rdotL ** 2
            # deal with division by zero (the first particle is guaranteed to
            # be in the centre)
            mask = Ri2 == 0.0
            Ri2[mask] = 1.0 * Ri2.units
            Krot = 0.5 * (Li ** 2 / (mass * Ri2))
            Kcorot = Krot[(~mask) & (Li > 0.0 * Li.units)].sum()
            kappa_corot += Kcorot / K

        if do_counterrot_mass:
            M_counterrot += mass[Li < 0.0 * Li.units].sum()

    if do_counterrot_mass:
        return Ltot, kappa_corot, M_counterrot
    else:
        return Ltot, kappa_corot


def get_vmax(
    mass: unyt.unyt_array, radius: unyt.unyt_array
) -> Tuple[unyt.unyt_quantity, unyt.unyt_quantity]:
    """
    Get the maximum circular velocity of a particle distribution.

    The value is computed from the cumulative mass profile after
    sorting the particles by radius, as
     vmax = sqrt(G*M/r)

    Parameters:
     - mass: unyt.unyt_array
       Mass of the particles.
     - radius: unyt.unyt_array
       Radius of the particles.

    Returns:
     - Radius at which the maximum circular velocity is reached.
     - Maximum circular velocity.
    """
    # obtain the gravitational constant in the right units
    # (this is read from the snapshot metadata, and is hence
    # guaranteed to be consistent with the value used by SWIFT)
    G = unyt.Unit("newton_G", registry=mass.units.registry)
    isort = np.argsort(radius)
    ordered_radius = radius[isort]
    cumulative_mass = mass[isort].cumsum()
    nskip = max(1, np.argmax(ordered_radius > 0.0 * ordered_radius.units))
    ordered_radius = ordered_radius[nskip:]
    if len(ordered_radius) == 0 or ordered_radius[0] == 0:
        return 0.0 * radius.units, np.sqrt(0.0 * G * mass.units / radius.units)
    cumulative_mass = cumulative_mass[nskip:]
    v_over_G = cumulative_mass / ordered_radius
    imax = np.argmax(v_over_G)
    return ordered_radius[imax], np.sqrt(v_over_G[imax] * G)


def get_inertia_tensor(
    mass: unyt.unyt_array, position: unyt.unyt_array
) -> unyt.unyt_array:
    """
    Get the inertia tensor of the given particle distribution.

    The axis lengths are defined as the eigenvalues of the (normal) moment of
    inertia tensor, computed as
     I_{ij} = 1/M * (m*x_i*x_j)

    Parameters:
     - mass: unyt.unyt_array
       Masses of the particles.
     - position: unyt.unyt_array
       Positions of the particles.

    Returns the inertia tensor.
    """
    Itensor = (mass[:, None, None]) * np.ones((mass.shape[0], 3, 3))
    # Note: unyt currently ignores the position units in the *=
    # i.e. Itensor is dimensionless throughout (even though it should not be)
    for i in range(3):
        for j in range(3):
            Itensor[:, i, j] *= position[:, i].value * position[:, j].value
    Itensor = Itensor.sum(axis=0)
    Itensor = (
        np.array(
            (
                Itensor[0, 0],
                Itensor[1, 1],
                Itensor[2, 2],
                Itensor[0, 1],
                Itensor[0, 2],
                Itensor[1, 2],
            )
        )
        * position.units
        * position.units
        * mass.units
    )
    return Itensor


def get_projected_inertia_tensor(
    mass: unyt.unyt_array, position: unyt.unyt_array, axis: int
) -> unyt.unyt_array:
    """
    Get the inertia tensor of the given particle distribution, projected along the
    given axis.

    Parameters:
     - mass: unyt.unyt_array
       Masses of the particles.
     - position: unyt.unyt_array
       Positions of the particles.
     - axis: 0, 1, 2
       Projection axis. Only the coordinates perpendicular to this axis are
       taken into account.

    Returns the inertia tensor.
    """

    projected_position = unyt.unyt_array(
        np.zeros((position.shape[0], 2)), units=position.units, dtype=position.dtype
    )
    if axis == 0:
        projected_position[:, 0] = position[:, 1]
        projected_position[:, 1] = position[:, 2]
    elif axis == 1:
        projected_position[:, 0] = position[:, 2]
        projected_position[:, 1] = position[:, 0]
    elif axis == 2:
        projected_position[:, 0] = position[:, 0]
        projected_position[:, 1] = position[:, 1]
    else:
        raise AttributeError(f"Invalid axis: {axis}!")

    Itensor = (mass[:, None, None]) * np.ones((mass.shape[0], 2, 2))
    # Note: unyt currently ignores the position units in the *=
    # i.e. Itensor is dimensionless throughout (even though it should not be)
    for i in range(2):
        for j in range(2):
            Itensor[:, i, j] *= (
                projected_position[:, i].value * projected_position[:, j].value
            )
    Itensor = Itensor.sum(axis=0)
    Itensor = (
        np.array((Itensor[0, 0], Itensor[1, 1], Itensor[0, 1]))
        * position.units
        * position.units
        * mass.units
    )
    return Itensor


if __name__ == "__main__":
    """
    Standalone version. TODO: add test to check if inertia tensor computation works.
    """
    pass
