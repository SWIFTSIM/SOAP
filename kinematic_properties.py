#! /usr/bin/env python

import numpy as np
import unyt


def get_velocity_dispersion_matrix(mass_fraction, velocity, ref_velocity):
    """
    Compute the velocity dispersion matrix for the particles with the given
    fractional mass (particle mass divided by total mass) and velocity, using
    the given reference velocity as the centre of mass velocity.

    The result is a 6 element vector containing the unique components XX, YY,
    ZZ, XY, XZ and YZ of the velocity dispersion matrix.
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
    mass, position, velocity, ref_position=None, ref_velocity=None
):
    """
    Compute the total angular momentum vector for the particles with the given
    masses, positions and velocities, and using the given reference position
    and velocity as the centre of mass (velocity).
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
    mass,
    position,
    velocity,
    ref_position=None,
    ref_velocity=None,
    do_counterrot_mass=False,
):
    """
    Get the total angular momentum vector (as in get_angular_momentum()) and
    kappa_corot (Correa et al., 2017) for the particles with the given masses,
    positions and velocities, and using the given reference position and
    velocity as centre of mass (velocity).

    If both kappa_corot and the angular momentum vector are desired, it is more
    efficient to use this function that calling get_angular_momentum() (and
    get_kappa_corot(), if that would ever exist).
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


def get_vmax(mass, radius):
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


def get_inertia_tensor(mass, position):

    # 3x3 inertia tensor
    Itensor = (mass[:, None, None] * position[:, None:, None] * position[:, None]).sum(
        axis=0
    )

    # Symmetric, so only return lower triangle
    Itensor = np.concatenate([np.diag(Itensor), Itensor[np.triu_indices(3, 1)]])

    return Itensor


def get_projected_inertia_tensor(mass, position, axis):
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
    pass
