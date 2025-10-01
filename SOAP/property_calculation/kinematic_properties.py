#! /usr/bin/env python

"""
kinematic_properties.py

Some utility functions to compute kinematic properies for particle
distributions.

We put them in a separate file to facilitate unit testing.
"""

import numpy as np
from typing import Union, Tuple
import unyt


def get_weighted_rotation_velocity(particle_weights: unyt.unyt_array,
                                   particle_azimuthal_velocities: unyt.unyt_array) -> unyt.unyt_quantity:
    """
    Get the weighted average azimuthal velocity of a particle distribution.

    Parameters:
     - particle_weights: unyt.unyt_array
       Weight assigned to each particle.
     - particle_azimuthal_velocities: unyt.unyt_array
       Azimuthal velocity of each particle.

    Returns:
     - Weighted average of the azimuthal velocity of particles.
    """
    return (particle_weights * particle_azimuthal_velocities).sum()

def get_rotation_velocity_mass_weighted(particle_masses, particle_azimuthal_velocities) -> unyt.unyt_quantity:
    """
    Return the mass-weighted average azimuthal velocity of a particle distribution.

    Parameters:
     - particle_masses: unyt.unyt_array
       Mass of particle.
     - particle_azimuthal_velocities: unyt.unyt_array
       Azimuthal velocity of each particle.

    Returns:
     - Mass-weighted average of the azimuthal velocity of particles.
    """
    mass_weights = particle_masses / particle_masses.sum()
    return get_weighted_rotation_velocity(mass_weights, particle_azimuthal_velocities)

def get_rotation_velocity_luminosity_weighted(particle_luminosities: unyt.unyt_array,
                                               particle_azimuthal_velocities: unyt.unyt_array) -> unyt.unyt_array:
    """
    Return the luminosity-weighted average azimuthal velocity of a particle distribution, for each
    provided luminosity band.

    Parameters:
     - particle_luminosities: unyt.unyt_array
       Luminosity of each particle in the provided bands.
     - particle_azimuthal_velocities: unyt.unyt_array
       Azimuthal velocity of each particle, pre-computed for each luminosity band.

    Returns:
     - Luminosity-weighted average of the azimuthal velocity of particles, with a value for each band.
    """

    number_luminosity_bands = particle_luminosities.shape[1]
    rotation_velocities = np.zeros(number_luminosity_bands.shape[0]) * particle_azimuthal_velocities.units

    for i_band, (particle_luminosities_i_band, particle_azimuthal_velocities_i_band) in enumerate(zip(particle_luminosities.T, particle_azimuthal_velocities.T)):
        luminosity_weights = particle_luminosities_i_band / particle_luminosities_i_band.sum()
        rotation_velocities[i_band] = get_weighted_rotation_velocity(luminosity_weights, particle_azimuthal_velocities_i_band)

    return rotation_velocities

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

    result = unyt.unyt_array(np.zeros(6), dtype=np.float32, units=velocity.units**2)

    vrel = velocity - ref_velocity[None, :]
    result[0] += (mass_fraction * vrel[:, 0] * vrel[:, 0]).sum()
    result[1] += (mass_fraction * vrel[:, 1] * vrel[:, 1]).sum()
    result[2] += (mass_fraction * vrel[:, 2] * vrel[:, 2]).sum()
    result[3] += (mass_fraction * vrel[:, 0] * vrel[:, 1]).sum()
    result[4] += (mass_fraction * vrel[:, 0] * vrel[:, 2]).sum()
    result[5] += (mass_fraction * vrel[:, 1] * vrel[:, 2]).sum()

    return result

def get_weighted_cylindrical_velocity_dispersion_vector(
    particle_weights: unyt.unyt_array,
    particle_cylindrical_velocities: unyt.unyt_array,
) -> unyt.unyt_array:
    """
    Compute the velocity dispersion along the radial, azimuthal and vertical
    directions for the input particles and their specified weights.

    Parameters:
     - particle_weights: unyt.unyt_array
       Weight assigned to each particle.
     - particle_cylindrical_velocities: unyt.unyt_array
       Velocity of the particles in a cylindrical coordinate system.

    Returns a 3 element vector containing [sigma_r, sigma_phi, sigma_z].
    """

    # This implementation of standard deviation is more numerically stable than using <x^2> - <x>^2
    mean_velocity = (particle_weights[:, None] * particle_cylindrical_velocities).sum(axis=0)
    squared_velocity_dispersion = (
        particle_weights[:, None] * (particle_cylindrical_velocities - mean_velocity) ** 2
    ).sum(axis=0)

    return np.sqrt(squared_velocity_dispersion)

def get_cylindrical_velocity_dispersion_vector_mass_weighted(
    particle_masses: unyt.unyt_array,
    particle_cylindrical_velocities: unyt.unyt_array,
) -> unyt.unyt_array:
    """
    Compute the mass-weighted velocity dispersion along the radial, azimuthal and vertical
    directions for the input particles.

    Parameters:
     - particle_masses: unyt.unyt_array
       Mass of each particle.
     - particle_cylindrical_velocities: unyt.unyt_array
       Velocity of each particle in a cylindrical coordinate system.

    Returns a 3 element vector containing [sigma_r, sigma_phi, sigma_z].
    """
    mass_weights = particle_masses / particle_masses.sum()
    return get_weighted_cylindrical_velocity_dispersion_vector(mass_weights, particle_cylindrical_velocities)

def get_cylindrical_velocity_dispersion_vector_luminosity_weighted(
    particle_luminosities: unyt.unyt_array,
    particle_cylindrical_velocities: unyt.unyt_array,
) -> unyt.unyt_array:
    """
    Compute the luminosity-weighted velocity dispersion along the radial, azimuthal and vertical
    directions for the input particles. We return a vector for each luminosity band.

    Parameters:
     - particle_luminosities: unyt.unyt_array
       Luminosity of each particle in different luminosity bands.
     - particle_cylindrical_velocities: unyt.unyt_array
       Velocity of each particle in a cylindrical coordinate system, which varies between different
       luminosity bands.

    Returns a 3 element vector for each luminosity band, which contains [sigma_r, sigma_phi, sigma_z].
    The velocity dispersion vectors for each band are appended to the same vector, hence the shape is
    (number_luminosity_bands, 3).
    """

    number_luminosity_bands = particle_luminosities.shape[1]
    velocity_dispersion_vectors = np.zeros((number_luminosity_bands.shape[0], 3)) * particle_cylindrical_velocities.units

    for i_band, (particle_luminosities_i_band, particle_cylindrical_velocities_i_band) in enumerate(zip(particle_luminosities.T, particle_cylindrical_velocities.T)):
        luminosity_weights = particle_luminosities_i_band / particle_luminosities_i_band.sum()
        velocity_dispersion_vectors[i_band] = get_weighted_cylindrical_velocity_dispersion_vector(luminosity_weights, particle_cylindrical_velocities_i_band)

    return velocity_dispersion_vectors

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
    return (mass[:, None] * np.cross(prel, vrel)).sum(axis=0)


def get_angular_momentum_and_kappa_corot_weighted(
    particle_masses: unyt.unyt_array,
    particle_positions: unyt.unyt_array,
    particle_velocities: unyt.unyt_array,
    particle_weights: Union[None, unyt.unyt_array] = None,
    reference_position: Union[None, unyt.unyt_array] = None,
    reference_velocity: Union[None, unyt.unyt_array] = None,
    do_counterrot_mass: bool = False,
    do_counterrot_weight: bool = False,
) -> Union[
    Tuple[unyt.unyt_array, unyt.unyt_array],
    Tuple[unyt.unyt_array, unyt.unyt_array, unyt.unyt_array],
    Tuple[unyt.unyt_array, unyt.unyt_array, unyt.unyt_array, unyt.unyt_array],
]:
    """
    Get the total angular momentum vector and kappa_corot (Correa et al., 2017)
    for the particles with the given masses, positions, velocities and weights.
    It uses the given reference position and velocity as the spatial and velocity
    centres. It optionally returns the total mass and total weight value of
    counterrotating particles.

    We use this function if both kappa_corot and the angular momentum vector are
    requested, as it is more efficient than get_angular_momentum() and
    get_kappa_corot() (if that would even exist).

    Parameters:
     - particle_masses: unyt.unyt_array
       Masses of the particles.
     - particle_positions: unyt.unyt_array
       Position of the particles.
     - particle_velocities: unyt.unyt_array
       Velocities of the particles.
     - particle_weights: unyt.unyt_array or None
       Weights given to each particle when computing the total angular momentum.
       If not provided, this function returns the mass-weighted inertia tensor.
     - reference_position: unyt.unyt_array or None
       Reference position used as centre for the angular momentum calculation.
       particle_positions and reference_position are assumed to use the same reference point upon
       entry into this function. If None, particle_positions is assumed to be already using
       the desired referece point.
     - reference_velocity: unyt.unyt_array or None
       Reference point in velocity space for the angular momentum calculation.
       particle_velocities and reference_velocity are assumed to use the same reference point upon
       entry into this function. If None, particle_velocities is assumed to be already using
       the desired reference point.
     - do_counterrot_mass: bool
       Also compute the counterrotating mass?
     - do_counterrot_weight: bool
       Also compute the counterrotating weight quantity?

    Returns:
     - The weighted total angular momentum vector, or if no weights are provided,
       the usual definition of angular momentum.
     - The ratio of the kinetic energy in counterrotating movement and the total
       kinetic energy, kappa_corot.
     - The total mass of counterrotating particles if do_counterrot_mass == True.
     - The total weight of counterrotating particles if do_counterrot_weight == True.
    """

    if reference_position is None:
        prel = particle_positions
    else:
        prel = particle_positions - reference_position[None, :]
    if reference_velocity is None:
        vrel = particle_velocities
    else:
        vrel = particle_velocities - reference_velocity[None, :]

    # We compute the normal angular momentum because we require it for the
    # kinetic energy calculation in kappa_corot.
    Lpart = particle_masses[:, None] * np.cross(prel, vrel)

    # If we weight the angular momentum, divide by the particle mass so that
    # it L_i = w_i * (r_i x v_i). We also estimate a total weighted mass in order
    # to return an angular momentum with the correct units.
    if particle_weights is not None:
        weights = particle_weights / particle_weights.sum() / particle_masses
        weighted_total_mass = (
            particle_weights / particle_weights.sum() * particle_masses
        ).sum() * len(particle_masses)

    # Normal (mass-weighted) definition of the angular momentum.
    else:
        weights = np.ones(1)
        weighted_total_mass = 1

    # Weighted average of the (specific) angular momentum. We multiply by the
    # weighted total mass to have the correct units for angular momentum, but we
    # are only really interested in its direction.
    Ltot = weighted_total_mass * (weights[:, None] * Lpart).sum(axis=0)
    Lnrm = np.linalg.norm(Ltot)

    # Define the output variables that we will use. Unit registry is the same for
    # all fields, hence why we use particle_masses.units.registry
    kappa_corot = unyt.unyt_array(
        0.0,
        dtype=np.float32,
        units="dimensionless",
        registry=particle_masses.units.registry,
    )

    if do_counterrot_mass:
        M_counterrot = unyt.unyt_array(
            0.0,
            dtype=np.float32,
            units=particle_masses.units,
            registry=particle_masses.units.registry,
        )

    if do_counterrot_weight:
        W_counterrot = unyt.unyt_array(
            0.0,
            dtype=np.float32,
            units=particle_weights.units,
            registry=particle_masses.units.registry,
        )

    if Lnrm > 0.0 * Lnrm.units:

        # Total kinetic energy
        K = 0.5 * (particle_masses[:, None] * vrel**2).sum()

        # Angular momentum of individual particles projected along the weighted
        # angular momentum direction.
        if K > 0.0 * K.units or do_counterrot_mass:
            Ldir = Ltot / Lnrm
            Li = (Lpart * Ldir[None, :]).sum(axis=1)

        if K > 0.0 * K.units:

            # Distance to origin of the coordinate system.
            r2 = prel[:, 0] ** 2 + prel[:, 1] ** 2 + prel[:, 2] ** 2

            # Get distance to axis of rotation for each particle.
            rdotL = (prel * Ldir[None, :]).sum(axis=1)
            Ri2 = r2 - rdotL**2

            # Deal with division by zero (the first particle may be in the centre)
            mask = Ri2 == 0.0
            Ri2[mask] = 1.0 * Ri2.units

            # Get kinetic energy in rotation & co-rotation, and hence kappa_corot.
            Krot = 0.5 * (Li**2 / (particle_masses * Ri2))
            Kcorot = Krot[(~mask) & (Li > 0.0 * Li.units)].sum()
            kappa_corot += Kcorot / K

        if do_counterrot_mass:
            M_counterrot += particle_masses[Li < 0.0 * Li.units].sum()

        if do_counterrot_weight:
            W_counterrot += particle_weights[Li < 0.0 * Li.units].sum()

    if do_counterrot_mass & do_counterrot_weight:
        return Ltot, kappa_corot, M_counterrot, W_counterrot
    elif do_counterrot_weight:
        return Ltot, kappa_corot, W_counterrot
    elif do_counterrot_mass:
        return Ltot, kappa_corot, M_counterrot
    else:
        return Ltot, kappa_corot


def get_angular_momentum_and_kappa_corot_mass_weighted(
    particle_masses: unyt.unyt_array,
    particle_positions: unyt.unyt_array,
    particle_velocities: unyt.unyt_array,
    reference_position: Union[None, unyt.unyt_array] = None,
    reference_velocity: Union[None, unyt.unyt_array] = None,
    do_counterrot_mass: bool = False,
) -> Union[
    Tuple[unyt.unyt_array, unyt.unyt_quantity],
    Tuple[unyt.unyt_array, unyt.unyt_quantity, unyt.unyt_quantity],
]:
    """
    Get the total angular momentum vector and kappa_corot (Correa et al., 2017)
    for the particles with the given masses, positions and velocities, and using
    the given reference position and velocity as centre of mass (velocity). It
    optionally returns the total mass of counterrotating particles.

    This function calls get_angular_momentum_and_kappa_corot_weighted without
    weighting particles. See get_angular_momentum_and_kappa_corot_weighted for
    input parameters and outputs.
    """
    return get_angular_momentum_and_kappa_corot_weighted(
        particle_masses=particle_masses,
        particle_positions=particle_positions,
        particle_velocities=particle_velocities,
        reference_position=reference_position,
        reference_velocity=reference_velocity,
        do_counterrot_mass=do_counterrot_mass,
    )


def get_angular_momentum_and_kappa_corot_luminosity_weighted(
    particle_masses: unyt.unyt_array,
    particle_positions: unyt.unyt_array,
    particle_velocities: unyt.unyt_array,
    particle_luminosities: unyt.unyt_array,
    reference_position: Union[None, unyt.unyt_array] = None,
    reference_velocity: Union[None, unyt.unyt_array] = None,
    do_counterrot_mass: bool = False,
    do_counterrot_luminosity: bool = False,
) -> Union[
    Tuple[unyt.unyt_array, unyt.unyt_array],
    Tuple[unyt.unyt_array, unyt.unyt_array, unyt.unyt_array],
    Tuple[unyt.unyt_array, unyt.unyt_array, unyt.unyt_array, unyt.unyt_array],
]:
    """
    Get the total angular momentum vector and kappa_corot (Correa et al., 2017)
    for the particles with the given masses, positions, velocities and luminosities,
    and using the given reference position and velocity as the spatial and velocity
    centres. It optionally returns the total mass and total luminosity of counterrotating
    particles.

    This function calls get_angular_momentum_and_kappa_corot_weighted and weights
    particles by their luminosity in a given band. See
    get_angular_momentum_and_kappa_corot_weighted for input parameters and outputs.
    """

    number_luminosity_bands = particle_luminosities.shape[1]

    # Create output arrays depending on what we have requested.
    Ltot = unyt.unyt_array(
        np.zeros(3 * number_luminosity_bands),
        dtype=np.float32,
        units=particle_masses.units
        * particle_positions.units
        * particle_velocities.units,
        registry=particle_masses.units.registry,
    )
    kappa_corot = unyt.unyt_array(
        np.zeros(number_luminosity_bands),
        dtype=np.float32,
        units="dimensionless",
        registry=particle_masses.units.registry,
    )
    if do_counterrot_mass:
        M_counterrot = unyt.unyt_array(
            np.zeros(number_luminosity_bands),
            dtype=np.float32,
            units=particle_masses.units,
            registry=particle_masses.units.registry,
        )
    if do_counterrot_luminosity:
        L_counterrot = unyt.unyt_array(
            np.zeros(number_luminosity_bands),
            dtype=np.float32,
            units=particle_luminosities.units,
            registry=particle_luminosities.units.registry,
        )

    for i_band, particle_luminosities_i_band in enumerate(particle_luminosities.T):
        output = get_angular_momentum_and_kappa_corot_weighted(
            particle_masses=particle_masses,
            particle_positions=particle_positions,
            particle_velocities=particle_velocities,
            particle_weights=particle_luminosities_i_band,
            reference_position=reference_position,
            reference_velocity=reference_velocity,
            do_counterrot_mass=do_counterrot_mass,
            do_counterrot_weight=do_counterrot_luminosity,
        )

        # These entries are always in the same order.
        Ltot[3 * i_band : 3 * (i_band + 1)] = output[0]
        kappa_corot[i_band] = output[1]

        # Need to handle different combinations of output requests.
        if do_counterrot_mass & do_counterrot_luminosity:
            M_counterrot[i_band] = output[2]
            L_counterrot[i_band] = output[3]
            continue
        elif do_counterrot_mass:
            M_counterrot[i_band] = output[2]
            continue
        elif do_counterrot_luminosity:
            L_counterrot[i_band] = output[2]
            continue

    if do_counterrot_mass & do_counterrot_luminosity:
        return Ltot, kappa_corot, M_counterrot, L_counterrot
    elif do_counterrot_luminosity:
        return Ltot, kappa_corot, L_counterrot
    elif do_counterrot_mass:
        return Ltot, kappa_corot, M_counterrot
    else:
        return Ltot, kappa_corot


def get_vmax(
    mass: unyt.unyt_array, radius: unyt.unyt_array, nskip: int = 0
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
     - nskip: int
       Number of particles to skip

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
    nskip = max(
        nskip, np.argmin(np.isclose(ordered_radius, 0.0 * ordered_radius.units))
    )
    ordered_radius = ordered_radius[nskip:]
    if len(ordered_radius) == 0 or ordered_radius[0] == 0:
        return 0.0 * radius.units, np.sqrt(0.0 * G * mass.units / radius.units)
    cumulative_mass = cumulative_mass[nskip:]
    v_over_G = cumulative_mass / ordered_radius
    imax = np.argmax(v_over_G)
    return ordered_radius[imax], np.sqrt(v_over_G[imax] * G)


if __name__ == "__main__":
    """
    Standalone version.
    """
    pass
