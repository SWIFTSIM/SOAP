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

from SOAP.particle_selection.halo_properties import SearchRadiusTooSmallError


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

    Lpart = mass[:, None] * np.cross(prel, vrel)
    Ltot = Lpart.sum(axis=0)
    Lnrm = np.linalg.norm(Ltot)

    if do_counterrot_mass:
        M_counterrot = unyt.unyt_array(
            0.0, dtype=np.float32, units=mass.units, registry=mass.units.registry
        )

    if Lnrm > 0.0 * Lnrm.units:
        K = 0.5 * (mass[:, None] * vrel**2).sum()
        if K > 0.0 * K.units or do_counterrot_mass:
            Ldir = Ltot / Lnrm
            Li = (Lpart * Ldir[None, :]).sum(axis=1)
        if K > 0.0 * K.units:
            r2 = prel[:, 0] ** 2 + prel[:, 1] ** 2 + prel[:, 2] ** 2
            rdotL = (prel * Ldir[None, :]).sum(axis=1)
            Ri2 = r2 - rdotL**2
            # deal with division by zero (the first particle is guaranteed to
            # be in the centre)
            mask = Ri2 == 0.0
            Ri2[mask] = 1.0 * Ri2.units
            Krot = 0.5 * (Li**2 / (mass * Ri2))
            Kcorot = Krot[(~mask) & (Li > 0.0 * Li.units)].sum()
            kappa_corot += Kcorot / K

        if do_counterrot_mass:
            M_counterrot += mass[Li < 0.0 * Li.units].sum()

    if do_counterrot_mass:
        return Ltot, kappa_corot, M_counterrot
    else:
        return Ltot, kappa_corot


def get_angular_momentum_and_kappa_corot_luminosity_weighted(
    mass: unyt.unyt_array,
    position: unyt.unyt_array,
    velocity: unyt.unyt_array,
    luminosities: unyt.unyt_array,
    ref_position: Union[None, unyt.unyt_array] = None,
    ref_velocity: Union[None, unyt.unyt_array] = None,
    do_counterrot_mass: bool = False,
    do_counterrot_luminosity: bool = False,
) -> Union[
    Tuple[unyt.unyt_array, unyt.unyt_array],
    Tuple[unyt.unyt_array, unyt.unyt_array, unyt.unyt_array],
    Tuple[unyt.unyt_array, unyt.unyt_array, unyt.unyt_array, unyt.unyt_array],
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
     - luminosities: unyt.unyt_array
       Luminosities of the particles in each of the GAMA bands.
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
     - do_counterrot_luminosity: bool
       Also compute the counterrotating luminosity in each GAMA band?

    Returns:
     - The luminosity-weighted total angular momentum vector for each GAMA band.
     - The ratio of the kinetic energy in counterrotating movement and the total
       kinetic energy, kappa_corot. Provided for each luminosity-weighted definition of the
       angular momentum.
     - The total mass of counterrotating particles for each luminosity-weighted definition of the
       angular momentum. (if do_counterrot_mass == True).
     - The total luminosity of counterrotating particles for each luminosity-weighted definition of the
       angular momentum (if do_counterrot_luminosity == True).
    """

    kappa_corot = unyt.unyt_array(
        unyt.unyt_array(np.zeros(luminosities.shape[-1]), units="Dimensionless"), dtype=np.float32, units="dimensionless", registry=mass.units.registry
    )

    # We use centre of mass velocity and position of the stars to not introduce
    # an offset from the true galaxy centre due to bright star forming clumps.
    if ref_position is None:
        prel = position
    else:
        prel = position - ref_position[None, :]
    if ref_velocity is None:
        vrel = velocity
    else:
        vrel = velocity - ref_velocity[None, :]

    # We compute the normal angular momentum because we require it for the 
    # kinetic energy calculation in kappa_corot.
    Lpart = mass[:, None] * np.cross(prel, vrel) # Shape (number_particles, 3)

    # Assign a weight to each particle based on each of its GAMA luminosity. We divide
    # by the particle mass to have an average based on L * (r x v). 
    particle_weights =  luminosities / luminosities.sum(axis=0) / mass[:, None] # Shape (number_particles, number_luminosity_bands)
    weighted_mass    = (luminosities / luminosities.sum(axis=0) * mass[:, None]).sum(axis=0) * len(mass)

    # Luminosity-weighted average of the (specific) angular momentum. We multiply
    # by the luminosity-weighted total mass to have the correct units for angular momentum,
    # but we are only really interested in its direction.
    Ltot = weighted_mass[:,None] * (particle_weights[:, :, None] * Lpart[:, None, :]).sum(axis=0) # Shape (number_luminosity_bands, 3)
    Lnrm = np.linalg.norm(Ltot,axis=1)                                                            # Shape (number_luminosity_bands, )

    if do_counterrot_mass:
        M_counterrot = unyt.unyt_array(
            np.zeros(luminosities.shape[-1]), dtype=np.float32, units=mass.units, registry=mass.units.registry
        )

    if do_counterrot_luminosity:
        L_counterrot = unyt.unyt_array(
            np.zeros(luminosities.shape[-1]), dtype=np.float32, units=luminosities.units, registry=luminosities.units.registry
        )

    if np.any(Lnrm > 0.0 * Lnrm.units):
        K = 0.5 * (mass[:, None] * vrel**2).sum()
        if K > 0.0 * K.units or do_counterrot_mass or do_counterrot_luminosity:
            Ldir = Ltot / Lnrm[:,None]                # Shape (number_particles, number_luminosity_bands, 3)
            Li   = (Lpart[:,None] * Ldir).sum(axis=2) # Shape (number_particles, number_luminosity_bands)
        if K > 0.0 * K.units:
            r2 = prel[:, 0] ** 2 + prel[:, 1] ** 2 + prel[:, 2] ** 2 # Shape (number_particles, )
            rdotL = (Ldir * prel[:,None]).sum(axis=2)                # Shape (number_particles, number_luminosity_bands)
            Ri2 = r2[:,None] - rdotL**2                              # Shape (number_particles, number_luminosity_bands)
            
            # Deal with division by zero, as the first particle may be at the centre.
            mask = Ri2 == 0.0
            Ri2[mask] = 1.0 * Ri2.units
            Krot = 0.5 * (Li**2 / (mass[:,None] * Ri2))

            # We create an array of shape (number_particles, number_luminosity_bands) rather
            # than directly indexing to preserve the 2D nature of the array. 
            # Counterrotating stars make no contribution.
            Kcorot = np.where((~mask) & (Li > 0.0 * Li.units), Krot, 0 * Krot.units) 
            kappa_corot = Kcorot.sum(axis=0) / K.sum(axis=0)

        if do_counterrot_mass:
            M_counterrot = (mass[:,None] * (Li < 0.0 * Li.units)).sum(axis=0)

        # No need to create new axis for luminosities as it has the correct shape
        # already.
        if do_counterrot_luminosity:
            L_counterrot = (luminosities * (Li < 0.0 * Li.units)).sum(axis=0)


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


def get_inertia_tensor(
    mass,
    position,
    sphere_radius,
    search_radius=None,
    reduced=False,
    max_iterations=20,
    min_particles=20,
):
    """
    Get the inertia tensor of the given particle distribution, computed as
    I_{ij} = m*x_i*x_j / Mtot.

    Parameters:
     - mass: unyt.unyt_array
       Masses of the particles.
     - position: unyt.unyt_array
       Positions of the particles.
     - sphere_radius: unyt.unyt_quantity
       Use all particles within a sphere of this size for the calculation
     - search_radius: unyt.unyt_quantity
       Radius of the region of the simulation for which we have particle data
       This function throws a SearchRadiusTooSmallError if we need particles outside
       of this region.
     - reduced: bool
       Whether to calculate the reduced inertia tensor
     - max_iterations: int
       The maximum number of iterations to repeat the inertia tensor calculation
     - min_particles: int
       The number of particles required within the initial sphere. The inertia tensor
       is not computed if this threshold is not met.

    Returns the inertia tensor.
    """

    # Check we have at least "min_particles" particles
    if mass.shape[0] < min_particles:
        return None

    # Remove particles at centre if calculating reduced tensor
    if reduced:
        norm = np.linalg.norm(position, axis=1) ** 2
        mask = np.logical_not(np.isclose(norm, 0))
        position = position[mask]
        mass = mass[mask]
        norm = norm[mask]

    # Set stopping criteria
    tol = 0.0001
    q = 1000

    # Ensure we have consistent units
    R = sphere_radius.to("kpc")
    position = position.to("kpc")

    # Start with a sphere
    eig_val = [1, 1, 1]
    eig_vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    for i_iter in range(max_iterations):
        # Calculate shape
        old_q = q
        q = np.sqrt(eig_val[1] / eig_val[2])
        s = np.sqrt(eig_val[0] / eig_val[2])
        p = np.sqrt(eig_val[0] / eig_val[1])

        # Break if converged
        if abs((old_q - q) / q) < tol:
            break

        # Calculate ellipsoid, determine which particles are inside
        axis = R * np.array(
            [1 * np.cbrt(s * p), 1 * np.cbrt(q / p), 1 / np.cbrt(q * s)]
        )
        p = np.dot(position, eig_vec) / axis
        r = np.linalg.norm(p, axis=1)
        # We want to skip the calculation if we have less than "min_particles"
        # inside the initial sphere. We do the check here since this is the first
        # time we calculate how many particles are within the sphere.
        if (i_iter == 0) and (np.sum(r <= 1) < min_particles):
            return None
        weight = mass / np.sum(mass[r <= 1])
        weight[r > 1] = 0

        # Check if we have exceeded the search radius. For subhalo_properties we
        # have all the bound particles, and so the search radius doesn't matter
        if (search_radius is not None) and (np.max(R) > search_radius):
            raise SearchRadiusTooSmallError("Inertia tensor required more particles")

        # Calculate inertia tensor
        tensor = weight[:, None, None] * position[:, :, None] * position[:, None, :]
        if reduced:
            tensor /= norm[:, None, None]
        tensor = tensor.sum(axis=0)
        eig_val, eig_vec = np.linalg.eigh(tensor.value)

    return np.concatenate([np.diag(tensor), tensor[np.triu_indices(3, 1)]])

def get_inertia_tensor_luminosity_weighted(
    mass,
    position,
    luminosity,
    sphere_radius,
    search_radius=None,
    reduced=False,
    max_iterations=20,
    min_particles=20,
):
    """
    Get the inertia tensor of the given particle distribution weighted by the 
    luminosity of individual particles. Computed as:
    I_{ij} = Li * x_i * x_j / Ltot.

    Parameters:
     - mass: unyt.unyt_array
       Masses of the particles.
     - position: unyt.unyt_array
       Positions of the particles.
     - luminosity: unyt.unyt_array
       Luminosities of the particles in each of the provided bands.
     - sphere_radius: unyt.unyt_quantity
       Use all particles within a sphere of this size for the calculation
     - search_radius: unyt.unyt_quantity
       Radius of the region of the simulation for which we have particle data
       This function throws a SearchRadiusTooSmallError if we need particles outside
       of this region.
     - reduced: bool
       Whether to calculate the reduced inertia tensor
     - max_iterations: int
       The maximum number of iterations to repeat the inertia tensor calculation
     - min_particles: int
       The number of particles required within the initial sphere. The inertia tensor
       is not computed if this threshold is not met.

    Returns the inertia tensor for each of the luminosity bands.
    """

    # Check we have at least "min_particles" particles
    if mass.shape[0] < min_particles:
        return None

    # Remove particles at centre if calculating reduced tensor
    if reduced:
        norm = np.linalg.norm(position, axis=1) ** 2
        mask = np.logical_not(np.isclose(norm, 0))
        position = position[mask]
        mass = mass[mask]
        luminosity = luminosity[mask]
        norm = norm[mask]

    # Set stopping criteria.
    tol = 0.0001
    q = 1000
    is_converged = np.zeros(luminosity.shape[1], dtype=bool)

    # Ensure we have consistent units
    R = sphere_radius.to("kpc")
    position = position.to("kpc")

    # Start with a sphere for each luminosity band.
    eig_val = np.ones((luminosity.shape[1],3))
    eig_vec = np.repeat(np.diag(np.ones(3))[np.newaxis, :, :], luminosity.shape[1], axis=0)

    for i_iter in range(max_iterations):
        # Calculate shape for each luminosity band
        old_q = q
        q = np.sqrt(eig_val[:,1] / eig_val[:,2])
        s = np.sqrt(eig_val[:,0] / eig_val[:,2])
        p = np.sqrt(eig_val[:,0] / eig_val[:,1])

        # Identify bands with converged results. The calculations will only be
        # done for bands that are not yet converged. If all are converged, we are 
        # done.
        is_converged[((old_q - q) / q) < tol] = 1
        if ~is_converged.sum() == 0:
          break

        # Calculate ellipsoid per band, and determine which particles are inside. 
        axis = R * np.array(
            [1 * np.cbrt(s * p), 1 * np.cbrt(q / p), 1 / np.cbrt(q * s)]
        )
        # The transpose is to handle the fact that axis has shape (3, number_luminosity_bands)
        p = np.dot(position, eig_vec) / axis.T
        r = np.linalg.norm(p, axis=2)
        # We want to skip the calculation if we have less than "min_particles"
        # inside the initial sphere. We do the check here since this is the first
        # time we calculate how many particles are within the sphere.
        if (i_iter == 0) and (np.all(np.sum(r <= 1,axis=0) < min_particles)):
            return None

        # Create a luminosity-weight array of the correct shape.
        weight = np.zeros(luminosity.shape)
        weight = np.where(r <= 1, luminosity, weight)
        weight /= weight.sum(axis=0)

        # Check if we have exceeded the search radius. For subhalo_properties we
        # have all the bound particles, and so the search radius doesn't matter
        if (search_radius is not None) and (np.max(R) > search_radius):
            raise SearchRadiusTooSmallError("Inertia tensor required more particles")

        # Calculate inertia tensor
        tensor = weight[:, None, None] * position[:, :, None] * position[:, None, :]
        if reduced:
            tensor /= norm[:, None, None]
        tensor = tensor.sum(axis=0)
        eig_val, eig_vec = np.linalg.eigh(tensor.value)

    return np.concatenate([np.diag(tensor), tensor[np.triu_indices(3, 1)]])

def get_projected_inertia_tensor(
    mass, position, axis, radius, reduced=False, max_iterations=20, min_particles=20
):
    """
    Takes in the particle distribution projected along a given axis, and calculates the inertia
    tensor using the projected values.

    Unlike get_inertia_tensor, we don't need to check if we have exceeded the search radius. This
    is because all the bound particles are passed to this function.

    Parameters:
     - mass: unyt.unyt_array
       Masses of the particles.
     - position: unyt.unyt_array
       Positions of the particles.
     - axis: 0, 1, 2
       Projection axis. Only the coordinates perpendicular to this axis are
       taken into account.
     - radius: unyt.unyt_quantity
       Exclude particles outside this radius for the inertia tensor calculation
     - reduced: bool
       Whether to calculate the reduced inertia tensor
     - max_iterations: int
       The maximum number of iterations to repeat the inertia tensor calculation
     - min_particles: int
       The number of particles required within the initial circle. The inertia tensor
       is not computed if this threshold is not met.

    Returns the inertia tensor.
    """

    # Check we have at least "min_particles" particles
    if mass.shape[0] < min_particles:
        return None

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

    # Remove particles at centre if calculating reduced tensor
    if reduced:
        norm = np.linalg.norm(projected_position, axis=1) ** 2
        mask = np.logical_not(np.isclose(norm, 0))
        projected_position = projected_position[mask]
        mass = mass[mask]
        norm = norm[mask]

    # Set stopping criteria
    tol = 0.0001
    q = 1000

    # Ensure we have consistent units
    R = radius.to("kpc")
    projected_position = projected_position.to("kpc")

    # Start with a circle
    eig_val = [1, 1]
    eig_vec = np.array([[1, 0], [0, 1]])

    for i_iter in range(max_iterations):
        # Calculate shape
        old_q = q
        q = np.sqrt(eig_val[0] / eig_val[1])

        # Break if converged
        if abs((old_q - q) / q) < tol:
            break

        # Calculate ellipse, determine which particles are inside
        axis = R * np.array([1 * np.sqrt(q), 1 / np.sqrt(q)])
        p = np.dot(projected_position, eig_vec) / axis
        r = np.linalg.norm(p, axis=1)
        # We want to skip the calculation if we have less than "min_particles"
        # inside the initial circle. We do the check here since this is the first
        # time we calculate how many particles are within the circle.
        if (i_iter == 0) and (np.sum(r <= 1) < min_particles):
            return None
        weight = mass / np.sum(mass[r <= 1])
        weight[r > 1] = 0

        # Calculate inertia tensor
        tensor = (
            weight[:, None, None]
            * projected_position[:, :, None]
            * projected_position[:, None, :]
        )
        if reduced:
            tensor /= norm[:, None, None]
        tensor = tensor.sum(axis=0)
        eig_val, eig_vec = np.linalg.eigh(tensor.value)

    return np.concatenate([np.diag(tensor), [tensor[(0, 1)]]])


if __name__ == "__main__":
    """
    Standalone version. TODO: add test to check if inertia tensor computation works.
    """
    pass
