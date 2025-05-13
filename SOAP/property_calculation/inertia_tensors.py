#! /usr/bin/env python

"""
inertia_tensors.py

Some utility functions to compute inertia tensors for particle spatial 
distributions.

We put them in a separate file to facilitate unit testing.
"""

import numpy as np
from typing import Union, Tuple
import unyt

from SOAP.particle_selection.halo_properties import SearchRadiusTooSmallError

def get_weighted_inertia_tensor(
    particle_weights,
    particle_positions,
    sphere_radius,
    search_radius=None,
    reduced=False,
    max_iterations=20,
    min_particles=20,
):
    """
    Get the inertia tensor of the given particle distribution weighted by a given
    quantity. Computed as:
    I_{ij} = w*x_i*x_j / Wtot.

    Parameters:
     - particle_weights: unyt.unyt_array
       Weight given to each particle.
     - particle_positions: unyt.unyt_array
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

    Returns a flattened representation of the weighted inertia tensor, with the 
    first 3 entries corresponding to the diagonal terms and the rest to the
    off-diagonal terms.
    """

    # Check we have at least "min_particles" particles
    if particle_weights.shape[0] < min_particles:
        return None

    # Remove particles at centre if calculating reduced tensor
    if reduced:
        norm = np.linalg.norm(particle_positions, axis=1) ** 2
        mask = np.logical_not(np.isclose(norm, 0))

        norm = norm[mask]
        particle_weights = particle_weights[mask]
        particle_positions = particle_positions[mask]

    # Set stopping criteria
    tol = 0.0001
    q = 1000

    # Ensure we have consistent units
    R = sphere_radius.to("kpc")
    particle_positions = particle_positions.to("kpc")

    # Start with a sphere of size equal to the initial aperture
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
        p = np.dot(particle_positions, eig_vec) / axis
        r = np.linalg.norm(p, axis=1)

        # We want to skip the calculation if we have less than "min_particles"
        # inside the initial sphere. We do the check here since this is the first
        # time we calculate how many particles are within the sphere.
        if (i_iter == 0) and (np.sum(r <= 1) < min_particles):
            return None
        weight = particle_weights / np.sum(particle_weights[r <= 1])
        weight[r > 1] = 0

        # Check if we have exceeded the search radius. For subhalo_properties we
        # have all the bound particles, and so the search radius doesn't matter
        if (search_radius is not None) and (np.max(R) > search_radius):
            raise SearchRadiusTooSmallError("Inertia tensor required more particles")

        # Calculate inertia tensor
        tensor = weight[:, None, None] * particle_positions[:, :, None] * particle_positions[:, None, :]
        if reduced:
            tensor /= norm[:, None, None]
        tensor = tensor.sum(axis=0)
        eig_val, eig_vec = np.linalg.eigh(tensor.value)

        # Handle cases where there is only one particle after iterating.
        if q == 0:
          tensor.fill(0)
          break

    return np.concatenate([np.diag(tensor), tensor[np.triu_indices(3, 1)]])

def get_inertia_tensor_mass_weighted(
    particle_masses,
    particle_positions,
    sphere_radius,
    search_radius=None,
    reduced=False,
    max_iterations=20,
    min_particles=20,
):
    """
    Get the mass-weighted inertia tensor of the given particle distribution. 
    Computed as:
    I_{ij} = m*x_i*x_j / Mtot.

    Parameters:
     - particle_masses: unyt.unyt_array
       Masses of the particles.
     - particle_positions: unyt.unyt_array
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

    Returns a flattened representation of the mass-weighted inertia tensor, with the 
    first 3 entries corresponding to the diagonal terms and the rest to the
    off-diagonal terms.
    """
    return get_weighted_inertia_tensor(particle_masses,
          particle_positions,
          sphere_radius,
          search_radius,
          reduced,
          max_iterations,
          min_particles)

def get_inertia_tensor_luminosity_weighted(
    particle_luminosities,
    particle_positions,
    sphere_radius,
    search_radius=None,
    reduced=False,
    max_iterations=20,
    min_particles=20,
):
    """
    Get the luminosity-weighted inertia tensors of the given particle distribution
    in each of the available luminosity bands. Computed as:
    I_{ij} = Li * x_i * x_j / Ltot.

    Parameters:
     - particle_luminosities: unyt.unyt_array
       Luminosities of the particles in each of the provided bands.
     - particle_positions: unyt.unyt_array
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

    Returns an array of concatenated flattened luminosity-weighted inertia tensors, 
    with each 6 consecutive  entries corresponding to 3 diagonal and 3 off-diagonal terms
    in a given band.
    """

    number_luminosity_bands = particle_luminosities.shape[1]

    # We need 6 elements per luminosity band (3 diagonal + 3 off-diagonal terms).
    flattened_inertia_tensors = np.zeros(6 * number_luminosity_bands)

    for i_band, particle_luminosities_i_band in enumerate(particle_luminosities.T):
        flattened_inertia_tensor_i_band = get_weighted_inertia_tensor(particle_luminosities_i_band,
                                                              particle_positions,
                                                              sphere_radius,
                                                              search_radius,
                                                              reduced,
                                                              max_iterations,
                                                              min_particles)

        # Not enough particles in the first band, which means not enough particles
        # in the other bands.
        if flattened_inertia_tensor_i_band is None:
          return None

        # Create the array to output here, once we know the units of the inertia tensor.
        if i_band == 0:
          flattened_inertia_tensors = unyt.unyt_array(
            np.zeros(6 * number_luminosity_bands), dtype=np.float32, units=flattened_inertia_tensor_i_band.units, 
            registry=flattened_inertia_tensor_i_band.units.registry)

        flattened_inertia_tensors[6 * i_band : 6 * (i_band + 1)] = flattened_inertia_tensor_i_band

    return flattened_inertia_tensors

def get_weighted_projected_inertia_tensor(
    particle_weights, particle_positions, axis, radius, reduced=False, max_iterations=20, min_particles=20
):
    """
    Takes in the particle distribution projected along a given axis, and calculates the inertia
    tensor using the projected values.

    Unlike get_inertia_tensor, we don't need to check if we have exceeded the search radius. This
    is because all the bound particles are passed to this function.

    Parameters:
     - particle_weights: unyt.unyt_array
       Weight given to each particle.
     - particle_positions: unyt.unyt_array
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
    if particle_weights.shape[0] < min_particles:
        return None

    projected_position = unyt.unyt_array(
        np.zeros((particle_positions.shape[0], 2)), units=particle_positions.units, dtype=particle_positions.dtype
    )
    if axis == 0:
        projected_position[:, 0] = particle_positions[:, 1]
        projected_position[:, 1] = particle_positions[:, 2]
    elif axis == 1:
        projected_position[:, 0] = particle_positions[:, 2]
        projected_position[:, 1] = particle_positions[:, 0]
    elif axis == 2:
        projected_position[:, 0] = particle_positions[:, 0]
        projected_position[:, 1] = particle_positions[:, 1]
    else:
        raise AttributeError(f"Invalid axis: {axis}!")

    # Remove particles at centre if calculating reduced tensor
    if reduced:
        norm = np.linalg.norm(projected_position, axis=1) ** 2
        mask = np.logical_not(np.isclose(norm, 0))

        norm = norm[mask]
        particle_weights = particle_weights[mask]
        projected_position = projected_position[mask]

    # Set stopping criteria
    tol = 0.0001
    q = 1000

    # Ensure we have consistent units
    R = radius.to("kpc")
    projected_position = projected_position.to("kpc")

    # Start with a circle of size equal to the initial aperture
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
        weight = particle_weights / np.sum(particle_weights[r <= 1])
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

        # Handle cases where there is only one particle after iterating.
        if q == 0:
          tensor.fill(0)
          break

    return np.concatenate([np.diag(tensor), [tensor[(0, 1)]]])

def get_projected_inertia_tensor_mass_weighted(
    particle_masses, particle_positions, axis, radius, reduced=False, max_iterations=20, min_particles=20
):
    """
    Takes in the particle distribution projected along a given axis, and calculates the inertia
    tensor using the projected values.

    Unlike get_inertia_tensor, we don't need to check if we have exceeded the search radius. This
    is because all the bound particles are passed to this function.

    Parameters:
     - particle_masses: unyt.unyt_array
       Masses of the particles.
     - particle_positions: unyt.unyt_array
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
    return get_weighted_projected_inertia_tensor(particle_masses, 
                                                 particle_positions, 
                                                 axis, 
                                                 radius, 
                                                 reduced, 
                                                 max_iterations, 
                                                 min_particles)

def get_projected_inertia_tensor_luminosity_weighted(
    particle_luminosities, particle_positions, axis, radius, reduced=False, max_iterations=20, min_particles=20
):
    """
    Takes in the particle distribution projected along a given axis, and calculates the inertia
    tensor using the projected values and weighting it by the fractional contribution of a particle
    to a given luminosity band.

    Unlike get_inertia_tensor, we don't need to check if we have exceeded the search radius. This
    is because all the bound particles are passed to this function.

    Parameters:
     - particle_luminosities: unyt.unyt_array
       Luminosities of the particles in each of the provided bands.
     - particle_positions: unyt.unyt_array
       Positions of the particles.
     - luminosity: unyt.unyt_array
       Luminosities of the particles.
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

    Returns an array of concatenated flattened inertia tensors, with each 3 consecutive 
    entries corresponding to 2 diagonal and 1 off-diagonal terms.
    """

    number_luminosity_bands = particle_luminosities.shape[1]

    # We need 3 elements per luminosity band (2 diagonal + 1 off-diagonal terms)
    flattened_inertia_tensors = np.zeros(3 * number_luminosity_bands)

    for i_band, particle_luminosities_i_band in enumerate(particle_luminosities.T):
        flattened_inertia_tensor_i_band = get_weighted_projected_inertia_tensor(particle_luminosities_i_band,
                                                              particle_positions,
                                                              axis, 
                                                              radius,
                                                              reduced,
                                                              max_iterations,
                                                              min_particles)

        # Not enough particles in the first band, which means not enough particles
        # in the other bands.
        if flattened_inertia_tensor_i_band is None:
          return None

        # Create the array to output here, once we know the units of the inertia tensor.
        if i_band == 0:
          flattened_inertia_tensors = unyt.unyt_array(
            np.zeros(3 * number_luminosity_bands), dtype=np.float32, units=flattened_inertia_tensor_i_band.units, 
            registry=flattened_inertia_tensor_i_band.units.registry)

        flattened_inertia_tensors[3 * i_band : 3 * (i_band + 1)] = flattened_inertia_tensor_i_band

    return flattened_inertia_tensors

if __name__ == "__main__":
    """
    Standalone version. TODO: add test to check if inertia tensor computation works.
    """
    pass