#! /usr/bin/env python

"""
euler_mass_terms.py

Some utility functions to compute SPH-smoothed quantities, which have been
implemented to compute Euler mass terms.

We put them in a separate file because a lot of functions are needed.

To run, we need numba and healpy installed.
"""

import unyt as u
import numpy as np
from math import sqrt
from numba import jit
from numba.typed import List

# G = unyt.Unit("newton_G", registry=masses.units.registry)
#===============================================================================
# Euler mass integrands
#===============================================================================

def get_integrand_thermal_support(dP_dr, rho):
    return  (-4*np.pi*u.G)**(-1) * dP_dr * rho**(-1)

def get_integrand_rotational_support(vtheta,vphi, R):
    return  (4*np.pi*u.G)**(-1) * (vtheta**2 + vphi**2) / R

def get_integrand_streaming_support(vr, vtheta, vphi, dvr_dr, dvr_dtheta_R, dvr_dphi_R_sin_theta):
    return  (-4*np.pi*u.G)**(-1) * ( (vr * dvr_dr) + ( vtheta*dvr_dtheta_R ) + (vphi * dvr_dphi_R_sin_theta) )

def do_surface_integral(integrand, surface_element):
    return np.nansum(integrand * surface_element, axis=1)

#===============================================================================
# Generation of query points around sphere.
#===============================================================================

def get_radial_bins(inner_radius, outer_radius, number_radial_bins, scale='log'):
    """
    Returns radial bins for the specified range and number of bins.

    Parameters
    ----------
    inner_radius: float
        The inner edge of the first radial bin.
    outer_radius: float
        The outer edge of the last radial bin.
    number_radial_bins: int
        The number of radial bins to generate.
    scale: str, opt
        Whether bins should be spaced linearly or logarithmically (base 10). It
        defaults to a logarithmic spacing.

    Returns
    -------
    bin_centres: np.ndarray
        The centres of each radial bin.
    bin_edges: np.ndarray
        The edges of all radial bins.
    """

    if scale not in ["log", "lin"]:
        raise ValueError("Accepted string values for scale are log or lin.")

    if scale == 'log':
        multiplicative_factor = (outer_radius / inner_radius).to_value()**(1.0 / number_radial_bins)

        # Do not use logspace because we otherwise lose the cosmo array.
        bin_edges   = inner_radius * multiplicative_factor**np.arange(number_radial_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    elif scale == 'lin':
        bin_edges = np.linspace(inner_radius, outer_radius, number_radial_bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, bin_edges

def get_angular_bins(number_angular_bins):
    """
    Generates angular bins from a Healpix distribution.

    Parameters
    ----------
    number_angular_bins: int
        The number of Healpix angular bins to generate. The value should be
        (12 * 2 ** power)

    Returns
    -------
    np.ndarray
        Polar and azimuthal angles of the bins.
    np.ndarray
        Cartesian location of the healpix centres for a unit sphere.

    """

    # We import within the function because the rest of SOAP does not use it.
    import healpy as hp

    # Obtain the power factor of 2 we require to generate the healpix sphere.
    # This also checks whether the requested number of angular bins is possible
    # to obtain.
    nside = hp.npix2nside(number_angular_bins)

    theta, phi = hp.pix2ang(nside, np.arange(number_angular_bins))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.vstack((theta, phi)), np.vstack((x, y, z)).T

#===============================================================================
# Main function
#===============================================================================

def convert1(x, dtype, depth):# found on a github page https://github.com/numba/numba/issues/7727
    if depth==0:
        # We have reached the end of the nesting-depth,
        # so convert x to a Numpy array.
        y = np.asarray(x, dtype=dtype)
    else:
        # Recursively call this function on all elements of x.
        y = [convert1(x_, dtype=dtype, depth=depth-1) for x_ in x]

        # Convert Python list to Numba list.
        y = List(y)

    return y

@jit(nopython=True, fastmath=True)
def compute_qp_sph_with_grad(x, y, z, particle_masses, particle_properties, particle_smoothing_lengths, particle_densities, neighbors, query_point):
    """
    Compute SPH-interpolated quantities at a query point using neighboring particles.

    This function evaluates a set of scalar quantities at a given point in space. It loops over neighboring particles, whose kernel (WC2) overlap with this points in space,
    Parameters
    - SPH interpolation uses the formula:

      Q_interp = Σ_j (m_j / ρ_j) * Q_j * W(|r - r_j|, h_j)

      where the sum is over neighbors and `W` is the kernel function.
    ----------
    x, y, z : array_like
        1D arrays containing the x, y, and z coordinates of all particles.
    particle_masses: np.ndarray
        Masses of the particles whose mass profile we are interested in.
    particle_properties : tuple of .
        Tuple of 1D arrays, each representing a scalar field (e.g., density, pressure, velocity components)
        defined on the particles. Each Q[i][j] corresponds to the value of the i-th field on the j-th particle.
    particle_smoothing_lengths : array_like
        Smoothning lengths of the particles whose mass profile we are interested in.
    particle_densities : ndarray
        Densities of the particles whose mass profile we are interested in.
    neighbors : List (numba list)
        List of indices of neighboring particles that lie within the search radius (maximum particle h).
    query_point : ndarray
        1D array of shape (3,) representing the 3D Cartesian coordinate of the query point where the SPH quantities
        are to be interpolated.

    Returns
    -------
    sum_w_Q: np.array with shape (N_quantities)
        representing Σ_j (m_j / ρ_j) * Q_j * W(|r - r_j|, h_j)
        saved per quantity
    sum_Gw_Q: np.array with shape (N_quantities, 3)
        representing Σ_j (m_j / ρ_j) * Q_j * ∇W(|r - r_j|, h_j)
        saved per quantity, in cartesian system!
    sum_w: np.array with shape (1)
        representing Σ_j (m_j / ρ_j) * W(|r - r_j|, h_j)
    sum_Gw: np.array with shape (3)
        representing Σ_j (m_j / ρ_j) * ∇W(|r - r_j|, h_j)
    """

    # Arrays to hold smoothed properties.
    sum_w_Q  = np.zeros((len(particle_properties)))
    sum_w    = np.zeros((1))

    # Arrays to hold smoothed gradients.
    sum_Gw_Q = np.zeros((len(particle_properties), 3) )
    sum_Gw   = np.zeros((3))

    for particle_index in neighbors:

        # Distance between query point and neighbour.
        dx = query_point[0] - x[particle_index]
        dy = query_point[1] - y[particle_index]
        dz = query_point[2] - z[particle_index]
        r  = sqrt(dx*dx + dy*dy + dz*dz)

        # The kernel will evalutate to zero if distance is larger than smoothing length.
        quantity_prefactor = (particle_masses[particle_index] / particle_densities[particle_index]) * kernel_M5(r, particle_smoothing_lengths[particle_index])
        gradient_prefactor = (particle_masses[particle_index] / particle_densities[particle_index]) * kernel_derivative_M5(r, particle_smoothing_lengths[particle_index])

        sum_w  += quantity_prefactor
        sum_Gw += gradient_prefactor
        for property_index in range(len(particle_properties)):
            sum_w_Q [property_index]    += quantity_prefactor * particle_properties[property_index][particle_index]
            sum_Gw_Q[property_index, :] += gradient_prefactor * particle_properties[property_index][particle_index]

    return sum_w_Q, sum_Gw_Q, sum_w, sum_Gw

def get_sph_quantities_with_grad(tree, particle_coordinates, particle_masses, particle_properties, particle_smoothing_lengths, particle_densities, query_points):
    """
    Compute SPH-interpolated quantities and gradient at a query point using
    neighboring particles.

    Parameters
    ----------
    tree : scipy.spatial.cKDTree
        KDTree built from particle coordinates for fast neighbor searching.
    particle_coordinates: np.ndarray
        Coordinates of the gas particles.
    particle_masses: np.ndarray
        Masses of the gas particles.
    particle_properties: tuple of np.ndarray .
        Tuple of 1D arrays, each representing a scalar property (e.g., density,
        pressure, a given velocity component, etc.) associated to each gas particle.
        The array particle_properties[i][j] corresponds to the value of the i-th
        field on the j-th particle.
    particle_smoothing_lengths : array_like
        Smoothing lengths of the gas particles.
    particle_densities : ndarray
        Densities of the particles whose mass profile we are interested in.
    query_points : ndarray
        2D array of shape (N_query, 3,) representing the 3D Cartesian coordinate
        of every query point where the SPH quantities are to be calculated.

    Returns
    -------
    sum_w_Q  : np.array with shape (N_query, N_quantities)
        Smoothed values of each quantity for every point in space
    SPH_Q_grad_spherical : np.array with shape (N_query, N_quantities, 3)
        Smoothed values of each gradient for each quantity for every point in space

    """

    query_points = np.asarray(query_points, dtype=np.float64)  #Jit did weird. might not need this

    sum_w_Q  = np.zeros((len(query_points), len(particle_properties)), dtype=np.float64)
    sum_Gw_Q = np.zeros((len(query_points), len(particle_properties), 3), dtype=np.float64)
    sum_Gw   = np.zeros((len(query_points), 3), dtype=np.float64)
    sum_w    = np.zeros((len(query_points)), dtype=np.float64)

    x, y, z = particle_coordinates[:, 0], particle_coordinates[:, 1], particle_coordinates[:, 2]
    tmp_neighbors = tree.query_ball_point(query_points, r=np.max(particle_smoothing_lengths),workers=1)

    neighbors_numba = List()
    neighbors_numba = convert1(tmp_neighbors, dtype=np.int64, depth=1) # found on a github page https://github.com/numba/numba/issues/7727
    # print (neighbors_numba)
    for i, neighbors_qp in enumerate(neighbors_numba):
        sum_w_Q[i, :], sum_Gw_Q[i, :, :], sum_w[i], sum_Gw[i,:] = compute_qp_sph_with_grad(x, y, z, particle_masses, particle_properties, particle_smoothing_lengths, particle_densities, neighbors_qp, query_points[i])
    print (sum_w.shape)
    print (sum_w)
    print ((sum_w[:, None, None])**2)
    SPH_Q_grad_car = (sum_Gw_Q * sum_w[:, None, None] -  sum_w_Q[:, :, None] * sum_Gw[:, None, :]) / (sum_w[:, None, None])**2
    sum_w_Q /= sum_w[:, None]

    SPH_Q_grad_spherical = project_to_spherical_components(SPH_Q_grad_car, query_points)

    return sum_w_Q, SPH_Q_grad_spherical

# Taken from Dehnen & Aly (2012), for a quartic spline kernel (used in COLIBRE)
kernel_gamma_M5 = 2.018932
kernel_constant_M5 = 15625.0 * 0.31830988618379067154 / 512.0

@jit(nopython=True, fastmath=True)
def kernel_M5(r: float | np.float32, H: float | np.float32) -> float:
    """
    Kernel implementation of a quartic spline.

    Parameters
    ----------
    r : float or np.float32
        Distance from particle.

    H : float or np.float32
        Kernel width (i.e. radius of compact support of kernel).

    Returns
    -------
    float
        Contribution to density by particle at distance `r`.

    """
    inverse_H = 1.0 / H
    ratio = r * inverse_H

    if ratio > 1.0:
        return 0

    kernel = max((1 - ratio), 0.0)**4 - 5 * max((3/5 - ratio), 0.0)**4 + 10 * max((1/5 - ratio), 0.0)**4
    kernel *= kernel_constant_M5 * inverse_H**3

    return kernel

@jit(nopython=True, fastmath=True)
def kernel_derivative_M5(r: float | np.float32, H: float | np.float32) -> float:
    """
    Kernel implementation of the derivative of a quartic spline.

    Parameters
    ----------
    r : float or np.float32
        Distance from particle.

    H : float or np.float32
        Kernel width (i.e. radius of compact support of kernel).

    Returns
    -------
    float
        Contribution to density by particle at distance `r`.

    """
    inverse_H = 1.0 / H
    ratio = r * inverse_H

    if ratio > 1.0:
        return 0

    kernel = - 4 * max((1 - ratio), 0.0)**3 + 20 * max((3/5 - ratio), 0.0)**3 - 40 * max((1/5 - ratio), 0.0)**3
    kernel *= kernel_constant_M5 * inverse_H**4

    return kernel