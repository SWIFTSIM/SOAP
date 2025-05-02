#! /usr/bin/env python

"""
half_mass_radius.py

Utility functions to compute the half mass or half light radius of a particle 
distribution.

We put this in a separate file to facilitate unit testing.
"""

import numpy as np
import unyt

def get_half_weight_radius(
    radius: unyt.unyt_array, weights: unyt.unyt_array, total_weight: unyt.unyt_quantity
) -> unyt.unyt_quantity:
    """
    Get the radius that encloses half of the total weight of the given particle distribution.

    We obtain the half weight radius by sorting the particles on radius and then computing
    the cumulative weight profile from this. We then determine in which "bin" the cumulative
    weight profile intersects the target half weight value and obtain the corresponding
    radius from linear interpolation.

    Parameters:
     - radius: unyt.unyt_array
       Radii of the particles.
     - weight: unyt.unyt_array
       Weight of individual particles.
     - total_weight: unyt.unyt_quantity
       Total weight of the particles. Should be weights.sum(). We pass this on as an argument
       because this value might already have been computed before. If it was not, then
       computing it in the function call is still an efficient way to do this.

    Returns the radius that encloses half of the summed weight, defined as the radius 
    at which the cumulative weight profile reaches 0.5 * total_weight.
    """
    if total_weight == 0.0 * total_weight.units or len(weights) < 1:
        return 0.0 * radius.units

    target_weight = 0.5 * total_weight

    isort = np.argsort(radius)
    sorted_radius = radius[isort]

    # Compute sum in double precision to avoid numerical overflow due to
    # weird unit conversions in unyt
    cumulative_weights = weights[isort].cumsum(dtype=np.float64)

    # Consistency check.
    # np.sum() and np.cumsum() use different orders, so we have to allow for
    # some small difference.
    if cumulative_weights[-1] < 0.999 * total_weight:
        raise RuntimeError(
            "Weights sum up to less than the given total weight value:"
            f" cumulative_weights[-1] = {cumulative_weights[-1]},"
            f" total_weights = {total_weight}!"
        )

    # Find the intersection point, abd if that is the first bin, set the lower limits to 0.
    ihalf = np.argmax(cumulative_weights >= target_weight)
    if ihalf == 0:
        rmin = 0.0 * radius.units
        WeightMin = 0.0 * weights.units
    else:
        rmin = sorted_radius[ihalf - 1]
        WeightMin = cumulative_weights[ihalf - 1]
    rmax = sorted_radius[ihalf]
    WeightMax = cumulative_weights[ihalf]

    # Now get the radius by linearly interpolating. If the bin edges coincide 
    # (two particles at exactly the same radius) then we simply take that radius
    if WeightMin == WeightMax:
        half_weight_radius = 0.5 * (rmin + rmax)
    else:
        half_weight_radius = rmin + (target_weight - WeightMin) / (WeightMax - WeightMin) * (rmax - rmin)

    # Consistency check.
    # We cannot use '>=', since equality would happen if half_mass_radius == 0.
    if half_weight_radius > sorted_radius[-1]:
        raise RuntimeError(
            "Half weight radius larger than input radii:"
            f" half_mass_radius = {half_weight_radius},"
            f" sorted_radius[-1] = {sorted_radius[-1]}!"
            f" ihalf = {ihalf}, Npart = {len(radius)},"
            f" target_weight = {target_weight},"
            f" rmin = {rmin}, rmax = {rmax},"
            f" WeightMin = {WeightMin}, WeightMax = {WeightMax},"
            f" sorted_radius = {sorted_radius},"
            f" cumulative_weights = {cumulative_weights}"
        )

    return half_weight_radius

def get_half_mass_radius(
    radius: unyt.unyt_array, mass: unyt.unyt_array, total_mass: unyt.unyt_quantity
) -> unyt.unyt_quantity:
    """
    Get the half mass radius of the given particle distribution.

    We obtain the half mass radius by sorting the particles on radius and then computing
    the cumulative mass profile from this. We then determine in which "bin" the cumulative
    mass profile intersects the target half mass value and obtain the corresponding
    radius from linear interpolation.

    Parameters:
     - radius: unyt.unyt_array
       Radii of the particles.
     - mass: unyt.unyt_array
       Mass of the particles.
     - total_mass: unyt.unyt_quantity
       Total mass of the particles. Should be mass.sum(). We pass this on as an argument
       because this value might already have been computed before. If it was not, then
       computing it in the function call is still an efficient way to do this.

    Returns the half mass radius, defined as the radius at which the cumulative mass profile
    reaches 0.5 * total_mass.
    """
    return get_half_weight_radius(radius, mass, total_mass) 

def get_half_light_radius(
    radius: unyt.unyt_array, mass: unyt.unyt_array, total_mass: unyt.unyt_quantity
) -> unyt.unyt_quantity:
    """
    Get the half mass radius of the given particle distribution.

    We obtain the half mass radius by sorting the particles on radius and then computing
    the cumulative mass profile from this. We then determine in which "bin" the cumulative
    mass profile intersects the target half mass value and obtain the corresponding
    radius from linear interpolation.

    Parameters:
     - radius: unyt.unyt_array
       Radii of the particles.
     - mass: unyt.unyt_array
       Mass of the particles.
     - total_mass: unyt.unyt_quantity
       Total mass of the particles. Should be mass.sum(). We pass this on as an argument
       because this value might already have been computed before. If it was not, then
       computing it in the function call is still an efficient way to do this.

    Returns the half mass radius, defined as the radius at which the cumulative mass profile
    reaches 0.5*total_mass.
    """
    if total_mass == 0.0 * total_mass.units or len(mass) < 1:
        return 0.0 * radius.units

    target_mass = 0.5 * total_mass

    isort = np.argsort(radius)
    sorted_radius = radius[isort]
    # compute sum in double precision to avoid numerical overflow due to
    # weird unit conversions in unyt
    cumulative_mass = mass[isort].cumsum(dtype=np.float64)

    # consistency check
    # np.sum() and np.cumsum() use different orders, so we have to allow for
    # some small difference
    if cumulative_mass[-1] < 0.999 * total_mass:
        raise RuntimeError(
            "Masses sum up to less than the given total mass:"
            f" cumulative_mass[-1] = {cumulative_mass[-1]},"
            f" total_mass = {total_mass}!"
        )

    # find the intersection point
    # if that is the first bin, set the lower limits to 0
    ihalf = np.argmax(cumulative_mass >= target_mass)
    if ihalf == 0:
        rmin = 0.0 * radius.units
        Mmin = 0.0 * mass.units
    else:
        rmin = sorted_radius[ihalf - 1]
        Mmin = cumulative_mass[ihalf - 1]
    rmax = sorted_radius[ihalf]
    Mmax = cumulative_mass[ihalf]

    # now get the radius by linearly interpolating
    # if the bin edges coincide (two particles at exactly the same radius)
    # then we simply take that radius
    if Mmin == Mmax:
        half_mass_radius = 0.5 * (rmin + rmax)
    else:
        half_mass_radius = rmin + (target_mass - Mmin) / (Mmax - Mmin) * (rmax - rmin)

    # consistency check
    # we cannot use '>=', since equality would happen if half_mass_radius==0
    if half_mass_radius > sorted_radius[-1]:
        raise RuntimeError(
            "Half mass radius larger than input radii:"
            f" half_mass_radius = {half_mass_radius},"
            f" sorted_radius[-1] = {sorted_radius[-1]}!"
            f" ihalf = {ihalf}, Npart = {len(radius)},"
            f" target_mass = {target_mass},"
            f" rmin = {rmin}, rmax = {rmax},"
            f" Mmin = {Mmin}, Mmax = {Mmax},"
            f" sorted_radius = {sorted_radius},"
            f" cumulative_mass = {cumulative_mass}"
        )

    return half_mass_radius