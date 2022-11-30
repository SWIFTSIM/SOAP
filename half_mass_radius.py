#! /usr/bin/env python

"""
half_mass_radius.py

Utility functions to compute the half mass radius of a particle distribution.

We put this in a separate file to facilitate unit testing.
"""

import numpy as np
import unyt


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


def test_get_half_mass_radius():
    """
    Unit test for get_half_mass_radius().

    We generate 1000 random particle distributions and check that the
    half mass radius returned by the function contains less than half
    the particles in mass.
    """
    np.random.seed(203)

    for i in range(1000):
        npart = np.random.choice([1, 10, 100, 1000, 10000])

        radius = np.random.exponential(1.0, npart) * unyt.kpc

        Mpart = 1.0e9 * unyt.Msun
        mass = Mpart * (1.0 + 0.2 * (np.random.random(npart) - 0.5))

        total_mass = mass.sum()

        half_mass_radius = get_half_mass_radius(radius, mass, total_mass)

        mask = radius <= half_mass_radius
        Mtest = mass[mask].sum()
        assert Mtest <= 0.5 * total_mass

    fail = False
    try:
        half_mass_radius = get_half_mass_radius(radius, mass, 2.0 * total_mass)
    except RuntimeError:
        fail = True
    assert fail
