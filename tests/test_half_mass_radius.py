import numpy as np
import unyt

from SOAP.property_calculation.half_mass_radius import get_half_mass_radius

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
