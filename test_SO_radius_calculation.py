#!/bin/env python

"""
test_SO_radius_calculation.py

Unit test for the SO radius calculation.

We put this in a separate file to avoid cluttering
SO_properties.py even more.
"""

import numpy as np
import unyt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl

from SO_properties import find_SO_radius_and_mass
from halo_properties import SearchRadiusTooSmallError


def test_SO_radius_calculation():
    """
    Unit test find_SO_radius_and_mass().

    We generate 100 random particle distributions and try
    to find the SO radius.

    This produces some figures for visual inspection.
    """

    np.random.seed(62)

    npart_choices = np.array([10, 100, 1000])
    for i in range(50):
        npart = np.random.choice(npart_choices)
        Mpart = 1.0e9 * unyt.Msun

        radius = np.random.exponential(1.0, npart) * unyt.kpc
        rmax = radius.max()

        ordered_radius = np.sort(radius)

        mass = Mpart * (1.0 + 0.2 * (np.random.random(npart) - 0.5))
        # add some (10%) random negative masses to simulate neutrinos
        idx = np.unique(np.random.randint(0, len(mass), int(0.1 * npart)))
        mass[idx] *= -1.0
        cumulative_mass = np.cumsum(mass)

        ipos = np.argmax(ordered_radius > 0.0)
        ordered_radius = ordered_radius[ipos:]
        cumulative_mass = cumulative_mass[ipos:]
        density = cumulative_mass / (4.0 * np.pi / 3.0 * ordered_radius ** 3)

        reference_density = 200.0 * Mpart * npart / (4.0 * np.pi / 3.0 * rmax ** 3)

        try:
            SO_r, SO_mass, SO_volume = find_SO_radius_and_mass(
                ordered_radius, density, cumulative_mass, reference_density
            )
            print(f"{i:03d}: SO_r: {SO_r}, SO_mass: {SO_mass}")
        except SearchRadiusTooSmallError:
            print(f"{i:03d}: Radius too small!")
            SO_r = -1.0 * unyt.kpc
            SO_mass = -1.0 * unyt.Msun

        fig, ax = pl.subplots(2, 2, sharex="col")

        ordered_radius.convert_to_units("kpc")
        density.convert_to_units("g/cm**3")
        cumulative_mass.convert_to_units("Msun")
        reference_density.convert_to_units("g/cm**3")
        SO_r.convert_to_units("kpc")
        SO_mass.convert_to_units("Msun")

        ax[0][0].semilogy(ordered_radius, density, "o-")
        ax[1][0].semilogy(ordered_radius, cumulative_mass, "o-")
        if SO_r >= 0.0 * unyt.kpc:
            rrange = np.linspace(0.0 * unyt.kpc, 2.0 * SO_r, 100)
            Mrange = reference_density * 4.0 * np.pi / 3.0 * rrange ** 3
            rrange.convert_to_units("kpc")
            Mrange.convert_to_units("Msun")
            ax[1][0].semilogy(rrange, Mrange, ":", color="C2")
            icross = np.argmin(np.abs(ordered_radius - SO_r))
            beg = max(0, icross - 10)
            end = min(len(ordered_radius) - 1, icross + 10)
            ax[0][1].semilogy(ordered_radius[beg:end], density[beg:end], "o-")
            ax[1][1].semilogy(ordered_radius[beg:end], cumulative_mass[beg:end], "o-")
            rrange = np.linspace(0.9 * SO_r, 1.1 * SO_r, 100)
            Mrange = reference_density * 4.0 * np.pi / 3.0 * rrange ** 3
            rrange.convert_to_units("kpc")
            Mrange.convert_to_units("Msun")
            ax[1][1].semilogy(rrange, Mrange, ":", color="C2")
        else:
            ax[0][1].semilogy(ordered_radius, density, "o-")
            ax[1][1].semilogy(ordered_radius, cumulative_mass, "o-")

        ax[0][0].axhline(y=reference_density, linestyle="--", color="C1")
        ax[0][1].axhline(y=reference_density, linestyle="--", color="C1")
        if SO_r >= 0.0 * unyt.kpc:
            ax[1][0].axhline(y=SO_mass, linestyle="--", color="C1")
            ax[0][0].axvline(x=SO_r, linestyle="--", color="C1")
            ax[1][0].axvline(x=SO_r, linestyle="--", color="C1")
            ax[1][1].axhline(y=SO_mass, linestyle="--", color="C1")
            ax[0][1].axvline(x=SO_r, linestyle="--", color="C1")
            ax[1][1].axvline(x=SO_r, linestyle="--", color="C1")
            ax[0][1].plot(SO_r, reference_density, "kx")
            ax[1][1].plot(SO_r, SO_mass, "kx")

        ax[0][0].set_ylabel("density")
        ax[1][0].set_ylabel("cumulative mass")
        ax[1][0].set_xlabel("radius")
        ax[1][1].set_xlabel("radius")

        if SO_r >= 0.0 * unyt.kpc:
            ax[0][0].set_title("Success")
        else:
            ax[0][0].set_title("Failure")
            print(f"{i:03d} SO calculation failed")

        pl.tight_layout()
        pl.savefig(f"test_SO_radius_{i:03d}.png", dpi=300)
        fig.clear()
        pl.close()


if __name__ == "__main__":
    """
    Standalone mode. Run the unit test.
    """

    test_SO_radius_calculation()
