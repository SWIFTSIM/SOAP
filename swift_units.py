#!/bin/env python

import unyt
import unyt.dimensions as dim


def unit_registry_from_snapshot(snap):

    # Read snapshot metadata
    physical_constants_cgs = {
        name: float(value[0])
        for name, value in snap["PhysicalConstants/CGS"].attrs.items()
    }
    cosmology = {
        name: float(value[0]) for name, value in snap["Cosmology"].attrs.items()
    }
    a = unyt.unyt_quantity(cosmology["Scale-factor"])
    h = unyt.unyt_quantity(cosmology["h"])

    # Create a new registry
    reg = unyt.unit_registry.UnitRegistry()

    # Define code and snapshot base units
    for group_name, prefix in (("Units", "snap"), ("InternalCodeUnits", "code")):
        units_cgs = {
            name: float(value[0]) for name, value in snap[group_name].attrs.items()
        }
        unyt.define_unit(
            prefix + "_length",
            units_cgs["Unit length in cgs (U_L)"] * unyt.cm,
            registry=reg,
        )
        unyt.define_unit(
            prefix + "_mass", units_cgs["Unit mass in cgs (U_M)"] * unyt.g, registry=reg
        )
        unyt.define_unit(
            prefix + "_time", units_cgs["Unit time in cgs (U_t)"] * unyt.s, registry=reg
        )
        unyt.define_unit(
            prefix + "_temperature",
            units_cgs["Unit temperature in cgs (U_T)"] * unyt.K,
            registry=reg,
        )
        unyt.define_unit(prefix + "_angle", 1.0 * unyt.rad, registry=reg)
        unyt.define_unit(
            prefix + "_current",
            units_cgs["Unit current in cgs (U_I)"] * unyt.A,
            registry=reg,
        )

    # Add the expansion factor as a dimensionless "unit"
    unyt.define_unit("a", a, dim.dimensionless, registry=reg)
    unyt.define_unit("h", h, dim.dimensionless, registry=reg)

    # Create a new unit system using the snapshot units as base units
    us = unyt.UnitSystem(
        "snap_units",
        unyt.Unit("snap_length", registry=reg),
        unyt.Unit("snap_mass", registry=reg),
        unyt.Unit("snap_time", registry=reg),
        unyt.Unit("snap_temperature", registry=reg),
        unyt.Unit("snap_angle", registry=reg),
        unyt.Unit("snap_current", registry=reg),
        registry=reg,
    )

    # Create a registry using this base unit system
    reg = unyt.unit_registry.UnitRegistry(lut=reg.lut, unit_system=us)

    # Add some units which might be useful for dealing with input halo catalogues
    unyt.define_unit(
        "swift_mpc", 1.0e6 * physical_constants_cgs["parsec"] * unyt.cm, registry=reg
    )
    unyt.define_unit(
        "swift_msun", physical_constants_cgs["solar_mass"] * unyt.g, registry=reg
    )
    unyt.define_unit(
        "newton_G",
        physical_constants_cgs["newton_G"] * unyt.cm ** 3 / unyt.g / unyt.s ** 2,
        registry=reg,
    )

    return reg


def units_from_attributes(attrs, registry):
    """
    Create a unyt.Unit object from dataset attributes

    attrs: the SWIFT dataset attributes dict
    registry: unyt unit registry with a, h and unit system for the snapshot

    Returns a unyt Unit object.
    """
    # Determine unyt unit for this quantity
    u = unyt.dimensionless
    unit_system = registry.unit_system
    base = registry.unit_system.base_units
    for symbol, baseunit in (
        ("I", base[dim.current_mks]),
        ("L", base[dim.length]),
        ("M", base[dim.mass]),
        ("T", base[dim.temperature]),
        ("t", base[dim.time]),
    ):
        unit = unyt.Unit(baseunit, registry=registry)
        exponent = attrs["U_%s exponent" % symbol][0]
        if exponent == 1.0:
            if u is unyt.dimensionless:
                u = unit
            else:
                u = u * unit
        elif exponent != 0.0:
            if u is unyt.dimensionless:
                u = unit ** exponent
            else:
                u = u * (unit ** exponent)

    # Add expansion factor
    a_scale_exponent = attrs["a-scale exponent"][0]
    a_unit = unyt.Unit("a", registry=registry) ** a_scale_exponent
    # Datasets in the FLAMINGO snapshots do not have a "Value stored as physical"
    # attribute, so default to comoving in that case
    physical = attrs.get("Value stored as physical", [0])[0] == 1
    if (a_scale_exponent != 0) and (not physical):
        if u is unyt.dimensionless:
            u = a_unit
        else:
            u = u * a_unit

    # Add h factor
    h_scale_exponent = attrs["h-scale exponent"][0]
    h_unit = unyt.Unit("h", registry=registry) ** h_scale_exponent
    if h_scale_exponent != 0:
        if u is unyt.dimensionless:
            u = h_unit
        else:
            u = u * h_unit

    return unyt.Unit(u, registry=registry)


def attributes_from_units(units, physical, a_exponent):
    """
    Given a unyt.Unit object, generate SWIFT dataset attributes

    units: the Unit object

    Returns a dict with the attributes
    """
    attrs = {}

    # Get CGS conversion factor. Note that this is the conversion to physical units,
    # because unyt multiplies out the dimensionless a factor.
    cgs_factor, offset = units.get_conversion_factor(units.get_cgs_equivalent())

    # Get a exponent
    a_unit = unyt.Unit("a", registry=units.registry)
    a_exponent_in_units = units.expr.as_powers_dict()[a_unit.expr]
    a_val = a_unit.base_value

    # Check a_exponent is consistent
    if a_exponent is None:
        assert physical
    else:
        if physical:
            assert a_exponent_in_units == 0
        else:
            assert float(a_exponent_in_units) == a_exponent

    # Get h exponent
    h_unit = unyt.Unit("h", registry=units.registry)
    h_exponent = units.expr.as_powers_dict()[h_unit.expr]
    h_val = h_unit.base_value
    # Something has gone wrong if we have h factors
    assert h_exponent == 0

    # Find the power associated with each dimension
    powers = units.get_mks_equivalent().dimensions.as_powers_dict()

    # Set the attributes
    attrs["Conversion factor to CGS (not including cosmological corrections)"] = [
        float(cgs_factor / (a_val ** a_exponent_in_units) / (h_val ** h_exponent))
    ]
    attrs["Conversion factor to physical CGS (including cosmological corrections)"] = [
        float(cgs_factor)
    ]
    attrs["U_I exponent"] = [float(powers[unyt.dimensions.current_mks])]
    attrs["U_L exponent"] = [float(powers[unyt.dimensions.length])]
    attrs["U_M exponent"] = [float(powers[unyt.dimensions.mass])]
    attrs["U_T exponent"] = [float(powers[unyt.dimensions.temperature])]
    attrs["U_t exponent"] = [float(powers[unyt.dimensions.time])]
    attrs["h-scale exponent"] = [float(h_exponent)]
    # Note that "a-scale exponent" is set even if the output is physical,
    # or if the quantity can't be converted to comoving
    attrs["a-scale exponent"] = [0.0 if a_exponent is None else a_exponent]
    attrs["Value stored as physical"] = [1 if physical else 0]
    attrs["Property can be converted to comoving"] = [0 if a_exponent is None else 1]

    return attrs
