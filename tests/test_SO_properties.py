import pytest
import numpy as np
import unyt

from SOAP.core.category_filter import CategoryFilter
from SOAP.core.parameter_file import ParameterFile
from SOAP.particle_selection.halo_properties import SearchRadiusTooSmallError
from SOAP.particle_selection.SO_properties import SOProperties, RadiusMultipleSOProperties

from dummy_halo_generator import DummyHaloGenerator


def test_SO_properties_random_halo():
    """
    Unit test for the SO property calculation.

    We generate 100 random halos and check that the various SO halo
    calculations return the expected results and do not lead to any
    errors.
    """
    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(4251)
    gas_filter = dummy_halos.get_recently_heated_gas_filter()
    cat_filter = CategoryFilter(dummy_halos.get_filters({"general": 100}))
    parameters = ParameterFile(
        parameter_dictionary={
            "aliases": {
                "PartType0/ElementMassFractions": "PartType0/SmoothedElementMassFractions",
                "PartType4/ElementMassFractions": "PartType4/SmoothedElementMassFractions",
                "PartType0/XrayLuminositiesRestframe": "PartType0/XrayLuminositiesRestframe",
                "PartType0/XrayPhotonLuminositiesRestframe": "PartType0/XrayPhotonLuminositiesRestframe",
            }
        }
    )
    dummy_halos.get_cell_grid().snapshot_datasets.setup_aliases(
        parameters.get_aliases()
    )
    parameters.get_halo_type_variations(
        "SOProperties",
        {
            "50_kpc": {"value": 50.0, "type": "physical"},
            "2500_mean": {"value": 2500.0, "type": "mean"},
            "2500_crit": {"value": 2500.0, "type": "crit"},
            "BN98": {"value": 0.0, "type": "BN98"},
            "5xR2500_mean": {"value": 2500.0, "type": "mean", "radius_multiple": 5.0},
        },
    )

    property_calculator_50kpc = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        50.0,
        "physical",
    )
    property_calculator_2500mean = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        2500.0,
        "mean",
    )
    property_calculator_2500crit = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        2500.0,
        "crit",
    )
    property_calculator_BN98 = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        0.0,
        "BN98",
    )
    property_calculator_5x2500mean = RadiusMultipleSOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        2500.0,
        5.0,
        "mean",
    )

    # Create a filter that no halos will satisfy
    fail_filter = CategoryFilter(dummy_halos.get_filters({"general": 10000000}))
    property_calculator_filter_test = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        fail_filter,
        "general",
        200.0,
        "crit",
    )
    property_calculator_filter_test.SO_name = "filter_test"
    property_calculator_filter_test.group_name = "SO/filter_test"

    for i in range(100):
        (
            input_halo,
            data,
            rmax,
            Mtot,
            Npart,
            particle_numbers,
        ) = dummy_halos.get_random_halo([10, 100, 1000, 10000], has_neutrinos=True)
        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)
        rho_ref = Mtot / (4.0 / 3.0 * np.pi * rmax ** 3)

        # force the SO radius to be outside the search sphere and check that
        # we get a SearchRadiusTooSmallError
        property_calculator_2500mean.reference_density = 0.01 * rho_ref
        property_calculator_2500crit.reference_density = 0.01 * rho_ref
        property_calculator_BN98.reference_density = 0.01 * rho_ref
        for prop_calc in [
            property_calculator_2500mean,
            property_calculator_2500crit,
            property_calculator_BN98,
        ]:
            fail = False
            try:
                halo_result = dict(halo_result_template)
                prop_calc.calculate(input_halo, rmax, data, halo_result)
            except SearchRadiusTooSmallError:
                fail = True
            # 1 particle halos don't fail, since we always assume that the first
            # particle is at the centre of potential (which means we exclude it
            # in the SO calculation)
            # non-centrals don't fail, since we do not calculate any SO
            # properties and simply return zeros in this case

            # TODO: This can fail due to how we calculate the SO if the
            # first particle is a neutrino with negative mass. In that case
            # we linearly interpolate the mass of the first non-negative particle
            # outwards.
            # TODO
            # assert (Npart == 1) or input_halo["is_central"] == 0 or fail

        # force the radius multiple to trip over not having computed the
        # required radius
        fail = False
        try:
            halo_result = dict(halo_result_template)
            property_calculator_5x2500mean.calculate(
                input_halo, rmax, data, halo_result
            )
        except RuntimeError:
            fail = True
        assert fail

        # force the radius multiple to trip over the search radius
        fail = False
        try:
            halo_result = dict(halo_result_template)
            halo_result.update(
                {
                    f"SO/2500_mean/{property_calculator_5x2500mean.radius_name}": (
                        0.1 * rmax,
                        "Dummy value.",
                    )
                }
            )
            property_calculator_5x2500mean.calculate(
                input_halo, 0.2 * rmax, data, halo_result
            )
        except SearchRadiusTooSmallError:
            fail = True
        assert fail

        # force the SO radius to be within the search sphere
        property_calculator_2500mean.reference_density = 2.0 * rho_ref
        property_calculator_2500crit.reference_density = 2.0 * rho_ref
        property_calculator_BN98.reference_density = 2.0 * rho_ref

        for SO_name, prop_calc in [
            ("50_kpc", property_calculator_50kpc),
            ("2500_mean", property_calculator_2500mean),
            ("2500_crit", property_calculator_2500crit),
            ("BN98", property_calculator_BN98),
            ("5xR_2500_mean", property_calculator_5x2500mean),
            ("filter_test", property_calculator_filter_test),
        ]:
            halo_result = dict(halo_result_template)
            # make sure the radius multiple is found this time
            if SO_name == "5xR_2500_mean":
                halo_result[
                    f"SO/2500_mean/{property_calculator_5x2500mean.radius_name}"
                ] = (0.1 * rmax, "Dummy value to force correct behaviour")
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            # TODO: remove this
            # Adding Restframe luminosties as they are calculated in halo_tasks
            if "PartType0" in input_data:
                for dset in [
                    "XrayLuminositiesRestframe",
                    "XrayPhotonLuminositiesRestframe",
                ]:
                    input_data["PartType0"][dset] = data["PartType0"][dset]
                    input_data["PartType0"][dset] = data["PartType0"][dset]
                halo_result[
                    f"SO/2500_mean/{property_calculator_5x2500mean.radius_name}"
                ] = (0.1 * rmax, "Dummy value to force correct behaviour")
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, rmax, input_data, halo_result)
            # make sure the calculation does not change the input
            assert input_halo_copy == input_halo
            assert input_data_copy == input_data

            for prop in prop_calc.property_list.values():
                outputname = prop.name
                size = prop.shape
                dtype = prop.dtype
                unit_string = prop.unit
                full_name = f"SO/{SO_name}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string, registry=dummy_halos.unit_registry)
                assert result.units.same_dimensions_as(unit.units)

            # Check properties were not calculated for filtered halos
            if SO_name == "filter_test":
                for prop in prop_calc.property_list.values():
                    outputname = prop.name
                    size = prop.shape
                    full_name = f"SO/{SO_name}/{outputname}"
                    assert np.all(halo_result[full_name][0].value == np.zeros(size))

    # Now test the calculation for each property individually, to make sure that
    # all properties read all the datasets they require
    all_parameters = parameters.get_parameters()
    for property in all_parameters["SOProperties"]["properties"]:
        print(f"Testing only {property}...")
        single_property = dict(all_parameters)
        for other_property in all_parameters["SOProperties"]["properties"]:
            single_property["SOProperties"]["properties"][other_property] = (
                other_property == property
            ) or other_property.startswith("NumberOf")
        single_parameters = ParameterFile(parameter_dictionary=single_property)

        property_calculator_50kpc = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            gas_filter,
            cat_filter,
            "basic",
            50.0,
            "physical",
        )
        property_calculator_2500mean = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            gas_filter,
            cat_filter,
            "basic",
            2500.0,
            "mean",
        )
        property_calculator_2500crit = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            gas_filter,
            cat_filter,
            "basic",
            2500.0,
            "crit",
        )
        property_calculator_BN98 = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            gas_filter,
            cat_filter,
            "basic",
            0.0,
            "BN98",
        )
        property_calculator_5x2500mean = RadiusMultipleSOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            gas_filter,
            cat_filter,
            "basic",
            2500.0,
            5.0,
            "mean",
        )

        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)
        rho_ref = Mtot / (4.0 / 3.0 * np.pi * rmax ** 3)

        # force the SO radius to be within the search sphere
        property_calculator_2500mean.reference_density = 2.0 * rho_ref
        property_calculator_2500crit.reference_density = 2.0 * rho_ref
        property_calculator_BN98.reference_density = 2.0 * rho_ref

        for SO_name, prop_calc in [
            ("50_kpc", property_calculator_50kpc),
            ("2500_mean", property_calculator_2500mean),
            ("2500_crit", property_calculator_2500crit),
            ("BN98", property_calculator_BN98),
            ("5xR_2500_mean", property_calculator_5x2500mean),
        ]:

            halo_result = dict(halo_result_template)
            # make sure the radius multiple is found this time
            if SO_name == "5xR_2500_mean":
                halo_result[
                    f"SO/2500_mean/{property_calculator_5x2500mean.radius_name}"
                ] = (0.1 * rmax, "Dummy value to force correct behaviour")
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            # Adding Restframe luminosties as they are calculated in halo_tasks
            if "PartType0" in input_data:
                for dset in [
                    "XrayLuminositiesRestframe",
                    "XrayPhotonLuminositiesRestframe",
                ]:
                    input_data["PartType0"][dset] = data["PartType0"][dset]
                    input_data["PartType0"][dset] = data["PartType0"][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, rmax, input_data, halo_result)
            # make sure the calculation does not change the input
            assert input_halo_copy == input_halo
            assert input_data_copy == input_data

            for prop in prop_calc.property_list.values():
                outputname = prop.name
                if not outputname == property:
                    continue
                size = prop.shape
                dtype = prop.dtype
                unit_string = prop.unit
                physical = prop.output_physical
                a_exponent = prop.a_scale_exponent
                full_name = f"SO/{SO_name}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string, registry=dummy_halos.unit_registry)
                if not physical:
                    unit = (
                        unit
                        * unyt.Unit("a", registry=dummy_halos.unit_registry)
                        ** a_exponent
                    )
                assert result.units == unit.units

    dummy_halos.get_cell_grid().snapshot_datasets.print_dataset_log()


def calculate_SO_properties_nfw_halo(seed, num_part, c):
    """
    Generates a halo with an NFW profile, and calculates SO properties for it
    """
    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(seed)
    gas_filter = dummy_halos.get_recently_heated_gas_filter()
    cat_filter = CategoryFilter(dummy_halos.get_filters({"general": 100}))
    parameters = ParameterFile(
        parameter_dictionary={
            "aliases": {
                "PartType0/ElementMassFractions": "PartType0/SmoothedElementMassFractions",
                "PartType4/ElementMassFractions": "PartType4/SmoothedElementMassFractions",
                "PartType0/XrayLuminositiesRestframe": "PartType0/XrayLuminositiesRestframe",
                "PartType0/XrayPhotonLuminositiesRestframe": "PartType0/XrayPhotonLuminositiesRestframe",
            }
        }
    )
    dummy_halos.get_cell_grid().snapshot_datasets.setup_aliases(
        parameters.get_aliases()
    )
    parameters.get_halo_type_variations(
        "SOProperties",
        {
            "50_kpc": {"value": 50.0, "type": "physical"},
            "2500_mean": {"value": 2500.0, "type": "mean"},
            "2500_crit": {"value": 2500.0, "type": "crit"},
            "BN98": {"value": 0.0, "type": "BN98"},
            "5xR2500_mean": {"value": 2500.0, "type": "mean", "radius_multiple": 5.0},
        },
    )

    property_calculator_200crit = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        200.0,
        "crit",
    )

    (input_halo, data, rmax, Mtot, Npart, particle_numbers) = dummy_halos.gen_nfw_halo(
        100, c, num_part
    )

    halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)

    property_calculator_200crit.cosmology["nu_density"] *= 0
    property_calculator_200crit.calculate(input_halo, rmax, data, halo_result_template)

    return halo_result_template


def test_concentration_nfw_halo():
    """
    Test if the calculated concentration is close to the input value.
    Only tests halos with for 10000 particles.
    Fails due to noise for small particle numbers.
    """
    n_part = 10000
    for seed in range(10):
        for concentration in [5, 10]:
            halo_result = calculate_SO_properties_nfw_halo(seed, n_part, concentration)
            calculated = halo_result["SO/200_crit/Concentration"][0]
            delta = np.abs(calculated - concentration) / concentration
            assert delta < 0.1


if __name__ == "__main__":
    """
    Standalone mode for running tests.
    """
    print("Calling test_SO_properties_random_halo()...")
    test_SO_properties_random_halo()
    print("Calling test_concentration_nfw_halo()...")
    test_concentration_nfw_halo()
    print("Tests passed.")
