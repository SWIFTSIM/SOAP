import pytest
import numpy as np
import unyt

from SOAP.core.category_filter import CategoryFilter
from SOAP.core.parameter_file import ParameterFile
from SOAP.particle_selection.projected_aperture_properties import (
    ProjectedApertureProperties,
)

from dummy_halo_generator import DummyHaloGenerator


def test_projected_aperture_properties():
    """
    Unit test for the projected aperture calculation.

    Generates 100 random halos and passes them on to
    ProjectedApertureProperties::calculate().
    Tests that all expected return values are computed and have the right size,
    dtype and units.
    """

    import pytest
    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(127)
    category_filter = CategoryFilter(dummy_halos.get_filters({"general": 100}))
    parameters = ParameterFile(
        parameter_dictionary={
            "aliases": {
                "PartType0/ElementMassFractions": "PartType0/SmoothedElementMassFractions",
                "PartType4/ElementMassFractions": "PartType4/SmoothedElementMassFractions",
            }
        }
    )
    dummy_halos.get_cell_grid().snapshot_datasets.setup_aliases(
        parameters.get_aliases()
    )
    parameters.get_halo_type_variations(
        "ProjectedApertureProperties", {"30_kpc": {"radius_in_kpc": 30.0}}
    )

    pc_projected = ProjectedApertureProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        30.0,
        None,
        category_filter,
        "basic",
        [30.0],
    )

    # Create a filter that no halos will satisfy
    fail_filter = CategoryFilter(dummy_halos.get_filters({"general": 10000000}))
    pc_filter_test = ProjectedApertureProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        30.0,
        None,
        fail_filter,
        "general",
        [30.0],
    )

    for i in range(100):
        input_halo, data, _, _, _, particle_numbers = dummy_halos.get_random_halo(
            [1, 10, 100, 1000, 10000]
        )
        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)

        for pc_name, pc_calc in [
            ("ProjectedAperture", pc_projected),
            ("filter_test", pc_filter_test),
        ]:
            input_data = {}
            for ptype in pc_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in pc_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()

            halo_result = dict(halo_result_template)
            pc_calc.calculate(input_halo, 50 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            for proj in ["projx", "projy", "projz"]:
                for prop in pc_calc.property_list.values():
                    outputname = prop.name
                    size = prop.shape
                    dtype = prop.dtype
                    unit_string = prop.unit
                    full_name = f"ProjectedAperture/30kpc/{proj}/{outputname}"
                    assert full_name in halo_result
                    result = halo_result[full_name][0]
                    assert (len(result.shape) == 0 and size == 1) or result.shape[
                        0
                    ] == size
                    assert result.dtype == dtype
                    unit = unyt.Unit(unit_string, registry=dummy_halos.unit_registry)
                    assert result.units.same_dimensions_as(unit.units)

            # Check properties were not calculated for filtered halos
            if pc_name == "filter_test":
                for proj in ["projx", "projy", "projz"]:
                    for prop in pc_calc.property_list.values():
                        outputname = prop.name
                        size = prop.shape
                        full_name = f"ProjectedAperture/30kpc/{proj}/{outputname}"
                        assert np.all(halo_result[full_name][0].value == np.zeros(size))

    # Now test the calculation for each property individually, to make sure that
    # all properties read all the datasets they require
    all_parameters = parameters.get_parameters()
    for property in all_parameters["ProjectedApertureProperties"]["properties"]:
        print(f"Testing only {property}...")
        single_property = dict(all_parameters)
        for other_property in all_parameters["ProjectedApertureProperties"][
            "properties"
        ]:
            single_property["ProjectedApertureProperties"]["properties"][
                other_property
            ] = (other_property == property) or other_property.startswith("NumberOf")
        single_parameters = ParameterFile(parameter_dictionary=single_property)

        property_calculator = ProjectedApertureProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            30.0,
            None,
            category_filter,
            "basic",
            [30.0],
        )

        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)

        input_data = {}
        for ptype in property_calculator.particle_properties:
            if ptype in data:
                input_data[ptype] = {}
                for dset in property_calculator.particle_properties[ptype]:
                    input_data[ptype][dset] = data[ptype][dset]
        input_halo_copy = input_halo.copy()
        input_data_copy = input_data.copy()
        halo_result = dict(halo_result_template)
        property_calculator.calculate(
            input_halo, 50 * unyt.kpc, input_data, halo_result
        )
        assert input_halo == input_halo_copy
        assert input_data == input_data_copy

        for proj in ["projx", "projy", "projz"]:
            for prop in property_calculator.property_list:
                outputname = prop[1]
                if not outputname == property:
                    continue
                size = prop.size
                dtype = prop.dtype
                unit_string = prop.unit
                physical = prop.output_physical
                a_exponent = prop.a_scale_exponent
                full_name = f"ProjectedAperture/30kpc/{proj}/{outputname}"
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


if __name__ == "__main__":
    """
    Standalone mode: simply run the unit test.

    Note that this can also be achieved by running
    python3 -m pytest *.py
    in the main folder.
    """

    print("Calling test_projected_aperture_properties()...")
    test_projected_aperture_properties()
    print("Test passed.")
