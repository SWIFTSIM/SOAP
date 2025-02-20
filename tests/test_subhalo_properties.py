import pytest
import numpy as np
import unyt

from SOAP.core.category_filter import CategoryFilter
from SOAP.core.parameter_file import ParameterFile
from SOAP.property_calculation.stellar_age_calculator import StellarAgeCalculator
from SOAP.particle_selection.subhalo_properties import SubhaloProperties

from dummy_halo_generator import DummyHaloGenerator


def test_subhalo_properties():
    """
    Unit test for the subhalo property calculations.

    We generate 100 random "dummy" halos and feed them to
    SubhaloProperties::calculate(). We check that the returned values
    are present, and have the right units, size and dtype
    """

    # initialise the DummyHaloGenerator with a random seed
    dummy_halos = DummyHaloGenerator(16902)
    cat_filter = CategoryFilter(
        dummy_halos.get_filters(
            {"general": 100, "gas": 100, "dm": 100, "star": 100, "baryon": 100}
        )
    )
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
        "SubhaloProperties",
        {},
    )

    recently_heated_gas_filter = dummy_halos.get_recently_heated_gas_filter()
    stellar_age_calculator = StellarAgeCalculator(dummy_halos.get_cell_grid())

    property_calculator_bound = SubhaloProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        recently_heated_gas_filter,
        stellar_age_calculator,
        cat_filter,
    )
    # generate 100 random halos
    for i in range(100):
        input_halo, data, _, _, _, _ = dummy_halos.get_random_halo(
            [1, 10, 100, 1000, 10000]
        )

        halo_result = {}
        for subhalo_name, prop_calc in [
            ("BoundSubhalo", property_calculator_bound),
        ]:
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, 0.0 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for prop in prop_calc.property_list.values():
                outputname = prop.name
                size = prop.shape
                dtype = prop.dtype
                unit_string = prop.unit
                physical = prop.output_physical
                a_exponent = prop.a_scale_exponent
                full_name = f"{subhalo_name}/{outputname}"
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

    # Now test the calculation for each property individually, to make sure that
    # all properties read all the datasets they require
    all_parameters = parameters.get_parameters()
    for property in all_parameters["SubhaloProperties"]["properties"]:
        print(f"Testing only {property}...")
        single_property = dict(all_parameters)
        for other_property in all_parameters["SubhaloProperties"]["properties"]:
            single_property["SubhaloProperties"]["properties"][other_property] = (
                other_property == property
            ) or other_property.startswith("NumberOf")
        single_parameters = ParameterFile(parameter_dictionary=single_property)
        property_calculator_bound = SubhaloProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            recently_heated_gas_filter,
            stellar_age_calculator,
            cat_filter,
        )
        halo_result = {}
        for subhalo_name, prop_calc in [
            ("BoundSubhalo", property_calculator_bound)
        ]:
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, 0.0 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for prop in prop_calc.property_list.values():
                outputname = prop.name
                if not outputname == property:
                    continue
                size = prop.shape
                dtype = prop.dtype
                unit_string = prop.unit
                full_name = f"{subhalo_name}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string, registry=dummy_halos.unit_registry)
                assert result.units.same_dimensions_as(unit.units)

    dummy_halos.get_cell_grid().snapshot_datasets.print_dataset_log()


if __name__ == "__main__":
    """
    Standalone version of the program: just run the unit test.

    Note that this can also be achieved by running "pytest *.py" in the folder.
    """
    print("Running test_subhalo_properties()...")
    test_subhalo_properties()
    print("Test passed.")
