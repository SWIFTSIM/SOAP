import pytest
import numpy as np
import unyt

from SOAP.core.category_filter import CategoryFilter
from SOAP.core.parameter_file import ParameterFile
from SOAP.property_calculation.stellar_age_calculator import StellarAgeCalculator
from SOAP.particle_filter.cold_dense_gas_filter import ColdDenseGasFilter
from SOAP.particle_filter.recently_heated_gas_filter import RecentlyHeatedGasFilter
from SOAP.particle_selection.halo_properties import SearchRadiusTooSmallError
from SOAP.particle_selection.aperture_properties import ExclusiveSphereProperties, InclusiveSphereProperties

from dummy_halo_generator import DummyHaloGenerator

def test_aperture_properties():
    """
    Unit test for the aperture property calculations.

    We generate 100 random "dummy" halos and feed them to
    ExclusiveSphereProperties::calculate() and
    InclusiveSphereProperties::calculate(). We check that the returned values
    are present, and have the right units, size and dtype
    """

    # initialise the DummyHaloGenerator with a random seed
    dummy_halos = DummyHaloGenerator(3256)
    recently_heated_filter = dummy_halos.get_recently_heated_gas_filter()
    stellar_age_calculator = StellarAgeCalculator(dummy_halos.get_cell_grid())
    cold_dense_gas_filter = dummy_halos.get_cold_dense_gas_filter()
    cat_filter = CategoryFilter(
        dummy_halos.get_filters(
            {"general": 0, "gas": 0, "dm": 0, "star": 0, "baryon": 0}
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
        "ApertureProperties",
        {
            "exclusive_50_kpc": {"radius_in_kpc": 50.0, "inclusive": False},
            "inclusive_50_kpc": {"radius_in_kpc": 50.0, "inclusive": True},
        },
    )

    pc_exclusive = ExclusiveSphereProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        50.0,
        recently_heated_filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        cat_filter,
        "basic",
        [50.0],
    )
    pc_inclusive = InclusiveSphereProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        50.0,
        recently_heated_filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        cat_filter,
        "basic",
        [50.0],
    )

    # Create a filter that no halos will satisfy
    fail_filter = CategoryFilter(dummy_halos.get_filters({"general": 10000000}))
    pc_filter_test = ExclusiveSphereProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        50.0,
        recently_heated_filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        fail_filter,
        "general",
        [50.0],
    )

    # generate 100 random halos
    for i in range(100):
        input_halo, data, _, _, _, particle_numbers = dummy_halos.get_random_halo(
            [1, 10, 100, 1000, 10000]
        )
        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)

        for pc_name, pc_type, pc_calc in [
            ("ExclusiveSphere", "ExclusiveSphere", pc_exclusive),
            ("InclusiveSphere", "InclusiveSphere", pc_inclusive),
            ("filter_test", "ExclusiveSphere", pc_filter_test),
        ]:
            input_data = {}
            for ptype in pc_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in pc_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()

            # Check halo fails if search radius is too small
            halo_result = dict(halo_result_template)
            if pc_name != "filter_test":
                with pytest.raises(SearchRadiusTooSmallError):
                    pc_calc.calculate(
                        input_halo, 10 * unyt.kpc, input_data, halo_result
                    )
            # Skipped halos shouldn't ever require a larger search radius
            else:
                pc_calc.calculate(input_halo, 10 * unyt.kpc, input_data, halo_result)

            halo_result = dict(halo_result_template)
            pc_calc.calculate(input_halo, 100 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for prop in pc_calc.property_list.values():
                outputname = prop.name
                size = prop.shape
                dtype = prop.dtype
                unit_string = prop.unit
                physical = prop.output_physical
                a_exponent = prop.a_scale_exponent
                full_name = f"{pc_type}/50kpc/{outputname}"
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

            # Check properties were not calculated for filtered halos
            if pc_name == "filter_test":
                for prop in pc_calc.property_list.values():
                    outputname = prop.name
                    size = prop.shape
                    full_name = f"{pc_type}/50kpc/{outputname}"
                    assert np.all(halo_result[full_name][0].value == np.zeros(size))

    # Now test the calculation for each property individually, to make sure that
    # all properties read all the datasets they require
    # we reuse the last random halo for this
    all_parameters = parameters.get_parameters()
    for property in all_parameters["ApertureProperties"]["properties"]:
        print(f"Testing only {property}...")
        single_property = dict(all_parameters)
        for other_property in all_parameters["ApertureProperties"]["properties"]:
            single_property["ApertureProperties"]["properties"][other_property] = (
                other_property == property
            ) or other_property.startswith("NumberOf")
        single_parameters = ParameterFile(parameter_dictionary=single_property)
        pc_exclusive = ExclusiveSphereProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            50.0,
            recently_heated_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            cat_filter,
            "basic",
            [50.0],
        )
        pc_inclusive = InclusiveSphereProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            50.0,
            recently_heated_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            cat_filter,
            "basic",
            [50.0],
        )

        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)

        for pc_type, pc_calc in [
            ("ExclusiveSphere", pc_exclusive),
            ("InclusiveSphere", pc_inclusive),
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
            pc_calc.calculate(input_halo, 100 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for prop in pc_calc.property_list.values():
                outputname = prop.name
                if not outputname == property:
                    continue
                size = prop.shape
                dtype = prop.dtype
                unit_string = prop.unit
                full_name = f"{pc_type}/50kpc/{outputname}"
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

    print("Running test_aperture_properties()...")
    test_aperture_properties()
    print("Test passed.")
