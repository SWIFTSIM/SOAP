import numpy as np
import unyt

from SOAP.core.category_filter import CategoryFilter
from SOAP.core.parameter_file import ParameterFile
from SOAP.property_calculation.stellar_age_calculator import StellarAgeCalculator
from SOAP.particle_selection.aperture_properties import (
    ExclusiveSphereProperties,
    InclusiveSphereProperties,
)
from SOAP.particle_selection.projected_aperture_properties import (
    ProjectedApertureProperties,
)
from SOAP.particle_selection.SO_properties import SOProperties
from SOAP.particle_selection.subhalo_properties import SubhaloProperties

from dummy_halo_generator import DummyHaloGenerator

# Aliases needed so that every property in the property table can be calculated
ALIASES = {
    "PartType0/ElementMassFractions": "PartType0/SmoothedElementMassFractions",
    "PartType4/ElementMassFractions": "PartType4/SmoothedElementMassFractions",
    "PartType0/XrayLuminositiesRestframe": "PartType0/XrayLuminositiesRestframe",
    "PartType0/XrayPhotonLuminositiesRestframe": "PartType0/XrayPhotonLuminositiesRestframe",
}


def _build_calculators(dummy_halos, parameters):
    """
    Construct one calculator of each HaloProperty type, with every property
    in the property table enabled.
    """
    rhg_filter = dummy_halos.get_recently_heated_gas_filter()
    stellar_age_calculator = StellarAgeCalculator(dummy_halos.get_cell_grid())
    cold_dense_gas_filter = dummy_halos.get_cold_dense_gas_filter()
    cat_filter = CategoryFilter(dummy_halos.get_filters({"general": 0}))

    calculators = {
        "ExclusiveSphere/50kpc": ExclusiveSphereProperties(
            dummy_halos.get_cell_grid(),
            parameters,
            50.0,
            None,
            rhg_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            cat_filter,
            "basic",
            [50.0],
        ),
        "InclusiveSphere/50kpc": InclusiveSphereProperties(
            dummy_halos.get_cell_grid(),
            parameters,
            50.0,
            None,
            rhg_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            cat_filter,
            "basic",
            [50.0],
        ),
        "ProjectedAperture/50kpc": ProjectedApertureProperties(
            dummy_halos.get_cell_grid(),
            parameters,
            50.0,
            None,
            cat_filter,
            "basic",
            [50.0],
        ),
        "SO/50_kpc": SOProperties(
            dummy_halos.get_cell_grid(),
            parameters,
            rhg_filter,
            cat_filter,
            "basic",
            50.0,
            "physical",
        ),
        "BoundSubhalo": SubhaloProperties(
            dummy_halos.get_cell_grid(),
            parameters,
            rhg_filter,
            stellar_age_calculator,
            cat_filter,
        ),
    }
    return calculators


def test_expected_dataset_names_matches_calculate():
    """
    Check that HaloProperty.expected_dataset_names() returns exactly the set of
    datasets that calculate() writes to halo_result. This is what the restart
    logic in ChunkTask relies on to detect that a pre-existing chunk file
    computed a different set of properties.
    """
    dummy_halos = DummyHaloGenerator(8161)
    parameters = ParameterFile(parameter_dictionary={"aliases": ALIASES})
    dummy_halos.get_cell_grid().snapshot_datasets.setup_aliases(
        parameters.get_aliases()
    )
    for halo_type in (
        "ApertureProperties",
        "SOProperties",
        "ProjectedApertureProperties",
        "SubhaloProperties",
    ):
        parameters.get_halo_type_variations(halo_type, {})

    calculators = _build_calculators(dummy_halos, parameters)

    for i in range(20):
        input_halo, data, _, _, _, particle_numbers = dummy_halos.get_random_halo(
            [10, 100, 1000, 10000]
        )
        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)

        for group_name, calc in calculators.items():
            input_data = {}
            for ptype in calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {
                        dset: data[ptype][dset]
                        for dset in calc.particle_properties[ptype]
                        if dset in data[ptype]
                    }
            # SubhaloProperties has no dependencies on other calculations, so we
            # can start from an empty result and see everything it writes. The
            # other calculators read values (filters, EncloseRadius, ...) that
            # SOAP would have computed earlier, so they get the dummy template.
            # None of the template keys are in those calculators' own groups.
            if group_name == "BoundSubhalo":
                halo_result = {}
                # SubhaloProperties works from the bound particles directly; a
                # zero search radius is what the dedicated unit test uses.
                search_radius = 0.0 * unyt.kpc
            else:
                halo_result = dict(halo_result_template)
                # Generous, so the calculation always completes
                search_radius = 10000.0 * unyt.kpc
            before = set(halo_result)

            calc.calculate(input_halo, search_radius, input_data, halo_result)

            written = {
                name
                for name in halo_result
                if (name == group_name or name.startswith(f"{group_name}/"))
                and name not in before
            }
            assert written == calc.expected_dataset_names(), (
                f"{group_name}: "
                f"missing={sorted(calc.expected_dataset_names() - written)}, "
                f"extra={sorted(written - calc.expected_dataset_names())}"
            )


def test_expected_dataset_names_tracks_property_filters():
    """
    Disabling a property in the parameter file should drop it (and only it) from
    expected_dataset_names(), so that a restart notices the difference.
    """
    dummy_halos = DummyHaloGenerator(8161)
    parameters = ParameterFile(parameter_dictionary={"aliases": ALIASES})
    dummy_halos.get_cell_grid().snapshot_datasets.setup_aliases(
        parameters.get_aliases()
    )
    parameters.get_halo_type_variations("SubhaloProperties", {})
    rhg_filter = dummy_halos.get_recently_heated_gas_filter()
    stellar_age_calculator = StellarAgeCalculator(dummy_halos.get_cell_grid())
    cat_filter = CategoryFilter(dummy_halos.get_filters({"general": 0}))

    full = SubhaloProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        rhg_filter,
        stellar_age_calculator,
        cat_filter,
    )
    full_names = full.expected_dataset_names()

    # Pick an enabled property that isn't a NumberOf* dependency and disable it
    disabled = None
    for prop in full.property_list.values():
        if full.property_filters[prop.name] and not prop.name.startswith("NumberOf"):
            disabled = prop.name
            break
    assert disabled is not None

    all_parameters = dict(parameters.get_parameters())
    all_parameters["SubhaloProperties"]["properties"][disabled] = False
    reduced_parameters = ParameterFile(parameter_dictionary=all_parameters)
    reduced = SubhaloProperties(
        dummy_halos.get_cell_grid(),
        reduced_parameters,
        rhg_filter,
        stellar_age_calculator,
        cat_filter,
    )

    assert full_names - reduced.expected_dataset_names() == {f"BoundSubhalo/{disabled}"}


if __name__ == "__main__":
    test_expected_dataset_names_matches_calculate()
    test_expected_dataset_names_tracks_property_filters()
    print("Tests passed.")
