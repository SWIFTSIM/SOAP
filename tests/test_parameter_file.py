#!/bin/env python

import glob
import os

import pytest

from SOAP.core.parameter_file import ParameterFile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every parameter file shipped in the repo, plus the full test config
SHIPPED_PARAMETER_FILES = sorted(
    glob.glob(os.path.join(REPO_ROOT, "parameter_files", "*.yml"))
) + [os.path.join(REPO_ROOT, "tests", "small_volume.yml")]


def make_parameter_file(
    section=None, calculate_missing_properties=True, snipshot=False, filters=None
):
    """
    Build a ParameterFile with a single ApertureProperties section.
    """
    parameters = {
        "calculations": {"calculate_missing_properties": calculate_missing_properties},
    }
    if section is not None:
        parameters["ApertureProperties"] = section
    if filters is not None:
        parameters["filters"] = filters
    return ParameterFile(parameter_dictionary=parameters, snipshot=snipshot)


VARIATIONS = {"exclusive_50_kpc": {"radius_in_kpc": 50.0, "inclusive": False}}

GENERAL_FILTER = {
    "general": {
        "limit": 100,
        "properties": ["BoundSubhalo/NumberOfDarkMatterParticles"],
    }
}


def variation_with_filter(filter_name):
    return {
        "exclusive_50_kpc": {
            "radius_in_kpc": 50.0,
            "inclusive": False,
            "filter": filter_name,
        }
    }


def test_missing_section_computes_nothing():
    pf = make_parameter_file(section=None)
    assert pf.get_halo_type_variations("ApertureProperties") == {}
    assert pf.variation_warnings == []


@pytest.mark.parametrize(
    "section",
    [
        {"properties": {}, "variations": {}},
        {"properties": {"Mtot": False}, "variations": {}},
        {},
    ],
)
def test_no_variations_no_properties_is_silent(section):
    pf = make_parameter_file(section=section)
    assert pf.get_halo_type_variations("ApertureProperties") == {}
    assert pf.variation_warnings == []


@pytest.mark.parametrize(
    "section",
    [
        {"properties": {"Mtot": True}, "variations": {}},
        {"properties": {"Mtot": "general"}},
        {"properties": {"Mtot": {"snapshot": True, "snipshot": False}}},
    ],
)
def test_properties_but_no_variations_warns_and_skips(section):
    pf = make_parameter_file(section=section)
    assert pf.get_halo_type_variations("ApertureProperties") == {}
    assert len(pf.variation_warnings) == 1
    assert "no variations" in pf.variation_warnings[0]


def test_variations_no_properties_with_calculate_missing():
    pf = make_parameter_file(
        section={"properties": {}, "variations": VARIATIONS},
        calculate_missing_properties=True,
    )
    assert list(pf.get_halo_type_variations("ApertureProperties")) == [
        "exclusive_50_kpc"
    ]
    assert pf.variation_warnings == []


@pytest.mark.parametrize(
    "properties",
    [{}, {"Mtot": False}, {"Mtot": {"snapshot": False, "snipshot": False}}],
)
def test_variations_no_properties_without_calculate_missing_warns_and_skips(properties):
    pf = make_parameter_file(
        section={"properties": properties, "variations": VARIATIONS},
        calculate_missing_properties=False,
    )
    assert pf.get_halo_type_variations("ApertureProperties") == {}
    assert len(pf.variation_warnings) == 1
    assert "calculate_missing_properties is False" in pf.variation_warnings[0]
    # The skipped variations should also be cleared from the stored parameters
    assert pf.parameters["ApertureProperties"]["variations"] == {}


def test_variations_and_properties_is_the_normal_case():
    pf = make_parameter_file(
        section={"properties": {"Mtot": True}, "variations": VARIATIONS},
        calculate_missing_properties=False,
    )
    assert list(pf.get_halo_type_variations("ApertureProperties")) == [
        "exclusive_50_kpc"
    ]
    assert pf.variation_warnings == []


def test_snipshot_mode_used_when_deciding_if_properties_enabled():
    # Property is enabled for snapshots only, but we are running a snipshot
    pf = make_parameter_file(
        section={
            "properties": {"Mtot": {"snapshot": True, "snipshot": False}},
            "variations": VARIATIONS,
        },
        calculate_missing_properties=False,
        snipshot=True,
    )
    assert pf.get_halo_type_variations("ApertureProperties") == {}
    assert len(pf.variation_warnings) == 1


def test_empty_variations_section_not_reported_as_invalid(capsys):
    pf = ParameterFile(
        parameter_dictionary={
            "ProjectedApertureProperties": {
                "properties": {"GasMass": True, "NotARealProperty": True},
                "variations": {},
            },
            "SubhaloProperties": {"properties": {"AlsoNotReal": True}},
            "calculations": {"calculate_missing_properties": True},
        }
    )
    pf.get_halo_type_variations("ProjectedApertureProperties")

    # halo_prop_list is empty since the section has no variations
    pf.print_invalid_properties([])
    captured = capsys.readouterr().out

    # Properties under the empty-variations section are unused, not invalid
    assert "ProjectedApertureProperties" not in captured
    assert "NotARealProperty" not in captured
    # Sections without a variations key at all are still validated
    assert "AlsoNotReal" in captured


def test_missing_variations_key_not_reported_as_invalid(capsys):
    # As above, but the section has no 'variations' key at all;
    # get_halo_type_variations should insert an empty one so that
    # print_invalid_properties skips the section
    pf = ParameterFile(
        parameter_dictionary={
            "ProjectedApertureProperties": {
                "properties": {"GasMass": True, "NotARealProperty": True},
            },
            "SubhaloProperties": {"properties": {"AlsoNotReal": True}},
            "calculations": {"calculate_missing_properties": True},
        }
    )
    pf.get_halo_type_variations("ProjectedApertureProperties")
    assert pf.parameters["ProjectedApertureProperties"]["variations"] == {}

    pf.print_invalid_properties([])
    captured = capsys.readouterr().out

    assert "ProjectedApertureProperties" not in captured
    assert "NotARealProperty" not in captured
    assert "AlsoNotReal" in captured


def test_variation_with_defined_filter_is_accepted():
    pf = make_parameter_file(
        section={
            "properties": {"Mtot": True},
            "variations": variation_with_filter("general"),
        },
        filters=GENERAL_FILTER,
    )
    assert list(pf.get_halo_type_variations("ApertureProperties")) == [
        "exclusive_50_kpc"
    ]


def test_variation_with_basic_filter_is_accepted_without_filters_section():
    pf = make_parameter_file(
        section={
            "properties": {"Mtot": True},
            "variations": variation_with_filter("basic"),
        },
    )
    assert list(pf.get_halo_type_variations("ApertureProperties")) == [
        "exclusive_50_kpc"
    ]


def test_variation_without_filter_key_is_accepted_without_filters_section():
    pf = make_parameter_file(
        section={"properties": {"Mtot": True}, "variations": VARIATIONS},
    )
    assert list(pf.get_halo_type_variations("ApertureProperties")) == [
        "exclusive_50_kpc"
    ]


def test_variation_with_undefined_filter_raises():
    pf = make_parameter_file(
        section={
            "properties": {"Mtot": True},
            "variations": variation_with_filter("not_a_filter"),
        },
    )
    with pytest.raises(ValueError, match="not_a_filter"):
        pf.get_halo_type_variations("ApertureProperties")


def test_property_with_defined_filter_is_accepted():
    pf = make_parameter_file(
        section={"properties": {"Mtot": "general"}}, filters=GENERAL_FILTER
    )
    filters = pf.get_property_filters("ApertureProperties", ["Mtot"])
    assert filters["Mtot"] == "general"


def test_property_true_resolves_to_basic_without_filters_section():
    pf = make_parameter_file(section={"properties": {"Mtot": True}})
    filters = pf.get_property_filters("ApertureProperties", ["Mtot"])
    assert filters["Mtot"] == "basic"


def test_property_with_undefined_filter_raises():
    pf = make_parameter_file(section={"properties": {"Mtot": "not_a_filter"}})
    with pytest.raises(ValueError, match="not_a_filter"):
        pf.get_property_filters("ApertureProperties", ["Mtot"])


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", SHIPPED_PARAMETER_FILES)
@pytest.mark.parametrize("snipshot", [False, True])
def test_shipped_parameter_files_pass_schema(path, snipshot):
    ParameterFile(file_name=path, snipshot=snipshot).check_schema()


def valid_schema_dict():
    return {
        "HaloFinder": {"type": "HBTplus", "filename": "hbt"},
        "SubhaloProperties": {"properties": {"TotalMass": True}},
        "ApertureProperties": {"properties": {}, "variations": {}},
        "calculations": {"calculate_missing_properties": True},
    }


def check_schema_of(params):
    ParameterFile(parameter_dictionary=params).check_schema()


def test_valid_schema_dict_passes():
    check_schema_of(valid_schema_dict())


def test_unknown_top_level_section_is_reported():
    params = valid_schema_dict()
    params["ApatureProperties"] = {}
    with pytest.raises(ValueError, match="ApatureProperties"):
        check_schema_of(params)


def test_unknown_key_in_halo_type_section_is_reported():
    params = valid_schema_dict()
    params["ApertureProperties"]["variation"] = {}
    with pytest.raises(ValueError, match=r'unknown key "ApertureProperties/variation"'):
        check_schema_of(params)


def test_unknown_key_in_halo_finder_is_reported():
    params = valid_schema_dict()
    params["HaloFinder"]["filenme"] = "typo"
    with pytest.raises(ValueError, match=r'unknown key "HaloFinder/filenme"'):
        check_schema_of(params)


def test_unknown_calculations_key_is_reported():
    params = valid_schema_dict()
    params["calculations"]["calculate_missing_property"] = True
    with pytest.raises(ValueError, match="calculate_missing_property"):
        check_schema_of(params)


def test_free_form_sections_are_not_checked():
    params = valid_schema_dict()
    params["filters"] = {"my_filter": {"limit": 100}}
    params["aliases"] = {"PartType0/Foo": "PartType0/Bar"}
    check_schema_of(params)


def test_check_schema_reports_all_errors_together():
    params = valid_schema_dict()
    params["Nonsense"] = {}
    params["HaloFinder"]["oops"] = 1
    with pytest.raises(ValueError) as excinfo:
        check_schema_of(params)
    message = str(excinfo.value)
    assert "Nonsense" in message
    assert "oops" in message
