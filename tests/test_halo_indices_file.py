#!/bin/env python

import numpy as np
import pytest

from SOAP.core.soap_args import get_halo_indices


def make_parameters(tmp_path, contents, snap_nr=10):
    """
    Write a halo indices file and return the parameters required by
    get_halo_indices
    """
    filename = tmp_path / f"halo_indices_{snap_nr:04d}.txt"
    filename.write_text(contents)
    return {
        "halo_indices": None,
        "halo_indices_file": str(tmp_path / "halo_indices_{snap_nr:04d}.txt"),
        "snap_nr": snap_nr,
    }


def test_no_halo_indices():
    parameters = {"halo_indices": None, "halo_indices_file": None, "snap_nr": 10}
    assert get_halo_indices(parameters) is None


def test_halo_indices_command_line():
    parameters = {
        "halo_indices": [3, 1, 2, 1],
        "halo_indices_file": None,
        "snap_nr": 10,
    }
    assert np.array_equal(get_halo_indices(parameters), [1, 2, 3])


def test_halo_indices_file_single_index(tmp_path):
    parameters = make_parameters(tmp_path, "7\n")
    halo_indices = get_halo_indices(parameters)
    assert halo_indices.shape == (1,)
    assert halo_indices[0] == 7


def test_halo_indices_file_comments_and_blank_lines(tmp_path):
    parameters = make_parameters(tmp_path, "# a comment\n\n1\n2 # another comment\n\n")
    assert np.array_equal(get_halo_indices(parameters), [1, 2])


def test_halo_indices_file_empty(tmp_path):
    parameters = make_parameters(tmp_path, "# no indices here\n")
    with pytest.raises(ValueError, match="empty"):
        get_halo_indices(parameters)


def test_halo_indices_file_multiple_columns(tmp_path):
    parameters = make_parameters(tmp_path, "1 2\n3 4\n")
    with pytest.raises(ValueError, match="one index per line"):
        get_halo_indices(parameters)


def test_halo_indices_file_not_integer(tmp_path):
    parameters = make_parameters(tmp_path, "1\n2.5\n")
    with pytest.raises(ValueError, match="Unable to read"):
        get_halo_indices(parameters)
