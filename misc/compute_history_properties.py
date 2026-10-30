#!/bin/env python

"""
compute_history_properties.py

This script computes properties which depend on the history of each subhalo,
by following its main progenitor branch through all previous SOAP
catalogues. The main progenitor branch is identified using the HBT TrackId.

The following property types can be computed:

  --bh-maximum: The maximum mass of the most massive black hole along the
    main progenitor branch. Black hole particles can be lost from subhaloes
    (e.g. satellites can lose their black holes due to repositioning, and
    backsplash galaxies may then start regrowing a new one), so this can be
    used to correct for missing black holes.
  --half-mass-formation: The snapshot at which the subhalo formed, defined
    as the earliest snapshot from which TotalMass remains at or above half
    its current value.
  --half-stellar-mass-formation: As above, but using StellarMass.
  --stellar-mass-maximum: The maximum StellarMass along the main progenitor
    branch.

Each argument takes a comma separated list of halo types to compute the
property for. If none of these arguments are passed then all property types
are computed for their default halo types. If any are passed then only those
property types are computed.

A single pass is made over the SOAP catalogues, and an output catalogue is
written for each of the requested target snapshots. The output catalogues
have the same row ordering as the input SOAP catalogues, and can be opened
using swiftsimio.

Usage:

  mpirun -- python -u -m mpi4py misc/compute_history_properties.py \
            --soap-basename SOAP_BASENAME \
            --output-basename OUTPUT_BASENAME \
            --target-snap-nrs SNAP_NR [SNAP_NR ...]

Run "python misc/compute_history_properties.py -h" for a full description
of the arguments, and a list of optional arguments.

"""

import argparse
import datetime
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

# The property types which can be computed. "input" is the SOAP dataset
# the property is computed from, and is used to determine which groups
# the property can be computed for.
property_types = {
    "bh_maximum": {
        "input": "MostMassiveBlackHoleMass",
        "default": ["BoundSubhalo", "ExclusiveSphere"],
    },
    "half_mass_formation": {
        "input": "TotalMass",
        "default": ["BoundSubhalo", "SO"],
    },
    "half_stellar_mass_formation": {
        "input": "StellarMass",
        "default": ["BoundSubhalo", "ExclusiveSphere"],
    },
    "stellar_mass_maximum": {
        "input": "StellarMass",
        "default": ["BoundSubhalo", "ExclusiveSphere"],
    },
}

# Groups in the SOAP catalogues which are copied to the output files
metadata_groups = [
    "Cells",
    "Code",
    "Cosmology",
    "Header",
    "Parameters",
    "PhysicalConstants",
    "SWIFT",
    "Units",
]

# Units for the dimensionless fields we will be saving
unit_attrs = {
    "Conversion factor to CGS (not including cosmological corrections)": [1.0],
    "Conversion factor to physical CGS (including cosmological corrections)": [1.0],
    "U_I exponent": [0.0],
    "U_L exponent": [0.0],
    "U_M exponent": [0.0],
    "U_t exponent": [0.0],
    "U_T exponent": [0.0],
    "a-scale exponent": [0.0],
    "h-scale exponent": [0.0],
    "Property can be converted to comoving": [0],
    "Value stored as physical": [1],
    "Masked": False,
}

# Output datasets. "units_from" is the input dataset to copy unit
# attributes from, or None if the output is dimensionless.
output_properties = {
    "MainBranchMaximumMostMassiveBlackHoleMass": {
        "units_from": "MostMassiveBlackHoleMass",
        "description": (
            "Maximum value of MostMassiveBlackHoleMass along the main progenitor "
            "branch of this subhalo (including the current snapshot). "
            "Zero if no black hole was ever present."
        ),
    },
    "MainBranchMaximumMostMassiveBlackHoleID": {
        "units_from": "MostMassiveBlackHoleID",
        "description": (
            "ID of the most massive black hole at the snapshot where "
            "MainBranchMaximumMostMassiveBlackHoleMass was reached. "
            "Zero if no black hole was ever present."
        ),
    },
    "MainBranchMaximumMostMassiveBlackHoleSnapshot": {
        "units_from": None,
        "description": (
            "Snapshot number where "
            "MainBranchMaximumMostMassiveBlackHoleMass was reached. "
            "-1 if no black hole was ever present."
        ),
    },
    "HalfMassFormationSnapshot": {
        "units_from": None,
        "description": (
            "First snapshot of the final continuous stretch, ending at the "
            "current snapshot, during which TotalMass along the main progenitor "
            "branch is at least half of its current value. "
            "-1 if the current TotalMass is zero."
        ),
    },
    "HalfStellarMassFormationSnapshot": {
        "units_from": None,
        "description": (
            "First snapshot of the final continuous stretch, ending at the "
            "current snapshot, during which StellarMass along the main progenitor "
            "branch is at least half of its current value. "
            "-1 if the current StellarMass is zero."
        ),
    },
    "MainBranchMaximumStellarMass": {
        "units_from": "StellarMass",
        "description": (
            "Maximum value of StellarMass along the main progenitor "
            "branch of this subhalo (including the current snapshot). "
            "Zero if no stars were ever present."
        ),
    },
    "MainBranchMaximumStellarMassSnapshot": {
        "units_from": None,
        "description": (
            "Snapshot number where MainBranchMaximumStellarMass was reached. "
            "-1 if no stars were ever present."
        ),
    },
}


def mpi_print(string, comm_rank):
    if comm_rank == 0:
        print(string)


def find_groups(filename, halo_types, dataset):
    """
    Return the list of groups in a SOAP catalogue which contain
    the specified dataset and are within one of the halo_types, and
    the list of halo_types which did not match any group
    """
    groups = set()
    matched = set()

    def visitor(name, obj):
        if not name.endswith(f"/{dataset}"):
            return
        group = name[: -len(f"/{dataset}")]
        for halo_type in halo_types:
            if (group == halo_type) or group.startswith(halo_type + "/"):
                groups.add(group)
                matched.add(halo_type)

    with h5py.File(filename, "r") as file:
        file.visititems(visitor)
    unmatched = [halo_type for halo_type in halo_types if halo_type not in matched]
    return sorted(groups), unmatched


def read_soap(filename, datasets):
    """
    Read the specified datasets from a SOAP catalogue, distributing the
    subhaloes across ranks. Returns None if the catalogue has no subhaloes.
    """
    if comm_rank == 0:
        with h5py.File(filename, "r") as file:
            n_subhalo = file["Header"].attrs["NumSubhalos_Total"][0]
    else:
        n_subhalo = None
    n_subhalo = comm.bcast(n_subhalo)
    if n_subhalo == 0:
        return None

    mf = phdf5.MultiFile([filename], comm=comm)
    return mf.read(datasets)


def copy_group(ingroup, outgroup):
    """
    Recursively copy a HDF5 group. We don't use h5py's copy method
    since it segfaults for some of the variable length string
    attributes in the SOAP catalogues.
    """
    for k, v in ingroup.attrs.items():
        outgroup.attrs[k] = v
    for name, obj in ingroup.items():
        if isinstance(obj, h5py.Group):
            copy_group(obj, outgroup.create_group(name))
        else:
            dset = outgroup.create_dataset(name, data=obj[()])
            for k, v in obj.attrs.items():
                dset.attrs[k] = v


def write_output(soap_filename, output_filename, groups, input_halos, output):
    """
    Write a minimal SOAP catalogue which can be opened with swiftsimio. The
    metadata groups are copied from the original SOAP catalogue.
    """
    if comm_rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
        with (
            h5py.File(soap_filename, "r") as infile,
            h5py.File(output_filename, "w") as outfile,
        ):
            for name in metadata_groups:
                copy_group(infile[name], outfile.create_group(name))
            # Only list the subhalo types present in this file
            outfile["Header"].attrs["SubhaloTypes"] = np.array(
                ["InputHalos", "InputHalos/HBTplus"] + groups, dtype=object
            )
            outfile.create_group("InputHalos/HBTplus")
            for group in groups:
                outfile.create_group(group)
    comm.barrier()

    with h5py.File(output_filename, "r+", driver="mpio", comm=comm) as outfile:
        for name, arr in input_halos.items():
            phdf5.collective_write(outfile, name, arr, comm=comm)
        for (group, name), arr in output.items():
            phdf5.collective_write(outfile[group], name, arr, comm=comm)
    comm.barrier()

    # Set the attributes of the datasets we have written
    if comm_rank == 0:
        with (
            h5py.File(soap_filename, "r") as infile,
            h5py.File(output_filename, "r+") as outfile,
        ):
            dset_attrs = {}
            for name in input_halos:
                dset_attrs[name] = dict(infile[name].attrs)
            for group, name in output:
                units_from = output_properties[name]["units_from"]
                if units_from is None:
                    attrs = dict(unit_attrs)
                else:
                    attrs = dict(infile[f"{group}/{units_from}"].attrs)
                attrs["Description"] = output_properties[name]["description"]
                dset_attrs[f"{group}/{name}"] = attrs
            for dset_name, attrs in dset_attrs.items():
                # We do not apply any lossy compression
                attrs["Is Compressed"] = False
                attrs["Lossy compression filter"] = "None"
                for k, v in attrs.items():
                    outfile[dset_name].attrs[k] = v
    comm.barrier()


class MaximumTracker:
    """
    Tracks the running maximum of a dataset along the main progenitor
    branch, and the snapshot where it was reached. Optionally also
    tracks the value of a second dataset at that snapshot.
    """

    def __init__(self, n_track, group, dataset, value_name, snap_name, extra=None):
        self.group = group
        self.dataset = dataset
        self.value_name = value_name
        self.snap_name = snap_name
        # extra is a (dataset, output_name, dtype) tuple, or None
        self.extra = extra
        self.max_value = np.zeros(n_track, dtype=np.float32)
        self.max_snap = -np.ones(n_track, dtype=np.int32)
        if extra is not None:
            self.max_extra = np.zeros(n_track, dtype=extra[2])

    def input_datasets(self):
        datasets = [f"{self.group}/{self.dataset}"]
        if self.extra is not None:
            datasets.append(f"{self.group}/{self.extra[0]}")
        return datasets

    def update(self, snap_nr, values, has_match):
        value = values[f"{self.group}/{self.dataset}"]
        new_max = value > self.max_value[has_match]
        is_max = has_match.copy()
        is_max[has_match] = new_max
        self.max_value[is_max] = value[new_max]
        self.max_snap[is_max] = snap_nr
        if self.extra is not None:
            extra = values[f"{self.group}/{self.extra[0]}"]
            self.max_extra[is_max] = extra[new_max]

    def get_output(self, idx):
        output = {
            (self.group, self.value_name): psort.fetch_elements(
                self.max_value, idx, comm=comm
            ),
            (self.group, self.snap_name): psort.fetch_elements(
                self.max_snap, idx, comm=comm
            ),
        }
        if self.extra is not None:
            output[(self.group, self.extra[1])] = psort.fetch_elements(
                self.max_extra, idx, comm=comm
            )
        return output


class HalfMassFormationTracker:
    """
    Tracks the earliest snapshot from which a dataset along the main
    progenitor branch remains at or above half its value at each of the
    target snapshots. Snapshots where the track is not present (including
    snapshots whose SOAP catalogue is missing) are ignored. SO properties
    are only computed for centrals, so for SO groups snapshots where the
    track is a satellite are also ignored.
    """

    def __init__(self, track_ids, group, dataset, snap_name, target_values):
        self.group = group
        self.dataset = dataset
        self.snap_name = snap_name
        self.centrals_only = group.split("/")[0] == "SO"
        # Value of the dataset at each target snapshot, NaN if the
        # track is not present in that target
        self.target_values = target_values
        # Earliest snapshot of the current run of snapshots at or above
        # half the target value, -1 if the last snapshot where the track
        # was present was below. We use int16 to reduce memory usage.
        self.formation_snap = {
            snap_nr: -np.ones(track_ids.shape[0], dtype=np.int16)
            for snap_nr in target_values
        }

    def input_datasets(self):
        datasets = [f"{self.group}/{self.dataset}"]
        if self.centrals_only:
            datasets.append("InputHalos/IsCentral")
        return datasets

    def update(self, snap_nr, values, has_match):
        value = values[f"{self.group}/{self.dataset}"]
        if self.centrals_only:
            # Treat satellites as if they were not present
            is_central = values["InputHalos/IsCentral"] == 1
            value = value[is_central]
            has_match = has_match.copy()
            has_match[has_match] = is_central
        for target_snap_nr, formation_snap in self.formation_snap.items():
            # Snapshots after the target do not contribute
            if snap_nr > target_snap_nr:
                continue
            # Only update tracks which are present in this snapshot
            target_value = self.target_values[target_snap_nr][has_match]
            is_below = value < 0.5 * target_value
            matched_formation_snap = formation_snap[has_match]
            matched_formation_snap[is_below] = -1
            matched_formation_snap[(~is_below) & (matched_formation_snap == -1)] = (
                snap_nr
            )
            formation_snap[has_match] = matched_formation_snap

    def get_output(self, idx, target_snap_nr):
        formation_snap = self.formation_snap[target_snap_nr].astype(np.int32)
        formation_snap[self.target_values[target_snap_nr] == 0] = -1
        return {
            (self.group, self.snap_name): psort.fetch_elements(
                formation_snap, idx, comm=comm
            )
        }


def read_target_values(soap_filename, track_ids, datasets):
    """
    Read the specified datasets from a target snapshot, and return them in
    the same order as track_ids. Values are NaN for tracks which are not
    present in this snapshot.
    """
    data = read_soap(soap_filename, ["InputHalos/HBTplus/TrackId"] + datasets)
    target_values = {}
    if data is None:
        for dataset in datasets:
            target_values[dataset] = np.full(track_ids.shape[0], np.nan)
        return target_values
    idx = psort.parallel_match(track_ids, data["InputHalos/HBTplus/TrackId"], comm=comm)
    has_match = idx != -1
    for dataset in datasets:
        values = np.full(track_ids.shape[0], np.nan, dtype=np.float32)
        values[has_match] = psort.fetch_elements(
            data[dataset], idx[has_match], comm=comm
        )
        target_values[dataset] = values
    return target_values


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(
        description=(
            "Script to calculate properties which depend on the main "
            "progenitor branch of each subhalo"
        ),
    )
    parser.add_argument(
        "--soap-basename",
        type=str,
        required=True,
        help=(
            "The basename of the SOAP catalogues (the filename without "
            "the .hdf5 suffix). Use {snap_nr:04d} instead of the snapshot number"
        ),
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        required=True,
        help=(
            "The basename of the output catalogues (the filename without "
            "the .hdf5 suffix). Use {snap_nr:04d} instead of the snapshot number"
        ),
    )
    parser.add_argument(
        "--target-snap-nrs",
        type=int,
        nargs="+",
        required=True,
        help="Snapshots for which to write output catalogues",
    )
    for name, prop in property_types.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=lambda s: s.split(","),
            default=None,
            help=(
                f"Comma separated list of halo types to compute {name} for, "
                "e.g. BoundSubhalo,ExclusiveSphere/50kpc. All groups within "
                f"these which contain {prop['input']} are processed. If none of "
                "the property type arguments are passed then all property "
                "types are computed with their default halo types. "
                f"Default: {','.join(prop['default'])}"
            ),
        )
    parser.add_argument(
        "--skip-missing-snapshots",
        action="store_true",
        help=(
            "Skip snapshots whose SOAP catalogue does not exist. By default "
            "the script exits with an error if a catalogue is missing"
        ),
    )

    args = parser.parse_args()

    # Log the arguments
    for k, v in vars(args).items():
        mpi_print(f"  {k}: {v}", comm_rank)

    target_snap_nrs = sorted(set(args.target_snap_nrs))
    final_snap_nr = target_snap_nrs[-1]

    # Determine which halo types to compute each property type for
    halo_types = {name: getattr(args, name) for name in property_types}
    if all(v is None for v in halo_types.values()):
        halo_types = {name: prop["default"] for name, prop in property_types.items()}
    else:
        halo_types = {k: v for k, v in halo_types.items() if v is not None}

    # Determine which groups we will process for each property type
    if comm_rank == 0:
        final_soap_filename = (
            args.soap_basename.format(snap_nr=final_snap_nr) + ".hdf5"
        )
        groups = {}
        unmatched = []
        for name, types in halo_types.items():
            groups[name], prop_unmatched = find_groups(
                final_soap_filename, types, property_types[name]["input"]
            )
            for halo_type in prop_unmatched:
                unmatched.append(
                    f"No groups found for {name} matching '{halo_type}' "
                    f"(groups must contain {property_types[name]['input']})"
                )
    else:
        groups, unmatched = None, None
    groups, unmatched = comm.bcast((groups, unmatched))
    # Every rank raises, so the job will exit
    if len(unmatched) > 0:
        raise ValueError("\n".join(unmatched))
    for name, prop_groups in groups.items():
        mpi_print(f"Computing {name} for groups: {prop_groups}", comm_rank)

    # Load the TrackIds present in each of the target snapshots. We
    # will follow every TrackId which appears in any of them.
    mpi_print("Loading TrackIds from target snapshots", comm_rank)
    target_track_ids = []
    for snap_nr in target_snap_nrs:
        data = read_soap(
            args.soap_basename.format(snap_nr=snap_nr) + ".hdf5",
            ["InputHalos/HBTplus/TrackId"],
        )
        if data is not None:
            target_track_ids.append(data["InputHalos/HBTplus/TrackId"])
    target_track_ids = np.concatenate(target_track_ids)
    track_ids = psort.parallel_unique(
        target_track_ids, comm=comm, repartition_output=True
    )
    del target_track_ids
    n_track = comm.allreduce(track_ids.shape[0])
    mpi_print(f"Following {n_track} TrackIds", comm_rank)

    # The half mass formation trackers need to know the value of
    # their input dataset at each of the target snapshots
    formation_datasets = []
    for name in ["half_mass_formation", "half_stellar_mass_formation"]:
        for group in groups.get(name, []):
            formation_datasets.append(f"{group}/{property_types[name]['input']}")
    formation_datasets = sorted(set(formation_datasets))
    target_values = {}
    if len(formation_datasets) > 0:
        mpi_print("Loading target values for formation snapshots", comm_rank)
        for snap_nr in target_snap_nrs:
            target_values[snap_nr] = read_target_values(
                args.soap_basename.format(snap_nr=snap_nr) + ".hdf5",
                track_ids,
                formation_datasets,
            )

    # Set up the objects which compute each property
    trackers = []
    for group in groups.get("bh_maximum", []):
        trackers.append(
            MaximumTracker(
                track_ids.shape[0],
                group,
                "MostMassiveBlackHoleMass",
                "MainBranchMaximumMostMassiveBlackHoleMass",
                "MainBranchMaximumMostMassiveBlackHoleSnapshot",
                extra=(
                    "MostMassiveBlackHoleID",
                    "MainBranchMaximumMostMassiveBlackHoleID",
                    np.uint64,
                ),
            )
        )
    for group in groups.get("stellar_mass_maximum", []):
        trackers.append(
            MaximumTracker(
                track_ids.shape[0],
                group,
                "StellarMass",
                "MainBranchMaximumStellarMass",
                "MainBranchMaximumStellarMassSnapshot",
            )
        )
    for name, dataset, snap_name in [
        ("half_mass_formation", "TotalMass", "HalfMassFormationSnapshot"),
        (
            "half_stellar_mass_formation",
            "StellarMass",
            "HalfStellarMassFormationSnapshot",
        ),
    ]:
        for group in groups.get(name, []):
            trackers.append(
                HalfMassFormationTracker(
                    track_ids,
                    group,
                    dataset,
                    snap_name,
                    {
                        snap_nr: target_values[snap_nr][f"{group}/{dataset}"]
                        for snap_nr in target_snap_nrs
                    },
                )
            )
    del target_values

    input_halos_datasets = [
        "InputHalos/HBTplus/TrackId",
        "InputHalos/HaloCatalogueIndex",
    ]
    datasets = set()
    for tracker in trackers:
        datasets.update(tracker.input_datasets())
    datasets = input_halos_datasets + sorted(datasets)
    output_groups = sorted(set(tracker.group for tracker in trackers))

    for snap_nr in range(0, final_snap_nr + 1):

        soap_filename = args.soap_basename.format(snap_nr=snap_nr) + ".hdf5"
        if not os.path.exists(soap_filename):
            if not args.skip_missing_snapshots:
                # Every rank raises, so the job will exit. Outputs for
                # earlier target snapshots are complete and are kept.
                raise FileNotFoundError(
                    f"SOAP catalogue not found: {soap_filename}. "
                    "Pass --skip-missing-snapshots to skip missing snapshots"
                )
            mpi_print(f"Skipping snapshot {snap_nr}, file not found", comm_rank)
            continue
        data = read_soap(soap_filename, datasets)
        if data is None:
            mpi_print(f"Skipping snapshot {snap_nr}, no subhaloes", comm_rank)
            continue
        mpi_print(f"Processing snapshot {snap_nr}", comm_rank)

        # Get the values of each dataset for the tracks we are following
        idx = psort.parallel_match(
            track_ids, data["InputHalos/HBTplus/TrackId"], comm=comm
        )
        has_match = idx != -1
        values = {}
        for dataset in datasets[len(input_halos_datasets) :]:
            values[dataset] = psort.fetch_elements(
                data[dataset], idx[has_match], comm=comm
            )

        # Update the properties of each track
        for tracker in trackers:
            tracker.update(snap_nr, values, has_match)
        del values

        if snap_nr not in target_snap_nrs:
            continue

        # Get the values for each subhalo in this snapshot, in the
        # same order as the SOAP catalogue
        mpi_print(f"Writing output for snapshot {snap_nr}", comm_rank)
        snap_track_id = data["InputHalos/HBTplus/TrackId"]
        idx = psort.parallel_match(snap_track_id, track_ids, comm=comm)
        assert np.all(idx != -1)
        output = {}
        for tracker in trackers:
            if isinstance(tracker, HalfMassFormationTracker):
                output.update(tracker.get_output(idx, snap_nr))
            else:
                output.update(tracker.get_output(idx))
        write_output(
            soap_filename,
            args.output_basename.format(snap_nr=snap_nr) + ".hdf5",
            output_groups,
            {name: data[name] for name in input_halos_datasets},
            output,
        )
        del output

    # Finished
    comm.barrier()
    mpi_print(f"Runtime: {datetime.datetime.now() - start_time}", comm_rank)
    mpi_print("Done!", comm_rank)
