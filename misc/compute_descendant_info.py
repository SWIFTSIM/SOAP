#!/bin/env python

"""
compute_descendant_info.py

Older SOAP catalogues only contain the main branch descendant of each
subhalo, i.e. SOAP/DescendantIndex is -1 if the subhalo is not resolved in
the next snapshot. In addition, older versions of SOAP could compute an
incorrect SOAP/ProgenitorIndex and SOAP/DescendantIndex for some subhalos,
since the order of the previous/next SOAP catalogue was not always
reconstructed correctly for subhalos lying on the boundary of a SWIFT cell.

This script computes SOAP/ProgenitorIndex, SOAP/DescendantIndex, and
SOAP/DescendantType using the same method as the current version of SOAP.

The script works directly from the HBT-HERONS catalogues, so it does not
require the SOAP catalogues to exist. For each snapshot the resolved
subhalos are sorted in the same way as SOAP sorts them, and an output
catalogue is written with this ordering. The SWIFT snapshots are only used
to obtain the cell structure and metadata. The output catalogues can be
opened using swiftsimio.

If the SOAP catalogues are passed then the script checks that the order of
the subhalos is the same as in the SOAP catalogues, and reports the number
of subhalos whose ProgenitorIndex/DescendantIndex differ.

Usage:

  mpirun -- python -u misc/compute_descendant_info.py \\
            --hbt-basename HBT_BASENAME \\
            --snapshot-basename SNAPSHOT_BASENAME \\
            --output-basename OUTPUT_BASENAME \\
            --final-snap-nr SNAP_NR

Run "python misc/compute_descendant_info.py -h" for a full description
of the arguments, and a list of optional arguments.

"""

import argparse
import datetime
import os
import socket
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

from SOAP.catalogue_readers import read_hbtplus
from SOAP.core import swift_cells
from SOAP.core.combine_chunks import (
    compute_descendant_index,
    spatial_sort,
    sub_snapnum,
)
from SOAP.core.soap_args import get_git_hash
from SOAP.property_table import PropertyTable

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
    "Is Compressed": False,
    "Lossy compression filter": "None",
}

# Datasets in the output files, and their key in the property table
output_datasets = {
    "InputHalos/HBTplus/TrackId": "HBTplus/TrackId",
    "InputHalos/HaloCatalogueIndex": "index",
    "SOAP/ProgenitorIndex": "SOAP/ProgenitorIndex",
    "SOAP/DescendantIndex": "SOAP/DescendantIndex",
    "SOAP/DescendantType": "SOAP/DescendantType",
}


def mpi_print(string, comm_rank):
    if comm_rank == 0:
        print(string)


def load_cellgrid(snapshot_filename):
    """
    Read the cell structure and metadata from a SWIFT snapshot
    """
    if comm_rank == 0:
        cellgrid = swift_cells.SWIFTCellGrid(snapshot_filename, [])
    else:
        cellgrid = None
    return comm.bcast(cellgrid)


def load_snapshot(hbt_basename, snapshot_filename, index_by_track_id):
    """
    Read the HBT-HERONS catalogue for a snapshot, and sort the resolved
    subhalos in the same order as SOAP. Returns a dict containing:

    full        - TrackId, number of bound particles, and descendant
                  information for every subhalo (including orphans)
    track_id    - TrackId of the resolved subhalos, in SOAP order
    index       - Catalogue index of the resolved subhalos, in SOAP order
    cell_counts - Number of subhalos in each cell (only on rank 0)
    n_total     - Total number of resolved subhalos
    cellgrid    - Cell structure and metadata from the snapshot
    """
    cellgrid = load_cellgrid(snapshot_filename)

    # Check if the catalogue has any subhalos
    if comm_rank == 0:
        if os.path.exists(f"{hbt_basename}.0.hdf5"):
            filename = f"{hbt_basename}.0.hdf5"
        else:
            filename = hbt_basename
        with h5py.File(filename, "r") as file:
            n_hbt = int(file["NumberOfSubhalosInAllFiles"][0])
    else:
        n_hbt = None
    n_hbt = comm.bcast(n_hbt)

    data = {
        "full": None,
        "track_id": np.zeros(0, dtype=np.int64),
        "index": np.zeros(0, dtype=np.int64),
        "cell_counts": np.zeros(cellgrid.nr_cells[0], dtype=np.int64),
        "n_total": 0,
        "cellgrid": cellgrid,
    }
    if n_hbt == 0:
        return data

    hbt = read_hbtplus.read_hbtplus_catalogue(
        comm,
        hbt_basename,
        cellgrid.a_unit,
        cellgrid.snap_unit_registry,
        cellgrid.boxsize,
        keep_orphans=True,
        index_by_track_id=index_by_track_id,
        read_descendant_info=True,
    )
    data["full"] = {
        name: np.asarray(hbt[name])
        for name in ["TrackId", "nr_bound_part", "SinkTrackId", "DescendantTrackId"]
    }

    # Sort the resolved subhalos in the same way as SOAP
    resolved = data["full"]["nr_bound_part"] > 0
    data["n_total"] = comm.allreduce(np.sum(resolved))
    if data["n_total"] == 0:
        return data
    order, cell_counts = spatial_sort(
        hbt["cofp"][resolved], hbt["index"][resolved], cellgrid, comm
    )
    data["track_id"] = np.asarray(
        psort.fetch_elements(data["full"]["TrackId"][resolved], order, comm=comm)
    )
    data["index"] = np.asarray(
        psort.fetch_elements(np.asarray(hbt["index"])[resolved], order, comm=comm)
    )
    data["cell_counts"] = cell_counts
    return data


def has_values(data, name):
    """
    Returns True if the dataset is present in the SOAP catalogue
    and is not -1 for every subhalo
    """
    arr = data.get(name, None)
    has = (arr is not None) and bool(np.any(arr != -1))
    return comm.allreduce(has, op=MPI.LOR)


def compare_with_soap(soap_filename, output, has_prev, has_next):
    """
    Check that the subhalos are in the same order as in the SOAP catalogue,
    and report the number of subhalos whose ProgenitorIndex/DescendantIndex
    differ from the SOAP catalogue. The old descendant index only follows
    the main branch, so it should agree with the new one for subhalos which
    survive, and should be -1 for subhalos which disrupt or sink.
    """
    track_id = output["InputHalos/HBTplus/TrackId"]
    n_total = comm.allreduce(track_id.shape[0])

    # Read the SOAP catalogue
    names = [
        "InputHalos/HBTplus/TrackId",
        "SOAP/ProgenitorIndex",
        "SOAP/DescendantIndex",
    ]
    if comm_rank == 0:
        with h5py.File(soap_filename, "r") as file:
            n_soap = int(file["Header"].attrs["NumSubhalos_Total"][0])
            names = [name for name in names if name in file]
    else:
        n_soap = None
        names = None
    n_soap, names = comm.bcast((n_soap, names))
    if n_soap != n_total:
        mpi_print(
            f"  {soap_filename} has {n_soap} subhalos, expected {n_total}", comm_rank
        )
        comm.Abort(1)
    if n_total == 0:
        return
    mf = phdf5.MultiFile([soap_filename], comm=comm)
    soap = mf.read(names)

    # Distribute in the same way as the output arrays
    counts = comm.allgather(track_id.shape[0])
    soap = {
        name: psort.repartition(arr, counts, comm=comm) for name, arr in soap.items()
    }

    # Check the subhalos are in the same order
    n_diff = comm.allreduce(np.sum(soap["InputHalos/HBTplus/TrackId"] != track_id))
    if n_diff > 0:
        mpi_print(
            f"  {n_diff} subhalos have a different TrackId to {soap_filename}",
            comm_rank,
        )
        comm.Abort(1)
    mpi_print("  TrackId order matches the SOAP catalogue", comm_rank)

    # Compare the indices
    if has_prev and has_values(soap, "SOAP/ProgenitorIndex"):
        old = soap["SOAP/ProgenitorIndex"]
        new = output["SOAP/ProgenitorIndex"]
        n_diff = comm.allreduce(np.sum(old != new))
        mpi_print(
            f"  ProgenitorIndex differs from SOAP for {n_diff} subhalos", comm_rank
        )
    else:
        mpi_print("  No ProgenitorIndex to compare with", comm_rank)

    if has_next and has_values(soap, "SOAP/DescendantIndex"):
        old = soap["SOAP/DescendantIndex"]
        new = output["SOAP/DescendantIndex"]
        new_type = output["SOAP/DescendantType"]
        survived = new_type == 0
        merged = new_type > 0
        n_survived = comm.allreduce(np.sum(old[survived] != new[survived]))
        n_merged = comm.allreduce(np.sum(old[merged] != -1))
        mpi_print(
            f"  DescendantIndex differs from SOAP for {n_survived} surviving "
            f"subhalos, and is not -1 in SOAP for {n_merged} merged subhalos",
            comm_rank,
        )
    else:
        mpi_print("  No DescendantIndex to compare with", comm_rank)


def write_metadata(outfile, cellgrid, cell_counts, n_total, snap_nr, args):
    """
    Write the metadata required for swiftsimio to read the output
    catalogue, following the format of the SOAP catalogues
    """
    code = outfile.create_group("Code")
    code.attrs["Code"] = "SOAP"
    code.attrs["git_hash"] = get_git_hash()
    code.attrs["Date"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Arguments used to run this script
    params = outfile.create_group("Parameters")
    params.attrs["script"] = os.path.basename(__file__)
    params.attrs["snapshot_nr"] = snap_nr
    for name, value in vars(args).items():
        params.attrs[name] = str(value)

    cellgrid.copy_swift_metadata(outfile)

    header = outfile.create_group("Header")
    for attr in [
        "BoxSize",
        "Dimension",
        "NumPartTypes",
        "Redshift",
        "RunName",
        "Scale-factor",
        "Time",
    ]:
        header.attrs[attr] = cellgrid.swift_header_group[attr]
    header.attrs["Code"] = "SOAP"
    header.attrs["NumFilesPerSnapshot"] = np.array([1], dtype="int32")
    header.attrs["NumSubhalos_ThisFile"] = np.array([n_total], dtype="int32")
    header.attrs["NumSubhalos_Total"] = np.array([n_total], dtype="int32")
    n_part_type = cellgrid.swift_header_group["NumPartTypes"][0]
    header.attrs["NumPart_ThisFile"] = np.zeros(n_part_type, dtype="int32")
    header.attrs["NumPart_Total"] = np.zeros(n_part_type, dtype="uint32")
    header.attrs["NumPart_Total_HighWord"] = np.zeros(n_part_type, dtype="uint32")
    header.attrs["OutputType"] = "SOAP"
    header.attrs["SnapshotDate"] = time.strftime(
        "%H:%M:%S %Y-%m-%d GMT", time.gmtime()
    )
    header.attrs["System"] = socket.gethostname()
    header.attrs["ThisFile"] = np.array([0], dtype="int32")
    header.attrs["SubhaloTypes"] = ["InputHalos", "InputHalos/HBTplus", "SOAP"]

    cosmo = outfile.create_group("Cosmology")
    for name, value in cellgrid.cosmology.items():
        cosmo.attrs[name] = [value]

    units = outfile.create_group("Units")
    for name, value in cellgrid.swift_units_group.items():
        units.attrs[name] = [value]

    const = outfile.create_group("PhysicalConstants").create_group("CGS")
    for name, value in cellgrid.constants.items():
        const.attrs[name] = [value]

    cells = outfile.create_group("Cells")
    cells_metadata = cells.create_group("Meta-data")
    cells_metadata.attrs["dimension"] = cellgrid.dimension
    cells_metadata.attrs["nr_cells"] = cellgrid.nr_cells
    cells_metadata.attrs["size"] = cellgrid.cell_size.to("a*snap_length").value
    cells.create_dataset("Centres", data=cellgrid.cell_centres)
    cells.create_dataset("Counts/Subhalos", data=cell_counts)
    cells.create_dataset(
        "Files/Subhalos", data=np.zeros(cellgrid.nr_cells[0], dtype="int32")
    )
    cells.create_dataset(
        "OffsetsInFile/Subhalos", data=np.cumsum(cell_counts) - cell_counts
    )


def write_output(
    output_filename, cellgrid, cell_counts, n_total, output, snap_nr, args
):
    """
    Write a minimal SOAP catalogue which can be opened with swiftsimio
    """
    if comm_rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
        with h5py.File(output_filename, "w") as outfile:
            write_metadata(outfile, cellgrid, cell_counts, n_total, snap_nr, args)
            outfile.create_group("InputHalos/HBTplus")
            outfile.create_group("SOAP")
            if n_total == 0:
                for name, key in output_datasets.items():
                    dtype = PropertyTable.full_property_list[key].dtype
                    outfile.create_dataset(name, shape=(0,), dtype=dtype)
    comm.barrier()

    if n_total > 0:
        with h5py.File(output_filename, "r+", driver="mpio", comm=comm) as outfile:
            for name, key in output_datasets.items():
                dtype = PropertyTable.full_property_list[key].dtype
                arr = output[name].astype(dtype)
                phdf5.collective_write(outfile, name, arr, comm=comm)
        comm.barrier()

    # Set the attributes of the datasets
    if comm_rank == 0:
        with h5py.File(output_filename, "r+") as outfile:
            for name, key in output_datasets.items():
                attrs = dict(unit_attrs)
                attrs["Description"] = PropertyTable.full_property_list[key].description
                for k, v in attrs.items():
                    outfile[name].attrs[k] = v
    comm.barrier()


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(
        description=(
            "Script to calculate the progenitor index, descendant index, and "
            "descendant type of each subhalo from HBT-HERONS catalogues"
        ),
    )
    parser.add_argument(
        "--hbt-basename",
        type=str,
        required=True,
        help=(
            "The basename of the HBT-HERONS catalogues, as passed to SOAP. "
            "Use {snap_nr:03d} instead of the snapshot number"
        ),
    )
    parser.add_argument(
        "--snapshot-basename",
        type=str,
        required=True,
        help=(
            "The filename of the SWIFT snapshots, as passed to SOAP. Use "
            "{snap_nr:04d} instead of the snapshot number, and {file_nr} "
            "instead of the file number"
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
        "--final-snap-nr",
        type=int,
        required=True,
        help=(
            "The final snapshot of the simulation. Subhalos in this snapshot "
            "have DescendantIndex=-1 and DescendantType=-1"
        ),
    )
    parser.add_argument(
        "--snap-nrs",
        type=int,
        nargs="+",
        default=None,
        help="Only process these snapshots. Default: all snapshots",
    )
    parser.add_argument(
        "--index-by-track-id",
        action="store_true",
        help=(
            "Pass this if SOAP was run with index_by_track_id. Only "
            "affects unsorted HBT-HERONS catalogues"
        ),
    )
    parser.add_argument(
        "--soap-basename",
        type=str,
        default=None,
        help=(
            "The basename of existing SOAP catalogues (the filename without "
            "the .hdf5 suffix) to compare with. Use {snap_nr:04d} instead of "
            "the snapshot number. Snapshots without a SOAP catalogue are not "
            "compared. The script exits with an error if the order of the "
            "subhalos differs from the SOAP catalogues"
        ),
    )
    parser.add_argument(
        "--skip-missing-snapshots",
        action="store_true",
        help=(
            "Skip snapshots whose HBT-HERONS catalogue or SWIFT snapshot does "
            "not exist. Subhalos in the previous snapshot will have "
            "DescendantIndex=-1 and DescendantType=-1, and subhalos in the "
            "next snapshot will have ProgenitorIndex=-1. By default the "
            "script exits with an error if an input file is missing"
        ),
    )

    args = parser.parse_args()

    # Log the arguments
    for k, v in vars(args).items():
        mpi_print(f"  {k}: {v}", comm_rank)

    if args.snap_nrs is None:
        snap_nrs = list(range(0, args.final_snap_nr + 1))
    else:
        snap_nrs = sorted(set(args.snap_nrs))
    assert snap_nrs[-1] <= args.final_snap_nr

    def hbt_basename(snap_nr):
        return args.hbt_basename.format(snap_nr=snap_nr)

    def snapshot_filename(snap_nr):
        return sub_snapnum(args.snapshot_basename, snap_nr)

    def output_filename(snap_nr):
        return args.output_basename.format(snap_nr=snap_nr) + ".hdf5"

    def soap_filename(snap_nr):
        return args.soap_basename.format(snap_nr=snap_nr) + ".hdf5"

    def missing_input_files(snap_nr):
        """Returns the list of missing input files for a snapshot"""
        missing = []
        basename = hbt_basename(snap_nr)
        if not (os.path.exists(basename) or os.path.exists(f"{basename}.0.hdf5")):
            missing.append(basename)
        filename = snapshot_filename(snap_nr).format(file_nr=0)
        if not os.path.exists(filename):
            missing.append(filename)
        return missing

    # Determine which snapshots have all their input files
    if comm_rank == 0:
        missing = {
            snap_nr: missing_input_files(snap_nr)
            for snap_nr in range(0, args.final_snap_nr + 1)
        }
    else:
        missing = None
    missing = comm.bcast(missing)

    def available(snap_nr):
        return (0 <= snap_nr <= args.final_snap_nr) and (not missing[snap_nr])

    # Check all the required input files exist before we start
    required = set()
    for snap_nr in snap_nrs:
        for s in [snap_nr - 1, snap_nr, snap_nr + 1]:
            if 0 <= s <= args.final_snap_nr:
                required.update(missing[s])
    if required:
        for filename in sorted(required):
            mpi_print(f"Input file not found: {filename}", comm_rank)
        if not args.skip_missing_snapshots:
            # Every rank raises, so the job will exit
            raise FileNotFoundError(
                "Input files are missing. "
                "Pass --skip-missing-snapshots to skip missing snapshots"
            )

    # Sorted catalogues for the snapshots we are currently using
    cache = {}

    def get_snapshot(snap_nr):
        if snap_nr not in cache:
            cache[snap_nr] = load_snapshot(
                hbt_basename(snap_nr),
                snapshot_filename(snap_nr),
                args.index_by_track_id,
            )
        return cache[snap_nr]

    for snap_nr in snap_nrs:

        if not available(snap_nr):
            mpi_print(f"Skipping snapshot {snap_nr}, input files not found", comm_rank)
            continue
        mpi_print(f"Processing snapshot {snap_nr}", comm_rank)

        # Remove snapshots we no longer need
        for s in list(cache.keys()):
            if s < snap_nr - 1:
                del cache[s]

        data = get_snapshot(snap_nr)
        n_local = data["track_id"].shape[0]
        has_prev = available(snap_nr - 1)
        has_next = available(snap_nr + 1)

        # Find where each TrackId appears in the previous snapshot
        progenitor_index = -1 * np.ones(n_local, dtype=np.int32)
        if not has_prev:
            mpi_print(f"  No previous snapshot for snapshot {snap_nr}", comm_rank)
        elif data["n_total"] > 0:
            prev = get_snapshot(snap_nr - 1)
            if prev["n_total"] > 0:
                progenitor_index[:] = psort.parallel_match(
                    data["track_id"], prev["track_id"], comm=comm
                )

        # Find the descendant of each subhalo in the next snapshot
        descendant_index = -1 * np.ones(n_local, dtype=np.int32)
        descendant_type = -1 * np.ones(n_local, dtype=np.int32)
        if not has_next:
            mpi_print(f"  No next snapshot for snapshot {snap_nr}", comm_rank)
        elif data["n_total"] > 0:
            next_data = get_snapshot(snap_nr + 1)
            descendant_index, descendant_type = compute_descendant_index(
                data["track_id"], next_data["full"], next_data["track_id"], comm
            )

        output = {
            "InputHalos/HBTplus/TrackId": data["track_id"],
            "InputHalos/HaloCatalogueIndex": data["index"],
            "SOAP/ProgenitorIndex": progenitor_index,
            "SOAP/DescendantIndex": descendant_index,
            "SOAP/DescendantType": descendant_type,
        }

        # Compare with the existing SOAP catalogue
        if args.soap_basename is not None:
            if os.path.exists(soap_filename(snap_nr)):
                compare_with_soap(soap_filename(snap_nr), output, has_prev, has_next)
            else:
                mpi_print(f"  No SOAP catalogue for snapshot {snap_nr}", comm_rank)

        write_output(
            output_filename(snap_nr),
            data["cellgrid"],
            data["cell_counts"],
            data["n_total"],
            output,
            snap_nr,
            args,
        )

        # The full catalogue is only needed when this is the next snapshot
        data["full"] = None

    # Finished
    comm.barrier()
    mpi_print(f"Runtime: {datetime.datetime.now() - start_time}", comm_rank)
    mpi_print("Done!", comm_rank)
