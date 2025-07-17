import h5py
from mpi4py import MPI
import numpy as np
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.util.partial_formatter import PartialFormatter

from SOAP.core import lustre, swift_units
import xray_calculator

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def recalculate_xrays(snap_file, output_filename, units, xray_calculator):
    """
    Calculate Xray properties for gas particles, write them to the output file
    """

    if comm_rank == 0:
        print("Loading data")
    datasets = ["Densities", "Temperatures", "SmoothedElementMassFractions", "Masses"]
    data = snap_file.read(datasets, group="PartType0")
    for dset in datasets:
        data[dset] *= units[dset]

    # Dictionary for holding recalculated values
    output = {}

    if comm_rank == 0:
        print("Recalculating xrays")
    (
        idx_he,
        idx_T,
        idx_n,
        t_z,
        d_z,
        t_T,
        d_T,
        t_nH,
        d_nH,
        t_He,
        d_He,
        abundance_to_solar,
        joint_mask,
        volumes,
        data_n,
    ) = xray_calculator.find_indices(
        data["Densities"],
        data["Temperatures"],
        data["SmoothedElementMassFractions"],
        data["Masses"],
        fill_value=0,
    )

    xray_bands = ["erosita-low", "erosita-high", "ROSAT"]
    observing_types = ["energies_intrinsic", "energies_intrinsic", "energies_intrinsic"]
    output["XrayLuminosities"] = xray_calculator.interpolate_X_Ray(
        idx_he,
        idx_T,
        idx_n,
        t_z,
        d_z,
        t_T,
        d_T,
        t_nH,
        d_nH,
        t_He,
        d_He,
        abundance_to_solar,
        joint_mask,
        volumes,
        data_n,
        bands=xray_bands,
        observing_types=observing_types,
        fill_value=0,
    ).to(units["XrayLuminosities"])

    observing_types = ["photons_intrinsic", "photons_intrinsic", "photons_intrinsic"]
    output["XrayPhotonLuminosities"] = xray_calculator.interpolate_X_Ray(
        idx_he,
        idx_T,
        idx_n,
        t_z,
        d_z,
        t_T,
        d_T,
        t_nH,
        d_nH,
        t_He,
        d_He,
        abundance_to_solar,
        joint_mask,
        volumes,
        data_n,
        bands=xray_bands,
        observing_types=observing_types,
        fill_value=0,
    ).to(units["XrayPhotonLuminosities"])

    observing_types = [
        "energies_intrinsic_restframe",
        "energies_intrinsic_restframe",
        "energies_intrinsic_restframe",
    ]
    output["XrayLuminositiesRestframe"] = xray_calculator.interpolate_X_Ray(
        idx_he,
        idx_T,
        idx_n,
        t_z,
        d_z,
        t_T,
        d_T,
        t_nH,
        d_nH,
        t_He,
        d_He,
        abundance_to_solar,
        joint_mask,
        volumes,
        data_n,
        bands=xray_bands,
        observing_types=observing_types,
        fill_value=0,
    ).to(units["XrayLuminosities"])

    observing_types = [
        "photons_intrinsic_restframe",
        "photons_intrinsic_restframe",
        "photons_intrinsic_restframe",
    ]
    output["XrayPhotonLuminositiesRestframe"] = xray_calculator.interpolate_X_Ray(
        idx_he,
        idx_T,
        idx_n,
        t_z,
        d_z,
        t_T,
        d_T,
        t_nH,
        d_nH,
        t_He,
        d_He,
        abundance_to_solar,
        joint_mask,
        volumes,
        data_n,
        bands=xray_bands,
        observing_types=observing_types,
        fill_value=0,
    ).to(units["XrayPhotonLuminosities"])

    attrs = {}
    for k in output.keys():
        attrs[k] = {
            "Description": f"{k} recalculated by SOAP using table {xray_calculator.table_path}"
        }
        attrs[k].update(swift_units.attributes_from_units(output[k].units, 1, 0))

    comm.barrier()
    # Write these particles out with the same layout as the input snapshot
    if comm_rank == 0:
        print("Writing out xray properties")
    elements_per_file = snap_file.get_elements_per_file(
        "ParticleIDs", group="PartType0"
    )
    snap_file.write(
        output,
        elements_per_file,
        filenames=output_filename,
        mode="w",
        group="PartType0",
        attrs=attrs,
    )


#
if __name__ == "__main__":

    import datetime

    start = datetime.datetime.now()

    # Read parameters from command line and config file
    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(
        comm=comm, description="Recalculate xray properties from SWIFT snapshots."
    )
    parser.add_argument(
        "swift_filename",
        type=str,
        help="Basename of the input snapshots, use {snap_nr:04d} for the snapshot number and {file_nr} for the file number.",
    )
    parser.add_argument(
        "output_filename",
        type=str,
        help="Basename of the output snapshots, use {snap_nr:04d} for the snapshot number and {file_nr} for the file number.",
    )
    parser.add_argument(
        "xray_table_path", type=str, help="Path to the xray tables to use"
    )
    parser.add_argument("--snap-nr", type=int, help="Snapshot number to process")
    args = parser.parse_args()

    # Substitute in the snapshot number where necessary
    pf = PartialFormatter()
    swift_filename = pf.format(args.swift_filename, snap_nr=args.snap_nr, file_nr=None)
    output_filename = pf.format(
        args.output_filename, snap_nr=args.snap_nr, file_nr=None
    )

    # Ensure output dir exists
    if comm_rank == 0:
        lustre.ensure_output_dir(output_filename)
    comm.barrier()

    # Load tables on rank 0
    if comm_rank == 0:
        units = {}
        with h5py.File(swift_filename.format(file_nr=0), "r") as snap_file:
            z = snap_file["Header"].attrs["Redshift"][0]
            unit_registry = swift_units.unit_registry_from_snapshot(snap_file)
            for dset in [
                "Densities",
                "Temperatures",
                "SmoothedElementMassFractions",
                "Masses",
                "XrayLuminosities",
                "XrayPhotonLuminosities",
            ]:
                attrs = snap_file[f"PartType0/{dset}"].attrs
                units[dset] = swift_units.units_from_attributes(attrs, unit_registry)

        xray_bands = [
            "erosita-low",
            "erosita-high",
            "ROSAT",
            "erosita-low",
            "erosita-high",
            "ROSAT",
            "erosita-low",
            "erosita-high",
            "ROSAT",
            "erosita-low",
            "erosita-high",
            "ROSAT",
        ]
        observing_types = [
            "energies_intrinsic",
            "energies_intrinsic",
            "energies_intrinsic",
            "photons_intrinsic",
            "photons_intrinsic",
            "photons_intrinsic",
            "energies_intrinsic_restframe",
            "energies_intrinsic_restframe",
            "energies_intrinsic_restframe",
            "photons_intrinsic_restframe",
            "photons_intrinsic_restframe",
            "photons_intrinsic_restframe",
        ]
        xray_calculator = xray_calculator.XrayCalculator(
            z, args.xray_table_path, xray_bands, observing_types
        )
    else:
        units = None
        xray_calculator = None
    units = comm.bcast(units)
    xray_calculator = comm.bcast(xray_calculator)

    # Open the snapshot
    snap_file = phdf5.MultiFile(
        swift_filename, file_nr_attr=("Header", "NumFilesPerSnapshot")
    )

    # Recalculate xrays and write the output
    recalculate_xrays(snap_file, output_filename, units, xray_calculator)

    comm.barrier()
    if comm_rank == 0:
        runtime = datetime.datetime.now() - start
        print(f"Done in {int(runtime.total_seconds())}s")
