#!/bin/env python

import os
import socket
import time

import h5py
from mpi4py import MPI
import numpy as np
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.util.partial_formatter import PartialFormatter

from catalogue_readers import read_hbtplus
from property_calculation.subhalo_rank import compute_subhalo_rank
from . import swift_units
from .mpi_timer import MPITimer
from SOAP.property_table import PropertyTable


def sub_snapnum(filename, snapnum):
    """
    Substitute the snapshot number into a filename format string
    without substituting the file number.
    """
    from virgo.util.partial_formatter import PartialFormatter

    pf = PartialFormatter()
    filename = pf.format(filename, snap_nr=snapnum, file_nr=None)
    return filename


def spatial_sort(halo_cofp, halo_index, cellgrid, comm):
    """
    Sort the halos based on what cell they are in, then based
    on their catalogue index
    """
    cell_indices = (halo_cofp // cellgrid.cell_size).value.astype("int64")
    assert cellgrid.dimension[0] >= cellgrid.dimension[1] >= cellgrid.dimension[2]
    # Assert that all halos are within the box
    if cell_indices.shape[0] > 0:
        assert np.min(cell_indices) >= 0
        for i_cell in range(3):
            assert np.max(cell_indices[:, i_cell]) < cellgrid.dimension[i_cell]
    # Sort first based on position, then on catalogue index
    sort_hash_dtype = [("cell_index", np.int64), ("catalogue_index", np.int64)]
    sort_hash = np.zeros(cell_indices.shape[0], dtype=sort_hash_dtype)
    sort_hash["cell_index"] += cell_indices[:, 0] * cellgrid.dimension[0] ** 2
    sort_hash["cell_index"] += cell_indices[:, 1] * cellgrid.dimension[1]
    sort_hash["cell_index"] += cell_indices[:, 2]
    sort_hash["catalogue_index"] = halo_index
    order = psort.parallel_sort(sort_hash, return_index=True, comm=comm)

    # Calculate local count of halos in each cell, and combine on rank 0
    local_cell_counts = np.bincount(
        sort_hash["cell_index"], minlength=cellgrid.nr_cells[0]
    ).astype("int64")
    assert local_cell_counts.shape[0] == np.prod(cellgrid.dimension)
    cell_counts = comm.reduce(local_cell_counts)

    return order, cell_counts


def combine_chunks(
    args,
    cellgrid,
    halo_prop_list,
    scratch_file_format,
    ref_metadata,
    nr_chunks,
    comm_world,
    category_filter,
    recently_heated_gas_filter,
    cold_dense_gas_filter,
):
    """
    Combine the per-chunk output files into a single, sorted output
    """

    # Open the per-chunk scratch files
    scratch_file = phdf5.MultiFile(
        scratch_file_format, file_idx=range(nr_chunks), comm=comm_world
    )

    # Determine units of halo centres:
    # ref_metadata is a list of (name, dimensions, units, description) for each property.
    cofp_metadata = [rm for rm in ref_metadata if rm[0] == "InputHalos/HaloCentre"][0]
    cofp_units = cofp_metadata[2]

    # Sort halos based on what cell their centre is in
    with MPITimer(
        "Establishing ordering of halos based on SWIFT cell structure", comm_world
    ):
        halo_cofp = scratch_file.read("InputHalos/HaloCentre") * cofp_units
        halo_index = scratch_file.read("InputHalos/HaloCatalogueIndex")
        order, cell_counts = spatial_sort(halo_cofp, halo_index, cellgrid, comm_world)

        # Free up some memory
        del halo_cofp
        del halo_index

    # Determine total number of halos
    total_nr_halos = comm_world.allreduce(len(order))

    # Handle derived (SOAP) quantities: these don't exist in the chunk
    # output but will be computed by combining other halo properties.
    soap_props = set()  # SOAP properties which will be calculated
    props_to_keep = set()  # Properties required from the chunk files
    if args.halo_format == "VR":
        # Subhalo rank
        soap_props.add("SOAP/SubhaloRankByBoundMass")
        props_to_keep.update(
            (
                "InputHalos/VR/ID",
                "BoundSubhalo/TotalMass",
                "InputHalos/VR/HostHaloID",
            )
        )
        # Host halo index
        soap_props.add("SOAP/HostHaloIndex")
        props_to_keep.update(("InputHalos/VR/ID", "InputHalos/VR/HostHaloID"))
    elif args.halo_format == "HBTplus":
        # Subhalo rank
        soap_props.add("SOAP/SubhaloRankByBoundMass")
        props_to_keep.update(
            (
                "InputHalos/HBTplus/HostFOFId",
                "BoundSubhalo/TotalMass",
                "InputHalos/HBTplus/TrackId",
            )
        )
        # Host halo index
        soap_props.add("SOAP/HostHaloIndex")
        props_to_keep.update(("InputHalos/HBTplus/HostFOFId", "InputHalos/IsCentral"))
        # Progenitor/Descendant SOAP index
        soap_props.update(("SOAP/ProgenitorIndex", "SOAP/DescendantIndex"))
        props_to_keep.add("InputHalos/HBTplus/TrackId")

    # Also keep M200c for calculating reduced_snapshot flag
    if "reduced_snapshots" in args.calculations:
        for metadata in ref_metadata:
            if metadata[0] == "SO/200_crit/TotalMass":
                soap_props.add("SOAP/IncludedInReducedSnapshot")
                props_to_keep.add("SO/200_crit/TotalMass")
                break
        else:
            if comm_world.Get_rank() == 0:
                print("IncludedInReducedSnapshot is enabled, but M200c is missing")

    # Get metadata for derived quantities
    soap_metadata = []
    for key, prop in PropertyTable.full_property_list.items():
        if not key.startswith("SOAP/"):
            continue
        if not key in soap_props:
            continue
        name = prop.name
        size = prop.shape
        if size == 1:
            # Scalar quantity
            size = ()
        else:
            # Vector quantity
            size = (size,)
        dtype = prop.dtype
        unit = cellgrid.get_unit(prop.unit)
        description = prop.description
        physical = prop.output_physical
        a_exponent = prop.a_scale_exponent
        if not physical:
            unit = unit * cellgrid.get_unit("a") ** a_exponent
        soap_metadata.append(
            (name, size, unit, dtype, description, physical, a_exponent)
        )

    # Add metadata for FOF properties
    fof_metadata = []
    if (args.fof_group_filename != "") and (args.halo_format == "HBTplus"):
        fof_keys = ["Centres", "Masses", "Sizes"]
        if (args.fof_radius_filename != ""):
            fof_keys.append('Radii')
        for fofkey in fof_keys:
            prop = PropertyTable.full_property_list[f"FOF/{fofkey}"]
            name = f"InputHalos/{prop.name}"
            size = prop.shape
            if size == 1:
                # Scalar quantity
                size = ()
            else:
                # Vector quantity
                size = (size,)
            dtype = prop.dtype
            unit = cellgrid.get_unit(prop.unit)
            description = prop.description
            physical = prop.output_physical
            a_exponent = prop.a_scale_exponent
            if not physical:
                unit = unit * cellgrid.get_unit("a") ** a_exponent
            fof_metadata.append(
                (name, size, unit, dtype, description, physical, a_exponent)
            )

        # Keep the properties required for matching subhalos to FOFs
        props_to_keep.update(("InputHalos/HBTplus/HostFOFId", "InputHalos/IsCentral"))

    # First MPI rank sets up the output file
    with MPITimer("Creating output file", comm_world):
        output_file = sub_snapnum(args.output_file, args.snapshot_nr)
        if comm_world.Get_rank() == 0:
            # Create the file
            outfile = h5py.File(output_file, "w")

            # Write parameters
            params = outfile.create_group("Parameters")
            params.attrs["swift_filename"] = args.swift_filename
            params.attrs["membership_filename"] = args.extra_input[-1]
            params.attrs["extra_input"] = args.extra_input[:-1]
            params.attrs["halo_basename"] = args.halo_basename
            params.attrs["halo_format"] = args.halo_format
            params.attrs["snapshot_nr"] = args.snapshot_nr
            params.attrs["centrals_only"] = 0 if args.centrals_only == False else 1
            calc_names = sorted([hp.name for hp in halo_prop_list])
            params.attrs["calculations"] = calc_names
            params.attrs["halo_indices"] = (
                args.halo_indices
                if args.halo_indices is not None
                else np.ndarray(0, dtype=int)
            )
            if recently_heated_gas_filter.initialised:
                recently_heated_gas_metadata = recently_heated_gas_filter.get_metadata()
                recently_heated_gas_params = params.create_group(
                    "RecentlyHeatedGasFilter"
                )
                for at, val in recently_heated_gas_metadata.items():
                    recently_heated_gas_params.attrs[at] = val
            if cold_dense_gas_filter.initialised:
                cold_dense_gas_params = params.create_group("ColdDenseGasFilter")
                for at, val in cold_dense_gas_filter.get_metadata().items():
                    cold_dense_gas_params.attrs[at] = val

            # Write code information
            code = outfile.create_group("Code")
            code.attrs["Code"] = "SOAP"
            code.attrs["git_hash"] = args.git_hash

            # Copy swift metadata
            params = cellgrid.copy_swift_metadata(outfile)

            # Generate header
            header = outfile.create_group("Header")
            for attr in [
                "BoxSize",
                "Dimension",
                "NumPartTypes",
                "Redshift",
                "RunName",
                "Scale-factor",
            ]:
                header.attrs[attr] = cellgrid.swift_header_group[attr]
            header.attrs["Code"] = "SOAP"
            header.attrs["Dimension"] = cellgrid.swift_header_group["Dimension"]
            header.attrs["NumFilesPerSnapshot"] = np.array([1], dtype="int32")
            header.attrs["NumSubhalos_ThisFile"] = np.array(
                [total_nr_halos], dtype="int32"
            )
            header.attrs["NumSubhalos_Total"] = np.array(
                [total_nr_halos], dtype="int32"
            )
            n_part_type = cellgrid.swift_header_group["NumPartTypes"][0]
            header.attrs["NumPart_ThisFile"] = np.zeros(n_part_type, dtype="int32")
            header.attrs["NumPart_Total"] = np.zeros(n_part_type, dtype="uint32")
            header.attrs["NumPart_Total_HighWord"] = np.zeros(
                n_part_type, dtype="uint32"
            )
            header.attrs["OutputType"] = "SOAP"
            snapshot_date = time.strftime("%H:%M:%S %Y-%m-%d GMT", time.gmtime())
            header.attrs["SnapshotDate"] = snapshot_date
            header.attrs["System"] = socket.gethostname()
            header.attrs["ThisFile"] = np.array([0], dtype="int32")

            # Write cosmology
            cosmo = outfile.create_group("Cosmology")
            for name, value in cellgrid.cosmology.items():
                cosmo.attrs[name] = [value]

            # Write units
            units = outfile.create_group("Units")
            for name, value in cellgrid.swift_units_group.items():
                units.attrs[name] = [value]

            # Write physical constants
            const = outfile.create_group("PhysicalConstants")
            const = const.create_group("CGS")
            for name, value in cellgrid.constants.items():
                const.attrs[name] = [value]

            # Write cell information
            cells = outfile.create_group("Cells")
            cells_metadata = cells.create_group("Meta-data")
            cells_metadata.attrs["dimension"] = cellgrid.dimension
            cells_metadata.attrs["nr_cells"] = cellgrid.nr_cells
            cell_size = cellgrid.cell_size.to("a*snap_length").value
            cells_metadata.attrs["size"] = cell_size
            cells.create_dataset("Centres", data=cellgrid.cell_centres)
            cells.create_dataset("Counts/Subhalos", data=cell_counts)
            cells.create_dataset(
                "Files/Subhalos", data=np.zeros(cellgrid.nr_cells[0], dtype="int32")
            )
            cell_offsets = np.cumsum(cell_counts) - cell_counts
            cells.create_dataset("OffsetsInFile/Subhalos", data=cell_offsets)

            # Create datasets for all halo properties
            for metadata in ref_metadata + soap_metadata + fof_metadata:
                name, size, unit, dtype, description, physical, a_exponent = metadata
                if description == "No description available":
                    print(f"{name} not found in property table")
                    compression_metadata = {
                        "Lossy compression filter": "None",
                        "Is Compressed": False,
                    }
                    mask_metadata = category_filter.get_filter_metadata(None)
                elif description.startswith("Time taken in seconds") or name in [
                    "InputHalos/n_loop",
                    "InputHalos/n_process",
                ]:
                    # Timing information
                    compression_metadata = {
                        "Lossy compression filter": "None",
                        "Is Compressed": False,
                    }
                    mask_metadata = category_filter.get_filter_metadata(None)
                else:
                    compression_metadata = category_filter.get_compression_metadata(
                        name
                    )
                    mask_metadata = category_filter.get_filter_metadata_for_property(
                        name
                    )

                shape = (total_nr_halos,) + size
                dataset = outfile.create_dataset(
                    name, shape=shape, dtype=dtype, fillvalue=None
                )
                attrs = swift_units.attributes_from_units(unit, physical, a_exponent)
                attrs["Description"] = description
                attrs.update(mask_metadata)
                attrs.update(compression_metadata)
                for attr_name, attr_value in attrs.items():
                    dataset.attrs[attr_name] = attr_value

            # Save the names of the groups containing the data
            subhalo_types = set()
            for metadata in ref_metadata + soap_metadata + fof_metadata:
                # Remove property name from full hdf5 path
                group_name = "/".join(metadata[0].split("/")[:-1])
                subhalo_types.add(group_name)
            header.attrs["SubhaloTypes"] = sorted(subhalo_types)

            # Save masks for each halo variation
            for halo_prop in halo_prop_list:
                for attr_name, attr_value in halo_prop.mask_metadata.items():
                    outfile[halo_prop.group_name].attrs[attr_name] = attr_value

            outfile.close()
    comm_world.barrier()

    # Reopen the output file in parallel mode
    outfile = h5py.File(output_file, "r+", driver="mpio", comm=comm_world)
    props_kept = {}

    with MPITimer("Writing output properties", comm_world):
        # Loop over halo properties, a few at a time
        total_nr_props = len(ref_metadata)
        props_per_iteration = min(
            total_nr_props, 100
        )  # TODO: how to choose this number?
        for i1 in range(0, total_nr_props, props_per_iteration):
            i2 = min(i1 + props_per_iteration, total_nr_props)

            # Find the properties to reorder on this iteration
            names = [metadata[0] for metadata in ref_metadata[i1:i2]]

            # Read in and reorder the properties
            data = scratch_file.read(names)
            for name in names:
                data[name] = psort.fetch_elements(data[name], order, comm=comm_world)

            # Keep a reference to any arrays we'll need later
            for name in names:
                if name in props_to_keep:
                    props_kept[name] = data[name]

            # Write these properties to the output file
            for name in names:
                phdf5.collective_write(
                    outfile, name, data[name], create_dataset=False, comm=comm_world
                )

            del data

    # Save the properties from the FOF catalogues
    if fof_metadata:
        # Extract units from FOF file
        if comm_world.Get_rank() == 0:
            with h5py.File(
                args.fof_group_filename.format(file_nr=0, snap_nr=args.snapshot_nr), "r"
            ) as fof_file:
                fof_reg = swift_units.unit_registry_from_snapshot(fof_file)
                fof_com_unit = swift_units.units_from_attributes(
                    dict(fof_file["Groups/Centres"].attrs), fof_reg
                )
                fof_mass_unit = swift_units.units_from_attributes(
                    dict(fof_file["Groups/Masses"].attrs), fof_reg
                )
        else:
            fof_reg = None
            fof_com_unit = None
            fof_mass_unit = None
        (fof_reg, fof_com_unit, fof_mass_unit) = comm_world.bcast(
            (fof_reg, fof_com_unit, fof_mass_unit)
        )

        # Open file in parallel
        pf = PartialFormatter()
        fof_filename = pf.format(
            args.fof_group_filename, snap_nr=args.snapshot_nr, file_nr=None
        )
        fof_file = phdf5.MultiFile(
            fof_filename,
            file_nr_attr=("Header", "NumFilesPerSnapshot"),
            comm=comm_world,
        )

        # Save data only for central halos which are not hostless
        keep = (props_kept["InputHalos/IsCentral"] == 1) & (
            props_kept["InputHalos/HBTplus/HostFOFId"] != -1
        )
        fof_ids = props_kept["InputHalos/HBTplus/HostFOFId"][keep]
        indices = psort.parallel_match(
            fof_ids, fof_file.read("Groups/GroupIDs"), comm=comm_world
        )
        # Assert that a FOF group has been found for all subhalos which should have one
        assert np.all(indices >= 0)

        fof_com = np.zeros((keep.shape[0], 3), dtype=np.float64)
        fof_com[keep] = psort.fetch_elements(
            fof_file.read("Groups/Centres"), indices, comm=comm_world
        )
        prop = PropertyTable.full_property_list[f"FOF/Centres"]
        soap_com_unit = cellgrid.get_unit(prop.unit)
        physical = prop.output_physical
        a_exponent = prop.a_scale_exponent
        if not physical:
            soap_com_unit = soap_com_unit * cellgrid.get_unit("a") ** a_exponent
        fof_com = (fof_com * fof_com_unit).to(soap_com_unit)
        phdf5.collective_write(
            outfile,
            "InputHalos/FOF/Centres",
            fof_com,
            create_dataset=False,
            comm=comm_world,
        )

        fof_mass = np.zeros(keep.shape[0], dtype=np.float64)
        fof_mass[keep] = psort.fetch_elements(
            fof_file.read("Groups/Masses"), indices, comm=comm_world
        )
        prop = PropertyTable.full_property_list[f"FOF/Masses"]
        soap_mass_unit = cellgrid.get_unit(prop.unit)
        physical = prop.output_physical
        a_exponent = prop.a_scale_exponent
        if not physical:
            soap_mass_unit = soap_mass_unit * cellgrid.get_unit("a") ** a_exponent
        fof_mass = (fof_mass * fof_mass_unit).to(soap_mass_unit)
        phdf5.collective_write(
            outfile,
            "InputHalos/FOF/Masses",
            fof_mass,
            create_dataset=False,
            comm=comm_world,
        )

        fof_size = np.zeros(keep.shape[0], dtype=np.int64)
        fof_size[keep] = psort.fetch_elements(
            fof_file.read("Groups/Sizes"), indices, comm=comm_world
        )
        phdf5.collective_write(
            outfile,
            "InputHalos/FOF/Sizes",
            fof_size,
            create_dataset=False,
            comm=comm_world,
        )

        # Handle radius differently since SWIFT did not always output radius
        # Assumes the FOF radii files are the same order as the other FOFs
        read_radii = 'InputHalos/FOF/Radii' in [m[0] for m in fof_metadata]
        if read_radii:
            # Open file in parallel
            pf = PartialFormatter()
            fof_filename = pf.format(
                args.fof_radius_filename, snap_nr=args.snapshot_nr, file_nr=None
            )
            fof_file = phdf5.MultiFile(
                fof_filename,
                file_nr_attr=("Header", "NumFilesPerSnapshot"),
                comm=comm_world,
            )

            fof_radii = np.zeros(keep.shape[0], dtype=np.float32)
            fof_radii[keep] = psort.fetch_elements(
                fof_file.read("Groups/Radii"), indices, comm=comm_world
            )
            prop = PropertyTable.full_property_list[f"FOF/Radii"]
            soap_radii_unit = cellgrid.get_unit(prop.unit)
            physical = prop.output_physical
            a_exponent = prop.a_scale_exponent
            if not physical:
                soap_radii_unit = soap_radii_unit * cellgrid.get_unit("a") ** a_exponent
            fof_radii = (fof_radii * fof_com_unit).to(soap_radii_unit)
            phdf5.collective_write(
                outfile,
                "InputHalos/FOF/Radii",
                fof_radii,
                create_dataset=False,
                comm=comm_world,
            )
        else:
            if comm_world.Get_rank() == 0:
                print("Groups/Radii not found in FOF files")

    # Calculate the index in the SOAP output of the host field halo (VR) or the central subhalo of the host FOF group (HBTplus)
    if "SOAP/HostHaloIndex" in soap_props:
        with MPITimer("Calculate and write host index of each satellite", comm_world):
            if args.halo_format == "VR":
                # We don't need to worry about hostless halos for VR
                host_ids = props_kept["InputHalos/VR/HostHaloID"].copy()
                cen_mask = host_ids == -1
                host_ids[cen_mask] = props_kept["InputHalos/VR/ID"][cen_mask]
                # If we run on an incomplete catalogue (e.g. for testing)
                # some satellites might point to themselves
                host_halo_index = psort.parallel_match(
                    host_ids, props_kept["InputHalos/VR/ID"], comm=comm_world
                )
            elif args.halo_format == "HBTplus":
                host_fof_id = props_kept["InputHalos/HBTplus/HostFOFId"]
                # Create array where FOF IDs are only set for centrals
                # so that we can match to it
                cen_fof_id = host_fof_id.copy()
                sat_mask = props_kept["InputHalos/IsCentral"] == 0
                cen_fof_id[sat_mask] = -1
                # We don't want to set HostHaloIndex for hostless halos
                has_host_mask = host_fof_id != -1
                host_ids = host_fof_id[has_host_mask]
                # If we run on an incomplete catalogue (e.g. for testing)
                # some satellites might point to themselves
                indices = psort.parallel_match(host_ids, cen_fof_id, comm=comm_world)
                host_halo_index = -1 * np.ones(sat_mask.shape[0], dtype=np.int64)
                host_halo_index[has_host_mask] = indices

            phdf5.collective_write(
                outfile,
                "SOAP/HostHaloIndex",
                host_halo_index,
                create_dataset=False,
                comm=comm_world,
            )
    else:
        if comm_world.Get_rank() == 0:
            print("Not calculating host halo index")

    # Write out subhalo ranking by mass within host halos, if we have all the required quantities.
    if "SOAP/SubhaloRankByBoundMass" in soap_props:
        with MPITimer("Calculate and write subhalo ranking by mass", comm_world):
            if args.halo_format == "VR":
                # Set field halos to be their own host (VR sets hostid=-1 in this case)
                field = props_kept["InputHalos/VR/HostHaloID"] < 0
                host_id = props_kept[
                    "InputHalos/VR/HostHaloID"
                ].copy()  # avoid modifying input
                host_id[field] = props_kept["InputHalos/VR/ID"][field]
            elif args.halo_format == "HBTplus":
                # Set hostless halos to have a unique FOF group by using -TrackId
                hostless = props_kept["InputHalos/HBTplus/HostFOFId"] < 0
                host_id = props_kept["InputHalos/HBTplus/HostFOFId"].copy()
                host_id[hostless] = -props_kept["InputHalos/HBTplus/TrackId"][hostless]
            subhalo_rank = compute_subhalo_rank(
                host_id, props_kept["BoundSubhalo/TotalMass"], comm_world
            )
            phdf5.collective_write(
                outfile,
                "SOAP/SubhaloRankByBoundMass",
                subhalo_rank,
                create_dataset=False,
                comm=comm_world,
            )
    else:
        if comm_world.Get_rank() == 0:
            print("Not calculating subhalo ranking by mass")

    # Determine which objects should be saved in the reduced snapshot files
    if "SOAP/IncludedInReducedSnapshot" in soap_props:
        with MPITimer("Calculate and write reduced snapshot membership", comm_world):
            # Load parameters. We create mass bins with the lower limit of the smallest mass bin
            # given by "min_halo_mass". The size of the bins is set by "halo_bin_size_dex".
            # For each bin we keep at most "halos_per_bin" objects.
            halos_per_bin = int(args.calculations["reduced_snapshots"]["halos_per_bin"])
            halo_bin_size = float(
                args.calculations["reduced_snapshots"]["halo_bin_size_dex"]
            )
            min_mass = np.log10(
                float(args.calculations["reduced_snapshots"]["min_halo_mass"])
            )

            # Load masses and convert to Msun
            mass_metadata = [
                metadata
                for metadata in ref_metadata
                if metadata[0] == "SO/200_crit/TotalMass"
            ]
            mass_unit = cellgrid.get_unit(mass_metadata[0][2])
            mass = (props_kept["SO/200_crit/TotalMass"] * mass_unit).to("Msun").value

            # Determine mass bins
            local_max_mass = np.max(mass) if mass.shape[0] else 0
            max_mass = comm_world.allreduce(local_max_mass, MPI.MAX)
            max_mass = np.log10(max_mass) + halo_bin_size
            bins = 10 ** np.arange(min_mass, max_mass, halo_bin_size)

            # Determine how many halos each rank should keep
            np.random.seed(0)
            n_halo_local, _ = np.histogram(mass, bins=bins)
            n_halo = np.array(comm_world.gather(n_halo_local))
            if comm_world.Get_rank() == 0:
                n_keep = n_halo.copy()
                n_halo_total = np.sum(n_halo, axis=0)
                for i_bin in range(bins.shape[0] - 1):
                    # Keep all halos in this bin
                    if n_halo_total[i_bin] <= halos_per_bin:
                        continue
                    # Add halos to a random rank until we have enough
                    n_keep[:, i_bin] = 0
                    p_keep = n_halo[:, i_bin] / n_halo_total[i_bin]
                    while np.sum(n_keep[:, i_bin]) < halos_per_bin:
                        i_rank = np.random.choice(comm_world.Get_size(), p=p_keep)
                        if n_keep[i_rank, i_bin] < n_halo[i_rank, i_bin]:
                            n_keep[i_rank, i_bin] += 1
            else:
                n_keep = None
            n_keep = comm_world.bcast(n_keep)[comm_world.Get_rank()]

            # Each rank determines which halos to keep
            reduced_snapshot = np.zeros(order.shape[0], dtype=np.int32)
            for i_bin in range(bins.shape[0] - 1):
                mask = (bins[i_bin] <= mass) & (mass < bins[i_bin + 1])
                idx = np.where(mask)[0]
                assert n_keep[i_bin] <= np.sum(mask)
                keep_idx = np.random.choice(idx, size=n_keep[i_bin], replace=False)
                reduced_snapshot[keep_idx] = 1
            phdf5.collective_write(
                outfile,
                "SOAP/IncludedInReducedSnapshot",
                reduced_snapshot,
                create_dataset=False,
                comm=comm_world,
            )
    else:
        if comm_world.Get_rank() == 0:
            print("Not calculating reduced snapshot membership")

    if "SOAP/ProgenitorIndex" in soap_props:
        assert "SOAP/DescendantIndex" in soap_props
        assert args.halo_format in ["HBTplus"]

        for name, snap_nr in [
            ("Progenitor", args.snapshot_nr - 1),
            ("Descendant", args.snapshot_nr + 1),
        ]:

            # Set the default value
            prev_index = -1 * np.ones(order.shape[0], dtype=np.int32)

            # Check if the previous/next HBT catalogue is available
            prev_basename = args.halo_basename.format(snap_nr=snap_nr)
            if comm_world.Get_rank() == 0:
                prev_filename = prev_basename + ".0.hdf5"
                if os.path.exists(prev_filename):
                    calculate_prev_index = True
                else:
                    print(f"Can't find halo catalogues for calculating {name}Index")
                    calculate_prev_index = False
            else:
                calculate_prev_index = None
            calculate_prev_index = comm_world.bcast(calculate_prev_index)

            if calculate_prev_index:
                track_id = props_kept["InputHalos/HBTplus/TrackId"]

                # Load data from previous/next halo catalogue and sort it
                # This assumes the metadata in the previous/next snapshot
                # is the same as in the current snapshot
                prev_data = read_hbtplus.read_hbtplus_catalogue(
                    comm_world,
                    prev_basename,
                    cellgrid.a_unit,
                    cellgrid.snap_unit_registry,
                    cellgrid.boxsize,
                )
                prev_order, _ = spatial_sort(
                    prev_data["cofp"],
                    prev_data["index"],
                    cellgrid,
                    comm_world,
                )
                prev_track_id = psort.fetch_elements(
                    prev_data["TrackId"], prev_order, comm=comm_world
                )
                # Find where each TrackId appears in the previous/next snapshot
                prev_index = psort.parallel_match(
                    track_id, prev_track_id, comm=comm_world
                )

            phdf5.collective_write(
                outfile,
                f"SOAP/{name}Index",
                prev_index,
                create_dataset=False,
                comm=comm_world,
            )

    # Done.
    outfile.close()
