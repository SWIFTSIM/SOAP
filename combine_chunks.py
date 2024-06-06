#!/bin/env python

import numpy as np
import h5py

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.util.partial_formatter import PartialFormatter

from subhalo_rank import compute_subhalo_rank
import swift_units
from mpi_timer import MPITimer
from property_table import PropertyTable


def sub_snapnum(filename, snapnum):
    """
    Substitute the snapshot number into a filename format string
    without substituting the file number.
    """
    from virgo.util.partial_formatter import PartialFormatter
    pf = PartialFormatter()
    filename = pf.format(filename, snap_nr=snapnum, file_nr=None)    
    return filename


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
):
    """
    Combine the per-chunk output files into a single, sorted output
    """

    # Open the per-chunk scratch files
    scratch_file = phdf5.MultiFile(
        scratch_file_format, file_idx=range(nr_chunks), comm=comm_world
    )

    # Read the halo index from the scratch files and make a sorting index to put them in order
    with MPITimer("Establishing ID ordering of halos", comm_world):
        halo_index = scratch_file.read("InputHalos/HaloCatalogueIndex")
        order = psort.parallel_sort(halo_index, return_index=True, comm=comm_world)
        del halo_index

    # Determine total number of halos
    total_nr_halos = comm_world.allreduce(len(order))

    # Get metadata for derived quantities: these don't exist in the chunk
    # output but will be computed by combining other halo properties.
    soap_metadata = []
    for soapkey in PropertyTable.soap_properties:
        props = PropertyTable.full_property_list[f"{soapkey}"]
        name = f"SOAP/{soapkey}"
        size = props[1]
        if size == 1:
            # Scalar quantity
            size = ()
        else:
            # Vector quantity
            size = (size,)
        dtype = props[2]
        unit = cellgrid.get_unit(props[3])
        description = props[4]
        soap_metadata.append((name, size, unit, dtype, description))

    # Add metadata for FOF properties
    fof_metadata = []
    if (args.fof_group_filename != '') and (args.halo_format == 'HBTplus'):
        for fofkey in ['Centres', 'Masses', 'Sizes']:
            props = PropertyTable.full_property_list[f"FOF/{fofkey}"]
            name = f"InputHalos/FOF/{fofkey}"
            size = props[1]
            if size == 1:
                # Scalar quantity
                size = ()
            else:
                # Vector quantity
                size = (size,)
            dtype = props[2]
            unit = cellgrid.get_unit(props[3])
            description = props[4]
            fof_metadata.append((name, size, unit, dtype, description))

    # First MPI rank sets up the output file
    with MPITimer("Creating output file", comm_world):
        output_file = sub_snapnum(args.output_file, args.snapshot_nr)
        if comm_world.Get_rank() == 0:
            # Create the file
            outfile = h5py.File(output_file, "w")
            # Write parameters etc
            cellgrid.write_metadata(outfile.create_group("SWIFT"))
            params = outfile.create_group("Parameters")
            params.attrs["swift_filename"] = args.swift_filename
            params.attrs["halo_basename"] = args.halo_basename
            params.attrs["halo_format"] = args.halo_format
            params.attrs["snapshot_nr"] = args.snapshot_nr
            params.attrs["centrals_only"] = 0 if args.centrals_only == False else 1
            calc_names = sorted([hp.name for hp in halo_prop_list])
            params.attrs["calculations"] = calc_names
            params.attrs["halo_indices"] = (
                args.halo_indices if args.halo_indices is not None else np.ndarray(0, dtype=int)
            )
            params.attrs["git_hash"] = args.git_hash
            # NOTE: FLAMINGO
            recently_heated_gas_metadata = recently_heated_gas_filter.get_metadata()
            recently_heated_gas_params = outfile.create_group("RecentlyHeatedGasFilter")
            for at, val in recently_heated_gas_metadata.items():
                recently_heated_gas_params.attrs[at] = val

            # Create datasets for all halo properties
            for name, size, unit, dtype, description in ref_metadata + soap_metadata + fof_metadata:
                if description == 'No description available':
                    print(f'{name} not found in property table')
                shape = (total_nr_halos,) + size
                dataset = outfile.create_dataset(
                    name, shape=shape, dtype=dtype, fillvalue=None
                )
                # Add units and description
                attrs = swift_units.attributes_from_units(unit)
                attrs["Description"] = description
                mask_metadata = category_filter.get_filter_metadata(name)
                attrs.update(mask_metadata)
                compression_metadata = category_filter.get_compression_metadata(name)
                attrs.update(compression_metadata)
                for attr_name, attr_value in attrs.items():
                    dataset.attrs[attr_name] = attr_value
            outfile.close()
    comm_world.barrier()

    # Reopen the output file in parallel mode
    outfile = h5py.File(output_file, "r+", driver="mpio", comm=comm_world)

    # Certain properties need to be kept for calculating the SOAP properties
    subhalo_rank_props = {
        'VR': ("InputHalos/VR/ID", "BoundSubhalo/TotalMass", "InputHalos/VR/HostHaloID"),
        'HBTplus': ("InputHalos/HBTplus/HostFOFId", "BoundSubhalo/TotalMass", "InputHalos/HBTplus/TrackId"),
    }.get(args.halo_format, ())
    host_halo_index_props = {
        'VR': ("InputHalos/VR/ID", "InputHalos/VR/HostHaloID"),
        'HBTplus': ("InputHalos/HBTplus/HostFOFId", "InputHalos/IsCentral"),
    }.get(args.halo_format, ())
    fof_props = {
        'HBTplus': ("InputHalos/HBTplus/HostFOFId", "InputHalos/IsCentral"),
    }.get(args.halo_format, ())
    props_to_keep = set((*subhalo_rank_props, *host_halo_index_props, *fof_props))
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
            names, sizes, units, dtypes, descriptions = zip(*ref_metadata[i1:i2])

            # Read in and reorder the properties
            data = scratch_file.read(names)
            for name in names:
                data[name] = psort.fetch_elements(data[name], order, comm=comm_world)

            # Keep a reference to any arrays we'll need later
            for name in names:
                if name in props_to_keep:
                    props_kept[name] = data[name]

            # Write these properties to the output file
            for name, size, unit, description in zip(names, sizes, units, descriptions):
                phdf5.collective_write(
                    outfile, name, data[name], create_dataset=False, comm=comm_world
                )

            del data

    # Save the properties from the FOF catalogues
    if fof_metadata:
        # Extract units from FOF file
        if comm_world.Get_rank() == 0:
            with h5py.File(args.fof_group_filename.format(file_nr=0, snap_nr= args.snapshot_nr), "r") as fof_file:
                fof_reg = swift_units.unit_registry_from_snapshot(fof_file)
                fof_com_unit = swift_units.units_from_attributes(fof_file['Groups/Centres'].attrs, fof_reg)
                fof_mass_unit = swift_units.units_from_attributes(fof_file['Groups/Masses'].attrs, fof_reg)
        else:
            fof_reg = None
            fof_com_unit = None
            fof_mass_unit = None
        (fof_reg, fof_com_unit, fof_mass_unit) = comm_world.bcast((fof_reg, fof_com_unit, fof_mass_unit))
            
        # Open file in parallel
        pf = PartialFormatter()
        fof_filename = pf.format(args.fof_group_filename, snap_nr=args.snapshot_nr, file_nr=None)
        fof_file = phdf5.MultiFile(
            fof_filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm_world
        )

        # Save data only for central halos which are not hostless
        keep = (props_kept["InputHalos/IsCentral"] == 1) & (props_kept["InputHalos/HBTplus/HostFOFId"] != -1)
        fof_ids = props_kept["InputHalos/HBTplus/HostFOFId"][keep]
        indices = psort.parallel_match(fof_ids, fof_file.read('Groups/GroupIDs'), comm=comm_world)
        # Assert that a FOF group has been found for all subhalos which should have one
        assert np.all(indices >= 0)

        fof_com = np.zeros((keep.shape[0], 3), dtype=np.float64)
        fof_com[keep] = psort.fetch_elements(fof_file.read('Groups/Centres'), indices, comm=comm_world)
        props = PropertyTable.full_property_list[f"FOF/Centres"]
        fof_com = (fof_com * fof_com_unit).to(cellgrid.get_unit(props[3]))
        dataset = phdf5.collective_write(outfile, "InputHalos/FOF/Centres", fof_com, create_dataset=False, comm=comm_world)

        fof_mass = np.zeros(keep.shape[0], dtype=np.float64)
        fof_mass[keep] = psort.fetch_elements(fof_file.read('Groups/Masses'), indices, comm=comm_world)
        props = PropertyTable.full_property_list[f"FOF/Masses"]
        fof_mass = (fof_mass * fof_mass_unit).to(cellgrid.get_unit(props[3]))
        dataset = phdf5.collective_write(outfile, "InputHalos/FOF/Masses", fof_mass, create_dataset=False, comm=comm_world)

        fof_size = np.zeros(keep.shape[0], dtype=np.int64)
        fof_size[keep] = psort.fetch_elements(fof_file.read('Groups/Sizes'), indices, comm=comm_world)
        dataset = phdf5.collective_write(outfile, "InputHalos/FOF/Sizes", fof_size, create_dataset=False, comm=comm_world)

    # Calculate the index in the SOAP output of the host field halo (VR) or the central subhalo of the host FOF group (HBTplus)
    if len(host_halo_index_props) > 0:
        with MPITimer("Calculate and write host index of each satellite", comm_world):
            if args.halo_format == 'VR':
                sat_mask = props_kept["InputHalos/VR/HostHaloID"] != -1
                host_ids = props_kept["InputHalos/VR/HostHaloID"][sat_mask]
                # If we run on an incomplete catalogue (e.g. for testing) some satellites will have an index == -1
                indices = psort.parallel_match(host_ids, props_kept["InputHalos/VR/ID"], comm=comm_world)
                host_halo_index = -1 * np.ones(sat_mask.shape[0], dtype=np.int64)
                host_halo_index[sat_mask] = indices
            elif args.halo_format == 'HBTplus':
                # Create array where FOF IDs are only set for centrals, so we can match to it
                cen_fof_id = props_kept["InputHalos/HBTplus/HostFOFId"].copy()
                sat_mask = props_kept["InputHalos/IsCentral"] == 0
                cen_fof_id[sat_mask] = -1
                host_ids = props_kept["InputHalos/HBTplus/HostFOFId"][sat_mask]
                # If we run on an incomplete catalogue (e.g. for testing) some satellites will have an index == -1
                indices = psort.parallel_match(host_ids, cen_fof_id, comm=comm_world)
                host_halo_index = -1 * np.ones(sat_mask.shape[0], dtype=np.int64)
                host_halo_index[sat_mask] = indices

            dataset = phdf5.collective_write(
                outfile,
                "SOAP/HostHaloIndex",
                host_halo_index,
                create_dataset=False,
                comm=comm_world,
            )

    # Now write out subhalo ranking by mass within host halos, if we have all the required quantities.
    if len(subhalo_rank_props) > 0:
        with MPITimer("Calculate and write subhalo ranking by mass", comm_world):
            if args.halo_format == 'VR':
                # Set field halos to be their own host (VR sets hostid=-1 in this case)
                field = props_kept["InputHalos/VR/HostHaloID"] < 0
                host_id = props_kept["InputHalos/VR/HostHaloID"].copy() # avoid modifying input
                host_id[field] = props_kept["InputHalos/VR/ID"][field]
            elif args.halo_format == 'HBTplus':
                # Set hostless halos to have a unique FOF group by using -TrackId
                hostless = props_kept["InputHalos/HBTplus/HostFOFId"] < 0
                host_id = props_kept["InputHalos/HBTplus/HostFOFId"].copy()
                host_id[hostless] = -props_kept["InputHalos/HBTplus/TrackId"][hostless]
            subhalo_rank = compute_subhalo_rank(
                host_id,
                props_kept["BoundSubhalo/TotalMass"],
                comm_world,
            )
            dataset = phdf5.collective_write(
                outfile,
                "SOAP/SubhaloRankByBoundMass",
                subhalo_rank,
                create_dataset=False,
                comm=comm_world,
            )
    else:
        if comm_world.Get_rank() == 0:
            print('Not calculating subhalo ranking by mass')

    # Done.
    outfile.close()
