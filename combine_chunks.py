#!/bin/env python

import numpy as np
import h5py
import unyt

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

from subhalo_rank import compute_subhalo_rank
import swift_units
from mpi_timer import MPITimer
from property_table import PropertyTable


def sub_snapnum(filename, snapnum):
    """
    Substitute the snapshot number into a filename format string
    without substituting the file number.
    """
    filename = filename.replace("%(file_nr)", "%%(file_nr)")
    return filename % {"snap_nr": snapnum}


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

    # Read the VR halo IDs from the scratch files and make a sorting index to put them in order
    with MPITimer("Establishing ID ordering of halos", comm_world):
        vr_id = scratch_file.read(("VR/ID",))["VR/ID"]
        order = psort.parallel_sort(vr_id, return_index=True, comm=comm_world)
        del vr_id

    # Determine total number of halos
    total_nr_halos = comm_world.allreduce(len(order))

    # Get metadata for derived quantities: these don't exist in the chunk
    # output but will be computed by combining other halo properties.
    soap_metadata = []
    for soapkey in PropertyTable.soap_properties:
        props = PropertyTable.full_property_list[f"SOAP{soapkey}"]
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
            params.attrs["vr_basename"] = args.vr_basename
            params.attrs["snapshot_nr"] = args.snapshot_nr
            params.attrs["centrals_only"] = 0 if args.centrals_only == False else 1
            calc_names = sorted([hp.name for hp in halo_prop_list])
            params.attrs["calculations"] = calc_names
            params.attrs["halo_ids"] = (
                args.halo_ids if args.halo_ids is not None else np.ndarray(0, dtype=int)
            )
            params.attrs["git_hash"] = args.git_hash
            # NOTE: FLAMINGO
            recently_heated_gas_metadata = recently_heated_gas_filter.get_metadata()
            recently_heated_gas_params = outfile.create_group("RecentlyHeatedGasFilter")
            for at, val in recently_heated_gas_metadata.items():
                recently_heated_gas_params.attrs[at] = val

            # Create datasets for all halo properties
            for name, size, unit, dtype, description in ref_metadata + soap_metadata:
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

    # Certain properties are needed to compute subhalo ranking by mass
    props_to_keep = ("VR/ID", "BoundSubhaloProperties/TotalMass", "VR/HostHaloID")
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

    with MPITimer("Writing subhalo ranking by mass", comm_world):
        # Now write out subhalo ranking by mass within host halos, if we computed all the required quantities.
        if len(props_kept) == len(props_to_keep):
            subhalo_rank = compute_subhalo_rank(
                props_kept["VR/HostHaloID"],
                props_kept["VR/ID"],
                props_kept["BoundSubhaloProperties/TotalMass"],
                comm_world,
            )
            dataset = phdf5.collective_write(
                outfile,
                "SOAP/SubhaloRankByBoundMass",
                subhalo_rank,
                create_dataset=False,
                comm=comm_world,
            )

    # Done.
    outfile.close()
