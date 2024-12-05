#!/bin/env python

# Initialize mpi4py with thread support
import mpi4py

mpi4py.rc.threads = True
from mpi4py import MPI

comm_world = MPI.COMM_WORLD
comm_world_rank = comm_world.Get_rank()
comm_world_size = comm_world.Get_size()

import os
import os.path
import time
import traceback
import numpy as np
import unyt


import halo_centres
import swift_cells
import chunk_tasks
import task_queue
import lustre
import soap_args
import SO_properties
import subhalo_properties
import aperture_properties
import result_set
from combine_chunks import combine_chunks, sub_snapnum
import projected_aperture_properties
from recently_heated_gas_filter import RecentlyHeatedGasFilter
from stellar_age_calculator import StellarAgeCalculator
from cold_dense_gas_filter import ColdDenseGasFilter
from category_filter import CategoryFilter
from parameter_file import ParameterFile
from mpi_timer import MPITimer


# Set numpy to raise divide by zero, overflow and invalid operation errors as exceptions
np.seterr(divide="raise", over="raise", invalid="raise")


def split_comm_world():

    # Communicator containing all ranks on this node
    comm_intra_node = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    comm_intra_node_rank = comm_intra_node.Get_rank()

    # Communicator containing first rank on each node only:
    # other ranks will have comm_inter_node=MPI_COMM_NULL.
    colour = 0 if comm_intra_node_rank == 0 else MPI.UNDEFINED
    key = MPI.COMM_WORLD.Get_rank()
    comm_inter_node = MPI.COMM_WORLD.Split(colour, key)
    return comm_intra_node, comm_inter_node


def get_rank_and_size(comm):
    if comm == MPI.COMM_NULL:
        return (-1, -1)
    else:
        return (comm.Get_rank(), comm.Get_size())


def compute_halo_properties():

    # Read command line parameters
    args = soap_args.get_soap_args(comm_world)

    # Enable profiling, if requested
    if args.profile == 2 or (args.profile == 1 and comm_world_rank == 0):
        import cProfile, pstats, io

        pr = cProfile.Profile()
        pr.enable()

    # Start the clock
    comm_world.barrier()
    t0 = time.time()

    # Split MPI ranks according to which node they are on.
    # Only the first rank on each node belongs to comm_inter_node.
    # Others have comm_inter_node=MPI_COMM_NULL and inter_node_rank=-1.
    comm_intra_node, comm_inter_node = split_comm_world()
    intra_node_rank, intra_node_size = get_rank_and_size(comm_intra_node)
    inter_node_rank, inter_node_size = get_rank_and_size(comm_inter_node)

    # Report number of ranks, compute nodes etc
    if comm_world_rank == 0:
        print("Starting halo properties calculation on %d MPI ranks" % comm_world_size)
        print(
            "Can process %d chunks in parallel using %d ranks per chunk"
            % (inter_node_size, intra_node_size)
        )
        print(
            "Number of MPI ranks per node reading snapshots: %d"
            % args.max_ranks_reading
        )
        print("Halo format is %s" % args.halo_format)
        print("Halo basename is %s" % args.halo_basename)
        print("Output file is %s" % args.output_file)
        print("Snapshot number is %d" % args.snapshot_nr)

    # Open the snapshot and read SWIFT cell structure, units etc
    if comm_world_rank == 0:
        swift_filename = sub_snapnum(args.swift_filename, args.snapshot_nr)
        extra_input = [
            sub_snapnum(filename, args.snapshot_nr) for filename in args.extra_input
        ]
        if args.reference_snapshot is not None:
            swift_filename_ref = sub_snapnum(
                args.swift_filename, args.reference_snapshot
            )
            extra_input_ref = [
                sub_snapnum(filename, args.reference_snapshot)
                for filename in args.extra_input
            ]
        else:
            swift_filename_ref = None
            extra_input_ref = None
        try:
            cellgrid = swift_cells.SWIFTCellGrid(
                swift_filename, extra_input, swift_filename_ref, extra_input_ref
            )
        except Exception as err_msg:
            print(err_msg)
            # Thrown if there are issues with the input files
            comm_world.Abort(1)
        parsec_cgs = cellgrid.constants["parsec"]
        solar_mass_cgs = cellgrid.constants["solar_mass"]
        a = cellgrid.a
    else:
        cellgrid = None
        parsec_cgs = None
        solar_mass_cgs = None
        a = None
    cellgrid, parsec_cgs, solar_mass_cgs, a = comm_world.bcast(
        (cellgrid, parsec_cgs, solar_mass_cgs, a)
    )

    # Check that the extra-input files are valid
    cellgrid.verify_extra_input(comm_world)

    # Process parameter file
    if args.snipshot is None:
        args.snipshot = cellgrid.snipshot
    if comm_world_rank == 0:
        parameter_file = ParameterFile(
            file_name=args.config_filename, snipshot=args.snipshot
        )
    else:
        parameter_file = None
    parameter_file = comm_world.bcast(parameter_file)
    cellgrid.snapshot_datasets.setup_aliases(parameter_file.get_aliases())
    cellgrid.snapshot_datasets.setup_defined_constants(
        parameter_file.get_defined_constants()
    )
    parameter_file.record_property_timings = args.record_property_timings

    # Try to load parameters for RecentlyHeatedGasFilter. If a property that uses the
    # filter is calculated when the parameters could not be found, the code will
    # crash.
    try:
        recently_heated_params = args.calculations["recently_heated_gas_filter"]
        if (not args.dmo) and (recently_heated_params["use_AGN_delta_T"]):
            assert cellgrid.AGN_delta_T.value != 0, "Invalid value for AGN_delta_T"
        recently_heated_gas_filter = RecentlyHeatedGasFilter(
            cellgrid,
            float(recently_heated_params["delta_time_myr"]) * unyt.Myr,
            float(recently_heated_params["use_AGN_delta_T"]),
            True,
            delta_logT_min=-1.0,
            delta_logT_max=0.3,
        )
    except KeyError:
        recently_heated_gas_filter = RecentlyHeatedGasFilter(
            cellgrid,
            0 * unyt.Myr,
            False,
            False,
        )

    stellar_age_calculator = StellarAgeCalculator(cellgrid)

    # Try to load parameters for ColdDenseGasFilter. If a property that uses the
    # filter is calculated when the parameters could not be found, the code will
    # crash.
    try:
        cold_dense_params = args.calculations["cold_dense_gas_filter"]
        cold_dense_gas_filter = ColdDenseGasFilter(
            float(cold_dense_params["maximum_temperature_K"]) * unyt.K,
            float(cold_dense_params["minimum_hydrogen_number_density_cm3"])
            / unyt.cm ** 3,
            True,
        )
    except KeyError:
        cold_dense_gas_filter = ColdDenseGasFilter(0 * unyt.K, 0 / unyt.cm ** 3, False)

    default_filters = {
        "general": {
            "limit": 100,
            "properties": [
                "BoundSubhalo/NumberOfDarkMatterParticles",
                "BoundSubhalo/NumberOfGasParticles",
                "BoundSubhalo/NumberOfStarParticles",
                "BoundSubhalo/NumberOfBlackHoleParticles",
            ],
            "combine_properties": "sum",
        },
        "dm": {
            "limit": 100,
            "properties": ["BoundSubhalo/NumberOfDarkMatterParticles"],
        },
        "gas": {"limit": 100, "properties": ["BoundSubhalo/NumberOfGasParticles"]},
        "star": {"limit": 100, "properties": ["BoundSubhalo/NumberOfStarParticles"]},
        "baryon": {
            "limit": 100,
            "properties": [
                "BoundSubhalo/NumberOfGasParticles",
                "BoundSubhalo/NumberOfStarParticles",
            ],
            "combine_properties": "sum",
        },
    }
    category_filter = CategoryFilter(
        parameter_file.get_filters(default_filters), dmo=args.dmo
    )

    # Get the full list of property calculations we can do
    # Note that the order matters: we need to do the BoundSubhalo first,
    # since quantities are filtered based on the particle numbers in there
    # Similarly, things like SO 5xR500_crit can only be done after
    # SO 500_crit for obvious reasons
    halo_prop_list = []

    # We require BoundSubhalo since it's used for filters
    if comm_world_rank == 0:
        if "SubhaloProperties" not in parameter_file.parameters:
            print('SubhaloProperties must be in the parameter file')
            comm_world.Abort(1)
    halo_prop_list.append(
        subhalo_properties.SubhaloProperties(
            cellgrid,
            parameter_file,
            recently_heated_gas_filter,
            stellar_age_calculator,
            category_filter,
        )
    )

    SO_variations = parameter_file.get_halo_type_variations(
        "SOProperties",
        {
            "200_mean": {"value": 200.0, "type": "mean"},
            "50_crit": {"value": 50.0, "type": "crit"},
            "100_crit": {"value": 100.0, "type": "crit"},
            "200_crit": {"value": 200.0, "type": "crit"},
            "500_crit": {"value": 500.0, "type": "crit"},
            "1000_crit": {"value": 1000.0, "type": "crit"},
            "2500_crit": {"value": 2500.0, "type": "crit"},
            "BN98": {"value": 0.0, "type": "BN98"},
            "5xR500_crit": {"value": 500.0, "type": "crit", "radius_multiple": 5.0},
        },
    )
    # first add non radius multiples to make sure the radius multiples can be
    # computed
    for variation in SO_variations:
        if (
            "radius_multiple" in SO_variations[variation]
            and SO_variations[variation]["radius_multiple"] > 0.0
        ):
            continue
        if "core_excision_fraction" in SO_variations[variation]:
            halo_prop_list.append(
                SO_properties.CoreExcisedSOProperties(
                    cellgrid,
                    parameter_file,
                    recently_heated_gas_filter,
                    category_filter,
                    SO_variations[variation].get("filter", "basic"),
                    SO_variations[variation]["value"],
                    SO_variations[variation]["type"],
                    core_excision_fraction=SO_variations[variation][
                        "core_excision_fraction"
                    ],
                )
            )
        else:
            halo_prop_list.append(
                SO_properties.SOProperties(
                    cellgrid,
                    parameter_file,
                    recently_heated_gas_filter,
                    category_filter,
                    SO_variations[variation].get("filter", "basic"),
                    SO_variations[variation]["value"],
                    SO_variations[variation]["type"],
                )
            )

    for variation in SO_variations:
        if (
            "radius_multiple" in SO_variations[variation]
            and SO_variations[variation]["radius_multiple"] > 0.0
        ):
            halo_prop_list.append(
                SO_properties.RadiusMultipleSOProperties(
                    cellgrid,
                    parameter_file,
                    recently_heated_gas_filter,
                    category_filter,
                    SO_variations[variation].get("filter", "basic"),
                    SO_variations[variation]["value"],
                    SO_variations[variation]["radius_multiple"],
                    SO_variations[variation]["type"],
                )
            )

    aperture_variations = parameter_file.get_halo_type_variations(
        "ApertureProperties",
        {
            "inclusive_10_kpc": {"radius_in_kpc": 10.0, "inclusive": True},
            "inclusive_30_kpc": {"radius_in_kpc": 30.0, "inclusive": True},
            "inclusive_50_kpc": {"radius_in_kpc": 50.0, "inclusive": True},
            "inclusive_100_kpc": {"radius_in_kpc": 100.0, "inclusive": True},
            "inclusive_300_kpc": {"radius_in_kpc": 300.0, "inclusive": True},
            "inclusive_500_kpc": {"radius_in_kpc": 500.0, "inclusive": True},
            "inclusive_1000_kpc": {"radius_in_kpc": 1000.0, "inclusive": True},
            "inclusive_3000_kpc": {"radius_in_kpc": 3000.0, "inclusive": True},
            "exclusive_10_kpc": {"radius_in_kpc": 10.0, "inclusive": False},
            "exclusive_30_kpc": {"radius_in_kpc": 30.0, "inclusive": False},
            "exclusive_50_kpc": {"radius_in_kpc": 50.0, "inclusive": False},
            "exclusive_100_kpc": {"radius_in_kpc": 100.0, "inclusive": False},
            "exclusive_300_kpc": {"radius_in_kpc": 300.0, "inclusive": False},
            "exclusive_500_kpc": {"radius_in_kpc": 500.0, "inclusive": False},
            "exclusive_1000_kpc": {"radius_in_kpc": 1000.0, "inclusive": False},
            "exclusive_3000_kpc": {"radius_in_kpc": 3000.0, "inclusive": False},
        },
    )
    for variation in aperture_variations:
        if aperture_variations[variation]["inclusive"]:
            halo_prop_list.append(
                aperture_properties.InclusiveSphereProperties(
                    cellgrid,
                    parameter_file,
                    aperture_variations[variation]["radius_in_kpc"],
                    recently_heated_gas_filter,
                    stellar_age_calculator,
                    cold_dense_gas_filter,
                    category_filter,
                    aperture_variations[variation].get("filter", "basic"),
                )
            )
        else:
            halo_prop_list.append(
                aperture_properties.ExclusiveSphereProperties(
                    cellgrid,
                    parameter_file,
                    aperture_variations[variation]["radius_in_kpc"],
                    recently_heated_gas_filter,
                    stellar_age_calculator,
                    cold_dense_gas_filter,
                    category_filter,
                    aperture_variations[variation].get("filter", "basic"),
                )
            )
    projected_aperture_variations = parameter_file.get_halo_type_variations(
        "ProjectedApertureProperties",
        {
            "10_kpc": {"radius_in_kpc": 10.0},
            "30_kpc": {"radius_in_kpc": 30.0},
            "50_kpc": {"radius_in_kpc": 50.0},
            "100_kpc": {"radius_in_kpc": 100.0},
        },
    )
    for variation in projected_aperture_variations:
        halo_prop_list.append(
            projected_aperture_properties.ProjectedApertureProperties(
                cellgrid,
                parameter_file,
                projected_aperture_variations[variation]["radius_in_kpc"],
                category_filter,
                projected_aperture_variations[variation].get("filter", "basic"),
            )
        )

    if comm_world_rank == 0 and args.output_parameters:
        parameter_file.write_parameters(args.output_parameters)

    if len(halo_prop_list) < 1:
        raise Exception("Must select at least one halo property calculation!")

    # Report calculations to do
    if comm_world_rank == 0:
        print("Halo property calculations enabled:")
        for hp in halo_prop_list:
            print("  %s" % hp.name)
        if args.centrals_only:
            print("for central halos only")
        else:
            print("for central and satellite halos")
        if args.snipshot:
            print("Running in snipshot mode")
        if args.record_halo_timings:
            print("Storing processing time for each halo")
        if args.record_property_timings:
            print("Storing processing time for each property")
        parameter_file.print_unregistered_properties()
        parameter_file.print_invalid_properties()
        category_filter.print_filters()

    # Ensure output dir exists
    if comm_world_rank == 0:
        lustre.ensure_output_dir(args.output_file)
    comm_world.barrier()

    # Read in the halo catalogue:
    # All ranks read the file(s) in then gather to rank 0. Also computes search radius for each halo.
    halo_basename = sub_snapnum(args.halo_basename, args.snapshot_nr)
    so_cat = halo_centres.SOCatalogue(
        comm_world,
        cellgrid.a_unit,
        cellgrid.snap_unit_registry,
        cellgrid.boxsize,
        halo_prop_list,
        args,
    )
    so_cat.start_request_thread()

    # Generate the chunk task list
    nr_chunks = so_cat.nr_chunks
    if comm_world_rank == 0:
        tasks = [
            chunk_tasks.ChunkTask(halo_prop_list, chunk_nr, nr_chunks)
            for chunk_nr in range(nr_chunks)
        ]
    else:
        tasks = None

    # Report initial set-up time
    comm_world.barrier()
    t1 = time.time()
    if comm_world_rank == 0:
        print(
            "Reading %d input halos and setting up %d chunk(s) took %.1fs"
            % (so_cat.nr_halos, len(tasks), t1 - t0)
        )

    # Make a format string to generate the name of the file each chunk task will write to
    scratch_file_format = (
        args.scratch_dir
        + f"/snapshot_{args.snapshot_nr:04d}/"
        + "chunk_%(file_nr)d.hdf5"
    )

    # Ensure that the directories which will contain the scratch files exist
    if comm_world_rank == 0:
        for file_nr in range(nr_chunks):
            scratch_file_name = scratch_file_format % {"file_nr": file_nr}
            scratch_file_dir = os.path.dirname(scratch_file_name)
            try:
                os.makedirs(scratch_file_dir)
            except OSError:
                pass
    comm_world.barrier()

    # Execute the chunk tasks. This writes one file per chunk with the halo properties.
    # For each chunk it returns a list with (name, size, units, description) for each
    # quantity that was calculated.
    timings = []
    task_args = (
        cellgrid,
        so_cat,
        comm_intra_node,
        inter_node_rank,
        timings,
        args.max_ranks_reading,
        scratch_file_format,
    )
    # Catch any errors so we can call MPI_ABORT
    try:
        metadata = task_queue.execute_tasks(
            tasks,
            args=task_args,
            comm_all=comm_world,
            comm_master=comm_inter_node,
            comm_workers=comm_intra_node,
        )
    except Exception as e:
        traceback.print_exc()
        comm_world.Abort(1)

    # Can stop the halo request thread now that all chunk tasks have executed
    so_cat.stop_request_thread()

    # Check metadata for consistency between chunks. Sets ref_metadata on all ranks,
    # including those that processed no halos.
    ref_metadata = result_set.check_metadata(metadata, comm_inter_node, comm_world)

    # Combine chunks into a single output file
    combine_chunks(
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
    )

    # Delete scratch files
    comm_world.barrier()
    if comm_world_rank == 0:
        for file_nr in range(nr_chunks):
            os.remove(scratch_file_format % {"file_nr": file_nr})
        print("Deleted scratch files.")
    comm_world.barrier()

    # Stop the clock
    comm_world.barrier()
    t1 = time.time()

    # Find total time spent running tasks
    if len(timings) > 0:
        task_time_local = sum(timings)
    else:
        task_time_local = 0.0
    task_time_total = comm_world.allreduce(task_time_local)
    task_time_fraction = task_time_total / (comm_world_size * (t1 - t0))

    # Save profiling results for each MPI rank
    if args.profile == 2 or (args.profile == 1 and comm_world_rank == 0):
        pr.disable()
        # Save profile so it can be loaded back into python for analysis
        pr.dump_stats("./profile.%d.dat" % comm_world_rank)
        # Dump text version of the profile
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open("./profile.%d.txt" % comm_world_rank, "w") as profile_file:
            profile_file.write(s.getvalue())

    if comm_world_rank == 0:
        print(
            "Fraction of time spent calculating halo properties = %.2f"
            % task_time_fraction
        )
        print("Total elapsed time: %.1f seconds" % (t1 - t0))
        print("Done.")


if __name__ == "__main__":

    compute_halo_properties()
