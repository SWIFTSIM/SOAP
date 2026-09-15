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
import sys
import time
import traceback
import numpy as np
import unyt

from SOAP.core import (
    chunk_tasks,
    halo_centres,
    lustre,
    result_set,
    soap_args,
    swift_cells,
    task_queue,
)
from SOAP.core.parameter_file import ParameterFile
from SOAP.core.mpi_timer import MPITimer
from SOAP.core.combine_chunks import combine_chunks, sub_snapnum
from SOAP.core.category_filter import CategoryFilter
from SOAP.particle_selection import (
    aperture_properties,
    SO_properties,
    projected_aperture_properties,
    subhalo_properties,
)
from SOAP.property_calculation.stellar_age_calculator import StellarAgeCalculator
from SOAP.particle_filter.cold_dense_gas_filter import ColdDenseGasFilter
from SOAP.particle_filter.recently_heated_gas_filter import RecentlyHeatedGasFilter

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

    # Start the clock
    t0 = time.time()

    # Read command line parameters
    args = soap_args.get_soap_args(comm_world)
    comm_world.barrier()

    # Enable profiling, if requested
    if args.profile == 2 or (args.profile == 1 and comm_world_rank == 0):
        import cProfile, pstats, io

        pr = cProfile.Profile()
        pr.enable()

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
            print(err_msg, flush=True)
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
        try:
            parameter_file.check_schema()
        except ValueError as e:
            print(e, flush=True)
            comm_world.Abort(1)
    else:
        parameter_file = None
    parameter_file = comm_world.bcast(parameter_file)
    cellgrid.snapshot_datasets.setup_aliases(parameter_file.get_aliases())
    cellgrid.snapshot_datasets.setup_defined_constants(
        parameter_file.get_defined_constants()
    )
    # Tell the parameter file which datasets are in the input files, so that
    # properties which cannot be computed can be skipped or reported
    parameter_file.set_available_datasets(cellgrid.snapshot_datasets.datasets_in_file)
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
    cold_dense_params = parameter_file.get_cold_dense_params()
    cold_dense_gas_filter = ColdDenseGasFilter(
        cold_dense_params["maximum_temperature_K"] * unyt.K,
        cold_dense_params["minimum_hydrogen_number_density_cm3"] / unyt.cm**3,
        cold_dense_params["initialised"],
    )

    filters = parameter_file.get_filters()
    for filter_name, filter_info in filters.items():
        for prop in filter_info.get("properties", []):
            assert prop.split("/")[0] == "BoundSubhalo", (
                f'Filter "{filter_name}" uses "{prop}", but filters can only '
                "use BoundSubhalo properties."
            )
    category_filter = CategoryFilter(filters, dmo=args.dmo)

    # Get the full list of property calculations we can do
    # Each kind of calculation is collected separately so that the final list
    # can be built in a deliberate order (see where it is assembled below),
    # rather than relying on the order things happen to be created in.
    subhalo_props = []
    so_props = []
    exclusive_apertures = []
    inclusive_apertures = []
    projected_apertures = []

    # We require BoundSubhalo since it's used for filters
    if comm_world_rank == 0:
        if "SubhaloProperties" not in parameter_file.parameters:
            print("SubhaloProperties must be in the parameter file", flush=True)
            comm_world.Abort(1)
    subhalo_props.append(
        subhalo_properties.SubhaloProperties(
            cellgrid,
            parameter_file,
            recently_heated_gas_filter,
            stellar_age_calculator,
            category_filter,
        )
    )

    SO_variations = parameter_file.get_halo_type_variations("SOProperties")
    # first add non radius multiples to make sure the radius multiples can be
    # computed
    for variation in SO_variations:
        if (
            "radius_multiple" in SO_variations[variation]
            and SO_variations[variation]["radius_multiple"] > 0.0
        ):
            continue
        if "core_excision_fraction" in SO_variations[variation]:
            so_props.append(
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
            so_props.append(
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
            so_props.append(
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

    aperture_variations = parameter_file.get_halo_type_variations("ApertureProperties")

    # Sort the aperture variations based on their radii, and create a list
    # of all apertures. This is required since we can skip some of the larger
    # apertures if all the particles were already included in the previous aperture
    aperture_variations = dict(
        sorted(aperture_variations.items(), key=lambda x: x[1].get("radius_in_kpc", 0))
    )
    inclusive_radii_kpc = []
    exclusive_radii_kpc = []
    for variation in aperture_variations:
        # We don't consider this for apertures defined based on properties
        if "radius_in_kpc" not in aperture_variations[variation]:
            continue
        if aperture_variations[variation]["inclusive"]:
            inclusive_radii_kpc.append(aperture_variations[variation]["radius_in_kpc"])
        else:
            exclusive_radii_kpc.append(aperture_variations[variation]["radius_in_kpc"])
    assert inclusive_radii_kpc == sorted(inclusive_radii_kpc)
    assert exclusive_radii_kpc == sorted(exclusive_radii_kpc)

    # Add the apertures defined with fixed physical radii, followed by those
    # whose radius is defined by a SOAP property. Exclusive and inclusive
    # apertures go into separate lists; within each, aperture_variations is
    # sorted by radius so they stay in ascending order, which is what the
    # skip_gt_enclose_radius logic requires.
    for variation in aperture_variations:
        if "radius_in_kpc" not in aperture_variations[variation]:
            continue
        assert "property" not in aperture_variations[variation]
        assert "radius_multiple" not in aperture_variations[variation]
        if aperture_variations[variation]["inclusive"]:
            # If skip_gt_enclose_radius is False (which is the default) then
            # we always want to calculate its properties, regardless of the
            # size of the next smallest aperture.
            radii_kpc = [aperture_variations[variation]["radius_in_kpc"]]
            if aperture_variations[variation].get("skip_gt_enclose_radius", False):
                radii_kpc = inclusive_radii_kpc

            inclusive_apertures.append(
                aperture_properties.InclusiveSphereProperties(
                    cellgrid,
                    parameter_file,
                    aperture_variations[variation]["radius_in_kpc"],
                    None,
                    recently_heated_gas_filter,
                    stellar_age_calculator,
                    cold_dense_gas_filter,
                    category_filter,
                    aperture_variations[variation].get("filter", "basic"),
                    radii_kpc,
                )
            )
        else:
            exclusive_apertures.append(
                aperture_properties.ExclusiveSphereProperties(
                    cellgrid,
                    parameter_file,
                    aperture_variations[variation]["radius_in_kpc"],
                    None,
                    recently_heated_gas_filter,
                    stellar_age_calculator,
                    cold_dense_gas_filter,
                    category_filter,
                    aperture_variations[variation].get("filter", "basic"),
                    exclusive_radii_kpc,
                )
            )

    # Apertures based on SOAP properties
    for variation in aperture_variations:
        if "radius_in_kpc" in aperture_variations[variation]:
            continue
        assert "property" in aperture_variations[variation]
        radius_multiple = aperture_variations[variation].get("radius_multiple", 1)
        # Only allow integer radius mutiples, otherwise swiftsimio will
        # struggle to handle the group names
        assert int(radius_multiple) == radius_multiple
        if aperture_variations[variation]["inclusive"]:
            inclusive_apertures.append(
                aperture_properties.InclusiveSphereProperties(
                    cellgrid,
                    parameter_file,
                    None,
                    (aperture_variations[variation]["property"], radius_multiple),
                    recently_heated_gas_filter,
                    stellar_age_calculator,
                    cold_dense_gas_filter,
                    category_filter,
                    aperture_variations[variation].get("filter", "basic"),
                    [],
                )
            )
        else:
            exclusive_apertures.append(
                aperture_properties.ExclusiveSphereProperties(
                    cellgrid,
                    parameter_file,
                    None,
                    (aperture_variations[variation]["property"], radius_multiple),
                    recently_heated_gas_filter,
                    stellar_age_calculator,
                    cold_dense_gas_filter,
                    category_filter,
                    aperture_variations[variation].get("filter", "basic"),
                    [],
                )
            )

    projected_aperture_variations = parameter_file.get_halo_type_variations(
        "ProjectedApertureProperties"
    )
    # Sort the aperture variations based on their radii, and create a list
    # of all apertures. This is required since we can skip some of the larger
    # apertures if all the particles were already included in the previous aperture
    projected_aperture_variations = dict(
        sorted(
            projected_aperture_variations.items(),
            key=lambda x: x[1].get("radius_in_kpc", 0),
        )
    )
    projected_radii_kpc = []
    for variation in projected_aperture_variations:
        # We don't consider this for apertures defined based on properties
        if "radius_in_kpc" not in projected_aperture_variations[variation]:
            continue
        projected_radii_kpc.append(
            projected_aperture_variations[variation]["radius_in_kpc"]
        )
    assert projected_radii_kpc == sorted(projected_radii_kpc)
    # Add the apertures defined with fixed physical radii
    for variation in projected_aperture_variations:
        if "radius_in_kpc" not in projected_aperture_variations[variation]:
            continue
        assert "property" not in projected_aperture_variations[variation]
        assert "radius_multiple" not in projected_aperture_variations[variation]
        projected_apertures.append(
            projected_aperture_properties.ProjectedApertureProperties(
                cellgrid,
                parameter_file,
                projected_aperture_variations[variation]["radius_in_kpc"],
                None,
                category_filter,
                projected_aperture_variations[variation].get("filter", "basic"),
                projected_radii_kpc,
            )
        )
    # Add the apertures based on SOAP properties
    for variation in projected_aperture_variations:
        if "radius_in_kpc" in projected_aperture_variations[variation]:
            continue
        assert "property" in projected_aperture_variations[variation]
        assert (
            projected_aperture_variations[variation]["property"].split("/")[0]
            == "BoundSubhalo"
        ), "Projected apertures can only be defined by a BoundSubhalo property"
        radius_multiple = projected_aperture_variations[variation].get(
            "radius_multiple", 1
        )
        assert int(radius_multiple) == radius_multiple
        projected_apertures.append(
            projected_aperture_properties.ProjectedApertureProperties(
                cellgrid,
                parameter_file,
                None,
                (projected_aperture_variations[variation]["property"], radius_multiple),
                category_filter,
                projected_aperture_variations[variation].get("filter", "basic"),
                projected_radii_kpc,
            )
        )

    # The category_filter needs access to the filters for each property
    # whenever we are writing the final output file. It needs to get this
    # information from the parmeter file. It would be better to get rid of
    # the category_filter object, and combine it with the parameter_file
    category_filter.set_property_filters(parameter_file.property_filters)

    if comm_world_rank == 0 and args.output_parameters:
        parameter_file.write_parameters(args.output_parameters)

    # Assemble the calculations in the order they will be run for each halo.
    # This order matters, for four separate reasons:
    #
    #  - BoundSubhalo must come first: its results are used by the category
    #    filters and by the enclose radius check of every aperture.
    #  - Within each group of apertures the radii must ascend, because an
    #    aperture may copy its results from the previous (smaller) aperture of
    #    the same type. aperture_variations is sorted by radius, so appending in
    #    order gives this.
    #  - Calculations which see the same particles are kept together, so that
    #    the shared particle arrays can be dropped as soon as the last
    #    calculation needing them has run. Everything using only the bound
    #    particles comes first, then everything using every particle in the
    #    search radius.
    #  - The SO calculations come last, after the inclusive apertures they share
    #    their particle arrays with. SO adds quantities to that shared object
    #    which no aperture uses (the sorted mass profile, the masks flagging
    #    particles bound to another halo), and for the largest halos those are
    #    several GB. Running SO last means they only exist while the
    #    calculations which need them are running.
    #
    halo_prop_list = (
        subhalo_props
        + exclusive_apertures
        + projected_apertures
        + inclusive_apertures
        + so_props
    )

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
        parameter_file.print_unregistered_properties(halo_prop_list, dmo=args.dmo)
        parameter_file.print_skipped_properties(halo_prop_list, dmo=args.dmo)
        parameter_file.print_invalid_properties(halo_prop_list)
        parameter_file.print_variation_warnings()
        if not parameter_file.renclose_enabled():
            print(
                "BoundSubhalo/EncloseRadius is not enabled. This means apertures with r > r_enclose will be calculated explicitly, rather than copying over values from smaller apertures"
            )
        category_filter.print_filters()

        # Properties enabled in the parameter file must be computed, so abort
        # if the input files do not contain the datasets they require
        parameter_file.print_uncomputable_properties()
        if len(parameter_file.uncomputable_properties):
            comm_world.Abort(1)

    # Ensure output dir exists
    if comm_world_rank == 0:
        try:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory: {e}", flush=True)
            comm_world.Abort(1)
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
                os.makedirs(scratch_file_dir, exist_ok=True)
            except OSError as e:
                print(f"Error creating scratch directory: {e}", flush=True)
                comm_world.Abort(1)
    comm_world.barrier()

    # Report initial set-up time
    setup_time_local = time.time() - t0
    t1 = time.time()
    if comm_world_rank == 0:
        print(
            "Reading %d input halos and setting up %d chunk(s) took %.1fs"
            % (so_cat.nr_halos, len(tasks), t1 - t0)
        )

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
        sys.stdout.flush()
        sys.stderr.flush()
        comm_world.Abort(1)

    # Can stop the halo request thread now that all chunk tasks have executed
    so_cat.stop_request_thread()

    # Check metadata for consistency between chunks. Sets ref_metadata on all ranks,
    # including those that processed no halos.
    ref_metadata = result_set.check_metadata(metadata, comm_inter_node, comm_world)

    # Combine chunks into a single output file
    comm_world.barrier()
    t0_combine = time.time()
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

    # Delete scratch files, unless we've been asked to keep them
    if comm_world_rank == 0:
        if args.keep_scratch_files:
            print("Keeping scratch files.")
        else:
            for file_nr in range(nr_chunks):
                os.remove(scratch_file_format % {"file_nr": file_nr})
            print("Deleted scratch files.")
    comm_world.barrier()

    # Stop the clock
    combine_time_local = time.time() - t0_combine
    t1 = time.time()

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

    # Find total time spent running tasks
    if len(timings) > 0:
        task_time_local = sum(timings)
    else:
        task_time_local = 0.0
    setup_time_total = comm_world.allreduce(setup_time_local)
    setup_time_fraction = setup_time_total / (comm_world_size * (t1 - t0))
    task_time_total = comm_world.allreduce(task_time_local)
    task_time_fraction = task_time_total / (comm_world_size * (t1 - t0))
    combine_time_total = comm_world.allreduce(combine_time_local)
    combine_time_fraction = combine_time_total / (comm_world_size * (t1 - t0))

    if comm_world_rank == 0:
        print("Fraction of time spent setting up = %.2f" % setup_time_fraction)
        print(
            "Fraction of time spent calculating halo properties = %.2f"
            % task_time_fraction
        )
        print("Fraction of time spent combining chunks = %.2f" % combine_time_fraction)
        print("Total elapsed time: %.1f seconds" % (t1 - t0))
        print("Done.")


if __name__ == "__main__":

    compute_halo_properties()
