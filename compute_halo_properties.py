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

from xray_calculator import XrayCalculator


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
        extra_input = sub_snapnum(args.extra_input, args.snapshot_nr)
        if args.reference_snapshot is not None:
            swift_filename_ref = sub_snapnum(
                args.swift_filename, args.reference_snapshot
            )
            extra_input_ref = sub_snapnum(args.extra_input, args.reference_snapshot)
        else:
            swift_filename_ref = None
            extra_input_ref = None
        cellgrid = swift_cells.SWIFTCellGrid(
            swift_filename, extra_input, swift_filename_ref, extra_input_ref
        )
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

    # Process parameter file
    if comm_world_rank == 0:
        parameter_file = ParameterFile(args.config_filename)
    else:
        parameter_file = None
    parameter_file = comm_world.bcast(parameter_file)
    cellgrid.snapshot_datasets.setup_aliases(parameter_file.get_aliases())
    cellgrid.snapshot_datasets.setup_defined_constants(
        parameter_file.get_defined_constants()
    )

    recently_heated_gas_filter = RecentlyHeatedGasFilter(
        cellgrid,
        delta_time=15.0 * unyt.Myr,
        delta_logT_min=-1.0,
        delta_logT_max=0.3,
        AGN_delta_T=8.80144197177e7 * unyt.K,
    )

    stellar_age_calculator = StellarAgeCalculator(cellgrid)
    cold_dense_gas_filter = ColdDenseGasFilter(10.0 ** 4.5 * unyt.K, 0.1 / unyt.cm ** 3)
    default_filters = {
        'general': {
                'limit': 100,
                'properties': [
                    'BoundSubhaloProperties/NumberOfDarkMatterParticles',
                    'BoundSubhaloProperties/NumberOfGasParticles',
                    'BoundSubhaloProperties/NumberOfStarParticles',
                    'BoundSubhaloProperties/NumberOfBlackHoleParticles',
                 ],
                'combine_properties': 'sum'
             },
        'dm': {
                'limit': 100,
                'properties': [
                    'BoundSubhaloProperties/NumberOfDarkMatterParticles',
                 ],
             },
        'gas': {
                'limit': 100,
                'properties': [
                    'BoundSubhaloProperties/NumberOfGasParticles',
                 ],
             },
        'star': {
                'limit': 100,
                'properties': [
                    'BoundSubhaloProperties/NumberOfStarParticles',
                 ],
             },
        'baryon': {
                'limit': 100,
                'properties': [
                    'BoundSubhaloProperties/NumberOfGasParticles',
                    'BoundSubhaloProperties/NumberOfStarParticles',
                 ],
                'combine_properties': 'sum'
             },
    }
    category_filter = CategoryFilter(
        parameter_file.get_filters(
            default_filters
        ),
        dmo=args.dmo,
    )

    # Get the full list of property calculations we can do
    # Note that the order matters: we need to do the BoundSubhaloProperties first,
    # since quantities are filtered based on the particle numbers in there
    # Similarly, things like SO 5xR500_crit can only be done after
    # SO 500_crit for obvious reasons
    halo_prop_list = []
    # Make sure BoundSubhaloProperties is always first
    subhalo_variations = parameter_file.get_halo_type_variations(
        "SubhaloProperties",
        {"Bound": {"bound_only": True}},
    )
    for variation in subhalo_variations:
        if subhalo_variations[variation]["bound_only"]:
            halo_prop_list.append(
                subhalo_properties.SubhaloProperties(
                    cellgrid,
                    parameter_file,
                    recently_heated_gas_filter,
                    stellar_age_calculator,
                    category_filter,
                    bound_only=subhalo_variations[variation]["bound_only"],
                )
            )
    assert len(halo_prop_list) > 0, 'BoundSubhaloProperties must be calculated'
    # Adding FOFSubhaloProperties if present
    for variation in subhalo_variations:
        if not subhalo_variations[variation]["bound_only"]:
            halo_prop_list.append(
                subhalo_properties.SubhaloProperties(
                    cellgrid,
                    parameter_file,
                    recently_heated_gas_filter,
                    stellar_age_calculator,
                    category_filter,
                    bound_only=subhalo_variations[variation]["bound_only"],
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
            )
        )

    if comm_world_rank == 0 and args.output_parameters:
        parameter_file.write_parameters(args.output_parameters)

    # Determine which calculations we're doing this time
    if args.calculations is not None:

        # Check we recognise all the names specified on the command line
        all_names = [hp.name for hp in halo_prop_list]
        for calc in args.calculations:
            if calc not in all_names:
                raise Exception("Don't recognise calculation name: %s" % calc)

        # Filter out calculations which were not selected
        halo_prop_list = [hp for hp in halo_prop_list if hp.name in args.calculations]

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
        parameter_file.print_unregistered_properties()
        if parameter_file.recalculate_xrays():
            print(f"Recalculating xray properties using table: {parameter_file.get_xray_table_path()}")
        category_filter.print_filters()

    # Ensure output dir exists
    if comm_world_rank == 0:
        lustre.ensure_output_dir(args.output_file)
    comm_world.barrier()

    if comm_world_rank == 0:
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
        xray_calculator = XrayCalculator(
            cellgrid.z,
            parameter_file.get_xray_table_path(),
            xray_bands,
            observing_types,
            parameter_file.recalculate_xrays(),
        )
    else:
        xray_calculator = None
    xray_calculator = comm_world.bcast(xray_calculator)

    # Read in the halo catalogue:
    # All ranks read the file(s) in then gather to rank 0. Also computes search radius for each halo.
    halo_basename = sub_snapnum(args.halo_basename, args.snapshot_nr)
    so_cat = halo_centres.SOCatalogue(
        comm_world,
        halo_basename,
        args.halo_format,
        cellgrid.a_unit,
        cellgrid.snap_unit_registry,
        cellgrid.boxsize,
        args.max_halos,
        args.centrals_only,
        args.halo_indices,
        halo_prop_list,
        args.chunks,
    )

    # Generate the chunk task list
    if comm_world_rank == 0:
        task_list = chunk_tasks.ChunkTaskList(
            cellgrid, so_cat, halo_prop_list=halo_prop_list
        )
        tasks = task_list.tasks
        nr_chunks = len(tasks)
    else:
        tasks = None
        nr_chunks = None
    nr_chunks = comm_world.bcast(nr_chunks)

    # Report initial set-up time
    comm_world.barrier()
    t1 = time.time()
    if comm_world_rank == 0:
        print(
            "Reading %d input halos and setting up %d chunk(s) took %.1fs"
            % (so_cat.nr_halos, len(tasks), t1 - t0)
        )

    # We no longer need the catalogue, since halo centres etc are stored in the chunk tasks
    del so_cat

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
        comm_intra_node,
        inter_node_rank,
        timings,
        args.max_ranks_reading,
        scratch_file_format,
        xray_calculator,
    )
    metadata = task_queue.execute_tasks(
        tasks,
        args=task_args,
        comm_all=comm_world,
        comm_master=comm_inter_node,
        comm_workers=comm_intra_node,
        task_type=chunk_tasks.ChunkTask,
    )

    # Check metadata for consistency between chunks. Sets ref_metadata on all ranks,
    # including those that processed no halos.
    ref_metadata = result_set.check_metadata(metadata, comm_inter_node, comm_world)

    # Combine chunks into a single output file
    with MPITimer("Sorting %d halo properties" % len(ref_metadata), comm_world):
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
