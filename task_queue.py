#!/bin/env python

import time
import threading
from mpi4py import MPI
import collections
import numpy as np
import shared_array

from mpi_tags import REQUEST_TASK_TAG, ASSIGN_TASK_TAG
from sleepy_recv import sleepy_recv


def distribute_tasks(tasks, comm, task_type):
    """
    Listen for and respond to requests for tasks to do
    """
    comm_size = comm.Get_size()
    next_task = 0
    nr_tasks = len(tasks)
    nr_done = 0
    while nr_done < comm_size:
        request_src = sleepy_recv(comm, REQUEST_TASK_TAG)
        if next_task < nr_tasks:
            if task_type is None:
                # Use generic mpi4py send
                comm.send(tasks[next_task], request_src, tag=ASSIGN_TASK_TAG)
            else:
                # Task provides its own send method
                tasks[next_task].send(comm, request_src, tag=ASSIGN_TASK_TAG)
            next_task += 1
        else:
            comm.send(None, request_src, tag=ASSIGN_TASK_TAG)
            nr_done += 1


def distribute_tasks_with_queue_per_rank(tasks, comm):
    """
    Listen for and respond to requests for tasks to do.
    In this case tasks is a sequence of comm_size task lists.
    Each rank will preferentially do tasks from it's own
    task list, but will do other tasks if it runs out.
    """
    comm_size = comm.Get_size()
    next_task = 0
    nr_tasks = sum([len(t) for t in tasks])
    tasks = [collections.deque(t) for t in tasks]
    nr_done = 0
    while nr_done < comm_size:
        request_src = sleepy_recv(comm, REQUEST_TASK_TAG)
        if next_task < nr_tasks:
            # If we have no tasks left for this rank, steal some!
            if len(tasks[request_src]) == 0:

                # Take one task from the longest queue
                i = np.argmax([len(t) for t in tasks])
                tasks[request_src].append(tasks[i].popleft())

            # Get the next task for this rank
            task = tasks[request_src].popleft()
            # Send back the task
            comm.send(task, request_src, tag=ASSIGN_TASK_TAG)
            next_task += 1
        else:
            comm.send(None, request_src, tag=ASSIGN_TASK_TAG)
            nr_done += 1


def execute_tasks(
    tasks,
    args,
    comm_all,
    comm_master,
    comm_workers,
    queue_per_rank=False,
    return_timing=False,
    task_type=None,
):
    """
    Execute the tasks in tasks, which should be a sequence of
    callables which each return a result. Task objects are
    communicated over MPI so they must be pickleable. Each task is
    called as task(*args). The tasks argument is only significant
    on rank 0 of comm_master. The args argument is used on all
    ranks and should be used to pass comm_workers into the code
    executing each task.

    MPI ranks are split into groups of workers. The first rank in
    each worker group belongs to comm_master too. The comm_master
    communicator is used to issue tasks to each group of workers.
    Each task is run in parallel by all of the MPI ranks in one
    group of workers.

    The intended use of this to assign tasks to compute nodes and
    have all of the ranks on a node cooperate on a single task.

    Use comm_master=MPI.COMM_WORLD and comm_workers=MPI.COMM_SELF
    to run each task on a single MPI rank instead.

    On each MPI rank a list of results of the tasks which that rank
    participated in is returned. Ordering of the results is likely
    to be unpredictable!
    """

    timing = {}

    # Clone communicators to prevent message confusion:
    # In particular, tasks are likely to be using comm_workers internally.
    if comm_master != MPI.COMM_NULL:
        comm_master_local = comm_master.Dup()
    else:
        comm_master_local = MPI.COMM_NULL
    comm_workers_local = comm_workers.Dup()
    comm_all_local = comm_all.Dup()

    # Start the clock
    comm_all.barrier()
    overall_t0 = time.time()
    tasks_elapsed_time = 0.0
    tasks_wait_time = 0.0

    # Get ranks in communicators
    master_rank = (
        -1 if comm_master_local == MPI.COMM_NULL else comm_master_local.Get_rank()
    )
    worker_rank = comm_workers_local.Get_rank()

    # First rank in comm_master starts a thread to hand out tasks
    if master_rank == 0:
        if queue_per_rank:
            if task_type is not None:
                raise NotImplementedError("Can't specify task type with queue per rank")
            task_queue_thread = threading.Thread(
                target=distribute_tasks_with_queue_per_rank,
                args=(tasks, comm_master_local),
            )
        else:
            task_queue_thread = threading.Thread(
                target=distribute_tasks, args=(tasks, comm_master_local, task_type)
            )
        task_queue_thread.start()

    # Request and run tasks until there are none left
    result = []
    while True:

        # Start timer for time spent waiting to be given a task
        wait_time_t0 = time.time()

        # The first rank in each group of workers requests a task and broadcasts it to the other workers.
        # If the task has a get_worker_task() method we broadcast whatever it returns to the other MPI ranks
        # and execute it. This allows having a different task object on the first rank, e.g. with extra data
        # which doesn't need to be duplicated to all ranks.
        if worker_rank == 0:
            comm_master_local.send(master_rank, 0, tag=REQUEST_TASK_TAG)
            if task_type is None:
                # generic recv
                task = comm_master_local.recv(tag=ASSIGN_TASK_TAG)
            else:
                # task provides its own recv
                task = task_type.recv(comm_master_local, 0, ASSIGN_TASK_TAG)
        else:
            task = None

        # Broadcast the task to all workers if necessary:
        # First they all need to know whether we have a task to do.
        task_not_none = int(task is not None)
        task_not_none = comm_workers_local.allreduce(task_not_none, MPI.MAX)
        if task_not_none:
            if worker_rank == 0:
                task_class = type(task)
                instance = task
            else:
                task_class = None
                instance = None
            task_class = comm_workers_local.bcast(task_class)
            if hasattr(task_class, "bcast"):
                # Task implements it's own broadcast
                task = task_class.bcast(comm_workers_local, instance)
            elif comm_workers_local.Get_size() > 1:
                # Use generic mpi4py broadcast
                task = comm_workers_local.bcast(task)

        # Accumulate time spent waiting for a task
        wait_time_t1 = time.time()
        tasks_wait_time += wait_time_t1 - wait_time_t0

        # All workers in the group execute the task as a collective operation
        if task_not_none:
            task_t0 = time.time()
            result.append(task(*args))
            task_t1 = time.time()
            tasks_elapsed_time += task_t1 - task_t0
        else:
            break

    # Measure how long we spend waiting with nothing to do at the end
    t0_out_of_work = time.time()

    # Wait for task distributing thread to finish
    if master_rank == 0:
        task_queue_thread.join()

    # Stop the clock
    comm_all_local.barrier()
    overall_t1 = time.time()
    t1_out_of_work = time.time()

    # Compute dead time etc
    time_total = comm_all_local.allreduce(overall_t1 - overall_t0)
    time_tasks = comm_all_local.allreduce(tasks_elapsed_time)
    time_out_of_work = comm_all_local.allreduce(t1_out_of_work - t0_out_of_work)
    time_wait_for_task = comm_all_local.allreduce(tasks_wait_time)

    # Report total task time
    timing["elapsed"] = overall_t1 - overall_t0
    timing["dead_time_fraction"] = (time_total - time_tasks) / time_total
    timing["out_of_work_fraction"] = time_out_of_work / time_total
    timing["wait_for_task_fraction"] = time_wait_for_task / time_total

    # Free local communicators
    if comm_master_local != MPI.COMM_NULL:
        comm_master_local.Free()
    comm_workers_local.Free()
    comm_all_local.Free()

    if return_timing:
        return result, timing
    else:
        return result
