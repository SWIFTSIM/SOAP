#!/bin/env python
 
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np
from glob import glob

import virgo.mpi.util
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

def get_SOAP_property_evolution(SOAP_paths, TrackIDs_to_follow, properties):
    """
    Loads the SOAP catalogues in parallel and retrieves for each catalogue 
    the specified properties for the provided TrackIDs.

    Parameters
    ----------
    SOAP_paths: list of str
        List of sorted strings where each one provides the path to a single SOAP
        catalogue.
    TrackIDs_to_follow: np.ndarray
        TrackId of the subhaloes we are interested in.
    properties: list of str
        The properties we want to retrieve for each fo the provided TrackIDs.

    Returns
    -------
    redshift_evolution: np.ndarray
        The redshifts at which each SOAP catalogue was saved.
    property_evolution: dict of np.ndarray
        Dictionary where each entry corresponds to the evolution of the requested
        properties for the TrackIDs that are in the local MPI rank.
    """

    # We balance the number of Tracks across the tasks we have
    TrackIDs_to_follow = psort.parallel_unique(TrackIDs_to_follow, repartition_output=True)

    # Final dictionary with the data we will save.
    property_evolution = dict([(property, []) for property in properties])

    # Redshifts at which we loaded the SOAP catalogues.
    redshift_evolution = -np.ones(len(SOAP_paths))

    for i, path in enumerate(SOAP_paths):

        if comm_rank ==0:
            print(f"Reading SOAP catalogue: {path}", end=' --- ')

        with h5py.File(path, 'r') as SOAP_catalogue:
            redshift_evolution[i] = SOAP_catalogue['Header'].attrs['Redshift'][0]

        mf = phdf5.MultiFile([path], comm=comm,)
        data = {}
        for property in properties:
            data[property] = mf.read(property)
        del mf

        # Each rank will find the entry with the TrackID they are supposed to use.
        order = psort.parallel_match(TrackIDs_to_follow, data["InputHalos/HBTplus/TrackId"])

        # Get the properties we want
        for property in properties:
            property_evolution[property].append([psort.fetch_elements(data[property], order, comm=comm)])

        if comm_rank == 0:
            print("DONE") 

    return redshift_evolution, property_evolution

def save_evolution(redshift_evolution, property_evolution, output_file):
    """
    Saves the property evolution of the subhaloes in a specified HDF5 file path.

    Parameters
    ----------
    redshift_evolution: np.ndarray
        The redshifts at which each SOAP catalogue was saved.
    property_evolution: dict of np.ndarray
        Dictionary where each entry corresponds to the evolution of the requested
        properties for the TrackIDs that are in the local MPI rank.
    output_file: str
        Location where to save the HDF5 containing the evolution of TrackIDs.
    """

    if comm_rank ==0:
        print(f"Saving output", end=' --- ')

    # We create a single array for each output property
    with h5py.File(output_file, 'w', driver='mpio', comm=comm) as outfile:

        for property in property_evolution.keys():
            data_to_save = np.concatenate(property_evolution[property]).T

            names = property.split('/')
            group_name = '/'.join(names[:-1])
            dset_name  = names[-1]

            if group_name in outfile:
                group = outfile[group_name]
            else:
                group = outfile.create_group(group_name)

            phdf5.collective_write(group, dset_name, data_to_save, comm)

        # Save the redshift for future use
        dset = outfile.create_dataset("Redshift", data=redshift_evolution)

    comm.Barrier()
    if comm_rank == 0:
        print("DONE")

def get_track_evolution(SOAP_basedir, output_file, track_path, property_path):
    """
    Loads in parallel the SOAP properties for each catalogue in the provided 
    base directory, retrieves the specified properties for each TrackID of interest.
    Saves the property evolution in a HDF5 file.

    Parameters
    ----------
    SOAP_basedir: str
        Path to the directory containing all SOAP catalogues.
    output_file: str
        Location the HDF5 containing the evolution of TrackIDs is saved.
    track_path: np.ndarray
        Path to a text file containing in each row a different TrackId, which we 
        use to flag which subhaloes we are interested in.
    property_path: np.ndarray
        Path to a text file containing in each row a different SOAP property, which we 
        use to flag which properties we are interested in following.
    """

    # Get all paths by default
    SOAP_paths = sorted(glob(f"{SOAP_basedir}/halo_properties_*.hdf5"))

    # Load which subhaloes and properties we are interested in.
    tracks = np.loadtxt(track_path, int)
    properties = np.loadtxt(property_path, str)

    if comm_rank ==0:
        print (f"Getting the evolution of {len(properties)} properties for {len(tracks)} TrackIDs. There are {len(SOAP_paths)} SOAP files available in the specified directory.")

    # Get the evolution
    redshift_evolution, property_evolution = get_SOAP_property_evolution(SOAP_paths, tracks, properties)

    comm.Barrier()

    # Save in a separate file for future use
    save_evolution(redshift_evolution, property_evolution, output_file)

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(comm, description="Obtain the evolution of the specified SOAP properties for the provided TrackIds.")
    parser.add_argument("SOAP_basedir", type=str, help="Root directory of the the SOAP catalogues.")
    parser.add_argument("output_file",  type=str, help="File in which to write the output.")
    parser.add_argument('-t', '--tracks', type=str, dest="track_path", help='Path to a text file containing in each row a TrackId whose evolution are interested in tracking.')
    parser.add_argument('-p', '--properties', type=str, dest="property_path", help='Path to a text file containing in each row a SOAP property that we are interested in tracking.')
    args = parser.parse_args()

    get_track_evolution(**vars(args))