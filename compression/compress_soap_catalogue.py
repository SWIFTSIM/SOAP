import argparse
import time
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import shutil

import numpy as np
import h5py
from mpi4py import MPI
import yaml

script_folder = os.path.realpath(os.path.dirname(__file__))

with open(f"{script_folder}/filters.yml", "r") as ffile:
    filterdict = yaml.safe_load(ffile)

with open(f"{script_folder}/wrong_compression.yml", "r") as cfile:
    # Load empty dictionary if wrong_compression.yml is empty
    compression_fixes = yaml.safe_load(cfile) or {}

chunksize = 1000
compression_opts = {"compression": "gzip", "compression_opts": 4}


class H5visiter:
    def __init__(self):
        self.totsize = 0
        self.tottime = 0.0

    def get_total_size_bytes(self):
        return self.totsize

    def get_total_size(self):
        totsize = self.totsize
        for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
            if abs(totsize) < 1024.0:
                return "%3.1f%s%s" % (totsize, unit, "B")
            totsize /= 1024.0
        return "%.1f%s%s" % (num, "Y", "B")

    def get_total_time(self):
        return 1000.0 * self.tottime


class H5copier(H5visiter):
    def __init__(self, ifile, ofile):
        super().__init__()
        self.ifile = ifile
        self.ofile = ofile
        self.dsets = []

    def __call__(self, name, h5obj):
        type = h5obj.__class__
        if isinstance(h5obj, h5py.Group):
            type = "group"
        elif isinstance(h5obj, h5py.Dataset):
            type = "dataset"
        else:
            raise RuntimeError(f"Unknown HDF5 object type: {name}")

        if type == "group":
            tic = time.time()
            self.ofile.create_group(name)
            for attr in self.ifile[name].attrs:
                self.ofile[name].attrs[attr] = self.ifile[name].attrs[attr]
            toc = time.time()
            self.tottime += toc - tic
        elif type == "dataset":
            size = h5obj.id.get_storage_size()
            self.totsize += size
            self.dsets.append(name)


class H5printer(H5visiter):
    def __init__(self, print=True):
        super().__init__()
        self.print = print

    def __call__(self, name, h5obj):
        if isinstance(h5obj, h5py.Dataset):
            size = h5obj.id.get_storage_size()
            self.totsize += size
            if self.print:
                print(name)


def create_lossy_dataset(file, name, shape, filter):
    fprops = filterdict[filter]
    type = h5py.h5t.decode(fprops["type"])
    new_plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    if len(shape) == 1:
        chunk = (min(shape[0], chunksize),)
    else:
        chunk = (min(shape[0], chunksize), shape[1])
    new_plist.set_chunk(chunk)
    for f in fprops["filters"]:
        new_plist.set_filter(f[0], f[1], tuple(f[2]))
    new_plist.set_deflate(compression_opts["compression_opts"])
    space = h5py.h5s.create_simple(shape, shape)
    h5py.h5d.create(file.id, name.encode("utf-8"), type, space, new_plist, None).close()


def compress_dataset(input_name, output_name, dset):

    # Setting hdf5 version of file
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_V18, h5py.h5f.LIBVER_LATEST)
    fid = h5py.h5f.create(output_name.encode('utf-8'), flags=h5py.h5f.ACC_TRUNC, fapl=fapl)

    with h5py.File(input_name, "r") as ifile, h5py.File(output_name, "r+") as ofile:
        group_name = dset.split("/")[0]
        if group_name == 'Cells':
            filter = "None"
        else:
            filter = ifile[dset].attrs["Lossy compression filter"]
        dset_name = dset.split("/")[-1]
        if dset_name in compression_fixes:
            filter = compression_fixes[dset_name]
        data = ifile[dset][:]
        if filter == "None":
            if len(data.shape) == 1:
                compression_opts["chunks"] = min(chunksize, data.shape[0])
            else:
                compression_opts["chunks"] = (
                    min(chunksize, data.shape[0]),
                    data.shape[1],
                )
            ofile.create_dataset("data", data=data, **compression_opts)
        else:
            create_lossy_dataset(ofile, "data", data.shape, filter)
            ofile["data"][:] = data
        for attr in ifile[dset].attrs:
            if attr == "Is Compressed":
                ofile["data"].attrs[attr] = True
            else:
                # This is needed if we have used the compression_fixes dictionary
                if attr == "Lossy compression filter":
                    ofile["data"].attrs[attr] = filter
                else:
                    ofile["data"].attrs[attr] = ifile[dset].attrs[attr]

    return dset


# Assign datasets to ranks
def assign_datasets(nr_files, nr_ranks, comm_rank):
    files_on_rank = np.zeros(nr_ranks, dtype=int)
    files_on_rank[:] = nr_files // nr_ranks
    remainder = nr_files % nr_ranks
    if remainder == 1:
        files_on_rank[0] += 1
    elif remainder > 1:
        for i in range(remainder):
            files_on_rank[int((nr_ranks - 1) * i/(remainder - 1))] += 1
    assert np.sum(files_on_rank) == nr_files, f'{nr_files=}, {nr_ranks=}'
    first_file = np.cumsum(files_on_rank) - files_on_rank
    return first_file[comm_rank], first_file[comm_rank] + files_on_rank[comm_rank]


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", help="Filename of uncompressed input SOAP catalogue")
    argparser.add_argument("output", help="Filename of output catalogue")
    argparser.add_argument("scratch", help="Directory to store temporary files")
    args = argparser.parse_args()

    mastertic = time.time()

    if comm_rank == 0:
        try:
            print(f'Creating output file at {args.output}')
            # Setting hdf5 version of file
            fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            fapl.set_libver_bounds(h5py.h5f.LIBVER_V18, h5py.h5f.LIBVER_LATEST)
            fid = h5py.h5f.create(args.output.encode('utf-8'), flags=h5py.h5f.ACC_TRUNC, fapl=fapl)

            print(f"Creating groups and datasets in output file")
            tic = time.time()
            with h5py.File(args.input, "r") as ifile, h5py.File(args.output, "r+") as ofile:
                h5copy = H5copier(ifile, ofile)
                ifile.visititems(h5copy)
                original_size = h5copy.get_total_size()
                original_size_bytes = h5copy.get_total_size_bytes()
                total_time = h5copy.get_total_time()
            toc = time.time()
            print(f"File structure copy took {toc-tic:.2f} s.")

            tmp_dir = f"{args.scratch}/{os.path.basename(args.output).removesuffix('.hdf5')}_temp"
            os.makedirs(tmp_dir, exist_ok=True)

            datasets = h5copy.dsets.copy()

        except Exception as e:
            print(f'Error: {e}')
            comm.Abort(1)
    else:
        tmp_dir = None
        datasets = None
    tmp_dir = comm.bcast(tmp_dir)
    datasets = comm.bcast(datasets)

    if comm_rank == 0:
        print(f'Creating compressed datasets in temporary files using {comm_size} ranks')
    tic = time.time()
    i_start, i_end = assign_datasets(len(datasets), comm_size, comm_rank)
    for i_dset, dset in enumerate(sorted(datasets)[i_start:i_end]):
        tmp_output = f"{tmp_dir}/{dset.replace('/','_')}.hdf5"
        compress_dataset(args.input, tmp_output, dset)
        print(f"{comm_rank}: [{i_dset+1:04d}/{i_end-i_start:04d}] {dset}")

    comm.barrier()
    toc = time.time()
    if comm_rank == 0:
        print(f"Temporary file writing took {toc-tic:.2f} s.")
        print(f"Copying datasets into {args.output}")

    # Only the first rank writes the final output file
    tic = time.time()
    if comm_rank == 0:
        with h5py.File(args.output, 'r+') as ofile:
            for i_dset, dset in enumerate(datasets):
                tmp_output = f"{tmp_dir}/{dset.replace('/','_')}.hdf5"
                with h5py.File(tmp_output, 'r') as ifile:
                    ifile.copy(ifile['data'], ofile, name=dset)
                print(f"{comm_rank}: [{i_dset+1:04d}/{len(datasets):04d}] {dset}")

    comm.barrier()
    toc = time.time()
    mastertoc = toc
    if comm_rank == 0:
        print(f"Writing output file took {toc-tic:.2f} s.")
        print('Removing temporary files')
        shutil.rmtree(tmp_dir)

        with h5py.File(args.output, "r") as ofile:
            h5print = H5printer(False)
            ofile.visititems(h5print)
            new_size = h5print.get_total_size()
            new_size_bytes = h5print.get_total_size_bytes()

        frac = new_size_bytes/original_size_bytes
        print(f"{original_size} -> {new_size} ({100.*frac:.2f}%)")
        print(f"Total writing time: {mastertoc-mastertic:.2f} s.")
        print("Done")
