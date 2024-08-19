import numpy as np
import h5py
import multiprocessing as mp
import argparse
import time
import os
import shutil
import yaml

script_folder = os.path.realpath(__file__).removesuffix("/compress_fast_metadata.py")

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


def compress_dataset(arguments):
    input_name, output_name, dset = arguments

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
        # TODO: Remove after removing DMantissa21 from property table
        if filter == 'DMantissa21':
            filter = 'DMantissa9'
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
                # TODO: Remove, this was only the case for a small number of catalogues
                elif attr == "Conversion factor to CGS (including cosmological corrections)":
                    ofile["data"].attrs["Conversion factor to physical CGS (including cosmological corrections)"] = filter
                else:
                    ofile["data"].attrs[attr] = ifile[dset].attrs[attr]

    return dset


if __name__ == "__main__":

    # disable CBLAS threading to avoid problems when spawning
    # parallel numpy imports
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    mp.set_start_method("forkserver")

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    argparser.add_argument("scratch")
    argparser.add_argument("--nproc", "-n", type=int, default=1)
    args = argparser.parse_args()

    # Setting hdf5 version of file
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_libver_bounds(h5py.h5f.LIBVER_V18, h5py.h5f.LIBVER_LATEST)
    # Creation will fail if file already exists
    fid = h5py.h5f.create(args.output.encode('utf-8'), flags=h5py.h5f.ACC_TRUNC, fapl=fapl)

    print(f"Copying over groups to {args.output} and listing datasets...")
    tic = time.time()
    mastertic = tic
    with h5py.File(args.input, "r") as ifile, h5py.File(args.output, "r+") as ofile:
        h5copy = H5copier(ifile, ofile)
        ifile.visititems(h5copy)
        original_size = h5copy.get_total_size()
        original_size_bytes = h5copy.get_total_size_bytes()
        total_time = h5copy.get_total_time()
    toc = time.time()
    print(f"File structure copy took {1000.*(toc-tic):.2f} ms.")

    tmpdir = (
        f"{args.scratch}/{os.path.basename(args.output).removesuffix('.hdf5')}_temp"
    )
    print(
        f"Copying over datasets to temporary files in {tmpdir} using {args.nproc} processes..."
    )
    tic = time.time()
    os.makedirs(tmpdir, exist_ok=True)

    arguments = []
    for dset in h5copy.dsets:
        arguments.append((args.input, f"{tmpdir}/{dset.replace('/','_')}.hdf5", dset))

    pool = mp.Pool(args.nproc)
    count = 0
    ntot = len(arguments)
    for dset in pool.imap_unordered(compress_dataset, arguments):
        count += 1
        print(f"[{count:04d}/{ntot:04d}] {dset}".ljust(110), end="\r")
    toc = time.time()
    print(f"Temporary file writing took {1000.*(toc-tic):.2f} ms.".ljust(110))

    print(f"Copying datasets into {args.output} and cleaning up temporary files...")
    tic = time.time()
    count = 0
    ntot = len(arguments)
    with h5py.File(args.output, "r+") as ofile:
        for _, tmpfile, dset in arguments:
            count += 1
            print(f"[{count:04d}/{ntot:04d}] {dset}".ljust(110), end="\r")
            with h5py.File(tmpfile, "r") as ifile:
                ifile.copy(ifile["data"], ofile, dset)

    shutil.rmtree(tmpdir)
    toc = time.time()
    mastertoc = toc
    print(f"Temporary file copy took {1000.*(toc-tic):.2f} ms.".ljust(110))

    with h5py.File(args.output, "r") as ofile:
        h5print = H5printer(False)
        ofile.visititems(h5print)
        new_size = h5print.get_total_size()
        new_size_bytes = h5print.get_total_size_bytes()

    print(
        f"{original_size} -> {new_size} ({100.*new_size_bytes/original_size_bytes:.2f}%)"
    )
    print(f"Total writing time: {1000.*(mastertoc-mastertic):.2f} ms.")
