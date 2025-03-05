#!/bin/bash

wget https://ftp.strw.leidenuniv.nl/mcgibbon/SOAP/swift_output/fof_output_0018.hdf5
for i in {0..18}; do
    snap_nr=$(printf "%04d" $i)
    wget "https://ftp.strw.leidenuniv.nl/mcgibbon/SOAP/swift_output/snap_${snap_nr}.hdf5"
done

