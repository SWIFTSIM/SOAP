#!/usr/bin/env

"""
extract_filters.py

Creates a yaml file which contains the information about the custom lossy
compression filters within a SWIFT snapshot. This yaml file is then used
when we compress SOAP output.

Usage:
  python3 extract_filters.py SNAPSHOT
where SNAPSHOT is the SWIFT snapshot from which we want to extract the filters.

SNAPSHOT can not be a virtual file, as the datasets in the virtual file do not
contain filter information. Use one of the chunk files instead

The chunk size is included in the filter that is output, meaning the output
will differ slightly for different SWIFT snapshots. However, this doesn't matter
since we set the chunk size explicitly when we create compressed SOAP catalogues. 
"""

import sys

import h5py
import yaml

def extract_filters(filename):
    file = h5py.File(filename, 'r')

    filters = {}
    # Loop through particle types
    for i in [0, 1, 4, 5]:
        group = file[f'PartType{i}']
        # Loop through properties
        for prop in group.keys():
            filter_name = group[prop].attrs['Lossy compression filter'].decode('utf-8')
            # Skip if we've already added this filter, or if there is no compression
            if (filter_name in filters) or (filter_name == 'None'):
                continue
            # Extract properties
            dset = h5py.h5d.open(group.id, prop.encode('utf-8'))
            plist = dset.get_create_plist()
            # Convert to list
            lossy_filter = list(plist.get_filter(0))
            lossy_filter[2] = list(lossy_filter[2])
            print(f'Adding the following lossy filter for {filter_name}:')
            print(lossy_filter, end='\n\n')
            filters[filter_name] = {'filters': [lossy_filter]}

            # Save dtype
            filters[filter_name]['type'] = dset.get_type().encode()

    # Add a Fletcher32 checksum filter
    fletcher_filter = [3, 0, [], 'fletcher32'.encode('utf-8')]
    for filter_name in filters:
        filters[filter_name]['filters'].append(fletcher_filter)

    file.close()
    return filters

if __name__ == '__main__':
    """
    Main entry point
    """
    filename = sys.argv[1]

    filters = extract_filters(filename)

    yaml.Dumper.ignore_aliases = lambda self, data: True
    with open('filters.yml', 'w') as outfile:
        yaml.dump(filters, outfile)
