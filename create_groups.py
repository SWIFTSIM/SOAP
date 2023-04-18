#!/bin/env python

import os.path


def find_groups_to_create(paths):
    """
    Given a list of paths to HDF5 objects, return a list of the names
    of the groups which must be created in the order in which to create
    them.
    """

    groups_to_create = set()
    for path in paths:
        dirname = path
        while True:
            dirname = os.path.dirname(dirname)
            if len(dirname) > 0:
                groups_to_create.add(dirname)
            else:
                break
    groups_to_create = list(groups_to_create)
    groups_to_create.sort(key=lambda x: len(x.split("/")))
    return groups_to_create
