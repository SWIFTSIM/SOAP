#!/bin/env python

"""
snapshot_datasets.py

Auxiliary object used to facilitate the access to particle datasets
when some dataset names can be aliased and some datasets have named
columns.

This object should be used as the sole way to access snapshot datasets
in halo property calculations to guarantee that dataset names are
consistent with the aliases defined in the parameter file, and to
guarantee that column indices match the correct column name.
"""

import unyt
import h5py

from typing import Dict
from numpy.typing import NDArray

from SOAP import property_table


class SnapshotDatasets:
    """
    Wrapper around the snapshot metadata that is relevant for dataset
    access: dataset names and aliases and named columns.
    """

    # dictionary containing the datasets per particle type that
    # are present in the snapshot
    datasets_in_file: Dict
    # mapping from generic dataset name to (particle type group,
    # dataset name) pairs.
    dataset_map: Dict
    # mapping from dataset + column names to column index
    named_columns: Dict
    # grain compositions in the dust model (currently not used)
    dust_grain_composition: NDArray[float]
    # constants defined in the parameter file
    defined_constants: Dict

    def __init__(self, filenames: list):
        """
        Constructor.

        Read dataset names and named column metadata from
        the filenames given.

        Parameters:
         - filenames: list
           Filenames of the snapshot and extra-input files.
           The snapshot filename must be the first element.
        """

        # Loop through all the files provided and record the datasets present
        self.datasets_in_file = {}
        self.named_columns = {}
        for filename in filenames:
            with h5py.File(filename.format(file_nr=0), "r") as file_handle:
                for group in file_handle:
                    if not group.startswith("PartType"):
                        continue
                    if group not in self.datasets_in_file:
                        self.datasets_in_file[group] = set()
                    for dset in file_handle[group]:
                        self.datasets_in_file[group].add(dset)

                # Try to read named columns
                if ("SubgridScheme" not in file_handle) or (
                    "NamedColumns" not in file_handle["SubgridScheme"]
                ):
                    continue
                # As the snapshot filename is done first, if one of the extra-input
                # files has a named column entry in common with the snapshot then
                # we use then one from the extra-input file.
                for name in file_handle["SubgridScheme"]["NamedColumns"]:
                    column_names = file_handle["SubgridScheme"]["NamedColumns"][name][:]
                    self.named_columns[name] = {}
                    # turn the list into a dictionary that maps a column name to
                    # a colum index
                    for iname, colname in enumerate(column_names):
                        self.named_columns[name][colname.decode("utf-8")] = iname

                try:
                    self.dust_grain_composition = file_handle["SubgridScheme"][
                        "GrainToElementMapping"
                    ][:]
                except KeyError:
                    try:
                        self.dust_grain_composition = file_handle["SubgridScheme"][
                            "DustMassFractionsToElementMassFractionsMapping"
                        ][:]
                    except KeyError:
                        pass

    def setup_aliases(self, aliases: Dict):
        """
        Set up alternative names (aliases) for some datasets.
        This method also creates the dataset_map dictionary that
        maps a full (generic) dataset path to a (particle type group,
        dataset name) pair.

        An alias is a pair
         (generic_name, snapshot_name)
        where 'generic_name' is used internally in halo property calculations,
        while 'snapshot_name' is the name of the dataset as it appears in the
        snapshot. This can be useful if a dataset has a different name than
        expected.
        Aliases can also affect named columns, since these are also defined
        for a particular dataset name as it appears in the snapshot.

        Note that this function simply adds new entries to the dataset_map
        for the generic name of an alias, which will then contain the data
        for the snapshot name. If the generic name was also present in the
        snapshot, the original data for that name will become inaccessible.
        Aliases can hence also be used to hide existing datasets.

        Parameters:
         - aliases: Dict
           Dictionary with (generic name, snapshot name) pairs.
        """
        self.dataset_map = {}
        for ptype in self.datasets_in_file:
            for dset in self.datasets_in_file[ptype]:
                snap_name = f"{ptype}/{dset}"
                self.dataset_map[snap_name] = (ptype, dset)
        for alias in aliases:
            SOAP_ptype, SOAP_dset = alias.split("/")
            snap_ptype, snap_dset = aliases[alias].split("/")
            self.dataset_map[alias] = (snap_ptype, snap_dset)
            if (snap_dset in self.named_columns) and (
                SOAP_dset not in self.named_columns
            ):
                self.named_columns[SOAP_dset] = dict(self.named_columns[snap_dset])

    def setup_defined_constants(self, defined_constants: Dict):
        """
        Set up defined constants based on the provided dictionary.

        This function also attaches units to the constants if appropriate.

        Parameters:
         - defined_constants:
           Dictionary with defined constants (name: value). Values can have
           unit strings attached to them. Can correspond to a raw YAML
           dictionary, for example the one read from the parameter file.
        """
        self.defined_constants = {}
        for name, value in defined_constants.items():
            self.defined_constants[name] = unyt.unyt_quantity.from_string(f"{value}")

    def get_defined_constant(self, name: str) -> unyt.unyt_quantity:
        """
        Get the value of the defined constant with the given name.

        Parameters:
         - name: str
           Name of a constant.

        Returns the corresponding value, with units attached.
        """
        try:
            return self.defined_constants[name]
        except KeyError:
            raise KeyError(f'Defined constant "{name}" not found in parameter file!')

    def get_dataset(self, name: str, data_dict: Dict) -> unyt.unyt_array:
        """
        Get the data for the dataset with the given generic name.

        Parameters:
         - name: str
           Generic name of a dataset, as used by halo property calculations.
         - data_dict: Dict
           Dictionary of particle properties, as read from the snapshot.

        Returns the corresponding data, taking into account potential
        aliases.
        """
        try:
            ptype, dset = self.dataset_map[name]
        except KeyError as e:
            print(f'Dataset "{name}" not found!')
            print("The following properties require this dataset:")
            full_property_list = property_table.PropertyTable.full_property_list
            for k, v in full_property_list.items():
                if name in v.particle_properties:
                    print(f'  {k}')
            raise e
        return data_dict[ptype][dset]

    def get_dataset_column(
        self, name: str, column_name: str, data_dict: Dict
    ) -> unyt.unyt_array:
        """
        Get the data for the given named column in the dataset with the given
        generic name.

        Parameters:
         - name: str
           Generic name of a dataset, as used by halo property calculations.
         - column_name: str
           Name of a named column, as defined in the snapshot metadata and used
           by halo property calculations.
         - data_dict: Dict
           Dictionary of particle properties, as read from the snapshot.

        Returns the corresponding data, taking into account potential
        aliases and the named column metadata.
        """
        ptype, dset = self.dataset_map[name]
        column_index = self.named_columns[dset][column_name]
        return data_dict[ptype][dset][:, column_index]

    def get_column_index(self, dset: str, column_name: str) -> int:
        """
        Get the index of the given named column of the dataset
        with the given name.

        Parameters:
         - dset: str
           Generic name of a dataset, as used by halo property calculations.
         - column_name: str
           Name of a named column, as defined in the snapshot metadata and
           used by halo property calculations.

        Returns the corresponding index number that can be used to
        access that specific column in a data array that was obtained earlier
        using get_dataset().
        """
        return self.named_columns[dset][column_name]

    def get_dust_grain_composition(self, grain_name: str) -> NDArray[float]:
        """
        Get the composition of the grain with the given name.

        Currently not used.

        Parameters:
         - grain_name: str
           Name of a dust grain.

        Returns the corresponding elemental composition of the grain.
        """
        return self.dust_grain_composition[
            self.named_columns["DustMassFractions"][grain_name]
        ]
