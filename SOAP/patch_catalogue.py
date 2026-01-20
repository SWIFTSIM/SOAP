import numpy as np
import unyt
import yaml
import argparse
import os
import h5py as h5

# Set simName
parser = argparse.ArgumentParser()

parser.add_argument(
    "simName",
    type=str,
    help="Simulation name.",
)

parser.add_argument(
    "snap",
    type=int,
    help="<Required> Snapshot number.",
)

args = parser.parse_args()

# Define filepaths from parameter file
dir_path =  '/users/ariadurr/ariadurr/COLIBRE/Processing/colibre-skirt'
param_file = 'vimf_SKIRT_parameters.yml'
with open(f'{dir_path}/{param_file}','r') as stream:
    params = yaml.safe_load(stream)

simPath = params['InputFilepaths']['simPath'].format(simName=args.simName)

# Load SOAP catalogue and required attributes for a SOAP dset
catalogue_file = params['InputFilepaths']['catalogueFile'].format(simPath=simPath,snap_nr=args.snap)
patch_file = catalogue_file.replace('_vimf','_vimf_patch')


# Delete existing luminosity files

ES_all_apertures = ['100kpc','10kpc','1kpc','2xHalfMassRadiusStars','30kpc','3kpc','50kpc']
SO_all_apertures = ['200_crit','200_mean','500_crit','BN98']
PA_all_apertures = ['100kpc','100pc','10kpc','1kpc','2xHalfMassRadiusStars','300pc','30kpc','3kpc','50kpc']

groups_to_delete = ['BoundSubhalo']
for ap in ES_all_apertures:
    groups_to_delete += [f'ExclusiveSphere/{ap}']
for ap in SO_all_apertures:
    groups_to_delete += [f'SO/{ap}']
for ap in PA_all_apertures:
    for proj in ['projx','projy','projz']:
        groups_to_delete += [f'ProjectedAperture/{ap}/{proj}']

with h5.File(catalogue_file,'a') as dst_file:
    for grp in groups_to_delete:

        def delete_dset(dset_name):
            try:
                del dst_file[grp + '/' + dset_name]
            except:
                print(f'Group {grp}/{dset_name} not found.')

        delete_dset('CorrectedStellarLuminosity')
        if 'ExclusiveSphere' in grp:
            delete_dset('HalfCorrectedLightRadiusStars')
            delete_dset('UVLuminosityWeightedMeanStellarAge')

        # if 'ExclusiveSphere' in grp or 'ProjectedAperture' in grp:
        #     delete_dset('ChabrierInferredStellarMass')
dst_file.close()


# Add back in from the patch files

PA_all_apertures = ['100kpc','10kpc','30kpc','3kpc','50kpc'] # recomputed less apertures here

groups_to_add = ['BoundSubhalo']
for ap in ES_all_apertures:
    groups_to_add += [f'ExclusiveSphere/{ap}']
for ap in SO_all_apertures:
    groups_to_add += [f'SO/{ap}']
for ap in PA_all_apertures:
    for proj in ['projx','projy','projz']:
        groups_to_add += [f'ProjectedAperture/{ap}/{proj}']

with h5.File(catalogue_file,'a') as dst_file, h5.File(patch_file,'r') as src_file:
    for grp in groups_to_add:
        dst_group = dst_file.require_group(grp)

        # First for luminosities
        dset_source = src_file[grp +'/CorrectedStellarLuminosity']
        dst_group.copy(dset_source,'CorrectedStellarLuminosity')

        # Second for HLR in ES only
        if 'ExclusiveSphere' in grp:
            dset_source = src_file[grp +'/UVLuminosityWeightedMeanStellarAge']
            dst_group.copy(dset_source,'UVLuminosityWeightedMeanStellarAge')

            dset_source = src_file[grp +'/HalfCorrectedLightRadiusStars']
            dst_group.copy(dset_source,'HalfCorrectedLightRadiusStars')

        # Third for Chab masses in ES and Proj only
        # if 'ExclusiveSphere' in grp or 'ProjectedAperture' in grp:
        #     dset_source = src_file[grp +'/ChabrierInferredStellarMass']
        #     dst_group.copy(dset_source,'ChabrierInferredStellarMass')

src_file.close()
dst_file.close()

