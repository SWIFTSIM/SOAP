#!/bin/env python

# Exclude neutrinos from SO masses for now
ptypes_for_so_masses = [
    "PartType0",
    "PartType1",
    "PartType4",
    "PartType5",
]

def mass_dataset(ptype):
    if ptype == "PartType5":
        return "DynamicalMasses"
    else:
        return "Masses"
