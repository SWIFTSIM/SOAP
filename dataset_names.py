#!/bin/env python

# Exclude neutrinos from SO masses for now
ptypes_for_so_masses = [f"PartType{i}" for i in range(6)]


def mass_dataset(ptype):
    if ptype == "PartType5":
        return "DynamicalMasses"
    else:
        return "Masses"
