#! /usr/bin/env python

"""
cylindrical_coordinates.py

Utility function for converting particles to a cylindrical coordinate system

"""

import numpy as np


def build_rotation_matrix(z_target):
    """
    Build a rotation matrix that aligns the new z-axis
    with the given `z_target` vector.

    Parameters:
        z_target: A 3-element array representing
        the target direction to align with the new z-axis.

    Returns:
        R: A (3, 3) rotation matrix. Applying R to a
        set of vectors rotates them into new frame:
          v_new = v_original @ R.T
        where v_original has shape (N, 3)
    """
    z_axis = z_target / np.linalg.norm(z_target)

    # Pick a helper vector that is not nearly parallel to z_axis
    helper = np.array([1, 0, 0])
    if np.allclose(z_axis, helper / np.linalg.norm(helper), rtol=0.1):
        helper = np.array([0, 1, 0])

    # Construct orthonormal basis
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    R = np.vstack([x_axis, y_axis, z_axis])
    return R


def calculate_cylindrical_velocities(positions, velocities, z_target, reference_position = None, reference_velocity = None):
    """
    Convert 3D Cartesian velocities to cylindrical coordinates (v_r, v_phi, v_z),
    after rotating the system such that the z-axis aligns with `z_target`.

    Parameters:
        positions: (N, 3) array of particle positions in the original Cartesian frame.
        velocities: (N, 3) array of particle velocities in the original Cartesian frame.
        z_target: A 3-element vector indicating the new z-axis direction.
        reference_position: (3,) array with a reference position on which to centre the Cartesian coordinate system.
        reference_velocity: (3,) array with a reference velocity on which to centre the Cartesian reference frame.

    Returns:
        cyl_velocities: (N, 3) array of velocities in cylindrical coordinates:
              [v_r, v_phi, v_z] for each particle.
    """

    if reference_position is not None:
        positions -= reference_position
    if reference_velocity is not None:
        velocities -= reference_velocity

    R = build_rotation_matrix(z_target)

    # Rotate positions and velocities into new frame
    positions_rot = positions @ R.T
    velocities_rot = velocities @ R.T

    x = positions_rot[:, 0]
    y = positions_rot[:, 1]
    vx = velocities_rot[:, 0]
    vy = velocities_rot[:, 1]
    vz = velocities_rot[:, 2]

    phi = np.arctan2(y, x)

    v_r = vx * np.cos(phi) + vy * np.sin(phi)
    v_phi = -vx * np.sin(phi) + vy * np.cos(phi)
    v_z = vz

    return np.stack([v_r, v_phi, v_z], axis=1)
