import numpy as np

def vecdot(a, b, axis=-1):
    return np.sum(np.conjugate(a) * b, axis=axis)

def get_scalar_projection(vector_to_project, projection_direction_vector):
    """
    Projects one or more vectors along the provided direction(s) and returns the
    (signed) scalar magnitude. NOTE: requires Numpy 2.X

    Parameters
    ----------
    vector_to_project: np.ndarray
        The vector(s) to project
    projection_direction_vector: np.ndarray
        The direction(s) along which to project the vector. If more than one
        direction is provided, it needs to be provided individually for each
        vector.

    Returns
    -------
    np.ndarray
    """

    if projection_direction_vector.shape != vector_to_project.shape:
        raise ValueError("The vectors to project and the direction to project along should have the shape.")

    if projection_direction_vector.ndim != 2:
        raise ValueError("The provided vectors should have two axis, even if there is only one vector.")

    return vecdot(vector_to_project, projection_direction_vector / np.linalg.norm(projection_direction_vector,axis=-1)[:,None])

def get_vector_projection(vector_to_project, projection_direction_vector):
    """
    Projects one or more vectors along the provided direction(s). NOTE: requires
    Numpy 2.X

    Parameters
    ----------
    vector_to_project: np.ndarray
        The vector(s) to project
    projection_direction_vector: np.ndarray
        The direction(s) along which to project the vector. If more than one
        direction is provided, it needs to be provided individually for each
        vector.

    Returns
    -------
    np.ndarray
    """

    if projection_direction_vector.ndim == 2 and projection_direction_vector.shape != vector_to_project.shape:
        raise ValueError("If multiple projection directions are input, there should be one per vector to project.")

    if projection_direction_vector.shape[-1] != vector_to_project.shape[-1]:
        raise ValueError("The vectors to project and the direction to project along should have the same dimensions along axis = -1")

    return (vecdot(vector_to_project, projection_direction_vector) / vecdot(projection_direction_vector, projection_direction_vector))[:,None] * projection_direction_vector

def cartesian_to_spherical_coordinates(cartesian_coordinates):
    """
    Transforms cartesian coordinates (x, y, z) into spherical coordinates
    (r, theta, phi) and returns the corresponding spherical unit vectors.

    Parameters:
    -----------
    cartesian_coordinates:
        3D position of particles in a cartesian coordinate system.

    Returns:
    -----------
    spherical_coordinates: list
        Tuple containing in each element coordinates along each spherical
        dimension (r, theta, phi).
    spherical_unit_vectors: list
        List containing in each element the unit vector of each point along
        each spherical dimension (u_r, u_theta, u_phi).
    """

    # To simplify algebra.
    x, y, z = cartesian_coordinates.T

    # We collect each dimension in a list to be able to have mixed units.
    # Otherwise we will lose units for the radial coordinate.
    r     = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi   = np.arctan2(y, x)
    spherical_vectors = [r, theta, phi]

    # Compute the unit vectors for each position and each spherical direction.
    # This will be used later to get velocities in spherical coordinates.
    theta = theta[:, np.newaxis]
    phi   = phi  [:, np.newaxis]

    e_x = np.array([1,0,0])
    e_y = np.array([0,1,0])
    e_z = np.array([0,0,1])

    e_r     =  np.sin(theta) * np.cos(phi) * e_x + np.sin(theta) * np.sin(phi) * e_y + np.cos(theta) * e_z
    e_theta =  np.cos(theta) * np.cos(phi) * e_x + np.cos(theta) * np.sin(phi) * e_y - np.sin(theta) * e_z
    e_phi   = -np.sin(phi) * e_x + np.cos(phi) * e_y

    spherical_unit_vectors = [e_r, e_theta, e_phi]

    return spherical_vectors, spherical_unit_vectors

def cartesian_to_spherical_system(cartesian_coordinates, cartesian_velocities = None):
    """
    Returns the spherical velocity coordinates and velocities for a given set of
    cartesian coordinates and velocities.

    Parameters
    ----------
    cartesian_coordinates: np.ndarray
        Cartesian particle positions.
    cartesian_velocities: np.ndarray, opt.
        Cartesian partic velocities. If not provided, the conversion to a spherical
        coordinate system is only done for the coordinates.

    Returns
    -------
    spherical_coordinates: list
        The coordinates of the provided points in a spherical coordinate system.
        We return it as a list of shape (r, theta, phi) because we cannot mix
        units in cosmo arrays.
    spherical_velocities: list
        The velocities of the provided points in a spherical coordinate system.
        We return it as a list of shape (v_r, v_theta, v_phi) for consistency
        with spherical_coordinates. If not cartesian velocities are provided, this
        is not returned.
    """

    spherical_coordinates, spherical_unit_vectors = cartesian_to_spherical_coordinates(cartesian_coordinates)

    if cartesian_velocities is None:
        return spherical_coordinates

    # We project every along every 3D spherical direction
    spherical_velocities = []
    for spherical_unit_vector in spherical_unit_vectors:
        spherical_velocities.append(get_scalar_projection(cartesian_velocities, spherical_unit_vector))

    return spherical_coordinates, spherical_velocities