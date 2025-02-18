import numpy as np
import swiftsimio as sw

def build_matrix(flattened_matrix):
    """
    Creates a (ndim,ndim) representation of a symmetric matrix, given
    a flattened SOAP representation of shape (ndim * (ndim + 1) / 2).

    flat_matrix: swiftsimio.cosmo_array
        One or more flattened representations of a (ndim x ndim) matrix, where
        the first ndim columns are diagonal elements and the rest are 
        off-diagonal. For example, the flattened representation of a single 
        2D array would be:

        np.array([11, 22, 12])

        where the values indicate the array location in the (2,2) matrix.

    Returns
    -------
    matrix: swiftsimio.cosmo_array
        The (ndim, ndim) representations of the provided matricies. For the 
        above 2D example, it would correspond to:

        np.array([[11,12],[12,22]])
    """

    # Guess number of dimensions based on the first flattened representation.
    for ndim in range(1, 5):
        if ndim * (ndim + 1) / 2 == len(flattened_matrix[0]):
            break

    # We check if we found a solution above. If not, the input may be incorrect
    if (ndim * (ndim + 1) / 2 != len(flattened_matrix[0])):
        print("Could not find number of dimensions based on input. Exiting.")
        return

    number_of_matricies, size_of_matricies = flattened_matrix[:,:ndim].shape

    # We create the output cosmo array with correct dtype and units
    matrix = sw.cosmo_array(np.ones((number_of_matricies,ndim, ndim)), units = flattened_matrix.units, cosmo_factor = flattened_matrix.cosmo_factor, comoving = flattened_matrix.comoving)

    # Handle diagonals
    matrix_index = np.arange(number_of_matricies).T[:,None]
    row_idx, col_idx = np.tril_indices(ndim)

    # Identify combinations for diagonals
    diagonal = (row_idx == col_idx)
    off_diagonal = ~diagonal

    # Fill in values: diag, lower and upper triangles.
    matrix[matrix_index, row_idx[diagonal], col_idx[diagonal]] = flattened_matrix[:,:ndim]
    matrix[matrix_index, row_idx[off_diagonal], col_idx[off_diagonal]] = flattened_matrix[:,ndim:]
    matrix[matrix_index, col_idx[off_diagonal], row_idx[off_diagonal]] = flattened_matrix[:,ndim:]

    return matrix


if __name__ == "__main__":

    # How many random matricies we generate
    number_test_matricies = 100

    # Test implementation in relevant dimensions
    for ndim in [2,3]:

        # Number elements in flattened array for the chosen
        # dimensions
        entries = int(ndim * (ndim + 1) / 2)

        # Generate random tests for 2D, currently in interval [0,1)
        random_flattened_matrix = sw.cosmo_array(np.random.random((number_test_matricies,entries)), units = sw.units.unyt.Mpc, comoving = False, cosmo_factor = sw.cosmo_factor(sw.a**1, scale_factor=1))

        # Make the (ndim, ndim) representation
        reconstructed_matrix = build_matrix(random_flattened_matrix)

        # Test if symmetric
        assert((reconstructed_matrix.swapaxes(1,2) == reconstructed_matrix).all())

        # Test diagonal elements
        assert((np.diagonal(reconstructed_matrix,axis1=1,axis2=2) == random_flattened_matrix[:,:ndim]).all())

        # Test off-diagonal elements. We vstack to retrieve off-diagonal elements for
        # all matricies in one go without using the same method I use
        # in the function we are testing. NOTE: if I do not use .vale, the hstack fails...
        flattened_off_diagonal = []
        for i in range(ndim - 1):
            flattened_off_diagonal.append(reconstructed_matrix[:,1 + i:,i].value)

        flattened_off_diagonal = np.hstack(flattened_off_diagonal)
        assert((flattened_off_diagonal == random_flattened_matrix[:,ndim:].value).all())

        # Test cosmo array properties
        assert(reconstructed_matrix.units == random_flattened_matrix.units)
        assert(reconstructed_matrix.comoving == random_flattened_matrix.comoving)
        assert(reconstructed_matrix.cosmo_factor == random_flattened_matrix.cosmo_factor)

        # Print the first example
        print (f"Dimension number {ndim} suceeded.")

        print ("Original flattened array: ")
        print (random_flattened_matrix[0])

        print ("Reconstructed array: ")
        print (reconstructed_matrix[0])

        print ()

