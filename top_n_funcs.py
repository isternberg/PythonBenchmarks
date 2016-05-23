import numpy as np


def convert_matrix_to_sparse_with_top_n(dense_matrix, n):
    """
    Converts a dense Matrix to a sparse matrix, using a top-n algorithm
    Parameters
    ----------
    dense_matrix - a dense matrix
    n - The number of top items that will be left in every matrix row.

    """
    if(type(dense_matrix) != np.ndarray):
        raise ValueError("Matrix must be a numpy array "
                         "(currently: {0})". format(type(dense_matrix)))
    for row in dense_matrix:
        convert_to_sparse_vector_with_top_n(row, n)


def convert_to_sparse_vector_with_top_n(dense_vector, n):
    """
    Converts a dense vector to sparse using a top-n algorithm
    Parameters
    ----------
    dense_vector - the dense vector
    n - the number of n numerically highest items that will be left in the vector

    """
    indices = get_indices_of_top_n_items(dense_vector, n)
    convert_to_sparse_vector_with_selected_indices(dense_vector, indices)


def get_indices_of_top_n_items(dense_vector: np.array, n: int):
    """
    Gets the indices of the top n items in a given vector
    Parameters
    ----------
    dense_vector - the vector
    n - the number of indices

    Returns the indices of the top n items
    -------

    """
    if n > dense_vector.size or n < 0:
        raise ValueError("N may not be greater than the"
                         " size of the vector's dimension or less than zero")
    elif n == dense_vector.size:
        return np.argpartition(dense_vector, 0)
    elif n == 0:
        return []
    else:
        return np.argpartition(dense_vector, -n)[-n:]


def convert_to_sparse_vector_with_selected_indices(dense_vector, indices):
    """
    Converts a vector to sparse by only keeping the given indices. Other values are replaced with zeros.
    Parameters
    ----------
    dense_vector - the vector
    indices - the indices that should be kept.

    """
    if indices.size == 0:
       dense_vector.fill(0)
       return
    if np.max(indices) >= len(dense_vector) or np.min(indices) < 0:
       raise IndexError("An index out of range has been used")
    top_n = {i: dense_vector[i] for i in indices}
    dense_vector.fill(0)
    for index in top_n:
        dense_vector[index] = top_n[index]
