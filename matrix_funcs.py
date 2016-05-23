import numpy as np
from scipy import sparse
from top_n_funcs import convert_matrix_to_sparse_with_top_n


def create_matrix(rows, cols, percent_zeros=0.99):
    """
    Creates a random matrix with a defined percentage of sparsity.
    The matrix contains only zeros and ones.
    Parameters
    ----------
    rows: number of rows in the matrix
    cols: number of columns in the matrix
    percent_zeros: percentage of zeros in the matrix

    Returns numpy 2-dim matrix
    -------

    """
    if percent_zeros < 0 or percent_zeros > 1:
        raise ValueError
    if rows <= 0 or cols <= 0:
        raise ValueError
    matrix = np.ones([rows,cols], dtype=np.int)
    desired_ones_count = int(rows*cols*percent_zeros)
    ones_count = 0
    while ones_count < desired_ones_count:
        row = np.random.random_integers(0,rows-1)
        col = np.random.random_integers(0,cols-1)
        if matrix[row][col]==1 :
            matrix[row][col] = 0
            ones_count+=1
    return matrix


def dot_numpy(matrix_1: np.ndarray, matrix_2: np.ndarray):
    """
    Calculates the dot product using numpy
    Parameters
    ----------
    matrix_1:  numpy-array
    matrix_2:  numpy-array

    Returns: a numpy-array which results from the dot product
    -------

    """
    return np.dot(matrix_1, matrix_2)


def dot_scipy_csc_with_conversion(matrix_1: np.ndarray, matrix_2: np.ndarray):
    """
    Calculates the dot product by converting the parameters to compressed Sparse Column matrices
    Parameters
    ----------
    matrix_1:  numpy-array
    matrix_2:  numpy-array

    Returns: a numpy-array which results from the dot product
    -------

    """
    sparse_result = sparse.csc_matrix(matrix_1).dot(sparse.csc_matrix(matrix_2))
    return np.array(sparse_result.todense())


def dot_scipy_bsr_with_conversion(matrix_1: np.ndarray, matrix_2: np.ndarray):
    """
    Calculates the dot product by converting the parameters to Block Sparse Row matrices
    Parameters
    ----------
    matrix_1:  numpy-array
    matrix_2:  numpy-array

    Returns: a numpy-array which results from the dot product
    -------

    """
    sparse_result = sparse.bsr_matrix(matrix_1).dot(sparse.bsr_matrix(matrix_2))
    return np.array(sparse_result.todense())


def dot_scipy_csr_with_conversion(matrix_1: np.ndarray, matrix_2: np.ndarray):
    """
    Calculates the dot product by converting the parameters to Compressed Sparse Row sparse matrices
    Parameters
    ----------
    matrix_1:  numpy-array
    matrix_2:  numpy-array

    Returns: a numpy-array which results from the dot product
    -------

    """
    sparse_result = sparse.csr_matrix(matrix_1).dot(sparse.csr_matrix(matrix_2))
    return np.array(sparse_result.todense())




def scipy_csc_dot_numpy_with_swap(matrix_dense: np.ndarray, matrix_sparse: np.ndarray):
    """
    Calculates the dot product of two numpy arrays. The matrices are converted to CSC format for fast
    multiplication.

    Parameters
    ----------
    matrix_dense - the first array
    matrix_sparse - the second array.

    Returns a numpy array, which is the result of the matrix multiplication.
    -------

    """
    result = sparse.csc_matrix(matrix_sparse.T).dot(matrix_dense.T)
    return result.T


def scipy_csr_dot_numpy_with_swap(dense_matrix: np.ndarray, sparse_matrix: np.ndarray):
    """
    Calculates the dot product of two numpy arrays. The matrices are converted to CSR format for fast
    multiplication.

    Parameters
    ----------
    matrix_dense - the first array
    matrix_sparse - the second array.

    Returns a numpy array, which is the result of the matrix multiplication.
    -------

    """
    result = sparse.csr_matrix(sparse_matrix.T).dot(dense_matrix.T)
    return result.T


def scipy_bsr_dot_numpy_with_swap(dense_matrix: np.ndarray, sparse_matrix: np.ndarray):
    """
    Calculates the dot product of two numpy arrays. The matrices are converted to BSR format for fast
    multiplication.
    Parameters
    ----------
    matrix_dense - the first array
    matrix_sparse - the second array.

    Returns a numpy array, which is the result of the matrix multiplication.
    -------

    """
    result = sparse.bsr_matrix(sparse_matrix.T).dot(dense_matrix.T)
    return result.T


def scipy_csc_dot_numpy_with_top_n(dense_matrix: np.ndarray, sparse_matrix: np.ndarray, n=20):
    """
    Calculates the dot product of two Matrices of type numpy array. The first array is convert to a sparse matrix with
    top N items in every row. Afterwards both matrices are converted to Sparse matrices from type CSC for
    fast multiplication.
    Parameters
    ----------
    dense_matrix - The first matrix, which will be converted to a top-n matrix
    sparse_matrix - the second matrix
    n = the n value for the top n matrix.

    Returns a numpy array, which is the result of the matrix multiplication.
    -------

    """

    convert_matrix_to_sparse_with_top_n(dense_matrix, n)
    result = sparse.csc_matrix(dense_matrix).dot(sparse.csc_matrix(sparse_matrix))
    return np.array(result.todense())


def scipy_csr_dot_numpy_with_top_n(dense_matrix: np.ndarray, sparse_matrix: np.ndarray, n=20):
    """
    Calculates the dot product of two Matrices of type numpy array. The first array is convert to a sparse matrix with
    top N items in every row. Afterwards both matrices are converted to Sparse matrices from type CSR for
    fast multiplication.
    Parameters
    ----------
    dense_matrix - The first matrix, which will be converted to a top-n matrix
    sparse_matrix - the second matrix
    n = the n value for the top n matrix.

    Returns a numpy array, which is the result of the matrix multiplication.
    -------

    """
    convert_matrix_to_sparse_with_top_n(dense_matrix, n)
    result = sparse.csr_matrix(dense_matrix).dot(sparse.csr_matrix(sparse_matrix))
    return np.array(result.todense())


def scipy_bsr_dot_numpy_with_top_n(dense_matrix: np.ndarray, sparse_matrix: np.ndarray, n=20):
    """
    Calculates the dot product of two Matrices of type numpy array. The first array is convert to a sparse matrix with
    top N items in every row. Afterwards both matrices are converted to Sparse matrices from type BSR for
    fast multiplication.
    Parameters
    ----------
    dense_matrix - The first matrix, which will be converted to a top-n matrix
    sparse_matrix - the second matrix
    n = the n value for the top n matrix.

    Returns a numpy array, which is the result of the matrix multiplication.
    -------

    """
    convert_matrix_to_sparse_with_top_n(dense_matrix, n)
    result = sparse.bsr_matrix(dense_matrix).dot(sparse.bsr_matrix(sparse_matrix))
    return np.array(result.todense())