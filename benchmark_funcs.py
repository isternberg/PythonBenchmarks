import numpy as np
import timeit
import platform
import multiprocessing
import scipy



def get_test_related_information():
    """
    Creates a string which contains the configuration data of the test.
    Returns  - the configuration data of the test.
    -------
    """
    data =  '\nMachine\t\t\t\t\t\tToshiba TECRA R950-191\n'
    data += 'Prozessor\t\t\t\t\tIntel(R) Core(TM) i5-3340M CPU @ 2.70GHz\n'
    data += 'System\t\t\t\t\t\t{0}\n'.format(platform.platform())
    data += 'CPUs-Zahl\t\t\t\t\t{0}\n'.format(multiprocessing.cpu_count())
    data += 'Pythoncompiler\t\t\t\t{0}\n'.format(platform.python_compiler())
    data += 'Pythonimplementaion\t\t\t{0}\n' .format(platform.python_implementation())
    data += 'Pythonversion\t\t\t\t{0}\n'.format(platform.python_version())
    data += 'Scipyversion\t\t\t\t{0}\n'.format(scipy.__version__)
    data += 'Numpyversion\t\t\t\t{0}\n'.format(np.__version__)
    data += 'Virtualenvversion\t\t\t13.1.2\n\n'
    return data


def test_performance(func, repeats, matrix_1, matrix_2):
    """
    creates a list of performance test results.
    Parameters
    ----------
    func - the function under test
    repeats - repeat for the time measurement
    matrix_1 - first parameter of the function under test
    matrix_2 - send parameter of the function under test

    Returns a results list. The list contains tuples in the following form: (mean, std. deviation)
    -------

    """
    all_results = []
    for i in range(repeats):
        all_results.append(measure_time(func,matrix_1, matrix_2))
    return (np.mean(all_results), np.std(all_results))


def measure_time(func, matrix_1, matrix_2):
    """
    Measures the clock time while running a function
    Parameters
    ----------
    func - the function under test
    matrix_1  - first parameter of the function under test
    matrix_2  - second parameter of the function under test

    Returns the measured time in seconds
    -------

    """

    start = timeit.default_timer()
    func(matrix_1, matrix_2)
    end = timeit.default_timer()
    return end - start


def test_performance_dot_sparse(repeats, matrix_1, matrix_2):
    """
    creates a list of performance test results for the sparse matrix multiplication.
    Parameters
    ----------
    repeats - number of repeats for the time measurements
    matrix_1 - the first matrix for the multiplication
    matrix_2 - the second matrix for the multiplication

    Returns a results list. The list contains tuples in the following form: (mean, std. deviation)
    -------

    """
    all_results = []
    for i in range(repeats):
       all_results.append(measure_time_dot_sparse(matrix_1, matrix_2))
    return (np.mean(all_results), np.std(all_results))


def measure_time_dot_sparse(matrix_1, matrix_2):
    """
    Measures the clock time while running a SciPy sparse dot function
    Parameters
    ----------
    matrix_1  - the first matrix
    matrix_2  - the second matrix

    Returns the measured time in seconds
    -------

    """
    start = timeit.default_timer()
    matrix_1.dot(matrix_2)
    end = timeit.default_timer()
    return end - start

