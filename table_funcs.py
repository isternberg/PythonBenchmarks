from collections import namedtuple
import prettytable

TESTOBJECT = "Testobjekt"

PERCENT_ZEROS = 'Nullenanteil'

PERFORMANCE_GAIN = 'rel. Performancegewinn'

STD_DIVIATION = 'Standardabweichung'

TIME_IN_SEC = 'Zeit in Sek.'

FLOAT_FORMATTING = 5

LEFT = 'l'


TableData = namedtuple('TableData', ['functions_labels', 'items_pro_dimension', 'functions_ranked_by_time', 'results',
                                     'timings'])

ExpendedTableData = namedtuple('ExpendedTableData', ['functions_labels', 'items_pro_dimension', 'functions_ranked_by_time', 'results',
                                     'timings', 'matrix_size_string', 'sparsities'])


def create_summery_tables_for_scipy_vs_numpy_benchmark(expended_table_data):
    """
    Creates summery tables to display the result of the benchmark.
    Parameters
    ----------
    expended_table_data -  All the data needed for the table, save in a ExpendedTableData object.

    """

    output = ""

    for index, sparsity in enumerate(expended_table_data.sparsities):
        output += create_table_for_scipy_vs_numpy_benchmark(index, expended_table_data, sparsity)

    return output


def create_table_for_scipy_vs_numpy_benchmark(index, table_data, sparsity):
    """
    Creates a single table for one part of the banchmark reuslts.
    Parameters
    ----------
    index - index of the given sparsity
    table_data - All the data needed for the table, save in a ExpendedTableData object
    sparsity - the percent of zeros in the matrix

    Returns the results table as string
    -------

    """
    test_objects_description="Testobjekt"
    # Create header
    fit_table = create_table_header_sparsity_benchmark(test_objects_description)
    fit_table.align[test_objects_description] = LEFT
    # Create Table
    for result_function_tuple in table_data.functions_ranked_by_time:
        create_table_row_for_sparsity_benchmark(result_function_tuple, fit_table, index, table_data, sparsity)
    output = fit_table.get_string() + "\n"
    return output


def create_table_row_for_sparsity_benchmark(result_function_tuple, fit_table, index, table_data, sparsity):
    """
    adds a row in the results table.
    Parameters
    ----------
    result_function_tuple - a tuple containing the results and the function's name
    fit_table - the table object
    index - the index of the row that will be added to the table
    table_data - All the data needed for the table, save in a ExpendedTableData object
    sparsity - the percent zeros.

    """
    func_name_string = result_function_tuple[1]
    foo = table_data.functions_labels[result_function_tuple[1]]
    zeros = str(int(sparsity * 100)) + '%'
    tested_func = table_data.functions_labels[result_function_tuple[1]]
    compared_result = table_data.timings['dot_numpy'][index]
    rounded_time_in_sec = round(table_data.timings[result_function_tuple[1]][index], FLOAT_FORMATTING)
    rounded_std_deviation = round(table_data.results[table_data.matrix_size_string][func_name_string][index][1], FLOAT_FORMATTING)
    rounded_performance_gain = round(compared_result / table_data.timings[result_function_tuple[1]][index], 3)
    fit_table.add_row([
        zeros, tested_func, rounded_time_in_sec, rounded_std_deviation, rounded_performance_gain
    ])


def create_table_header_sparsity_benchmark(test_objects_description):
    """
    Creates the table header.
    Parameters
    ----------
    test_objects_description - a string describing the tested object

    """
    return prettytable.PrettyTable([PERCENT_ZEROS,
                                    test_objects_description,
                                    TIME_IN_SEC,
                                    STD_DIVIATION,
                                    PERFORMANCE_GAIN])


def create_table_header(n, test_objects_description):
    """

    Parameters
    ----------
    n - the number of items pro matrix dimension
    test_objects_description - a string describing the tested object

    """
    return prettytable.PrettyTable(['n=%-5s' % n,
                                    test_objects_description,
                                    TIME_IN_SEC,
                                    STD_DIVIATION,
                                    PERFORMANCE_GAIN])


def create_summery_tables_dense_dot_sparse_benchmark(table_data):
    """
    Creates summery tables to display the result of the benchmark.
    Parameters
    ----------
    table_data -  All the data needed for the table, save in a TableData object.

    """
    output=""
    for index, n in enumerate(table_data.items_pro_dimension):
        output += create_table_dense_dot_sparse_benchmark(table_data, index, n)
    return output


def create_summery_tables_for_sparse_matrices_benchmark(table_data, zero_ness):
    """
    Creates summery tables to display the result of the benchmark.
    Parameters
    ----------
    table_data - All the data needed for the table, save in a TableData object.
    zero_ness - the percent zeros

    Returns - a string representation of the tables
    -------

    """
    test_objects_description=TESTOBJECT
    output=""
    for index, n in enumerate(table_data.items_pro_dimension):
        output += create_table_for_sparse_matrices_benchmark(index, n, table_data, zero_ness)
    return output


def create_table_for_sparse_matrices_benchmark(index, n, table_data,  zero_ness):
    """
    Creates a single table for the benchmark results tables
    Parameters
    ----------
    index - the index of the table
    n - the numebr of items pro matrix dimension
    table_data - TableData object containing the results
    zero_ness - the percent zeros

    Returns
    -------

    """
    test_objects_description= TESTOBJECT
    # Create header
    fit_table = create_table_header(n, test_objects_description)
    # Create Table
    fit_table.align[test_objects_description] = LEFT
    for entry in table_data.functions_ranked_by_time:
        create_table_row_for_sparse_matrices_benchmark(entry, fit_table, index, table_data, zero_ness)
    output = fit_table.get_string() + "\n"
    return output


def create_table_dense_dot_sparse_benchmark(table_data, index, n):
    """

    Parameters
    ----------
    table_data - TableData object containing the results
    index - the index of the table
    n - the numebr of items pro matrix dimension

    Returns a string representation of the table
    -------

    """
    test_objects_description=TESTOBJECT
    # Create header
    fit_table = create_table_header(n, test_objects_description)
    # Create Table
    fit_table.align[test_objects_description] = LEFT
    for entry in table_data.functions_ranked_by_time:
        create_table_row(table_data, entry, fit_table, index)
    output = fit_table.get_string() + "\n"
    return output


def create_table_row(table_data, entry, fit_table, index):
    """
    Adds one row to the results table.
    Parameters
    ----------
    table_data - TableData object containing the results
    entry - a tuple containing the result and the function name
    fit_table - the table object
    index - the index of the table row

    """
    empty_cell = ''
    tested_func = table_data.functions_labels[entry[1]]
    compared_result = table_data.timings[table_data.functions_ranked_by_time[0][1]][index]
    rounded_time_in_sec = round(table_data.timings[entry[1]][index], 5)
    rounded_std_deviation = round(table_data.results[entry[1]][index][1], 5)
    rounded_performance_gain = round(compared_result / table_data.timings[entry[1]][index], 3)
    fit_table.add_row([
        empty_cell, tested_func, rounded_time_in_sec, rounded_std_deviation, rounded_performance_gain
    ])


def create_table_row_for_sparse_matrices_benchmark(entry, fit_table, index, table_data, zero_ness):
    """
    Adds a row to the results table
    Parameters
    ----------
    entry  - a tuple containing the result and the function name
    fit_table - the table object
    index - the index of the table row
    table_data - TableData object containing the results
    zero_ness - the index of the table row

    """
    empty_cell = ''
    tested_func = table_data.functions_labels[entry[1]]
    compared_result = table_data.timings[table_data.functions_ranked_by_time[0][1]][index]
    rounded_time_in_sec = round(table_data.timings[entry[1]][index], 5)
    rounded_std_deviation = round(table_data.results[zero_ness][entry[1]][index][1], 5)
    rounded_performance_gain = round(compared_result / table_data.timings[entry[1]][index], 3)
    fit_table.add_row([
        empty_cell, tested_func, rounded_time_in_sec, rounded_std_deviation, rounded_performance_gain
    ])



