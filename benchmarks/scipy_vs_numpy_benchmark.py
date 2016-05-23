from data_containers.plot_data import PlotData
import benchmark_funcs as bf
import matrix_funcs as mf
import table_funcs as tf
import plotting_funcs as pf
import date_funcs as df
import io_funcs as io

RESULTS_DIRECTORY = "results/"

RESULTS_FILENAME = "scipy_vs_numpy_benchmark_results"

BENCHMARK_DIRECTORY = "02_scipy_vs_numpy/"

SUMMERY_TABLE_FILE = "summery_table_scipy_vs_numpy_benchmark.txt"

PLOT_X_LABEL = "Anteil der Nullen "

PLOT_Y_LABEL = "Durchschnittliche Rechenzeit [Sek]"




def run_performance_test(functions, items_pro_dimension, sparsities):
    """
    Runs the benchmark.
    Parameters
    ----------
    functions - the functions under test
    items_pro_dimension - number of items pro matrix dimenstion
    sparsities - a list of values between 0 and 1 which define the percent zeros in each matrix.

    Returns a dictionary with the avg. results and std. for each function
    -------

    """
    results = {size: {f.__name__: [] for f in functions} for size in items_pro_dimension}
    for key, dimension in items_pro_dimension.items():
        for sparsity in sparsities:
            matrix_1 = mf.create_matrix(dimension, dimension, sparsity)
            matrix_2 = matrix_1.T
            for func in functions:
                results[key][func.__name__].append(bf.test_performance(func, number_of_timings,
                                                                       matrix_1, matrix_2))
                print(key, sparsity, func)
    return results


def get_timings_from_results(results):
    """
    Scraps the timings from the results dictionary
    Parameters
    ----------
    results - a dictionary containing the entire test results

    Returns  a dictionary with the timings for each function under test.
    -------

    """
    return {key: {func_name: [result_pair[0] for result_pair in results[key][func_name]]
                  for func_name in results[key]} for key in results}


def create_functions_aliases():
    """
    Creates aliases to the function names in order to display them in the plots.
    Returns a dictionary with the function names as keys and aliases as values.
    -------

    """
    return {'dot_numpy': 'Numpy (Referenz)',
            'dot_scipy_bsr_with_conversion': 'Block Sparse Row',
            'dot_scipy_csc_with_conversion': 'Compressed Sparse Column',
            'dot_scipy_csr_with_conversion': 'Compressed Sparse Row'}


def rank_functions_by_performance(timings):
    """
    Creates a list of the functions ranked by their timing (slowest first).
    Parameters
    ----------
    timings - the avg. timing of each function

    Returns a list if the following form: [(timing, function name), ..]
    -------

    """
    return {key: sorted([(time[1][-1], time[0]) for time in timings[key].items()],
                        reverse=True) for key in timings.keys()}

def persist_benchmark_data_to_summery_table_file(number_of_timings, results_path):
    """
    Persists the benchmark'S related data to the results file.
    Parameters
    ----------
    number_of_timings - the amount of repeats for each test.
    results_path - path to results file.
    -------

    """
    io.persist_to_text_file('Ergebnisse des 2. Benchmarks\n\nFür jede Matrixgröße wurde die Ausführungszeit {0} Mal gemessen.\n'
                            .format(number_of_timings), results_path, SUMMERY_TABLE_FILE)
    io.persist_test_related_info(results_path, SUMMERY_TABLE_FILE)


def persist_summery_table(table_data, results_path):
    """
    Persists the test results to a summery table in a text file.
    Parameters
    ----------
    table_data - an ExpendedTableData object, containing all the data needed for the table
    results_path  -path to results file
    -------

    """
    results_table = tf.create_summery_tables_for_scipy_vs_numpy_benchmark(table_data)
    io.persist_to_text_file('\nResults for {0} matrix:\n'.format(table_data.matrix_size_string),
                            results_path, SUMMERY_TABLE_FILE)
    io.persist_to_text_file(results_table, results_path, SUMMERY_TABLE_FILE)

def get_functions_under_test():
    """
    Returns a list with all functions under test.
    -------

    """
    return [mf.dot_numpy,  mf.dot_scipy_csc_with_conversion,
             mf.dot_scipy_bsr_with_conversion, mf.dot_scipy_csr_with_conversion]



if __name__ == "__main__":
    benchmark_timestamp = df.get_date()
    results_path = RESULTS_DIRECTORY + BENCHMARK_DIRECTORY + benchmark_timestamp + "/"
    functions = get_functions_under_test()
    matrix_sizes_values = [500, 1000, 2000, 3000]
    matrix__sizes_keys = ['{0}x{1}' .format(str(i), str(i)) for i in matrix_sizes_values]
    items_pro_dimension =dict(zip(matrix__sizes_keys, matrix_sizes_values))

    sparsities = [0.4, 0.45, 0.5, 0.55, 0.6] #percent zeros
    number_of_timings = 5
    functions_labels = create_functions_aliases()

    results = run_performance_test(functions, items_pro_dimension, sparsities)
    io.save_results_to_pkl(results_path, results, RESULTS_FILENAME)

    timings = get_timings_from_results(results)
    functions_ranked_by_performance = rank_functions_by_performance(timings)
    persist_benchmark_data_to_summery_table_file(number_of_timings, results_path)

    for key in matrix__sizes_keys:
        plot_title = 'Matrixmultiplikation-Performance: {0} Matrix' .format(key)
        plot_data =  PlotData(plot_title, functions_labels, results, functions_ranked_by_performance, PLOT_X_LABEL,
                             PLOT_Y_LABEL)
        pf.plot_timing_scipy_vs_numpy_benchmark(plot_data, sparsities, functions, key, results_path)


        table_data = tf.ExpendedTableData(functions_labels, matrix_sizes_values, functions_ranked_by_performance[key],
                                  results, timings[key], key, sparsities)
        persist_summery_table(table_data, results_path)

