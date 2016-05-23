import date_funcs as df
import matrix_funcs as mf
import benchmarks.dense_dot_sparse_benchmark as dds
import table_funcs as tf
import benchmark_funcs as bf
from data_containers.plot_data import PlotData
import plotting_funcs as pf

TABLE_HEADLINE = "Ergebnisse des 5. Benchmarks"

RESULTS_DIRECTORY = "results/"

BENCHMARK_DIRECTORY = "05_TopNBenchmark/"

FILENAME = "top_n_benchmark_results"

TEST_NAME = 'Matrixmultiplikation: Top-N-Optimierung  '

PLOT_Y_LABEL = 'Durchschnittliche Rechenzeit [Sek]'

PLOT_X_LABEL = 'Matrixgröße $N \\times N$'

def create_functions_aliases():
    """
    Creates aliases to the function names in order to display them in the plots.
    Returns a dictionary with the function names as keys and aliases as values.
    -------

    """
    return {'dot_numpy': 'Numpy (Referenz)', 'scipy_csc_dot_numpy_with_top_n': 'Compressed Sparse Column',
            'scipy_csr_dot_numpy_with_top_n': 'Compressed Sparse Row', 'scipy_bsr_dot_numpy_with_top_n': 'Block Sparse Row'}

def get_functions_under_test():
    """
    Returns a list with all functions under test.
    -------

    """
    return [mf.dot_numpy, mf.scipy_csc_dot_numpy_with_top_n,
            mf.scipy_csr_dot_numpy_with_top_n, mf.scipy_bsr_dot_numpy_with_top_n]


def persist_plots(items_pro_matrix_dimension, results_directory, *argv: PlotData):
    """
    Persists the Benchmarks results as a plot in a PNG file.
    Parameters
    ----------
    items_pro_matrix_dimension  - amount of items pro matrix dimension
    results_directory - path to results directory
    argv - one or more PlotData objects, containing te necessary data for the plots.
    -------

    """
    for plot_data in argv:
        pf.plot_timing_top_n_benchmark(plot_data, items_pro_matrix_dimension, results_directory)


def run_performance_test(items_in_matrix, number_of_timings, functions):
    test_results = {f.__name__:[] for f in functions}
    for n in items_in_matrix:
        dense_matrix = mf.create_matrix(n, n, 0.01)
        sparse_matrix = mf.create_matrix(n, n, 0.99)
        for func in functions:
            test_results[func.__name__].append(bf.test_performance(func, number_of_timings, dense_matrix, sparse_matrix))
            print(func.__name__, n)
    return test_results

if __name__ == '__main__':
    benchmark_timestamp = df.get_date()
    results_path = RESULTS_DIRECTORY + BENCHMARK_DIRECTORY + benchmark_timestamp + "/"
    functions_under_test = get_functions_under_test()
    items_pro_dimension = [500, 1000, 2000, 3000, 4000]
    number_of_timings_pro_function_and_matrix_dimension = 5

    results = run_performance_test(items_pro_dimension, number_of_timings_pro_function_and_matrix_dimension,
                               functions_under_test)
    dds.backup_results(results_path, results, FILENAME)
    timings = dds.get_timings_from_results(results)
    functions_ranked_by_time = dds.rank_functions_by_performance(timings)
    functions_labels = create_functions_aliases()
    table_data = tf.TableData(functions_labels, items_pro_dimension, functions_ranked_by_time, results, timings)

    plot__data = dds.generate_reduced_plot_data(functions_labels, results, functions_ranked_by_time, PLOT_X_LABEL,
                                                PLOT_Y_LABEL, TEST_NAME)
    persist_plots(items_pro_dimension, results_path, plot__data)

    results_table = dds.create_summery_table(table_data)
    dds.persist_summery_table(number_of_timings_pro_function_and_matrix_dimension, results_path, results_table,
                              TABLE_HEADLINE)