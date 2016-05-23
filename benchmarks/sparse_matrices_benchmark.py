import benchmark_funcs as bf
import plotting_funcs as pf
import date_funcs as df
import io_funcs as io
import table_funcs as tf
from data_containers import plot_data
from data_containers.plot_data import PlotData
from scipy.sparse import *


from matrix_funcs import create_matrix

SUMMERY_TABLE_FILE = 'summery_table_sparse_matrices_benchmark.txt'

RESULTS_FILENAME = "sparse_matrices_benchmark_results"

BENCHMARK_SUBDIRECTORY = "01_SparseMatricesBenchmark/"

RESULTS_DIRECTORY = "results/"

PLOT_X_LABEL = 'Matrixgröße $N \\times N$'

PLOT_Y_LABEL = "Durchschnittliche Rechenzeit [Sek]"


def create_functions_aliases():
    """
    Creates aliases to the function names in order to display them in the plots.
    Returns a dictionary with the function names as keys and aliases as values.
    -------

    """
    return{'bsr_matrix': 'Block Sparse Row', 'coo_matrix': 'Coordinate',
                        'csc_matrix': 'Compressed Sparse Column',
                        'csr_matrix': 'Compressed Sparse Row', 'dia_matrix': 'Diagonal storage',
                        'dok_matrix': 'Dictionary Of Keys ', 'lil_matrix': 'Row-based linked list'}


def run_performance_test(sparsities, sparse_matrices, items_pro_dimension, number_of_timings):
    """
    Runs the benchmark
    Parameters
    ----------
    sparsities a list of values between 0 and 1 the represent the tested sparsities
    sparse_matrices - the matrices under test
    items_pro_dimension - amount of items pro matrix dimension
    number_of_timings - amount of repeats pro test

    Returns a dictionary containing the avg timing and std. for each tested matrix.
    -------

    """
    results = {sparsity: {sm.__name__: [] for sm in sparse_matrices} for sparsity in sparsities}
    for sparsity in sparsities:
        for n in items_pro_dimension:
            M_1 = create_matrix(n, n, sparsity)
            M_2 = M_1.T

            for sm in sparse_matrices:
                M_1 = sm(M_1)
                M_2 = sm(M_2)
                results[sparsity][sm.__name__].append(bf.test_performance_dot_sparse(number_of_timings, M_1, M_2))
                print(sparsity, sm.__name__, n)
    return results


def get_timings_from_performance_test_results(results):
    """
    Scraps the timings from the results dictionary
    Parameters
    ----------
    results - a dictionary containing the entire test results

    Returns  a dictionary with the timings for each function under test.
    -------

    """
    return {key:
                   {func_name:
                        [result_pair[0] for result_pair in results[key][func_name]]
                    for func_name in results[key]}
               for key in results}


def rank_functions_by_performance(timings):
    """
    Creates a list of the functions ranked by their timing (slowest first).
    Parameters
    ----------
    timings - the avg. timing of each matrix and sparsity

    Returns a dict in the following form: {sparsity: (timing, matrix name), ..}
    -------

    """
    functions_ranked_by_time = {key: sorted([(time[1][-1], time[0]) for time in timings[key].items()], reverse=True)
            for key in timings.keys()}
    return functions_ranked_by_time


def backup_results(results_path, results):
    """
    Backs up the results to a PKL file
    Parameters
    ----------
    results_path - path to Results directory
    results - the results dictionary
    -------

    """
    io.save_results_to_pkl(results_path, results, RESULTS_FILENAME)


def persist_summery_table(table_data, key, results_path):
    """
    Persists the benchmark results to a PrettyTable in a text file.
    Parameters
    ----------
    table_data - a TableData object, containing the data needed for creating the table
    key - a key in the results dict, whose the values are persisted in the table
    results_path - path to results directory
    -------

    """
    results_table = tf.create_summery_tables_for_sparse_matrices_benchmark(table_data, key)
    io.persist_to_text_file('\nErgebnisse bei {0} Prozent Nullen in jeder Matrix:\n'.format(int(key * 100)),
                            results_path, SUMMERY_TABLE_FILE)
    io.persist_to_text_file(results_table, results_path, SUMMERY_TABLE_FILE)


def persist_benchmark_data_to_summery_table_file(number_of_timings, results_path):
    """
    Persists general benchmark data to the results text file.
    Parameters
    ----------
    number_of_timings - number of repeats for each test.
    results_path - path to the results directory.
    -------

    """
    io.persist_to_text_file('Ergebnisse des 1. Benchmarks\n\nFür jede Matrixgröße wurde die Ausführungszeit {0} Mal gemessen.\n'
                            .format(number_of_timings), results_path, SUMMERY_TABLE_FILE)
    io.persist_test_related_info(results_path, SUMMERY_TABLE_FILE)


def create_plot_title(key):
    """
    Creates the title of a plot.
    Parameters
    ----------
    key - a float which represent the amount of zeros in the tested matrix.

    Returns the plot title.
    -------

    """
    return 'Sparsematrizenmultiplikation-Performance: Anteil der Nullen: {0}% '.format(int(key * 100))


if __name__ == "__main__":
    benchmark_timestamp = df.get_date()
    results_path = RESULTS_DIRECTORY + BENCHMARK_SUBDIRECTORY + benchmark_timestamp + "/"
    sparse_matrices= [bsr_matrix, coo_matrix, csc_matrix, csr_matrix, dia_matrix, dok_matrix, lil_matrix]
    sparsities = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    items_pro_matrix_dimension = [100, 500, 1000, 2000, 3000]
    number_of_timings = 5

    results = run_performance_test(sparsities, sparse_matrices, items_pro_matrix_dimension, number_of_timings)
    backup_results(results_path, results)

    timings = get_timings_from_performance_test_results(results)
    functions_labels = create_functions_aliases()
    functions_ranked_by_performance = rank_functions_by_performance(timings)
    persist_benchmark_data_to_summery_table_file(number_of_timings, results_path)

    for key in sparsities:
        plot_title = create_plot_title(key)
        plot_data = PlotData(plot_title, functions_labels, results, functions_ranked_by_performance, PLOT_X_LABEL,
                             PLOT_Y_LABEL)
        pf.plot_timing_sparse_matrices_benchmark(plot_data, items_pro_matrix_dimension, sparse_matrices, key,
                                                 results_path)

        pf.make_reduced_plot_for_sparse_matrices_benchmark(plot_data, items_pro_matrix_dimension, sparse_matrices, key,
                                                 results_path)

        table_data = tf.TableData(functions_labels, items_pro_matrix_dimension, functions_ranked_by_performance[key],
                                  results, timings[key])
        persist_summery_table(table_data, key, results_path)
