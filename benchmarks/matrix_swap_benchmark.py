import matrix_funcs as mf
import date_funcs as df
import table_funcs as tf
import benchmarks.dense_dot_sparse_benchmark as dds
import numpy as np
from data_containers.plot_data import PlotData

FILENAME = "matrix_swap_benchmark_results"

TEST_NAME = 'Matrixmultiplikation: Dünn besetzt mal dicht besetzt '

TABLE_HEADLINE = "Ergebnisse des 4. Benchmarks\n"

RESULTS_DIRECTORY = "results/"

BENCHMARK_DIRECTORY = "04_MatrixSwapBenchmark/"

PLOT_Y_LABEL = 'Durchschnittliche Rechenzeit [Sek]'

PLOT_X_LABEL = 'Matrixgröße $N \\times N$'


def get_functions_under_test():
    """
    Returns a list with all functions under test.
    -------
    """
    return [mf.dot_numpy,  mf.scipy_csc_dot_numpy_with_swap, mf.scipy_csr_dot_numpy_with_swap,
            mf.scipy_bsr_dot_numpy_with_swap ]


def create_functions_aliases():
    """
    Creates aliases to the function names in order to display them in the plots.
    Returns a dictionary with the function names as keys and aliases as values.
    -------

    """
    return  {'dot_numpy':'Numpy x Numpy (Referenz)', 'scipy_csc_dot_numpy_with_swap':'Compressed Sparse Column x Numpy',
             'scipy_csr_dot_numpy_with_swap':'Compressed Sparse Row x Numpy',
             'scipy_bsr_dot_numpy_with_swap':'Block Sparse Row x Numpy'}


if __name__ == '__main__':
    benchmark_timestamp = df.get_date()
    results_path = RESULTS_DIRECTORY + BENCHMARK_DIRECTORY + benchmark_timestamp + "/"
    functions_under_test = get_functions_under_test()
    items_pro_dimension = [500, 1000, 2000, 3000, 4000, 5000]
    number_of_timings_pro_function_and_matrix_dimension = 5

    results = dds.run_performance_test(items_pro_dimension, number_of_timings_pro_function_and_matrix_dimension,
                               functions_under_test)
    dds.backup_results(results_path, results, FILENAME)

    timings = dds.get_timings_from_results(results)
    functions_ranked_by_time = dds.rank_functions_by_performance(timings)
    functions_labels = create_functions_aliases()
    table_data = tf.TableData(functions_labels, items_pro_dimension, functions_ranked_by_time, results, timings)

    ranked_times = [ranked_label for time, ranked_label in functions_ranked_by_time]
    plot__data = PlotData(TEST_NAME, functions_labels, results, ranked_times, PLOT_X_LABEL, PLOT_Y_LABEL)
    dds.persist_plots(items_pro_dimension, results_path, plot__data)
    results_table = dds.create_summery_table(table_data)
    dds.persist_summery_table(number_of_timings_pro_function_and_matrix_dimension, results_path, results_table, TABLE_HEADLINE)
