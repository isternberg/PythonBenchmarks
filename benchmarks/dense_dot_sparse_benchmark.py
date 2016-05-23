import benchmark_funcs as bf
import matrix_funcs as mf
import io_funcs as io
import date_funcs as df
import plotting_funcs as pf
import table_funcs as tf
from data_containers.plot_data import PlotData

FILENAME = "Matrixmultiplikation: Dicht besetzt dot dünn besetzt"

TEST_OBJECTS_DESCRIPTION = 'Testobjekt'

PLOT_Y_LABEL = 'Durchschnittliche Rechenzeit [Sek]'

PLOT_X_LABEL = 'Matrixgröße $N \\times N$'

TEST_NAME = 'Ergebnisse des 3. Benchmarks\n '

RESULTS_DIRECTORY = "results/"

BENCHMARK_DIRECTORY = "03_DenseDotSparseBenchmark/"


def run_performance_test(items_pro_dimension, number_of_timings, functions):
    """
    Runs the Benchmark.
    Parameters
    ----------
    items_pro_dimension - number of items in each matrix dimension
    number_of_timings  - number of repeats for each timing
    functions - the functions under test

    Returns a dictionary with the avg. results and std. for each function
    -------

    """
    test_results = {f.__name__:[] for f in functions}
    for n in items_pro_dimension:
        dense_matrix = mf.create_matrix(n, n, 0.01)
        sparse_matrix = mf.create_matrix(n, n, 0.99)
        for func in functions:
            test_results[func.__name__].append(bf.test_performance(func, number_of_timings, dense_matrix, sparse_matrix))
            print(func.__name__, n)
    return test_results


def get_timings_from_results(results):
    """
    Scraps the timings from the results dictionary
    Parameters
    ----------
    results - a dictionary containing the entire test results

    Returns  a dictionary with the timings for each function under test.
    -------

    """
    return {func_name:[result_pair[0] for result_pair in results[func_name]] for func_name in results}



def get_functions_under_test():
    """
    Returns a list with all functions under test.
    -------

    """
    return [mf.dot_numpy,  mf.dot_scipy_csc_with_conversion,
             mf.dot_scipy_bsr_with_conversion, mf.dot_scipy_csr_with_conversion]



def create_functions_aliases():
    """
    Creates aliases to the function names in order to display them in the plots.
    Returns a dictionary with the function names as keys and aliases as values.
    -------

    """
    return  {'dot_numpy': 'Numpy (Referenz)', 'dot_scipy_csc_with_conversion': 'Compressed Sparse Column',
         'dot_scipy_bsr_with_conversion': 'Block Sparse Row', 'dot_scipy_csr_with_conversion':'Compressed Sparse Row'}



def generate_reduced_plot_data(labels, results, functions_ranked_by_time, x_label =PLOT_X_LABEL, y_label=PLOT_Y_LABEL,
                               test_name = TEST_NAME,  omitted_functions=[],
                               comment_about_omitted_function=""):
    """
    Recreates the plot data with less functions.
    Parameters
    ----------
    labels - a dictionary with the functions as keys and their aliases as values
    results - the entire test results.
    functions_ranked_by_time - a list of tuples in the form:[(resuls, function name)..]
    x_label - the plot's x label
    y_label - the plot's y label
    test_name - the name of the test
    omitted_functions - a list of functions that should be ignored in the plot.
    comment_about_omitted_function - a comment referring to the ommited functions.

    Returns a PlotData object with reduced information.
    -------

    """
    plot_title = test_name + comment_about_omitted_function
    plot_labels = {k: v for k, v in labels.items() if k not in omitted_functions}
    plot_results = {k: v for k, v in results.items() if k not in omitted_functions}
    plot_ranking = [ranked_label for time, ranked_label in functions_ranked_by_time if
                    ranked_label not in omitted_functions]
    plot_data = PlotData(plot_title, plot_labels, plot_results, plot_ranking, x_label, y_label)
    return plot_data


def persist_summery_table(number_of_timings, file_path, results_table, test_name):
    """
    Persists the benchmark results-table to a text file.
    Parameters
    ----------
    number_of_timings - the number of time every test was repeated.
    file_path - path to results file
    results_table - the results table
    test_name - the name of the test
    -------

    """
    file_name = 'summery_table.txt'
    io.persist_to_text_file(test_name +'\nFür jede Matrixgröße wurde die Ausführungszeit {0} Mal gemessen.\n'
                            .format(number_of_timings), file_path, file_name)
    io.persist_test_related_info(file_path, file_name)
    io.persist_to_text_file(results_table, file_path, file_name)


def create_summery_table(table_data):
    """
    Creates a PrettyTable from the test resuls.
    Parameters
    ----------
    table_data - A TableData object, containing the test results and other necessary data.

    Returns a string representation of the PrettyTable.
    -------

    """
    table = tf.create_summery_tables_dense_dot_sparse_benchmark(table_data)
    return table


def persist_plots(items_pro_matrix_dimension, results_directory, *argv: PlotData):
    """
    Creates plots and persists them in a png file.
    Parameters
    ----------
    items_pro_matrix_dimension - amount of items pro matrix dimension
    results_directory - path to results directory
    argv - the plot data.
    -------

    """
    for plot_data in argv:
        pf.plot_timing_dense_dot_sparse_benchmark(plot_data, items_pro_matrix_dimension, results_directory)


def rank_functions_by_performance(timings):
    """
    Creates a list of the functions ranked by their timing (slowest first).
    Parameters
    ----------
    timings - the avg. timing of each function

    Returns a list in the following form: [(timing, function name), ..]
    -------

    """
    functions_ranked_by_time = sorted([(time[1][-1], time[0]) for time in timings.items()], reverse=True)
    return functions_ranked_by_time


def backup_results(results_path, results, file_name):
    """
    Backs up the test results to a PKL file
    Parameters
    ----------
    results_path - path to backup file
    results - the results object
    file_name - name of back up file.
    -------

    """
    io.save_results_to_pkl(results_path, results, file_name)

if __name__ == '__main__':
    benchmark_timestamp = df.get_date()
    results_path = RESULTS_DIRECTORY + BENCHMARK_DIRECTORY + benchmark_timestamp + "/"
    functions_under_test = get_functions_under_test()
    items_pro_dimension = [1000, 2000, 3000, 4000, 5000]
    number_of_timings_pro_function_and_matrix_dimension = 5

    results = run_performance_test(items_pro_dimension, number_of_timings_pro_function_and_matrix_dimension,
                                   functions_under_test)
    backup_results(results_path, results, 'dense_dot_sparse')

    timings = get_timings_from_results(results)
    functions_ranked_by_time = rank_functions_by_performance(timings)
    functions_labels = create_functions_aliases()
    table_data = tf.TableData(functions_labels, items_pro_dimension, functions_ranked_by_time, results, timings)

    ranked_times = [ranked_label for time, ranked_label in functions_ranked_by_time]
    plot__data = PlotData(FILENAME, functions_labels, results, ranked_times, PLOT_X_LABEL, PLOT_Y_LABEL)
    persist_plots(items_pro_dimension, results_path, plot__data)
    results_table = create_summery_table(table_data)
    persist_summery_table(number_of_timings_pro_function_and_matrix_dimension, results_path, results_table, TEST_NAME)
