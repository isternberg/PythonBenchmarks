import matplotlib.pyplot as plt
import io_funcs as io


def configure_plot(data_object, x_axis_data, padding=3):
    """
    configures the display of the plot.
    Parameters
    ----------
    data_object - an object from type PlotData, containing data such as the x-label, y-label and plot title.
    x_axis_data - a list of values which will be represented on the x axis
    padding - the rightand left padding for the x axis

    Returns
    -------

    """
    markers = ['D', 'p', 's', '*', 'v', '.', 'H']
    line_width = 1
    plt.rcParams.update({'font.size': 14})  # Font size
    fig = plt.figure(figsize=(11, 10))  # width and height of plot
    plt.xlabel(data_object.xlabel)
    plt.ylabel(data_object.ylabel)
    plt.xlim([min(x_axis_data) - padding, max(x_axis_data) + padding])  # values on x-axis
    #plt.title(data_object.title)  # title of plot
    return fig, line_width, markers


def plot_timing_dense_dot_sparse_benchmark(data_object, items_pro_dimension, results_directory=""):
    """
    Creates a plot for the dense_dot_sparse benchmark.
    Parameters
    ----------
    data_object  - an object from type PlotData, containing data such as the results and labels.
    items_pro_dimension - the number of items per matrix dimension
    results_directory - path to the results directory

    """
    fig, line_width, markers = configure_plot(data_object, items_pro_dimension, padding=50)
    for i, ranked_label in enumerate(data_object.data_ranking):
        plt.errorbar(items_pro_dimension, [tup[0] for tup in data_object.results[ranked_label]],
                     [tup[1] for tup in data_object.results[ranked_label]], alpha=0.5, label=data_object.labels[ranked_label],
                     marker=markers[i], lw=line_width, markersize=10, elinewidth=2)
    plt.legend(loc=2, fancybox=True, framealpha=0.8) # add legend with transparent background
    plt.yscale('log')
    io.save_plot_to_file(results_directory, data_object.title, fig)


def plot_timing_sparse_matrices_benchmark(data_object, items_pro_dimension, sparse_functions, sparsity, results_directory=""):
    """
    Creates a plot for the matrix sparsity benchmark.
    Parameters
    ----------
    data_object  - an object from type PlotData, containing data such as the results and labels.
    items_pro_dimension - the number of items per matrix dimension
    sparse_functions - the SciPy sparse implementation which is used,
    sparsity - the percent zeros which is used
    results_directory - path to the results directory

    """
    functions_names = [i.__name__ for i in sparse_functions]
    fig, line_width, markers = configure_plot(data_object, items_pro_dimension, padding=50)

    handles = [] # this will be used to show the legend in reversed order
    for i, sparse_function in enumerate(functions_names):
        timings = get_timings_from_results(data_object, sparse_function, sparsity)
        errors = get_errors_from_results(data_object, sparse_function, sparsity)
        label = get_label_from_data_object(data_object, sparse_function)
        h = plt.errorbar(items_pro_dimension, timings, errors, alpha=0.5, label=label,
                     marker=markers[i], lw=line_width, markersize=10, elinewidth=2)
        handles.append(h)
    plt.legend(loc=2, fancybox=True, framealpha=0.8, handles=handles[::-1])

    io.save_plot_to_file(results_directory, data_object.title, fig)


def get_errors_from_results(data_object, sparse_function, sparsity):
    """
    scraps the std deviations from a list of tuples (mean, std deviation).
    Parameters
    ----------
    data_object - an object from type PlotData, containing data such as the results and labels.
    sparse_function - the SciPy sparse implementation which is used,
    sparsity - the percent zeros which is used

    Returns a list of all std. deviations
    -------

    """
    return [tup[1] for tup in data_object.results[sparsity][sparse_function]]


def make_reduced_plot_for_sparse_matrices_benchmark(data_object, items_pro_dimension, sparse_functions, sparsity, results_directory=""):
    functions_names = [i.__name__ for i in sparse_functions if i.__name__ not in ['dia_matrix', 'dok_matrix', 'lil_matrix']]
    fig, line_width, markers = configure_plot(data_object, items_pro_dimension, padding=50)

    for i, sparse_function in enumerate(functions_names):
        timings = get_timings_from_results(data_object, sparse_function, sparsity)
        errors = get_errors_from_results(data_object, sparse_function, sparsity)
        label = get_label_from_data_object(data_object, sparse_function)
        plt.errorbar(items_pro_dimension, timings, errors, alpha=0.5, label=label,
                     marker=markers[i], lw=line_width, markersize=10, elinewidth=2)
    plt.legend(loc=2, fancybox=True, framealpha=0.8)

    io.save_plot_to_file(results_directory+ "/reduced/", data_object.title, fig)


def get_label_from_data_object(data_object, sparse_function):
    """
    Get the label string  from a data object of type PlotData.
    Parameters
    ----------
    data_object - the PlotData object
    sparse_function  - the function, for which the label is scraped.

    Returns the label
    -------

    """
    return data_object.labels[sparse_function]


def get_timings_from_results(data_object, sparse_function, sparsity):
    """
    Get the timings from the data object from type PlotData
    Parameters
    ----------
    data_object - object of type PlotData
    sparse_function - the funciton for which the timing is scraped
    sparsity - the percent zeros

    Returns - a list of all timings
    -------

    """
    return [tup[0] for tup in data_object.results[sparsity][sparse_function]]


def plot_timing_scipy_vs_numpy_benchmark(data_object, sparsities, sparse_functions, matrix_size_string, results_directory=""):
    """
    Creates a plot for the scipy vs numpy benchmark
    Parameters
    ----------
    data_object - an object from type PlotData, containing data such as the results and labels.
    sparsities - a list of floats representing the percent zero-values in a matrix
    sparse_functions - the functions under test
    matrix_size_string - the matrix dimensions sizes as a string
    results_directory - path to the directory where the plot will be saved.

    """
    functions_names = [i.__name__ for i in sparse_functions]
    fig, line_width, markers = configure_plot(data_object, sparsities, padding=0.01)

    for i, sparse_function in enumerate(functions_names):
        timings = get_timing_from_results_scipy_vs_numpy(data_object, matrix_size_string, sparse_function)
        errors = get_errors_from_results_scipy_vs_numpy(data_object, matrix_size_string, sparse_function)
        label = get_label_from_data_object(data_object, sparse_function)
        plt.errorbar(sparsities, timings, errors, alpha=0.5, label=label,
                     marker=markers[i], lw=line_width, markersize=10, elinewidth=2)
    max_ylim =  (data_object.results[matrix_size_string]['dot_numpy'][0][0]) * 2
    plt.ylim(0, max_ylim)
    plt.legend(loc=1, fancybox=True, framealpha=0.8)

    io.save_plot_to_file(results_directory, data_object.title, fig)


def get_errors_from_results_scipy_vs_numpy(data_object, matrix_size_string, sparse_function):
    """
    Scatps the errors from a the PlotData object.
    Parameters
    ----------
    data_object - a PlotData object.
    matrix_size_string - the size of the matrix dimensions as string.
    sparse_function - the function theat is represented in the plot.

    Returns - a list of all errors
    -------

    """
    return [tup[1] for tup in data_object.results[matrix_size_string][sparse_function]]


def get_timing_from_results_scipy_vs_numpy(data_object, matrix_size_string, sparse_function):
    """
    Scryps the timings from a PlotData object.
    Parameters
    ----------
    data_object - a PlotData object.
    matrix_size_string - the size of the matrix dimensions as string.
    sparse_function - the function theat is represented in the plot.

    Returns - a list of all timings
    -------

    """
    return [tup[0] for tup in data_object.results[matrix_size_string][sparse_function]]


def plot_timing_top_n_benchmark(data_object, items_pro_dimension, results_directory=""):
    """
    Creates a plot for the top-n benchmark.

    Parameters
    ----------
    data_object - an object from type PlotData
    items_pro_dimension - number of items pro matrix dimension
    results_directory - path to where the plot will be saved

    """
    fig, line_width, markers = configure_plot(data_object, items_pro_dimension, padding =100)

    for i, ranked_label in enumerate(data_object.data_ranking):
        plt.errorbar(items_pro_dimension, [tup[0] for tup in data_object.results[ranked_label]],
                     [tup[1] for tup in data_object.results[ranked_label]], alpha=0.5, label=data_object.labels[ranked_label],
                     marker=markers[i], lw=line_width, markersize=10, elinewidth=2)
    plt.yscale('log')
    plt.legend(loc=2, fancybox=True, framealpha=0.8)
    io.save_plot_to_file(results_directory, data_object.title, fig)




