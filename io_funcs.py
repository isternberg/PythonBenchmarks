import os
import pickle
import benchmark_funcs as bf


def persist_to_text_file(data:str, path, filename):
    """
    Persists a a string to a text file to a given path. If the the given file and directory do not exist,
    they will be created.

    Parameters
    ----------
    data - The string that will be persisted
    path - Path to file
    filename  -The name of the file

    """
    assert(type(data) == str)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + filename, 'at') as f:
        f.write(data)


def save_plot_to_file(path, filename, fig):
    """
    saves a plot to a PNG file ina given directory. If the the given file and directory do not exist,
    they will be created.

    Parameters
    ----------
    path - the file path
    filename - the file name
    fig - the plot

    """
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path,'{0}.png'.format(filename)), bbox_inches='tight')


def save_results_to_pkl(path, obj_to_save, filename):
    """
    Saves a python object to a okl file in a given path. If the the given file and directory do not exist,
    they will be created.

    Parameters
    ----------
    path -  the file path
    obj_to_save - the python object
    filename - the file name

    """
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + filename + '.pkl', 'wb') as f:
        pickle.dump(obj_to_save, f, pickle.HIGHEST_PROTOCOL)


def load_results_from_pkl(path, filename):
    """
    loads a python object from a pkl file.
    Parameters
    ----------
    path - path to file
    filename - the file name

    Returns -  the python object
    -------

    """
    with open(path + filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def persist_test_related_info(results_path, filename):
    """
    persists the test configuration data to a test file.
    Parameters
    ----------
    results_path -  path to text file
    filename -  the file name.

    """
    data = bf.get_test_related_information()
    persist_to_text_file(data, results_path, filename)


