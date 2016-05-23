from unittest import TestCase
import matplotlib.pyplot as plt
import plotting_funcs as pf
from data_containers.plot_data import PlotData


class TestPlottingFuncs(TestCase):
    """Tests for the functions in the module plotting_funcs.py"""

    data_object = PlotData("title", {"csc": "CSC", "csr": "CSR"}, {0.1: {'csc': [(0.11, 0.22), (0.12, 0.23)]},
                                                                0.2: {'csc': [(0.12, 0.23), (0.15, 0.45)]}},
                               [100, 200], "foo", "bar")

    data_object_2 = PlotData("title", {"csc": "CSC"}, {'100x100': {'csc': [(0.11, 0.22), (0.12, 0.23)]}},
                           [100], "foo", "bar")

    def test_configure_plot(self):
        fig = plt.figure(figsize=(11, 10))
        line_width = 1
        markers = ['D', 'p', 's', '*', 'v', '.', 'H']
        x_axis_data = [100, 200]
        result = pf.configure_plot(self.data_object, x_axis_data, padding=3)
        self.assertEquals(line_width, result[1])
        self.assertEquals(markers, result[2])

    def test_get_timings_from_results(self):
        sparse_function = 'csc'
        sparsity = 0.2
        expected = [0.12, 0.15]
        self.assertEquals(expected, pf.get_timings_from_results(self.data_object, sparse_function, sparsity))

    def test_get_errors_from_results(self):
        sparse_function = 'csc'
        sparsity = 0.2
        expected = [0.23, 0.45]
        self.assertEquals(expected, pf.get_errors_from_results(self.data_object, sparse_function, sparsity))

    def test_get_label_from_data_object(self):
        sparse_function = 'csc'
        expected = 'CSC'
        self.assertEquals(expected, pf.get_label_from_data_object(self.data_object, sparse_function))

    def test_get_errors_from_results_scipy_vs_numpy(self):
        actual = pf.get_errors_from_results_scipy_vs_numpy(self.data_object_2, '100x100', 'csc')
        expected = [0.22, 0.23]
        self.assertEquals(expected, actual)

    def test_get_timing_from_results_scipy_vs_numpy(self):
        actual = pf.get_timing_from_results_scipy_vs_numpy(self.data_object_2, '100x100', 'csc')
        expected = [0.11, 0.12]
        self.assertEquals(expected, actual)