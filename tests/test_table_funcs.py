import table_funcs as tf
import io_funcs as io
from prettytable import PrettyTable
import shutil
from unittest import TestCase


class TestTableFuncs(TestCase):
    """Tests for the functions in the module table_funcs.py"""

    def test_table_for_scipy_vs_numpy(self):
        expected = self.create_expected_table_scipy_vs_numpy()
        actual= self.create_actual_table_scipy_vs_numpy()
        self.assertEquals(expected, actual)

    def test_table_for_dense_dot_sparse(self):
        expected = self.create_expected_table_dense_dot_sparse()
        actual =  self.create_actual_table_dense_dot_sparse()
        self.assertEquals(expected, actual)

    def test_table_for_sparsity_benchmark(self):
        expected = self.create_expected_table_sparsity_benchmark()
        actual =  self.create_actual_table_for_sparsity_benchmark()
        io.persist_to_text_file(expected,"test_files/", "table_test_file")
        io.persist_to_text_file(actual,"test_files/", "table_test_file")
        self.assertEquals(expected, actual)
        shutil.rmtree('test_files')

    def create_actual_table_for_sparsity_benchmark(self):
        labels = {'bsr_matrix': 'Block Sparse Row'}
        items_pro_dim = [100]
        funcs_ranked_by_time = [(0.1, 'bsr_matrix')]
        results = {0.1: {'bsr_matrix': [(0.1, 0.001), (0.4, 0.004)]}}
        timings = {'bsr_matrix': [0.1, 0.4]}
        test_data = tf.TableData(labels, items_pro_dim, funcs_ranked_by_time, results, timings)
        return tf.create_summery_tables_for_sparse_matrices_benchmark(test_data, 0.1)

    def create_expected_table_sparsity_benchmark(self):
        table_1 = PrettyTable(
            ["n=100  ", "Testobjekt", "Zeit in Sek.", "Standardabweichung", "rel. Performancegewinn"])
        table_1.align["Testobjekt"] = "l"
        table_1.add_row(["", "Block Sparse Row", 0.1, 0.001, 1.0])
        return table_1.get_string() + "\n"


    def create_actual_table_dense_dot_sparse(self):
        labels = {'dot_numpy': 'numpy (Referenz)'}
        items_pro_dim = [100]
        funcs_ranked_by_time = [(0.1, 'dot_numpy')]
        results = {'dot_numpy': [(0.1, 0.001)]}
        timings = {'dot_numpy': [0.1]}
        test_data = tf.TableData(labels, items_pro_dim, funcs_ranked_by_time, results, timings)
        return tf.create_summery_tables_dense_dot_sparse_benchmark(test_data)


    def create_expected_table_dense_dot_sparse(self):
        table_1 = PrettyTable(
            ["n=100  ", "Testobjekt", "Zeit in Sek.", "Standardabweichung", "rel. Performancegewinn"])
        table_1.align["Testobjekt"] = "l"
        table_1.add_row(["", "numpy (Referenz)", 0.1, 0.001, 1.0])
        return table_1.get_string() + "\n"

    def create_actual_table_scipy_vs_numpy(self):
        labels = {'dot_numpy': 'numpy (Referenz)'}
        items_pro_dim = [100, 120]
        funcs_ranked_by_time = [(0.1, 'dot_numpy')]
        results = {'120x120': {'dot_numpy': [(0.1, 0.001), (0.2, 0.002)]},
                   '100x100': {'dot_numpy': [(0.3, 0.003), (0.4, 0.004)]}}
        timings = {'dot_numpy': [0.1, 0.2]}
        matrix_size_string = '120x120'
        sparsities = [0.45]
        test_data = tf.ExpendedTableData(labels, items_pro_dim, funcs_ranked_by_time, results, timings,
                                         matrix_size_string, sparsities)
        return tf.create_summery_tables_for_scipy_vs_numpy_benchmark(test_data)

    def create_expected_table_scipy_vs_numpy(self):
        table_1 = PrettyTable(
            ["Nullenanteil", "Testobjekt", "Zeit in Sek.", "Standardabweichung", "rel. Performancegewinn"])
        table_1.align["Testobjekt"] = "l"
        table_1.add_row(["45%", "numpy (Referenz)", 0.1, 0.001, 1.0])
        return table_1.get_string() + "\n"




