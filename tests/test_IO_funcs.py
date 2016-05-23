from unittest import TestCase
import io_funcs as io
import os
import shutil
import benchmark_funcs as bf
import matplotlib.pyplot as plt


class TestIOFuncs(TestCase):
    """Tests for the functions in the module io_funcs.py"""

    def test_persist_to_text_file(self):
        test_data = 'This is a test'
        io.persist_to_text_file(test_data, 'my_test_file/', 'foo')
        result = open('my_test_file/foo', 'r')
        shutil.rmtree('my_test_file')
        self.assertEquals(test_data, result.read())

    def test_save_and_load_results(self):
        test_dict = {"foo":[1,2,3], "bar": [42], "baz": []}
        io.save_results_to_pkl("test_files/", test_dict, "my_test_file")
        result = io.load_results_from_pkl("test_files/", "my_test_file")
        shutil.rmtree('test_files')
        self.assertEquals(test_dict, result)

    def test_persist_test_related_info(self):
        io.persist_test_related_info("test_files/", "my_test_file")
        data_from_file = open("test_files/my_test_file", 'r')
        shutil.rmtree('test_files')
        expected_result = bf.get_test_related_information()
        self.assertEquals(data_from_file.read(), expected_result)

    def test_save_plot_to_file(self):
        fig = plt.figure(figsize=(11, 10))
        path = "test_files/"
        filename = "my_test_file"
        io.save_plot_to_file(path, filename, fig)
        self.assertTrue(os.path.isfile("{0}{1}{2}" .format(path, filename, ".png")))
        shutil.rmtree('test_files')
