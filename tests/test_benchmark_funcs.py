from unittest import TestCase
import benchmark_funcs as bf
import mock
import numpy as np
from time import sleep

TWO_SEC = 2

RANDOM_SIZE = 100

EXPECTED_STD = 0.0

EXPECTED_MEAN = 2.0

REPEATS = 3


class TestBenchmarkFuncs(TestCase):
    """ Tests for the functions in the module benchmark_funcs.py"""

    def test_test_performance(self):
        bf.measure_time = mock.Mock(return_value=2.00)
        self.assertEquals((EXPECTED_MEAN, EXPECTED_STD),
                          bf.test_performance(self.dummy_func, REPEATS, np.zeros(RANDOM_SIZE), np.zeros(RANDOM_SIZE)))

    def test_test_performance_dot_sparse(self):
        bf.measure_time_dot_sparse= mock.Mock(return_value=2.00)
        self.assertEquals((EXPECTED_MEAN, EXPECTED_STD), bf.test_performance_dot_sparse(REPEATS, np.zeros(RANDOM_SIZE),
                                                                                        np.zeros(RANDOM_SIZE)))

    def test_measure_time(self):
        a = np.zeros(4)
        b = np.zeros(4)
        time = bf.measure_time(self.wait_2_sec,a, b)
        self.assertGreater(time, TWO_SEC)

    def wait_2_sec(self, a, b):
        sleep(2)


    def dummy_func(self):
        pass