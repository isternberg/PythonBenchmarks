from unittest import TestCase
import matrix_funcs as mf
import nose
import numpy as np


class TestMatrixFuncs(TestCase):
    """Tests for the functions in the module matrix_funcs.py"""

    def test_create_matrix_all_zeros(self):
        zeros_matrix = mf.create_matrix(10, 10, 1)
        if zeros_matrix.any() != 0:
            self.fail()
        self.assertTrue(True)

    def test_create_matrix_all_ones(self):
        ones_matrix = mf.create_matrix(10, 10, 0)
        if ones_matrix.any() != 1:
            self.fail()
        self.assertTrue(True)

    def test_create_matrix_half_zeros(self):
        matrix = mf.create_matrix(10, 10, 0.5)
        zeros_count = 0
        for row in matrix:
            for i in row:
                if i == 0:
                    zeros_count+=1
        self.assertEquals(50, zeros_count)

    @nose.tools.raises(ValueError)
    def test_create_matrix_invalid_1st_dimension(self):
        matrix = mf.create_matrix(0, 10, 0.5)

    @nose.tools.raises(ValueError)
    def test_create_matrix_invalid_2nd_dimension(self):
        matrix = mf.create_matrix(10, 0, 0.5)

    @nose.tools.raises(ValueError)
    def test_create_matrix_invalid_percent_zeros_too_low(self):
        matrix = mf.create_matrix(10, 10, -0.1)

    @nose.tools.raises(ValueError)
    def test_create_matrix_invalid_percent_zeros_too_high(self):
        matrix = mf.create_matrix(10, 10, 1.1)


    def test_dot_numpy(self):
        M1 = mf.create_matrix(100, 100)
        M2 = mf.create_matrix(100, 100)
        expected = np.dot(M1, M2)
        np.testing.assert_array_equal(expected, mf.dot_numpy(M1,M2))

    def test_dot_scipy_funcs_with_conversion(self):
        M1 = mf.create_matrix(100, 100)
        expected = np.dot(M1, M1)
        np.testing.assert_array_equal(expected, mf.dot_scipy_csc_with_conversion(M1, M1))
        np.testing.assert_array_equal(expected, mf.dot_scipy_csr_with_conversion(M1, M1))
        np.testing.assert_array_equal(expected, mf.dot_scipy_bsr_with_conversion(M1, M1))

    def test_swap_funcs(self):
        M1 = mf.create_matrix(100, 100)
        M2 = mf.create_matrix(100, 100)
        expected = np.dot(M1, M2)
        np.testing.assert_array_equal(expected, mf.scipy_csc_dot_numpy_with_swap(M1, M2))
        np.testing.assert_array_equal(expected, mf.scipy_csr_dot_numpy_with_swap(M1, M2))
        np.testing.assert_array_equal(expected, mf.scipy_bsr_dot_numpy_with_swap(M1, M2))

    def test_scipy_dot_numpy_funcs(self):
        M1 = np.array([[2, 4, 6, 8], [2, 3, 4, 5], [9, 8 ,7, 6], [3, 7, 5, 9]])
        M2 = np.copy(M1)
        M1_top_2 = np.array([[0, 0, 6, 8], [0, 0, 4, 5], [9, 8, 0, 0], [0, 7, 0, 9]])
        expected = np.dot(M1_top_2, M2)
        np.testing.assert_array_equal(expected, mf.scipy_csc_dot_numpy_with_top_n(M1, M2, 2))
        np.testing.assert_array_equal(expected, mf.scipy_csr_dot_numpy_with_top_n(M1, M2, 2))
        np.testing.assert_array_equal(expected, mf.scipy_bsr_dot_numpy_with_top_n(M1, M2, 2))