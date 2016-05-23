from unittest import TestCase
import nose
from top_n_funcs import *

TOP_2_ITEMS = 2

ZERO_ELEMENTS = 0

INDEX_OUT_OF_BOUNDS = 42

NEGATIVE_ITEMS_COUNT = -1


class TopNTest(TestCase):
    """Tests for the module top_n.py"""

    @nose.tools.raises(ValueError)
    def test_get_indices_of_top_n_items_with_negative_n(self):
        arr = np.array([2, 4, 6])
        get_indices_of_top_n_items(arr, NEGATIVE_ITEMS_COUNT)

    @nose.tools.raises(ValueError)
    def test_get_indices_of_top_n_items_with_n_out_of_range(self):
        arr = np.array([2, 4, 6])
        get_indices_of_top_n_items(arr, INDEX_OUT_OF_BOUNDS)

    def test_get_indices_of_top_n_items_with_n_equals_all_returns_indices_of_entire_array(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        top_n_indices = get_indices_of_top_n_items(arr, len(arr))
        self.assertEquals(len(top_n_indices), len(arr))
        self.assertEqual(set(top_n_indices), set([i for i in range(len(arr))]))

    def test_get_indices_of_top_n_items_with_n_equals_all_returns_empty_array(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        top_n_indices = get_indices_of_top_n_items(arr, ZERO_ELEMENTS)
        np.testing.assert_array_equal(np.empty, top_n_indices)

    def test_get_indices_of_top_n_items_with_n_equals_2(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        top_n_indices = get_indices_of_top_n_items(arr, TOP_2_ITEMS)
        self.assertEquals(set([2, 5]), set(top_n_indices))

    @nose.tools.raises(IndexError)
    def test_convert_to_sparse_vector_with_selected_indices_with_invalid_indices(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        indices = np.array([INDEX_OUT_OF_BOUNDS])
        convert_to_sparse_vector_with_selected_indices(arr, indices)

    @nose.tools.raises(IndexError)
    def test_convert_to_sparse_vector_with_selected_indices_with_negative_indices(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        indices = np.array([-1,-2,4])
        convert_to_sparse_vector_with_selected_indices(arr, indices)

    def test_convert_to_sparse_vector_with_selected_indices_with_no_indices_selected(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        indices = np.array([])
        convert_to_sparse_vector_with_selected_indices(arr, indices)
        np.testing.assert_array_equal(np.zeros(len(arr)), arr)

    def test_convert_to_sparse_vector_with_selected_indices_positive_test(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        indices = np.array([2, 5])
        convert_to_sparse_vector_with_selected_indices(arr, indices)
        np.testing.assert_array_equal(np.array([0, 0, 6, 0, 0, 5, 0]), arr)

    def test_convert_to_sparse_vector_with_top_n_positive_test(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        convert_to_sparse_vector_with_top_n(arr, 3)
        np.testing.assert_array_equal(np.array([0, 4, 6, 0, 0, 5, 0]), arr)

    @nose.tools.raises(ValueError)
    def test_convert_to_sparse_vector_with_top_n_with_n_larger_than_length(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        convert_to_sparse_vector_with_top_n(arr, len(arr) + 1)

    @nose.tools.raises(ValueError)
    def test_convert_to_sparse_vector_with_top_n_with_n_negative(self):
        arr = np.array([2, 4, 6, 1, 3, 5, -1])
        convert_to_sparse_vector_with_top_n(arr, NEGATIVE_ITEMS_COUNT)

    def test_convert_matrix_to_sparse_with_top_n_positive_test(self):
        arr = np.array([[2, 3, 4, 5],[-1,-2,-3,-4],[0, 1, -1, 0]])
        expected = np.array([[0, 0, 4, 5],[-1, -2, 0, 0],[0, 1, 0, 0]])
        convert_matrix_to_sparse_with_top_n(arr, 2)
        np.testing.assert_array_equal(expected, arr)

    @nose.tools.raises(ValueError)
    def test_convert_matrix_to_sparse_with_top_n_bad_input(self):
        arr = [1, 2, 3]
        convert_matrix_to_sparse_with_top_n(arr, 2)
