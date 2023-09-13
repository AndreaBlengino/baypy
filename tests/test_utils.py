from baypy.utils import flatten_matrix, matrices_to_frame, dot_product
import numpy as np
import pandas as pd
from pytest import mark, raises


@mark.utils
class TestFlattenMatrix:


    @mark.genuine
    def test_function(self):
        matrix = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        flat_matrix = flatten_matrix(matrix = matrix)

        assert isinstance(flat_matrix, np.ndarray)
        assert flat_matrix.size == np.prod(matrix.shape)
        assert flat_matrix.shape == (np.prod(matrix.shape), )


    @mark.error
    def test_raises_type_error(self, utils_flatten_matrix_type_error):
        with raises(TypeError):
            flatten_matrix(matrix = utils_flatten_matrix_type_error)


    @mark.error
    def test_raises_value_error(self):
        with raises(ValueError):
            flatten_matrix(matrix = np.array([]))


@mark.utils
class TestMatricesToFrame:


    @mark.genuine
    def test_function(self):
        matrix_1 = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        matrix_2 = np.array([[0, -1], [-2, -3], [-4, -5], [-6, -7], [-8, -9]])
        matrices_dict = {'a': matrix_1, 'b': matrix_2}
        frame = matrices_to_frame(matrices_dict = matrices_dict)

        assert isinstance(frame, pd.DataFrame)
        assert all(frame.columns == list(matrices_dict.keys()))
        assert not frame.empty
        assert len(frame) == np.prod(matrices_dict['a'].shape)


    @mark.error
    def test_raises_type_error(self, utils_matrices_to_frame_type_error):
        with raises(TypeError):
            matrices_to_frame(matrices_dict = utils_matrices_to_frame_type_error)


    @mark.error
    def test_raises_value_error(self):
        with raises(ValueError):
            matrices_to_frame(matrices_dict = {'a': np.array([])})


@mark.utils
class TestDotProduct:


    @mark.genuine
    def test_function(self):
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        regressors = {'a': 1, 'b': 2}
        product = dot_product(data = data, regressors = regressors)

        assert isinstance(product, np.ndarray)
        assert len(product) == len(data)


    @mark.error
    def test_raises_type_error(self, utils_dot_product_type_error):
        with raises(TypeError):
            dot_product(data = utils_dot_product_type_error['data'],
                        regressors = utils_dot_product_type_error['regressors'])


    @mark.error
    def test_raises_key_error(self):
        with raises(KeyError):
            dot_product(data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
                        regressors = {'a': 1, 'b': 2, 'c': 3})


    @mark.error
    def test_raises_value_error(self, utils_dot_product_value_error):
        with raises(ValueError):
            dot_product(data = utils_dot_product_value_error['data'],
                        regressors = utils_dot_product_value_error['regressors'])
