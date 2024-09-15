from baypy.utils import flatten_matrix, matrices_to_frame, dot_product
from hypothesis import given, settings
from hypothesis.strategies import integers
import numpy as np
import pandas as pd
from pytest import mark, raises


@mark.utils
class TestFlattenMatrix:

    @mark.genuine
    @given(
        n_rows=integers(min_value=1, max_value=10000),
        n_columns=integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, deadline=1000)
    def test_function(self, n_rows, n_columns):
        matrix = np.random.randn(n_rows, n_columns)
        flat_matrix = flatten_matrix(matrix=matrix)

        assert isinstance(flat_matrix, np.ndarray)
        assert flat_matrix.size == n_rows*n_columns
        assert flat_matrix.shape == (n_rows*n_columns, )
        for i in range(n_rows):
            for j in range(n_columns):
                assert matrix[i, j] == flat_matrix[i*n_columns + j]

    @mark.error
    def test_raises_type_error(self, utils_flatten_matrix_type_error):
        with raises(TypeError):
            flatten_matrix(matrix=utils_flatten_matrix_type_error)

    @mark.error
    def test_raises_value_error(self):
        with raises(ValueError):
            flatten_matrix(matrix=np.array([]))


@mark.utils
class TestMatricesToFrame:

    @mark.genuine
    @given(
        n_rows=integers(min_value=1, max_value=1000),
        n_columns=integers(min_value=1, max_value=5),
        n_matrices=integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, deadline=None)
    def test_function(self, n_rows, n_columns, n_matrices):
        matrix_names = np.random.choice(
            list('abcdefghij'),
            n_matrices,
            replace=False
        ).tolist()
        matrices_dict = {
            matrix_name: np.random.randn(n_rows, n_columns)
            for matrix_name in matrix_names
        }
        frame = matrices_to_frame(matrices_dict=matrices_dict)

        assert isinstance(frame, pd.DataFrame)
        assert all(frame.columns == list(matrices_dict.keys()))
        assert not frame.empty
        assert len(frame) == n_rows*n_columns

        for col in matrices_dict.keys():
            for i in range(n_rows):
                for j in range(n_columns):
                    assert matrices_dict[col][i, j] == \
                        frame.loc[i*n_columns + j, col]

    @mark.error
    def test_raises_type_error(self, utils_matrices_to_frame_type_error):
        with raises(TypeError):
            matrices_to_frame(matrices_dict=utils_matrices_to_frame_type_error)

    @mark.error
    def test_raises_value_error(self):
        with raises(ValueError):
            matrices_to_frame(matrices_dict={'a': np.array([])})


@mark.utils
class TestDotProduct:

    @mark.genuine
    @given(
        n_rows=integers(min_value=1, max_value=10000),
        n_columns=integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, deadline=None)
    def test_function(self, n_rows, n_columns):
        column_names = np.random.choice(
            list('abcdefghij'),
            n_columns,
            replace=False
        ).tolist()
        data = pd.DataFrame(
            {
                column_name: np.random.randn(n_rows)
                for column_name in column_names
            }
        )
        regressors = {
            column_name: np.random.randn(1)[0]
            for column_name in column_names
        }
        product = dot_product(data=data, regressors=regressors)

        assert isinstance(product, np.ndarray)
        assert len(product) == n_rows
        for i in range(n_rows):
            s = 0
            for col, regressor in regressors.items():
                s += data.loc[i, col]*regressor
            assert np.abs(product[i] - s) < 1e-14

    @mark.error
    def test_raises_type_error(self, utils_dot_product_type_error):
        with raises(TypeError):
            dot_product(
                data=utils_dot_product_type_error['data'],
                regressors=utils_dot_product_type_error['regressors']
            )

    @mark.error
    def test_raises_key_error(self):
        with raises(KeyError):
            dot_product(
                data=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
                regressors={'a': 1, 'b': 2, 'c': 3}
            )

    @mark.error
    def test_raises_value_error(self, utils_dot_product_value_error):
        with raises(ValueError):
            dot_product(
                data=utils_dot_product_value_error['data'],
                regressors=utils_dot_product_value_error['regressors']
            )
