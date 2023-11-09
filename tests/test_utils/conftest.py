import baypy as bp
from baypy.regression import LinearRegression
from hypothesis.strategies import composite, integers, floats, lists
import numpy as np
import pandas as pd
from pytest import fixture
from tests.conftest import types_to_check


@fixture(params = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, np.ndarray)])
def utils_flatten_matrix_type_error(request):
    return request.param


utils_matrices_to_frame_type_error_1 = [type_to_check for type_to_check in types_to_check
                                        if not isinstance(type_to_check, dict)]

utils_matrices_to_frame_type_error_2 = [{'a': type_to_check} for type_to_check in types_to_check
                                        if not isinstance(type_to_check, np.ndarray)]

@fixture(params = [*utils_matrices_to_frame_type_error_1,
                   *utils_matrices_to_frame_type_error_2])
def utils_matrices_to_frame_type_error(request):
    return request.param


utils_dot_product_type_error_1 = [{'data': type_to_check, 'regressors': {}}
                                  for type_to_check in types_to_check if not isinstance(type_to_check, pd.DataFrame)]

utils_dot_product_type_error_2 = [{'data': pd.DataFrame(columns = ['a', 'b'], index = [0]), 'regressors': type_to_check}
                                  for type_to_check in types_to_check if not isinstance(type_to_check, dict)]

utils_dot_product_type_error_3 = [{'data': pd.DataFrame(columns = ['a', 'b'], index = [0]),
                                   'regressors': {'a': type_to_check}} for type_to_check in types_to_check
                                  if not isinstance(type_to_check, int) and not isinstance(type_to_check, float)]

@fixture(params = [*utils_dot_product_type_error_1,
                   *utils_dot_product_type_error_2,
                   *utils_dot_product_type_error_3])
def utils_dot_product_type_error(request):
    return request.param


@fixture(params = [{'data': pd.DataFrame(), 'regressors': {'a': 1, 'b': 2}},
                   {'data': pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), 'regressors': {}}])
def utils_dot_product_value_error(request):
    return request.param
