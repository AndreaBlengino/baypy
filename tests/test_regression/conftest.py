import baypy as bp
from baypy.regression import LinearRegression
from hypothesis.strategies import composite, integers, floats, lists
import numpy as np
import pandas as pd
from pytest import fixture
from tests.conftest import types_to_check, model


linear_regression_sample_type_error_1 = [{'model': type_to_check, 'n_iterations': 100, 'burn_in_iterations': 50,
                                          'n_chains': 3, 'seed': 137} for type_to_check in types_to_check
                                         if not isinstance(type_to_check, bp.model.Model)]

linear_regression_sample_type_error_2 = [{'model': model, 'n_iterations': type_to_check,
                                          'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137}
                                         for type_to_check in types_to_check if not isinstance(type_to_check, int)]

linear_regression_sample_type_error_3 = [{'model': model, 'n_iterations': 100,
                                          'burn_in_iterations': type_to_check, 'n_chains': 3, 'seed': 137}
                                         for type_to_check in types_to_check if not isinstance(type_to_check, int)]

linear_regression_sample_type_error_4 = [{'model': model, 'n_iterations': 100,
                                          'burn_in_iterations': 50, 'n_chains': type_to_check, 'seed': 137}
                                         for type_to_check in types_to_check if not isinstance(type_to_check, int)]

linear_regression_sample_type_error_5 = [{'model': model, 'n_iterations': 100, 'burn_in_iterations': 50,
                                          'n_chains': 3, 'seed': type_to_check} for type_to_check in types_to_check
                                         if not isinstance(type_to_check, int) and type_to_check is not None]

@fixture(params = [*linear_regression_sample_type_error_1,
                   *linear_regression_sample_type_error_2,
                   *linear_regression_sample_type_error_3,
                   *linear_regression_sample_type_error_4,
                   *linear_regression_sample_type_error_5])
def linear_regression_sample_type_error(request):
    return request.param


linear_regression_sample_value_error_1 = bp.model.LinearModel()
linear_regression_sample_value_error_1.priors = {'x': {'mean': 0,
                                                       'variance': 1},
                                                 'intercept': {'mean': 0,
                                                               'variance': 1},
                                                 'variance': {'shape': 1,
                                                              'scale': 1}}

linear_regression_sample_value_error_2 = bp.model.LinearModel()
linear_regression_sample_value_error_2.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
linear_regression_sample_value_error_2.priors = {'x': {'mean': 0,
                                                       'variance': 1},
                                                 'y': {'mean': 0,
                                                       'variance': 1},
                                                 'intercept': {'mean': 0,
                                                               'variance': 1},
                                                 'variance': {'shape': 1,
                                                              'scale': 1}}

linear_regression_sample_value_error_3 = bp.model.LinearModel()
linear_regression_sample_value_error_3.data = pd.DataFrame(columns = ['x', 'z'], index = [0])
linear_regression_sample_value_error_3.response_variable = 'z'

linear_regression_sample_value_error_4 = bp.model.LinearModel()
linear_regression_sample_value_error_4.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
linear_regression_sample_value_error_4.response_variable = 'w'
linear_regression_sample_value_error_4.priors = {'x': {'mean': 0,
                                                       'variance': 1},
                                                 'y': {'mean': 0,
                                                       'variance': 1},
                                                 'intercept': {'mean': 0,
                                                               'variance': 1},
                                                 'variance': {'shape': 1,
                                                              'scale': 1}}

linear_regression_sample_value_error_5 = bp.model.LinearModel()
linear_regression_sample_value_error_5.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
linear_regression_sample_value_error_5.response_variable = 'z'
linear_regression_sample_value_error_5.priors = {'x': {'mean': 0,
                                                       'variance': 1},
                                                 'y': {'mean': 0,
                                                       'variance': 1},
                                                 'w': {'mean': 0,
                                                       'variance': 1},
                                                 'intercept': {'mean': 0,
                                                               'variance': 1},
                                                 'variance': {'shape': 1,
                                                              'scale': 1}}

@fixture(params = [{'model': linear_regression_sample_value_error_1, 'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'model': linear_regression_sample_value_error_2, 'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'model': linear_regression_sample_value_error_3, 'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'model': linear_regression_sample_value_error_4, 'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'model': linear_regression_sample_value_error_5, 'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'model': model, 'n_iterations': -1, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'model': model, 'n_iterations': 1000, 'burn_in_iterations': -1, 'n_chains': 3, 'seed': 137},
                   {'model': model, 'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': -1, 'seed': 137},
                   {'model': model, 'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': -1},
                   {'model': model, 'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 2**32}])
def linear_regression_sample_value_error(request):
    return request.param
