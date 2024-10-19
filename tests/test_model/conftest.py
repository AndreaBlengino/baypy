import baypy as bp
import numpy as np
import pandas as pd
from pytest import fixture
from tests.conftest import types_to_check, response_variable


@fixture(
    params=[
        type_to_check for type_to_check in types_to_check
        if not isinstance(type_to_check, pd.DataFrame)
    ]
)
def model_data_type_error(request):
    return request.param


@fixture(
    params=[
        type_to_check for type_to_check in types_to_check
        if not isinstance(type_to_check, str)
    ]
)
def model_response_variable_type_error(request):
    return request.param


model_priors_type_error_1 = [
    type_to_check for type_to_check in types_to_check
    if not isinstance(type_to_check, dict)
]
model_priors_type_error_2 = [
    {
        'intercept': type_to_check,
        'variance': type_to_check
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, dict)
]


@fixture(
    params=[
        *model_priors_type_error_1,
        *model_priors_type_error_2
    ]
)
def model_priors_type_error(request):
    return request.param


@fixture(
    params=[
        {
            'regressor': 1,
            'variance': 2
        },
        {
            'intercept': 1,
            'regressor': 2
        },
        {
            'intercept': {'m': 0, 'variance': 1},
            'variance': {'shape': 1, 'scale': 1}
        },
        {
            'intercept': {'mean': 0, 'v': 1},
            'variance': {'shape': 1, 'scale': 1}
        },
        {
            'intercept': {'mean': 0, 'variance': 1},
            'variance': {'s': 1, 'scale': 1}
        },
        {
            'intercept': {'mean': 0, 'variance': 1},
            'variance': {'shape': 1, 's': 1}
        }
    ]
)
def model_priors_key_error(request):
    return request.param


@fixture(
    params=[
        {},
        {
            'intercept': {},
            'variance': {}
        },
        {
            'intercept': {'mean': 0, 'variance': -1e6},
            'variance': {'shape': 1, 'scale': 1e-6}
        },
        {
            'intercept': {'mean': 0, 'variance': 1e6},
            'variance': {'shape': -1, 'scale': 1e-6}
        },
        {
            'intercept': {'mean': 0, 'variance': 1e6},
            'variance': {'shape': 1, 'scale': -1e-6}
        }
    ]
)
def model_priors_value_error(request):
    return request.param


model_posteriors_type_error_1 = [
    type_to_check for type_to_check in types_to_check
    if not isinstance(type_to_check, dict)
]
model_posteriors_type_error_2 = [
    {'intercept': type_to_check} for type_to_check in types_to_check
    if not isinstance(type_to_check, np.ndarray)
]


@fixture(
    params=[
        *model_posteriors_type_error_1,
        *model_posteriors_type_error_2
    ]
)
def model_posteriors_type_error(request):
    return request.param


@fixture(params=[{'intercept': np.array([0])}, {'variance': np.array([0])}])
def model_posteriors_key_error(request):
    return request.param


model_residuals_value_error_1 = bp.model.LinearModel()

model_residuals_value_error_2 = bp.model.LinearModel()
model_residuals_value_error_2.data = pd.DataFrame(
    columns=['not_response_variable'],
    index=[0]
)
model_residuals_value_error_2.response_variable = 'response_variable'

model_residuals_value_error_3 = bp.model.LinearModel()
model_residuals_value_error_3.data = pd.DataFrame(
    columns=['response_variable'],
    index=[0]
)
model_residuals_value_error_3.response_variable = 'response_variable'


@fixture(
    params=[
        model_residuals_value_error_1,
        model_residuals_value_error_2,
        model_residuals_value_error_3
    ]
)
def model_residuals_value_error(request):
    return request.param


@fixture(
    params=[
        type_to_check for type_to_check in types_to_check
        if not isinstance(type_to_check, dict)
    ]
)
def model_predict_distribution_type_error(request):
    return request.param


@fixture(
    params=[
        type_to_check for type_to_check in types_to_check
        if not isinstance(type_to_check, pd.DataFrame)
    ]
)
def model_likelihood_type_error(request):
    return request.param


@fixture(
    params=[
        pd.DataFrame(),
        pd.DataFrame({'mean': [1, 2, 3], 'variance': [1, 2, 3]}),
        pd.DataFrame({response_variable: [1, 2, 3], 'variance': [1, 2, 3]}),
        pd.DataFrame({response_variable: [1, 2, 3], 'mean': [1, 2, 3]})
    ]
)
def model_likelihood_value_error(request):
    return request.param


@fixture(
    params=[
        type_to_check for type_to_check in types_to_check
        if not isinstance(type_to_check, pd.DataFrame)
    ]
)
def model_log_likelihood_type_error(request):
    return request.param


@fixture(
    params=[
        pd.DataFrame(),
        pd.DataFrame({'mean': [1, 2, 3], 'variance': [1, 2, 3]}),
        pd.DataFrame({response_variable: [1, 2, 3], 'variance': [1, 2, 3]}),
        pd.DataFrame({response_variable: [1, 2, 3], 'mean': [1, 2, 3]})
    ]
)
def model_log_likelihood_value_error(request):
    return request.param
