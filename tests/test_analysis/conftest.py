import baypy as bp
import numpy as np
import pandas as pd
from pytest import fixture
from tests.conftest import types_to_check


analysis_trace_plot_type_error_1 = [
    type_to_check for type_to_check in types_to_check
    if not isinstance(type_to_check, dict)
]

analysis_trace_plot_type_error_2 = [
    {'intercept': type_to_check}
    for type_to_check in types_to_check
    if not isinstance(type_to_check, np.ndarray)
]


@fixture(
    params=[
        *analysis_trace_plot_type_error_1,
        *analysis_trace_plot_type_error_2
    ]
)
def analysis_trace_plot_type_error(request):
    return request.param


analysis_summary_type_error_1 = [
    {
        'posteriors': type_to_check,
        'alpha': 0.05,
        'quantiles': [0.1, 0.9],
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, dict)
]

analysis_summary_type_error_2 = [
    {
        'posteriors': {'intercept': type_to_check},
        'alpha': 0.05,
        'quantiles': [0.1, 0.9],
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, np.ndarray)
]

analysis_summary_type_error_3 = [
    {
        'posteriors': {'intercept': np.array([0])},
        'alpha': type_to_check,
        'quantiles': [0.1, 0.9],
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, float | bool)
]

analysis_summary_type_error_4 = [
    {
        'posteriors': {'intercept': np.array([0])},
        'alpha': 0.05,
        'quantiles': type_to_check,
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, list) and type_to_check is not None
]

analysis_summary_type_error_5 = [
    {
        'posteriors': {'intercept': np.array([0])},
        'alpha': 0.05,
        'quantiles': [type_to_check, type_to_check],
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, float)
]

analysis_summary_type_error_6 = [
    {
        'posteriors': {'intercept': np.array([0])},
        'alpha': 0.05,
        'quantiles': [0.1, 0.9],
        'print_summary': type_to_check
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, bool)
]


@fixture(
    params=[
        *analysis_summary_type_error_1,
        *analysis_summary_type_error_2,
        *analysis_summary_type_error_3,
        *analysis_summary_type_error_4,
        *analysis_summary_type_error_5,
        *analysis_summary_type_error_6
    ]
)
def analysis_summary_type_error(request):
    return request.param


@fixture(
    params=[
        {
            'posteriors': {'intercept': np.array([])},
            'alpha': 0.05,
            'quantiles': [0.1, 0.9]
        },
        {
            'posteriors': {'intercept': np.array([0])},
            'alpha': -0.5,
            'quantiles': [0.1, 0.9]
        },
        {
            'posteriors': {'intercept': np.array([0])},
            'alpha': 1.5,
            'quantiles': [0.1, 0.9]
        },
        {
            'posteriors': {'intercept': np.array([0])},
            'alpha': 0.05,
            'quantiles': []
        },
        {
            'posteriors': {'intercept': np.array([0])},
            'alpha': 0.05,
            'quantiles': [-0.5]
        },
        {
            'posteriors': {'intercept': np.array([0])},
            'alpha': 0.05,
            'quantiles': [1.5]
        }
    ]
)
def analysis_summary_value_error(request):
    return request.param


@fixture(
    params=[
        type_to_check for type_to_check in types_to_check
        if not isinstance(type_to_check, bp.model.Model)
    ]
)
def analysis_residuals_plot_type_error(request):
    return request.param


analysis_residuals_plot_value_error_1 = bp.model.LinearModel()
analysis_residuals_plot_value_error_1.data = pd.DataFrame(
    columns=['response_variable'],
    index=[0]
)
analysis_residuals_plot_value_error_1.response_variable = 'response_variable'

analysis_residuals_plot_value_error_2 = bp.model.LinearModel()
analysis_residuals_plot_value_error_2.data = pd.DataFrame(
    columns=['response_variable'],
    index=[0]
)
analysis_residuals_plot_value_error_2.response_variable = 'response_variable'
analysis_residuals_plot_value_error_2.posteriors = {
    'intercept': np.array([0]),
    'variance': np.array([0]),
    'x': np.array([0])
}

analysis_residuals_plot_value_error_3 = bp.model.LinearModel()
analysis_residuals_plot_value_error_3.data = pd.DataFrame(
    columns=['not_response_variable'],
    index=[0]
)
analysis_residuals_plot_value_error_3.data.drop(
    index=[0],
    inplace=True
)
analysis_residuals_plot_value_error_3.response_variable = 'response_variable'
analysis_residuals_plot_value_error_3.posteriors = {
    'intercept': np.array([0]),
    'variance': np.array([0])
}

analysis_residuals_plot_value_error_4 = bp.model.LinearModel()
analysis_residuals_plot_value_error_4.data = pd.DataFrame(
    columns=['not_response_variable'],
    index=[0]
)
analysis_residuals_plot_value_error_4.response_variable = 'response_variable'
analysis_residuals_plot_value_error_4.posteriors = {
    'intercept': np.array([0]),
    'variance': np.array([0])
}


@fixture(
    params=[
        analysis_residuals_plot_value_error_1,
        analysis_residuals_plot_value_error_2,
        analysis_residuals_plot_value_error_3,
        analysis_residuals_plot_value_error_4
    ]
)
def analysis_residuals_plot_value_error(request):
    return request.param


analysis_compute_dic_type_error_1 = [
    {
        'model': type_to_check,
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, bp.model.Model)
]

analysis_compute_dic_type_error_2 = [
    {
        'model': bp.model.LinearModel(),
        'print_summary': type_to_check
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, bool)
]
for args in analysis_compute_dic_type_error_2:
    args['model'].data = pd.DataFrame(
        columns=['response_variable'],
        index=[0]
    )
    args['model'].response_variable = 'response_variable'
    args['model'].posteriors = {
        'intercept': np.array([0]),
        'variance': np.array([0])
    }


@fixture(
    params=[
        *analysis_compute_dic_type_error_1,
        *analysis_compute_dic_type_error_2
    ]
)
def analysis_compute_dic_type_error(request):
    return request.param


analysis_compute_dic_value_error_value_error_1 = bp.model.LinearModel()
analysis_compute_dic_value_error_value_error_1.data = pd.DataFrame(
    columns=['response_variable'],
    index=[0]
)
analysis_compute_dic_value_error_value_error_1.response_variable = \
    'response_variable'

analysis_compute_dic_value_error_value_error_2 = bp.model.LinearModel()
analysis_compute_dic_value_error_value_error_2.data = pd.DataFrame(
    columns=['response_variable'],
    index=[0]
)
analysis_compute_dic_value_error_value_error_2.response_variable = \
    'response_variable'
analysis_compute_dic_value_error_value_error_2.posteriors = {
    'intercept': np.array([0]),
    'variance': np.array([0]),
    'x': np.array([0])
}

analysis_compute_dic_value_error_value_error_3 = bp.model.LinearModel()
analysis_compute_dic_value_error_value_error_3.data = pd.DataFrame(
    columns=['not_response_variable'],
    index=[0]
)
analysis_compute_dic_value_error_value_error_3.data.drop(
    index=[0],
    inplace=True
)
analysis_compute_dic_value_error_value_error_3.response_variable = \
    'response_variable'
analysis_compute_dic_value_error_value_error_3.posteriors = {
    'intercept': np.array([0]),
    'variance': np.array([0])
}

analysis_compute_dic_value_error_value_error_4 = bp.model.LinearModel()
analysis_compute_dic_value_error_value_error_4.data = pd.DataFrame(
    columns=['not_response_variable'],
    index=[0]
)
analysis_compute_dic_value_error_value_error_4.response_variable = \
    'response_variable'
analysis_compute_dic_value_error_value_error_4.posteriors = {
    'intercept': np.array([0]),
    'variance': np.array([0])
}


@fixture(
    params=[
        analysis_compute_dic_value_error_value_error_1,
        analysis_compute_dic_value_error_value_error_2,
        analysis_compute_dic_value_error_value_error_3,
        analysis_compute_dic_value_error_value_error_4
    ]
)
def analysis_compute_dic_value_error(request):
    return request.param
