import numpy as np
from pytest import fixture
from tests.conftest import types_to_check
from typing import Iterable


diagnostics_autocorrelation_plot_type_error_1 = [
    {
        'posteriors': type_to_check,
        'max_lags': 30
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, dict)
]

diagnostics_autocorrelation_plot_type_error_2 = [
    {
        'posteriors': {'intercept': type_to_check},
        'max_lags': 30
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, np.ndarray)
]

diagnostics_autocorrelation_plot_type_error_3 = [
    {
        'posteriors': {'intercept': np.array([0])},
        'max_lags': type_to_check
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, int)
]


@fixture(
    params=[
        *diagnostics_autocorrelation_plot_type_error_1,
        *diagnostics_autocorrelation_plot_type_error_2,
        *diagnostics_autocorrelation_plot_type_error_3
    ]
)
def diagnostics_autocorrelation_plot_type_error(request):
    return request.param


@fixture(
    params=[
        {
            'posteriors': {'intercept': np.array([])},
            'max_lags': 30
        },
        {
            'posteriors': {'intercept': np.array([0])},
            'max_lags': -1
        }
    ]
)
def diagnostics_autocorrelation_plot_value_error(request):
    return request.param


diagnostics_autocorrelation_summary_type_error_1 = [
    {
        'posteriors': type_to_check,
        'lags': [0, 1],
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, dict)
]

diagnostics_autocorrelation_summary_type_error_2 = [
    {
        'posteriors': {'intercept': type_to_check},
        'lags': [0, 1],
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, np.ndarray)
]

diagnostics_autocorrelation_summary_type_error_3 = [
    {
        'posteriors': {'intercept': np.array([0])},
        'lags': type_to_check,
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, Iterable) and type_to_check is not None
]

diagnostics_autocorrelation_summary_type_error_4 = [
    {
        'posteriors': {'intercept': np.array([0])},
        'lags': [type_to_check, type_to_check],
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, int)
]

diagnostics_autocorrelation_summary_type_error_5 = [
    {
        'posteriors': {'intercept': np.array([0])},
        'lags': [0, 1],
        'print_summary': type_to_check
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, bool)
]


@fixture(
    params=[
        *diagnostics_autocorrelation_summary_type_error_1,
        *diagnostics_autocorrelation_summary_type_error_2,
        *diagnostics_autocorrelation_summary_type_error_3,
        *diagnostics_autocorrelation_summary_type_error_4,
        *diagnostics_autocorrelation_summary_type_error_5
    ]
)
def diagnostics_autocorrelation_summary_type_error(request):
    return request.param


@fixture(
    params=[
        {
            'posteriors': {'intercept': np.array([])},
            'lags': [0, 1, 5, 10, 30]
        },
        {
            'posteriors': {'intercept': np.array([0])},
            'lags': []
        },
        {
            'posteriors': {'intercept': np.array([0])},
            'lags': [-1]
        }
    ]
)
def diagnostics_autocorrelation_summary_value_error(request):
    return request.param


diagnostics_effective_sample_size_type_error_1 = [
    {
        'posteriors': type_to_check,
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, dict)
]

diagnostics_effective_sample_size_type_error_2 = [
    {
        'posteriors': {'intercept': type_to_check},
        'print_summary': False
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, np.ndarray)
]

diagnostics_effective_sample_size_type_error_3 = [
    {
        'posteriors': {'intercept': np.array([0])},
        'print_summary': type_to_check
    } for type_to_check in types_to_check
    if not isinstance(type_to_check, bool)
]


@fixture(
    params=[
        *diagnostics_effective_sample_size_type_error_1,
        *diagnostics_effective_sample_size_type_error_2,
        *diagnostics_effective_sample_size_type_error_3
    ]
)
def diagnostics_effective_sample_size_type_error(request):
    return request.param
