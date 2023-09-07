import baypy as bp
import numpy as np
import pandas as pd
from pytest import fixture


np.random.seed(42)

N = 50
data = pd.DataFrame()
data['x_1'] = np.random.uniform(low = 0, high = 100, size = N)
data['x_2'] = np.random.uniform(low = -10, high = 10, size = N)
data['x_3'] = np.random.uniform(low = -50, high = -40, size = N)
data['x_1 * x_2'] = data['x_1']*data['x_2']

data['y'] = 3*data['x_1'] - 20*data['x_2'] - data['x_3'] - 5*data['x_1 * x_2'] + 13 + 1*np.random.randn(N)

response_variable = 'y'

variance_sample_size = 5
variance_of_variance = 10

priors = {'x_1': {'mean': 0,
                  'variance': 1e6},
          'x_2': {'mean': 0,
                  'variance': 1e6},
          'x_3': {'mean': 0,
                  'variance': 1e6},
          'x_1 * x_2': {'mean': 0,
                        'variance': 1e6},
          'intercept': {'mean': 0,
                        'variance': 1e6},
          'variance': {'shape': variance_sample_size,
                       'scale': variance_sample_size*variance_of_variance}}


regressor_names = ['intercept', 'x_1', 'x_2', 'x_3', 'x_1 * x_2']

q_min = 0.025
q_max = 0.975

predictors = {'x_1': 20, 'x_2': 5, 'x_3': -45}
predictors['x_1 * x_2'] = predictors['x_1']*predictors['x_2']

linear_regression_model_no_data = bp.model.LinearModel()
linear_regression_model_no_data.priors = {'x': {'mean': 0,
                              'variance': 1},
                        'intercept': {'mean': 0,
                                      'variance': 1},
                        'variance': {'shape': 1,
                                     'scale': 1}}

linear_regression_model_no_response_variable = bp.model.LinearModel()
linear_regression_model_no_response_variable.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
linear_regression_model_no_response_variable.priors = {'x': {'mean': 0,
                                           'variance': 1},
                                     'y': {'mean': 0,
                                           'variance': 1},
                                     'intercept': {'mean': 0,
                                                   'variance': 1},
                                     'variance': {'shape': 1,
                                                  'scale': 1}}

linear_regression_model_no_priors = bp.model.LinearModel()
linear_regression_model_no_priors.data = pd.DataFrame(columns = ['x', 'z'], index = [0])
linear_regression_model_no_priors.response_variable = 'z'

linear_regression_model_response_variable_not_in_data = bp.model.LinearModel()
linear_regression_model_response_variable_not_in_data.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
linear_regression_model_response_variable_not_in_data.response_variable = 'w'
linear_regression_model_response_variable_not_in_data.priors = {'x': {'mean': 0,
                                                    'variance': 1},
                                              'y': {'mean': 0,
                                                    'variance': 1},
                                              'intercept': {'mean': 0,
                                                            'variance': 1},
                                              'variance': {'shape': 1,
                                                           'scale': 1}}

linear_regression_model_prior_not_in_data = bp.model.LinearModel()
linear_regression_model_prior_not_in_data.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
linear_regression_model_prior_not_in_data.response_variable = 'z'
linear_regression_model_prior_not_in_data.priors = {'x': {'mean': 0,
                                        'variance': 1},
                                  'y': {'mean': 0,
                                        'variance': 1},
                                  'w': {'mean': 0,
                                        'variance': 1},
                                  'intercept': {'mean': 0,
                                                'variance': 1},
                                  'variance': {'shape': 1,
                                               'scale': 1}}


analysis_value_error_posterior_data_empty = bp.model.LinearModel()
analysis_value_error_posterior_data_empty.data = pd.DataFrame(columns = ['response_variable'], index = [0])
analysis_value_error_posterior_data_empty.response_variable = 'response_variable'
analysis_value_error_posterior_data_empty.posteriors = {'intercept': np.array([])}

analysis_value_error_posterior_not_in_data = bp.model.LinearModel()
analysis_value_error_posterior_not_in_data.data = pd.DataFrame(columns = ['response_variable'], index = [0])
analysis_value_error_posterior_not_in_data.response_variable = 'response_variable'
analysis_value_error_posterior_not_in_data.posteriors = {'intercept': np.array([0]), 'x': np.array([0])}

analysis_value_error_data_empty = bp.model.LinearModel()
analysis_value_error_data_empty.data = pd.DataFrame(columns = ['not_response_variable'], index = [0])
analysis_value_error_data_empty.data.drop(index = [0], inplace = True)
analysis_value_error_data_empty.response_variable = 'response_variable'
analysis_value_error_data_empty.posteriors = {'intercept': np.array([0])}

analysis_value_error_response_variable_not_in_data = bp.model.LinearModel()
analysis_value_error_response_variable_not_in_data.data = pd.DataFrame(columns = ['not_response_variable'], index = [0])
analysis_value_error_response_variable_not_in_data.response_variable = 'response_variable'
analysis_value_error_response_variable_not_in_data.posteriors = {'intercept': np.array([0])}


types_to_check = ['string', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, {0: 1}, None,
                  pd.DataFrame(columns = ['response_variable'], index = [0]), np.ndarray([0])]

analysis_residuals_plot_type_error_1 = [bp.model.LinearModel() for type_to_check in types_to_check if not isinstance(type_to_check, dict)]
for model, type_to_check in zip(analysis_residuals_plot_type_error_1, types_to_check):
    if not isinstance(type_to_check, dict):
        model.data = pd.DataFrame(columns = ['response_variable'], index = [0])
        model.response_variable = 'response_variable'
        model.posteriors = type_to_check

analysis_residuals_plot_type_error_2 = [bp.model.LinearModel() for type_to_check in types_to_check if not isinstance(type_to_check, np.ndarray)]
for model, type_to_check in zip(analysis_residuals_plot_type_error_2, types_to_check):
    if not isinstance(type_to_check, np.ndarray):
        model.data = pd.DataFrame(columns = ['response_variable'], index = [0])
        model.response_variable = 'response_variable'
        model.posteriors = {'intercept': type_to_check}


analysis_compute_dic_type_error_1 = [{'model': type_to_check, 'print_summary': False}
                                     for type_to_check in types_to_check]

analysis_compute_dic_type_error_2 = [{'model': bp.model.LinearModel(),
                                      'print_summary': False} for type_to_check in types_to_check
                                     if not isinstance(type_to_check, dict)]
for args, type_to_check in zip(analysis_compute_dic_type_error_2, types_to_check):
    if not isinstance(type_to_check, dict):
        args['model'].data = pd.DataFrame(columns = ['response_variable'], index = [0])
        args['model'].response_variable = 'response_variable'
        args['model'].posteriors = type_to_check

analysis_compute_dic_type_error_3 = [{'model': bp.model.LinearModel(),
                                      'print_summary': False} for type_to_check in types_to_check
                                     if not isinstance(type_to_check, np.ndarray)]
for args, type_to_check in zip(analysis_compute_dic_type_error_3, types_to_check):
    if not isinstance(type_to_check, np.ndarray):
        args['model'].data = pd.DataFrame(columns = ['response_variable'], index = [0])
        args['model'].response_variable = 'response_variable'
        args['model'].posteriors = {'intercept': type_to_check}

analysis_compute_dic_type_error_4 = [{'model': bp.model.LinearModel(),
                                      'print_summary': type_to_check} for type_to_check in types_to_check
                                     if not isinstance(type_to_check, bool)]
for args in analysis_compute_dic_type_error_4:
    args['model'].data = pd.DataFrame(columns = ['response_variable'], index = [0])
    args['model'].response_variable = 'response_variable'
    args['model'].posteriors = {'intercept': np.ndarray([0])}


@fixture(scope = 'session',
         params = [{'data': data,
                    'response_variable': response_variable,
                    'priors': priors,
                    'n_iterations': 1000,
                    'burn_in_iterations': 50,
                    'n_chains': 3,
                    'seed': 137,
                    'regressor_names': regressor_names,
                    'q_min': q_min,
                    'q_max': q_max,
                    'predictors': predictors},
                   {'data': data,
                    'response_variable': response_variable,
                    'priors': priors,
                    'n_iterations': 100,
                    'burn_in_iterations': 50,
                    'n_chains': 5,
                    'seed': 137,
                    'regressor_names': regressor_names,
                    'q_min': q_min,
                    'q_max': q_max,
                    'predictors': predictors},
                   {'data': data,
                    'response_variable': response_variable,
                    'priors': priors,
                    'n_iterations': 1000,
                    'burn_in_iterations': 50,
                    'n_chains': 1,
                    'seed': 137,
                    'regressor_names': regressor_names,
                    'q_min': q_min,
                    'q_max': q_max,
                    'predictors': predictors},
                   {'data': data,
                    'response_variable': response_variable,
                    'priors': priors,
                    'n_iterations': 1000,
                    'burn_in_iterations': 1,
                    'n_chains': 3,
                    'seed': 137,
                    'regressor_names': regressor_names,
                    'q_min': q_min,
                    'q_max': q_max,
                    'predictors': predictors}])
def general_testing_data(request):
    return request.param


@fixture(scope = 'session')
def empty_model():
    model = bp.model.LinearModel()
    return model


@fixture(params = ['data', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, None])
def model_data_type_error(request):
    return request.param


@fixture(params = [1, 1.1, True, (0, 1), [0, 1], {0, 1}, None])
def model_response_variable_type_error(request):
    return request.param


@fixture(params = ['priors', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, None,
                   {'intercept': 'intercept', 'variance': 'variance'},
                   {'intercept': 1, 'variance': 1},
                   {'intercept': 1.1, 'variance': 1.1},
                   {'intercept': True, 'variance': True},
                   {'intercept': (0, 1), 'variance': (0, 1)},
                   {'intercept': [0, 1], 'variance': [0, 1]},
                   {'intercept': {0, 1}, 'variance': {0, 1}},
                   {'intercept': None, 'variance': None}])
def model_priors_type_error(request):
    return request.param


@fixture(params = [{'regressor': 1, 'variance': 2},
                   {'intercept': 1, 'regressor': 2},
                   {'intercept': {'m': 0, 'variance': 1}, 'variance': {'shape': 1, 'scale': 1}},
                   {'intercept': {'mean': 0, 'v': 1}, 'variance': {'shape': 1, 'scale': 1}},
                   {'intercept': {'mean': 0, 'variance': 1}, 'variance': {'s': 1, 'scale': 1}},
                   {'intercept': {'mean': 0, 'variance': 1}, 'variance': {'shape': 1, 's': 1}}])
def model_priors_key_error(request):
    return request.param


@fixture(params = [{},
                   {'intercept': {}, 'variance': {}},
                   {'intercept': {'mean': 0, 'variance': -1e6}, 'variance': {'shape': 1, 'scale': 1e-6}},
                   {'intercept': {'mean': 0, 'variance': 1e6}, 'variance': {'shape': -1, 'scale': 1e-6}},
                   {'intercept': {'mean': 0, 'variance': 1e6}, 'variance': {'shape': 1, 'scale': -1e-6}}])
def model_priors_value_error(request):
    return request.param


@fixture(scope = 'session')
def model(general_testing_data):
    complete_model = bp.model.LinearModel()
    complete_model.data = general_testing_data['data']
    complete_model.response_variable = general_testing_data['response_variable']
    complete_model.priors = general_testing_data['priors']
    return complete_model


@fixture(params = ['model', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, {'model': 1}, None])
def linear_regression_init_type_error(request):
    return request.param


@fixture(params = [linear_regression_model_no_data,
                   linear_regression_model_no_response_variable,
                   linear_regression_model_no_priors,
                   linear_regression_model_response_variable_not_in_data,
                   linear_regression_model_prior_not_in_data])
def linear_regression_init_value_error(request):
    return request.param


@fixture(params = [{'n_iterations': '100', 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100.0, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': (100, 200), 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': [100, 200], 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': {100, 200}, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': {'100': 100}, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': None, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': '50', 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50.0, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': (50, 100), 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': [50, 100], 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': {50, 100}, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': {'50': 50}, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': None, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': '3', 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3.0, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': (3, 6), 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': [4, 6], 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': {3, 6}, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': {'3': 3}, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': None, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': '137'},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137.0},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': (137, 137)},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': [137, 137]},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': {137, 137}},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': {'137': 137}}])
def linear_regression_sample_type_error(request):
    return request.param


@fixture(params = [{'n_iterations': -1, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 1000, 'burn_in_iterations': -1, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': -1, 'seed': 137},
                   {'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': -1},
                   {'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 2**32}])
def linear_regression_sample_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'max_lags': 30},
                   {'posteriors': 1, 'max_lags': 30},
                   {'posteriors': 1.1, 'max_lags': 30},
                   {'posteriors': True, 'max_lags': 30},
                   {'posteriors': (0, 1), 'max_lags': 30},
                   {'posteriors': [0, 1], 'max_lags': 30},
                   {'posteriors': {0, 1}, 'max_lags': 30},
                   {'posteriors': None, 'max_lags': 30},
                   {'posteriors': {'intercept': '1'}, 'max_lags': 30},
                   {'posteriors': {'intercept': 1}, 'max_lags': 30},
                   {'posteriors': {'intercept': 1.1}, 'max_lags': 30},
                   {'posteriors': {'intercept': True}, 'max_lags': 30},
                   {'posteriors': {'intercept': (0, 1)}, 'max_lags': 30},
                   {'posteriors': {'intercept': [0, 1]}, 'max_lags': 30},
                   {'posteriors': {'intercept': {0, 1}}, 'max_lags': 30},
                   {'posteriors': {'intercept': {0 : 1}}, 'max_lags': 30},
                   {'posteriors': {'intercept': None}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': 'max_lags'},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': 1.1},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': (0, 1)},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': [0, 1]},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': {0, 1}},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': {'30': 30}},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': None}])
def diagnostics_autocorrelation_plot_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([])}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': -1}])
def diagnostics_autocorrelation_plot_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': 1, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': 1.1, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': True, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': (0, 1), 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': [0, 1], 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {0, 1}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': None, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': '1'}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': 1}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': 1.1}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': True}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': (0, 1)}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': [0, 1]}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': {0, 1}}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': {0: 1}}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': None}, 'lags': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': 'lags', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': 1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': 1.1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': True, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': (0, 1), 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': {'30': 30}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': {0, 1}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': ['lag'], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [1.1], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [(0, 1)], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [[0, 1]], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [{0, 1}], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [{'lag': 1}], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [None], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [0, 1], 'print_summary': 'False'},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [0, 1], 'print_summary': 1.1},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [0, 1], 'print_summary': (0, 1)},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [0, 1], 'print_summary': [0, 1]},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [0, 1], 'print_summary': {0, 1}},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [0, 1], 'print_summary': {0: 1}},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [0, 1], 'print_summary': None}])
def diagnostics_autocorrelation_summary_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([])}, 'lags': [0, 1, 5, 10, 30]},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': []},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [-1]}])
def diagnostics_autocorrelation_summary_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'print_summary': False},
                   {'posteriors': 1, 'print_summary': False},
                   {'posteriors': 1.1, 'print_summary': False},
                   {'posteriors': True, 'print_summary': False},
                   {'posteriors': (0, 1), 'print_summary': False},
                   {'posteriors': [0, 1], 'print_summary': False},
                   {'posteriors': {0, 1}, 'print_summary': False},
                   {'posteriors': None, 'print_summary': False},
                   {'posteriors': {'intercept': '1'}, 'print_summary': False},
                   {'posteriors': {'intercept': 1}, 'print_summary': False},
                   {'posteriors': {'intercept': 1.1}, 'print_summary': False},
                   {'posteriors': {'intercept': True}, 'print_summary': False},
                   {'posteriors': {'intercept': (0, 1)}, 'print_summary': False},
                   {'posteriors': {'intercept': [0, 1]}, 'print_summary': False},
                   {'posteriors': {'intercept': {0, 1}}, 'print_summary': False},
                   {'posteriors': {'intercept': {0: 1}}, 'print_summary': False},
                   {'posteriors': {'intercept': None}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'print_summary': 'False'},
                   {'posteriors': {'intercept': np.array([0])}, 'print_summary': 1.1},
                   {'posteriors': {'intercept': np.array([0])}, 'print_summary': (0, 1)},
                   {'posteriors': {'intercept': np.array([0])}, 'print_summary': [0, 1]},
                   {'posteriors': {'intercept': np.array([0])}, 'print_summary': {0, 1}},
                   {'posteriors': {'intercept': np.array([0])}, 'print_summary': {0: 1}}])
def diagnostics_effective_sample_size_type_error(request):
    return request.param


@fixture(params = ['posteriors', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, None,
                   {'intercept': [0]}])
def analysis_trace_plot_type_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': 1, 'alpha': 0.05,'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': 1.1, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': True, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': (0, 1), 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': [0, 1], 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {0, 1}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': None, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': '1'}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': 1}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': 1.1}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': True}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': (0, 1)}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': [0, 1]}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': {0, 1}}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': {0: 1}}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': None}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 'alpha', 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 2, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': (0, 1), 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': [0, 1], 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': {0, 1}, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': {'0': 1}, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': None, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': 'quantiles', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': 1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': 1.1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': True, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': (0, 1), 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': {'0.1': 0.1}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': {0, 1}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': ['quantiles'], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [1], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [(0, 1)], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [[0, 1]], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [{0, 1}], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [{'quantiles': 1}], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [None], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': 'False'},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': 1.1},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': (0, 1)},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': [0, 1]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': {0, 1}},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': None}])
def analysis_summary_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': -0.5, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 1.5, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': []},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [-0.5]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [1.5]}])
def analysis_summary_value_error(request):
    return request.param


@fixture(params = [*types_to_check,
                   *analysis_residuals_plot_type_error_1,
                   *analysis_residuals_plot_type_error_2])
def analysis_residuals_plot_type_error(request):
    return request.param


@fixture(params = [analysis_value_error_posterior_data_empty,
                   analysis_value_error_posterior_not_in_data,
                   analysis_value_error_data_empty,
                   analysis_value_error_response_variable_not_in_data])
def analysis_residuals_plot_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'predictors': {'predictor': 1}},
                   {'posteriors': 1, 'predictors': {'regressor': 1}},
                   {'posteriors': 1.1, 'predictors': {'regressor': 1}},
                   {'posteriors': True, 'predictors': {'regressor': 1}},
                   {'posteriors': (0, 1), 'predictors': {'regressor': 1}},
                   {'posteriors': [0, 1], 'predictors': {'regressor': 1}},
                   {'posteriors': {0, 1}, 'predictors': {'regressor': 1}},
                   {'posteriors': None, 'predictors': {'regressor': 1}},
                   {'posteriors': {'intercept': [0], 'variance': np.array([0])}, 'predictors': {'predictor': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'predictors': 'predictors'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'predictors': 1},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'predictors': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'predictors': True},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'predictors': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'predictors': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'predictors': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'predictors': pd.DataFrame()},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'predictors': None}])
def analysis_predict_distribution_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'variance': np.array([0]), 'x': np.array([0])}, 'predictors': {'x': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0]), 'x': np.array([0])}, 'predictors': {'x': 1, 'z': 1}}])
def analysis_predict_distribution_key_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([]), 'variance': np.array([0]), 'x': np.array([0])}, 'predictors': {'x': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0]), 'x': np.array([0])}, 'predictors': {}}])
def analysis_predict_distribution_value_error(request):
    return request.param


@fixture(params = [*analysis_compute_dic_type_error_1,
                   *analysis_compute_dic_type_error_2,
                   *analysis_compute_dic_type_error_3,
                   *analysis_compute_dic_type_error_4])
def analysis_compute_dic_type_error(request):
    return request.param


@fixture(params = [analysis_value_error_posterior_data_empty,
                   analysis_value_error_posterior_not_in_data,
                   analysis_value_error_data_empty,
                   analysis_value_error_response_variable_not_in_data])
def analysis_compute_dic_value_error(request):
    return request.param


@fixture(scope = 'session')
def posteriors(general_testing_data):
    model = bp.model.LinearModel()
    model.data = general_testing_data['data']
    model.response_variable = general_testing_data['response_variable']
    model.priors = general_testing_data['priors']
    sampler = bp.regression.LinearRegression(model = model)
    sampler.sample(n_iterations = general_testing_data['n_iterations'],
                   burn_in_iterations = general_testing_data['burn_in_iterations'],
                   n_chains = general_testing_data['n_chains'])
    return model.posteriors


@fixture(scope = 'session')
def solved_model(general_testing_data):
    model = bp.model.LinearModel()
    model.data = general_testing_data['data']
    model.response_variable = general_testing_data['response_variable']
    model.priors = general_testing_data['priors']
    sampler = bp.regression.LinearRegression(model = model)
    sampler.sample(n_iterations = general_testing_data['n_iterations'],
                   burn_in_iterations = general_testing_data['burn_in_iterations'],
                   n_chains = general_testing_data['n_chains'])
    return model


@fixture(params = ['1', 1, 1.1, True, (0, 1), [0, 1], {1, 2}, {0: 1}, None])
def utils_flatten_matrix_type_error(request):
    return request.param


@fixture(params = ['1', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, None, {'a': 'a'}, {'a': 1}, {'a': 1.1}, {'a': True},
                   {'a': (0, 1)}, {'a': [0, 1]}, {'a': {0, 1}}, {'a': {0: 1}}, {'a': None}])
def utils_matrices_to_frame_type_error(request):
    return request.param
