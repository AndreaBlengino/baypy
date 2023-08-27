import GibbsSampler as gs
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

model_no_data = gs.model.LinearModel()
model_no_data.priors = {'x': {'mean': 0,
                              'variance': 1},
                        'intercept': {'mean': 0,
                                      'variance': 1},
                        'variance': {'shape': 1,
                                     'scale': 1}}

model_no_response_variable = gs.model.LinearModel()
model_no_response_variable.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
model_no_response_variable.priors = {'x': {'mean': 0,
                                           'variance': 1},
                                     'y': {'mean': 0,
                                           'variance': 1},
                                     'intercept': {'mean': 0,
                                                   'variance': 1},
                                     'variance': {'shape': 1,
                                                  'scale': 1}}

model_no_priors = gs.model.LinearModel()
model_no_priors.data = pd.DataFrame(columns = ['x', 'z'], index = [0])
model_no_priors.response_variable = 'z'

model_response_variable_not_in_data = gs.model.LinearModel()
model_response_variable_not_in_data.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
model_response_variable_not_in_data.response_variable = 'w'
model_response_variable_not_in_data.priors = {'x': {'mean': 0,
                                                    'variance': 1},
                                              'y': {'mean': 0,
                                                    'variance': 1},
                                              'intercept': {'mean': 0,
                                                            'variance': 1},
                                              'variance': {'shape': 1,
                                                           'scale': 1}}

model_prior_not_in_data = gs.model.LinearModel()
model_prior_not_in_data.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
model_prior_not_in_data.response_variable = 'z'
model_prior_not_in_data.priors = {'x': {'mean': 0,
                                        'variance': 1},
                                  'y': {'mean': 0,
                                        'variance': 1},
                                  'w': {'mean': 0,
                                        'variance': 1},
                                  'intercept': {'mean': 0,
                                                'variance': 1},
                                  'variance': {'shape': 1,
                                               'scale': 1}}


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
    model = gs.model.LinearModel()
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
def sampler(general_testing_data):
    model = gs.model.LinearModel()
    model.data = general_testing_data['data']
    model.response_variable = general_testing_data['response_variable']
    model.priors = general_testing_data['priors']
    sampler = gs.regression.LinearRegression(model = model)
    return sampler


@fixture(params = ['model', 1, 1.1, {'model': 1}, True, (0, 1), [0, 1], {0, 1}, None])
def linear_regression_init_type_error(request):
    return request.param


@fixture(params = [model_no_data,
                   model_no_response_variable,
                   model_no_priors,
                   model_response_variable_not_in_data,
                   model_prior_not_in_data])
def linear_regression_init_value_error(request):
    return request.param


@fixture(params = [{'n_iterations': '100', 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100.0, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': {'100': 100}, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': (100, 200), 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': [100, 200], 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': {100, 200}, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': None, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': '50', 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50.0, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': {'50': 50}, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': (50, 100), 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': [50, 100], 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': {50, 100}, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': None, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': '3', 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3.0, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': {'3': 3}, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': (3, 6), 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': [4, 6], 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': {3, 6}, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': None, 'seed': 137},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': '137'},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137.0},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': {'137': 137}},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': (137, 137)},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': [137, 137]},
                   {'n_iterations': 100, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': {137, 137}}])
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
                   {'posteriors': {'intercept': [0], 'variance': np.array([0])}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'max_lags': {'30': 30}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'max_lags': 'max_lags'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'max_lags': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'max_lags': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'max_lags': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'max_lags': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'max_lags': None}])
def diagnostics_autocorrelation_plot_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'variance': np.array([0])}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': 30}])
def diagnostics_autocorrelation_plot_key_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([]), 'variance': np.array([0])}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'max_lags': -1}])
def diagnostics_autocorrelation_plot_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'lags': 30, 'print_summary': False},
                   {'posteriors': 1, 'lags': 30, 'print_summary': False},
                   {'posteriors': 1.1, 'lags': 30, 'print_summary': False},
                   {'posteriors': True, 'lags': 30, 'print_summary': False},
                   {'posteriors': (0, 1), 'lags': 30, 'print_summary': False},
                   {'posteriors': [0, 1], 'lags': 30, 'print_summary': False},
                   {'posteriors': {0, 1}, 'lags': 30, 'print_summary': False},
                   {'posteriors': None, 'lags': 30, 'print_summary': False},
                   {'posteriors': {'intercept': [0], 'variance': np.array([0])}, 'lags': 30, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': 'lags', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': 1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': 1.1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': True, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': (0, 1), 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': {'30': 30}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': {0, 1}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': ['lag'], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': [1.1], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': [(0, 1)], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': [[0, 1]], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': [{0, 1}], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': [{'lag': 1}], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': [None], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': 30, 'print_summary': 'False'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': 30, 'print_summary': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': 30, 'print_summary': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': 30, 'print_summary': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': 30, 'print_summary': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': 30, 'print_summary': None}])
def diagnostics_autocorrelation_summary_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'variance': np.array([0])}, 'lags': [0, 1, 5, 10, 30]},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [0, 1, 5, 10, 30]}])
def diagnostics_autocorrelation_summary_key_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([]), 'variance': np.array([0])}, 'lags': [0, 1, 5, 10, 30]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': []},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'lags': [-1]}])
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
                   {'posteriors': {'intercept': [0], 'variance': np.array([0])}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'print_summary': 'False'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'print_summary': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'print_summary': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'print_summary': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'print_summary': {0, 1}}])
def diagnostics_effective_sample_size_type_error(request):
    return request.param


@fixture(params = [{'variance': np.array([0])},
                   {'intercept': np.array([0])}])
def diagnostics_effective_sample_size_key_error(request):
    return request.param


@fixture(params = ['posteriors', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, None,
                   {'intercept': [0], 'variance': np.array([0])}])
def analysis_trace_plot_type_error(request):
    return request.param


@fixture(params = [{'variance': np.array([0])},
                   {'intercept': np.array([0])}])
def analysis_trace_plot_key_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': 1, 'alpha': 0.05,'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': 1.1, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': True, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': (0, 1), 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': [0, 1], 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {0, 1}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': None, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': [0], 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 'alpha', 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 2, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': (0, 1), 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': [0, 1], 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': {0, 1}, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': {'0': 1}, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': None, 'quantiles': [0.1, 0.9], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': 'quantiles', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': 1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': 1.1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': True, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': (0, 1), 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': {'0.1': 0.1}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': {0, 1}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': ['quantiles'], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [1], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [(0, 1)], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [[0, 1]], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [{0, 1}], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [{'quantiles': 1}], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [None], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': 'False'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9], 'print_summary': None}])
def analysis_summary_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]}])
def analysis_summary_key_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': -0.5, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 1.5, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': []},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [-0.5]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'alpha': 0.05, 'quantiles': [1.5]}])
def analysis_summary_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': 1, 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': 1.1, 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': True, 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': (0, 1), 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': [0, 1], 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': {0, 1}, 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': None, 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': [0], 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': 'data', 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': 1, 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': 1.1, 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': True, 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': (0, 1), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': [0, 1], 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': {0, 1}, 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': {'0': 1}, 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': None, 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 1},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': True},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': {'response_variable': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': None}])
def analysis_residuals_plot_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'variance': np.array([0])}, 'data': pd.DataFrame(columns = ['response_variable'], index = [0]), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0])}, 'data': pd.DataFrame(columns = ['response_variable'], index = [0]), 'response_variable': 'response_variable'}])
def analysis_residuals_plot_key_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([]), 'variance': np.array([0])}, 'data': pd.DataFrame(columns = ['response_variable'], index = [0]), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0]), 'x': np.array([0])}, 'data': pd.DataFrame(columns = ['response_variable'], index = [0]), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(columns = ['not_response_variable'], index = [0]), 'response_variable': 'response_variable'}])
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
                   {'posteriors': {'intercept': np.array([0]), 'x': np.array([0])}, 'predictors': {'x': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0]), 'x': np.array([0])}, 'predictors': {'x': 1, 'z': 1}}])
def analysis_predict_distribution_key_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([]), 'variance': np.array([0]), 'x': np.array([0])}, 'predictors': {'x': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0]), 'x': np.array([0])}, 'predictors': {}}])
def analysis_predict_distribution_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': 1, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': 1.1, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': True, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': (0, 1), 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': [0, 1], 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {0, 1}, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': None, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': [0], 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': 'data', 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': 1, 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': 1.1, 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': True, 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': (0, 1), 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': [0, 1], 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': {0, 1}, 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': {'0': 1}, 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': None, 'response_variable': 'response_variable', 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 1.1, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': True, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': (0, 1), 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': [0, 1], 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': {0, 1}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': { 'response_variable': 1}, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': None, 'print_summary': False},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': 'False'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable', 'print_summary': None}])
def analysis_compute_dic_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'variance': np.array([0])}, 'data': pd.DataFrame(columns = ['response_variable'], index = [0]), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0])}, 'data': pd.DataFrame(columns = ['response_variable'], index = [0]), 'response_variable': 'response_variable'}])
def analysis_compute_dic_key_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([]), 'variance': np.array([0])}, 'data': pd.DataFrame(columns = ['response_variable'], index = [0]), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0]), 'x': np.array([0])}, 'data': pd.DataFrame(columns = ['response_variable'], index = [0]), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(), 'response_variable': 'response_variable'},
                   {'posteriors': {'intercept': np.array([0]), 'variance': np.array([0])}, 'data': pd.DataFrame(columns = ['not_response_variable'], index = [0]), 'response_variable': 'response_variable'}])
def analysis_compute_dic_value_error(request):
    return request.param


@fixture(scope = 'session')
def posteriors(sampler, general_testing_data):
    sampler.sample(n_iterations = general_testing_data['n_iterations'],
                   burn_in_iterations = general_testing_data['burn_in_iterations'],
                   n_chains = general_testing_data['n_chains'])
    return sampler.posteriors
