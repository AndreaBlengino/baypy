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

initial_values = {'x_1': 1,
                  'x_2': 2,
                  'x_3': 3,
                  'x_1 * x_2': 4,
                  'intercept': 5}

sigma2_sample_size = 5
sigma2_variance = 10

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
          'sigma2': {'shape': sigma2_sample_size,
                     'scale': sigma2_sample_size*sigma2_variance}}

n_iterations = 1000
burn_in_iterations = 50
n_chains = 3

regressor_names = ['intercept', 'x_1', 'x_2', 'x_3', 'x_1 * x_2']

q_min = 0.025
q_max = 0.975

prediction_data = {'x_1': 20, 'x_2': 5, 'x_3': -45}
prediction_data['x_1 * x_2'] = prediction_data['x_1']*prediction_data['x_2']


model_initial_value_not_in_data = gs.Model()
model_initial_value_not_in_data.set_data(data = pd.DataFrame(columns = ['x', 'z'], index = [0]),
                                         y_name = 'z')
model_initial_value_not_in_data.set_initial_values(values = {'x': 0,
                                                             'y': 0,
                                                             'intercept': 0})
model_initial_value_not_in_data.set_priors(priors = {'x': {'mean': 0,
                                                           'variance': 1},
                                                     'y': {'mean': 0,
                                                           'variance': 1},
                                                     'intercept': {'mean': 0,
                                                                   'variance': 1},
                                                     'sigma2': {'shape': 1,
                                                                'scale': 1}})

model_initial_value_not_in_priors = gs.Model()
model_initial_value_not_in_priors.set_data(data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0]),
                                           y_name = 'z')
model_initial_value_not_in_priors.set_initial_values(values = {'x': 0,
                                                               'y': 0,
                                                               'intercept': 0})
model_initial_value_not_in_priors.set_priors(priors = {'x': {'mean': 0,
                                                             'variance': 1},
                                                       'intercept': {'mean': 0,
                                                                     'variance': 1},
                                                       'sigma2': {'shape': 1,
                                                                  'scale': 1}})

model_prior_not_in_data = gs.Model()
model_prior_not_in_data.set_data(data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0]),
                                 y_name = 'z')
model_prior_not_in_data.set_initial_values(values = {'x': 0,
                                                     'y': 0,
                                                     'intercept': 0})
model_prior_not_in_data.set_priors(priors = {'x': {'mean': 0,
                                                   'variance': 1},
                                             'y': {'mean': 0,
                                                   'variance': 1},
                                             'w': {'mean': 0,
                                                   'variance': 1},
                                             'intercept': {'mean': 0,
                                                           'variance': 1},
                                             'sigma2': {'shape': 1,
                                                        'scale': 1}})

model_prior_not_in_initial_values = gs.Model()
model_prior_not_in_initial_values.set_data(data = pd.DataFrame(columns = ['x', 'y', 'w', 'z'], index = [0]),
                                           y_name = 'z')
model_prior_not_in_initial_values.set_initial_values(values = {'x': 0,
                                                               'y': 0,
                                                               'intercept': 0})
model_prior_not_in_initial_values.set_priors(priors = {'x': {'mean': 0,
                                                             'variance': 1},
                                                       'y': {'mean': 0,
                                                             'variance': 1},
                                                       'w': {'mean': 0,
                                                             'variance': 1},
                                                       'intercept': {'mean': 0,
                                                                     'variance': 1},
                                                       'sigma2': {'shape': 1,
                                                                  'scale': 1}})


@fixture(scope = 'session')
def empty_model():
    model = gs.Model()
    return model


@fixture(params = [{'data': 'data', 'y_name': 'y_name'},
                   {'data': 1, 'y_name': 'y_name'},
                   {'data': 1.1, 'y_name': 'y_name'},
                   {'data': True, 'y_name': 'y_name'},
                   {'data': (0, 1), 'y_name': 'y_name'},
                   {'data': [0, 1], 'y_name': 'y_name'},
                   {'data': {0, 1}, 'y_name': 'y_name'},
                   {'data': None, 'y_name': 'y_name'},
                   {'data': pd.DataFrame(), 'y_name': 1},
                   {'data': pd.DataFrame(), 'y_name': 1.1},
                   {'data': pd.DataFrame(), 'y_name': True},
                   {'data': pd.DataFrame(), 'y_name': (0, 1)},
                   {'data': pd.DataFrame(), 'y_name': [0, 1]},
                   {'data': pd.DataFrame(), 'y_name': {0, 1}},
                   {'data': pd.DataFrame(), 'y_name': None}])
def model_set_data_type_error(request):
    return request.param


@fixture(params = [{'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'data': pd.DataFrame(columns = ['not_y_name'], index = [0]), 'y_name': 'y_name'}])
def model_set_data_value_error(request):
    return request.param


@fixture(params = ['initial_values', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, None])
def model_set_initial_value_type_error(request):
    return request.param


@fixture(params = ['priors', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, None,
                   {'intercept': 'intercept', 'sigma2': 'sigma2'},
                   {'intercept': 1, 'sigma2': 1},
                   {'intercept': 1.1, 'sigma2': 1.1},
                   {'intercept': True, 'sigma2': True},
                   {'intercept': (0, 1), 'sigma2': (0, 1)},
                   {'intercept': [0, 1], 'sigma2': [0, 1]},
                   {'intercept': {0, 1}, 'sigma2': {0, 1}},
                   {'intercept': None, 'sigma2': None}])
def model_set_priors_type_error(request):
    return request.param


@fixture(params = [{},
                   {'intercept': {}, 'sigma2': {}},
                   {'intercept': {'m': 0, 'variance': 1}, 'sigma2': {'shape': 1, 'scale': 1}},
                   {'intercept': {'mean': 0, 'v': 1}, 'sigma2': {'shape': 1, 'scale': 1}},
                   {'intercept': {'mean': 0, 'variance': 1}, 'sigma2': {'s': 1, 'scale': 1}},
                   {'intercept': {'mean': 0, 'variance': 1}, 'sigma2': {'shape': 1, 's': 1}}])
def model_set_priors_value_error(request):
    return request.param


@fixture(params = [{'regressor': 1, 'sigma2': 2},
                   {'intercept': 1, 'regressor': 2}])
def model_set_priors_key_error(request):
    return request.param


@fixture(scope = 'session')
def sampler():
    model = gs.Model()
    model.set_data(data = data, y_name = 'y')
    model.set_initial_values(values = initial_values)
    model.set_priors(priors = priors)
    sampler = gs.LinearRegression(model = model)
    return sampler


@fixture(params = ['model', 1, 1.1, {'model': 1}, True, (0, 1), [0, 1], {0, 1}, None])
def linear_regression_init_type_error(request):
    return request.param


@fixture(params = [model_initial_value_not_in_data,
                   model_initial_value_not_in_priors,
                   model_prior_not_in_data,
                   model_prior_not_in_initial_values])
def linear_regression_init_value_error(request):
    return request.param


@fixture(params = [{'n_iterations': 1000,
                    'burn_in_iterations': 50,
                    'n_chains': 3},
                   {'n_iterations': 100,
                    'burn_in_iterations': 50,
                    'n_chains': 5},
                   {'n_iterations': 100,
                    'burn_in_iterations': 0,
                    'n_chains': 3}])
def linear_regression_sample(request):
    return request.param


@fixture(params = [{'n_iterations': -1,
                    'burn_in_iterations': 50,
                    'n_chains': 3},
                   {'n_iterations': 1000,
                    'burn_in_iterations': -1,
                    'n_chains': 3},
                   {'n_iterations': 1000,
                    'burn_in_iterations': 50,
                    'n_chains': -1}])
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
                   {'posteriors': {'intercept': [0], 'sigma2': np.array([0])}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'max_lags': {'30': 30}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'max_lags': 'max_lags'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'max_lags': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'max_lags': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'max_lags': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'max_lags': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'max_lags': None}])
def diagnostics_autocorrelation_plot_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'sigma2': np.array([0])}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([]), 'sigma2': np.array([0])}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'max_lags': -1}])
def diagnostics_autocorrelation_plot_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'lags': 30},
                   {'posteriors': 1, 'lags': 30},
                   {'posteriors': 1.1, 'lags': 30},
                   {'posteriors': True, 'lags': 30},
                   {'posteriors': (0, 1), 'lags': 30},
                   {'posteriors': [0, 1], 'lags': 30},
                   {'posteriors': {0, 1}, 'lags': 30},
                   {'posteriors': None, 'lags': 30},
                   {'posteriors': {'intercept': [0], 'sigma2': np.array([0])}, 'lags': 30},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': 'lags'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': 1},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': True},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': {'30': 30}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': ['lag']},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': [1.1]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': [(0, 1)]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': [[0, 1]]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': [{0, 1}]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': [{'lag': 1}]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': [None]}])
def diagnostics_autocorrelation_summary_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'sigma2': np.array([0])}, 'lags': [0, 1, 5, 10, 30]},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [0, 1, 5, 10, 30]},
                   {'posteriors': {'intercept': np.array([]), 'sigma2': np.array([0])}, 'lags': [0, 1, 5, 10, 30]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': []},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'lags': [-1]}])
def diagnostics_autocorrelation_summary_value_error(request):
    return request.param


@fixture(params = ['posteriors', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, None,
                   {'intercept': [0], 'sigma2': np.array([0])}])
def diagnostics_effective_sample_size_type_error(request):
    return request.param


@fixture(params = [{'sigma2': np.array([0])},
                   {'intercept': np.array([0])},
                   {'intercept': np.array([]), 'sigma2': np.array([0])}])
def diagnostics_effective_sample_size_value_error(request):
    return request.param


@fixture(params = ['posteriors', 1, 1.1, True, (0, 1), [0, 1], {0, 1}, None,
                   {'intercept': [0], 'sigma2': np.array([0])}])
def analysis_trace_plot_type_error(request):
    return request.param


@fixture(params = [{'sigma2': np.array([0])},
                   {'intercept': np.array([0])},
                   {'intercept': np.array([]), 'sigma2': np.array([0])}])
def analysis_trace_plot_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': 1, 'alpha': 0.05,'quantiles': [0.1, 0.9]},
                   {'posteriors': 1.1, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': True, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': (0, 1), 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': [0, 1], 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {0, 1}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': None, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': [0], 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 'alpha', 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 2, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': (0, 1), 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': [0, 1], 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': {0, 1}, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': {'0': 1}, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': None, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': 'quantiles'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': 1},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': True},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': {'0.1': 0.1}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': ['quantiles']},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [1]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [(0, 1)]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [[0, 1]]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [{0, 1}]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [{'quantiles': 1}]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [None]}])
def analysis_summary_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': -0.5, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 1.5, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': []},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [-0.5]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'alpha': 0.05, 'quantiles': [1.5]}])
def analysis_summary_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': 1, 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': 1.1, 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': True, 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': (0, 1), 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': [0, 1], 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': {0, 1}, 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': None, 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': {'intercept': [0], 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': 'data', 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': 1, 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': 1.1, 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': True, 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': (0, 1), 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': [0, 1], 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': {0, 1}, 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': {'0': 1}, 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': None, 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': 1},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': True},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': {'y_name': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': None}])
def analysis_residuals_plot_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'sigma2': np.array([0])}, 'data': pd.DataFrame(columns = ['y_name'], index = [0]), 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0])}, 'data': pd.DataFrame(columns = ['y_name'], index = [0]), 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(columns = ['y_name'], index = [0]), 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0]), 'x': np.array([0])}, 'data': pd.DataFrame(columns = ['y_name'], index = [0]), 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(), 'y_name': 'y_name'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame(columns = ['not_y_name'], index = [0]), 'y_name': 'y_name'}])
def analysis_residuals_plot_value_error(request):
    return request.param


@fixture(params = [{'posteriors': 'posteriors', 'data': {'regressor': 1}},
                   {'posteriors': 1, 'data': {'regressor': 1}},
                   {'posteriors': 1.1, 'data': {'regressor': 1}},
                   {'posteriors': True, 'data': {'regressor': 1}},
                   {'posteriors': (0, 1), 'data': {'regressor': 1}},
                   {'posteriors': [0, 1], 'data': {'regressor': 1}},
                   {'posteriors': {0, 1}, 'data': {'regressor': 1}},
                   {'posteriors': None, 'data': {'regressor': 1}},
                   {'posteriors': {'intercept': [0], 'sigma2': np.array([0])}, 'data': {'regressor': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': 'data'},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': 1},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': 1.1},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': True},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': (0, 1)},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': [0, 1]},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': {0, 1}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': pd.DataFrame()},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0])}, 'data': None}])
def analysis_predict_distribution_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'sigma2': np.array([0]), 'x': np.array([0])}, 'data': {'x': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'x': np.array([0])}, 'data': {'x': 1}},
                   {'posteriors': {'intercept': np.array([]), 'sigma2': np.array([0]), 'x': np.array([0])}, 'data': {'x': 1}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0]), 'x': np.array([0])}, 'data': {}},
                   {'posteriors': {'intercept': np.array([0]), 'sigma2': np.array([0]), 'x': np.array([0])}, 'data': {'x': 1, 'z': 1}}])
def analysis_predict_distribution_value_error(request):
    return request.param


@fixture(scope = 'session')
def posteriors(sampler):
    sampler.sample(n_iterations = n_iterations,
                   burn_in_iterations = burn_in_iterations,
                   n_chains = n_chains)
    return sampler.posteriors
