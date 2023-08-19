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
def regression_model_type_error(request):
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
def regression_parameters(request):
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
def regression_parameters_value_error(request):
    return request.param


@fixture(scope = 'session')
def posteriors(sampler):
    sampler.sample(n_iterations = n_iterations,
                   burn_in_iterations = burn_in_iterations,
                   n_chains = n_chains)
    return sampler.posteriors
