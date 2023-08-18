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


@fixture(scope = 'function')
def empty_model():
    model = gs.Model()
    return model


@fixture(scope = 'function')
def model(empty_model):
    empty_model.set_data(data = data, y_name = 'y')
    empty_model.set_initial_values(values = initial_values)
    empty_model.set_priors(priors = priors)
    return empty_model


@fixture(scope = 'function')
def sampler(model):
    sampler = gs.LinearRegression(model = model)
    return sampler


@fixture(scope = 'function')
def posteriors(sampler):
    sampler.sample(n_iterations = n_iterations,
                   burn_in_iterations = burn_in_iterations,
                   n_chains = n_chains)
    return sampler.posteriors
