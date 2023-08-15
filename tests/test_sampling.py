from pytest import mark
import numpy as np
import pandas as pd


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

prior = {'x_1': {'mean': 0,
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


@mark.set_input
def test_set_data(sampler):
    sampler.set_data(data = data,
                     y_name = 'y')

    assert sampler.data.equals(data)
    assert sampler.y_name == 'y'

@mark.set_input
def test_set_initial_values(sampler):
    sampler.set_initial_values(values = initial_values)

    assert sampler.initial_values == initial_values

@mark.set_input
def test_set_prior(sampler):
    sampler.set_prior(prior = prior)

    assert sampler.prior == prior
