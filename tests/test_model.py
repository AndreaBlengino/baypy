import numpy as np
import pandas as pd
from pytest import mark


np.random.seed(42)

N = 50
data = pd.DataFrame()
data['x_1'] = np.random.uniform(low = 0, high = 100, size = N)
data['x_2'] = np.random.uniform(low = -10, high = 10, size = N)
data['x_3'] = np.random.uniform(low = -50, high = -40, size = N)
data['x_1 * x_2'] = data['x_1']*data['x_2']

data['y'] = 3*data['x_1'] - 20*data['x_2'] - data['x_3'] - 5*data['x_1 * x_2'] + 13 + 1*np.random.randn(N)

regressor_names = ['intercept', 'x_1', 'x_2', 'x_3', 'x_1 * x_2']

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


@mark.model
class TestModel:

    def test_set_data(self, model):
        model.set_data(data = data,
                       y_name = 'y')

        assert model.data.equals(data)
        assert model.y_name == 'y'


    def test_set_initial_values(self, model):
        model.set_initial_values(values = initial_values)

        assert model.initial_values == initial_values
        assert 'intercept' in model.initial_values.keys()


    def test_set_priors(self, model):
        model.set_priors(priors = priors)

        assert model.priors == priors
        assert 'intercept' in model.priors.keys()
        assert 'sigma2' in model.priors.keys()
        assert all(['mean' in priors[regressor].keys() for regressor in priors.keys() if regressor != 'sigma2'])
        assert all(['variance' in priors[regressor].keys() for regressor in priors.keys() if regressor != 'sigma2'])
        assert 'shape' in model.priors['sigma2'].keys()
        assert 'scale' in model.priors['sigma2'].keys()
