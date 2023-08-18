import GibbsSampler as gs
import numpy as np
from pytest import fixture, mark, raises


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


@mark.regression
class TestRegression:

    def test___init__raises_type_error(self, regression_model_type_error):
        with raises(TypeError):
            gs.LinearRegression(model = regression_model_type_error)


    def test_sample(self, sampler, regression_parameters):
        sampler.sample(n_iterations = regression_parameters['n_iterations'],
                       burn_in_iterations = regression_parameters['burn_in_iterations'],
                       n_chains = regression_parameters['n_chains'])

        assert sampler.posteriors.keys() == priors.keys()
        assert all(np.array([posterior.shape for posterior in sampler.posteriors.values()])[:, 0] == regression_parameters['n_iterations'])
        assert all(np.array([posterior.shape for posterior in sampler.posteriors.values()])[:, 1] == regression_parameters['n_chains'])


    def test_sample_raises_value_error(self, sampler, regression_parameters_value_error):
        with raises(ValueError):
            sampler.sample(n_iterations = regression_parameters_value_error['n_iterations'],
                           burn_in_iterations = regression_parameters_value_error['burn_in_iterations'],
                           n_chains = regression_parameters_value_error['n_chains'])
