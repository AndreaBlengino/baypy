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
class TestLinearRegressionInit:


    def test_raises_type_error(self, linear_regression_init_type_error):
        with raises(TypeError):
            gs.LinearRegression(model = linear_regression_init_type_error)


    def test_raises_value_error(self, linear_regression_init_value_error):
        with raises(ValueError):
            gs.LinearRegression(model = linear_regression_init_value_error)


@mark.regression
class TestLinearRegressionSample:


    def test_method(self, sampler, linear_regression_sample):
        sampler.sample(n_iterations = linear_regression_sample['n_iterations'],
                       burn_in_iterations = linear_regression_sample['burn_in_iterations'],
                       n_chains = linear_regression_sample['n_chains'])

        assert sampler.posteriors.keys() == priors.keys()
        assert all(np.array([posterior.shape for posterior in sampler.posteriors.values()])[:, 0] == linear_regression_sample['n_iterations'])
        assert all(np.array([posterior.shape for posterior in sampler.posteriors.values()])[:, 1] == linear_regression_sample['n_chains'])


    def test_raises_value_error(self, sampler, linear_regression_sample_value_error):
        with raises(ValueError):
            sampler.sample(n_iterations = linear_regression_sample_value_error['n_iterations'],
                           burn_in_iterations = linear_regression_sample_value_error['burn_in_iterations'],
                           n_chains = linear_regression_sample_value_error['n_chains'])
