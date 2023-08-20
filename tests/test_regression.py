import GibbsSampler as gs
import numpy as np
from pytest import fixture, mark, raises


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


    def test_method(self, sampler, general_testing_data):
        sampler.sample(n_iterations = general_testing_data['n_iterations'],
                       burn_in_iterations = general_testing_data['burn_in_iterations'],
                       n_chains = general_testing_data['n_chains'])

        assert sampler.posteriors.keys() == general_testing_data['priors'].keys()
        assert all(np.array([posterior_samples.shape for posterior_samples in sampler.posteriors.values()])[:, 0] == general_testing_data['n_iterations'])
        assert all(np.array([posterior_samples.shape for posterior_samples in sampler.posteriors.values()])[:, 1] == general_testing_data['n_chains'])


    def test_raises_value_error(self, sampler, linear_regression_sample_value_error):
        with raises(ValueError):
            sampler.sample(n_iterations = linear_regression_sample_value_error['n_iterations'],
                           burn_in_iterations = linear_regression_sample_value_error['burn_in_iterations'],
                           n_chains = linear_regression_sample_value_error['n_chains'])
