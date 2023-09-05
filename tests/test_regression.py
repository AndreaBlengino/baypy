import baypy as bp
import numpy as np
import pandas as pd
from pytest import mark, raises


@mark.regression
class TestLinearRegressionInit:


    def test_raises_type_error(self, linear_regression_init_type_error):
        with raises(TypeError):
            bp.regression.LinearRegression(model = linear_regression_init_type_error)


    def test_raises_value_error(self, linear_regression_init_value_error):
        with raises(ValueError):
            bp.regression.LinearRegression(model = linear_regression_init_value_error)


@mark.regression
class TestLinearRegressionSample:


    def test_method(self, model, general_testing_data):
        sampler = bp.regression.LinearRegression(model = model)
        sampler.sample(n_iterations = general_testing_data['n_iterations'],
                       burn_in_iterations = general_testing_data['burn_in_iterations'],
                       n_chains = general_testing_data['n_chains'],
                       seed = general_testing_data['seed'])

        assert model.posteriors.keys() == general_testing_data['priors'].keys()
        assert all(np.array([posterior_samples.shape for posterior_samples in model.posteriors.values()])[:, 0] == general_testing_data['n_iterations'])
        assert all(np.array([posterior_samples.shape for posterior_samples in model.posteriors.values()])[:, 1] == general_testing_data['n_chains'])


    def test_raises_type_error(self, model, linear_regression_sample_type_error):
        sampler = bp.regression.LinearRegression(model = model)
        with raises(TypeError):
            sampler.sample(n_iterations = linear_regression_sample_type_error['n_iterations'],
                           burn_in_iterations = linear_regression_sample_type_error['burn_in_iterations'],
                           n_chains = linear_regression_sample_type_error['n_chains'],
                           seed = linear_regression_sample_type_error['seed'])


    def test_raises_value_error(self, model, linear_regression_sample_value_error):
        sampler = bp.regression.LinearRegression(model = model)
        with raises(ValueError):
            sampler.sample(n_iterations = linear_regression_sample_value_error['n_iterations'],
                           burn_in_iterations = linear_regression_sample_value_error['burn_in_iterations'],
                           n_chains = linear_regression_sample_value_error['n_chains'],
                           seed = linear_regression_sample_value_error['seed'])
