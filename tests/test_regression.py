import baypy as bp
from hypothesis import given, settings, HealthCheck
import numpy as np
import pandas as pd
from pytest import mark, raises
from tests.conftest import model_set_up, model


@mark.regression
class TestLinearRegressionInit:


    @mark.genuine
    def test_raises_type_error(self, linear_regression_init_type_error):
        with raises(TypeError):
            bp.regression.LinearRegression(model = linear_regression_init_type_error)


    @mark.error
    def test_raises_value_error(self, linear_regression_init_value_error):
        with raises(ValueError):
            bp.regression.LinearRegression(model = linear_regression_init_value_error)


@mark.regression
class TestLinearRegressionSample:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_method(self, model_options):
        model = bp.model.LinearModel()
        model.data = model_options['data']
        model.response_variable = model_options['response_variable']
        model.priors = model_options['priors']
        sampler = bp.regression.LinearRegression(model = model)
        sampler.sample(n_iterations = model_options['n_samples'],
                       burn_in_iterations = model_options['burn_in_iterations'],
                       n_chains = model_options['n_chains'],
                       seed = model_options['seed'])

        assert isinstance(model.posteriors, dict)
        assert len(model.posteriors) > 0
        assert model.posteriors.keys() == model_options['priors'].keys()
        assert all(np.array([posterior_samples.shape for posterior_samples
                             in model.posteriors.values()])[:, 0] == model_options['n_samples'])
        assert all(np.array([posterior_samples.shape for posterior_samples
                             in model.posteriors.values()])[:, 1] == model_options['n_chains'])
        assert (model.posteriors['variance'] > 0).all()

        regressor_names = [posterior for posterior in model.posteriors.keys() if posterior != 'variance']
        data = model.data.copy()
        data['intercept'] = 1
        linear_model_results = np.linalg.lstsq(a = data[regressor_names],
                                               b = data[model.response_variable],
                                               rcond = None)[0]

        for i, regressor in enumerate(regressor_names, 0):
            lower_bound = np.quantile(np.asarray(model.posteriors[regressor]).reshape(-1), model_options['q_min'])
            upper_bound = np.quantile(np.asarray(model.posteriors[regressor]).reshape(-1), model_options['q_max'])

            assert lower_bound <= linear_model_results[i] <= upper_bound


    @mark.error
    def test_raises_type_error(self, linear_regression_sample_type_error):
        sampler = bp.regression.LinearRegression(model = model)
        with raises(TypeError):
            sampler.sample(n_iterations = linear_regression_sample_type_error['n_iterations'],
                           burn_in_iterations = linear_regression_sample_type_error['burn_in_iterations'],
                           n_chains = linear_regression_sample_type_error['n_chains'],
                           seed = linear_regression_sample_type_error['seed'])


    @mark.error
    def test_raises_value_error(self, linear_regression_sample_value_error):
        sampler = bp.regression.LinearRegression(model = model)
        with raises(ValueError):
            sampler.sample(n_iterations = linear_regression_sample_value_error['n_iterations'],
                           burn_in_iterations = linear_regression_sample_value_error['burn_in_iterations'],
                           n_chains = linear_regression_sample_value_error['n_chains'],
                           seed = linear_regression_sample_value_error['seed'])
