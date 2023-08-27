import GibbsSampler as gs
import numpy as np
import pandas as pd
from pytest import mark, raises


@mark.regression
class TestLinearRegressionInit:


    def test_raises_type_error(self, linear_regression_init_type_error):
        with raises(TypeError):
            gs.regression.LinearRegression(model = linear_regression_init_type_error)


    def test_raises_value_error(self, linear_regression_init_value_error):
        with raises(ValueError):
            gs.regression.LinearRegression(model = linear_regression_init_value_error)


@mark.regression
class TestLinearRegressionSample:


    def test_method(self, sampler, general_testing_data):
        sampler.sample(n_iterations = general_testing_data['n_iterations'],
                       burn_in_iterations = general_testing_data['burn_in_iterations'],
                       n_chains = general_testing_data['n_chains'],
                       seed = general_testing_data['seed'])

        assert sampler.posteriors.keys() == general_testing_data['priors'].keys()
        assert all(np.array([posterior_samples.shape for posterior_samples in sampler.posteriors.values()])[:, 0] == general_testing_data['n_iterations'])
        assert all(np.array([posterior_samples.shape for posterior_samples in sampler.posteriors.values()])[:, 1] == general_testing_data['n_chains'])


    def test_raises_type_error(self, sampler, linear_regression_sample_type_error):
        with raises(TypeError):
            sampler.sample(n_iterations = linear_regression_sample_type_error['n_iterations'],
                           burn_in_iterations = linear_regression_sample_type_error['burn_in_iterations'],
                           n_chains = linear_regression_sample_type_error['n_chains'],
                           seed = linear_regression_sample_type_error['seed'])


    def test_raises_value_error(self, sampler, linear_regression_sample_value_error):
        with raises(ValueError):
            sampler.sample(n_iterations = linear_regression_sample_value_error['n_iterations'],
                           burn_in_iterations = linear_regression_sample_value_error['burn_in_iterations'],
                           n_chains = linear_regression_sample_value_error['n_chains'],
                           seed = linear_regression_sample_value_error['seed'])


@mark.regression
class TestLinearRegressionPosteriorsToFrame:

    def test_method(self, sampler, general_testing_data):
        posteriors = sampler.sample(n_iterations = general_testing_data['n_iterations'],
                                    burn_in_iterations = general_testing_data['burn_in_iterations'],
                                    n_chains = general_testing_data['n_chains'])
        posteriors_frame = sampler.posteriors_to_frame()

        assert isinstance(posteriors_frame, pd.DataFrame)
        assert not posteriors_frame.empty
        assert all(posteriors_frame.columns == list(posteriors.keys()))
        assert len(posteriors_frame) == general_testing_data['n_iterations']*general_testing_data['n_chains']


    def test_raises_value_error(self, general_testing_data):
        model = gs.model.LinearModel()
        model.data = general_testing_data['data']
        model.response_variable = general_testing_data['response_variable']
        model.priors = general_testing_data['priors']
        sampler = gs.regression.LinearRegression(model = model)
        with raises(ValueError):
            sampler.posteriors_to_frame()
