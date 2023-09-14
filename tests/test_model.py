from baypy.model import LinearModel
from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import composite, integers, floats
import pandas as pd
import numpy as np
from pytest import mark, raises
from tests.conftest import model_set_up


@composite
def likelihood_data(draw):
    n_data_points = draw(integers(min_value = 20, max_value = 100))
    mean_minimum = draw(floats(min_value = -1000, max_value = 0))
    mean_maximum = draw(floats(min_value = 1, max_value = 1000))
    variance_minimum = draw(floats(min_value = 1e-10, max_value = 10))
    variance_maximum = draw(floats(min_value = 11, max_value = 1000))

    return pd.DataFrame({'mean': np.random.uniform(low = mean_minimum,
                                                   high = mean_maximum,
                                                   size = n_data_points),
                         'variance': np.random.uniform(low = variance_minimum,
                                                       high = variance_maximum,
                                                       size = n_data_points)})


@mark.model
class TestModelData:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_property(self, model_options):
        model = LinearModel()
        model.data = model_options['data']

        assert isinstance(model.data, pd.DataFrame)
        assert model.data.equals(model_options['data'])
        assert not model.data.empty


    @mark.error
    def test_raises_type_error(self, empty_model, model_data_type_error):
        with raises(TypeError):
            empty_model.data = model_data_type_error


    @mark.error
    def test_raises_value_error(self, empty_model):
        with raises(ValueError):
            empty_model.data = pd.DataFrame()


@mark.model
class TestModelResponseVariable:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_property(self, model_options):
        model = LinearModel()
        model.response_variable = model_options['response_variable']

        assert isinstance(model.response_variable, str)
        assert model.response_variable == model_options['response_variable']


    @mark.error
    def test_raises_type_error(self, empty_model, model_response_variable_type_error):
        with raises(TypeError):
            empty_model.response_variable = model_response_variable_type_error


@mark.model
class TestModelPriors:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_property(self, model_options):
        model = LinearModel()
        model.priors = model_options['priors']

        assert isinstance(model.priors, dict)
        assert len(model.priors) > 0
        assert model.priors == model_options['priors']
        assert 'intercept' in model.priors.keys()
        assert 'variance' in model.priors.keys()
        assert all(['mean' in regressor_data.keys() for regressor, regressor_data in model.priors.items()
                    if regressor != 'variance'])
        assert all(['variance' in regressor_data.keys() for regressor, regressor_data in model.priors.items()
                    if regressor != 'variance'])
        assert 'shape' in model.priors['variance'].keys()
        assert 'scale' in model.priors['variance'].keys()
        assert model.variable_names is not None
        assert model.variable_names[0] == 'intercept'
        assert 'variance' in model.variable_names


    @mark.error
    def test_raises_type_error(self, empty_model, model_priors_type_error):
        with raises(TypeError):
            empty_model.priors = model_priors_type_error


    @mark.error
    def test_raises_key_error(self, empty_model, model_priors_key_error):
        with raises(KeyError):
            empty_model.priors = model_priors_key_error


    @mark.error
    def test_raises_value_error(self, empty_model, model_priors_value_error):
        with raises(ValueError):
            empty_model.priors = model_priors_value_error


@mark.model
class TestModelPosteriors:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_property(self, model_options):
        model = LinearModel()
        model.posteriors = model_options['posteriors']

        assert isinstance(model.posteriors, dict)
        assert len(model.posteriors) > 0
        assert model.posteriors == model_options['posteriors']
        assert 'intercept' in model.posteriors.keys()
        assert 'variance' in model.posteriors.keys()
        assert all([posterior_samples.shape == (model_options['n_samples'], model_options['n_chains'])
                    for posterior_samples in model.posteriors.values()])
        assert (model.posteriors['variance'] > 0).all()


    @mark.error
    def test_raises_type_error(self, empty_model, model_posteriors_type_error):
        with raises(TypeError):
            empty_model.posteriors = model_posteriors_type_error


    @mark.error
    def test_raises_key_error(self, empty_model, model_posteriors_key_error):
        with raises(KeyError):
            empty_model.posteriors = model_posteriors_key_error


    @mark.error
    def test_raises_value_error(self, empty_model):
        with raises(ValueError):
            empty_model.posteriors = {'intercept': np.array([]), 'variance': np.array([0])}


@mark.model
class TestModelPosteriorsToFrame:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_method(self, model_options):
        model = LinearModel()
        model.posteriors = model_options['posteriors']
        posteriors_frame = model.posteriors_to_frame()

        assert isinstance(posteriors_frame, pd.DataFrame)
        assert not posteriors_frame.empty
        assert all(posteriors_frame.columns == list(model_options['posteriors'].keys()))
        assert len(posteriors_frame) == model_options['n_samples']*model_options['n_chains']
        for col in model.posteriors.keys():
            for i in range(model_options['n_samples']):
                for j in range(model_options['n_chains']):
                    assert model.posteriors[col][i, j] == posteriors_frame.loc[i*model_options['n_chains'] + j, col]


    @mark.error
    def test_raises_value_error(self, complete_model):
        with raises(ValueError):
            complete_model.posteriors_to_frame()


@mark.model
class TestModelResiduals:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_method(self, model_options):
        model = LinearModel()
        model.data = model_options['data']
        model.response_variable = model_options['response_variable']
        model.posteriors = model_options['posteriors']
        residuals = model.residuals()

        assert isinstance(residuals, pd.DataFrame)
        assert not residuals.empty
        assert 'predicted' in residuals.columns
        assert 'residuals' in residuals.columns
        assert len(residuals) == len(model_options['data'])
        cols = list(residuals.columns)
        cols.remove('intercept')
        cols.remove('predicted')
        cols.remove('residuals')
        assert set(cols) == set(model_options['data'].columns)


    @mark.error
    def test_raises_value_error(self, model_residuals_value_error):
        with raises(ValueError):
            model_residuals_value_error.residuals()


@mark.model
class TestModelPredictDistribution:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_method(self, model_options):
        if len(model_options['predictors']) != 0:
            model = LinearModel()
            model.posteriors = model_options['posteriors']
            predicted = model.predict_distribution(predictors = model_options['predictors'])

            assert isinstance(predicted, np.ndarray)
            assert predicted.size != 0
            assert len(predicted) == model_options['n_samples']*model_options['n_chains']


    @mark.error
    def test_raises_type_error(self, solved_model, model_predict_distribution_type_error):
        with raises(TypeError):
            solved_model.predict_distribution(predictors = model_predict_distribution_type_error)


    @mark.error
    def test_raises_key_error(self, empty_model):
        empty_model.posteriors = {'intercept': np.array([0]), 'variance': np.array([0])}
        predictors = {'x': 5}
        with raises(KeyError):
            empty_model.predict_distribution(predictors = predictors)


    @mark.error
    def test_raises_value_error(self, empty_model):
        with raises(ValueError):
            empty_model.predict_distribution(predictors = {})


@mark.model
class TestModelLikelihood:


    @mark.genuine
    @given(l_data = likelihood_data())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_method(self, l_data):
        model = LinearModel()
        model.response_variable = 'y'
        l_data['y'] = np.random.uniform(low = l_data['mean'].min(),
                                        high = l_data['mean'].max(),
                                        size = len(l_data))
        likelihood = model.likelihood(data = l_data)

        assert isinstance(likelihood, np.ndarray)
        assert len(likelihood) == len(l_data)


    @mark.error
    def test_raises_type_error(self, complete_model, model_likelihood_type_error):
        with raises(TypeError):
            complete_model.likelihood(data = model_likelihood_type_error)


    @mark.error
    def test_raises_value_error(self, complete_model, model_likelihood_value_error):
        with raises(ValueError):
            complete_model.likelihood(data = model_likelihood_value_error)


@mark.model
class TestModelLogLikelihood:


    @mark.genuine
    @given(l_data = likelihood_data())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_method(self, l_data):
        model = LinearModel()
        model.response_variable = 'y'
        l_data['y'] = np.random.uniform(low = l_data['mean'].min(),
                                        high = l_data['mean'].max(),
                                        size = len(l_data))
        log_likelihood = model.log_likelihood(data = l_data)

        assert isinstance(log_likelihood, np.ndarray)
        assert len(log_likelihood) == len(l_data)


    @mark.error
    def test_raises_type_error(self, complete_model, model_log_likelihood_type_error):
        with raises(TypeError):
            complete_model.log_likelihood(data = model_log_likelihood_type_error)


    @mark.error
    def test_raises_value_error(self, complete_model, model_log_likelihood_value_error):
        with raises(ValueError):
            complete_model.log_likelihood(data = model_log_likelihood_value_error)
