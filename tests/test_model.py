import pandas as pd
import numpy as np
from pytest import mark, raises


@mark.model
class TestModelData:


    def test_property(self, empty_model, general_testing_data):
        empty_model.data = general_testing_data['data']

        assert empty_model.data.equals(general_testing_data['data'])


    def test_raises_type_error(self, empty_model, model_data_type_error):
        with raises(TypeError):
            empty_model.data = model_data_type_error


    def test_raises_value_error(self, empty_model):
        with raises(ValueError):
            empty_model.data = pd.DataFrame()


@mark.model
class TestModelResponseVariable:


    def test_property(self, empty_model, general_testing_data):
        empty_model.response_variable = general_testing_data['response_variable']

        assert empty_model.response_variable == general_testing_data['response_variable']


    def test_raises_type_error(self, empty_model, model_response_variable_type_error):
        with raises(TypeError):
            empty_model.response_variable = model_response_variable_type_error


@mark.model
class TestModelPriors:


    def test_property(self, empty_model, general_testing_data):
        empty_model.priors = general_testing_data['priors']

        assert empty_model.priors == general_testing_data['priors']
        assert 'intercept' in empty_model.priors.keys()
        assert 'variance' in empty_model.priors.keys()
        assert all(['mean' in regressor_data.keys() for regressor, regressor_data in empty_model.priors.items()
                    if regressor != 'variance'])
        assert all(['variance' in regressor_data.keys() for regressor, regressor_data in empty_model.priors.items()
                    if regressor != 'variance'])
        assert 'shape' in empty_model.priors['variance'].keys()
        assert 'scale' in empty_model.priors['variance'].keys()
        assert empty_model.variable_names is not None
        assert empty_model.variable_names[0] == 'intercept'
        assert 'variance' in empty_model.variable_names


    def test_raises_type_error(self, empty_model, model_priors_type_error):
        with raises(TypeError):
            empty_model.priors = model_priors_type_error


    def test_raises_key_error(self, empty_model, model_priors_key_error):
        with raises(KeyError):
            empty_model.priors = model_priors_key_error


    def test_raises_value_error(self, empty_model, model_priors_value_error):
        with raises(ValueError):
            empty_model.priors = model_priors_value_error


@mark.model
class TestModelPosteriors:


    def test_property(self, empty_model, general_testing_data):
        posteriors = {variable: np.zeros((general_testing_data['n_iterations'], general_testing_data['n_chains']))
                      for variable in general_testing_data['priors'].keys()}
        empty_model.posteriors = posteriors

        assert empty_model.posteriors == posteriors
        assert 'intercept' in empty_model.posteriors.keys()
        assert 'variance' in empty_model.posteriors.keys()
        assert all([posterior_samples.shape == (general_testing_data['n_iterations'], general_testing_data['n_chains'])
                    for posterior_samples in empty_model.posteriors.values()])


    def test_raises_type_error(self, empty_model, model_posteriors_type_error):
        with raises(TypeError):
            empty_model.posteriors = model_posteriors_type_error


    def test_raises_key_error(self, empty_model, model_posteriors_key_error):
        with raises(KeyError):
            empty_model.posteriors = model_posteriors_key_error


    def test_raises_value_error(self, empty_model):
        with raises(ValueError):
            empty_model.posteriors = {'intercept': np.array([]), 'variance': np.array([0])}


@mark.model
class TestModelPosteriorsToFrame:


    def test_method(self, empty_model, general_testing_data):
        posteriors = {variable: np.zeros((general_testing_data['n_iterations'], general_testing_data['n_chains']))
                      for variable in general_testing_data['priors'].keys()}
        empty_model.posteriors = posteriors
        posteriors_frame = empty_model.posteriors_to_frame()

        assert isinstance(posteriors_frame, pd.DataFrame)
        assert not posteriors_frame.empty
        assert all(posteriors_frame.columns == list(posteriors.keys()))
        assert len(posteriors_frame) == general_testing_data['n_iterations']*general_testing_data['n_chains']


    def test_raises_value_error(self, complete_model):
        with raises(ValueError):
            complete_model.posteriors_to_frame()


@mark.model
class TestModelResiduals:

    def test_method(self, solved_model, general_testing_data):
        residuals = solved_model.residuals()

        assert isinstance(residuals, pd.DataFrame)
        assert not residuals.empty
        assert 'predicted' in residuals.columns
        assert 'residuals' in residuals.columns
        assert len(residuals) == len(general_testing_data['data'])
        cols = list(residuals.columns)
        cols.remove('intercept')
        cols.remove('predicted')
        cols.remove('residuals')
        assert set(cols) == set(general_testing_data['data'].columns)


    def test_raises_value_error(self, model_residuals_value_error):
        with raises(ValueError):
            model_residuals_value_error.residuals()


@mark.model
class TestModelPredictDistribution:

    def test_method(self, solved_model, general_testing_data):
        predictors = {predictor: 1 for predictor in general_testing_data['priors'].keys()
                      if predictor not in ['intercept', 'variance']}
        predicted = solved_model.predict_distribution(predictors = predictors)

        assert isinstance(predicted, np.ndarray)
        assert predicted.size != 0
        assert len(predicted) == general_testing_data['n_iterations']*general_testing_data['n_chains']


    def test_raises_type_error(self, solved_model, model_predict_distribution_type_error):
        with raises(TypeError):
            solved_model.predict_distribution(predictors = model_predict_distribution_type_error)


    def test_raises_key_error(self, empty_model):
        empty_model.posteriors = {'intercept': np.array([0]), 'variance': np.array([0])}
        predictors = {'x': 5}
        with raises(KeyError):
            empty_model.predict_distribution(predictors = predictors)


    def test_raises_value_error(self, empty_model):
        with raises(ValueError):
            empty_model.predict_distribution(predictors = {})


@mark.model
class TestModelLikelihood:

    def test_method(self, complete_model):
        data = pd.DataFrame({complete_model.response_variable: [0, 1, 2, 3, 4],
                             'mean': [0, 1, 2, 3, 4],
                             'variance': [1, 2, 3, 4, 5]})
        likelihood = complete_model.likelihood(data = data)

        assert isinstance(likelihood, np.ndarray)
        assert len(likelihood) == len(data)


    def test_raises_type_error(self, complete_model, model_likelihood_type_error):
        with raises(TypeError):
            complete_model.likelihood(data = model_likelihood_type_error)


    def test_raises_value_error(self, complete_model, model_likelihood_value_error):
        with raises(ValueError):
            complete_model.likelihood(data = model_likelihood_value_error)


@mark.model
class TestModelLogLikelihood:

    def test_method(self, complete_model):
        data = pd.DataFrame({complete_model.response_variable: [0, 1, 2, 3, 4],
                             'mean': [0, 1, 2, 3, 4],
                             'variance': [1, 2, 3, 4, 5]})
        log_likelihood = complete_model.log_likelihood(data = data)

        assert isinstance(log_likelihood, np.ndarray)
        assert len(log_likelihood) == len(data)


    def test_raises_type_error(self, complete_model, model_log_likelihood_type_error):
        with raises(TypeError):
            complete_model.log_likelihood(data = model_log_likelihood_type_error)


    def test_raises_value_error(self, complete_model, model_log_likelihood_value_error):
        with raises(ValueError):
            complete_model.log_likelihood(data = model_log_likelihood_value_error)
