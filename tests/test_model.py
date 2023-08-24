import pandas as pd
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
        assert all(['mean' in regressor_data.keys() for regressor, regressor_data in empty_model.priors.items() if regressor != 'variance'])
        assert all(['variance' in regressor_data.keys() for regressor, regressor_data in empty_model.priors.items() if regressor != 'variance'])
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
