from pytest import mark, raises


@mark.model
class TestModelSetData:


    def test_method(self, empty_model, general_testing_data):
        empty_model.set_data(data = general_testing_data['data'],
                             response_variable = general_testing_data['response_variable'])

        assert empty_model.data.equals(general_testing_data['data'])
        assert empty_model.response_variable == general_testing_data['response_variable']


    def test_raises_type_error(self, empty_model, model_set_data_type_error):
        with raises(TypeError):
            empty_model.set_data(data = model_set_data_type_error['data'],
                                 response_variable = model_set_data_type_error['response_variable'])


    def test_raises_value_error(self, empty_model, model_set_data_value_error):
        with raises(ValueError):
            empty_model.set_data(data = model_set_data_value_error['data'],
                                 response_variable = model_set_data_value_error['response_variable'])


@mark.model
class TestModelSetInitialValues:


    def test_method(self, empty_model, general_testing_data):
        empty_model.set_initial_values(values = general_testing_data['initial_values'])

        assert empty_model.initial_values == general_testing_data['initial_values']
        assert 'intercept' in empty_model.initial_values.keys()


    def test_raises_type_error(self, empty_model, model_set_initial_value_type_error):
        with raises(TypeError):
            empty_model.set_initial_values(model_set_initial_value_type_error)


    def test_raises_key_error(self, empty_model):
        with raises(KeyError):
            empty_model.set_initial_values({'regressor': 1})


    def test_raises_value_error(self, empty_model):
        with raises(ValueError):
            empty_model.set_initial_values({})


@mark.model
class TestModelSetPriors:


    def test_method(self, empty_model, general_testing_data):
        empty_model.set_priors(priors = general_testing_data['priors'])

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


    def test_raises_type_error(self, empty_model, model_set_priors_type_error):
        with raises(TypeError):
            empty_model.set_priors(model_set_priors_type_error)


    def test_raises_key_error(self, empty_model, model_set_priors_key_error):
        with raises(KeyError):
            empty_model.set_priors(model_set_priors_key_error)


    def test_raises_value_error(self, empty_model, model_set_priors_value_error):
        with raises(ValueError):
            empty_model.set_priors(model_set_priors_value_error)
