import baypy as bp
from hypothesis.strategies import composite, integers, floats, lists
import numpy as np
import pandas as pd
from pytest import fixture


np.random.seed(42)

N = 50
data = pd.DataFrame()
data['x_1'] = np.random.uniform(low = 0, high = 100, size = N)
data['x_2'] = np.random.uniform(low = -10, high = 10, size = N)
data['x_3'] = np.random.uniform(low = -50, high = -40, size = N)
data['x_1 * x_2'] = data['x_1']*data['x_2']

data['y'] = 3*data['x_1'] - 20*data['x_2'] - data['x_3'] - 5*data['x_1 * x_2'] + 13 + 1*np.random.randn(N)

response_variable = 'y'

variance_sample_size = 5
variance_of_variance = 10

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
          'variance': {'shape': variance_sample_size,
                       'scale': variance_sample_size*variance_of_variance}}


regressor_names = ['intercept', 'x_1', 'x_2', 'x_3', 'x_1 * x_2']

q_min = 0.025
q_max = 0.975

predictors = {'x_1': 20, 'x_2': 5, 'x_3': -45}
predictors['x_1 * x_2'] = predictors['x_1']*predictors['x_2']


types_to_check = ['string', 2, 2.2, True, (0, 1), [0, 1], {0, 1}, {0: 1}, None,
                  pd.DataFrame(columns = ['response_variable'], index = [0]), np.array([0])]


@composite
def model_set_up(draw):
    n_regressors = draw(integers(min_value = 0, max_value = 5))
    n_data_points = draw(integers(min_value = 20, max_value = 100))
    regressors_minimum = draw(floats(min_value = -1000, max_value = 0))
    regressors_maximum = draw(floats(min_value = 1, max_value = 1000))
    regressors_parameters_minimum = draw(floats(min_value = -100, max_value = -1))
    regressors_parameters_maximum = draw(floats(min_value = 1, max_value = 100))
    n_samples = draw(integers(min_value = 20, max_value = 1000))
    burn_in_iterations = draw(integers(min_value = 1, max_value = 100))
    n_chains = draw(integers(min_value = 1, max_value = 5))
    seed = draw(integers(min_value = 1, max_value = 1000))
    quantiles = draw(lists(integers(min_value = 0, max_value = 100), min_size = 1, max_size = 10, unique = True))

    regressors_names = np.random.choice(list('abcdefghijklmnopqrstuvxyz'), n_regressors, replace = False).tolist()
    data = pd.DataFrame({regressor_name: np.random.uniform(low = regressors_minimum,
                                                           high = regressors_maximum,
                                                           size = n_data_points)
                         for regressor_name in regressors_names})
    data['intercept'] = data['intercept'] = np.ones(n_data_points)
    regressors_parameters = np.random.uniform(low = regressors_parameters_minimum,
                                              high = regressors_parameters_maximum,
                                              size = n_regressors + 1)
    response_variable = np.random.choice([name for name in list('abcdefghijklmnopqrstuvxyz')
                                          if name not in data.columns], 1).tolist()[0]
    data[response_variable] = (data*regressors_parameters).sum(axis = 1) + \
                              np.random.normal(loc = 0, scale = 1, size = n_data_points)
    data.drop(columns = ['intercept'], inplace = True)

    priors = {regressor_name: {'mean': 0, 'variance': 1e14} for regressor_name in regressors_names}
    priors['intercept'] = {'mean': 0, 'variance': 1e14}
    priors['variance'] = {'shape': 1, 'scale': 1e-14}

    posteriors = {posterior_name: np.random.randn(n_samples, n_chains) for posterior_name in priors.keys()}
    posteriors['variance'] = np.abs(posteriors['variance'])

    predictors = {predictor: np.random.uniform(low = regressors_parameters_minimum,
                                               high = regressors_parameters_maximum,
                                               size = 1)[0]
                  for predictor in priors.keys() if predictor not in ['intercept', 'variance']}

    return {'data': data,
            'response_variable': response_variable,
            'priors': priors,
            'posteriors': posteriors,
            'n_samples': n_samples,
            'burn_in_iterations': burn_in_iterations,
            'n_chains': n_chains,
            'predictors': predictors,
            'seed': seed,
            'q_min': 0.025,
            'q_max': 0.975,
            'quantiles': [quantile/100 for quantile in quantiles]}


@composite
def posteriors_data(draw):
    n_samples = draw(integers(min_value = 20, max_value = 1000))
    n_chains = draw(integers(min_value = 1, max_value = 5))
    n_posteriors = draw(integers(min_value = 1, max_value = 9))

    posterior_names = np.random.choice(list('abcdefghij'), n_posteriors, replace = False).tolist()
    posterior_names.append('intercept')
    return {'n_samples': n_samples,
            'n_chains': n_chains,
            'n_posteriors': n_posteriors,
            'posteriors': {posterior_name: np.random.randn(n_samples, n_chains) for posterior_name in posterior_names}}


@fixture(scope = 'session',
         params = [{'data': data,
                    'response_variable': response_variable,
                    'priors': priors,
                    'n_iterations': 1000,
                    'burn_in_iterations': 50,
                    'n_chains': 3,
                    'seed': 137,
                    'regressor_names': regressor_names,
                    'q_min': q_min,
                    'q_max': q_max,
                    'predictors': predictors},
                   {'data': data,
                    'response_variable': response_variable,
                    'priors': priors,
                    'n_iterations': 100,
                    'burn_in_iterations': 50,
                    'n_chains': 5,
                    'seed': 137,
                    'regressor_names': regressor_names,
                    'q_min': q_min,
                    'q_max': q_max,
                    'predictors': predictors},
                   {'data': data,
                    'response_variable': response_variable,
                    'priors': priors,
                    'n_iterations': 1000,
                    'burn_in_iterations': 50,
                    'n_chains': 1,
                    'seed': 137,
                    'regressor_names': regressor_names,
                    'q_min': q_min,
                    'q_max': q_max,
                    'predictors': predictors},
                   {'data': data,
                    'response_variable': response_variable,
                    'priors': priors,
                    'n_iterations': 1000,
                    'burn_in_iterations': 1,
                    'n_chains': 3,
                    'seed': 137,
                    'regressor_names': regressor_names,
                    'q_min': q_min,
                    'q_max': q_max,
                    'predictors': predictors}])
def general_testing_data(request):
    return request.param


@fixture(scope = 'session')
def empty_model():
    model = bp.model.LinearModel()
    return model


@fixture(params = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, pd.DataFrame)])
def model_data_type_error(request):
    return request.param


@fixture(params = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, str)])
def model_response_variable_type_error(request):
    return request.param


model_priors_type_error_1 = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, dict)]
model_priors_type_error_2 = [{'intercept': type_to_check, 'variance': type_to_check} for type_to_check in types_to_check
                             if not isinstance(type_to_check, dict)]

@fixture(params = [*model_priors_type_error_1,
                   *model_priors_type_error_2])
def model_priors_type_error(request):
    return request.param


@fixture(params = [{'regressor': 1, 'variance': 2},
                   {'intercept': 1, 'regressor': 2},
                   {'intercept': {'m': 0, 'variance': 1}, 'variance': {'shape': 1, 'scale': 1}},
                   {'intercept': {'mean': 0, 'v': 1}, 'variance': {'shape': 1, 'scale': 1}},
                   {'intercept': {'mean': 0, 'variance': 1}, 'variance': {'s': 1, 'scale': 1}},
                   {'intercept': {'mean': 0, 'variance': 1}, 'variance': {'shape': 1, 's': 1}}])
def model_priors_key_error(request):
    return request.param


@fixture(params = [{},
                   {'intercept': {}, 'variance': {}},
                   {'intercept': {'mean': 0, 'variance': -1e6}, 'variance': {'shape': 1, 'scale': 1e-6}},
                   {'intercept': {'mean': 0, 'variance': 1e6}, 'variance': {'shape': -1, 'scale': 1e-6}},
                   {'intercept': {'mean': 0, 'variance': 1e6}, 'variance': {'shape': 1, 'scale': -1e-6}}])
def model_priors_value_error(request):
    return request.param


model_posteriors_type_error_1 = [type_to_check for type_to_check in types_to_check
                                 if not isinstance(type_to_check, dict)]
model_posteriors_type_error_2 = [{'intercept': type_to_check} for type_to_check in types_to_check
                                 if not isinstance(type_to_check, np.ndarray)]
@fixture(params = [*model_posteriors_type_error_1,
                   *model_posteriors_type_error_2])
def model_posteriors_type_error(request):
    return request.param


@fixture(params = [{'intercept': np.array([0])},
                   {'variance': np.array([0])}])
def model_posteriors_key_error(request):
    return request.param


model_residuals_value_error_1 = bp.model.LinearModel()

model_residuals_value_error_2 = bp.model.LinearModel()
model_residuals_value_error_2.data = pd.DataFrame(columns = ['not_response_variable'], index = [0])
model_residuals_value_error_2.response_variable = 'response_variable'

model_residuals_value_error_3 = bp.model.LinearModel()
model_residuals_value_error_3.data = pd.DataFrame(columns = ['response_variable'], index = [0])
model_residuals_value_error_3.response_variable = 'response_variable'

@fixture(params = [model_residuals_value_error_1,
                   model_residuals_value_error_2,
                   model_residuals_value_error_3])
def model_residuals_value_error(request):
    return request.param


@fixture(params = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, dict)])
def model_predict_distribution_type_error(request):
    return request.param


@fixture(params = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, pd.DataFrame)])
def model_likelihood_type_error(request):
    return request.param


@fixture(params = [pd.DataFrame(),
                   pd.DataFrame({'mean': [1, 2, 3], 'variance': [1, 2, 3]}),
                   pd.DataFrame({response_variable: [1, 2, 3], 'variance': [1, 2, 3]}),
                   pd.DataFrame({response_variable: [1, 2, 3], 'mean': [1, 2, 3]})])
def model_likelihood_value_error(request):
    return request.param


@fixture(params = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, pd.DataFrame)])
def model_log_likelihood_type_error(request):
    return request.param


@fixture(params = [pd.DataFrame(),
                   pd.DataFrame({'mean': [1, 2, 3], 'variance': [1, 2, 3]}),
                   pd.DataFrame({response_variable: [1, 2, 3], 'variance': [1, 2, 3]}),
                   pd.DataFrame({response_variable: [1, 2, 3], 'mean': [1, 2, 3]})])
def model_log_likelihood_value_error(request):
    return request.param


@fixture(scope = 'session')
def complete_model(general_testing_data):
    complete_model = bp.model.LinearModel()
    complete_model.data = general_testing_data['data']
    complete_model.response_variable = general_testing_data['response_variable']
    complete_model.priors = general_testing_data['priors']
    return complete_model


@fixture(params = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, bp.model.Model)])
def linear_regression_init_type_error(request):
    return request.param


linear_regression_init_value_error_1 = bp.model.LinearModel()
linear_regression_init_value_error_1.priors = {'x': {'mean': 0,
                                                     'variance': 1},
                                               'intercept': {'mean': 0,
                                                             'variance': 1},
                                               'variance': {'shape': 1,
                                                            'scale': 1}}

linear_regression_init_value_error_2 = bp.model.LinearModel()
linear_regression_init_value_error_2.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
linear_regression_init_value_error_2.priors = {'x': {'mean': 0,
                                                     'variance': 1},
                                               'y': {'mean': 0,
                                                     'variance': 1},
                                               'intercept': {'mean': 0,
                                                             'variance': 1},
                                               'variance': {'shape': 1,
                                                            'scale': 1}}

linear_regression_init_value_error_3 = bp.model.LinearModel()
linear_regression_init_value_error_3.data = pd.DataFrame(columns = ['x', 'z'], index = [0])
linear_regression_init_value_error_3.response_variable = 'z'

linear_regression_init_value_error_4 = bp.model.LinearModel()
linear_regression_init_value_error_4.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
linear_regression_init_value_error_4.response_variable = 'w'
linear_regression_init_value_error_4.priors = {'x': {'mean': 0,
                                                     'variance': 1},
                                               'y': {'mean': 0,
                                                     'variance': 1},
                                               'intercept': {'mean': 0,
                                                             'variance': 1},
                                               'variance': {'shape': 1,
                                                            'scale': 1}}

linear_regression_init_value_error_5 = bp.model.LinearModel()
linear_regression_init_value_error_5.data = pd.DataFrame(columns = ['x', 'y', 'z'], index = [0])
linear_regression_init_value_error_5.response_variable = 'z'
linear_regression_init_value_error_5.priors = {'x': {'mean': 0,
                                                     'variance': 1},
                                               'y': {'mean': 0,
                                                     'variance': 1},
                                               'w': {'mean': 0,
                                                     'variance': 1},
                                               'intercept': {'mean': 0,
                                                             'variance': 1},
                                               'variance': {'shape': 1,
                                                            'scale': 1}}

@fixture(params = [linear_regression_init_value_error_1,
                   linear_regression_init_value_error_2,
                   linear_regression_init_value_error_3,
                   linear_regression_init_value_error_4,
                   linear_regression_init_value_error_5])
def linear_regression_init_value_error(request):
    return request.param


linear_regression_sample_type_error_1 = [{'n_iterations': type_to_check, 'burn_in_iterations': 50,
                                          'n_chains': 3, 'seed': 137} for type_to_check in types_to_check
                                         if not isinstance(type_to_check, int)]

linear_regression_sample_type_error_2 = [{'n_iterations': 100, 'burn_in_iterations': type_to_check,
                                          'n_chains': 3, 'seed': 137} for type_to_check in types_to_check
                                         if not isinstance(type_to_check, int)]

linear_regression_sample_type_error_3 = [{'n_iterations': 100, 'burn_in_iterations': 50,
                                          'n_chains': type_to_check, 'seed': 137} for type_to_check in types_to_check
                                         if not isinstance(type_to_check, int)]

linear_regression_sample_type_error_4 = [{'n_iterations': 100, 'burn_in_iterations': 50,
                                          'n_chains': 3, 'seed': type_to_check} for type_to_check in types_to_check
                                         if not isinstance(type_to_check, int) and type_to_check is not None]

@fixture(params = [*linear_regression_sample_type_error_1,
                   *linear_regression_sample_type_error_2,
                   *linear_regression_sample_type_error_3,
                   *linear_regression_sample_type_error_4])
def linear_regression_sample_type_error(request):
    return request.param


@fixture(params = [{'n_iterations': -1, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 1000, 'burn_in_iterations': -1, 'n_chains': 3, 'seed': 137},
                   {'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': -1, 'seed': 137},
                   {'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': -1},
                   {'n_iterations': 1000, 'burn_in_iterations': 50, 'n_chains': 3, 'seed': 2**32}])
def linear_regression_sample_value_error(request):
    return request.param


diagnostics_autocorrelation_plot_type_error_1 = [{'posteriors': type_to_check, 'max_lags': 30}
                                                 for type_to_check in types_to_check
                                                 if not isinstance(type_to_check, dict)]

diagnostics_autocorrelation_plot_type_error_2 = [{'posteriors': {'intercept': type_to_check}, 'max_lags': 30}
                                                 for type_to_check in types_to_check
                                                 if not isinstance(type_to_check, np.ndarray)]

diagnostics_autocorrelation_plot_type_error_3 = [{'posteriors': {'intercept': np.array([0])},
                                                  'max_lags': type_to_check}
                                                 for type_to_check in types_to_check
                                                 if not isinstance(type_to_check, int)]

@fixture(params = [*diagnostics_autocorrelation_plot_type_error_1,
                   *diagnostics_autocorrelation_plot_type_error_2,
                   *diagnostics_autocorrelation_plot_type_error_3])
def diagnostics_autocorrelation_plot_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([])}, 'max_lags': 30},
                   {'posteriors': {'intercept': np.array([0])}, 'max_lags': -1}])
def diagnostics_autocorrelation_plot_value_error(request):
    return request.param


diagnostics_autocorrelation_summary_type_error_1 = [{'posteriors': type_to_check,
                                                     'lags': [0, 1], 'print_summary': False}
                                                    for type_to_check in types_to_check
                                                    if not isinstance(type_to_check, dict)]

diagnostics_autocorrelation_summary_type_error_2 = [{'posteriors': {'intercept': type_to_check},
                                                     'lags': [0, 1], 'print_summary': False}
                                                    for type_to_check in types_to_check
                                                    if not isinstance(type_to_check, np.ndarray)]

diagnostics_autocorrelation_summary_type_error_3 = [{'posteriors': {'intercept': np.array([0])},
                                                     'lags': type_to_check, 'print_summary': False}
                                                    for type_to_check in types_to_check
                                                    if not isinstance(type_to_check, list) and
                                                    type_to_check is not None]

diagnostics_autocorrelation_summary_type_error_4 = [{'posteriors': {'intercept': np.array([0])},
                                                     'lags': [type_to_check, type_to_check], 'print_summary': False}
                                                    for type_to_check in types_to_check
                                                    if not isinstance(type_to_check, int)]

diagnostics_autocorrelation_summary_type_error_5 = [{'posteriors': {'intercept': np.array([0])},
                                                     'lags': [0, 1], 'print_summary': type_to_check}
                                                    for type_to_check in types_to_check
                                                    if not isinstance(type_to_check, bool)]


@fixture(params = [*diagnostics_autocorrelation_summary_type_error_1,
                   *diagnostics_autocorrelation_summary_type_error_2,
                   *diagnostics_autocorrelation_summary_type_error_3,
                   *diagnostics_autocorrelation_summary_type_error_4,
                   *diagnostics_autocorrelation_summary_type_error_5])
def diagnostics_autocorrelation_summary_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([])}, 'lags': [0, 1, 5, 10, 30]},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': []},
                   {'posteriors': {'intercept': np.array([0])}, 'lags': [-1]}])
def diagnostics_autocorrelation_summary_value_error(request):
    return request.param


diagnostics_effective_sample_size_type_error_1 = [{'posteriors': type_to_check, 'print_summary': False}
                                                  for type_to_check in types_to_check
                                                  if not isinstance(type_to_check, dict)]

diagnostics_effective_sample_size_type_error_2 = [{'posteriors': {'intercept': type_to_check}, 'print_summary': False}
                                                  for type_to_check in types_to_check
                                                  if not isinstance(type_to_check, np.ndarray)]

diagnostics_effective_sample_size_type_error_3 = [{'posteriors': {'intercept': np.array([0])},
                                                   'print_summary': type_to_check}
                                                  for type_to_check in types_to_check
                                                  if not isinstance(type_to_check, bool)]

@fixture(params = [*diagnostics_effective_sample_size_type_error_1,
                   *diagnostics_effective_sample_size_type_error_2,
                   *diagnostics_effective_sample_size_type_error_3])
def diagnostics_effective_sample_size_type_error(request):
    return request.param


analysis_trace_plot_type_error_1 = [type_to_check for type_to_check in types_to_check
                                    if not isinstance(type_to_check, dict)]

analysis_trace_plot_type_error_2 = [{'intercept': type_to_check}
                                    for type_to_check in types_to_check
                                    if not isinstance(type_to_check, np.ndarray)]

@fixture(params = [*analysis_trace_plot_type_error_1,
                   *analysis_trace_plot_type_error_2])
def analysis_trace_plot_type_error(request):
    return request.param


analysis_summary_type_error_1 = [{'posteriors': type_to_check, 'alpha': 0.05,
                                  'quantiles': [0.1, 0.9], 'print_summary': False}
                                 for type_to_check in types_to_check if not isinstance(type_to_check, dict)]

analysis_summary_type_error_2 = [{'posteriors': {'intercept': type_to_check}, 'alpha': 0.05,
                                  'quantiles': [0.1, 0.9], 'print_summary': False}
                                 for type_to_check in types_to_check if not isinstance(type_to_check, np.ndarray)]

analysis_summary_type_error_3 = [{'posteriors': {'intercept': np.array([0])}, 'alpha': type_to_check,
                                  'quantiles': [0.1, 0.9], 'print_summary': False}
                                 for type_to_check in types_to_check
                                 if not isinstance(type_to_check, float) and not isinstance(type_to_check, bool)]

analysis_summary_type_error_4 = [{'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05,
                                  'quantiles': type_to_check, 'print_summary': False}
                                 for type_to_check in types_to_check if not isinstance(type_to_check, list) and type_to_check is not None]

analysis_summary_type_error_5 = [{'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05,
                                  'quantiles': [type_to_check, type_to_check], 'print_summary': False}
                                 for type_to_check in types_to_check if not isinstance(type_to_check, float)]

analysis_summary_type_error_6 = [{'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05,
                                  'quantiles': [0.1, 0.9], 'print_summary': type_to_check}
                                 for type_to_check in types_to_check if not isinstance(type_to_check, bool)]

@fixture(params = [*analysis_summary_type_error_1,
                   *analysis_summary_type_error_2,
                   *analysis_summary_type_error_3,
                   *analysis_summary_type_error_4,
                   *analysis_summary_type_error_5,
                   *analysis_summary_type_error_6])
def analysis_summary_type_error(request):
    return request.param


@fixture(params = [{'posteriors': {'intercept': np.array([])}, 'alpha': 0.05, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': -0.5, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 1.5, 'quantiles': [0.1, 0.9]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': []},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [-0.5]},
                   {'posteriors': {'intercept': np.array([0])}, 'alpha': 0.05, 'quantiles': [1.5]}])
def analysis_summary_value_error(request):
    return request.param


@fixture(params = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, bp.model.Model)])
def analysis_residuals_plot_type_error(request):
    return request.param


analysis_residuals_plot_value_error_1 = bp.model.LinearModel()
analysis_residuals_plot_value_error_1.data = pd.DataFrame(columns = ['response_variable'], index = [0])
analysis_residuals_plot_value_error_1.response_variable = 'response_variable'

analysis_residuals_plot_value_error_2 = bp.model.LinearModel()
analysis_residuals_plot_value_error_2.data = pd.DataFrame(columns = ['response_variable'], index = [0])
analysis_residuals_plot_value_error_2.response_variable = 'response_variable'
analysis_residuals_plot_value_error_2.posteriors = {'intercept': np.array([0]),
                                                    'variance': np.array([0]),
                                                    'x': np.array([0])}

analysis_residuals_plot_value_error_3 = bp.model.LinearModel()
analysis_residuals_plot_value_error_3.data = pd.DataFrame(columns = ['not_response_variable'], index = [0])
analysis_residuals_plot_value_error_3.data.drop(index = [0], inplace = True)
analysis_residuals_plot_value_error_3.response_variable = 'response_variable'
analysis_residuals_plot_value_error_3.posteriors = {'intercept': np.array([0]),
                                                    'variance': np.array([0])}

analysis_residuals_plot_value_error_4 = bp.model.LinearModel()
analysis_residuals_plot_value_error_4.data = pd.DataFrame(columns = ['not_response_variable'], index = [0])
analysis_residuals_plot_value_error_4.response_variable = 'response_variable'
analysis_residuals_plot_value_error_4.posteriors = {'intercept': np.array([0]),
                                                    'variance': np.array([0])}

@fixture(params = [analysis_residuals_plot_value_error_1,
                   analysis_residuals_plot_value_error_2,
                   analysis_residuals_plot_value_error_3,
                   analysis_residuals_plot_value_error_4])
def analysis_residuals_plot_value_error(request):
    return request.param


analysis_compute_dic_type_error_1 = [{'model': type_to_check, 'print_summary': False}
                                     for type_to_check in types_to_check
                                     if not isinstance(type_to_check, bp.model.Model)]

analysis_compute_dic_type_error_2 = [{'model': bp.model.LinearModel(),
                                      'print_summary': type_to_check} for type_to_check in types_to_check
                                     if not isinstance(type_to_check, bool)]
for args in analysis_compute_dic_type_error_2:
    args['model'].data = pd.DataFrame(columns = ['response_variable'], index = [0])
    args['model'].response_variable = 'response_variable'
    args['model'].posteriors = {'intercept': np.array([0]), 'variance': np.array([0])}

@fixture(params = [*analysis_compute_dic_type_error_1,
                   *analysis_compute_dic_type_error_2])
def analysis_compute_dic_type_error(request):
    return request.param


analysis_compute_dic_value_error_value_error_1 = bp.model.LinearModel()
analysis_compute_dic_value_error_value_error_1.data = pd.DataFrame(columns = ['response_variable'], index = [0])
analysis_compute_dic_value_error_value_error_1.response_variable = 'response_variable'

analysis_compute_dic_value_error_value_error_2 = bp.model.LinearModel()
analysis_compute_dic_value_error_value_error_2.data = pd.DataFrame(columns = ['response_variable'], index = [0])
analysis_compute_dic_value_error_value_error_2.response_variable = 'response_variable'
analysis_compute_dic_value_error_value_error_2.posteriors = {'intercept': np.array([0]),
                                                             'variance': np.array([0]),
                                                             'x': np.array([0])}

analysis_compute_dic_value_error_value_error_3 = bp.model.LinearModel()
analysis_compute_dic_value_error_value_error_3.data = pd.DataFrame(columns = ['not_response_variable'], index = [0])
analysis_compute_dic_value_error_value_error_3.data.drop(index = [0], inplace = True)
analysis_compute_dic_value_error_value_error_3.response_variable = 'response_variable'
analysis_compute_dic_value_error_value_error_3.posteriors = {'intercept': np.array([0]),
                                                             'variance': np.array([0])}

analysis_compute_dic_value_error_value_error_4 = bp.model.LinearModel()
analysis_compute_dic_value_error_value_error_4.data = pd.DataFrame(columns = ['not_response_variable'], index = [0])
analysis_compute_dic_value_error_value_error_4.response_variable = 'response_variable'
analysis_compute_dic_value_error_value_error_4.posteriors = {'intercept': np.array([0]),
                                                             'variance': np.array([0])}

@fixture(params = [analysis_compute_dic_value_error_value_error_1,
                   analysis_compute_dic_value_error_value_error_2,
                   analysis_compute_dic_value_error_value_error_3,
                   analysis_compute_dic_value_error_value_error_4])
def analysis_compute_dic_value_error(request):
    return request.param


@fixture(scope = 'session')
def solved_model(general_testing_data):
    model = bp.model.LinearModel()
    model.data = general_testing_data['data']
    model.response_variable = general_testing_data['response_variable']
    model.priors = general_testing_data['priors']
    sampler = bp.regression.LinearRegression(model = model)
    sampler.sample(n_iterations = general_testing_data['n_iterations'],
                   burn_in_iterations = general_testing_data['burn_in_iterations'],
                   n_chains = general_testing_data['n_chains'])
    return model


@fixture(params = [type_to_check for type_to_check in types_to_check if not isinstance(type_to_check, np.ndarray)])
def utils_flatten_matrix_type_error(request):
    return request.param


utils_matrices_to_frame_type_error_1 = [type_to_check for type_to_check in types_to_check
                                        if not isinstance(type_to_check, dict)]

utils_matrices_to_frame_type_error_2 = [{'a': type_to_check} for type_to_check in types_to_check
                                        if not isinstance(type_to_check, np.ndarray)]

@fixture(params = [*utils_matrices_to_frame_type_error_1,
                   *utils_matrices_to_frame_type_error_2])
def utils_matrices_to_frame_type_error(request):
    return request.param


utils_dot_product_type_error_1 = [{'data': type_to_check, 'regressors': {}}
                                  for type_to_check in types_to_check if not isinstance(type_to_check, pd.DataFrame)]

utils_dot_product_type_error_2 = [{'data': pd.DataFrame(columns = ['a', 'b'], index = [0]), 'regressors': type_to_check}
                                  for type_to_check in types_to_check if not isinstance(type_to_check, dict)]

utils_dot_product_type_error_3 = [{'data': pd.DataFrame(columns = ['a', 'b'], index = [0]),
                                   'regressors': {'a': type_to_check}} for type_to_check in types_to_check
                                  if not isinstance(type_to_check, int) and not isinstance(type_to_check, float)]

@fixture(params = [*utils_dot_product_type_error_1,
                   *utils_dot_product_type_error_2,
                   *utils_dot_product_type_error_3])
def utils_dot_product_type_error(request):
    return request.param


@fixture(params = [{'data': pd.DataFrame(), 'regressors': {'a': 1, 'b': 2}},
                   {'data': pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), 'regressors': {}}])
def utils_dot_product_value_error(request):
    return request.param
