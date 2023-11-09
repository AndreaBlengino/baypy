import baypy as bp
from baypy.regression import LinearRegression
from hypothesis.strategies import composite, integers, floats, lists
import numpy as np
import pandas as pd
from pytest import fixture


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


np.random.seed(42)

N = 50
data = pd.DataFrame()
data['x_1'] = np.random.uniform(low = 0, high = 100, size = N)
data['x_2'] = np.random.uniform(low = -10, high = 10, size = N)
data['x_3'] = np.random.uniform(low = -50, high = -40, size = N)
data['x_1 * x_2'] = data['x_1']*data['x_2']

response_variable = 'y'
data[response_variable] = 3*data['x_1'] - 20*data['x_2'] - data['x_3'] - 5*data['x_1 * x_2'] + 13 + 1*np.random.randn(N)

model = bp.model.LinearModel()
model.data = data
model.response_variable = response_variable
model.priors = priors = {'x_1': {'mean': 0,
                                 'variance': 1e6},
                         'x_2': {'mean': 0,
                                 'variance': 1e6},
                         'x_3': {'mean': 0,
                                 'variance': 1e6},
                         'x_1 * x_2': {'mean': 0,
                                       'variance': 1e6},
                         'intercept': {'mean': 0,
                                       'variance': 1e6},
                         'variance': {'shape': 1,
                                      'scale': 1e-6}}
LinearRegression.sample(model = model, n_iterations = 500, burn_in_iterations = 50, n_chains = 3)


types_to_check = ['string', 2, 2.2, True, (0, 1), [0, 1], {0, 1}, {0: 1}, None,
                  pd.DataFrame(columns = ['response_variable'], index = [0]), np.array([0])]
