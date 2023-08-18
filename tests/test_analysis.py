import GibbsSampler as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytest import mark


np.random.seed(42)

N = 50
data = pd.DataFrame()
data['x_1'] = np.random.uniform(low = 0, high = 100, size = N)
data['x_2'] = np.random.uniform(low = -10, high = 10, size = N)
data['x_3'] = np.random.uniform(low = -50, high = -40, size = N)
data['x_1 * x_2'] = data['x_1']*data['x_2']

data['y'] = 3*data['x_1'] - 20*data['x_2'] - data['x_3'] - 5*data['x_1 * x_2'] + 13 + 1*np.random.randn(N)

regressor_names = ['intercept', 'x_1', 'x_2', 'x_3', 'x_1 * x_2']

initial_values = {'x_1': 1,
                  'x_2': 2,
                  'x_3': 3,
                  'x_1 * x_2': 4,
                  'intercept': 5}

sigma2_sample_size = 5
sigma2_variance = 10

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
          'sigma2': {'shape': sigma2_sample_size,
                     'scale': sigma2_sample_size*sigma2_variance}}

n_iterations = 1000
burn_in_iterations = 50
n_chains = 3

q_min = 0.025
q_max = 0.975

prediction_data = {'x_1': 20, 'x_2': 5, 'x_3': -45}
prediction_data['x_1 * x_2'] = prediction_data['x_1']*prediction_data['x_2']


@mark.analysis
class TestAnalysis:

    def test_summary(self, model, sampler):
        model.set_data(data = data,
                       y_name = 'y')
        model.set_initial_values(values = initial_values)
        model.set_priors(priors = priors)

        sampler.sample(n_iterations = n_iterations,
                       burn_in_iterations = burn_in_iterations,
                       n_chains = n_chains)

        gs.analysis.summary(sampler.posteriors)

        data_tmp = data.copy()
        data_tmp['intercept'] = 1
        linear_model_results = np.linalg.lstsq(a = data_tmp[regressor_names],
                                               b = data_tmp['y'],
                                               rcond = None)[0]

        for i, regressor in enumerate(regressor_names, 0):
            lower_bound = np.quantile(np.asarray(sampler.posteriors[regressor]).reshape(-1), q_min)
            upper_bound = np.quantile(np.asarray(sampler.posteriors[regressor]).reshape(-1), q_max)

            assert lower_bound <= linear_model_results[i] <= upper_bound


    def test_trace_plot(self, model, sampler, monkeypatch):
        model.set_data(data = data,
                       y_name = 'y')
        model.set_initial_values(values = initial_values)
        model.set_priors(priors = priors)

        sampler.sample(n_iterations = n_iterations,
                       burn_in_iterations = burn_in_iterations,
                       n_chains = n_chains)

        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.analysis.trace_plot(sampler.posteriors)


    def test_residuals_plot(self, model, sampler, monkeypatch):
        model.set_data(data = data,
                       y_name = 'y')
        model.set_initial_values(values = initial_values)
        model.set_priors(priors = priors)

        sampler.sample(n_iterations = n_iterations,
                       burn_in_iterations = burn_in_iterations,
                       n_chains = n_chains)

        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.analysis.residuals_plot(posteriors = sampler.posteriors,
                                   data = data,
                                   y_name = 'y')


    def test_predict_distribution(self, model, sampler):
        model.set_data(data = data,
                       y_name = 'y')
        model.set_initial_values(values = initial_values)
        model.set_priors(priors = priors)

        sampler.sample(n_iterations = n_iterations,
                       burn_in_iterations = burn_in_iterations,
                       n_chains = n_chains)

        predicted = gs.analysis.predict_distribution(posteriors = sampler.posteriors,
                                                     data = prediction_data)

        lower_bound = np.quantile(predicted, q_min)
        upper_bound = np.quantile(predicted, q_max)

        data_tmp = data.copy()
        data_tmp['intercept'] = 1
        linear_model_results = np.linalg.lstsq(a = data_tmp[regressor_names],
                                               b = data_tmp['y'],
                                               rcond = None)[0]
        linear_model_prediction = linear_model_results[0] + np.dot(np.array(list(prediction_data.values())),
                                                                   linear_model_results[1:])

        assert lower_bound <= linear_model_prediction <= upper_bound
