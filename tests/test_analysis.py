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

q_min = 0.025
q_max = 0.975

prediction_data = {'x_1': 20, 'x_2': 5, 'x_3': -45}
prediction_data['x_1 * x_2'] = prediction_data['x_1']*prediction_data['x_2']


@mark.analysis
class TestAnalysis:

    def test_summary(self, posteriors):
        gs.analysis.summary(posteriors)

        data_tmp = data.copy()
        data_tmp['intercept'] = 1
        linear_model_results = np.linalg.lstsq(a = data_tmp[regressor_names],
                                               b = data_tmp['y'],
                                               rcond = None)[0]

        for i, regressor in enumerate(regressor_names, 0):
            lower_bound = np.quantile(np.asarray(posteriors[regressor]).reshape(-1), q_min)
            upper_bound = np.quantile(np.asarray(posteriors[regressor]).reshape(-1), q_max)

            assert lower_bound <= linear_model_results[i] <= upper_bound


    def test_trace_plot(self, posteriors, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.analysis.trace_plot(posteriors)


    def test_residuals_plot(self, posteriors, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.analysis.residuals_plot(posteriors = posteriors, data = data, y_name = 'y')


    def test_predict_distribution(self, posteriors):
        predicted = gs.analysis.predict_distribution(posteriors = posteriors, data = prediction_data)

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
