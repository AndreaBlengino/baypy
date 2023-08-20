import GibbsSampler as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytest import mark, raises


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
class TestAnalysisTracePlot:


    def test_method(self, posteriors, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.analysis.trace_plot(posteriors)


    def test_raises_type_error(self, analysis_trace_plot_type_error):
        with raises(TypeError):
            gs.analysis.trace_plot(analysis_trace_plot_type_error)


    def test_raises_value_error(self, analysis_trace_plot_value_error):
        with raises(ValueError):
            gs.analysis.trace_plot(analysis_trace_plot_value_error)


@mark.analysis
class TestAnalysisSummary:


    def test_method(self, posteriors):
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


    def test_raises_type_error(self, analysis_summary_type_error):
        with raises(TypeError):
            gs.analysis.summary(posteriors = analysis_summary_type_error['posteriors'],
                                alpha = analysis_summary_type_error['alpha'],
                                quantiles = analysis_summary_type_error['quantiles'])


    def test_raises_value_error(self, analysis_summary_value_error):
        with raises(ValueError):
            gs.analysis.summary(posteriors = analysis_summary_value_error['posteriors'],
                                alpha = analysis_summary_value_error['alpha'],
                                quantiles = analysis_summary_value_error['quantiles'])


@mark.analysis
class TestAnalysisResidualsPlot:


    def test_method(self, posteriors, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.analysis.residuals_plot(posteriors = posteriors, data = data, y_name = 'y')


    def test_raises_type_error(self, analysis_residuals_plot_type_error):
        with raises(TypeError):
            gs.analysis.residuals_plot(posteriors = analysis_residuals_plot_type_error['posteriors'],
                                       data = analysis_residuals_plot_type_error['data'],
                                       y_name = analysis_residuals_plot_type_error['y_name'])

    def test_raises_value_error(self, analysis_residuals_plot_value_error):
        with raises(ValueError):
            gs.analysis.residuals_plot(posteriors = analysis_residuals_plot_value_error['posteriors'],
                                       data = analysis_residuals_plot_value_error['data'],
                                       y_name = analysis_residuals_plot_value_error['y_name'])


@mark.analysis
class TestAnalysisPredictDistribution:


    def test_method(self, posteriors):
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

    def test_raises_type_error(self, analysis_predict_distribution_type_error):
        with raises(TypeError):
            gs.analysis.predict_distribution(posteriors = analysis_predict_distribution_type_error['posteriors'],
                                             data = analysis_predict_distribution_type_error['data'])

    def test_raises_value_error(self, analysis_predict_distribution_value_error):
        with raises(ValueError):
            gs.analysis.predict_distribution(posteriors = analysis_predict_distribution_value_error['posteriors'],
                                             data = analysis_predict_distribution_value_error['data'])
