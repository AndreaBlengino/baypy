import GibbsSampler as gs
import matplotlib.pyplot as plt
import numpy as np
from pytest import mark, raises


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


    def test_method(self, posteriors, general_testing_data):
        gs.analysis.summary(posteriors)

        data_tmp = general_testing_data['data'].copy()
        data_tmp['intercept'] = 1
        linear_model_results = np.linalg.lstsq(a = data_tmp[general_testing_data['regressor_names']],
                                               b = data_tmp[general_testing_data['y_name']],
                                               rcond = None)[0]

        for i, regressor in enumerate(general_testing_data['regressor_names'], 0):
            lower_bound = np.quantile(np.asarray(posteriors[regressor]).reshape(-1), general_testing_data['q_min'])
            upper_bound = np.quantile(np.asarray(posteriors[regressor]).reshape(-1), general_testing_data['q_max'])

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


    def test_method(self, posteriors, general_testing_data, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.analysis.residuals_plot(posteriors = posteriors, data = general_testing_data['data'], y_name = general_testing_data['y_name'])


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


    def test_method(self, posteriors, general_testing_data):
        predicted = gs.analysis.predict_distribution(posteriors = posteriors, data = general_testing_data['prediction_data'])

        lower_bound = np.quantile(predicted, general_testing_data['q_min'])
        upper_bound = np.quantile(predicted, general_testing_data['q_max'])

        data_tmp = general_testing_data['data'].copy()
        data_tmp['intercept'] = 1
        linear_model_results = np.linalg.lstsq(a = data_tmp[general_testing_data['regressor_names']],
                                               b = data_tmp[general_testing_data['y_name']],
                                               rcond = None)[0]
        linear_model_prediction = linear_model_results[0] + np.dot(np.array(list(general_testing_data['prediction_data'].values())),
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
