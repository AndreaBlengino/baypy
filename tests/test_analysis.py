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


    def test_raises_key_error(self, analysis_trace_plot_key_error):
        with raises(KeyError):
            gs.analysis.trace_plot(analysis_trace_plot_key_error)


    def test_raises_value_error(self):
        with raises(ValueError):
            gs.analysis.trace_plot(posteriors = {'intercept': np.array([]), 'variance': np.array([0])})


@mark.analysis
class TestAnalysisSummary:


    def test_method(self, posteriors, general_testing_data):
        gs.analysis.summary(posteriors)

        data_tmp = general_testing_data['data'].copy()
        data_tmp['intercept'] = 1
        linear_model_results = np.linalg.lstsq(a = data_tmp[general_testing_data['regressor_names']],
                                               b = data_tmp[general_testing_data['response_variable']],
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


    def test_raises_key_error(self, analysis_summary_key_error):
        with raises(KeyError):
            gs.analysis.summary(posteriors = analysis_summary_key_error['posteriors'],
                                alpha = analysis_summary_key_error['alpha'],
                                quantiles = analysis_summary_key_error['quantiles'])


    def test_raises_value_error(self, analysis_summary_value_error):
        with raises(ValueError):
            gs.analysis.summary(posteriors = analysis_summary_value_error['posteriors'],
                                alpha = analysis_summary_value_error['alpha'],
                                quantiles = analysis_summary_value_error['quantiles'])


@mark.analysis
class TestAnalysisResidualsPlot:


    def test_method(self, posteriors, general_testing_data, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.analysis.residuals_plot(posteriors = posteriors,
                                   data = general_testing_data['data'],
                                   response_variable = general_testing_data['response_variable'])


    def test_raises_type_error(self, analysis_residuals_plot_type_error):
        with raises(TypeError):
            gs.analysis.residuals_plot(posteriors = analysis_residuals_plot_type_error['posteriors'],
                                       data = analysis_residuals_plot_type_error['data'],
                                       response_variable = analysis_residuals_plot_type_error['response_variable'])

    def test_raises_key_error(self, analysis_residuals_plot_key_error):
        with raises(KeyError):
            gs.analysis.residuals_plot(posteriors = analysis_residuals_plot_key_error['posteriors'],
                                       data = analysis_residuals_plot_key_error['data'],
                                       response_variable = analysis_residuals_plot_key_error['response_variable'])


    def test_raises_value_error(self, analysis_residuals_plot_value_error):
        with raises(ValueError):
            gs.analysis.residuals_plot(posteriors = analysis_residuals_plot_value_error['posteriors'],
                                       data = analysis_residuals_plot_value_error['data'],
                                       response_variable = analysis_residuals_plot_value_error['response_variable'])


@mark.analysis
class TestAnalysisPredictDistribution:


    def test_method(self, posteriors, general_testing_data):
        predicted = gs.analysis.predict_distribution(posteriors = posteriors, predictors = general_testing_data['predictors'])

        lower_bound = np.quantile(predicted, general_testing_data['q_min'])
        upper_bound = np.quantile(predicted, general_testing_data['q_max'])

        data_tmp = general_testing_data['data'].copy()
        data_tmp['intercept'] = 1
        linear_model_results = np.linalg.lstsq(a = data_tmp[general_testing_data['regressor_names']],
                                               b = data_tmp[general_testing_data['response_variable']],
                                               rcond = None)[0]
        linear_model_prediction = linear_model_results[0] + np.dot(np.array(list(general_testing_data['predictors'].values())),
                                                                   linear_model_results[1:])

        assert lower_bound <= linear_model_prediction <= upper_bound

    def test_raises_type_error(self, analysis_predict_distribution_type_error):
        with raises(TypeError):
            gs.analysis.predict_distribution(posteriors = analysis_predict_distribution_type_error['posteriors'],
                                             predictors = analysis_predict_distribution_type_error['predictors'])

    def test_raises_key_error(self, analysis_predict_distribution_key_error):
        with raises(KeyError):
            gs.analysis.predict_distribution(posteriors = analysis_predict_distribution_key_error['posteriors'],
                                             predictors = analysis_predict_distribution_key_error['predictors'])


    def test_raises_value_error(self, analysis_predict_distribution_value_error):
        with raises(ValueError):
            gs.analysis.predict_distribution(posteriors = analysis_predict_distribution_value_error['posteriors'],
                                             predictors = analysis_predict_distribution_value_error['predictors'])


@mark.analysis
class TestAnalysisComputeDIC:


    def test_method(self, posteriors, general_testing_data):
        gs.analysis.compute_DIC(posteriors = posteriors,
                                data = general_testing_data['data'],
                                response_variable = general_testing_data['response_variable'])


    def test_raises_type_error(self, analysis_compute_dic_type_error):
        with raises(TypeError):
            gs.analysis.compute_DIC(posteriors = analysis_compute_dic_type_error['posteriors'],
                                    data = analysis_compute_dic_type_error['data'],
                                    response_variable = analysis_compute_dic_type_error['response_variable'])


    def test_raises_key_error(self, analysis_compute_dic_key_error):
        with raises(KeyError):
            gs.analysis.compute_DIC(posteriors = analysis_compute_dic_key_error['posteriors'],
                                    data = analysis_compute_dic_key_error['data'],
                                    response_variable = analysis_compute_dic_key_error['response_variable'])


    def test_raises_value_error(self, analysis_compute_dic_value_error):
        with raises(ValueError):
            gs.analysis.compute_DIC(posteriors = analysis_compute_dic_value_error['posteriors'],
                                    data = analysis_compute_dic_value_error['data'],
                                    response_variable = analysis_compute_dic_value_error['response_variable'])
