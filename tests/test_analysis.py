import baypy as bp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytest import mark, raises


@mark.analysis
class TestAnalysisTracePlot:


    def test_method(self, posteriors, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        bp.analysis.trace_plot(posteriors)


    def test_raises_type_error(self, analysis_trace_plot_type_error):
        with raises(TypeError):
            bp.analysis.trace_plot(analysis_trace_plot_type_error)


    def test_raises_key_error(self, analysis_trace_plot_key_error):
        with raises(KeyError):
            bp.analysis.trace_plot(analysis_trace_plot_key_error)


    def test_raises_value_error(self):
        with raises(ValueError):
            bp.analysis.trace_plot(posteriors = {'intercept': np.array([]), 'variance': np.array([0])})


@mark.analysis
class TestAnalysisSummary:


    def test_method(self, posteriors, general_testing_data):
        summary = bp.analysis.summary(posteriors)

        data_tmp = general_testing_data['data'].copy()
        data_tmp['intercept'] = 1
        linear_model_results = np.linalg.lstsq(a = data_tmp[general_testing_data['regressor_names']],
                                               b = data_tmp[general_testing_data['response_variable']],
                                               rcond = None)[0]

        for i, regressor in enumerate(general_testing_data['regressor_names'], 0):
            lower_bound = np.quantile(np.asarray(posteriors[regressor]).reshape(-1), general_testing_data['q_min'])
            upper_bound = np.quantile(np.asarray(posteriors[regressor]).reshape(-1), general_testing_data['q_max'])

            assert lower_bound <= linear_model_results[i] <= upper_bound

        assert summary['n_chains'] == general_testing_data['n_chains']
        assert summary['n_iterations'] == general_testing_data['n_iterations']
        assert isinstance(summary['summary'], pd.DataFrame)
        assert not summary['summary'].empty
        assert list(summary['summary'].index) == list(posteriors.keys())
        assert all(summary['summary'].columns == ['Mean', 'SD', 'HPD min', 'HPD max'])
        assert isinstance(summary['quantiles'], pd.DataFrame)
        assert not summary['quantiles'].empty
        assert list(summary['quantiles'].index) == list(posteriors.keys())
        assert all(summary['quantiles'].columns == ['2.5%', '25%', '50%', '75%', '97.5%'])


    def test_raises_type_error(self, analysis_summary_type_error):
        with raises(TypeError):
            bp.analysis.summary(posteriors = analysis_summary_type_error['posteriors'],
                                alpha = analysis_summary_type_error['alpha'],
                                quantiles = analysis_summary_type_error['quantiles'],
                                print_summary = analysis_summary_type_error['print_summary'])


    def test_raises_key_error(self, analysis_summary_key_error):
        with raises(KeyError):
            bp.analysis.summary(posteriors = analysis_summary_key_error['posteriors'],
                                alpha = analysis_summary_key_error['alpha'],
                                quantiles = analysis_summary_key_error['quantiles'])


    def test_raises_value_error(self, analysis_summary_value_error):
        with raises(ValueError):
            bp.analysis.summary(posteriors = analysis_summary_value_error['posteriors'],
                                alpha = analysis_summary_value_error['alpha'],
                                quantiles = analysis_summary_value_error['quantiles'])


@mark.analysis
class TestAnalysisResidualsPlot:


    def test_method(self, posteriors, general_testing_data, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        bp.analysis.residuals_plot(posteriors = posteriors,
                                   data = general_testing_data['data'],
                                   response_variable = general_testing_data['response_variable'])


    def test_raises_type_error(self, analysis_residuals_plot_type_error):
        with raises(TypeError):
            bp.analysis.residuals_plot(posteriors = analysis_residuals_plot_type_error['posteriors'],
                                       data = analysis_residuals_plot_type_error['data'],
                                       response_variable = analysis_residuals_plot_type_error['response_variable'])

    def test_raises_key_error(self, analysis_residuals_plot_key_error):
        with raises(KeyError):
            bp.analysis.residuals_plot(posteriors = analysis_residuals_plot_key_error['posteriors'],
                                       data = analysis_residuals_plot_key_error['data'],
                                       response_variable = analysis_residuals_plot_key_error['response_variable'])


    def test_raises_value_error(self, analysis_residuals_plot_value_error):
        with raises(ValueError):
            bp.analysis.residuals_plot(posteriors = analysis_residuals_plot_value_error['posteriors'],
                                       data = analysis_residuals_plot_value_error['data'],
                                       response_variable = analysis_residuals_plot_value_error['response_variable'])


@mark.analysis
class TestAnalysisPredictDistribution:


    def test_method(self, posteriors, general_testing_data):
        predicted = bp.analysis.predict_distribution(posteriors = posteriors, predictors = general_testing_data['predictors'])

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
            bp.analysis.predict_distribution(posteriors = analysis_predict_distribution_type_error['posteriors'],
                                             predictors = analysis_predict_distribution_type_error['predictors'])

    def test_raises_key_error(self, analysis_predict_distribution_key_error):
        with raises(KeyError):
            bp.analysis.predict_distribution(posteriors = analysis_predict_distribution_key_error['posteriors'],
                                             predictors = analysis_predict_distribution_key_error['predictors'])


    def test_raises_value_error(self, analysis_predict_distribution_value_error):
        with raises(ValueError):
            bp.analysis.predict_distribution(posteriors = analysis_predict_distribution_value_error['posteriors'],
                                             predictors = analysis_predict_distribution_value_error['predictors'])


@mark.analysis
class TestAnalysisComputeDIC:


    def test_method(self, posteriors, general_testing_data):
        summary = bp.analysis.compute_DIC(posteriors = posteriors,
                                          data = general_testing_data['data'],
                                          response_variable = general_testing_data['response_variable'])

        assert isinstance(summary, dict)
        assert list(summary.keys()) == ['deviance at posterior means', 'posterior mean deviance',
                                        'effective number of parameters', 'DIC']
        assert all([isinstance(value, float) for value in summary.values()])


    def test_raises_type_error(self, analysis_compute_dic_type_error):
        with raises(TypeError):
            bp.analysis.compute_DIC(posteriors = analysis_compute_dic_type_error['posteriors'],
                                    data = analysis_compute_dic_type_error['data'],
                                    response_variable = analysis_compute_dic_type_error['response_variable'],
                                    print_summary = analysis_compute_dic_type_error['print_summary'])


    def test_raises_key_error(self, analysis_compute_dic_key_error):
        with raises(KeyError):
            bp.analysis.compute_DIC(posteriors = analysis_compute_dic_key_error['posteriors'],
                                    data = analysis_compute_dic_key_error['data'],
                                    response_variable = analysis_compute_dic_key_error['response_variable'])


    def test_raises_value_error(self, analysis_compute_dic_value_error):
        with raises(ValueError):
            bp.analysis.compute_DIC(posteriors = analysis_compute_dic_value_error['posteriors'],
                                    data = analysis_compute_dic_value_error['data'],
                                    response_variable = analysis_compute_dic_value_error['response_variable'])
