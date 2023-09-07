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


    def test_raises_key_error(self):
        with raises(KeyError):
            bp.analysis.trace_plot(posteriors = {'variance': np.array([0])})


    def test_raises_value_error(self):
        with raises(ValueError):
            bp.analysis.trace_plot(posteriors = {'intercept': np.array([])})


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


    def test_raises_key_error(self):
        with raises(KeyError):
            bp.analysis.summary(posteriors = {'variance': np.array([0])},
                                alpha = 0.05,
                                quantiles = [0.1, 0.9])


    def test_raises_value_error(self, analysis_summary_value_error):
        with raises(ValueError):
            bp.analysis.summary(posteriors = analysis_summary_value_error['posteriors'],
                                alpha = analysis_summary_value_error['alpha'],
                                quantiles = analysis_summary_value_error['quantiles'])


@mark.analysis
class TestAnalysisResidualsPlot:


    def test_method(self, solved_model, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        bp.analysis.residuals_plot(model = solved_model)


    def test_raises_type_error(self, analysis_residuals_plot_type_error):
        with raises(TypeError):
            bp.analysis.residuals_plot(model = analysis_residuals_plot_type_error)


    def test_raises_value_error(self, analysis_residuals_plot_value_error):
        with raises(ValueError):
            bp.analysis.residuals_plot(model = analysis_residuals_plot_value_error)


@mark.analysis
class TestAnalysisComputeDIC:


    def test_method(self, solved_model):
        summary = bp.analysis.compute_DIC(model = solved_model)

        assert isinstance(summary, dict)
        assert list(summary.keys()) == ['deviance at posterior means', 'posterior mean deviance',
                                        'effective number of parameters', 'DIC']
        assert all([isinstance(value, float) for value in summary.values()])


    def test_raises_type_error(self, analysis_compute_dic_type_error):
        with raises(TypeError):
            bp.analysis.compute_DIC(model = analysis_compute_dic_type_error['model'],
                                    print_summary = analysis_compute_dic_type_error['print_summary'])


    def test_raises_value_error(self, analysis_compute_dic_value_error):
        with raises(ValueError):
            bp.analysis.compute_DIC(model = analysis_compute_dic_value_error)
