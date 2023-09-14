import baypy as bp
from hypothesis import given, settings, HealthCheck
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytest import mark, raises
from tests.conftest import model_set_up, posteriors_data


@mark.analysis
class TestAnalysisTracePlot:


    @mark.genuine
    @given(p_data = posteriors_data())
    @settings(max_examples = 10, deadline = None)
    def test_method(self, p_data):
        bp.analysis.trace_plot(posteriors = p_data['posteriors'])
        plt.close()


    @mark.error
    def test_raises_type_error(self, analysis_trace_plot_type_error):
        with raises(TypeError):
            bp.analysis.trace_plot(analysis_trace_plot_type_error)


    @mark.error
    def test_raises_key_error(self):
        with raises(KeyError):
            bp.analysis.trace_plot(posteriors = {'variance': np.array([0])})


    @mark.error
    def test_raises_value_error(self):
        with raises(ValueError):
            bp.analysis.trace_plot(posteriors = {'intercept': np.array([])})


@mark.analysis
class TestAnalysisSummary:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_method(self, model_options):
        summary = bp.analysis.summary(model_options['posteriors'], quantiles = model_options['quantiles'])

        assert summary['n_chains'] == model_options['n_chains']
        assert summary['n_iterations'] == model_options['n_samples']
        assert isinstance(summary['summary'], pd.DataFrame)
        assert not summary['summary'].empty
        assert list(summary['summary'].index) == list(model_options['posteriors'].keys())
        assert all(summary['summary'].columns == ['Mean', 'SD', 'HPD min', 'HPD max'])
        assert isinstance(summary['quantiles'], pd.DataFrame)
        assert not summary['quantiles'].empty
        assert list(summary['quantiles'].index) == list(model_options['posteriors'].keys())
        assert all(summary['quantiles'].columns == [f'{100*quantile}%'.replace('.0%', '%')
                                                    for quantile in model_options['quantiles']])

        for posterior, posterior_samples in model_options['posteriors'].items():
            assert summary['summary'].loc[posterior, 'Mean'] == posterior_samples.mean()
            assert summary['summary'].loc[posterior, 'SD'] == posterior_samples.std()

        for posterior, posterior_samples in model_options['posteriors'].items():
            for quantile in model_options['quantiles']:
                assert summary['quantiles'].loc[posterior, f'{100*quantile}%'.replace('.0%', '%')] == \
                       np.quantile(np.asarray(posterior_samples).reshape(-1), quantile)


    @mark.error
    def test_raises_type_error(self, analysis_summary_type_error):
        with raises(TypeError):
            bp.analysis.summary(posteriors = analysis_summary_type_error['posteriors'],
                                alpha = analysis_summary_type_error['alpha'],
                                quantiles = analysis_summary_type_error['quantiles'],
                                print_summary = analysis_summary_type_error['print_summary'])


    @mark.error
    def test_raises_key_error(self):
        with raises(KeyError):
            bp.analysis.summary(posteriors = {'variance': np.array([0])},
                                alpha = 0.05,
                                quantiles = [0.1, 0.9])


    @mark.error
    def test_raises_value_error(self, analysis_summary_value_error):
        with raises(ValueError):
            bp.analysis.summary(posteriors = analysis_summary_value_error['posteriors'],
                                alpha = analysis_summary_value_error['alpha'],
                                quantiles = analysis_summary_value_error['quantiles'])


@mark.analysis
class TestAnalysisResidualsPlot:


    @mark.genuine
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_method(self, model_options):
        model = bp.model.LinearModel()
        model.data = model_options['data']
        model.response_variable = model_options['response_variable']
        model.posteriors = model_options['posteriors']
        bp.analysis.residuals_plot(model = model)
        plt.close()


    @mark.error
    def test_raises_type_error(self, analysis_residuals_plot_type_error):
        with raises(TypeError):
            bp.analysis.residuals_plot(model = analysis_residuals_plot_type_error)


    @mark.error
    def test_raises_value_error(self, analysis_residuals_plot_value_error):
        with raises(ValueError):
            bp.analysis.residuals_plot(model = analysis_residuals_plot_value_error)


@mark.analysis
class TestAnalysisComputeDIC:


    @mark.genuine
    @mark.filterwarnings('ignore::RuntimeWarning')
    @given(model_options = model_set_up())
    @settings(max_examples = 20, deadline = None, suppress_health_check = [HealthCheck.data_too_large])
    def test_method(self, model_options):
        model = bp.model.LinearModel()
        model.data = model_options['data']
        model.response_variable = model_options['response_variable']
        model.posteriors = model_options['posteriors']
        summary = bp.analysis.compute_DIC(model = model)

        assert isinstance(summary, dict)
        assert list(summary.keys()) == ['deviance at posterior means', 'posterior mean deviance',
                                        'effective number of parameters', 'DIC']
        assert all([isinstance(value, float) for value in summary.values()])


    @mark.error
    def test_raises_type_error(self, analysis_compute_dic_type_error):
        with raises(TypeError):
            bp.analysis.compute_DIC(model = analysis_compute_dic_type_error['model'],
                                    print_summary = analysis_compute_dic_type_error['print_summary'])


    @mark.error
    def test_raises_value_error(self, analysis_compute_dic_value_error):
        with raises(ValueError):
            bp.analysis.compute_DIC(model = analysis_compute_dic_value_error)
