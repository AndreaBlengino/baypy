import baypy as bp
from hypothesis import given, settings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytest import mark, raises
from tests.conftest import posteriors_data


@mark.diagnostics
class TestDiagnosticsAutocorrelationPlot:


    @mark.genuine
    @given(p_data = posteriors_data())
    @settings(max_examples = 10, deadline = None)
    def test_method(self, p_data):
        bp.diagnostics.autocorrelation_plot(posteriors = p_data['posteriors'],
                                            max_lags = p_data['n_samples'])
        plt.close()


    @mark.error
    def test_raises_type_error(self, diagnostics_autocorrelation_plot_type_error):
        with raises(TypeError):
            bp.diagnostics.autocorrelation_plot(posteriors = diagnostics_autocorrelation_plot_type_error['posteriors'],
                                                max_lags = diagnostics_autocorrelation_plot_type_error['max_lags'])


    @mark.error
    def test_raises_key_error(self):
        with raises(KeyError):
            bp.diagnostics.autocorrelation_plot(posteriors = {'variance': np.array([0])},
                                                max_lags = 30)


    @mark.error
    def test_raises_value_error(self, diagnostics_autocorrelation_plot_value_error):
        with raises(ValueError):
            bp.diagnostics.autocorrelation_plot(posteriors = diagnostics_autocorrelation_plot_value_error['posteriors'],
                                                max_lags = diagnostics_autocorrelation_plot_value_error['max_lags'])


@mark.diagnostics
class TestDiagnosticsAutocorrelationSummary:


    @mark.genuine
    @given(p_data = posteriors_data())
    @settings(max_examples = 20, deadline = None)
    def test_method(self, p_data):
        lags = [lag for lag in [0, 1, 5, 10, 20, 30] if lag <= p_data['n_samples'] - 1]
        acorr_summary = bp.diagnostics.autocorrelation_summary(posteriors = p_data['posteriors'],
                                                               lags = lags)

        assert isinstance(acorr_summary, pd.DataFrame)
        assert not acorr_summary.empty
        assert all([index.startswith('Lag ') for index in acorr_summary.index])
        assert list(acorr_summary.columns) == list(p_data['posteriors'].keys())
        assert all(acorr_summary.loc['Lag 0', :] - 1 < 1e-14)


    @mark.error
    def test_raises_type_error(self, diagnostics_autocorrelation_summary_type_error):
        with raises(TypeError):
            bp.diagnostics.autocorrelation_summary(posteriors = diagnostics_autocorrelation_summary_type_error['posteriors'],
                                                   lags = diagnostics_autocorrelation_summary_type_error['lags'],
                                                   print_summary = diagnostics_autocorrelation_summary_type_error['print_summary'])


    @mark.error
    def test_raises_key_error(self):
        with raises(KeyError):
            bp.diagnostics.autocorrelation_summary(posteriors = {'variance': np.array([0])},
                                                   lags = [0, 1, 5, 10, 30])


    @mark.error
    def test_raises_value_error(self, diagnostics_autocorrelation_summary_value_error):
        with raises(ValueError):
            bp.diagnostics.autocorrelation_summary(posteriors = diagnostics_autocorrelation_summary_value_error['posteriors'],
                                                   lags = diagnostics_autocorrelation_summary_value_error['lags'])


@mark.diagnostics
class TestDiagnosticsEffectiveSampleSize:


    @mark.genuine
    @given(p_data = posteriors_data())
    @settings(max_examples = 20, deadline = None)
    def test_method(self, p_data):
        ess_summary = bp.diagnostics.effective_sample_size(posteriors = p_data['posteriors'])

        assert isinstance(ess_summary, pd.DataFrame)
        assert not ess_summary.empty
        assert ess_summary.index[0] == 'Effective Sample Size'
        assert list(ess_summary.columns) == list(p_data['posteriors'].keys())
        assert all(ess_summary <= p_data['n_samples']*p_data['n_chains'])
        assert all(ess_summary >= 1)


    @mark.error
    def test_raises_type_error(self, diagnostics_effective_sample_size_type_error):
        with raises(TypeError):
            bp.diagnostics.effective_sample_size(posteriors = diagnostics_effective_sample_size_type_error['posteriors'],
                                                 print_summary = diagnostics_effective_sample_size_type_error['print_summary'])


    @mark.error
    def test_raises_key_error(self):
        with raises(KeyError):
            bp.diagnostics.effective_sample_size({'variance': np.array([0])})


    @mark.error
    def test_raises_value_error(self):
        with raises(ValueError):
            bp.diagnostics.effective_sample_size(posteriors = {'intercept': np.array([])})
