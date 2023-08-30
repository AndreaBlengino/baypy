import baypy as bp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytest import mark, raises


@mark.diagnostics
class TestDiagnosticsAutocorrelationPlot:


    def test_method(self, posteriors, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        bp.diagnostics.autocorrelation_plot(posteriors)


    def test_raises_type_error(self, diagnostics_autocorrelation_plot_type_error):
        with raises(TypeError):
            bp.diagnostics.autocorrelation_plot(posteriors = diagnostics_autocorrelation_plot_type_error['posteriors'],
                                                max_lags = diagnostics_autocorrelation_plot_type_error['max_lags'])


    def test_raises_key_error(self, diagnostics_autocorrelation_plot_key_error):
        with raises(KeyError):
            bp.diagnostics.autocorrelation_plot(posteriors = diagnostics_autocorrelation_plot_key_error['posteriors'],
                                                max_lags = diagnostics_autocorrelation_plot_key_error['max_lags'])


    def test_raises_value_error(self, diagnostics_autocorrelation_plot_value_error):
        with raises(ValueError):
            bp.diagnostics.autocorrelation_plot(posteriors = diagnostics_autocorrelation_plot_value_error['posteriors'],
                                                max_lags = diagnostics_autocorrelation_plot_value_error['max_lags'])


@mark.diagnostics
class TestDiagnosticsAutocorrelationSummary:


    def test_method(self, posteriors):
        acorr_summary = bp.diagnostics.autocorrelation_summary(posteriors)

        assert isinstance(acorr_summary, pd.DataFrame)
        assert not acorr_summary.empty
        assert all([index.startswith('Lag ') for index in acorr_summary.index])
        assert list(acorr_summary.columns) == list(posteriors.keys())


    def test_raises_type_error(self, diagnostics_autocorrelation_summary_type_error):
        with raises(TypeError):
            bp.diagnostics.autocorrelation_summary(posteriors = diagnostics_autocorrelation_summary_type_error['posteriors'],
                                                   lags = diagnostics_autocorrelation_summary_type_error['lags'],
                                                   print_summary = diagnostics_autocorrelation_summary_type_error['print_summary'])


    def test_raises_key_error(self, diagnostics_autocorrelation_summary_key_error):
        with raises(KeyError):
            bp.diagnostics.autocorrelation_summary(posteriors = diagnostics_autocorrelation_summary_key_error['posteriors'],
                                                   lags = diagnostics_autocorrelation_summary_key_error['lags'])


    def test_raises_value_error(self, diagnostics_autocorrelation_summary_value_error):
        with raises(ValueError):
            bp.diagnostics.autocorrelation_summary(posteriors = diagnostics_autocorrelation_summary_value_error['posteriors'],
                                                   lags = diagnostics_autocorrelation_summary_value_error['lags'])


@mark.diagnostics
class TestDiagnosticsEffectiveSampleSize:


    def test_method(self, posteriors):
        ess_summary = bp.diagnostics.effective_sample_size(posteriors)

        assert isinstance(ess_summary, pd.DataFrame)
        assert not ess_summary.empty
        assert ess_summary.index[0] == 'Effective Sample Size'
        assert list(ess_summary.columns) == list(posteriors.keys())


    def test_raises_type_error(self, diagnostics_effective_sample_size_type_error):
        with raises(TypeError):
            bp.diagnostics.effective_sample_size(posteriors = diagnostics_effective_sample_size_type_error['posteriors'],
                                                 print_summary = diagnostics_effective_sample_size_type_error['print_summary'])


    def test_raises_key_error(self, diagnostics_effective_sample_size_key_error):
        with raises(KeyError):
            bp.diagnostics.effective_sample_size(diagnostics_effective_sample_size_key_error)


    def test_raises_value_error(self):
        with raises(ValueError):
            bp.diagnostics.effective_sample_size(posteriors = {'intercept': np.array([]), 'variance': np.array([0])})
