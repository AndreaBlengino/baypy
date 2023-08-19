import GibbsSampler as gs
import matplotlib.pyplot as plt
from pytest import mark, raises


@mark.diagnostics
class TestDiagnosticsAutocorrelationPlot:


    def test_method(self, posteriors, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.diagnostics.autocorrelation_plot(posteriors)


    def test_raises_type_error(self, diagnostics_autocorrelation_plot_type_error):
        with raises(TypeError):
            gs.diagnostics.autocorrelation_plot(posteriors = diagnostics_autocorrelation_plot_type_error['posteriors'],
                                                max_lags = diagnostics_autocorrelation_plot_type_error['max_lags'])


    def test_raises_value_error(self, diagnostics_autocorrelation_plot_value_error):
        with raises(ValueError):
            gs.diagnostics.autocorrelation_plot(posteriors = diagnostics_autocorrelation_plot_value_error['posteriors'],
                                                max_lags = diagnostics_autocorrelation_plot_value_error['max_lags'])


@mark.diagnostics
class TestDiagnosticsAutocorrelationSummary:


    def test_method(self, posteriors):
        gs.diagnostics.autocorrelation_summary(posteriors)


    def test_raises_type_error(self, diagnostics_autocorrelation_summary_type_error):
        with raises(TypeError):
            gs.diagnostics.autocorrelation_summary(posteriors = diagnostics_autocorrelation_summary_type_error['posteriors'],
                                                   lags = diagnostics_autocorrelation_summary_type_error['lags'])


    def test_raises_value_error(self, diagnostics_autocorrelation_summary_value_error):
        with raises(ValueError):
            gs.diagnostics.autocorrelation_summary(posteriors = diagnostics_autocorrelation_summary_value_error['posteriors'],
                                                   lags = diagnostics_autocorrelation_summary_value_error['lags'])


@mark.diagnostics
class TestDiagnosticsEffectiveSampleSize:


    def test_method(self, posteriors):
        gs.diagnostics.effective_sample_size(posteriors)


    def test_raises_type_error(self, diagnostics_effective_sample_size_type_error):
        with raises(TypeError):
            gs.diagnostics.effective_sample_size(diagnostics_effective_sample_size_type_error)


    def test_raises_value_error(self, diagnostics_effective_sample_size_value_error):
        with raises(ValueError):
            gs.diagnostics.effective_sample_size(diagnostics_effective_sample_size_value_error)
