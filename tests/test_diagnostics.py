import GibbsSampler as gs
import matplotlib.pyplot as plt
from pytest import mark


@mark.diagnostics
class TestDiagnostics:

    def test_autocorrelation_summary(self, posteriors):
        gs.diagnostics.autocorrelation_summary(posteriors)


    def test_effective_sample_size(self, posteriors):
        gs.diagnostics.effective_sample_size(posteriors)


    def test_plot_autocorrelation(self, posteriors, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        gs.diagnostics.autocorrelation_plot(posteriors)
