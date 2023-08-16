from .functions import sampler
from .functions import plot
from .functions import plot_autocorrelation
from .functions import print_autocorrelation
from .functions import compute_effective_sample_size
from .functions import print_summary
from .functions import plot_residuals
from .functions import predict_distribution


class GibbsSampler:

    def __init__(self):

        self.data = None
        self.y_name = None
        self.initial_values = None
        self.prior = None
        self.variable_names = None
        self.traces = None

    def set_data(self, data, y_name):

        self.data = data
        self.y_name = y_name

    def set_initial_values(self, values):

        self.initial_values = values

    def set_prior(self, prior):

        self.prior = prior

    def run(self, n_iterations, burn_in_iterations, n_chains):

        self.variable_names = list(self.prior.keys())
        self.variable_names.insert(0, self.variable_names.pop(self.variable_names.index('intercept')))

        self.traces = sampler(n_iterations = n_iterations,
                              burn_in_iterations = burn_in_iterations,
                              n_chains = n_chains,
                              data = self.data,
                              y_name = self.y_name,
                              variable_names = self.variable_names,
                              initial_values = self.initial_values,
                              prior = self.prior)

    def plot(self):

        plot(traces = self.traces)

    def plot_autocorrelation(self, max_lags = 30):

        plot_autocorrelation(traces = self.traces,
                             max_lags = max_lags)

    def autocorrelation_summary(self, lags = None):

        lags = [0, 1, 5, 10, 30] if lags is None else lags

        print_autocorrelation(traces = self.traces,
                              lags = lags)

    def effective_sample_size(self):

        compute_effective_sample_size(traces = self.traces)

    def summary(self, alpha = 0.05, quantiles = None):

        quantiles = [0.025, 0.25, 0.5, 0.75, 0.975] if quantiles is None else quantiles

        print_summary(traces = self.traces,
                      alpha = alpha,
                      quantiles = quantiles)

    def plot_residuals(self):

        plot_residuals(traces = self.traces,
                       data = self.data.copy(),
                       y_name = self.y_name)

    def predict_distribution(self, data):

        return predict_distribution(traces = self.traces,
                                    data = data)
