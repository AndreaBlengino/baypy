from .functions import sampler
from .functions import plot
from .functions import plot_autocorrelation
from .functions import print_autocorrelation
from .functions import compute_effective_sample_size
from .functions import print_summary


class GibbsSampler:

    def __init__(self):

        self.data = None
        self.y_name = None
        self.initial_values = None
        self.initial_values_intercept = None
        self.prior = None
        self.traces = None

    def set_data(self, data, y_name):

        self.data = data
        self.y_name = y_name

    def set_initial_values(self, initial_values, initial_value_intercept):

        self.initial_values = initial_values
        self.initial_values_intercept = initial_value_intercept

    def set_prior(self, prior):

        self.prior = prior

    def run(self, n_iterations, burn_in_iterations, n_chains):

        self.traces = sampler(n_iterations = n_iterations,
                              burn_in_iterations = burn_in_iterations,
                              n_chains = n_chains,
                              data = self.data,
                              y_name = self.y_name,
                              initial_values = self.initial_values,
                              initial_value_intercept = self.initial_values_intercept,
                              prior = self.prior)

    def plot(self):

        plot(traces = self.traces,
             x_names = list(self.initial_values.keys()))

    def plot_autocorrelation(self, max_lags = 30):

        plot_autocorrelation(traces = self.traces,
                             x_names = list(self.initial_values.keys()),
                             max_lags = max_lags)

    def autocorrelation_summary(self, lags = None):

        lags = [0, 1, 5, 10, 30] if lags is None else lags

        print_autocorrelation(traces = self.traces,
                              x_names = list(self.initial_values.keys()),
                              lags = lags)

    def effective_sample_size(self):

        compute_effective_sample_size(traces = self.traces,
                                      x_names = list(self.initial_values.keys()))

    def summary(self, quantiles = None):

        quantiles = [0.025, 0.25, 0.5, 0.75, 0.975] if quantiles is None else quantiles

        print_summary(traces = self.traces,
                      x_names = list(self.initial_values.keys()),
                      quantiles = quantiles)
