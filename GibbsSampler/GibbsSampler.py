from .functions import sampler
from .functions import plot


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
