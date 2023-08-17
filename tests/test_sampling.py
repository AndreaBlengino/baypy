import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytest import mark


np.random.seed(42)

N = 50
data = pd.DataFrame()
data['x_1'] = np.random.uniform(low = 0, high = 100, size = N)
data['x_2'] = np.random.uniform(low = -10, high = 10, size = N)
data['x_3'] = np.random.uniform(low = -50, high = -40, size = N)
data['x_1 * x_2'] = data['x_1']*data['x_2']

data['y'] = 3*data['x_1'] - 20*data['x_2'] - data['x_3'] - 5*data['x_1 * x_2'] + 13 + 1*np.random.randn(N)

initial_values = {'x_1': 1,
                  'x_2': 2,
                  'x_3': 3,
                  'x_1 * x_2': 4,
                  'intercept': 5}

sigma2_sample_size = 5
sigma2_variance = 10

prior = {'x_1': {'mean': 0,
                 'variance': 1e6},
         'x_2': {'mean': 0,
                 'variance': 1e6},
         'x_3': {'mean': 0,
                 'variance': 1e6},
         'x_1 * x_2': {'mean': 0,
                       'variance': 1e6},
         'intercept': {'mean': 0,
                       'variance': 1e6},
         'sigma2': {'shape': sigma2_sample_size,
                    'scale': sigma2_sample_size*sigma2_variance}}

n_iterations = 1000
burn_in_iterations = 50
n_chains = 3

q_min = 0.025
q_max = 0.975


@mark.set_input
class TestSetInput:

    def test_set_data(self, sampler):
        sampler.set_data(data = data,
                         y_name = 'y')

        assert sampler.data.equals(data)
        assert sampler.y_name == 'y'


    def test_set_initial_values(self, sampler):
        sampler.set_initial_values(values = initial_values)

        assert sampler.initial_values == initial_values
        assert 'intercept' in sampler.initial_values.keys()


    def test_set_prior(self, sampler):
        sampler.set_prior(prior = prior)

        assert sampler.prior == prior
        assert 'intercept' in sampler.prior.keys()
        assert 'sigma2' in sampler.prior.keys()
        assert all(['mean' in prior[regressor].keys() for regressor in prior.keys() if regressor != 'sigma2'])
        assert all(['variance' in prior[regressor].keys() for regressor in prior.keys() if regressor != 'sigma2'])
        assert 'shape' in sampler.prior['sigma2'].keys()
        assert 'scale' in sampler.prior['sigma2'].keys()


@mark.analysis
def test_run(sampler):
    sampler.set_data(data = data,
                     y_name = 'y')
    sampler.set_initial_values(values = initial_values)
    sampler.set_prior(prior = prior)

    sampler.run(n_iterations = n_iterations,
                burn_in_iterations = burn_in_iterations,
                n_chains = n_chains)

    assert sampler.traces.keys() == prior.keys()
    assert all(np.array([trace.shape for trace in sampler.traces.values()])[:, 0] == n_iterations)
    assert all(np.array([trace.shape for trace in sampler.traces.values()])[:, 1] == n_chains)


@mark.diagnostics
class TestDiagnostics:

    def test_autocorrelation_summary(self, sampler):
        sampler.set_data(data = data,
                         y_name = 'y')
        sampler.set_initial_values(values = initial_values)
        sampler.set_prior(prior = prior)

        sampler.run(n_iterations = n_iterations,
                    burn_in_iterations = burn_in_iterations,
                    n_chains = n_chains)

        sampler.autocorrelation_summary()


    def test_effective_sample_size(self, sampler):
        sampler.set_data(data = data,
                         y_name = 'y')
        sampler.set_initial_values(values = initial_values)
        sampler.set_prior(prior = prior)

        sampler.run(n_iterations = n_iterations,
                    burn_in_iterations = burn_in_iterations,
                    n_chains = n_chains)

        sampler.effective_sample_size()


@mark.results
class TestResults:

    def test_summary(self, sampler):
        sampler.set_data(data = data,
                         y_name = 'y')
        sampler.set_initial_values(values = initial_values)
        sampler.set_prior(prior = prior)

        sampler.run(n_iterations = n_iterations,
                    burn_in_iterations = burn_in_iterations,
                    n_chains = n_chains)

        sampler.summary()

        regressor_names = ['intercept', 'x_1', 'x_2', 'x_3', 'x_1 * x_2']

        data_tmp = data.copy()
        data_tmp['intercept'] = 1
        results = np.linalg.lstsq(a = data[regressor_names],
                                  b = data['y'],
                                  rcond = None)[0]

        for i, regressor in enumerate(regressor_names, 0):
            lower_bound = np.quantile(np.asarray(sampler.traces[regressor]).reshape(-1), q_min)
            upper_bound = np.quantile(np.asarray(sampler.traces[regressor]).reshape(-1), q_max)

            assert lower_bound <= results[i] <= upper_bound


    def test_plot(self, sampler, monkeypatch):
        sampler.set_data(data = data,
                         y_name = 'y')
        sampler.set_initial_values(values = initial_values)
        sampler.set_prior(prior = prior)

        sampler.run(n_iterations = n_iterations,
                    burn_in_iterations = burn_in_iterations,
                    n_chains = n_chains)

        monkeypatch.setattr(plt, 'show', lambda: None)
        sampler.plot()
