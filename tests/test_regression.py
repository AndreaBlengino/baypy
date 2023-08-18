import numpy as np
from pytest import mark


sigma2_sample_size = 5
sigma2_variance = 10

priors = {'x_1': {'mean': 0,
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


@mark.regression
def test_sample(sampler):
    sampler.sample(n_iterations = n_iterations,
                   burn_in_iterations = burn_in_iterations,
                   n_chains = n_chains)

    assert sampler.posteriors.keys() == priors.keys()
    assert all(np.array([posterior.shape for posterior in sampler.posteriors.values()])[:, 0] == n_iterations)
    assert all(np.array([posterior.shape for posterior in sampler.posteriors.values()])[:, 1] == n_chains)
