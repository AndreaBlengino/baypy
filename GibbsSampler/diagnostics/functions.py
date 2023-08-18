import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..utils import flatten_matrix


def autocorrelation_plot(posteriors, max_lags = 30):

    variable_names = list(posteriors.keys())
    n_variables = len(variable_names)
    n_chains = posteriors['intercept'].shape[1]

    fig, ax = plt.subplots(nrows = n_variables,
                           ncols = n_chains,
                           figsize = (min(1.5*n_chains + 3, 10), min(1.5*n_variables + 2, 10)),
                           sharex = 'all',
                           sharey = 'all')

    for i in range(n_chains):
        ax[0, i].set_title(f'Chain {i + 1}')
        for j, variable in enumerate(variable_names, 0):
            acorr = _compute_autocorrelation(vector = flatten_matrix(posteriors[variable][:, i]),
                                             max_lags = max_lags)
            ax[j, i].stem(acorr, markerfmt = ' ', basefmt = ' ')

    for i, variable in enumerate(variable_names, 0):
        if variable != 'sigma2':
            ax[i, 0].set_ylabel(variable)
        else:
            ax[i, 0].set_ylabel(r'$\sigma^2$')
        ax[i, 0].set_yticks([-1, 0, 1])

    ax[0, 0].set_xlim(-1, max_lags)
    ax[0, 0].set_ylim(-1, 1)

    plt.tight_layout()
    plt.subplots_adjust(left = 0.1)

    plt.show()


def autocorrelation_summary(posteriors, lags = None):

    lags = [0, 1, 5, 10, 30] if lags is None else lags

    n_chains = posteriors['intercept'].shape[1]
    acorr_summary = pd.DataFrame(columns = list(posteriors.keys()),
                                 index = [f'Lag {lag}' for lag in lags])

    for variable in acorr_summary.columns:
        variable_acorr = []
        for i in range(n_chains):
            variable_chain_acorr = _compute_autocorrelation(vector = flatten_matrix(posteriors[variable][:, i]),
                                                            max_lags = max(lags) + 1)
            variable_acorr.append(variable_chain_acorr[lags])
        variable_acorr = np.array(variable_acorr)
        acorr_summary[variable] = variable_acorr.mean(axis = 0)

    print(acorr_summary.to_string())


def effective_sample_size(posteriors):

    n_chains = posteriors['intercept'].shape[1]
    ess_summary = pd.DataFrame(columns = list(posteriors.keys()),
                               index = ['Effective Sample Size'])

    for variable in ess_summary.columns:
        variable_ess = []
        for i in range(n_chains):
            vector = flatten_matrix(posteriors[variable][:, i])
            n = len(vector)
            variable_chain_acorr = _compute_autocorrelation(vector = vector, max_lags = n)
            indexes = np.arange(2, len(variable_chain_acorr), 1)
            indexes = indexes[(variable_chain_acorr[1:-1] + variable_chain_acorr[2:] < 0) & (indexes%2 == 1)]
            i = indexes[0] if indexes.size > 0 else len(variable_chain_acorr) + 1
            ess = n/(1 + 2*np.abs(variable_chain_acorr[1:i - 1].sum()))
            variable_ess.append(ess)

        ess_summary[variable] = np.sum(variable_ess)

    with pd.option_context('display.precision', 2):
        print(ess_summary.to_string())


def _compute_autocorrelation(vector, max_lags):

    normalized_vector = vector - vector.mean()
    autocorrelation = np.correlate(normalized_vector, normalized_vector, 'full')[len(normalized_vector) - 1:]
    autocorrelation = autocorrelation/vector.var()/len(vector)
    autocorrelation = autocorrelation[:max_lags]

    return autocorrelation