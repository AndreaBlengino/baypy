import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..utils import flatten_matrix


def autocorrelation_plot(posteriors: dict, max_lags: int = 30) -> None:
    """Plots the auto-correlation for each Markov chain for each regression variable.
    The plot shows the auto-correlation trend from lag ``0`` (when auto-correlation is always ``1``) up to ``max_lags``.
    The plot layout has number of rows equal to the number of regression variables and a number of columns equal to the
    number of chains.

    Parameters
    ----------
    posteriors : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equal to the number of iterations and a number of columns equal to the number of Markov
        chains.
    max_lags : int, optional
        Maximum number of lags to which compute the auto-correlation. The default is ``30``.

    Raises
    ------
    TypeError
        - If ``posteriors`` is not a ``dict``,
        - if a posterior sample is not a ``numpy.ndarray``,
        - if ``max_lags`` is not a ``int``.
    KeyError
        If ``posteriors`` does not contain ``intercept`` key.
    ValueError
        - If a posterior sample is an empty ``numpy.ndarray``,
        - if ``max_lags`` is less or equal to ``0``.

    See Also
    --------
    :py:func:`baypy.diagnostics.functions.autocorrelation_summary`
    :py:func:`baypy.diagnostics.functions.effective_sample_size`
    """
    if not isinstance(posteriors, dict):
        raise TypeError("Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    if 'intercept' not in posteriors.keys():
        raise KeyError("Parameter 'posteriors' must contain a 'intercept' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")

    if not isinstance(max_lags, int):
        raise TypeError("Parameter 'max_lags' must be a integer")
    if max_lags <= 0:
        raise ValueError("Parameter 'max_lags' must be greater than 0")

    variable_names = list(posteriors.keys())
    n_variables = len(variable_names)
    n_iterations, n_chains = posteriors['intercept'].shape

    fig, ax = plt.subplots(nrows = n_variables,
                           ncols = n_chains,
                           figsize = (min(1.5*n_chains + 3, 10), min(1.5*n_variables + 2, 10)),
                           sharex = 'all',
                           sharey = 'all')

    if n_chains > 1:
        for k in range(n_chains):
            ax[0, k].set_title(f'Chain {k + 1}')
            for j, variable in enumerate(variable_names, 0):
                acorr = _compute_autocorrelation(vector = flatten_matrix(posteriors[variable][:, k]),
                                                 max_lags = max_lags)
                ax[j, k].stem(acorr, markerfmt = ' ', basefmt = ' ')
                ax[j, k].tick_params(bottom = False, top = False, left = False, right = False)

        for j, variable in enumerate(variable_names, 0):
            ax[j, 0].set_ylabel(variable)
            ax[j, 0].set_yticks([-1, 0, 1])

        ax[0, 0].set_xlim(-1, min(max_lags, n_iterations))
        ax[0, 0].set_ylim(-1, 1)

        plt.tight_layout()
        plt.subplots_adjust(left = 0.1)

    else:
        ax[0].set_title('Chain 1')
        for j, variable in enumerate(variable_names, 0):
            acorr = _compute_autocorrelation(vector = flatten_matrix(posteriors[variable]),
                                             max_lags = max_lags)
            ax[j].stem(acorr, markerfmt = ' ', basefmt = ' ')

        for j, variable in enumerate(variable_names, 0):
            ax[j].set_ylabel(variable)
            ax[j].set_yticks([-1, 0, 1])

        ax[0].set_xlim(-1, min(max_lags, n_iterations))
        ax[0].set_ylim(-1, 1)

        plt.tight_layout()
        plt.subplots_adjust(left = 0.14)

    plt.show()


def autocorrelation_summary(posteriors: dict, lags: list = None, print_summary: bool = True) -> pd.DataFrame:
    """Prints the auto-correlation summary for each regression variable.
    The summary reports the auto-correlation values at the lags listed in ``lags``.

    Parameters
    ----------
    posteriors : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equal to the number of iterations and a number of columns equal to the number of Markov
        chains.
    lags : list, optional
        List of the lags to which compute the auto-correlation. It cannot be a empty ``list``. It must contain only
        positive integers. The default is ``[0, 1, 5, 10, 30]``.
    print_summary : bool, optional
        If ``True`` prints the autocorrelation summary report. Default is ``True``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a number of row equal to the number of element in ``lags`` and a number of columns equal to the
        number of model variables. Lags are reported in dataframe index.

    Raises
    ------
    TypeError
        - If ``posteriors`` is not a ``dict``,
        - if a posterior sample is not a ``numpy.ndarray``,
        - if ``lags`` is not a ``list``,
        - if ``lags`` does not contain only ``int``,
        - if ``print_summary`` is not a ``bool``.
    KeyError
        If ``posteriors`` does not contain ``intercept`` key.
    ValueError
        - If a posterior sample is an empty ``numpy.ndarray``,
        - if ``lags`` is an empty ``list``,
        - if a value in ``lags`` is a negative ``int``.

    See Also
    --------
    :py:func:`baypy.diagnostics.functions.autocorrelation_plot`
    :py:func:`baypy.diagnostics.functions.effective_sample_size`

    Notes
    -----
    The reported auto-correlation for each variable is a mean of auto-correlations for the chains of that variable, for
    each chain.
    """
    if not isinstance(posteriors, dict):
        raise TypeError("Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    if not isinstance(print_summary, bool):
        raise TypeError("Parameter 'print_summary' must be a boolean")

    if 'intercept' not in posteriors.keys():
        raise KeyError("Parameter 'posteriors' must contain a 'intercept' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")

    if lags is not None:
        if not isinstance(lags, list):
            raise TypeError("Parameter 'lags' must be a list")
        if not lags:
            raise ValueError("Parameter 'lags' cannot be an empty list")
        if not all([isinstance(lag, int) for lag in lags]):
            raise TypeError("Parameter 'lags' must contain only integers")
        if any([lag < 0 for lag in lags]):
            raise ValueError("Parameter 'lags' cannot contain negative integers")

    lags = [0, 1, 5, 10, 30] if lags is None else list(set(lags))

    n_chains = posteriors['intercept'].shape[1]
    acorr_summary = pd.DataFrame(columns = list(posteriors.keys()),
                                 index = [f'Lag {lag}' for lag in lags])

    for variable in acorr_summary.columns:
        variable_acorr = []
        for k in range(n_chains):
            variable_chain_acorr = _compute_autocorrelation(vector = flatten_matrix(posteriors[variable][:, k]),
                                                            max_lags = max(lags) + 1)
            variable_acorr.append(variable_chain_acorr[lags])
        variable_acorr = np.array(variable_acorr)
        acorr_summary[variable] = variable_acorr.mean(axis = 0)

    if print_summary:
        print(acorr_summary.to_string())

    return acorr_summary


def effective_sample_size(posteriors: dict, print_summary: bool = True) -> pd.DataFrame:
    """Computes and prints the effective number of sample for each posterior.

    Parameters
    ----------
    posteriors : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equals to the number of iterations and a number of columns equal to the number of Markov
        chains.
    print_summary : bool, optional
        If ``True`` prints the effective sample size summary report. Default is ``True``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single row and a number of columns equal to the number of model variables. The unique index of
        the dataframe is ``Effective Sample Size``.

    Raises
    ------
    TypeError
        - If ``posteriors`` is not a ``dict``,
        - if a posterior sample is not a ``numpy.ndarray``,
        - if ``print_summary`` is not a ``bool``.
    KeyError
        If ``posteriors`` does not contain ``intercept`` key.
    ValueError
        If a posterior sample is an empty ``numpy.ndarray``.

    See Also
    --------
    :py:func:`baypy.diagnostics.functions.autocorrelation_plot`
    :py:func:`baypy.diagnostics.functions.autocorrelation_summary`

    Notes
    -----
    The effective number of sample could be theoretically equal to the number of iterations in case of no
    auto-correlation of the Markov chain. The greater the auto-correlation of the Markov chain, the smaller the
    effective sample size of the posterior.
    """
    if not isinstance(posteriors, dict):
        raise TypeError("Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    if not isinstance(print_summary, bool):
        raise TypeError("Parameter 'print_summary' must be a boolean")

    if 'intercept' not in posteriors.keys():
        raise KeyError("Parameter 'posteriors' must contain a 'intercept' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")

    n_chains = posteriors['intercept'].shape[1]
    ess_summary = pd.DataFrame(columns = list(posteriors.keys()),
                               index = ['Effective Sample Size'])

    for variable in ess_summary.columns:
        variable_ess = []
        for k in range(n_chains):
            vector = flatten_matrix(posteriors[variable][:, k])
            n = len(vector)
            variable_chain_acorr = _compute_autocorrelation(vector = vector, max_lags = n)
            indexes = np.arange(2, len(variable_chain_acorr), 1)
            indexes = indexes[(variable_chain_acorr[1:-1] + variable_chain_acorr[2:] < 0) & (indexes%2 == 1)]
            index = indexes[0] if indexes.size > 0 else len(variable_chain_acorr) + 1
            ess = n/(1 + 2*np.abs(variable_chain_acorr[1:index - 1].sum()))
            variable_ess.append(ess)

        ess_summary[variable] = np.sum(variable_ess)

    if print_summary:
        with pd.option_context('display.precision', 2):
            print(ess_summary.to_string())

    return ess_summary


def _compute_autocorrelation(vector: np.ndarray, max_lags: int) -> np.ndarray:

    normalized_vector = vector - vector.mean()
    autocorrelation = np.correlate(normalized_vector, normalized_vector, 'full')[len(normalized_vector) - 1:]
    autocorrelation = autocorrelation/vector.var()/len(vector)
    autocorrelation = autocorrelation[:max_lags]

    return autocorrelation
