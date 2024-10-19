from ._functions import _compute_autocorrelation
import matplotlib.pyplot as plt
import numpy as np
from ..utils import flatten_matrix


def autocorrelation_plot(
    posteriors: dict[str, np.ndarray],
    max_lags: int = 30
) -> None:
    """It plots the auto-correlation for each Markov chain for each regression
    variable. \n
    The plot shows the auto-correlation trend from lag ``0`` (when
    auto-correlation is always ``1``) up to ``max_lags``.
    The plot layout has number of rows equal to the number of regression
    variables and a number of columns equal to the number of chains.

    Parameters
    ----------
    ``posteriors`` : :py:class:`dict`
        Posterior samples. Posteriors and relative samples are key-value pairs.
        Each sample is a :py:class:`numpy.ndarray` with a number of rows equal
        to the number of iterations and a number of columns equal to the number
        of Markov chains.
    ``max_lags`` : :py:class:`int`, optional
        Maximum number of lags to which compute the auto-correlation. The
        default is ``30``.

    .. admonition:: Raises
       :class: warning

       ``TypeError``
           - If ``posteriors`` is not a :py:class:`dict`,
           - if a posterior sample is not a :py:class:`numpy.ndarray`,
           - if ``max_lags`` is not an :py:class:`int`.
       ``KeyError``
           If ``posteriors`` does not contain ``'intercept'`` key.
       ``ValueError``
           - If a posterior sample is an empty :py:class:`numpy.ndarray`,
           - if ``max_lags`` is less or equal to ``0``.

    .. admonition:: See Also
       :class: seealso

       :py:func:`autocorrelation_summary <baypy.diagnostics.functions.autocorrelation_summary>` \n
       :py:func:`effective_sample_size <baypy.diagnostics.functions.effective_sample_size>`
    """
    if not isinstance(posteriors, dict):
        raise TypeError("Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray)
                for posterior_sample in posteriors.values()]):
        raise TypeError(
            "All posteriors data must be an instance of 'numpy.ndarray'"
        )

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

    _, ax = plt.subplots(
        nrows=n_variables,
        ncols=n_chains,
        figsize=(min(1.5*n_chains + 3, 10), min(1.5*n_variables + 2, 10)),
        sharex='all',
        sharey='all'
    )

    if n_chains > 1:
        for k in range(n_chains):
            ax[0, k].set_title(f"Chain {k + 1}")
            for j, variable in enumerate(variable_names, 0):
                acorr = _compute_autocorrelation(
                    vector=flatten_matrix(posteriors[variable][:, k]),
                    max_lags=max_lags
                )
                ax[j, k].stem(acorr, markerfmt=' ', basefmt=' ')
                ax[j, k].tick_params(
                    bottom=False,
                    top=False,
                    left=False,
                    right=False
                )

        for j, variable in enumerate(variable_names, 0):
            ax[j, 0].set_ylabel(variable)
            ax[j, 0].set_yticks([-1, 0, 1])

        ax[0, 0].set_xlim(-1, min(max_lags, n_iterations))
        ax[0, 0].set_ylim(-1, 1)

        plt.tight_layout()
        plt.subplots_adjust(left=0.1)

    else:
        ax[0].set_title('Chain 1')
        for j, variable in enumerate(variable_names, 0):
            acorr = _compute_autocorrelation(
                vector=flatten_matrix(posteriors[variable]),
                max_lags=max_lags
            )
            ax[j].stem(acorr, markerfmt=' ', basefmt=' ')

        for j, variable in enumerate(variable_names, 0):
            ax[j].set_ylabel(variable)
            ax[j].set_yticks([-1, 0, 1])

        ax[0].set_xlim(-1, min(max_lags, n_iterations))
        ax[0].set_ylim(-1, 1)

        plt.tight_layout()
        plt.subplots_adjust(left=0.14)

    plt.show()
