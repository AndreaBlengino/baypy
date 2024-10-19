from ._functions import _compute_autocorrelation
import numpy as np
import pandas as pd
from typing import Iterable
from ..utils import flatten_matrix


def autocorrelation_summary(
    posteriors: dict[str, np.ndarray],
    lags: Iterable[int] = None,
    print_summary: bool = True
) -> pd.DataFrame:
    """It prints the auto-correlation summary for each regression variable. \n
    The summary reports the auto-correlation values at the lags listed in
    ``lags``.

    Parameters
    ----------
    ``posteriors`` : :py:class:`dict`
        Posterior samples. Posteriors and relative samples are key-value pairs.
        Each sample is a :py:class:`numpy.ndarray` with a number of rows equal
        to the number of iterations and a number of columns equal to the number
        of Markov chains.
    ``lags`` : :py:class:`Iterable <typing.Iterable>`, optional
        List of the lags to which compute the auto-correlation. It cannot be an
        empty :py:class:`Iterable <typing.Iterable>`. It must contain only
        positive integers. The default is ``[0, 1, 5, 10, 30]``.
    ``print_summary`` : :py:class:`bool`, optional
        If ``True`` prints the autocorrelation summary report. Default is
        ``True``.

    Returns
    -------
    :py:class:`pandas.DataFrame`
        The dataframe with a number of row equal to the number of element in
        ``lags`` and a number of columns equal to the number of model
        variables. Lags are reported in dataframe index.

    .. admonition:: Raises
       :class: warning

       ``TypeError``
           - If ``posteriors`` is not a :py:class:`dict`,
           - if a posterior sample is not a :py:class:`numpy.ndarray`,
           - if ``lags`` is not an :py:class:`Iterable <typing.Iterable>`,
           - if ``lags`` does not contain only :py:class:`int`,
           - if ``print_summary`` is not a :py:class:`bool`.
       ``KeyError``
           If ``posteriors`` does not contain ``'intercept'`` key.
       ``ValueError``
           - If a posterior sample is an empty :py:class:`numpy.ndarray`,
           - if ``lags`` is an empty :py:class:`Iterable <typing.Iterable>`,
           - if a value in ``lags`` is a negative :py:class:`int`.

    .. admonition:: Notes
       :class: tip

       The reported auto-correlation for each variable is a mean of
       auto-correlations for the chains of that variable, for each chain.

    .. admonition:: See Also
       :class: seealso

       :py:func:`autocorrelation_plot <baypy.diagnostics.functions.autocorrelation_plot>` \n
       :py:func:`effective_sample_size <baypy.diagnostics.functions.effective_sample_size>`
    """
    if not isinstance(posteriors, dict):
        raise TypeError("Parameter 'posteriors' must be a dictionary")

    if not all(
        [
            isinstance(posterior_sample, np.ndarray)
            for posterior_sample in posteriors.values()
        ]
    ):
        raise TypeError(
            "All posteriors data must be an instance of 'numpy.ndarray'"
        )

    if not isinstance(print_summary, bool):
        raise TypeError("Parameter 'print_summary' must be a boolean")

    if 'intercept' not in posteriors.keys():
        raise KeyError("Parameter 'posteriors' must contain a 'intercept' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")

    if lags is not None:
        if not isinstance(lags, Iterable):
            raise TypeError("Parameter 'lags' must be an iterable")
        if not lags:
            raise ValueError("Parameter 'lags' cannot be an empty iterable")
        if not all([isinstance(lag, int) for lag in lags]):
            raise TypeError("Parameter 'lags' must contain only integers")
        if any([lag < 0 for lag in lags]):
            raise ValueError(
                "Parameter 'lags' cannot contain negative integers"
            )

    lags = [0, 1, 5, 10, 30] if lags is None else list(set(lags))

    n_chains = posteriors['intercept'].shape[1]
    acorr_summary = pd.DataFrame(
        columns=list(posteriors.keys()),
        index=[f"Lag {lag}" for lag in lags]
    )

    for variable in acorr_summary.columns:
        variable_acorr = []
        for k in range(n_chains):
            variable_chain_acorr = _compute_autocorrelation(
                vector=flatten_matrix(posteriors[variable][:, k]),
                max_lags=max(lags) + 1
            )
            variable_acorr.append(variable_chain_acorr[lags])
        variable_acorr = np.array(variable_acorr)
        acorr_summary[variable] = variable_acorr.mean(axis=0)

    if print_summary:
        print(acorr_summary.to_string())

    return acorr_summary
