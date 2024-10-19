from ._functions import _compute_autocorrelation
import numpy as np
import pandas as pd
from ..utils import flatten_matrix


def effective_sample_size(
    posteriors: dict[str, np.ndarray],
    print_summary: bool = True
) -> pd.DataFrame:
    """It computes and prints the effective number of sample for each
    posterior.

    Parameters
    ----------
    ``posteriors`` : :py:class:`dict`
        Posterior samples. Posteriors and relative samples are key-value pairs.
        Each sample is a :py:class:`numpy.ndarray` with a number of rows
        equals to the number of iterations and a number of columns equal to the
        number of Markov chains.
    ``print_summary`` : :py:class:`bool`, optional
        If ``True`` prints the effective sample size summary report. Default is
        ``True``.

    Returns
    -------
    :py:class:`pandas.DataFrame`
        The dataframe with a single row and a number of columns equal to the
        number of model variables. The unique index of the dataframe is
        ``'Effective Sample Size'``.

    .. admonition:: Raises
       :class: warning

       ``TypeError``
           - If ``posteriors`` is not a :py:class:`dict`,
           - if a posterior sample is not a :py:class:`numpy.ndarray`,
           - if ``print_summary`` is not a :py:class:`bool`.
       ``KeyError``
           If ``posteriors`` does not contain ``'intercept'`` key.
       ``ValueError``
           If a posterior sample is an empty :py:class:`numpy.ndarray`.

    .. admonition:: Notes
       :class: tip

       The effective number of sample could be theoretically equal to the
       number of iterations in case of no auto-correlation of the Markov chain.
       The greater the auto-correlation of the Markov chain, the smaller the
       effective sample size of the posterior.

    .. admonition:: See Also
       :class: seealso

       :py:func:`autocorrelation_plot <baypy.diagnostics.functions.autocorrelation_plot>` \n
       :py:func:`autocorrelation_summary <baypy.diagnostics.functions.autocorrelation_summary>`
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

    n_chains = posteriors['intercept'].shape[1]
    ess_summary = pd.DataFrame(
        columns=list(posteriors.keys()),
        index=["Effective Sample Size"]
    )

    for variable in ess_summary.columns:
        variable_ess = []
        for k in range(n_chains):
            vector = flatten_matrix(posteriors[variable][:, k])
            n = len(vector)
            variable_chain_acorr = _compute_autocorrelation(
                vector=vector,
                max_lags=n
            )
            indexes = np.arange(2, len(variable_chain_acorr), 1)
            indexes = indexes[
                (variable_chain_acorr[1:-1] + variable_chain_acorr[2:] < 0) &
                (indexes % 2 == 1)
            ]
            index = indexes[0] if indexes.size > 0 \
                else len(variable_chain_acorr) + 1
            ess = n/(1 + 2*np.abs(variable_chain_acorr[1:index - 1].sum()))
            variable_ess.append(ess)

        ess_summary[variable] = np.sum(variable_ess)

    if print_summary:
        with pd.option_context('display.precision', 2):
            print(ess_summary.to_string())

    return ess_summary
