import numpy as np
import pandas as pd
from ..utils import flatten_matrix


def summary(
    posteriors: dict[str, np.ndarray],
    alpha: float = 0.05,
    quantiles: list[float] = None,
    print_summary: bool = True
) -> dict[str, int | str]:
    """It prints a statistical summary for each posterior.

    Parameters
    ----------
    ``posteriors`` : :py:class:`dict`
        Posterior samples. Posteriors and relative samples are key-value pairs.
        Each sample is a :py:class:`numpy.ndarray` with a number of rows equal
        to the number of iterations and a number of columns equal to the number
        of Markov chains.
    ``alpha`` : :py:class:`float`, optional
        Significance level. It is used to compute the Highest Posterior Density
        (HPD) interval. It must be between ``0`` and ``1``. Default is
        ``0.05``.
    ``quantiles`` : :py:class:`list`, optional
        List of the quantiles to compute, for each posterior. It cannot be
        empty. It must contain only float between ``0`` and ``1``. Default is
        ``[0.025, 0.25, 0.5, 0.75, 0.975]``.
    ``print_summary`` : :py:class:`bool`, optional
        If ``True`` prints the statistical posterior summary report. Default is
        ``True``.

    Returns
    -------
    :py:class:`dict`
        Dictionary with statistical summary of posteriors. It contains:
            - key ``'n_chain'``, the number of Markov chains,
            - key ``'n_iterations'``, the number of regression iterations,
            - key ``'summary'``, the statistical summary of the posteriors, as
              a :py:class:`pandas.DataFrame`,
            - key ``'quantiles'``, quantiles summary of the posteriors, as a
              :py:class:`pandas.DataFrame`.

    .. admonition:: Raises
       :class: warning

       ``TypeError``
           - If ``posteriors`` is not a :py:class:`dict`,
           - if a posterior sample is not a :py:class:`numpy.ndarray`,
           - if ``alpha`` is not a :py:class:`float`,
           - if ``quantiles`` is not a :py:class:`list`,
           - if a ``quantiles`` value is not a :py:class:`float`,
           - if ``print_summary`` is not a :py:class:`bool`.
       ``KeyError``
           If ``posteriors`` does not contain ``'intercept'`` key.
       ``ValueError``
           - If a posterior sample is an empty :py:class:`numpy.ndarray`,
           - if ``alpha`` is not between ``0`` and ``1``,
           - if ``quantiles`` is an empty :py:class:`list`,
           - if a ``quantiles`` value is not between ``0`` and ``1``.

    .. admonition:: See Also
       :class: seealso

       :py:class:`LinearRegression <baypy.regression.linear_regression.LinearRegression>`
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

    if not isinstance(alpha, float):
        raise TypeError("Parameter 'alpha' must be a float")
    if (alpha < 0) or (alpha > 1):
        raise ValueError("Parameter 'alpha' must be between 0 and 1")

    if quantiles is not None:
        if not isinstance(quantiles, list):
            raise TypeError("Parameter 'quantiles' must be a list")
        if not quantiles:
            raise ValueError("Parameter 'quantiles' cannot be an empty list")
        if not all([isinstance(quantile, float) for quantile in quantiles]):
            raise TypeError("Parameter 'quantiles' must contain only float")
        if any([(quantile < 0) or (quantile > 1) for quantile in quantiles]):
            raise ValueError(
                "Parameter 'quantiles' cannot contain only floats between 0 "
                "and 1"
            )

    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975] if quantiles is None \
        else quantiles

    n_iterations, n_chains = posteriors['intercept'].shape

    general_summary = pd.DataFrame(index=list(posteriors.keys()))
    quantiles_summary = pd.DataFrame(index=list(posteriors.keys()))
    general_summary['Mean'] = np.nan
    general_summary['SD'] = np.nan
    general_summary['HPD min'] = np.nan
    general_summary['HPD max'] = np.nan
    for q in quantiles:
        quantiles_summary[f'{100*q}%'.replace('.0%', '%')] = np.nan

    for variable in general_summary.index:
        general_summary.loc[variable, 'Mean'] = posteriors[variable].mean()
        general_summary.loc[variable, 'SD'] = posteriors[variable].std()
        hpdi_min, hpdi_max = _compute_hpd_interval(
            x=np.sort(flatten_matrix(posteriors[variable])),
            alpha=alpha
        )
        general_summary.loc[variable, 'HPD min'] = hpdi_min
        general_summary.loc[variable, 'HPD max'] = hpdi_max
        for q in quantiles:
            quantiles_summary.loc[
                variable,
                f'{100*q}%'.replace('.0%', '%')
            ] = np.quantile(
                flatten_matrix(posteriors[variable]),
                q
            )

    credibility_mass = f'{100*(1 - alpha)}%'.replace('.0%', '%')

    if print_summary:
        summary_output = (
            f"Number of chains:      {n_chains:>6}\n"
            f"Sample size per chian: {n_iterations:>6}\n"
            f"\n"
            f"Empirical mean, standard deviation, {credibility_mass} HPD "
            f"interval for each variable:\n"
            f"\n"
            f"{general_summary.to_string()}\n"
            f"\n"
            f"Quantiles for each variable:\n"
            f"\n"
            f"{quantiles_summary.to_string()}"
        )
        print(summary_output)

    return {
        'n_chains': n_chains,
        'n_iterations': n_iterations,
        'summary': general_summary,
        'quantiles': quantiles_summary
    }


def _compute_hpd_interval(x: np.ndarray, alpha: float) -> tuple[float, float]:

    n = len(x)
    credibility_mass = 1 - alpha

    interval_idx_included = int(np.floor(credibility_mass*n))
    n_intervals = n - interval_idx_included
    interval_width = x[interval_idx_included:] - x[:n_intervals]
    min_idx = np.argmin(interval_width)
    hpdi_min = x[min_idx]
    hpdi_max = x[min_idx + interval_idx_included]

    return hpdi_min, hpdi_max
