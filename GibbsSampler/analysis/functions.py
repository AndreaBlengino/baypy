import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from ..utils import flatten_matrix, matrix_to_frame


def trace_plot(posteriors: dict) -> None:
    """Plots the traces and the probability density for each posterior.
    The plot shows the traces for each Markov chain, for each regression variable and the relative posterior density.
    The plot layout has number of rows equal to the number of regression variables and two columns: traces on the left
    and densities on the right.

    Parameters
    ----------
    posteriors : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equal to the number of iterations and a number of columns equal to the number of Markov
        chains.

    Raises
    ------
    TypeError
        - If ``posteriors`` is not a ``dict``,
        - if a posterior sample is not a ``numpy.ndarray``.
    KeyError
        If ``posteriors`` does not contain both ``intercept`` and ``variance`` keys.
    ValueError
        If a posterior sample is an empty ``numpy.ndarray``.

    See Also
    --------
    :meth:`GibbsSampler.regression.linear_regression.LinearRegression`
    """
    if not isinstance(posteriors, dict):
        raise TypeError(f"Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    for posterior in ['intercept', 'variance']:
        if posterior not in posteriors.keys():
            raise KeyError(f"Parameter 'posteriors' must contain a '{posterior}' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")

    variable_names = list(posteriors.keys())
    n_variables = len(variable_names)
    n_iterations = len(posteriors['intercept'])

    fig = plt.figure(figsize = (10, min(1.5*n_variables + 2, 10)))
    trace_axes = []
    for i, variable in zip(range(1, 2*n_variables, 2), variable_names):
        ax_i_trace = fig.add_subplot(n_variables, 2, i)
        ax_i_density = fig.add_subplot(n_variables, 2, i + 1)

        ax_i_trace.plot(posteriors[variable], linewidth = 0.5)
        ax_i_density.plot(*_compute_kde(posteriors[variable].flatten()))

        if variable not in ['intercept', 'variance']:
            ax_i_trace.set_title(f'Trace of {variable} parameter')
            ax_i_density.set_title(f'Density of {variable} parameter')
        else:
            ax_i_trace.set_title(f'Trace of {variable}')
            ax_i_density.set_title(f'Density of {variable}')
        ax_i_trace.tick_params(bottom = False, top = False, left = False, right = False)
        ax_i_density.tick_params(bottom = False, top = False, left = False, right = False)
        ax_i_density.set_ylim(0, )
        trace_axes.append(ax_i_trace)

    for ax_i in trace_axes[1:]:
        ax_i.sharex(trace_axes[0])
    trace_axes[0].set_xlim(0, n_iterations)

    plt.tight_layout()

    plt.show()


def _compute_kde(posterior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posterior_support = np.linspace(np.min(posterior), np.max(posterior), 1000)
    posterior_kde = gaussian_kde(posterior)(posterior_support)

    return posterior_support, posterior_kde


def summary(posteriors: dict, alpha: float = 0.05, quantiles: list = None, print_summary: bool = True) -> dict:
    """Prints a statistical summary for each posterior.

    Parameters
    ----------
    posteriors : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equal to the number of iterations and a number of columns equal to the number of Markov
        chains.
    alpha : float
        Significance level. It is used to compute the Highest Posterior Density (HPD) interval. It must be between ``0``
        and ``1``.
    quantiles : list, optional
        List of the quantiles to compute, for each posterior. It cannot be empty. It must contain only float between
        ``0`` and ``1``. Default is ``[0.025, 0.25, 0.5, 0.75, 0.975]``.
    print_summary : bool, optional
        If ``True`` prints the statistical posterior summary report. Default is ``True``.

    Returns
    -------
    dict
        Dictionary with statistical summary of posteriors. It contains:
            - key ``n_chain``, the number of Markov chains,
            - key ``n_iterations``, the number of regression iterations,
            - key ``summary``, the statistical summary of the posteriors, as a pandas.DataFrame,
            - key ``quantiles``, quantiles summary of the posteriors, as a pandas.DataFrame.

    Raises
    ------
    TypeError
        - If ``posteriors`` is not a ``dict``,
        - if a posterior sample is not a ``numpy.ndarray``,
        - if ``alpha`` is not a ``float``,
        - if ``quantiles`` is not a ``list``,
        - if a ``quantiles`` value is not a ``float``,
        - if ``print_summary`` is not a ``bool``.
    KeyError
        If ``posteriors`` does not contain both ``intercept`` and ``variance`` keys.
    ValueError
        - If a posterior sample is an empty ``numpy.ndarray``,
        - if ``alpha`` is not between ``0`` and ``1``,
        - if ``quantiles`` is an empty ``list``,
        - if a ``quantiles`` value is not between ``0`` and ``1``.

    See Also
    --------
    :meth:`GibbsSampler.regression.linear_regression.LinearRegression`
    """
    if not isinstance(posteriors, dict):
        raise TypeError(f"Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    if not isinstance(print_summary, bool):
        raise TypeError("Parameter 'print_summary' must be a boolean")

    for posterior in ['intercept', 'variance']:
        if posterior not in posteriors.keys():
            raise KeyError(f"Parameter 'posteriors' must contain a '{posterior}' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")

    if (not isinstance(alpha, float)) and (alpha not in [0, 1]):
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
            raise ValueError("Parameter 'quantiles' cannot contain only floats between 0 and 1")

    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975] if quantiles is None else quantiles

    n_iterations, n_chains = posteriors['intercept'].shape

    summary = pd.DataFrame(index = list(posteriors.keys()))
    quantiles_summary = pd.DataFrame(index = list(posteriors.keys()))
    summary['Mean'] = np.nan
    summary['SD'] = np.nan
    summary['HPD min'] = np.nan
    summary['HPD max'] = np.nan
    for q in quantiles:
        quantiles_summary[f'{100*q}%'.replace('.0%', '%')] = np.nan

    for variable in summary.index:
        summary.loc[variable, 'Mean'] = posteriors[variable].mean()
        summary.loc[variable, 'SD'] = posteriors[variable].std()
        hpdi_min, hpdi_max = _compute_hpd_interval(x = np.sort(flatten_matrix(posteriors[variable])),
                                                   alpha = alpha)
        summary.loc[variable, 'HPD min'] = hpdi_min
        summary.loc[variable, 'HPD max'] = hpdi_max
        for q in quantiles:
            quantiles_summary.loc[variable, f'{100*q}%'.replace('.0%', '%')] = np.quantile(flatten_matrix(posteriors[variable]), q)

    credibility_mass = f'{100*(1 - alpha)}%'.replace('.0%', '%')

    if print_summary:
        print(f'Number of chains:      {n_chains:>6}')
        print(f'Sample size per chian: {n_iterations:>6}')
        print()
        print(f'Empirical mean, standard deviation, {credibility_mass} HPD interval for each variable:')
        print()
        print(summary.to_string())
        print()
        print(f'Quantiles for each variable:')
        print()
        print(quantiles_summary.to_string())

    return {'n_chains': n_chains, 'n_iterations': n_iterations, 'summary': summary, 'quantiles': quantiles_summary}


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


def residuals_plot(posteriors: dict, data: pd.DataFrame, response_variable: str) -> None:
    r"""Plots the residuals :math:`\epsilon` with respect to predicted values :math:`\hat{y}`.

    Parameters
    ----------
    posteriors : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equal to the number of iterations and a number of columns equal to the number of Markov
        chains.
    data : pandas.DataFrame
        Observed data of the model. It cannot be empty. It must contain regressor variables :math:`X` and the
        response variable :math:`y`.
    response_variable : string
        Name of the response variable :math:`y`. In must be one of the columns of ``data``.

    Raises
    ------
    TypeError
        - If ``posteriors`` is not a ``dict``,
        - if a posterior sample is not a ``numpy.ndarray``,
        - if ``data`` is not an instance of ``pandas.DataFrame``,
        - if ``response_variable`` is not a ``str``.
    KeyError
        If ``posteriors`` does not contain both ``intercept`` and ``variance`` keys.
    ValueError
        - If a posterior sample is an empty ``numpy.ndarray``,
        - if a posterior key is not a column of ``data``,
        - if ``data`` is an empty ``pandas.DataFrame``,
        - if ``response_variable`` is not a column of ``data``.

    See Also
    --------
    :meth:`GibbsSampler.regression.linear_regression.LinearRegression`

    Notes
    -----
    Predicted values are computed at data points :math:`X` using the posteriors means for each regressor's parameter:

    .. math::
        \hat{y_i} = \beta_0 + \sum_{j = 1}^{m} \beta_j x_{i,j}

    while residuals are the difference between the observed values and the predicted values of the
    ``response_variable``:

    .. math::
        \epsilon_i = y_i - \hat{y_i}
    """
    if not isinstance(posteriors, dict):
        raise TypeError(f"Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    for posterior in ['intercept', 'variance']:
        if posterior not in posteriors.keys():
            raise KeyError(f"Parameter 'posteriors' must contain a '{posterior}' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")
        if (posterior not in ['intercept', 'variance']) and (posterior not in data.columns):
            raise ValueError(f"Column '{posterior}' not found in 'data'")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Parameter 'data' must be an instance of 'pandas.DataFrame'")

    if not isinstance(response_variable, str):
        raise TypeError("Parameter 'response_variable' must be a string")

    if data.empty:
        raise ValueError("Parameter 'data' cannot be an empty 'pandas.DataFrame'")

    if response_variable not in data.columns:
        raise ValueError(f"Column '{response_variable}' not found in 'data'")

    data_tmp = data.copy()
    data_tmp['intercept'] = 1
    data_tmp['predicted'] = 0

    for posterior, posterior_samples in posteriors.items():
        if posterior != 'variance':
            data_tmp['predicted'] += data_tmp[posterior]*flatten_matrix(posterior_samples).mean()
    data_tmp['residuals'] = data_tmp[response_variable] - data_tmp['predicted']

    fig, ax = plt.subplots()

    ax.plot(data_tmp['predicted'], data_tmp['residuals'],
            marker = 'o', linestyle = '', alpha = 0.5)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')

    ax.tick_params(bottom = False, top = False, left = False, right = False)

    plt.tight_layout()

    plt.show()


def predict_distribution(posteriors: dict, predictors: dict) -> np.ndarray:
    """Predicts a posterior distribution for an unobserved values. It takes the posteriors samples of the model
    parameters, and for each of those, draws a sample from the likelihood.

    Parameters
    ----------
    posteriors : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equal to the number of iterations and a number of columns equal to the number of Markov
        chains.
    predictors : dict
        Values of predictors :math:`X` at which compute the posterior distribution. Each predictor has to be set as a
        key-value pair.

    Returns
    -------
    numpy.ndarray
        Array of the predicted posterior distribution. It contains a number of element equal to the number of
        regression iterations times the number of model Markov chains.

    Raises
    ------
    TypeError
        - If ``posteriors`` is not a ``dict``,
        - if a posterior sample is not a ``numpy.ndarray``,
        - if ``predictors`` is not a ``dict``.
    KeyError
        - If ``posteriors`` does not contain both ``intercept`` and ``variance`` keys,
        - if a ``predictors`` key is not a key of ``posteriors``.
    ValueError
        - If a posterior sample is an empty ``numpy.ndarray``,
        - if ``predictors`` is an empty ``dict``.

    See Also
    --------
    :meth:`GibbsSampler.regression.linear_regression.LinearRegression`
    """
    if not isinstance(posteriors, dict):
        raise TypeError(f"Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    for posterior in ['intercept', 'variance']:
        if posterior not in posteriors.keys():
            raise KeyError(f"Parameter 'posteriors' must contain a '{posterior}' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")

    if not isinstance(predictors, dict):
        raise TypeError("Parameter 'predictors' must be a dictionary")

    if len(predictors) == 0:
        raise ValueError("Parameter 'predictors' cannot be an empty dictionary")

    for regressor in predictors.keys():
        if regressor not in posteriors.keys():
            raise KeyError(f"Regressor '{regressor}' not found in 'posteriors' keys")

    prediction = matrix_to_frame(posteriors = posteriors)

    prediction['mean'] = prediction['intercept']
    for regressor in posteriors.keys():
        if regressor not in ['intercept', 'variance']:
            prediction['mean'] += prediction[regressor]*predictors[regressor]
    prediction['standard deviation'] = np.sqrt(prediction['variance'])

    return norm.rvs(loc = prediction['mean'],
                    scale = prediction['standard deviation'],
                    size = len(prediction))


def compute_DIC(posteriors: dict, data: pd.DataFrame, response_variable: str, print_summary: bool = True) -> dict:
    r"""Computes and prints the Deviance Information Criterion (DIC) for the fitted linear model.

    Parameters
    ----------
    posteriors : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equal to the number of iterations and a number of columns equal to the number of Markov
        chains.
    data : pandas.DataFrame
        Observed data of the model. It cannot be empty. It must contain regressor variables :math:`X` and the
        response variable :math:`y`.
    response_variable : string
        Name of the response variable :math:`y`. In must be one of the columns of ``data``.
    print_summary : bool, optional
        If ``True`` prints the deviance summary report. Default is ``True``.

    Returns
    -------
    dict
        Dictionary with deviance summary. It contains:
            - key ``deviance at posterior means``,
            - key ``posterior mean deviance``,
            - key ``effective number of parameters``,
            - key ``DIC``.

    Raises
    ------
    TypeError
        - If ``posteriors`` is not a ``dict``,
        - if a posterior sample is not a ``numpy.ndarray``,
        - if ``data`` is not an instance of ``pandas.DataFrame``,
        - if ``response_variable`` is not a ``str``,
        - if ``print_summary`` is not a ``bool``.
    KeyError
        If ``posteriors`` does not contain both ``intercept`` and ``variance`` keys.
    ValueError
        - If a posterior sample is an empty ``numpy.ndarray``,
        - if a posterior key is not a column of ``data``,
        - if ``data`` is an empty ``pandas.DataFrame``,
        - if ``response_variable`` is not a column of ``data``.

    See Also
    --------
    :meth:`GibbsSampler.regression.linear_regression.LinearRegression`

    Notes
    -----
    The DIC measures posterior predictive error by penalizing the fit of a model (deviance) by its complexity,
    determined by the effective number of parameters.
    Comparing some alternative models, the smaller the DIC of a model, the *better* the model.
    Pretending to fit a linear regression of the response variable :math:`y` with respect to regressors :math:`X`,
    according to the following model:

    .. math::
        y \sim N(\mu, \sigma^2)
    .. math::
        \mu = \beta_0 + B X = \beta_0 + \sum_{j = 1}^m \beta_j x_j

    then the *likelyhood* is:

    .. math::
        p \left( y \left\vert B,\sigma^2 \right. \right) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp{\frac{\left(y - \mu
        \right)^2}{\sigma^2}} .

    The *deviance* is defined as:

    .. math::
        D \left( y, B, \sigma^2 \right) = -2\log p \left( y \left\vert B,\sigma^2 \right. \right) .

    The *deviance* at posterior mean of :math:`B` and :math:`\sigma^2`, denoted by :math:`\overline{B}` and
    :math:`\overline{\sigma^2}` is:

    .. math::
        D_{{\overline{\beta}}, \overline{\sigma^2}} (y) = D \left( y, \overline{B}, \overline{\sigma^2} \right)

    while the posterior mean deviance is:

    .. math::
        \overline{D} \left( y, B, \sigma^2 \right) = E \left( D(y, B, \sigma^2) \left. \right\vert y \right) .

    and the *effective number of parameter* is defined as:

    .. math::
        pD = \overline{D} \left( y, B, \sigma^2 \right) - D_{{\overline{\beta}}, \overline{\sigma^2}} (y) .

    The *Deviance Information Criterion* is:

    .. math::
        DIC = 2 \overline{D} \left( y, B, \sigma^2 \right) - D_{{\overline{\beta}}, \overline{\sigma^2}} (y) =
        \overline{D} \left( y, B, \sigma^2 \right) + pD =
        D_{{\overline{B}}, \overline{\sigma^2}} (y) + 2pD .

    References
    ----------
    .. [1] O. Spiegelhalter DJ, Best NG, Carlin BP, van der Linde A. Bayesian measures of model complexity and fit.
       J R Statist Soc B. 2002;64:583–616.
    .. [2] Gelman A, Carlin JB, Stern HS, Rubin DS. Bayesian Data Analysis. 2. Chapman & Hall/CRC; Boca Raton,
       Florida: 2004.
    """
    if not isinstance(posteriors, dict):
        raise TypeError(f"Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    if not isinstance(print_summary, bool):
        raise TypeError("Parameter 'print_summary' must be a boolean")

    for posterior in ['intercept', 'variance']:
        if posterior not in posteriors.keys():
            raise KeyError(f"Parameter 'posteriors' must contain a '{posterior}' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")
        if (posterior not in ['intercept', 'variance']) and (posterior not in data.columns):
            raise ValueError(f"Column '{posterior}' not found in 'data'")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Parameter 'data' must be an instance of 'pandas.DataFrame'")

    if not isinstance(response_variable, str):
        raise TypeError("Parameter 'response_variable' must be a string")

    if data.empty:
        raise ValueError("Parameter 'data' cannot be an empty 'pandas.DataFrame'")

    if response_variable not in data.columns:
        raise ValueError(f"Column '{response_variable}' not found in 'data'")

    data_tmp = data.copy()
    deviance_at_posterior_means = _compute_deviace_at_posterior_means(posteriors = posteriors,
                                                                      data = data_tmp,
                                                                      response_variable = response_variable)
    posterior_mean_deviance = _compute_posterior_mean_deviance(posteriors = posteriors,
                                                               data = data_tmp,
                                                               response_variable = response_variable)
    effective_number_of_parameters = posterior_mean_deviance - deviance_at_posterior_means
    DIC = effective_number_of_parameters + posterior_mean_deviance

    if print_summary:
        print(f"Deviance at posterior means     {deviance_at_posterior_means:>12.2f}")
        print(f"Posterior mean deviance         {posterior_mean_deviance:>12.2f}")
        print(f"Effective number of parameteres {effective_number_of_parameters:>12.2f}")
        print(f"Deviace Information Criterion   {DIC:>12.2f}")

    return {'deviance at posterior means': deviance_at_posterior_means,
            'posterior mean deviance': posterior_mean_deviance,
            'effective number of parameters': effective_number_of_parameters,
            'DIC': DIC}


def _compute_deviace_at_posterior_means(posteriors: dict, data: pd.DataFrame, response_variable: str) -> float:

    posterior_means = {posterior: flatten_matrix(posterior_samples).mean()
                       for posterior, posterior_samples in posteriors.items() if posterior != 'variance'}
    variance = flatten_matrix(posteriors['variance']).mean()

    data['intercept'] = 1
    data['mean'] = 0
    for posterior, posterior_mean in posterior_means.items():
        data['mean'] += data[posterior]*posterior_mean

    data['likelyhood'] = 1/np.sqrt(2*np.pi*variance)*np.exp((data[response_variable] - data['mean'])**2/2/variance)

    return -2*np.sum(np.log(data['likelyhood']))


def _compute_posterior_mean_deviance(posteriors: dict, data: pd.DataFrame, response_variable: str) -> float:

    data['intercept'] = 1
    deviance = []

    for i in range(posteriors['intercept'].shape[0]):
        data['mean'] = 0
        data['variance'] = 0
        for posterior, posterior_samples in posteriors.items():
            if posterior != 'variance':
                data['mean'] += data[posterior]*posterior_samples[i, :].mean()
            else:
                data['variance'] = posterior_samples[i, :].mean()

        data['likelyhood'] = 1/np.sqrt(2*np.pi*data['variance'])*np.exp((data[response_variable] - data['mean'])**2/2/data['variance'])

        deviance.append(-2*np.sum(np.log(data['likelyhood'])))

    return np.mean(deviance)
