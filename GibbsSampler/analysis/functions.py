import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from ..utils import flatten_matrix


def trace_plot(posteriors):

    if not isinstance(posteriors, dict):
        raise TypeError(f"Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    for posterior in ['intercept', 'sigma2']:
        if posterior not in posteriors.keys():
            raise ValueError(f"Parameter 'posteriors' must contain a '{posterior}' key")

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

        if variable != 'sigma2':
            ax_i_trace.set_title(f'Trace of {variable}')
            ax_i_density.set_title(f'Density of {variable}')
        else:
            ax_i_trace.set_title(r'Trace of $\sigma^2$')
            ax_i_density.set_title(r'Density of $\sigma^2$')

        trace_axes.append(ax_i_trace)

    for ax_i in trace_axes[1:]:
        ax_i.sharex(trace_axes[0])
    trace_axes[0].set_xlim(0, n_iterations)

    plt.tight_layout()

    plt.show()


def _compute_kde(posterior):
    posterior_support = np.linspace(np.min(posterior), np.max(posterior), 1000)
    posterior_kde = gaussian_kde(posterior)(posterior_support)

    return posterior_support, posterior_kde


def summary(posteriors, alpha = 0.05, quantiles = None):

    if not isinstance(posteriors, dict):
        raise TypeError(f"Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    for posterior in ['intercept', 'sigma2']:
        if posterior not in posteriors.keys():
            raise ValueError(f"Parameter 'posteriors' must contain a '{posterior}' key")

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


def _compute_hpd_interval(x, alpha):

    n = len(x)
    credibility_mass = 1 - alpha

    interval_idx_included = int(np.floor(credibility_mass*n))
    n_intervals = n - interval_idx_included
    interval_width = x[interval_idx_included:] - x[:n_intervals]
    min_idx = np.argmin(interval_width)
    hpdi_min = x[min_idx]
    hpdi_max = x[min_idx + interval_idx_included]

    return hpdi_min, hpdi_max


def residuals_plot(posteriors, data, y_name):

    if not isinstance(posteriors, dict):
        raise TypeError(f"Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    for posterior in ['intercept', 'sigma2']:
        if posterior not in posteriors.keys():
            raise ValueError(f"Parameter 'posteriors' must contain a '{posterior}' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")
        if (posterior not in ['intercept', 'sigma2']) and (posterior not in data.columns):
            raise ValueError(f"Column '{posterior}' not found in 'data'")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Parameter 'data' must be an instance of 'pandas.DataFrame'")

    if not isinstance(y_name, str):
        raise TypeError("Parameter 'y_name' must be a string")

    if data.empty:
        raise ValueError("Parameter 'data' cannot be an empty 'pandas.DataFrame'")

    if y_name not in data.columns:
        raise ValueError(f"Column '{y_name}' not found in 'data'")

    data_tmp = data.copy()
    data_tmp['intercept'] = 1
    data_tmp['predicted'] = 0

    for regressor, posterior in posteriors.items():
        if regressor != 'sigma2':
            data_tmp['predicted'] += data_tmp[regressor]*flatten_matrix(posterior).mean()
    data_tmp['residuals'] = data_tmp[y_name] - data_tmp['predicted']

    fig, ax = plt.subplots()

    ax.plot(data_tmp['predicted'], data_tmp['residuals'],
            marker = 'o', linestyle = '', alpha = 0.5)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')

    plt.show()


def predict_distribution(posteriors, data):

    if not isinstance(posteriors, dict):
        raise TypeError(f"Parameter 'posteriors' must be a dictionary")

    if not all([isinstance(posterior_sample, np.ndarray) for posterior_sample in posteriors.values()]):
        raise TypeError("All posteriors data must be an instance of 'numpy.ndarray'")

    for posterior in ['intercept', 'sigma2']:
        if posterior not in posteriors.keys():
            raise ValueError(f"Parameter 'posteriors' must contain a '{posterior}' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")

    if not isinstance(data, dict):
        raise TypeError("Parameter 'data' must be a dictionary")

    if len(data) == 0:
        raise ValueError("Parameter 'data' cannot be an empty dictionary")

    for regressor in data.keys():
        if regressor not in posteriors.keys():
            raise ValueError(f"Regressor '{regressor}' not found in 'posteriors' keys")

    pred = pd.DataFrame()
    for regressor, posterior in posteriors.items():
        pred[regressor] = flatten_matrix(posterior)

    pred['mean'] = pred['intercept']
    for regressor in posteriors.keys():
        if regressor not in ['intercept', 'sigma2']:
            pred['mean'] += pred[regressor]*data[regressor]
    pred['standard deviation'] = np.sqrt(pred['sigma2'])

    return norm.rvs(loc = pred['mean'],
                    scale = pred['standard deviation'],
                    size = len(pred))
