import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import invgamma, multivariate_normal, gaussian_kde, norm


def sampler(n_iterations, burn_in_iterations, n_chains, data, y_name, variable_names, initial_values, prior):

    regressor_names = variable_names.copy()
    regressor_names.pop(regressor_names.index('sigma2'))

    beta_0 = [prior[x]['mean'] for x in regressor_names]
    Beta_0 = np.array(beta_0)[np.newaxis].transpose()

    sigma_0 = [prior[x]['variance'] for x in regressor_names]
    Sigma_0 = np.zeros((len(sigma_0), len(sigma_0)))
    np.fill_diagonal(Sigma_0, sigma_0)
    Sigma_0_inv = np.linalg.inv(Sigma_0)

    T_0 = prior['sigma2']['shape']
    theta_0 = prior['sigma2']['scale']

    n = len(data)
    T_1 = T_0 + n

    Y = data[y_name]
    data['intercept'] = 1
    X = np.array(data[regressor_names])

    XtX = np.dot(X.transpose(), X)
    XtY = np.dot(X.transpose(), Y)[np.newaxis].transpose()
    Sigma_0_inv_Beta_0 = np.dot(Sigma_0_inv, Beta_0)

    traces = {variable: [[] for _ in range(n_chains)] for variable in variable_names}

    beta = [[initial_values[regressor] for regressor in regressor_names] for _ in range(n_chains)]

    for i in range(burn_in_iterations + n_iterations):

        sigma2 = [sample_sigma2(Y = Y,
                                X = X,
                                beta = beta[i],
                                T_1 = T_1,
                                theta_0 = theta_0) for i in range(n_chains)]

        beta = [sample_beta(XtX = XtX,
                            XtY = XtY,
                            sigma2 = sigma2[i],
                            Sigma_0_inv = Sigma_0_inv,
                            Sigma_0_inv_Beta_0 = Sigma_0_inv_Beta_0) for i in range(n_chains)]

        if i >= burn_in_iterations:
            for j in range(n_chains):
                [traces[regressor][j].append(beta[j][k]) for k, regressor in enumerate(regressor_names, 0)]
                traces['sigma2'][j].append(sigma2[j])

    traces = {variable: np.array(trace).transpose() for variable, trace in traces.items()}

    return traces


def sample_sigma2(Y, X, beta, T_1, theta_0):

    Y_X_beta = Y - np.asarray(np.dot(X, beta)).reshape(-1)
    theta_1 = theta_0 + np.dot(Y_X_beta.transpose(), Y_X_beta)

    return invgamma.rvs(a = T_1/2,
                        scale = theta_1/2,
                        size = 1)[0]


def sample_beta(XtX, XtY, sigma2, Sigma_0_inv, Sigma_0_inv_Beta_0):

    V = np.linalg.inv(Sigma_0_inv + XtX/sigma2)
    M = np.dot(V, Sigma_0_inv_Beta_0 + XtY/sigma2)

    return multivariate_normal.rvs(mean = np.asarray(M).reshape(-1),
                                   cov = V,
                                   size = 1)


def plot(traces):

    variable_names = list(traces.keys())
    n_variables = len(variable_names)
    n_iterations = len(traces['intercept'])

    fig = plt.figure(figsize = (10, min(1.5*n_variables + 2, 10)))
    trace_axes = []
    for i, variable in zip(range(1, 2*n_variables, 2), variable_names):
        ax_i_trace = fig.add_subplot(n_variables, 2, i)
        ax_i_density = fig.add_subplot(n_variables, 2, i + 1)

        ax_i_trace.plot(traces[variable], linewidth = 0.5)
        ax_i_density.plot(*compute_kde(traces[variable].flatten()))

        if variable != 'sigma2':
            ax_i_trace.set_title(f'Trace of {variable}')
            ax_i_density.set_title(f'Density of {variable}')
        else:
            ax_i_trace.set_title(f'Trace of $\sigma^2$')
            ax_i_density.set_title(f'Density of $\sigma^2$')

        trace_axes.append(ax_i_trace)

    for ax_i in trace_axes[1:]:
        ax_i.sharex(trace_axes[0])
    trace_axes[0].set_xlim(0, n_iterations)

    plt.tight_layout()

    plt.show()


def compute_kde(trace):
    posterior_support = np.linspace(np.min(trace), np.max(trace), 1000)
    posterior_kde = gaussian_kde(trace)(posterior_support)

    return posterior_support, posterior_kde


def plot_autocorrelation(traces, max_lags):

    variable_names = list(traces.keys())
    n_variables = len(variable_names)
    n_chains = traces['intercept'].shape[1]

    fig, ax = plt.subplots(nrows = n_variables,
                           ncols = n_chains,
                           figsize = (min(1.5*n_chains + 3, 10), min(1.5*n_variables + 2, 10)),
                           sharex = 'all',
                           sharey = 'all')

    for i in range(n_chains):
        ax[0, i].set_title(f'Chain {i + 1}')
        for j, variable in enumerate(variable_names, 0):
            acorr = compute_autocorrelation(vector = np.asarray(traces[variable][:, i]).reshape(-1),
                                            max_lags = max_lags)
            ax[j, i].stem(acorr, markerfmt = ' ', basefmt = ' ')

    for i, variable in enumerate(variable_names, 0):
        if variable != 'sigma2':
            ax[i, 0].set_ylabel(variable)
        else:
            ax[i, 0].set_ylabel('$\sigma^2$')
        ax[i, 0].set_yticks([-1, 0, 1])

    ax[0, 0].set_xlim(-1, max_lags)
    ax[0, 0].set_ylim(-1, 1)

    plt.tight_layout()
    plt.subplots_adjust(left = 0.1)

    plt.show()


def print_autocorrelation(traces, lags):

    n_chains = traces['intercept'].shape[1]
    acorr_summary = pd.DataFrame(columns = list(traces.keys()),
                                 index = [f'Lag {lag}' for lag in lags])

    for variable in acorr_summary.columns:
        variable_acorr = []
        for i in range(n_chains):
            variable_chain_acorr = compute_autocorrelation(vector = np.asarray(traces[variable][:, i]).reshape(-1),
                                                           max_lags = max(lags) + 1)
            variable_acorr.append(variable_chain_acorr[lags])
        variable_acorr = np.array(variable_acorr)
        acorr_summary[variable] = variable_acorr.mean(axis = 0)

    print(acorr_summary.to_string())


def compute_effective_sample_size(traces):

    n_chains = traces['intercept'].shape[1]
    ess_summary = pd.DataFrame(columns = list(traces.keys()),
                               index = ['Effective Sample Size'])

    for variable in ess_summary.columns:
        variable_ess = []
        for i in range(n_chains):
            vector = np.asarray(traces[variable][:, i]).reshape(-1)
            n = len(vector)
            variable_chain_acorr = compute_autocorrelation(vector = vector, max_lags = n)
            indexes = np.arange(2, len(variable_chain_acorr), 1)
            indexes = indexes[(variable_chain_acorr[1:-1] + variable_chain_acorr[2:] < 0) & (indexes%2 == 1)]
            i = indexes[0] if indexes.size > 0 else len(variable_chain_acorr) + 1
            ess = n/(1 + 2*np.abs(variable_chain_acorr[1:i - 1].sum()))
            variable_ess.append(ess)

        ess_summary[variable] = np.sum(variable_ess)

    with pd.option_context('display.precision', 2):
        print(ess_summary.to_string())


def compute_autocorrelation(vector, max_lags):

    normalized_vector = vector - vector.mean()
    autocorrelation = np.correlate(normalized_vector, normalized_vector, 'full')[len(normalized_vector) - 1:]
    autocorrelation = autocorrelation/vector.var()/len(vector)
    autocorrelation = autocorrelation[:max_lags]

    return autocorrelation


def print_summary(traces, alpha, quantiles):

    n_iterations, n_chains = traces['intercept'].shape

    summary = pd.DataFrame(index = list(traces.keys()))
    quantiles_summary = pd.DataFrame(index = list(traces.keys()))
    summary['Mean'] = np.nan
    summary['SD'] = np.nan
    summary['HPD min'] = np.nan
    summary['HPD max'] = np.nan
    for q in quantiles:
        quantiles_summary[f'{100*q}%'.replace('.0%', '%')] = np.nan

    for variable in summary.index:
        summary.loc[variable, 'Mean'] = traces[variable].mean()
        summary.loc[variable, 'SD'] = traces[variable].std()
        hpdi_min, hpdi_max = compute_HPD_interval(x = np.sort(np.asarray(traces[variable]).reshape(-1)),
                                                  alpha = alpha)
        summary.loc[variable, 'HPD min'] = hpdi_min
        summary.loc[variable, 'HPD max'] = hpdi_max
        for q in quantiles:
            quantiles_summary.loc[variable, f'{100*q}%'.replace('.0%', '%')] = np.quantile(np.asarray(traces[variable]).reshape(-1), q)

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


def compute_HPD_interval(x, alpha):

    n = len(x)
    credibility_mass = 1 - alpha

    interval_idx_included = int(np.floor(credibility_mass*n))
    n_intervals = n - interval_idx_included
    interval_width = x[interval_idx_included:] - x[:n_intervals]
    min_idx = np.argmin(interval_width)
    hpdi_min = x[min_idx]
    hpdi_max = x[min_idx + interval_idx_included]

    return hpdi_min, hpdi_max


def plot_residuals(traces, data, y_name):

    data['intercept'] = 1
    data['predicted'] = 0

    for regressor, trace in traces.items():
        if regressor != 'sigma2':
            data['predicted'] += data[regressor]*np.asarray(trace).reshape(-1).mean()
    data['residuals'] = data[y_name] - data['predicted']

    fig, ax = plt.subplots()

    ax.plot(data['predicted'], data['residuals'],
            marker = 'o', linestyle = '', alpha = 0.5)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')

    plt.show()


def predict_distribution(traces, data):

    pred = pd.DataFrame()
    for regressor, trace in traces.items():
        pred[regressor] = np.asarray(trace).reshape(-1)

    pred['mean'] = pred['intercept']
    for regressor in traces.keys():
        if regressor not in ['intercept', 'sigma2']:
            pred['mean'] += pred[regressor]*data[regressor]
    pred['standard deviation'] = np.sqrt(pred['sigma2'])

    return norm.rvs(loc = pred['mean'],
                    scale = pred['standard deviation'],
                    size = len(pred))
