import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import invgamma, multivariate_normal, gaussian_kde


def sampler(n_iterations, burn_in_iterations, n_chains, data, y_name, initial_values, initial_value_intercept, prior):

    beta_0 = [prior[x]['mean'] for x in list(initial_values.keys())]
    beta_0.insert(0, prior['intercept']['mean'])
    Beta_0 = np.matrix(beta_0).transpose()

    sigma_0 = [prior[x]['variance'] for x in list(initial_values.keys())]
    sigma_0.insert(0, prior['intercept']['variance'])
    Sigma_0 = np.zeros((len(sigma_0), len(sigma_0)))
    np.fill_diagonal(Sigma_0, sigma_0)
    Sigma_0_inv = np.linalg.inv(Sigma_0)

    T_0 = prior['sigma2']['shape']
    theta_0 = prior['sigma2']['scale']

    n = len(data)
    T_1 = T_0 + n

    Y = data[y_name]
    X = np.asmatrix(data[list(initial_values.keys())])
    X = np.hstack((np.ones((n, 1)), X))

    XtX = np.dot(X.transpose(), X)
    XtY = np.dot(X.transpose(), Y).transpose()
    Sigma_0_inv_Beta_0 = np.dot(Sigma_0_inv, Beta_0)


    traces = {parameter: [[] for _ in range(n_chains)] for parameter in list(initial_values.keys())}
    traces['intercept'] = [[] for _ in range(n_chains)]
    traces['sigma2'] = [[] for _ in range(n_chains)]

    beta = [np.hstack((initial_value_intercept, list(initial_values.values()))) for _ in range(n_chains)]

    for _ in range(burn_in_iterations):

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

    for _ in range(n_iterations):

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


        for i in range(n_chains):
            traces['sigma2'][i].append(sigma2[i])
            traces['intercept'][i].append(beta[i][0])
            [traces[x_k][i].append(beta[i][k]) for k, x_k in enumerate(list(initial_values.keys()), 1)]

    traces = {parameter: np.matrix(trace).transpose() for parameter, trace in traces.items()}

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


def plot(traces, x_names):

    n_variables = len(traces.keys())
    n_iterations = len(traces['intercept'])

    fig = plt.figure(figsize = (10, min(1.5*n_variables + 2, 10)))
    ax_intercept_trace = fig.add_subplot(n_variables, 2, 1)
    ax_intercept_density = fig.add_subplot(n_variables, 2, 2)

    ax_intercept_trace.plot(traces['intercept'], linewidth = 0.5)
    ax_intercept_density.plot(*compute_kde(traces['intercept'].flatten()))

    ax_intercept_trace.set_title('Trace of intercept')
    ax_intercept_density.set_title('Density of intercept')

    for i, x_i in zip(range(3, 2*n_variables - 2, 2), x_names):
        ax_i_trace = fig.add_subplot(n_variables, 2, i, sharex = ax_intercept_trace)
        ax_i_density = fig.add_subplot(n_variables, 2, i + 1)

        ax_i_trace.plot(traces[x_i], linewidth = 0.5)
        ax_i_density.plot(*compute_kde(traces[x_i].flatten()))

        ax_i_trace.set_title(f'Trace of {x_i}')
        ax_i_density.set_title(f'Density of {x_i}')

    ax_sigma2_trace = fig.add_subplot(n_variables, 2, 2*n_variables - 1, sharex = ax_intercept_trace)
    ax_sigma2_density = fig.add_subplot(n_variables, 2, 2*n_variables)

    ax_sigma2_trace.plot(traces['sigma2'], linewidth = 0.5)
    ax_sigma2_density.plot(*compute_kde(traces['sigma2'].flatten()))

    ax_sigma2_trace.set_title(r'Trace of $\sigma^2$')
    ax_sigma2_density.set_title('Density of $\sigma^2$')

    ax_intercept_trace.set_xlim(0, n_iterations)

    plt.tight_layout()

    plt.show()


def compute_kde(trace):
    posterior_support = np.linspace(np.min(trace), np.max(trace), 1000)
    posterior_kde = gaussian_kde(trace)(posterior_support)

    return posterior_support, posterior_kde


def plot_autocorrelation(traces, x_names, max_lags):

    n_variables = len(traces.keys())
    n_chains = traces['intercept'].shape[1]

    fig, ax = plt.subplots(nrows = n_variables,
                           ncols = n_chains,
                           figsize = (min(1.5*n_chains + 3, 10), min(1.5*n_variables + 2, 10)),
                           sharex = 'all',
                           sharey = 'all')

    for i in range(n_chains):
        autocorrelation = compute_autocorrelation(vector = np.asarray(traces['intercept'][:, i]).reshape(-1),
                                                  max_lags = max_lags)
        ax[0, i].stem(autocorrelation, markerfmt = ' ', basefmt = ' ')
        ax[0, i].set_title(f'Chain {i + 1}')

        for j, x_j in zip(range(1, n_variables - 1), x_names):
            autocorrelation = compute_autocorrelation(vector = np.asarray(traces[x_j][:, i]).reshape(-1),
                                                      max_lags = max_lags)
            ax[j, i].stem(autocorrelation, markerfmt = ' ', basefmt = ' ')
            ax[j, 0].set_ylabel(f'{x_j}')

        autocorrelation = compute_autocorrelation(vector = np.asarray(traces['sigma2'][:, i]).reshape(-1),
                                                  max_lags = max_lags)
        ax[n_variables - 1, i].stem(autocorrelation, markerfmt = ' ', basefmt = ' ')

    for i in range(n_variables):
        ax[i, 0].set_yticks([-1, 0, 1])

    ax[0, 0].set_ylabel('intercept')
    ax[n_variables - 1, 0].set_ylabel('$\sigma^2$')

    ax[0, 0].set_xlim(-1, max_lags)
    ax[0, 0].set_ylim(-1, 1)

    plt.tight_layout()
    plt.subplots_adjust(left = 0.1)

    plt.show()


def print_autocorrelation(traces, x_names, lags):

    n_chains = traces['intercept'].shape[1]
    acorr_summary = pd.DataFrame(columns = ['intercept', *x_names, 'sigma2'],
                                 index = [f'Lag {lag}' for lag in lags])

    for parameter in acorr_summary.columns:
        param_acorr = []
        for i in range(n_chains):
            param_chain_acorr = compute_autocorrelation(vector = np.asarray(traces[parameter][:, i]).reshape(-1),
                                                        max_lags = max(lags) + 1)
            param_acorr.append(param_chain_acorr[lags])
        param_acorr = np.array(param_acorr)
        acorr_summary[parameter] = param_acorr.mean(axis = 0)

    print(acorr_summary)


def compute_autocorrelation(vector, max_lags):

    normalized_vector = vector - vector.mean()
    autocorrelation = np.correlate(normalized_vector, normalized_vector, 'full')[len(normalized_vector) - 1:]
    autocorrelation = autocorrelation/vector.var()/len(vector)
    autocorrelation = autocorrelation[:max_lags]

    return autocorrelation
