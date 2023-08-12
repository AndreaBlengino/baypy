import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, multivariate_normal, gaussian_kde


def sampler(n_iterations, burn_in_iterations, data, y_name, initial_values, initial_value_intercept, prior):

    beta_0 = [prior[x]['mean'] for x in list(initial_values.keys())]
    beta_0.insert(0, prior['intercept']['mean'])
    Beta_0 = np.matrix(beta_0).transpose()

    sigma_0 = [prior[x]['variance'] for x in list(initial_values.keys())]
    sigma_0.insert(0, prior['intercept']['variance'])
    Sigma_0 = np.zeros((len(sigma_0), len(sigma_0)))
    np.fill_diagonal(Sigma_0, sigma_0)
    Sigma_0_inv = np.linalg.inv(Sigma_0)

    T_0 = prior['sigma2']['shape']
    theta_0 = prior['sigma2']['rate']

    n = len(data)
    T_1 = T_0 + n

    Y = data[y_name]
    X = np.asmatrix(data[list(initial_values.keys())])
    X = np.hstack((np.ones((n, 1)), X))

    XtX = np.dot(X.transpose(), X)
    XtY = np.dot(X.transpose(), Y).transpose()
    Sigma_0_inv_Beta_0 = np.dot(Sigma_0_inv, Beta_0)


    traces = {parameter: [] for parameter in list(initial_values.keys())}
    traces['intercept'] = []
    traces['sigma2'] = []

    beta = np.hstack((initial_value_intercept, list(initial_values.values())))

    for _ in range(burn_in_iterations):

        sigma2 = sample_sigma2(Y = Y,
                               X = X,
                               beta = beta,
                               T_1 = T_1,
                               theta_0 = theta_0)

        beta = sample_beta(XtX = XtX,
                           XtY = XtY,
                           sigma2 = sigma2,
                           Sigma_0_inv = Sigma_0_inv,
                           Sigma_0_inv_Beta_0 = Sigma_0_inv_Beta_0)

    for _ in range(n_iterations):

        sigma2 = sample_sigma2(Y = Y,
                               X = X,
                               beta = beta,
                               T_1 = T_1,
                               theta_0 = theta_0)

        beta = sample_beta(XtX = XtX,
                           XtY = XtY,
                           sigma2 = sigma2,
                           Sigma_0_inv = Sigma_0_inv,
                           Sigma_0_inv_Beta_0 = Sigma_0_inv_Beta_0)


        traces['sigma2'].append(sigma2)
        traces['intercept'].append(beta[0])
        [traces[x_k].append(beta[k]) for k, x_k in enumerate(list(initial_values.keys()), 1)]

    return traces


def sample_sigma2(Y, X, beta, T_1, theta_0):

    Y_X_beta = Y - np.asarray(np.dot(X, beta)).reshape(-1)
    theta_1 = theta_0 + np.dot(Y_X_beta.transpose(), Y_X_beta)

    return 1/gamma.rvs(a = T_1/2,
                       scale = 1/(theta_1/2),
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
    ax_intercept_density.plot(*compute_kde(traces['intercept']))

    ax_intercept_trace.set_title('Trace of intercept')
    ax_intercept_density.set_title('Density of intercept')

    for i, x_i in zip(range(3, 2*n_variables - 2, 2), x_names):
        ax_i_trace = fig.add_subplot(n_variables, 2, i, sharex = ax_intercept_trace)
        ax_i_density = fig.add_subplot(n_variables, 2, i + 1)

        ax_i_trace.plot(traces[x_i], linewidth = 0.5)
        ax_i_density.plot(*compute_kde(traces[x_i]))

        ax_i_trace.set_title(f'Trace of {x_i}')
        ax_i_density.set_title(f'Density of {x_i}')

    ax_sigma2_trace = fig.add_subplot(n_variables, 2, 2*n_variables - 1, sharex = ax_intercept_trace)
    ax_sigma2_density = fig.add_subplot(n_variables, 2, 2*n_variables)

    ax_sigma2_trace.plot(traces['sigma2'], linewidth = 0.5)
    ax_sigma2_density.plot(*compute_kde(traces['sigma2']))

    ax_sigma2_trace.set_title(r'Trace of $\sigma^2$')
    ax_sigma2_density.set_title('Density of $\sigma^2$')

    ax_intercept_trace.set_xlim(0, n_iterations)

    plt.tight_layout()

    plt.show()

def compute_kde(trace):
    posterior_support = np.linspace(np.min(trace), np.max(trace), 1000)
    posterior_kde = gaussian_kde(trace)(posterior_support)

    return posterior_support, posterior_kde
