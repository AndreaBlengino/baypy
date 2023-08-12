import numpy as np
from scipy.stats import gamma, multivariate_normal


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
