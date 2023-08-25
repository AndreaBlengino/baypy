import numpy as np
import pandas as pd
from scipy.stats import invgamma, multivariate_normal
from GibbsSampler.utils import flatten_matrix


def sample_sigma2(Y: pd.Series, X: np.ndarray, beta: np.ndarray, T_1: float, theta_0: float) -> float:

    Y_X_beta = Y - flatten_matrix(np.dot(X, beta))
    theta_1 = theta_0 + np.dot(Y_X_beta.transpose(), Y_X_beta)

    return invgamma.rvs(a = T_1/2,
                        scale = theta_1/2,
                        size = 1)[0]


def sample_beta(XtX: np.ndarray, XtY: np.ndarray, sigma2: float, Sigma_0_inv: np.ndarray,
                Sigma_0_inv_Beta_0: np.ndarray) -> np.ndarray:

    V = np.linalg.inv(Sigma_0_inv + XtX/sigma2)
    M = np.dot(V, Sigma_0_inv_Beta_0 + XtY/sigma2)

    return multivariate_normal.rvs(mean = flatten_matrix(M),
                                   cov = V,
                                   size = 1)
