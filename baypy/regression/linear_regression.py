from baypy.regression.functions import sample_sigma2
from baypy.regression.functions import sample_beta
from baypy.model import Model
from .regression import Regression
import numpy as np
import pandas as pd
from scipy.stats import norm, invgamma


class LinearRegression(Regression):
    r"""baypy.regression.linear_regression.LinearRegression object.

    Parameters
    ----------
    model : baypy.model.model.Model
        Model with data, regressors, response variable and priors to be solved through Monte Carlo sampling.

    Attributes
    ----------
    model : baypy.model.model.Model
        Model with data, regressors, response variable and priors to be solved through Monte Carlo sampling.

    Methods
    -------
    :meth:`baypy.regression.linear_regression.LinearRegression.sample()`
        Samples a sequence of observations from the full posterior distribution of regressors' parameters
        :math:`\beta_j` and ``variance`` :math:`\sigma^2`.

    Raises
    ------
    TypeError
        If ``model`` is not a ``baypy.model.model.Model``.
    ValueError
        - If ``model.data`` is ``None``,
        - if ``model.response_variable`` is ``None``,
        - if ``model.response_variable`` is not a column of ``model.data``
        - if ``model.priors`` is ``None``,
        - if a ``model.priors`` key is not a column of ``model.data``,

    See Also
    --------
    :meth:`baypy.model.linear_model.LinearModel`
    """


    def __init__(self, model: Model) -> None:
        super().__init__(model = model)
        self.model = model


    def sample(self, n_iterations: int, burn_in_iterations: int, n_chains: int, seed: int = None) -> None:
        r"""Samples a sequence of observations from the full posterior distribution of regressors' parameters
        :math:`\beta_j` and ``variance`` :math:`\sigma^2`.
        First ``burn_in_iterations`` are discarded since they may not accurately represent the desired distribution.
        For each variable, it generates ``n_chain`` Markov chains.

        Parameters
        ----------
        n_iterations : int
            Number of total sampling iteration for each chain. It must be a strictly positive integer.
        burn_in_iterations : int
            Number of burn-in iteration for each chain. It must be a positive integer or ``0``.
        n_chains : int
            Number of chains. It must be a strictly positive integer.
        seed : int, optional
            Random seed to use for reproducibility of the sampling.

        Raises
        ------
        TypeError
            - If ``n_iterations`` is not a ``int``,
            - if ``burn_in_iterations`` is not a ``int``,
            - if ``n_chains`` is not a ``int``,
            - if ``seed`` is not a ``int``.
        ValueError
            - If ``n_iterations`` is equal to or less than ``0``,
            - if ``burn_in_iterations`` is less than ``0``,
            - if ``n_chains`` is equal to or less than ``0``,
            - if ``seed`` is not between ``0`` and ``2**32 - 1``.

        Notes
        -----
        The linear regression model of the response variable :math:`y` with respect to regressors :math:`X` is:

        .. math::
            y \sim N(\mu, \sigma^2)
        .. math::
            \mu = \beta_0 + B X = \beta_0 + \sum_{j = 1}^m \beta_j x_j

        and the likelihood is:

        .. math::
            p \left( y \left\vert B,\sigma^2 \right. \right) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp{- \frac{\left(y -
            \mu \right)^2}{2 \sigma^2}} .
        """
        super().sample(n_iterations = n_iterations, burn_in_iterations = burn_in_iterations,
                       n_chains = n_chains, seed = seed)
        data = self.model.data.copy()

        regressor_names = self.model.variable_names.copy()
        regressor_names.pop(regressor_names.index('variance'))

        beta_0 = [self.model.priors[x]['mean'] for x in regressor_names]
        Beta_0 = np.array(beta_0)[np.newaxis].transpose()

        sigma_0 = [self.model.priors[x]['variance'] for x in regressor_names]
        Sigma_0 = np.zeros((len(sigma_0), len(sigma_0)))
        np.fill_diagonal(Sigma_0, sigma_0)
        Sigma_0_inv = np.linalg.inv(Sigma_0)

        k_0 = self.model.priors['variance']['shape']
        theta_0 = self.model.priors['variance']['scale']

        n = len(data)
        k_1 = k_0 + n

        y = data[self.model.response_variable]
        data['intercept'] = 1
        X = np.array(data[regressor_names])

        Xt_X = np.dot(X.transpose(), X)
        Xt_y = np.dot(X.transpose(), y)[np.newaxis].transpose()
        Sigma_0_inv_Beta_0 = np.dot(Sigma_0_inv, Beta_0)

        posteriors = {variable: [[] for _ in range(n_chains)] for variable in self.model.variable_names}

        if seed is not None:
            np.random.seed(seed)

        beta = [[norm.rvs(loc = self.model.priors[regressor]['mean'],
                          scale = np.sqrt(self.model.priors[regressor]['variance']))
                 for regressor in regressor_names] for _ in range(n_chains)]
        sigma2 = [invgamma.rvs(a = k_0, scale = theta_0) for _ in range(n_chains)]

        for i in range(burn_in_iterations + n_iterations + 1):
            for k in range(n_chains):
                [posteriors[regressor][k].append(beta[k][j])
                 for j, regressor in enumerate(regressor_names, 0)]
                posteriors['variance'][k].append(sigma2[k])

            beta = [sample_beta(Xt_X = Xt_X,
                                Xt_y = Xt_y,
                                sigma2 = sigma2[k],
                                Sigma_0_inv = Sigma_0_inv,
                                Sigma_0_inv_Beta_0 = Sigma_0_inv_Beta_0) for k in range(n_chains)]

            sigma2 = [sample_sigma2(y = y,
                                    X = X,
                                    beta = beta[k],
                                    k_1 = k_1,
                                    theta_0 = theta_0) for k in range(n_chains)]

        self.model.posteriors = {posterior: np.array(posterior_samples).transpose()[burn_in_iterations + 1:, :]
                                 for posterior, posterior_samples in posteriors.items()}
