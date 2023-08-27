from GibbsSampler.regression.functions import sample_sigma2
from GibbsSampler.regression.functions import sample_beta
from GibbsSampler.model import Model
from .regression import Regression
import numpy as np
import pandas as pd
from ..utils import matrices_to_frame


class LinearRegression(Regression):
    r"""GibbsSampler.regression.linear_regression.LinearRegression object.

    Constructor Parameters
    ----------------------
    model : GibbsSampler.model.model.Model
        Model with data, regressors, response variable, initial values and priors to be solved through Monte Carlo
        sampling.

    Attributes
    ----------
    model : GibbsSampler.model.model.Model
        Model with data, regressors, response variable, initial values and priors to be solved through Monte Carlo
        sampling.
    posteriors : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equals to the number of iterations and a number of columns equal to the number of Markov
        chains.

    Methods
    -------
    :meth:`GibbsSampler.regression.linear_regression.LinearRegression.sample()`
        Samples a sequence of observations from the full posterior distribution of regressors' parameters
        :math:`\beta_j` and ``variance`` :math:`\sigma^2`.
    :meth:`GibbsSampler.regression.linear_regression.LinearRegression.posteriors_to_frame()`
        Organizes the ``posteriors`` in a ``pandas.DataFrame``.

    Constructor Raises
    ------------------
    TypeError
        If ``model`` is not a ``GibbsSampler.model.model.Model``.
    ValueError
        - If ``model.data`` is ``None``,
        - if ``model.response_variable`` is ``None``,
        - if ``model.initial_values`` is ``None``,
        - if ``model.priors`` is ``None``,
        - if a ``model.initial_values`` key is not a column of ``model.data``,
        - if a ``model.initial_values`` key is not a key of ``model.priors``,
        - if a ``model.priors`` key is not a column of ``model.data``,
        - if a ``model.priors`` key is not a key of ``model.initial_values``.

    See Also
    --------
    :meth:`GibbsSampler.model.linear_model.LinearModel`
    """


    def __init__(self, model: Model) -> None:
        super().__init__(model = model)


    def sample(self, n_iterations: int, burn_in_iterations: int, n_chains: int, seed: int = None) -> dict:
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

        Returns
        -------
        dict
            Returns posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a
            ``numpy.ndarray`` with ``n_iterations`` rows and ``n_chains`` columns.

        Notes
        -----
        The linear regression model of the response variable :math:`y` with respect to regressors :math:`X` is:

        .. math::
            y \sim N(\mu, \sigma^2)
        .. math::
            \mu = \beta_0 + B X = \beta_0 + \sum_{j = 1}^m \beta_j x_j

        and the likelyhood is:

        .. math::
            p \left( y \left\vert B,\sigma^2 \right. \right) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp{\frac{\left(y -
            \mu \right)^2}{\sigma^2}} .
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

        self.posteriors = {variable: [[] for _ in range(n_chains)] for variable in self.model.variable_names}

        beta = [[0 for _ in regressor_names] for _ in range(n_chains)]

        if seed is not None:
            np.random.seed(seed)

        for i in range(burn_in_iterations + n_iterations):

            sigma2 = [sample_sigma2(y = y,
                                    X = X,
                                    beta = beta[i],
                                    k_1 = k_1,
                                    theta_0 = theta_0) for i in range(n_chains)]

            beta = [sample_beta(Xt_X = Xt_X,
                                Xt_y = Xt_y,
                                sigma2 = sigma2[i],
                                Sigma_0_inv = Sigma_0_inv,
                                Sigma_0_inv_Beta_0 = Sigma_0_inv_Beta_0) for i in range(n_chains)]

            if i >= burn_in_iterations:
                for j in range(n_chains):
                    [self.posteriors[regressor][j].append(beta[j][k]) for k, regressor in enumerate(regressor_names, 0)]
                    self.posteriors['variance'][j].append(sigma2[j])

        self.posteriors = {posterior: np.array(posterior_samples).transpose() for posterior, posterior_samples in self.posteriors.items()}

        return self.posteriors


    def posteriors_to_frame(self) -> pd.DataFrame:
        """Organizes the ``posteriors`` in a ``pandas.DataFrame``. Each posterior is a frame column. The length of the
        frame is the number of sampling iterations times the number of sampling chains.

        Returns
        -------
        pandas.DataFrame
            Returns posterior samples. Posteriors are organized in a ``pandas.DataFrame``, one for each columns. The
            length of the frame is the number of sampling iterations times the number of sampling chains.

        Raises
        ------
        ValueError
            If ``posteriors`` are not available because the method ``LinearRegression.sample`` is not been called yet.
        """
        super().posteriors_to_frame()
        return matrices_to_frame(matrices_dict = self.posteriors)
