from abc import ABC, abstractmethod
from .functions import sample_sigma2
from .functions import sample_beta
from .model import Model
import numpy as np


class Regression(ABC):


    @abstractmethod
    def __init__(self, model: Model) -> None:
        if not isinstance(model, Model):
            raise TypeError(f"Parameter 'model' must be an instance of '{Model.__module__}.{Model.__name__}'")

        if (model.data is None) or (model.response_variable is None):
            raise ValueError("Missing 'data' or 'response_variable' in 'model'")

        if model.initial_values is None:
            raise ValueError("Missing 'initial_values' in 'model'")

        if model.priors is None:
            raise ValueError("Missing 'priors' in 'model'")

        for initial_value in model.initial_values.keys():
            if (initial_value != 'intercept') and (initial_value not in model.data.columns):
                raise ValueError(f"Column '{initial_value}' not found in 'Model.data'")

            if initial_value not in model.priors.keys():
                raise ValueError(f"Missing '{initial_value}' prior value")

        for prior in model.priors.keys():
            if (prior not in  ['intercept', 'variance']) and (prior not in model.data.columns):
                raise ValueError(f"Column '{initial_value}' not found in 'Model.data'")

            if (prior != 'variance') and (prior not in model.initial_values.keys()):
                raise ValueError(f"Missing '{prior}' initial value")


    @abstractmethod
    def sample(self, n_iterations: int, burn_in_iterations: int, n_chains: int) -> dict:
        if not isinstance(n_iterations, int):
            raise TypeError("Parameter 'n_iteration' must be an integer")

        if not isinstance(burn_in_iterations, int):
            raise TypeError("Parameter 'burn_in_iterations' must be an integer")

        if not isinstance(n_chains, int):
            raise TypeError("Parameter 'n_chains' must be an integer")

        if n_iterations <= 0:
            raise ValueError("Parameter 'n_iteration' must be greater than 0")

        if burn_in_iterations < 0:
            raise ValueError("Parameter 'burn_in_iterations' must be greater than or equal to 0")

        if n_chains <= 0:
            raise ValueError("Parameter 'n_chains' must be greater than 0")


class LinearRegression(Regression):


    def __init__(self, model: Model) -> None:

        super().__init__(model = model)
        self.model = model
        self.posteriors = None


    def sample(self, n_iterations: int, burn_in_iterations: int, n_chains: int) -> dict:

        super().sample(n_iterations = n_iterations, burn_in_iterations = burn_in_iterations, n_chains = n_chains)
        data = self.model.data.copy()

        regressor_names = self.model.variable_names.copy()
        regressor_names.pop(regressor_names.index('variance'))

        beta_0 = [self.model.priors[x]['mean'] for x in regressor_names]
        Beta_0 = np.array(beta_0)[np.newaxis].transpose()

        sigma_0 = [self.model.priors[x]['variance'] for x in regressor_names]
        Sigma_0 = np.zeros((len(sigma_0), len(sigma_0)))
        np.fill_diagonal(Sigma_0, sigma_0)
        Sigma_0_inv = np.linalg.inv(Sigma_0)

        T_0 = self.model.priors['variance']['shape']
        theta_0 = self.model.priors['variance']['scale']

        n = len(data)
        T_1 = T_0 + n

        Y = data[self.model.response_variable]
        data['intercept'] = 1
        X = np.array(data[regressor_names])

        XtX = np.dot(X.transpose(), X)
        XtY = np.dot(X.transpose(), Y)[np.newaxis].transpose()
        Sigma_0_inv_Beta_0 = np.dot(Sigma_0_inv, Beta_0)

        self.posteriors = {variable: [[] for _ in range(n_chains)] for variable in self.model.variable_names}

        beta = [[self.model.initial_values[regressor] for regressor in regressor_names] for _ in range(n_chains)]

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
                    [self.posteriors[regressor][j].append(beta[j][k]) for k, regressor in enumerate(regressor_names, 0)]
                    self.posteriors['variance'][j].append(sigma2[j])

        self.posteriors = {posterior: np.array(posterior_samples).transpose() for posterior, posterior_samples in self.posteriors.items()}

        return self.posteriors
