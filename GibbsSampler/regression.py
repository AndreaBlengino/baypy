from .functions import sample_sigma2
from .functions import sample_beta
from .model import Model
import numpy as np


class LinearRegression:


    def __init__(self, model):

        if not isinstance(model, Model):
            raise TypeError("Parameter 'model' must be an instance of 'GibbsSampler.model.Model'")

        self.model = model
        self.posteriors = None


    def sample(self, n_iterations, burn_in_iterations, n_chains):

        if n_iterations <= 0:
            raise ValueError("Parameter 'n_iteration' must be greater than 0")

        if burn_in_iterations < 0:
            raise ValueError("Parameter 'burn_in_iterations' must be greater than or equal to 0")

        if n_chains <= 0:
            raise ValueError("Parameter 'n_chains' must be greater than 0")

        data = self.model.data.copy()

        regressor_names = self.model.variable_names.copy()
        regressor_names.pop(regressor_names.index('sigma2'))

        beta_0 = [self.model.priors[x]['mean'] for x in regressor_names]
        Beta_0 = np.array(beta_0)[np.newaxis].transpose()

        sigma_0 = [self.model.priors[x]['variance'] for x in regressor_names]
        Sigma_0 = np.zeros((len(sigma_0), len(sigma_0)))
        np.fill_diagonal(Sigma_0, sigma_0)
        Sigma_0_inv = np.linalg.inv(Sigma_0)

        T_0 = self.model.priors['sigma2']['shape']
        theta_0 = self.model.priors['sigma2']['scale']

        n = len(data)
        T_1 = T_0 + n

        Y = data[self.model.y_name]
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
                    self.posteriors['sigma2'][j].append(sigma2[j])

        self.posteriors = {variable: np.array(posterior).transpose() for variable, posterior in self.posteriors.items()}

        return self.posteriors
