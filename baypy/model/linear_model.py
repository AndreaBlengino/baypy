from .model import Model
import pandas as pd
import numpy as np
from scipy.stats import norm
from ..utils import flatten_matrix, matrices_to_frame, dot_product


class LinearModel(Model):
    r"""baypy.model.model.LinearModel object.

    Attributes
    ----------
    :py:attr:`baypy.model.linear_model.LinearModel.data` : pandas.DataFrame
        Data for the linear regression model, is a ``pandas.DataFrame`` containing all regressor variables
        :math:`X` and the response variable :math:`y`.
    :py:attr:`baypy.model.linear_model.LinearModel.response_variable` : string
        Response variable :math:`y` of the linear model.
    :py:attr:`baypy.model.linear_model.LinearModel.priors` : dict
        Priors for the regressors' and variance parameters.
    :py:attr:`baypy.model.linear_model.LinearModel.variable_names` : list
        List of all model variables: the regressors :math:`X`, including the ``intercept`` and the ``variance``
        :math:`\sigma^2`.
    :py:attr:`baypy.model.linear_model.LinearModel.posteriors` : dict
        Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equals to the number of iterations and a number of columns equal to the number of Markov
        chains.

    Methods
    -------
    :py:meth:`baypy.model.linear_model.LinearModel.posteriors_to_frame()`
        Organizes the ``posteriors`` in a ``pandas.DataFrame``.
    :py:meth:`baypy.model.linear_model.LinearModel.residuals`
        Compute the residuals :math:`\epsilon` with respect to predicted values :math:`\hat{y}`.
    :py:meth:`baypy.model.linear_model.LinearModel.predict_distribution`
        Predicts a posterior distribution for an unobserved values.
    :py:meth:`baypy.model.linear_model.LinearModel.likelihood`
        Computes the likelihood of observations ``model.response_variable`` given a model ``'mean'`` and ``'variance'``.
    :py:meth:`baypy.model.linear_model.LinearModel.log_likelihood`
        Computes the log likelihood of observations ``model.response_variable`` given a model ``'mean'`` and
        ``'variance'``.
    """


    def __init__(self):

        self.__data = None
        self.__response_variable = None
        self.__priors = None
        self.__variable_names = None
        self.__posteriors = None


    @property
    def data(self) -> pd.DataFrame:
        r"""Data for the linear regression model, is a ``pandas.DataFrame`` containing all regressor variables
        :math:`X` and the response variable :math:`y`.

        Returns
        -------
        pandas.DataFrame
            Observed data of the model. It cannot be empty. It must contain regressor variables :math:`X` and the
            response variable :math:`y`.

        Raises
        ------
        TypeError
            If ``data`` is not an instance of ``pandas.DataFrame``.
        ValueError
            If ``data`` is an empty ``pandas.DataFrame``.
        """
        assert super().data is None
        return self.__data


    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        super(LinearModel, type(self)).data.fset(self, data)
        self.__data = data


    @property
    def response_variable(self) -> str:
        r"""Response variable :math:`y` of the linear model.

        Returns
        -------
        string
            Name of the response variable :math:`y`. In must be one of the columns of ``data``.

        Raises
        ------
        TypeError
            If ``response_variable`` is not a ``str``.
        """
        assert super().response_variable is None
        return self.__response_variable


    @response_variable.setter
    def response_variable(self, response_variable: str) -> None:
        super(LinearModel, type(self)).response_variable.fset(self, response_variable)
        self.__response_variable = response_variable


    @property
    def priors(self) -> dict:
        r"""Priors for the regressors' and variance parameters.
        Each prior is a key-value pair, where the value is a ``dict`` with:

            - hyperparameter names as keys
            - hyperparameter values as values.

        Returns
        -------
        dict
            Priors for each random variable. It must contain an ``intercept`` and a ``variance`` keys. Each value must
            be a ``dict`` with hyperparameter names as key and hyperparameter values as values.

        Raises
        ------
        TypeError
            - If ``priors`` is not a ``dict``,
            - if a ``priors``' value is not a ``dict``.
        ValueError
            - If ``priors`` is an empty ``dict``,
            - if a ``priors``' value is a empty ``dict``,
            - if a ``variance`` value is not positive,
            - if a ``shape`` value is not positive,
            - if a ``scale`` value is not positive.
        KeyError
            - If ``priors`` does not contain both ``intercept`` and ``variance`` keys,
            - if a prior's hyperparameters are not:
                + ``mean`` and ``variance`` for a regression parameter :math:`\beta_j` or
                + ``shape`` and ``scale`` for ``variance`` :math:`\sigma^2`.

        Notes
        -----
        To each random variables is assigned a prior distribution:

            - to each regressor parameter :math:`\beta_j` is assigned a normal prior distribution with hyperparameters
              ``mean`` :math:`\beta_j^0` and ``variance`` :math:`\Sigma_{\beta_j}^0`:

              .. math::
                \beta_j \sim N(\beta_j^0 , \Sigma_{\beta_j}^0)

            - to variance :math:`\sigma^2` is assigned a inverse gamma distribution with hyperparameters ``shape``
              :math:`\kappa^0` and ``scale`` :math:`\theta^0`:

              .. math::
                \sigma^2 \sim \text{Inv-}\Gamma(\kappa^0, \theta^0)

        Examples
        --------
        Consider a linear regression of the response variable :math:`y` with respect to regressors :math:`x_1`,
        :math:`x_2` and :math:`x_3`, according to the following model:

        .. math::
            y \sim N(\mu, \sigma^2)
        .. math::
            \mu = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3

        then the sampler would require priors for:
            - parameter :math:`\beta_0` of variable ``intercept``, with ``mean`` :math:`\beta_0^0` and ``variance`` :math:`\Sigma_{\beta_0}^0`
            - parameter :math:`\beta_1` of variable :math:`x_1`, with ``mean`` :math:`\beta_1^0` and ``variance`` :math:`\Sigma_{\beta_1}^0`
            - parameter :math:`\beta_2` of variable :math:`x_2`, with ``mean`` :math:`\beta_2^0` and ``variance`` :math:`\Sigma_{\beta_2}^0`
            - parameter :math:`\beta_3` of variable :math:`x_3`, with ``mean`` :math:`\beta_3^0` and ``variance`` :math:`\Sigma_{\beta_3}^0`
            - variable :math:`\sigma^2`, with ``shape`` :math:`\kappa^0` and ``scale`` :math:`\theta^0`

        >>> model = baypy.model.LinearModel()
        >>> model.set_priors({'intercept': {'mean': 0, 'variance': 1e6},
        ...                   'x_1': {'mean': 0, 'variance': 1e6},
        ...                   'x_2': {'mean': 0, 'variance': 1e6},
        ...                   'x_3': {'mean': 0, 'variance': 1e6},
        ...                   'variance': {'shape': 1, 'scale': 1e-6}})
        """
        assert super().priors is None
        return self.__priors


    @priors.setter
    def priors(self, priors: dict) -> None:
        super(LinearModel, type(self)).priors.fset(self, priors)

        if 'variance' not in priors.keys():
            raise KeyError("Parameter 'priors' must contain a 'variance' key")

        for prior, values in priors.items():
            if not isinstance(values, dict):
                raise TypeError(f"The value of prior '{prior}' must be a dictionary")
            if len(values) == 0:
                raise ValueError(f"The value of prior '{prior}' cannot be an empty dictionary")
            if prior != 'variance':
                if set(values.keys()) != {'mean', 'variance'}:
                    raise KeyError(f"The value of prior '{prior}' must be a dictionary "
                                   f"containing 'mean' and 'variance' keys only")
                if values['variance'] <= 0:
                    raise ValueError(f"The 'variance' of prior '{prior}' must be positive")
            else:
                if set(values.keys()) != {'shape', 'scale'}:
                    raise KeyError(f"The value of prior '{prior}' must be a dictionary "
                                   f"containing 'shape' and 'scale' keys only")
                for parameter in ['shape', 'scale']:
                    if values[parameter] <= 0:
                        raise ValueError(f"The '{parameter}' of prior '{prior}' must be positive")

        self.__priors = priors
        self.__variable_names = list(priors.keys())
        self.__variable_names.insert(0, self.__variable_names.pop(self.__variable_names.index('intercept')))


    @property
    def variable_names(self) -> list:
        r"""Variables of the linear model.

        Returns
        -------
        list
            List of all model variables: the regressors :math:`X`, including the ``intercept`` and the ``variance``
            :math:`\sigma^2`.
        """
        assert super().variable_names is None
        return self.__variable_names


    @property
    def posteriors(self) -> dict:
        r"""Posteriors of the regressors' and variance parameters.
        Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
        with a number of rows equals to the number of iterations and a number of columns equal to the number of Markov
        chains.

        Returns
        -------
        dict
            Posterior samples. Posteriors and relative samples are key-value pairs. Each sample is a ``numpy.ndarray``
            with a number of rows equals to the number of iterations and a number of columns equal to the number of
            Markov chains.

        Raises
        ------
        TypeError
            - If ``posteriors`` is not a ``dict``,
            - if a posterior sample is not a ``numpy.ndarray``.
        KeyError
            If ``posteriors`` does not contain both ``intercept`` and ``variance`` keys.
        ValueError
            If a posterior sample is an empty ``numpy.ndarray``.
        """
        assert super().posteriors is None
        return self.__posteriors


    @posteriors.setter
    def posteriors(self, posteriors: dict) -> None:
        super(LinearModel, type(self)).posteriors.fset(self, posteriors)

        if 'variance' not in posteriors.keys():
            raise KeyError("Parameter 'posteriors' must contain a 'variance' key")

        self.__posteriors = posteriors


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
            If ``posteriors`` are not available because the method ``baypy.regression.LinearRegression.sample`` is not
            been called yet.
        """
        assert super().posteriors_to_frame() is None
        if self.__posteriors is None:
            raise ValueError("Posteriors not available, run 'baypy.regression.LinearRegression.sample' to generate "
                             "posteriors")
        return matrices_to_frame(matrices_dict = self.__posteriors)


    def residuals(self) -> pd.DataFrame:
        r"""Compute the residuals :math:`\epsilon` with respect to predicted values :math:`\hat{y}`.

        Returns
        -------
        pandas.DataFrame
            Returns a copy of ``data`` with 3 more columns: ``intercept``, ``predicted`` and ``residuals``.

        Raises
        ------
        ValueError
            - If ``model.data`` is ``None`` because the property ``baypy.model.LinearModel.data`` has not been set
            - if ``model.response_variable`` is not a column of ``model.data``,
            - If a ``model.posteriors`` is ``None`` because the sampling has not been done yet.

        Notes
        -----
        Predicted values are computed at data points :math:`X` using the posteriors means for each regressor's
        parameter:

        .. math::
            \hat{y_i} = \beta_0 + \sum_{j = 1}^{m} \beta_j x_{i,j}

        while residuals are the difference between the observed values and the predicted values of the
        ``response_variable``:

        .. math::
            \epsilon_i = y_i - \hat{y_i}
        """
        assert super().residuals() is None

        if self.__data is None:
            raise ValueError("Data not available, set data with 'baypy.model.LinearModel.data")

        if self.__response_variable not in self.__data.columns:
            raise ValueError(f"Column '{self.__response_variable}' not found in 'data'")

        if self.__posteriors is None:
            raise ValueError("Posteriors not available, run 'baypy.regression.LinearRegression.sample' to generate "
                             "posteriors")

        posterior_means = {posterior: flatten_matrix(posterior_samples).mean()
                           for posterior, posterior_samples in self.__posteriors.items() if posterior != 'variance'}

        data = self.__data.copy()
        data['intercept'] = 1
        data['predicted'] = dot_product(data = data, regressors = posterior_means)
        data['residuals'] = data[self.__response_variable] - data['predicted']

        return data


    def predict_distribution(self, predictors: dict) -> np.ndarray:
        """Predicts a posterior distribution for an unobserved values. For each posterior sample, it draws a sample from
         the likelihood.

        Parameters
        ----------
        predictors : dict
            Values of predictors :math:`X` at which compute the posterior distribution. Each predictor has to be set as
            a key-value pair.

        Returns
        -------
        numpy.ndarray
            Array of the predicted posterior distribution. It contains a number of element equal to the number of
            regression iterations times the number of model Markov chains.

        Raises
        ------
        TypeError
            If ``predictors`` is not a ``dict``.
        KeyError
            If a ``predictors`` key is not a key of ``posteriors``.
        ValueError
            If ``predictors`` is an empty ``dict``.

        See Also
        --------
        :py:class:`baypy.regression.linear_regression.LinearRegression`
        """
        super().predict_distribution(predictors = predictors)

        for regressor in predictors.keys():
            if regressor not in self.__posteriors.keys():
                raise KeyError(f"Regressor '{regressor}' not found in 'posteriors' keys")

        prediction = matrices_to_frame(matrices_dict = self.__posteriors)
        prediction['mean'] = prediction['intercept'] + dot_product(data = prediction, regressors = predictors)

        return norm.rvs(loc = prediction['mean'],
                        scale = np.sqrt(prediction['variance']),
                        size = len(prediction))


    def likelihood(self, data: pd.DataFrame) -> np.ndarray:
        r"""Computes the likelihood of observations ``model.response_variable`` given a model ``'mean'`` and
        ``'variance'``.

        Parameters
        ----------
        data: pandas.DataFrame
            Data to use for likelihood computation. It cannot be empty. It must contain columns
            ``model.response_variable``, ``'mean'`` and ``'variance'``.

        Returns
        -------
        numpy.ndarray
            Array of computed likelihood. It has the same length of ``data``. Each element is a likelihood computation
            of each row of ``data``.

        Raises
        ------
        TypeError
            If ``data`` is not an instance of ``pandas.DataFrame``.
        ValueError
            - If ``data`` is an empty ``pandas.DataFrame``,
            - if ``model.response_variable`` is not a column of ``data``,
            - if ``'mean'`` is not a column of ``data``,
            - if ``'variance'`` is not a column of ``data``.

        Notes
        -----
        The likelihood is computed with the normal distribution probability density function:

        .. math::
            L(y) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp{- \frac{\left(y - \mu \right)^2}{2 \sigma^2}}

        where :math:`\mu` is the ``'mean'`` column and :math:`\sigma^2` is the ``'variance'`` column.
        """
        super().likelihood(data = data)

        for col in [self.__response_variable, 'mean', 'variance']:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in 'data'")

        return norm.pdf(x = data[self.__response_variable],
                        loc = data['mean'],
                        scale = np.sqrt(data['variance']))


    def log_likelihood(self, data: pd.DataFrame) -> np.ndarray:
        r"""Computes the log likelihood of observations ``model.response_variable`` given a model ``'mean'`` and
        ``'variance'``.

        Parameters
        ----------
        data: pandas.DataFrame
            Data to use for log likelihood computation. It cannot be empty. It must contain columns
            ``model.response_variable``, ``'mean'`` and ``'variance'``.

        Returns
        -------
        numpy.ndarray
            Array of computed log likelihood. It has the same length of ``data``. Each element is a log likelihood
            computation of each row of ``data``.

        Raises
        ------
        TypeError
            If ``data`` is not an instance of ``pandas.DataFrame``.
        ValueError
            - If ``data`` is an empty ``pandas.DataFrame``,
            - if ``model.response_variable`` is not a column of ``data``,
            - if ``'mean'`` is not a column of ``data``,
            - if ``'variance'`` is not a column of ``data``.

        Notes
        -----
        The log likelihood is computed as the log of the normal distribution probability density function:

        .. math::
            l(y) = - \frac{1}{2} \log{2 \pi \sigma^2} - \frac{1}{2} \frac{\left(y - \mu \right)^2}{\sigma^2}

        where :math:`\mu` is the ``'mean'`` column and :math:`\sigma^2` is the ``'variance'`` column.
        """
        super().log_likelihood(data = data)

        for col in [self.__response_variable, 'mean', 'variance']:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in 'data'")

        return norm.logpdf(x = data[self.__response_variable],
                           loc = data['mean'],
                           scale = np.sqrt(data['variance']))


    def _compute_model_parameters_at_posterior_means(self) -> pd.DataFrame:

        posterior_means = {posterior: flatten_matrix(posterior_samples).mean()
                           for posterior, posterior_samples in self.__posteriors.items() if posterior != 'variance'}

        data = self.__data.copy()
        data['intercept'] = 1
        data['mean'] = dot_product(data = data, regressors = posterior_means)
        data['variance'] = flatten_matrix(self.__posteriors['variance']).mean()

        return data


    def _compute_model_parameters_at_observation(self, i: int):

        posterior_means = {posterior: posterior_samples[i, :].mean()
                           for posterior, posterior_samples in self.__posteriors.items() if posterior != 'variance'}

        data = self.__data.copy()
        data['intercept'] = 1
        data['mean'] = dot_product(data = data, regressors = posterior_means)
        data['variance'] = self.__posteriors['variance'][i, :].mean()

        return data
