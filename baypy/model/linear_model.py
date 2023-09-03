from .model import Model
import pandas as pd


class LinearModel(Model):
    r"""baypy.model.model.LinearModel object.

    Attributes
    ----------
    :meth:`baypy.model.linear_model.LinearModel.data`: pandas.DataFrame
        Data for the linear regression model, is a ``pandas.DataFrame`` containing all regressor variables
        :math:`X` and the response variable :math:`y`.
    :meth:`baypy.model.linear_model.LinearModel.response_variable`: string
        Response variable :math:`y` of the linear model.
    :meth:`baypy.model.linear_model.LinearModel.priors` : dict
        Priors for the regressors' and variance parameters.
    :meth:`baypy.model.linear_model.LinearModel.variable_names` : list
        List of all model variables: the regressors :math:`X`, including the ``intercept`` and the ``variance``
        :math:`\sigma^2`.
    """


    def __init__(self):

        self.__data = None
        self.__response_variable = None
        self.__priors = None
        self.__variable_names = None


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
            - if a ``priors``' value is a empty ``dict``.
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
        Pretending to fit a linear regression of the response variable :math:`y` with respect to regressors :math:`x_1`,
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
            raise KeyError(f"Parameter 'priors' must contain a 'variance' key")

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
