from .model import Model
import pandas as pd


class LinearModel(Model):
    """GibbsSampler.model.model.LinearModel object.

    Attributes
    ----------
    :meth:`GibbsSampler.model.linear_model.LinearModel.data`: pandas.DataFrame
        Observed data of the model. It cannot be empty. It must contain regressor variables :math:`X` and the
        response variable :math:`y`.
    :meth:`GibbsSampler.model.linear_model.LinearModel.response_variable`: string
        Name of the response variable :math:`y`. In must be one of the columns of ``data``.
    :meth:`GibbsSampler.model.linear_model.LinearModel.priors` : dict
        Priors for each random variable. It must contain an ``intercept`` and a ``variance`` keys. Each value must
        be a ``dict`` with hyperparameter names as key and hyperparameter values as values.
    :meth:`GibbsSampler.model.linear_model.LinearModel.variable_names` : list
        List of all model variables: the regressors :math:`X`, including the ``intercept`` and the ``variance``
        :math:`\sigma^2`.
    """


    def __init__(self):
        super().__init__()


    @property
    def data(self) -> pd.DataFrame:
        r"""Sets data for the linear regression model. The parameter ``data`` is a ``pandas.DataFrame`` containing all
        regressor variables :math:`X` and the response variable :math:`y`.

        Returns
        -------
        pandas.DataFrame
            Observed data of the model. It cannot be empty. It must contain regressor variables :math:`X` and the
            response variable :math:`y`.

        Setter Raises
        -------------
        TypeError
            If ``data`` is not an instance of ``pandas.DataFrame``.
        ValueError
            If ``data`` is an empty ``pandas.DataFrame``.
        """
        return super().data


    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        super(LinearModel, type(self)).data.fset(self, data)


    @property
    def response_variable(self) -> str:
        r"""Sets the response variable :math:`y` of the linear model.

        Returns
        -------
        string
            Name of the response variable :math:`y`. In must be one of the columns of ``data``.

        Setter Raises
        -------------
        TypeError
            If ``response_variable`` is not a ``str``.
        """
        return super().response_variable


    @response_variable.setter
    def response_variable(self, response_variable: str) -> None:
        super(LinearModel, type(self)).response_variable.fset(self, response_variable)


    @property
    def priors(self) -> dict:
        r"""Sets priors for the regressors' and variance parameters.
        Each prior has to be set as a key-value pair, where the value is a ``dict`` with:

             - hyperparameter names as keys
             - hyperparameter values as values.

        Returns
        -------
        dict
            Priors for each random variable. It must contain an ``intercept`` and a ``variance`` keys. Each value must
            be a ``dict`` with hyperparameter names as key and hyperparameter values as values.

        Setter Raises
        -------------
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
        To each random variables is assigned a prior
        distribution:

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

        >>> model = GibbsSampler.model.LinearModel()
        >>> model.set_priors({'intercept': {'mean': 0, 'variance': 1e6},
        ...                   'x_1': {'mean': 0, 'variance': 1e6},
        ...                   'x_2': {'mean': 0, 'variance': 1e6},
        ...                   'x_3': {'mean': 0, 'variance': 1e6},
        ...                   'variance': {'shape': 1, 'scale': 1e-6}})
        """
        return super().priors


    @priors.setter
    def priors(self, priors: dict) -> None:
        super(LinearModel, type(self)).priors.fset(self, priors)


    @property
    def variable_names(self) -> list:
        r"""Variables of the linear model.

        Returns
        -------
        list
            List of all model variables: the regressors :math:`X`, including the ``intercept`` and the ``variance``
            :math:`\sigma^2`.
        """
        return super().variable_names
