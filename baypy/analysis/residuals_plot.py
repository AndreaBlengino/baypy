import matplotlib.pyplot as plt
from ..model import Model


def residuals_plot(model: Model) -> None:
    r"""It plots the residuals :math:`\epsilon` with respect to predicted
    values :math:`\hat{y}`.

    Parameters
    ----------
    ``model`` : :py:class:`Model <baypy.model.model.Model>`
        The model with data, regressors, response variable and priors to be
        solved through Monte Carlo sampling.

    .. admonition:: Raises
       :class: warning

       ``TypeError``
           If ``model`` is not a :py:class:`Model <baypy.model.model.Model>`.
       ``ValueError``
           - If a ``model.posteriors`` is :py:obj:`None` because the sampling
             has not been done yet,
           - if a posterior key is not a column of ``model.data``,
           - if ``model.data`` is an empty :py:class:`pandas.DataFrame`,
           - if ``model.response_variable`` is not a column of ``model.data``.

    .. admonition:: Notes
       :class: tip

       Predicted values are computed at data points :math:`X` using the
       posteriors means for each regressor's parameter. In the case of linear
       model:

       .. math::
           \hat{y_i} = \beta_0 + \sum_{j = 1}^{m} \beta_j x_{i,j}

       while residuals are the difference between the observed values and the
       predicted values of the ``response_variable``:

       .. math::
           \epsilon_i = y_i - \hat{y_i}

    .. admonition:: See Also
       :class: seealso

       :py:class:`LinearRegression <baypy.regression.linear_regression.LinearRegression>`
    """
    if not isinstance(model, Model):
        raise TypeError(
            f"Parameter 'model' must be an instance of "
            f"'{Model.__module__}.{Model.__name__}'"
        )

    if model.posteriors is None:
        raise ValueError(
            "Posteriors not available, run "
            "'baypy.regression.Regression.sample' to generate posteriors"
        )

    for posterior in model.posteriors.keys():
        if (posterior not in ['intercept', 'variance']) and \
           (posterior not in model.data.columns):
            raise ValueError(f"Column '{posterior}' not found in 'model.data'")

    if model.data.empty:
        raise ValueError(
            "Parameter 'model.data' cannot be an empty 'pandas.DataFrame'"
        )

    if model.response_variable not in model.data.columns:
        raise ValueError(
            f"Column '{model.response_variable}' not found in 'model.data'"
        )

    data = model.residuals()

    _, ax = plt.subplots()

    ax.plot(
        data['predicted'],
        data['residuals'],
        marker='o',
        linestyle='',
        alpha=0.5
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')

    ax.tick_params(
        bottom=False,
        top=False,
        left=False,
        right=False
    )

    plt.tight_layout()

    plt.show()
