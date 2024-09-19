from ..model import Model
import numpy as np


def compute_DIC(model: Model, print_summary: bool = True) -> dict[str, float]:
    r"""It computes and prints the Deviance Information Criterion (DIC) for
    the fitted model.

    Parameters
    ----------
    ``model`` : :py:class:`Model <baypy.model.model.Model>`
        The model with data, regressors, response variable and priors to be
        solved through Monte Carlo sampling.
    ``print_summary`` : :py:class:`bool`, optional
        If ``True`` prints the deviance summary report. Default is ``True``.

    Returns
    -------
    :py:class:`dict`
        Dictionary with deviance summary. It contains:
            - key ``'Deviance at posterior means'``,
            - key ``'Posterior mean deviance'``,
            - key ``'Effective number of parameters'``,
            - key ``'Deviance Information Criterion'``.

    .. admonition:: Raises
       :class: warning

       ``TypeError``
           - If ``model`` is not a :py:class:`Model <baypy.model.model.Model>`,
           - if ``print_summary`` is not a :py:class:`bool`.
       ``ValueError``
           - If a ``model.posteriors`` is :py:obj:`None` because the sampling
             has not been done yet,
           - if a posterior key is not a column of ``model.data``,
           - if ``model.data`` is an empty :py:class:`pandas.DataFrame`,
           - if ``model.response_variable`` is not a column of ``model.data``.

    .. admonition:: Notes
       :class: tip

       The DIC measures posterior predictive error by penalizing the fit of a
       model (deviance) by its complexity, determined by the effective number
       of parameters.
       Comparing some alternative models, the smaller the DIC of a model, the
       *better* the model.
       Consider a linear regression of the response variable :math:`y` with
       respect to regressors :math:`X`, according to the following model:

       .. math::
           y \sim N(\mu, \sigma^2)
       .. math::
           \mu = \beta_0 + B X = \beta_0 + \sum_{j = 1}^m \beta_j x_j

       then the *likelyhood* is:

       .. math::
           p \left( y \left\vert B,\sigma^2 \right. \right) = \frac{1}{\sqrt{2
           \pi \sigma^2}} \exp{- \frac{\left(y - \mu \right)^2}{2 \sigma^2}} .

       The *deviance* [1]_ [2]_ is defined as:

       .. math::
           D \left( y, B, \sigma^2 \right) = -2\log p \left( y \left\vert B,
           \sigma^2 \right. \right) .

       The *deviance* at posterior mean of :math:`B` and :math:`\sigma^2`,
       denoted by :math:`\overline{B}` and :math:`\overline{\sigma^2}` is:

       .. math::
           D_{{\overline{\beta}}, \overline{\sigma^2}} (y) = D \left( y,
           \overline{B}, \overline{\sigma^2} \right)

       while the posterior mean deviance is:

       .. math::
           \overline{D} \left( y, B, \sigma^2 \right) = E \left( D(y, B,
           \sigma^2) \left. \right\vert y \right) .

       and the *effective number of parameter* is defined as:

       .. math::
           pD = \overline{D} \left( y, B, \sigma^2 \right) -
           D_{{\overline{\beta}}, \overline{\sigma^2}} (y) .

       The *Deviance Information Criterion* [1]_ is:

       .. math::
           DIC = 2 \overline{D} \left( y, B, \sigma^2 \right) -
           D_{{\overline{\beta}}, \overline{\sigma^2}} (y) =
           \overline{D} \left( y, B, \sigma^2 \right) + pD =
           D_{{\overline{B}}, \overline{\sigma^2}} (y) + 2pD .

    .. admonition:: References
       :class: note

       .. [1] O. Spiegelhalter DJ, Best NG, Carlin BP, van der Linde A.
          Bayesian measures of model complexity and fit. J R Statist Soc B.
          2002;64:583–616.
       .. [2] Gelman A, Carlin JB, Stern HS, Rubin DS. Bayesian Data Analysis.
          2. Chapman & Hall/CRC; Boca Raton, Florida: 2004.

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

    if not isinstance(print_summary, bool):
        raise TypeError("Parameter 'print_summary' must be a boolean")

    for posterior, _ in model.posteriors.items():
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

    deviance_at_posterior_means = _compute_deviance_at_posterior_means(
        model=model
    )
    posterior_mean_deviance = _compute_posterior_mean_deviance(model=model)
    effective_number_of_parameters = \
        posterior_mean_deviance - deviance_at_posterior_means
    DIC = effective_number_of_parameters + posterior_mean_deviance

    if print_summary:
        dic_output = (
            f"Deviance at posterior means     "
            f"{deviance_at_posterior_means:>12.2f}\n"
            f"Posterior mean deviance         "
            f"{posterior_mean_deviance:>12.2f}\n"
            f"Effective number of parameters  "
            f"{effective_number_of_parameters:>12.2f}\n"
            f"Deviance Information Criterion  {DIC:>12.2f}"
        )
        print(dic_output)

    return {
        "Deviance at posterior means": deviance_at_posterior_means,
        "Posterior mean deviance": posterior_mean_deviance,
        "Effective number of parameters": effective_number_of_parameters,
        "Deviance Information Criterion": DIC
    }


def _compute_deviance_at_posterior_means(model: Model) -> float:
    data = model._compute_model_parameters_at_posterior_means()
    likelihood = model.likelihood(data=data)

    return -2*np.sum(np.log(likelihood))


def _compute_posterior_mean_deviance(model: Model) -> float:
    deviance = []
    for i in range(model.posteriors['intercept'].shape[0]):
        data = model._compute_model_parameters_at_observation(i)
        likelihood = model.likelihood(data=data)
        deviance.append(-2*np.sum(np.log(likelihood)))

    return np.mean(deviance)
