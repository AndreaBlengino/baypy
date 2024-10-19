import numpy as np
import pandas as pd
from scipy.stats import invgamma, multivariate_normal
from baypy.utils import flatten_matrix


def sample_sigma2(
    y: pd.Series,
    X: np.ndarray,
    beta: np.ndarray,
    k_1: float,
    theta_0: float
) -> float:
    r"""It draws a sample for the variance from an inverse-gamma conditional
    posterior distribution at each regression iteration.

    Parameters
    ----------
    ``y`` : :py:class:`pandas.Series`
        Vector of the response variable :math:`y`. It has a number of rows
        equal to the number of observed data.
    ``X`` : :py:class:`numpy.ndarray`
        Regressors matrix :math:`X`. It has a number of rows equal to the
        number of observed data and a number of columns equal to the number of
        regressors.
    ``beta`` : :py:class:`numpy.ndarray`
        Regressors' parameters array of the model. It is sampled by the
        function
        :py:func:`sample_beta <baypy.regression.functions.sample_beta>` at each
        iteration.
    ``k_1`` : :py:class:`float`
        Sum of the variance prior ``'shape'`` and the number of observed data.
    ``theta_0`` : :py:class:`float`
        Variance prior ``'scale'``.

    Returns
    -------
    :py:class:`numpy.ndarray`
        Sampled variance :math:`\sigma^2`.

    .. admonition:: Notes
       :class: tip

       The variance is drawn from an inverse-gamma distribution. The
       conditional posterior distribution is:

       .. math::
           H \left( \sigma^2 \left\vert B, y \right. \right) \sim
           \text{Inv-}\Gamma \left( \frac{\kappa_1}{2}, \frac{\theta_1}{2}
           \right)

       where :math:`B` is the :math:`\beta_j` vector, :math:`\kappa_1` is:

       .. math::
           \kappa_1 = \kappa_0 + n

       where :math:`n` is the number of observed data and :math:`\theta_1` is:

       .. math::
           \theta_1 = \theta_0 + \left( y - X B \right)^T \left( y - X B
           \right) .

    .. admonition:: See Also
       :class: seealso

       :py:func:`sample_beta <baypy.regression.functions.sample_beta>`
    """
    y_X_beta = y - flatten_matrix(np.dot(X, beta))
    theta_1 = theta_0 + np.dot(y_X_beta.transpose(), y_X_beta)

    return invgamma.rvs(a=k_1/2, scale=theta_1/2)


def sample_beta(
    Xt_X: np.ndarray,
    Xt_y: np.ndarray,
    sigma2: float,
    Sigma_0_inv: np.ndarray,
    Sigma_0_inv_Beta_0: np.ndarray
) -> np.ndarray:
    r"""it draws a sample for each regressor parameter from a normal
    conditional posterior distribution at each regression iteration.

    Parameters
    ----------
    ``Xt_X`` : :py:class:`numpy.ndarray`
       Dot product of transposed regressors with themselves: :math:`X^T X`. It
       is a symmetric matrix with a number of rows and columns equal to the
       number of regressors.
    ``Xt_y`` : :py:class:`numpy.ndarray`
        Dot product of transposed regressors with the response variable:
        :math:`X^T y`. It is a one-dimensional array with a number of rows
        equal to the number of regressors.
    ``sigma2`` : :py:class:`float`
        Variance of the model. It is sampled by the function
        :py:func:`sample_sigma2 <baypy.regression.functions.sample_sigma2>` at
        each iteration.
    ``Sigma_0_inv`` : :py:class:`numpy.ndarray`
        Regressors' parameters variance priors matrix. It is a symmetric matrix
        with a number of rows and columns equal to the number of regressors. It
        is the inverse of a matrix which has regressors' parameters' prior
        variances on the diagonal and ``0`` everywhere else.
    ``Sigma_0_inv_Beta_0`` : :py:class:`numpy.ndarray`
        Dot product of ``Sigma_0_inv`` with a one-dimensional vector which has
        regressors' parameters' prior means on rows.

    Returns
    -------
    :py:class:`numpy.ndarray`
        Array of the sampled regressors' parameters :math:`B`. It has a number
        of element equal to the number of regressors.

    .. admonition:: Notes
       :class: tip

       The regressors' parameters are drawn from a multivariate normal
       distribution. The conditional posterior distribution is:

       .. math::
           H \left( B \left\vert \sigma^2, y \right. \right) \sim N(M, V)

       where :math:`M` is a one-dimensional vector with a number of rows equal
       to the number of regressors:

       .. math::
           M = \left( \Sigma_0^{-1} + \frac{1}{\sigma^2} X^T X \right)^{-1}
           \left( \Sigma_0^{-1} B_0 + \frac{1}{\sigma^2} X^T y \right)

       and :math:`V` is a symmetric matrix with a number of rows and columns
       equal to the number of regressors:

       .. math::
           V = \left( \Sigma_0^{-1} + \frac{1}{\sigma^2} X^T X \right)^{-1} .

       :math:`B_0` is:

       .. math::
           B_0 =
           \begin{bmatrix}
               \beta_0^0 \\
               \beta_1^0 \\
               \vdots    \\
               \beta_m^0
           \end{bmatrix}

       and :math:`\Sigma_0^{-1}` is:

       .. math::
           \Sigma_0^{-1} =
           \begin{bmatrix}
               \Sigma_{\beta_0}^0 & 0 & \dots & 0 \\
               0 & \Sigma_{\beta_1}^0 & \dots & 0 \\
               \vdots & \vdots & \ddots & 0       \\
               0 & 0 & 0 & \Sigma_{\beta_m}^0
           \end{bmatrix}

    .. admonition:: See Also
       :class: seealso

       :py:func:`sample_sigma2 <baypy.regression.functions.sample_sigma2>`
    """
    V = np.linalg.inv(Sigma_0_inv + Xt_X/sigma2)
    M = np.dot(V, Sigma_0_inv_Beta_0 + Xt_y/sigma2)
    beta = multivariate_normal.rvs(mean=flatten_matrix(M), cov=V, size=1)

    if Xt_X.shape != (1, 1):
        return beta
    else:
        return np.array([beta])
