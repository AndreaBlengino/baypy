import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def trace_plot(posteriors: dict[str, np.ndarray]) -> None:
    """It plots the traces and the probability density for each posterior. \n
    The plot shows the traces for each Markov chain, for each regression
    variable and the relative posterior density.
    The plot layout has number of rows equal to the number of regression
    variables and two columns: traces on the left and densities on the right.

    Parameters
    ----------
    ``posteriors`` : :py:class:`dict`
        Posterior samples. Posteriors and relative samples are key-value
        pairs. Each sample is a :py:class:`numpy.ndarray` with a number of rows
        equal to the number of iterations and a number of columns equal to the
        number of Markov chains.

    .. admonition:: Raises
       :class: warning

       ``TypeError``
           - If ``posteriors`` is not a :py:class:`dict`,
           - if a posterior sample is not a :py:class:`numpy.ndarray`.
       ``KeyError``
           If ``posteriors`` does not contain ``'intercept'`` key.
       ``ValueError``
           If a posterior sample is an empty :py:class:`numpy.ndarray`.

    .. admonition:: See Also
       :class: seealso

       :py:class:`LinearRegression <baypy.regression.linear_regression.LinearRegression>`
    """
    if not isinstance(posteriors, dict):
        raise TypeError("Parameter 'posteriors' must be a dictionary")

    if not all(
        [
            isinstance(posterior_sample, np.ndarray)
            for posterior_sample in posteriors.values()
        ]
    ):
        raise TypeError(
            "All posteriors data must be an instance of 'numpy.ndarray'"
        )

    if 'intercept' not in posteriors.keys():
        raise KeyError("Parameter 'posteriors' must contain a 'intercept' key")

    for posterior, posterior_samples in posteriors.items():
        if posterior_samples.size == 0:
            raise ValueError(f"Posterior '{posterior}' data is empty")

    variable_names = list(posteriors.keys())
    n_variables = len(variable_names)
    n_iterations = len(posteriors['intercept'])

    fig = plt.figure(figsize=(10, min(1.5*n_variables + 2, 10)))
    trace_axes = []
    for j, variable in zip(range(1, 2*n_variables, 2), variable_names):
        ax_j_trace = fig.add_subplot(n_variables, 2, j)
        ax_j_density = fig.add_subplot(n_variables, 2, j + 1)

        ax_j_trace.plot(posteriors[variable], linewidth=0.5)
        ax_j_density.plot(*_compute_kde(posteriors[variable].flatten()))

        ax_j_trace.set_title(f"Trace of {variable} parameter")
        ax_j_density.set_title(f"Density of {variable} parameter")
        ax_j_trace.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False
        )
        ax_j_density.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False
        )
        ax_j_density.set_ylim(0, )
        trace_axes.append(ax_j_trace)

    for ax_j in trace_axes[1:]:
        ax_j.sharex(trace_axes[0])
    trace_axes[0].set_xlim(0, n_iterations - 1)

    plt.tight_layout()

    plt.show()


def _compute_kde(posterior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posterior_support = np.linspace(np.min(posterior), np.max(posterior), 1000)
    posterior_kde = gaussian_kde(posterior)(posterior_support)

    return posterior_support, posterior_kde
