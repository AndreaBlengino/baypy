from abc import ABC, abstractmethod
from baypy.model import Model


class Regression(ABC):
    """:py:class:`Regression <baypy.regression.regression.Regression>`
    object. \n
    Abstract base class for creating regression objects.

    .. admonition:: See Also
       :class: seealso

       :py:class:`LinearRegression <baypy.regression.linear_regression.LinearRegression>`
    """

    @staticmethod
    @abstractmethod
    def sample(
        model: Model,
        n_iterations: int,
        burn_in_iterations: int,
        n_chains: int,
        seed: int = None
    ) -> None:
        if not isinstance(model, Model):
            raise TypeError(
                f"Parameter 'model' must be an instance of "
                f"'{Model.__module__}.{Model.__name__}'"
            )

        if model.data is None:
            raise ValueError("Missing 'data' in 'model'")

        if model.response_variable is None:
            raise ValueError("Missing 'response_variable' in 'model'")

        if model.response_variable not in model.data.columns:
            raise ValueError(
                f"Column '{model.response_variable}' not found in 'data'"
            )

        if model.priors is None:
            raise ValueError("Missing 'priors' in 'model'")

        for prior in model.priors.keys():
            if (prior not in ['intercept', 'variance']) and \
               (prior not in model.data.columns):
                raise ValueError(f"Column '{prior}' not found in 'model.data'")

        if not isinstance(n_iterations, int):
            raise TypeError("Parameter 'n_iteration' must be an integer")

        if not isinstance(burn_in_iterations, int):
            raise TypeError(
                "Parameter 'burn_in_iterations' must be an integer"
            )

        if not isinstance(n_chains, int):
            raise TypeError("Parameter 'n_chains' must be an integer")

        if (not isinstance(seed, int)) and (seed is not None):
            raise TypeError("Parameter 'seed' must be an integer")

        if n_iterations <= 0:
            raise ValueError("Parameter 'n_iteration' must be greater than 0")

        if burn_in_iterations < 0:
            raise ValueError(
                "Parameter 'burn_in_iterations' must be greater than or equal "
                "to 0"
            )

        if n_chains <= 0:
            raise ValueError("Parameter 'n_chains' must be greater than 0")

        if seed is not None:
            if (seed < 0) or (seed > 2**32 - 1):
                raise ValueError(
                    "Parameter 'seed' must be between 0 and 2**32 - 1"
                )
