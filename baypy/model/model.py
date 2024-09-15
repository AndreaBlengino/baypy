from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Model(ABC):
    """:py:class:`Model <baypy.model.model.Model>` object. \n
    Abstract base class for creating model objects.

    .. admonition:: See Also
       :class: seealso

       :py:class:`LinearModel <baypy.model.linear_model.LinearModel>`
    """

    @property
    @abstractmethod
    def data(self) -> None: ...

    @data.setter
    @abstractmethod
    def data(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Parameter 'data' must be an instance of 'pandas.DataFrame'"
            )

        if data.empty:
            raise ValueError(
                "Parameter 'data' cannot be an empty 'pandas.DataFrame'"
            )

    @property
    @abstractmethod
    def response_variable(self) -> None: ...

    @response_variable.setter
    @abstractmethod
    def response_variable(self, response_variable: str) -> None:
        if not isinstance(response_variable, str):
            raise TypeError("Parameter 'response_variable' must be a string")

    @property
    @abstractmethod
    def priors(self) -> None: ...

    @priors.setter
    @abstractmethod
    def priors(self, priors: dict[str, dict[str, float | int]]) -> None:
        if not isinstance(priors, dict):
            raise TypeError("Parameter 'priors' must be a dictionary")

        if len(priors) == 0:
            raise ValueError(
                "Parameter 'priors' cannot be an empty dictionary"
            )

        if 'intercept' not in priors.keys():
            raise KeyError("Parameter 'priors' must contain a 'intercept' key")

    @property
    @abstractmethod
    def variable_names(self) -> None: ...

    @property
    @abstractmethod
    def posteriors(self) -> None: ...

    @posteriors.setter
    @abstractmethod
    def posteriors(self, posteriors: dict[str, np.ndarray]) -> None:
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
            raise KeyError(
                "Parameter 'posteriors' must contain a 'intercept' key"
            )

        for posterior, posterior_samples in posteriors.items():
            if posterior_samples.size == 0:
                raise ValueError(f"Posterior '{posterior}' data is empty")

    @abstractmethod
    def posteriors_to_frame(self) -> None: ...

    @abstractmethod
    def residuals(self) -> None: ...

    @abstractmethod
    def predict_distribution(self, predictors: dict[str, float | int]) -> None:
        if not isinstance(predictors, dict):
            raise TypeError("Parameter 'predictors' must be a dictionary")

        if len(predictors) == 0:
            raise ValueError(
                "Parameter 'predictors' cannot be an empty dictionary"
            )

    @abstractmethod
    def likelihood(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Parameter 'data' must be an instance of 'pandas.DataFrame'"
            )

        if data.empty:
            raise ValueError(
                "Parameter 'data' cannot be an empty 'pandas.DataFrame'"
            )

    @abstractmethod
    def log_likelihood(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Parameter 'data' must be an instance of 'pandas.DataFrame'"
            )

        if data.empty:
            raise ValueError(
                "Parameter 'data' cannot be an empty 'pandas.DataFrame'"
            )

    @abstractmethod
    def _compute_model_parameters_at_posterior_means(self) -> None: ...

    @abstractmethod
    def _compute_model_parameters_at_observation(self, i: int) -> None: ...
