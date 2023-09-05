from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):


    @property
    @abstractmethod
    def data(self) -> None: ...


    @data.setter
    @abstractmethod
    def data(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Parameter 'data' must be an instance of 'pandas.DataFrame'")

        if data.empty:
            raise ValueError("Parameter 'data' cannot be an empty 'pandas.DataFrame'")


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
    def priors(self, priors: dict) -> None:
        if not isinstance(priors, dict):
            raise TypeError("Parameter 'priors' must be a dictionary")

        if len(priors) == 0:
            raise ValueError("Parameter 'priors' cannot be an empty dictionary")

        if 'intercept' not in priors.keys():
            raise KeyError(f"Parameter 'priors' must contain a 'intercept' key")


    @property
    @abstractmethod
    def variable_names(self) -> None: ...


    @property
    @abstractmethod
    def posteriors(self) -> None: ...


    @posteriors.setter
    @abstractmethod
    def posteriors(self, posteriors: dict) -> None: ...


    @abstractmethod
    def posteriors_to_frame(self) -> None: ...


    @abstractmethod
    def compute_model_parameters_at_posterior_means(self) -> None: ...


    @abstractmethod
    def compute_model_parameters_at_observation(self, i: int) -> None: ...


    @abstractmethod
    def likelihood(self, data: pd.DataFrame) -> None: ...


    @abstractmethod
    def log_likelihood(self, data: pd.DataFrame) -> None: ...
