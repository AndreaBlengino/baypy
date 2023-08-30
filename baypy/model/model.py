from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):


    @abstractmethod
    def __init__(self):

        self.__data = None
        self.__response_variable = None
        self.__priors = None
        self.__variable_names = None


    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        return self.__data


    @data.setter
    @abstractmethod
    def data(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Parameter 'data' must be an instance of 'pandas.DataFrame'")

        if data.empty:
            raise ValueError("Parameter 'data' cannot be an empty 'pandas.DataFrame'")

        self.__data = data


    @property
    @abstractmethod
    def response_variable(self) -> str:
        return self.__response_variable


    @response_variable.setter
    @abstractmethod
    def response_variable(self, response_variable: str) -> None:
        if not isinstance(response_variable, str):
            raise TypeError("Parameter 'response_variable' must be a string")

        self.__response_variable = response_variable


    @property
    @abstractmethod
    def priors(self) -> dict:
        return self.__priors


    @priors.setter
    @abstractmethod
    def priors(self, priors: dict) -> None:
        if not isinstance(priors, dict):
            raise TypeError("Parameter 'priors' must be a dictionary")

        if len(priors) == 0:
            raise ValueError("Parameter 'priors' cannot be an empty dictionary")

        for prior in ['intercept', 'variance']:
            if prior not in priors.keys():
                raise KeyError(f"Parameter 'priors' must contain a '{prior}' key")

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
        self.__variable_names = list(self.priors.keys())
        self.__variable_names.insert(0, self.__variable_names.pop(self.__variable_names.index('intercept')))


    @property
    @abstractmethod
    def variable_names(self) -> list:
        return self.__variable_names
