from .model import Model
import pandas as pd


class LinearModel(Model):


    def __init__(self):

        super().__init__()


    @property
    def data(self) -> pd.DataFrame:
        return super().data


    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        super(LinearModel, type(self)).data.fset(self, data)


    @property
    def response_variable(self) -> str:
        return super().response_variable


    @response_variable.setter
    def response_variable(self, response_variable: str) -> None:
        super(LinearModel, type(self)).response_variable.fset(self, response_variable)


    @property
    def priors(self) -> dict:
        return super().priors


    @priors.setter
    def priors(self, priors: dict) -> None:
        super(LinearModel, type(self)).priors.fset(self, priors)


    @property
    def variable_names(self) -> list:
        return super().variable_names
