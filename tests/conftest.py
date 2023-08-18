from pytest import fixture
import GibbsSampler as gs


@fixture(scope = 'function')
def model():
    model = gs.Model()
    return model


@fixture(scope = 'function')
def sampler(model):
    sampler = gs.LinearRegression(model = model)
    return sampler
