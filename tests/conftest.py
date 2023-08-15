from pytest import fixture
from GibbsSampler import GibbsSampler


@fixture(scope = 'function')
def sampler():
    sampler = GibbsSampler()
    return sampler
