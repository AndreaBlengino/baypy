from pytest import mark
import numpy as np
import pandas as pd


np.random.seed(42)

N = 50
data = pd.DataFrame()
data['x_1'] = np.random.uniform(low = 0, high = 100, size = N)
data['x_2'] = np.random.uniform(low = -10, high = 10, size = N)
data['x_3'] = np.random.uniform(low = -50, high = -40, size = N)
data['x_1 * x_2'] = data['x_1']*data['x_2']

data['y'] = 3*data['x_1'] - 20*data['x_2'] - data['x_3'] - 5*data['x_1 * x_2'] + 13 + 1*np.random.randn(N)


@mark.set_input
def test_set_data(sampler):
    sampler.set_data(data = data,
                     y_name = 'y')

    assert data.equals(sampler.data)
    assert 'y' == sampler.y_name
