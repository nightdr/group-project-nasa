import pytest
from monte_carlo import MonteCarlo



def test_monte_carlo_init(mocker):
    model = mocker.Mock()
    theta_distribution = mocker.Mock()

    mc = MonteCarlo(model, theta_distribution)

    assert mc.model == model
    assert mc.theta_distribution == theta_distribution

@pytest.mark.parametrize('num_samples', (100, 1000))
def test_monte_carlo_sample_type(mocker, num_samples): 
    import numpy as np
    model = mocker.Mock()
    theta_distribution = mocker.Mock()

    mc = MonteCarlo(model, theta_distribution)
    samples = mc.sample(num_samples)

    assert isinstance(samples, np.ndarray)

@pytest.mark.parametrize('num_samples', range(0, 100))
def test_monte_carlo_sample_type(mocker, num_samples): 
    model = mocker.Mock()
    theta_distribution = mocker.Mock()

    mc = MonteCarlo(model, theta_distribution)
    samples = mc.sample(num_samples)

    assert samples.size == num_samples