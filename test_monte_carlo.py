import pytest
from monte_carlo import MonteCarlo



def test_monte_carlo_init(mocker):
    model = mocker.Mock()
    theta = mocker.Mock()
    theta_distribution = (theta,)
    
    mc = MonteCarlo(model, theta_distribution)

    assert mc.model == model
    assert mc.theta_distribution == theta_distribution


@pytest.mark.parametrize('num_samples', (100, 1000))
def test_monte_carlo_sample_type(mocker, num_samples): 
    import numpy as np
    model = mocker.Mock()
    model.evaluate_distance = mocker.Mock(return_value=1.)
    theta = mocker.Mock()
    theta_distribution = (theta,)
    
    mc = MonteCarlo(model, theta_distribution)
    samples = mc.sample(num_samples)

    assert isinstance(samples, np.ndarray)

@pytest.mark.parametrize('num_samples', range(0, 100))
def test_monte_carlo_sample_size(mocker, num_samples): 
    model = mocker.Mock()
    model.evaluate_distance = mocker.Mock(return_value=1.)
    theta = mocker.Mock()
    theta_distribution = (theta,)

    mc = MonteCarlo(model, theta_distribution)
    samples = mc.sample(num_samples)

    assert samples.size == num_samples

@pytest.mark.parametrize('num_samples', (1000,))
def test_monte_carlo_sample___model_uses_theta_dist(mocker, num_samples): 
    model = mocker.Mock()
    model.evaluate_distance = mocker.Mock(return_value=1.)
    theta = mocker.Mock(return_value=1.)
    theta_distribution = (theta,)
    
    mc = MonteCarlo(model, theta_distribution)
    mc.sample(num_samples)

    model.evaluate_distance.assert_called_with(theta)
