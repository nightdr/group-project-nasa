import pytest
from monte_carlo import MonteCarlo



def test_init_monte_carlo(mocker):
    model = mocker.Mock()
    theta_distribution = mocker.Mock()

    mc = MonteCarlo(model, theta_distribution)

    assert mc.model == model
    assert mc.theta_distribution == theta_distribution
