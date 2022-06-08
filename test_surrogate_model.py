import pytest
import h5py
import sys

from surrogate_model import SurrogateModel

def test_surrogate_model_init(mocker):
    sub_model = mocker.Mock()
    model = SurrogateModel(sub_model)

    assert model.method == sub_model


def test_surrogate_model_evaluate_distance(mocker):
    sub_model = mocker.Mock()
    sub_model.predict = mocker.Mock(return_value=1.5)
    model = SurrogateModel(sub_model)
    theta = mocker.Mock()

    output = model.evaluate_distance(theta)

    sub_model.predict.assert_called_with(theta)
    assert isinstance(output, float)

def test_surrogate_model_evaluate_distance_sub_model_int(mocker):
    sub_model = mocker.Mock()
    sub_model.predict = mocker.Mock(return_value=1)
    model = SurrogateModel(sub_model)
    theta = mocker.Mock()

    output = model.evaluate_distance(theta)

    sub_model.predict.assert_called_with(theta)
    assert isinstance(output, float)

def test_surrogate_model_train_surrogate(mocker):
    sub_model = mocker.Mock()
    model = SurrogateModel(sub_model)

    hdf5_file = h5py.File('data.h5')

    model.train_surrogate(hdf5_file)

    sub_model.fit.assert_called_with(hdf5_file['\/']['explicit_model']['theta'], hdf5_file['\/']['explicit_model']['distance'])

