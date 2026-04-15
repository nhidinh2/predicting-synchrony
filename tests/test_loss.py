import numpy as np

from pipeline.evaluate.asymmetric_loss import asymmetric_mae, asymmetric_mape


def test_underprediction_penalized_more():
    y_true = np.array([100.0])
    under = asymmetric_mae(y_true, np.array([80.0]))  # underpredicted by 20
    over = asymmetric_mae(y_true, np.array([120.0]))  # overpredicted by 20
    assert under > over


def test_perfect_prediction_zero_loss():
    y = np.array([10.0, 20.0, 30.0])
    assert asymmetric_mae(y, y) == 0.0
    assert asymmetric_mape(y, y) == 0.0


def test_loss_scales_with_error():
    y = np.array([100.0])
    small = asymmetric_mae(y, np.array([95.0]))
    big = asymmetric_mae(y, np.array([80.0]))
    assert big > small
