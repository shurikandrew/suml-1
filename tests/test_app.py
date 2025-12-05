import pytest
import app.predict as pr

@pytest.mark.parametrize(
    "features, expected_class",
    [
        ([5.1, 3.5, 1.4, 0.2], 0),
        ([7.0, 3.2, 4.7, 1.4], 1),
        ([6.3, 3.3, 6.0, 2.5], 2),
    ]
)
def test_predict_returns_expected_class(features, expected_class):
    result = pr.predict(features)
    assert result == expected_class