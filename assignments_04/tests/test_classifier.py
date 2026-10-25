from pathlib import Path

import joblib
import pandas as pd
import pytest

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import Prediction, WeatherClassifier


FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

GOOD_DAY = {
    "temperature_2m_max": 20.0,
    "temperature_2m_min": 10.0,
    "precipitation_sum": 0.0,
    "wind_speed_10m_max": 5.0,
}

BAD_DAY = {
    "temperature_2m_max": -2.0,
    "temperature_2m_min": -8.0,
    "precipitation_sum": 15.0,
    "wind_speed_10m_max": 50.0,
}

SECOND_GOOD_DAY = {
    "temperature_2m_max": 22.0,
    "temperature_2m_min": 12.0,
    "precipitation_sum": 0.5,
    "wind_speed_10m_max": 8.0,
}


@pytest.fixture
def classifier(tmp_path: Path) -> WeatherClassifier:
    """Create, save, and load a tiny predictable classification pipeline."""
    X = pd.DataFrame(
        [GOOD_DAY, BAD_DAY, SECOND_GOOD_DAY],
        columns=FEATURES,
    )
    y = [1, 0, 1]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=1)),
        ]
    )
    pipeline.fit(X, y)

    model_path = tmp_path / "tiny_classifier.pkl"
    joblib.dump(pipeline, model_path)

    return WeatherClassifier(model_path)


# Behaviors 1 and 2
@pytest.mark.parametrize(
    "day, expected_label",
    [
        pytest.param(GOOD_DAY, "good", id="clearly-good"),
        pytest.param(BAD_DAY, "skip", id="cold-rainy-windy"),
    ],
)
def test_predict_labels(classifier: WeatherClassifier,day: dict[str, float],expected_label: str,) -> None:
    """Mapping model classes to the expected readable labels."""
    result = classifier.predict([day])

    assert len(result) == 1
    assert result[0].label == expected_label


# Behavior 3
def test_probability_is_valid(classifier: WeatherClassifier) -> None:
    """Returning P(good), within the valid probability range."""
    results = classifier.predict([GOOD_DAY, BAD_DAY])

    for result in results:
        assert isinstance(result.probability, float)
        assert 0.0 <= result.probability <= 1.0

    # With this exact-match k=1 fixture, P(good) is known.
    # These assertions also catch accidentally returning P(skip).
    assert results[0].probability == pytest.approx(1.0)
    assert results[1].probability == pytest.approx(0.0)


# Behavior 4
def test_three_predictions_preserve_order(classifier: WeatherClassifier,) -> None:
    """Returning three Prediction objects in the same order as their inputs."""
    days = [GOOD_DAY, BAD_DAY, SECOND_GOOD_DAY]
    results = classifier.predict(days)

    assert len(results) == 3
    assert all(isinstance(result, Prediction) for result in results)
    assert [result.label for result in results] == [
        "good",
        "skip",
        "good",
    ]

    # Reordering the inputs must reorder the corresponding outputs.
    reordered_results = classifier.predict([BAD_DAY, SECOND_GOOD_DAY, GOOD_DAY])
    assert reordered_results == [results[1], results[2], results[0]]


# Behavior 5
def test_empty_input_returns_empty_list(classifier: WeatherClassifier,) -> None:
    """Handling an empty prediction batch."""
    assert classifier.predict([]) == []


# Behavior 6
@pytest.mark.parametrize("missing_feature", FEATURES)
def test_missing_feature_raises(classifier: WeatherClassifier, missing_feature: str,) -> None:
    """Name any required feature missing from an individual day."""
    incomplete_day = GOOD_DAY.copy()
    del incomplete_day[missing_feature]

    # Include a complete day first to ensure validation checks every day,
    # rather than only checking the DataFrame's combined column names.
    with pytest.raises(ValueError, match=missing_feature):
        classifier.predict([GOOD_DAY, incomplete_day])


# Behavior 7
def test_missing_model_file_raises(tmp_path: Path) -> None:
    """Provides a helpful error when the saved model does not exist."""
    missing_path = tmp_path / "missing_classifier.pkl"

    with pytest.raises(FileNotFoundError, match="model file not found"):
        WeatherClassifier(missing_path)

#=================================================== test session starts ====================================================
#platform win32 -- Python 3.12.15, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\alexp\OneDrive\Documents\Code_The_Dream\Cloud_AI_26.4\python200-homework\.venv\Scripts\python.exe
#cachedir: .pytest_cache
#rootdir: C:\Users\alexp\OneDrive\Documents\Code_The_Dream\Cloud_AI_26.4\python200-homework
#configfile: pyproject.toml
#collected 10 items                                                                                                          

#tests\test_classifier.py::test_predict_labels[clearly-good] PASSED                                                    [ 10%]
#tests\test_classifier.py::test_predict_labels[cold-rainy-windy] PASSED                                                [ 20%]
#tests\test_classifier.py::test_probability_is_valid PASSED                                                            [ 30%]
#tests\test_classifier.py::test_three_predictions_preserve_order PASSED                                                [ 40%]
#tests\test_classifier.py::test_empty_input_returns_empty_list PASSED                                                  [ 50%]
#tests\test_classifier.py::test_missing_feature_raises[temperature_2m_max] PASSED                                      [ 60%]
#tests\test_classifier.py::test_missing_feature_raises[temperature_2m_min] PASSED                                      [ 70%]
#tests\test_classifier.py::test_missing_feature_raises[precipitation_sum] PASSED                                       [ 80%]
#tests\test_classifier.py::test_missing_feature_raises[wind_speed_10m_max] PASSED                                      [ 90%]
#tests\test_classifier.py::test_missing_model_file_raises PASSED                                                       [100%]

#=================================================== 10 passed in 15.62s ====================================================

#=================================================== test session starts ====================================================
#platform win32 -- Python 3.12.15, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\alexp\OneDrive\Documents\Code_The_Dream\Cloud_AI_26.4\python200-homework\.venv\Scripts\python.exe
#cachedir: .pytest_cache
#rootdir: C:\Users\alexp\OneDrive\Documents\Code_The_Dream\Cloud_AI_26.4\python200-homework
#configfile: pyproject.toml
#collected 2 items                                                                                                           

#tests\test_classifier.py::test_predict_labels[clearly-good] FAILED                                                    [ 50%]
#tests\test_classifier.py::test_predict_labels[cold-rainy-windy] PASSED                                                [100%]

#========================================================= FAILURES =========================================================
#____________________________________________ test_predict_labels[clearly-good] _____________________________________________

#classifier = <weather_model.classifier.WeatherClassifier object at 0x000001A4A75C7A70>
#day = {'temperature_2m_max': 20.0, 'temperature_2m_min': 10.0, 'precipitation_sum': 0.0, 'wind_speed_10m_max': 5.0}
#expected_label = 'good'

#    @pytest.mark.parametrize(
#        "day, expected_label",
#        [
#            pytest.param(GOOD_DAY, "good", id="clearly-good"),
#            pytest.param(BAD_DAY, "skip", id="cold-rainy-windy"),
#        ],
#    )
#    def test_predict_labels(classifier: WeatherClassifier,day: dict[str, float],expected_label: str,) -> None:
#        """Mapping model classes to the expected readable labels."""
#        result = classifier.predict([day])
#    
#        assert len(result) == 1
#>       assert result[0].label == expected_label
#E       AssertionError: assert 'skip' == 'good'
#E         
#E         - good
#E         + skip

#tests\test_classifier.py:79: AssertionError
#================================================= short test summary info ==================================================
#FAILED tests\test_classifier.py::test_predict_labels[clearly-good] - AssertionError: assert 'skip' == 'good'
#=============================================== 1 failed, 1 passed in 2.88s ================================================

# Deliberate break check:
# Temporarily made WeatherClassifier.predict() always return "skip".
# test_predict_labels[clearly-good] caught the change because it expected
# "good" but received "skip". Restored the mapping and confirmed that
# it passes again. This test is a good example of how to catch accidental changes to the code.