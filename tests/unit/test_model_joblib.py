import pytest
import joblib


def test_artifact():
    artifact = joblib.load('model.joblib')
    assert list(artifact.keys()) == ['model', 'features', 'best_params', 'cv_score']

