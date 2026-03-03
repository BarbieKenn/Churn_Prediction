import pytest

from app.api.pyndantic_models import ClientFeatures


@pytest.fixture
def payload_basic() -> dict:
    feats = {
        'Age': 20,
        'Membership_Type': 'Monthly',
        'Join_Date': '2025-10-10',
        'Last_Visit_Date': '2026-10-10',
        'Favorite_Exercise': 'Squads',
        'Avg_Workout_Duration_Min': 120,
        'Visits_Per_Month': 15
    }
    return feats


def test_valid_feats_to_model(payload_basic) -> None:
    actual_feats = ClientFeatures(**payload_basic)
    assert actual_feats.model_dump() == payload_basic


def test_invalid_age(payload_basic) -> None:
    data = payload_basic.copy()
    data['Age'] = 0

    with pytest.raises(ValueError):
        ClientFeatures(**data)


def test_invalid_data(payload_basic) -> None:
    data = payload_basic.copy()
    data['Join_Date'] = '123456789'

    with pytest.raises(ValueError):
        ClientFeatures(**data)


def test_optional_field(payload_basic) -> None:
    data = payload_basic.copy()
    data['Favorite_Exercise'] = None
    data['Avg_Workout_Duration_Min'] = None
    actual_feats = ClientFeatures(**data)
    assert actual_feats.model_dump() == data


