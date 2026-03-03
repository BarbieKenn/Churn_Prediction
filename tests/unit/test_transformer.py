import pytest
from app.ml.transformer import Transformer
import pandas as pd


@pytest.fixture
def payload_basic() -> pd.DataFrame:
    data_dict = {
        'Age': 20,
        'Membership_Type': 'Monthly',
        'Join_Date': '2025-10-10',
        'Last_Visit_Date': '2026-10-10',
        'Favorite_Exercise': 'Squads',
        'Avg_Workout_Duration_Min': 120,
        'Visits_Per_Month': 15
    }
    return pd.DataFrame([data_dict])


def test_output_columns(payload_basic: pd.DataFrame) -> None:
    input_data = payload_basic.copy()
    transformer = Transformer()
    actual_output_data = transformer.transform(input_data).columns
    expected_output_data = input_data.drop(['Join_Date', 'Last_Visit_Date'], axis=1)
    expected_output_data['Days_Since_Last_Visit'] = 3
    expected_output_data['Membership_Days'] = 3
    assert list(actual_output_data) == list(expected_output_data.columns)


# 2025-05-30
def test_neg_date(payload_basic: pd.DataFrame) -> None:
    data = payload_basic.copy()
    data['Last_Visit_Date'] = '2025-10-20'
    transformer = Transformer()
    actual = transformer.transform(data)['Days_Since_Last_Visit'].values
    assert actual[0] == 0
