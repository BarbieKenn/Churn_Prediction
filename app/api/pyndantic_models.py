from pydantic import BaseModel, Field
from typing import Optional


class ClientFeatures(BaseModel):
    Age: int = Field(gt=0, lt=120)
    Membership_Type: str = Field(min_length=5, max_length=50)
    Join_Date: str = Field(min_length=10, max_length=10)
    Last_Visit_Date: str = Field(min_length=10, max_length=10)
    Favorite_Exercise: Optional[str] = Field(min_length=5, max_length=100)
    Avg_Workout_Duration_Min: Optional[int] = Field(gt=15, lt=600)
    Visits_Per_Month: int = Field(gt=-1, lt=100)
