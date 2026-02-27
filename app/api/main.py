from fastapi import FastAPI
import uvicorn
from app.api.pyndantic_models import ClientFeatures
import joblib
from pathlib import Path
from app.ml.transformer import Transformer
import pandas as pd
import os
import psycopg


app = FastAPI()
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / 'model.joblib'
DB_URL = os.getenv("DB_URL", "postgresql://app:app_password@localhost:5432/gym")


def insert_user(payload: dict) -> int:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into users (Age, Membership_Type, Join_Date,Last_Visit_Date, 
                Favorite_Exercise,Avg_Workout_Duration_Min, Visits_Per_Month)
                values (%s, %s, %s, %s, %s, %s, %s)
                returning id;
                """,
                (
                    payload.get('Age'),
                    payload.get('Membership_Type'),
                    payload.get('Join_Date'),
                    payload.get('Last_Visit_Date'),
                    payload.get('Favorite_Exercise'),
                    payload.get('Avg_Workout_Duration_Min'),
                    payload.get('Visits_Per_Month'),
                ),
            )
            user_id = cur.fetchone()[0]
        conn.commit()
    return user_id


def insert_prediction(user_id: int, proba: float, pred: bool) -> int:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into predictions (user_id, proba, pred)
                values(%s, %s, %s)
                returning id;
                """,
                (user_id, proba, pred),
            )
            prediction_id = cur.fetchone()[0]
        conn.commit()
    return prediction_id


@app.on_event('startup')
def load_model():
    artifact = joblib.load(MODEL_PATH)
    app.state.pipe = artifact["model"]


@app.get('/health')
async def health():
    return {'message': 'Сайт поднят'}


@app.post('/predict')
async def predict(features: ClientFeatures):
    features_dict = features.model_dump()
    df = pd.DataFrame([features_dict])
    df['Reference_Date'] = pd.Timestamp.today().normalize()
    proba = float(app.state.pipe.predict_proba(df)[0][1])
    threshold = 0.7
    pred = proba > threshold
    user_id = insert_user(features_dict)
    prediction_id = insert_prediction(user_id, proba, pred)


    return {
        "user_id": user_id,
        'prediction_id': prediction_id,
        'proba': proba,
        'pred': pred,
    }


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", reload=True, port=8004)
