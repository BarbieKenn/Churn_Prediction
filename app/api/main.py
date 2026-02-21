from fastapi import FastAPI
import uvicorn
from app.api.pyndantic_models import ClientFeatures
import joblib
from pathlib import Path
from app.ml.transformer import Transformer
import pandas as pd


app = FastAPI()
fake_bd = []
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / 'model.joblib'


@app.on_event('startup')
def load_model():
    artifact = joblib.load(MODEL_PATH)
    app.state.pipe = artifact["model"]


@app.get('/health')
async def health():
    return {'message': 'Сайт поднят'}


@app.post('/predict')
async def predict(features: ClientFeatures):
    fake_bd.append(features)
    features_dict = features.model_dump()
    df = pd.DataFrame([features_dict])
    pred = app.state.pipe.predict(df)
    proba = app.state.pipe.predict_proba(df)[0][1]
    return {'prediction': int(pred[0]),
            'proba': float(proba)}

if __name__ == "__main__":
    uvicorn.run("app.api.main:app", reload=True, port=8004)
