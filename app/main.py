from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from app.model import BearingClassifier
from app.schemas import HealthResponse, PredictRequest, PredictResponse

PREDICTION_COUNTER = Counter(
    'bearing_predictions_total',
    'Bearing fault predictions broken down by predicted class',
    ['predicted_class'],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = BearingClassifier()
    yield


app = FastAPI(title='PdM-Lite', version='0.1.0', lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.get('/health', response_model=HealthResponse)
def health() -> dict:
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> dict:
    result = request.app.state.classifier.predict(req.signal)
    PREDICTION_COUNTER.labels(predicted_class=result['predicted_class']).inc()
    return result
