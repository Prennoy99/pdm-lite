from pydantic import BaseModel, field_validator

WINDOW_SIZE = 1024


class PredictRequest(BaseModel):
    signal: list[float]

    @field_validator('signal')
    @classmethod
    def must_be_1024_samples(cls, v: list[float]) -> list[float]:
        if len(v) != WINDOW_SIZE:
            raise ValueError(f'signal must contain exactly {WINDOW_SIZE} samples, got {len(v)}')
        return v


class PredictResponse(BaseModel):
    predicted_class: str
    label: int
    probabilities: dict[str, float]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
