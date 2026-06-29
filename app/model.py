import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'


class BearingClassifier:
    def __init__(self) -> None:
        config_path = ARTIFACTS_DIR / 'model_config.json'
        model_path  = ARTIFACTS_DIR / 'model.onnx'

        with open(config_path) as f:
            config = json.load(f)

        self.class_names = config['class_names']
        self.input_name  = config['input_name']
        self.output_name = config['output_name']
        self.window_size = config['window_size']

        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider'],
        )

    def predict(self, signal: list[float]) -> dict:
        x = np.array(signal, dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-8)   # per-window z-score, matches training
        x = x[np.newaxis, np.newaxis, :]           # (1, 1, 1024)

        t0 = time.perf_counter()
        logits = self.session.run([self.output_name], {self.input_name: x})[0][0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        exp   = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        label = int(probs.argmax())

        return {
            'predicted_class': self.class_names[label],
            'label': label,
            'probabilities': {
                name: float(p) for name, p in zip(self.class_names, probs)
            },
            'inference_time_ms': round(elapsed_ms, 3),
        }
