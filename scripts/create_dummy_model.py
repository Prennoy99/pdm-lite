"""Generates a dummy ONNX model with correct I/O shapes for CI and local dev.

Run once to create a placeholder when the real Colab-trained model is not yet
available (e.g. fresh clone before running notebooks/train_colab.ipynb).

Usage:
    python scripts/create_dummy_model.py
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ARTIFACTS_DIR = Path(__file__).parent.parent / 'app' / 'artifacts'
CLASS_NAMES   = ['normal', 'inner_race', 'ball', 'outer_race']
WINDOW_SIZE   = 1024


def _build_model() -> onnx.ModelProto:
    rng = np.random.default_rng(seed=0)

    W         = rng.standard_normal((1024, 4)).astype(np.float32)
    b         = np.zeros(4, dtype=np.float32)
    shape_val = np.array([-1, 1024], dtype=np.int64)

    signal_in  = helper.make_tensor_value_info('signal', TensorProto.FLOAT, [None, 1, 1024])
    logits_out = helper.make_tensor_value_info('logits', TensorProto.FLOAT, [None, 4])

    graph = helper.make_graph(
        nodes=[
            helper.make_node('Reshape', inputs=['signal', 'reshape_shape'], outputs=['flat']),
            helper.make_node('Gemm',    inputs=['flat', 'W', 'b'],          outputs=['logits']),
        ],
        name='dummy_bearing_classifier',
        inputs=[signal_in],
        outputs=[logits_out],
        initializer=[
            numpy_helper.from_array(shape_val, name='reshape_shape'),
            numpy_helper.from_array(W,         name='W'),
            numpy_helper.from_array(b,         name='b'),
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    onnx.save(_build_model(), str(ARTIFACTS_DIR / 'model.onnx'))
    print(f'Saved dummy model  → {ARTIFACTS_DIR / "model.onnx"}')

    config = {
        'class_names': CLASS_NAMES,
        'input_name':  'signal',
        'output_name': 'logits',
        'window_size': WINDOW_SIZE,
        'n_classes':   len(CLASS_NAMES),
    }
    with open(ARTIFACTS_DIR / 'model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Saved model config → {ARTIFACTS_DIR / "model_config.json"}')


if __name__ == '__main__':
    main()
