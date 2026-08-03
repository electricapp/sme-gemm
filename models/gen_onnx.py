# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx"]
# ///
"""Generate single-MatMul ONNX models (dynamic M/K/N) for the ORT benchmark.
Run: uv run models/gen_onnx.py"""

import os

import onnx
from onnx import TensorProto, helper


def make(dtype: int, fname: str) -> None:
    a = helper.make_tensor_value_info("A", dtype, ["M", "K"])
    b = helper.make_tensor_value_info("B", dtype, ["K", "N"])
    c = helper.make_tensor_value_info("C", dtype, ["M", "N"])
    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "gemm", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, fname)
    print("wrote", fname)


here = os.path.dirname(os.path.abspath(__file__))
make(TensorProto.FLOAT, os.path.join(here, "gemm_f32.onnx"))
make(TensorProto.FLOAT16, os.path.join(here, "gemm_f16.onnx"))
