import os
from onnxruntime.quantization import shape_inference

# ───────────────────────────────────────────────────────────────
# Custom Setting
# ───────────────────────────────────────────────────────────────
ONNX_FP32_DIR  = "onnx_TinyLlama_fp32"
OPT_MODEL_PATH = os.path.join(ONNX_FP32_DIR, "model_opt.onnx")

# ───────────────────────────────────────────────────────────────
# 정의부: preprocessing 함수
# ───────────────────────────────────────────────────────────────
def preprocess_onnx(input_fp32: str, output_opt: str):
    print(f"[Preprocessing] Running quant_pre_process on {input_fp32} → {output_opt}")
    shape_inference.quant_pre_process(
        input_model_path = input_fp32,
        output_model_path= output_opt,
        skip_symbolic_shape=False,
        skip_optimization    =False,
        skip_onnx_shape      =False,
        auto_merge          =True,
        verbose             =1
    )
    print(f"[Preprocessing] Saved fully optimized model to {output_opt}")

# ───────────────────────────────────────────────────────────────
# 실행부
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fp32_model = os.path.join(ONNX_FP32_DIR, "model.onnx")
    if not os.path.exists(OPT_MODEL_PATH):
        print("[Preprocessing] Starting Shape Inference and Graph Optimization...")
        preprocess_onnx(fp32_model, OPT_MODEL_PATH)
        print("[Preprocessing] Completed all preprocessing steps")
    else:
        print("[Preprocessing] Skipping (already exists)")
