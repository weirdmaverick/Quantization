import os
from onnxruntime.quantization import shape_inference

# ───────────────────────────────────────────────────────────────
# Preprocessing
# ───────────────────────────────────────────────────────────────
ONNX_FP32_DIR = "onnx_bloom_fp32"
OPT_MODEL_FP32 = os.path.join(ONNX_FP32_DIR, "model_optimized.onnx")

def preprocess(input_fp32:str, output_opt: str):
    print(f"[Preprocessing] Run preprocessing on {input_fp32} → {output_opt}")
    shape_inference.quant_pre_process(
        input_model_path = input_fp32,
        output_model_path = output_opt,
        skip_symbolic_shape = True,
        skip_optimization = True,
        skip_onnx_shape = True,
        auto_merge = False,
        verbose = 1        
    )
    print(f"[Preprocessing] Saved optimized model to {output_opt}")
    
# ───────────────────────────────────────────────────────────────
# Execution
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fp32_model = os.path.join(ONNX_FP32_DIR, "model.onnx")
    optimized_fp32_model = os.path.join(ONNX_FP32_DIR, "model_optimized.onnx")
    if not os.path.exists(OPT_MODEL_FP32):
        print("[Preprocessing] Start Processing")
        preprocess(fp32_model, OPT_MODEL_FP32)
        print("[Preprocessing] Completed all preprocessing steps")
    else:
        print("[Preprocessing] Skip preprocessing (already exists)")
  