import os 
from optimum.onnxruntime import ORTModelForCausalLM

# ───────────────────────────────────────────────────────────────
# Custom Setting
# ───────────────────────────────────────────────────────────────
MODEL_ID = "bigscience/bloom-560m"
ONNX_FP32_DIR = "onnx_bloom_fp32"

# ───────────────────────────────────────────────────────────────
# ONNX Export
# ───────────────────────────────────────────────────────────────
def export_model():
    print(f"[Export] Exporting FP32 model to {ONNX_FP32_DIR}")
    os.makedirs(ONNX_FP32_DIR, exist_ok = True)
    onnx_model : ORTModelForCausalLM = ORTModelForCausalLM.from_pretrained(
        MODEL_ID,
        export = True,
        use_cache = False,
        use_io_binding = False,
        trust_remote_code = True
    )
    onnx_model.save_pretrained(ONNX_FP32_DIR, opset_version=20)
    print(f"[Export] Completed ONNX export and saved to {ONNX_FP32_DIR}")

# ───────────────────────────────────────────────────────────────
# execution
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    export_model()
    