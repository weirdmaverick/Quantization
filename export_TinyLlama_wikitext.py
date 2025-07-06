import os
import datetime
from optimum.onnxruntime import ORTModelForCausalLM

# ───────────────────────────────────────────────────────────────
# Custom Setting
# ───────────────────────────────────────────────────────────────
MODEL_ID      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ONNX_FP32_DIR = "onnx_TinyLlama_fp32"

# ───────────────────────────────────────────────────────────────
# 정의부: export 함수
# ───────────────────────────────────────────────────────────────
def export_model():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[Export] Exporting model to ONNX directory: {ONNX_FP32_DIR} ({timestamp})")
    os.makedirs(ONNX_FP32_DIR, exist_ok=True)
    onnx_model = ORTModelForCausalLM.from_pretrained(
        MODEL_ID,
        export=True,
        use_cache=False,
        use_io_binding=False,
        use_auth_token=True,
        trust_remote_code=True
    )
    onnx_model.save_pretrained(
        ONNX_FP32_DIR,
        opset_version=20,
        use_external_data_format = False
        )
    print(f"[Export] Completed ONNX export and saved to {ONNX_FP32_DIR}")

# ───────────────────────────────────────────────────────────────
# 실행부
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    export_model()
