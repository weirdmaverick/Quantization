import os
import datetime
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoConfig

# ───────────────────────────────────────────────────────────────
# Model Export
# ───────────────────────────────────────────────────────────────
MODEL_ID      = "ajrae/bert-base-uncased-finetuned-mrpc"
ONNX_FP32_DIR = "onnx_bert_fp32"

def export_model():
    # 1) directory 
    os.makedirs(ONNX_FP32_DIR, exist_ok=True)
    # 2) config 
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.use_cache = False
    # 3) export & save
    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        export=True,
        config=config,
        use_io_binding=False,
        use_auth_token=True,
        trust_remote_code=True
    )
    onnx_model.save_pretrained(ONNX_FP32_DIR, opset_version=20)
    print(f"[Export] Completed ONNX export and saved to {ONNX_FP32_DIR}")

# ───────────────────────────────────────────────────────────────
# execution
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    export_model()
