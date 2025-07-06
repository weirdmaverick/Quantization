# quant.py

import re
import glob
import json
import os
import time
import datetime
import sys
import numpy as np
import onnx
from datasets import load_dataset
from tqdm import tqdm
import evaluate as eval_lib
from transformers import AutoTokenizer
import onnxruntime as ort
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnxruntime import ORTModelForSequenceClassification

# ────────────────────────────────────────────────────
# 0. 사용자 설정: 이미 fine-tuned된 MRPC model ID
# ────────────────────────────────────────────────────
MODEL_ID = "ajrae/bert-base-uncased-finetuned-mrpc"
BATCH_SIZE = 1
WARMUP_STEPS = 1
RUN_PROFILE = True
WEIGHT_TYPE = QuantType.QInt8

# Profiling file name 설정
safe_model = MODEL_ID.replace("/", "_")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ────────────────────────────────────────────────────
# 1. Dataset 준비
# ────────────────────────────────────────────────────
print("[Data] Loading and tokenizing GLUE MRPC validation set...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = load_dataset("glue", "mrpc", split="validation")
# dataset = dataset.select(range(400))  # 400
dataset = dataset.map(
    lambda x: tokenizer(
        x["sentence1"],
        x["sentence2"],
        truncation=True,
        padding="max_length"
    ),
    batched=True
)
dataset.set_format(
    type="np",
    columns=["input_ids", "attention_mask", "token_type_ids", "label"]
)
# print("[Data] Dataset ready (400 samples)\n")  # 400 samples

# ────────────────────────────────────────────────────
# 2. ONNX export (fine-tuned model load → ONNX)
# ────────────────────────────────────────────────────
print(f"[Export] Exporting FP32 model to ONNX (dir: onnx_bert_fp32)...")
onnx_model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    export=True
)
onnx_model.save_pretrained("onnx_bert_fp32")
print("[Export] Completed ONNX export and saved to onnx_bert_fp32\n")

# ────────────────────────────────────────────────────
# 3. Preprocessing (Shape Inference & Model Optimization)
# ────────────────────────────────────────────────────


def preprocess_onnx(input_fp32: str, output_opt: str):
    """
    1) Symbolic & ONNX Shape Inference
    2) Graph Optimization (operator fusion, constant folding)
    """
    print(f"[Preprocessing] Loading FP32 model from {input_fp32}")
    model = onnx.load(input_fp32)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferred, output_opt)
    print(f"[Preprocessing] Saved optimized model to {output_opt}")


def create_session(onnx_path: str, sess_opts: SessionOptions) -> InferenceSession:
    sess_opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    return InferenceSession(onnx_path, sess_options=sess_opts)


fp32_model = "onnx_bert_fp32/model.onnx"
opt_model = "onnx_bert_fp32/model_optimized.onnx"

print("[Preprocessing] Starting shape inference and graph optimization...")
preprocess_onnx(fp32_model, opt_model)
print("[Preprocessing] Completed shape inference and graph optimization\n")

# ────────────────────────────────────────────────────
# 4. Dynamic QOperator Quantization (Weight-only INT8)
# ────────────────────────────────────────────────────
print("[Quantization] Starting dynamic weight-only INT8 quantization...")
os.makedirs("onnx_bert_int8", exist_ok=True)
quantize_dynamic(
    model_input=opt_model,
    model_output="onnx_bert_int8/model.onnx",
    weight_type=WEIGHT_TYPE,
    per_channel=False
)
print("[Quantization] Completed dynamic quantization and saved to quant_model\n")

# ────────────────────────────────────────────────────
# 5. Evaluation Function (Profiling 포함)
# ────────────────────────────────────────────────────


def evaluate(onnx_dir, batch_size=BATCH_SIZE, warmup=WARMUP_STEPS, run_profile=RUN_PROFILE):
    paths = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    sess_opts = ort.SessionOptions()
    if run_profile:
        sess_opts.enable_profiling = True
        prefix = f"{safe_model}_{onnx_dir}_{timestamp}"
        sess_opts.profile_file_prefix = prefix
    sess = create_session(paths[0], sess_opts)
    inp_names = [inp.name for inp in sess.get_inputs()]

    # warm-up
    print(f"[Evaluate] Warming up for {warmup} steps...")
    for ex in dataset.select(range(min(warmup, len(dataset)))):
        feed = {n: np.expand_dims(ex[n], 0) for n in inp_names}
        sess.run(None, feed)

    # 본 측정
    metric = eval_lib.load("accuracy")
    total_t = 0.0
    print(f"[Evaluate] Running inference on {len(dataset)} samples...")
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"[{onnx_dir}] Evaluating"):
        batch = dataset[start:start+batch_size]
        feed = {n: batch[n] for n in inp_names}
        t0 = time.time()
        out = sess.run(None, feed)
        total_t += time.time() - t0
        preds = np.argmax(out[0], axis=1)
        metric.add_batch(predictions=preds.tolist(),
                         references=batch["label"].tolist())

    prof_path = None
    if run_profile:
        prof_path = sess.end_profiling()
        print(f"[Profile] Data saved to: {prof_path}")

    size_mb = os.path.getsize(paths[0]) / 1e6
    acc = metric.compute()["accuracy"]
    return total_t, size_mb, acc, prof_path


# ────────────────────────────────────────────────────
# 6. FP32 vs Quant 비교 & Overhead 계산
# ────────────────────────────────────────────────────
results = {}
for name, path in [("FP32", "onnx_bert_fp32"), ("Quant", "onnx_bert_int8")]:
    print(f"\n[Compare] Evaluating {name} model...")
    t, s, a, prof = evaluate(path)
    results[name] = (t, s, a, prof)
    print(f"{name} → Acc: {a:.4f}, Time: {t:.2f}s, Size: {s:.2f}MB")

fp_t, fp_s, _, _ = results["FP32"]
q_t, q_s, _, _ = results["Quant"]
print(f"\n[Result] Size Reduction   : {(fp_s - q_s) / fp_s * 100:.1f}%")
print(f"[Result] Latency Speed-up : {(fp_t - q_t) / fp_t * 100:.1f}%")

# ────────────────────────────────────────────────────
# 7. Quant Operator Overhead (Profiling JSON 분석)
# ────────────────────────────────────────────────────
events = []
pattern = re.compile(
    r'^(DynamicQuantizeLinear|DynamicQuantizeMatMul|MatMulIntegerToFloat)$'
)

for f in glob.glob(f"{safe_model}_*.json"):
    if os.path.getsize(f) == 0:
        print(f"[Profile] Skipping empty profile file: {f}")
        continue
    with open(f, "r") as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError:
            print(f"[Profile] Invalid JSON in {f}, skipping.")
            continue
    if isinstance(data, dict) and "events" in data:
        events.extend(data["events"])
    elif isinstance(data, list):
        events.extend(data)
    else:
        raise ValueError(f"Unexpected profile format in {f}")

qt_us = sum(
    ev.get("dur", 0)
    for ev in events
    if pattern.match(ev.get("args", {}).get("op_name", ev.get("name", "")))
)
tot_us = sum(ev.get("dur", 0) for ev in events)

if tot_us > 0:
    print(f"[Profile] Quantized Kernel Time: {qt_us/1e6:.3f}s / "
          f"Total: {tot_us/1e6:.3f}s ({qt_us/tot_us*100:.1f}%)")
else:
    print("[Profile] No profile data or zero total time; skipping Quant Kernel ratio.")

# ────────────────────────────────────────────────────
# 8. ONNX Quant Ops 비율 측정 (Graph 분석)
# ────────────────────────────────────────────────────


def quant_op_ratio(onnx_path):
    model = onnx.load(onnx_path)
    ops = [node.op_type for node in model.graph.node]
    total = len(ops)
    quant_ops = [op for op in ops if any(q in op for q in [
        "DynamicQuantizeLinear", "DynamicQuantizeMatMul", "MatMulIntegerToFloat"
    ])]
    ratio = len(quant_ops)/total*100 if total else 0
    return len(quant_ops), total, ratio


for name, path in [("FP32", "onnx_bert_fp32/model.onnx"), ("Quant", "onnx_bert_int8/model.onnx")]:
    qn, tot, pct = quant_op_ratio(path)
    print(f"[Graph] {name} → {qn}/{tot} QuantOps ({pct:.2f}%)")
sys.exit(0)
