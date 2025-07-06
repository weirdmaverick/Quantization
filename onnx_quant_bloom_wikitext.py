import os
import time
import datetime
import glob
import re
import json
import numpy as np
import onnx
from datasets import load_dataset
from transformers import AutoTokenizer
import onnxruntime as ort
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnxruntime import ORTModelForCausalLM
from tqdm import tqdm
from onnx import shape_inference

# ────────────────────────────────────────────────────
# 0. 사용자 설정
# ────────────────────────────────────────────────────
MODEL_ID = "bigscience/bloom-560m"
DATASET = ("wikitext", "wikitext-2-raw-v1", "validation")
BATCH_SIZE = 1
WARMUP_STEPS = 1
RUN_PROFILE = True
WEIGHT_TYPE = QuantType.QInt8
MAX_LENGTH = 512
ONNX_FP32_DIR = "onnx_bloom_fp32"
ONNX_INT8_DIR = "onnx_bloom_int8"

safe_model = MODEL_ID.replace("/", "_")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ────────────────────────────────────────────────────
# 1. 데이터셋 준비
# ────────────────────────────────────────────────────
print("[Data] Loading and tokenizing Wikitext-2 validation set...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, local_files_only=False, use_auth_token=True)
tokenizer.model_max_length = MAX_LENGTH

ds_raw = load_dataset(
    DATASET[0],        # "wikitext"
    DATASET[1],        # "wikitext-2-raw-v1"
    split=DATASET[2]   # "validation"
)
ds_tok = ds_raw.map(
    lambda x: tokenizer(
        x["text"], truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_attention_mask=True
    ),
    batched=True,
)
ds_tok.set_format(type="np", columns=["input_ids", "attention_mask"])
# ds_tok = ds_tok.select(range(400))  # 테스트용 1샘플
print(f"[Data] Dataset ready ({len(ds_tok)} samples)\n")

# ────────────────────────────────────────────────────
# 2. ONNX export
# ────────────────────────────────────────────────────
print(f"[Export] Exporting FP32 model to {ONNX_FP32_DIR} (use_cache=False)...")
os.makedirs(ONNX_FP32_DIR, exist_ok=True)
onnx_model: ORTModelForCausalLM = ORTModelForCausalLM.from_pretrained(
    MODEL_ID, export=True, use_cache=False, use_io_binding=False, use_auth_token=True,
    trust_remote_code=True, local_files_only=False
)
onnx_model.save_pretrained(ONNX_FP32_DIR, safe_serialization=False)
print("[Export] Completed ONNX export\n")

# ────────────────────────────────────────────────────
# 3. Shape Inference
# ────────────────────────────────────────────────────


def preprocess_onnx(input_fp32_path: str, output_path: str):
    print(f"[Preprocessing] Shape-inference on large model: {input_fp32_path}")
    inferred_model = shape_inference.infer_shapes_path(
        input_fp32_path, output_path)
    print(f"[Preprocessing] Saved shape-inferred model to {output_path}")


fp32_model = os.path.join(ONNX_FP32_DIR, "model.onnx")
opt_model_fp32 = os.path.join(ONNX_FP32_DIR, "model_opt.onnx")

if not os.path.exists(opt_model_fp32):
    preprocess_onnx(fp32_model, opt_model_fp32)
else:
    print("[Preprocessing] Skipping shape inference (already exists)\n")

# ────────────────────────────────────────────────────
# 4. Dynamic Quantization
# ────────────────────────────────────────────────────
print("[Quantization] Starting dynamic weight-only INT8 quantization...")
os.makedirs(ONNX_INT8_DIR, exist_ok=True)
quant_output = os.path.join(ONNX_INT8_DIR, "model.onnx")

quantize_dynamic(
    model_input=opt_model_fp32,
    model_output=quant_output,
    weight_type=WEIGHT_TYPE,
    per_channel=False,
)
print(f"[Quantization] Completed quantization, saved to {ONNX_INT8_DIR}\n")

# ────────────────────────────────────────────────────
# 5. 평가 함수
# ────────────────────────────────────────────────────


def create_session(path: str, profiling: bool) -> InferenceSession:
    so = SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if profiling:
        so.enable_profiling = True
        so.profile_file_prefix = f"{safe_model}_{os.path.basename(path)}_{timestamp}"
    return InferenceSession(path, sess_options=so)


def evaluate(onnx_dir: str, ds, model_tag: str):
    if model_tag == "FP32":
        model_path = os.path.join(onnx_dir, "model_opt.onnx")
        size_files = [
            os.path.join(onnx_dir, "model_opt.onnx"),
            os.path.join(onnx_dir, "model.onnx_data"),
        ]
        profiling = RUN_PROFILE
    else:
        model_path = os.path.join(onnx_dir, "model.onnx")
        size_files = [
            os.path.join(onnx_dir, "model.onnx"),
            os.path.join(onnx_dir, "model.onnx_data"),
        ]
        profiling = RUN_PROFILE

    sess = create_session(model_path, profiling)

    print(f"[Input Info] {model_tag} ONNX Input Names & Shapes:")
    for i in sess.get_inputs():
        print(f" - {i.name}: shape={i.shape}, type={i.type}")
    input_names = ["input_ids", "attention_mask"]

    total_time = 0.0

    # Warm-up
    for ex in ds.select(range(min(WARMUP_STEPS, len(ds)))):
        feed = {
            "input_ids": np.array(ex["input_ids"], dtype=np.int64).reshape(1, MAX_LENGTH),
            "attention_mask": np.array(ex["attention_mask"], dtype=np.int64).reshape(1, MAX_LENGTH),
        }
        sess.run(None, feed)

    # Evaluation
    for ex in tqdm(ds, desc=f"[{onnx_dir}] Evaluating"):
        ids = ex["input_ids"][:MAX_LENGTH]
        msk = ex["attention_mask"][:MAX_LENGTH]
        if ids.shape[0] < MAX_LENGTH:
            pad = MAX_LENGTH - ids.shape[0]
            ids = np.pad(ids, (0, pad), constant_values=tokenizer.pad_token_id)
            msk = np.pad(msk, (0, pad), constant_values=0)
        feed = {
            "input_ids": np.array(ids, dtype=np.int64).reshape(1, MAX_LENGTH),
            "attention_mask": np.array(msk, dtype=np.int64).reshape(1, MAX_LENGTH),
        }
        try:
            t0 = time.time()
            sess.run(None, feed)
            total_time += time.time() - t0
        except Exception as e:
            print(f"[Error] sess.run() failed: {e}")
            break

    prof_file = None
    if profiling:
        prof_file = sess.end_profiling()
        print(f"[Profile] Saved: {prof_file}")

    # Model size 계산
    total_bytes = sum(os.path.getsize(f)
                      for f in size_files if os.path.exists(f))
    size_mb = total_bytes / 1e6
    return total_time, size_mb, prof_file


# ────────────────────────────────────────────────────
# 6. FP32 vs INT8 비교 출력
# ────────────────────────────────────────────────────
results = {}
for tag, d in [("FP32", ONNX_FP32_DIR), ("INT8", ONNX_INT8_DIR)]:
    print(f"\n[Compare] Evaluating {tag} model...")
    t, s, p = evaluate(d, ds_tok, model_tag=tag)
    results[tag] = {"time": t, "size": s}
    print(f"{tag}: time={t:.2f}s, size={s:.2f}MB")
    if p:
        print(f"  → Profile: {p}")

fp = results["FP32"]
qt = results["INT8"]
print(
    f"\n[Result] Size Reduction   : {(fp['size']-qt['size'])/fp['size']*100:.1f}%")
print(
    f"[Result] Latency Speed-up : {(fp['time']-qt['time'])/fp['time']*100:.1f}%")

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
