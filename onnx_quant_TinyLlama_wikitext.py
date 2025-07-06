import os
import time
import datetime
import glob
import re
import json
import time
import numpy as np
import onnx
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from onnx import shape_inference
import onnx
import onnxruntime as ort
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnxruntime import ORTModelForCausalLM

# ────────────────────────────────────────────────────
# 0. 사용자 설정
# ────────────────────────────────────────────────────
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET = ("wikitext", "wikitext-2-raw-v1", "validation")
BATCH_SIZE = 1
WARMUP_STEPS = 1
RUN_PROFILE = True
WEIGHT_TYPE = QuantType.QInt8
MAX_LEN = 512
ONNX_DIR_FP32 = "onnx_TinyLlama_fp32"
ONNX_DIR_QUANT = "onnx_TinyLlama_int8"

safe_model = MODEL_ID.replace("/", "_")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ────────────────────────────────────────────────────
# 1. 데이터셋 준비 (Wikitext-2 Validation)
# ────────────────────────────────────────────────────
print("[Data] Loading and tokenizing dataset...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_auth_token=True,
    local_files_only=False,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.model_max_length = MAX_LEN
ds_raw = load_dataset(
    DATASET[0],        # "wikitext"
    DATASET[1],        # "wikitext-2-raw-v1"
    split=DATASET[2]   # "validation"
)


def preprocess_batch(batch):
    tok = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    seq_len = len(tok["input_ids"][0])
    pos = list(range(seq_len))
    tok["position_ids"] = [pos] * len(tok["input_ids"])
    return tok


ds_tok = ds_raw.map(
    preprocess_batch,
    batched=True,
    remove_columns=ds_raw.column_names,
)
ds_tok.set_format(type="np", columns=[
                  "input_ids", "attention_mask", "position_ids"])
# ds_tok = ds_tok.select(range(400))

# ────────────────────────────────────────────────────
# 2. ONNX export (fine-tuned Llama2 모델 로드 → ONNX)
# ────────────────────────────────────────────────────
print(f"[Export] Exporting model to ONNX directory: {ONNX_DIR_FP32}")
os.makedirs(ONNX_DIR_FP32, exist_ok=True)
onnx_model = ORTModelForCausalLM.from_pretrained(
    MODEL_ID,
    export=True,
    use_cache=False,
    use_io_binding=False,
    use_auth_token=True,
    local_files_only=False,
    trust_remote_code=True
)
onnx_model.save_pretrained(ONNX_DIR_FP32)
print(f"[Export] Completed ONNX export and saved to {ONNX_DIR_FP32}")

# ────────────────────────────────────────────────────
# 3. Preprocessing (Shape Inference & Model Optimization)
# ────────────────────────────────────────────────────
print("[Preprocessing] Starting shape inference and graph optimization...")


def preprocess_onnx(input_fp32_path: str, output_path: str):
    print(f"[Preprocessing] Shape-inference (safe) on {input_fp32_path}")
    inferred_model = shape_inference.infer_shapes_path(
        input_fp32_path, output_path)
    print(f"[Preprocessing] Saved shape-inferred model to {output_path}\n")


def create_session(path: str, profiling: bool) -> InferenceSession:
    so = SessionOptions()
    so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if profiling:
        so.enable_profiling = True
        so.profile_file_prefix = f"{safe_model}_{os.path.basename(path)}_{timestamp}"
    return InferenceSession(path, sess_options=so)


fp32_model = os.path.join(ONNX_DIR_FP32, "model.onnx")
opt_model = os.path.join(ONNX_DIR_FP32, "model_opt.onnx")
preprocess_onnx(fp32_model, opt_model)
print("[Preprocessing] Shape inference and graph optimization completed\n")

# ────────────────────────────────────────────────────
# 4. Dynamic Quantization (Weight-only INT8)
# ────────────────────────────────────────────────────
print("[Quantization] Starting dynamic weight-only INT8 quantization...")
os.makedirs(ONNX_DIR_QUANT, exist_ok=True)
quantize_dynamic(
    model_input=opt_model,
    model_output=os.path.join(ONNX_DIR_QUANT, "model.onnx"),
    weight_type=WEIGHT_TYPE,
    per_channel=False,
)
print(
    "[Quantization] Completed dynamic quantization and saved to {ONNX_DIR_QUANT}\n")

# ────────────────────────────────────────────────────
# 5. 평가 함수 (Profiling 포함)
# ────────────────────────────────────────────────────


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
    input_names = ["input_ids", "attention_mask", "position_ids"]

    total_time = 0.0

    # Warm-up
    for ex in ds.select(range(min(WARMUP_STEPS, len(ds)))):
        feed = {
            "input_ids": np.array(ex["input_ids"], dtype=np.int64).reshape(1, MAX_LEN),
            "attention_mask": np.array(ex["attention_mask"], dtype=np.int64).reshape(1, MAX_LEN),
            "position_ids": np.array(ex["position_ids"], dtype=np.int64).reshape(1, MAX_LEN)
        }
        sess.run(None, feed)

    # Evaluation
    for ex in tqdm(ds, desc=f"[{onnx_dir}] Evaluating"):
        ids = ex["input_ids"][:MAX_LEN]
        msk = ex["attention_mask"][:MAX_LEN]
        if ids.shape[0] < MAX_LEN:
            pad = MAX_LEN - ids.shape[0]
            ids = np.pad(ids, (0, pad), constant_values=tokenizer.pad_token_id)
            msk = np.pad(msk, (0, pad), constant_values=0)
        feed = {
            "input_ids": np.array(ids, dtype=np.int64).reshape(1, MAX_LEN),
            "attention_mask": np.array(msk, dtype=np.int64).reshape(1, MAX_LEN),
            "position_ids": np.array(ex["position_ids"], dtype=np.int64).reshape(1, MAX_LEN)
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
# 6. FP32 vs Quant 비교 & 결과 출력
# ────────────────────────────────────────────────────
results = {}
for name, dir_ in [("FP32", ONNX_DIR_FP32), ("INT8", ONNX_DIR_QUANT)]:
    print(f"\n[Compare] Evaluating {name} model...")
    t, s, prof = evaluate(dir_, ds_tok, model_tag=name)
    results[name] = {"time": t, "size": s, "prof": prof}
    print(f"{name}: time={t:.2f}s, size={s:.2f}MB")

fp = results["FP32"]
q = results["INT8"]
print(
    f"\n[Result] Size Reduction   : {(fp['size']-q['size'])/fp['size']*100:.1f}%")

if fp["time"] > 0:
    print(
        f"[Result] Latency Speed-up : {(fp['time']-q['time'])/fp['time']*100:.1f}%")
else:
    print("[Result] Latency Speed-up : N/A (FP32 time = 0)")
# ────────────────────────────────────────────────────
# 7. Quant Kernel 실행시간 비율 (Profiling JSON 분석)
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
# 8. Quant/Dequant 연산자 비율 (Graph 분석)
# ────────────────────────────────────────────────────


def quant_op_ratio(onnx_path):
    mdl = onnx.load(onnx_path)
    ops = [node.op_type for node in mdl.graph.node]
    total = len(ops)
    q_ops = [op for op in ops if any(q in op for q in [
                                     "DynamicQuantizeLinear", "DynamicQuantizeMatMul", "MatMulIntegerToFloat"])]
    return len(q_ops), total, len(q_ops)/total*100 if total else 0


for name, path in [("FP32", os.path.join(ONNX_DIR_FP32, "model.onnx")),
                   ("INT8", os.path.join(ONNX_DIR_QUANT, "model.onnx"))]:
    qn, tot, pct = quant_op_ratio(path)
    print(f"[{name}] {qn}/{tot} QuantOps ({pct:.2f}%)")
