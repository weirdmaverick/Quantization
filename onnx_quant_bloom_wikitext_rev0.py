import sys
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
from onnxruntime import SessionOptions, GraphOptimizationLevel, ExecutionMode, InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference
from optimum.onnxruntime import ORTModelForCausalLM
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────────────
# 0. 사용자 설정
# ───────────────────────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────────────────────
# 1. 데이터셋 준비
# ───────────────────────────────────────────────────────────────────────────────
print("[Data] Loading and tokenizing Wikitext-2 validation set...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
tokenizer.model_max_length = MAX_LENGTH

ds_raw = load_dataset(
    DATASET[0],        # "wikitext"
    DATASET[1],        # "wikitext-2-raw-v1"
    split=DATASET[2]   # "validation"
)

ds_tok = ds_raw.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_attention_mask=True
    ),
    batched=True
)
ds_tok.set_format(type="np", columns=["input_ids", "attention_mask"])
ds_tok = ds_tok.select(range(1))
print(f"[Data] Dataset ready ({len(ds_tok)} samples)\n")

# ───────────────────────────────────────────────────────────────────────────────
# 2. ONNX Export (fine-tuned 모델 → ONNX)
# ───────────────────────────────────────────────────────────────────────────────
print(f"[Export] Exporting FP32 model to {ONNX_FP32_DIR} (use_cache=False)...")
os.makedirs(ONNX_FP32_DIR, exist_ok=True)
onnx_model: ORTModelForCausalLM = ORTModelForCausalLM.from_pretrained(
    MODEL_ID,
    export=True,
    use_cache=False,
    use_io_binding=False,
    trust_remote_code=True
)
onnx_model.save_pretrained(ONNX_FP32_DIR, safe_serialization=False)
print("[Export] Completed ONNX export\n")

# ───────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing: Shape Inference + Graph Transformer(Fusion 등) + ONNX Shape Inference
# ───────────────────────────────────────────────────────────────────────────────


def preprocess_onnx_with_transformer(input_fp32: str, output_opt: str):
    """
    1) Symbolic Shape Inference
    2) Graph Optimization (Fusion, Constant Folding, Dead Code Elimination)
    3) ONNX Shape Inference
    """
    print(
        f"[Preprocessing] Running quant_pre_process on {input_fp32} → {output_opt}")
    shape_inference.quant_pre_process(
        input_model_path=input_fp32,
        output_model_path=output_opt,
        skip_symbolic_shape=False,
        skip_optimization=False,
        skip_onnx_shape=False,
        auto_merge=True,
        verbose=1
    )
    print(f"[Preprocessing] Saved fully optimized model to {output_opt}\n")


fp32_model = os.path.join(ONNX_FP32_DIR, "model.onnx")
opt_model_fp32 = os.path.join(ONNX_FP32_DIR, "model_opt.onnx")

if not os.path.exists(opt_model_fp32):
    print("[Preprocessing] Starting Shape Inference and Graph Optimization...")
    preprocess_onnx_with_transformer(fp32_model, opt_model_fp32)
    print("[Preprocessing] Completed all preprocessing steps\n")
else:
    print("[Preprocessing] Skipping preprocessing (already exists)\n")

# ───────────────────────────────────────────────────────────────────────────────
# 4. Dynamic Weight-Only Quantization (QOperator 방식)
# ───────────────────────────────────────────────────────────────────────────────
print("[Quantization] Starting dynamic weight-only INT8 quantization...")
os.makedirs(ONNX_INT8_DIR, exist_ok=True)
quant_output = os.path.join(ONNX_INT8_DIR, "model.onnx")

quantize_dynamic(
    model_input=opt_model_fp32,
    model_output=quant_output,
    weight_type=WEIGHT_TYPE,
    per_channel=False
)
print(f"[Quantization] Completed quantization, saved to {ONNX_INT8_DIR}\n")

# ───────────────────────────────────────────────────────────────────────────────
# 5. 평가 함수 (Inference & Profiling 포함)
# ───────────────────────────────────────────────────────────────────────────────


def evaluate_tuned(onnx_dir, ds, run_profile=RUN_PROFILE):
    """
    1) InferenceSession 생성 (스레드/메모리/프로파일링 옵션 포함)
    2) Warm-up
    3) 실제 추론 및 시간 측정
    4) 프로파일링 파일 경로 반환
    5) 모델 파일 크기 반환
    """
    # ONNX 파일 경로
    onnx_paths = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    model_path = onnx_paths[0]

    # SessionOptions 튜닝
    sess_opts = SessionOptions()

    # Profiling 활성화
    if run_profile:
        sess_opts.enable_profiling = True
        sess_opts.profile_file_prefix = f"{safe_model}_{onnx_dir}_{timestamp}"

    # 스레드 및 메모리 옵션
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 8
    sess_opts.enable_cpu_mem_arena = True
    sess_opts.enable_mem_pattern = True
    sess_opts.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 2  # WARNING

    # InferenceSession 생성
    session = InferenceSession(model_path, sess_opts)

    # 입력 이름
    input_names = ["input_ids", "attention_mask"]

    # Warm-up
    print(f"[Evaluate] Warming up for {WARMUP_STEPS} steps on {onnx_dir}...")
    for ex in ds.select(range(min(WARMUP_STEPS, len(ds)))):
        feed = {
            "input_ids":      np.array(ex["input_ids"],     dtype=np.int64).reshape(1, MAX_LENGTH),
            "attention_mask": np.array(ex["attention_mask"], dtype=np.int64).reshape(1, MAX_LENGTH),
        }
        session.run(None, feed)

    # 실제 추론 및 시간 측정
    total_t = 0.0
    for ex in tqdm(ds, desc=f"[Evaluate] Running inference on {onnx_dir}"):
        ids = ex["input_ids"][:MAX_LENGTH]
        msk = ex["attention_mask"][:MAX_LENGTH]
        if ids.shape[0] < MAX_LENGTH:
            pad = MAX_LENGTH - ids.shape[0]
            ids = np.pad(ids, (0, pad), constant_values=tokenizer.pad_token_id)
            msk = np.pad(msk, (0, pad), constant_values=0)
        feed = {
            "input_ids":      np.array(ids, dtype=np.int64).reshape(1, MAX_LENGTH),
            "attention_mask": np.array(msk, dtype=np.int64).reshape(1, MAX_LENGTH),
        }
        t0 = time.time()
        session.run(None, feed)
        total_t += time.time() - t0

    # Profiling 파일 경로 반환
    prof_path = None
    if run_profile:
        prof_path = session.end_profiling()
        print(f"[Profile] Saved: {prof_path}")

    # 모델 파일 크기 계산 (.onnx + .onnx_data)
    size_files = []
    size_files.append(os.path.join(onnx_dir, "model.onnx"))
    data_file = os.path.join(onnx_dir, "model.onnx_data")
    if os.path.exists(data_file):
        size_files.append(data_file)
    total_bytes = sum(os.path.getsize(f)
                      for f in size_files if os.path.exists(f))
    size_mb = total_bytes / 1e6

    return total_t, size_mb, prof_path


# ───────────────────────────────────────────────────────────────────────────────
# 6. FP32 vs INT8 비교 및 출력
# ───────────────────────────────────────────────────────────────────────────────
results = {}
for tag, d in [("FP32", ONNX_FP32_DIR), ("INT8", ONNX_INT8_DIR)]:
    print(f"\n[Compare] Evaluating {tag} model...")
    t, s, p = evaluate_tuned(d, ds_tok)
    results[tag] = {"time": t, "size": s, "profile": p}
    print(f"{tag}: time={t:.2f}s | size={s:.2f}MB")
    if p:
        print(f"  → Profile: {p}")

fp = results["FP32"]
qt = results["INT8"]
print(
    f"\n[Result] Size Reduction   : {(fp['size'] - qt['size']) / fp['size'] * 100:.1f}%")
print(
    f"[Result] Latency Speed-up : {(fp['time'] - qt['time']) / fp['time'] * 100:.1f}%\n")

# ───────────────────────────────────────────────────────────────────────────────
# 7. Quant Kernel 오버헤드 계산 (Profiling JSON 분석)
# ───────────────────────────────────────────────────────────────────────────────
events = []
pattern = re.compile(
    r'^(DynamicQuantizeLinear|DynamicQuantizeMatMul|MatMulIntegerToFloat|QLinearGemm)$')

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
    # ONNX Runtime 프로파일 형식에 따라 events 수집
    if isinstance(data, dict) and "events" in data:
        events.extend(data["events"])
    elif isinstance(data, list):
        events.extend(data)
    else:
        raise ValueError(f"Unexpected profile format in {f}")

qt_us = sum(ev.get("dur", 0) for ev in events if pattern.match(
    ev.get("args", {}).get("op_name", ev.get("name", ""))))
tot_us = sum(ev.get("dur", 0) for ev in events)

if tot_us > 0:
    print(
        f"[Profile] Quantized Kernel Time: {qt_us/1e6:.3f}s / Total: {tot_us/1e6:.3f}s ({qt_us/tot_us*100:.1f}%)")
else:
    print("[Profile] No profile data or zero total time; skipping Quant Kernel ratio.")

# ───────────────────────────────────────────────────────────────────────────────
# 8. ONNX Quant Ops 비율 측정 (Graph 분석)
# ───────────────────────────────────────────────────────────────────────────────


def quant_op_ratio(onnx_path):
    model = onnx.load(onnx_path)
    ops = [node.op_type for node in model.graph.node]
    total = len(ops)
    quant_ops = [op for op in ops if any(q in op for q in [
        "DynamicQuantizeLinear", "DynamicQuantizeMatMul", "MatMulIntegerToFloat", "QLinearGemm"
    ])]
    ratio = len(quant_ops) / total * 100 if total else 0.0
    return len(quant_ops), total, ratio


for name, path in [
    ("FP32", os.path.join(ONNX_FP32_DIR, "model.onnx")),
    ("Quant", os.path.join(ONNX_INT8_DIR, "model.onnx"))
]:
    qn, tot, pct = quant_op_ratio(path)
    print(f"[Graph] {name} → {qn}/{tot} QuantOps ({pct:.2f}%)")

# ───────────────────────────────────────────────────────────────────────────────
# 스크립트 종료
# ───────────────────────────────────────────────────────────────────────────────
sys.exit(0)
