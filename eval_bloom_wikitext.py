import os
import glob
import time
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from onnxruntime import SessionOptions, GraphOptimizationLevel, ExecutionMode, InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType

# ───────────────────────────────────────────────────────────────
# Custom Setting
# ───────────────────────────────────────────────────────────────
MODEL_ID      = "bigscience/bloom-560m"
DATASET       = ("wikitext","wikitext-2-raw-v1","validation")
RUN_PROFILE   = True
WEIGHT_TYPE   = QuantType.QInt8
ONNX_FP32_DIR = "onnx_bloom_fp32"
ONNX_INT8_DIR = "onnx_bloom_int8"
WARMUP_STEPS  = 0

# ───────────────────────────────────────────────────────────────
# 정의부: quantize 함수
# ───────────────────────────────────────────────────────────────
def quantize_model(input_model: str, output_dir: str):
    print("[Quantization] Starting dynamic weight-only INT8 quantization...")
    os.makedirs(output_dir, exist_ok=True)
    quantize_dynamic(
        model_input = input_model,
        model_output= os.path.join(output_dir, "model.onnx"),
        weight_type = WEIGHT_TYPE,
        per_channel = False
    )
    print(f"[Quantization] Completed → {output_dir}")

# ───────────────────────────────────────────────────────────────
# 정의부: evaluate 함수
# ───────────────────────────────────────────────────────────────
def evaluate(onnx_dir: str, ds, max_length: int, profile_prefix: str):
    # 1) 세션 옵션
    paths = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    sess_opts = SessionOptions()
    if RUN_PROFILE:
        sess_opts.enable_profiling          = True
        sess_opts.profile_file_prefix       = profile_prefix
    sess_opts.inter_op_num_threads        = 1
    sess_opts.intra_op_num_threads        = 1
    sess_opts.enable_cpu_mem_arena        = False
    sess_opts.enable_mem_pattern          = False
    sess_opts.execution_mode              = ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level    = GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_opts.log_severity_level          = 2

    session = InferenceSession(paths[0], sess_opts)

    # 2) Warm-up (optional)
    for ex in ds.select(range(min(WARMUP_STEPS, len(ds)))):
        feed = {
            "input_ids":      np.array(ex["input_ids"],     dtype=np.int64).reshape(1, -1),
            "attention_mask": np.array(ex["attention_mask"], dtype=np.int64).reshape(1, -1),
        }
        session.run(None, feed)

    # 3) 본 추론
    total_t = 0.0
    for ex in tqdm(ds, desc=f"[Evaluate] {os.path.basename(onnx_dir)}"):
        ids = ex["input_ids"][:max_length]
        msk = ex["attention_mask"][:max_length]
        if ids.shape[0] < max_length:
            pad = max_length - ids.shape[0]
            ids = np.pad(ids, (0,pad), constant_values=session.get_inputs()[0].type)
            msk = np.pad(msk, (0,pad), constant_values=0)
        feed = {
            "input_ids":      ids.reshape(1, max_length),
            "attention_mask": msk.reshape(1, max_length),
        }
        t0 = time.time()
        session.run(None, feed)
        total_t += time.time() - t0

    # 4) 프로파일링 종료
    prof_path = None
    if RUN_PROFILE:
        prof_path = session.end_profiling()
        print(f"[Profile] Saved → {prof_path}")

    # 5) 모델 크기 계산
    files = [os.path.join(onnx_dir,"model.onnx")]
    dataf = os.path.join(onnx_dir,"model.onnx_data")
    if os.path.exists(dataf): files.append(dataf)
    size_mb = sum(os.path.getsize(f) for f in files)/1e6

    return total_t, size_mb, prof_path

# ───────────────────────────────────────────────────────────────
# 실행부: main()
# ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_length",  type=int, required=True, help="max sequence length")
    p.add_argument("--output_json", type=str, required=True, help="profiling JSON path")
    return p.parse_args()

if __name__ == "__main__":
    args       = parse_args()
    max_length = args.max_length
    out_json   = args.output_json

    # 0) 데이터 준비
    print("[Data] Loading & tokenizing Wikitext-2 validation set...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.model_max_length = max_length
    ds_raw = load_dataset(
        DATASET[0],
        DATASET[1],
        split = DATASET[2]
    )
    ds_tok = ds_raw.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_length),
        batched=True
    )
    ds_tok.set_format(type="np", columns=["input_ids","attention_mask"])
    ds_tok = ds_tok.select(range(100))  # original 코드처럼 1개 샘플만

    # 1) preprocess된 ONNX 경로
    opt_model = os.path.join(ONNX_FP32_DIR, "model_opt.onnx")

    # 2) quantize
    quantize_model(opt_model, ONNX_INT8_DIR)

    # 3) evaluate FP32
    t_fp32, s_fp32, p_fp32 = evaluate(ONNX_FP32_DIR, ds_tok, max_length, out_json.rstrip(".json") + "_fp32")
    print(f"FP32 → time: {t_fp32:.2f}s | size: {s_fp32:.2f}MB")

    # 4) evaluate INT8
    t_int8, s_int8, p_int8 = evaluate(ONNX_INT8_DIR, ds_tok, max_length, out_json.rstrip(".json") + "_int8")
    print(f"INT8 → time: {t_int8:.2f}s | size: {s_int8:.2f}MB")

    # 5) 최종 비교
    print(f"\n[Result] Size↓: {(s_fp32 - s_int8)/s_fp32*100:.1f}%  Latency↑: {(t_fp32 - t_int8)/t_fp32*100:.1f}%")
