import sys
import os
import glob
import time
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from onnxruntime import SessionOptions, GraphOptimizationLevel, ExecutionMode, InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType

# ───────────────────────────────────────────────────────────────
# Custom Setting
# ───────────────────────────────────────────────────────────────
MODEL_ID        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET         = ("wikitext","wikitext-2-raw-v1","validation")
RUN_PROFILE     = True
WEIGHT_TYPE     = QuantType.QInt8
ONNX_FP32_DIR   = "onnx_TinyLlama_fp32"
ONNX_INT8_DIR   = "onnx_TinyLlama_int8"
WARMUP_STEPS    = 1

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
    print(f"[Quantization] Completed quantization and saved to {output_dir}")

# ───────────────────────────────────────────────────────────────
# 정의부: evaluate 함수
# ───────────────────────────────────────────────────────────────
def evaluate(onnx_dir: str, ds, max_length: int, output_json: str):
    # SessionOptions 설정
    paths = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    sess_opts = SessionOptions()
    if RUN_PROFILE:
        sess_opts.enable_profiling    = True
        sess_opts.profile_file_prefix = output_json.rstrip(".json")
    sess_opts.inter_op_num_threads     = 1
    sess_opts.intra_op_num_threads     = 1
    sess_opts.enable_cpu_mem_arena     = False
    sess_opts.enable_mem_pattern       = False
    sess_opts.execution_mode           = ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_opts.log_severity_level       = 2

    session = InferenceSession(paths[0], sess_opts)
    input_names = ["input_ids", "attention_mask", "position_ids"]

    # Warm-up
    print(f"[Evaluate] Warming up for {WARMUP_STEPS} steps on {onnx_dir}...")
    for ex in ds.select(range(min(WARMUP_STEPS, len(ds)))):
        feed = {
            "input_ids":      np.array(ex["input_ids"],   dtype=np.int64).reshape(1, max_length),
            "attention_mask": np.array(ex["attention_mask"],dtype=np.int64).reshape(1, max_length),
            "position_ids":   np.array(ex["position_ids"],  dtype=np.int64).reshape(1, max_length),
        }
        session.run(None, feed)

    # 실제 추론 및 시간 측정
    total_t = 0.0
    for ex in tqdm(ds, desc=f"[Evaluate] {os.path.basename(onnx_dir)}"):
        ids = ex["input_ids"][:max_length]
        msk = ex["attention_mask"][:max_length]
        pos = ex["position_ids"][:max_length]
        if ids.shape[0] < max_length:
            pad = max_length - ids.shape[0]
            ids = np.pad(ids, (0,pad), constant_values=tokenizer.pad_token_id)
            msk = np.pad(msk, (0,pad), constant_values=0)
            pos = np.pad(pos, (0,pad), constant_values=0)
        feed = {
            "input_ids":      ids.reshape(1, max_length),
            "attention_mask": msk.reshape(1, max_length),
            "position_ids":   pos.reshape(1, max_length),
        }
        t0 = time.time()
        session.run(None, feed)
        total_t += time.time() - t0

    # 프로파일링 종료
    prof_path = None
    if RUN_PROFILE:
        prof_path = session.end_profiling()
        print(f"[Profile] Saved → {prof_path}")

    # 모델 파일 크기 계산
    files = [os.path.join(onnx_dir,"model.onnx")]
    dataf = os.path.join(onnx_dir,"model.onnx_data")
    if os.path.exists(dataf):
        files.append(dataf)
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
    output_json   = args.output_json

    # 1) 데이터 준비
    print("[Data] Loading and tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_auth_token=True,
        trust_remote_code=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = max_length

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
            max_length=max_length
        )
        seq_len = len(tok["input_ids"][0])
        tok["position_ids"] = [list(range(seq_len))] * len(tok["input_ids"])
        return tok

    ds_tok = ds_raw.map(
        preprocess_batch,
        batched=True,
        remove_columns=ds_raw.column_names
    )
    ds_tok.set_format(type="np", columns=["input_ids","attention_mask","position_ids"])
    ds_tok = ds_tok.select(range(80))
    print(f"[Data] Dataset ready ({len(ds_tok)} samples)\n")

    # 2) quantize
    opt_model = os.path.join(ONNX_FP32_DIR, "model_opt.onnx")
    quantize_model(opt_model, ONNX_INT8_DIR)

    # 3) evaluate FP32
    t_fp32, s_fp32, p_fp32 = evaluate(
        ONNX_FP32_DIR, ds_tok, max_length, f"fp32_{output_json}"
    )
    print(f"FP32 → time: {t_fp32:.2f}s | size: {s_fp32:.2f}MB")

    # 4) evaluate INT8
    t_int8, s_int8, p_int8 = evaluate(
        ONNX_INT8_DIR, ds_tok, max_length, f"int8_{output_json}"
    )
    print(f"INT8 → time: {t_int8:.2f}s | size: {s_int8:.2f}MB")

    # 5) 결과 비교
    print(f"\n[Result] Size↓: {(s_fp32 - s_int8)/s_fp32*100:.1f}%  Latency↑: {(t_fp32 - t_int8)/t_fp32*100:.1f}%")