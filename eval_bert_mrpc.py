import sys
import os
import glob
import time
import datetime
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate as eval_lib
from tqdm import tqdm
from onnxruntime import SessionOptions, GraphOptimizationLevel, ExecutionMode, InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType

# ───────────────────────────────────────────────────────────────
# Evaluation
# ───────────────────────────────────────────────────────────────
MODEL_ID      = "ajrae/bert-base-uncased-finetuned-mrpc"
DATASET       = ("glue", "mrpc", "validation")
BATCH_SIZE    = 1
WARMUP_STEPS  = 0
RUN_PROFILE   = True
WEIGHT_TYPE   = QuantType.QInt8
ONNX_FP32_DIR = "onnx_bert_fp32"
ONNX_INT8_DIR = "onnx_bert_int8"

def quantize_model(input_model: str, output_dir: str):
    print("[Quantization] Starting dynamic weight-only INT8 quantization...")
    os.makedirs(output_dir, exist_ok=True)
    quantize_dynamic(
        model_input=input_model,
        model_output=os.path.join(output_dir, "model.onnx"),
        weight_type=WEIGHT_TYPE,
        per_channel=False,
        use_external_data_format=False
    )
    print(f"[Quantization] Completed dynamic weight-only quantization and saved to {output_dir}")

def evaluate(onnx_dir: str, max_length: int, output_json: str):
    # 1) dataset load & tokenizer
    print("[Data] Loading and tokenizing GLUE MRPC validation set...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset(DATASET[0], DATASET[1], split=DATASET[2])
    dataset = dataset.select(range(100))
    dataset = dataset.map(
        lambda x: tokenizer(
            x["sentence1"], x["sentence2"],
            truncation=True, padding="max_length", max_length=max_length
        ),
        batched=True
    )
    dataset.set_format(type="np", columns=["input_ids","attention_mask","token_type_ids","label"])
    print(f"[Data] Dataset ready ({len(dataset)} samples)")

    # 2) session option
    paths = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    sess_opts = SessionOptions()
    if RUN_PROFILE:
        sess_opts.enable_profiling = True
        sess_opts.profile_file_prefix = output_json.rstrip(".json")
    sess_opts.inter_op_num_threads    = 1
    sess_opts.intra_op_num_threads    = 1
    sess_opts.enable_cpu_mem_arena    = False
    sess_opts.enable_mem_pattern      = False
    sess_opts.execution_mode          = ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level= GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_opts.log_severity_level      = 2

    session = InferenceSession(paths[0], sess_opts)
    inp_names = [inp.name for inp in session.get_inputs()]

    # 3) warm-up
    print(f"[Evaluate] Warming up for {WARMUP_STEPS} steps on {onnx_dir}...")
    for ex in dataset.select(range(min(WARMUP_STEPS, len(dataset)))):
        feed = {n: np.expand_dims(ex[n],0) for n in inp_names}
        session.run(None, feed)

    # 4) inference
    total_t = 0.0
    for ex in tqdm(dataset, desc=f"[Evaluate] Running inference on {onnx_dir}"):
        feed = {n: np.expand_dims(ex[n],0) for n in inp_names}
        start = time.time()
        session.run(None, feed)
        total_t += time.time() - start

    # 5) accuracy
    metric = eval_lib.load(*DATASET[:2])
    for ex in dataset:
        feed = {n: np.expand_dims(ex[n],0) for n in inp_names}
        logits = session.run(None, feed)[0]
        metric.add_batch(predictions=[int(np.argmax(logits,axis=-1))],
                         references=[int(ex["label"])])
    acc = metric.compute()["accuracy"]

    # 6) profiling file save
    prof_path = None
    if RUN_PROFILE:
        prof_path = session.end_profiling()
        os.rename(prof_path, output_json)
        prof_path = output_json
        print(f"[Profile] Data saved to: {prof_path}")

    size_mb = os.path.getsize(paths[0]) / 1e6
    return total_t, size_mb, acc, prof_path

# ───────────────────────────────────────────────────────────────
# Execution
# ───────────────────────────────────────────────────────────────
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    max_length  = args.max_length
    output_json = args.output_json

    # 1) preprocessed model path
    opt_model = os.path.join(ONNX_FP32_DIR, "model_optimized.onnx")
    # 2) quantize & save
    quantize_model(opt_model, ONNX_INT8_DIR)

    # 3) evaluate FP32
    t_fp32, s_fp32, acc_fp32, _ = evaluate(ONNX_FP32_DIR, max_length, f"fp32_{output_json}")
    # 4) evaluate INT8
    t_int8, s_int8, acc_int8, _ = evaluate(ONNX_INT8_DIR, max_length, f"int8_{output_json}")

    print(f"\nFP32 → time: {t_fp32:.3f}s | size: {s_fp32:.3f}MB | acc: {acc_fp32:.4f}")
    print(f"INT8 → time: {t_int8:.3f}s | size: {s_int8:.3f}MB | acc: {acc_int8:.4f}")
