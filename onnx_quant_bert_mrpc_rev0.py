# onnx_quant_bert_mrpc_tuned.py

import os
import glob
import time
import datetime
import numpy as np
import onnx
from datasets import load_dataset
from tqdm import tqdm
import evaluate as eval_lib
from transformers import AutoTokenizer
import onnxruntime as ort
from onnxruntime import SessionOptions, GraphOptimizationLevel, ExecutionMode, InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference

# ───────────────────────────────────────────────────────────────────────────────
# 0. 사용자 설정
# ───────────────────────────────────────────────────────────────────────────────
MODEL_ID     = "ajrae/bert-base-uncased-finetuned-mrpc"
BATCH_SIZE   = 1
WARMUP_STEPS = 1
RUN_PROFILE  = True
WEIGHT_TYPE  = QuantType.QInt8

# profiling 파일명에 쓰일 안전한 모델명 및 타임스탬프
safe_model = MODEL_ID.replace("/", "_")
timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ───────────────────────────────────────────────────────────────────────────────
# 1. Dataset 준비
# ───────────────────────────────────────────────────────────────────────────────
print("[Data] Loading and tokenizing GLUE MRPC validation set...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = load_dataset("glue", "mrpc", split="validation")
# 테스트 용으로 첫 1개 샘플만 사용 (실험 시에는 전체 사용 권장)
dataset = dataset.select(range(1))
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
print(f"[Data] Dataset ready ({len(dataset)} samples)\n")

# ───────────────────────────────────────────────────────────────────────────────
# 2. ONNX Export (fine-tuned 모델 → ONNX)
# ───────────────────────────────────────────────────────────────────────────────
print(f"[Export] Exporting FP32 model to ONNX (dir: onnx_bert_fp32)...")
from optimum.onnxruntime import ORTModelForSequenceClassification

onnx_model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    export=True
)
onnx_model.save_pretrained("onnx_bert_fp32")
print("[Export] Completed ONNX export and saved to onnx_bert_fp32\n")

# ───────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing: Shape Inference + Graph Transformer(Fusion 등) + ONNX Shape Inference
# ───────────────────────────────────────────────────────────────────────────────
def preprocess_onnx_with_transformer(input_fp32: str, output_opt: str):
    """
    1) Symbolic Shape Inference
    2) Graph Optimization (Fusion, Constant Folding, Dead Code Elimination)
    3) ONNX Shape Inference
    """
    print(f"[Preprocessing] Running quant_pre_process on {input_fp32} → {output_opt}")
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

fp32_model = "onnx_bert_fp32/model.onnx"
opt_model  = "onnx_bert_fp32/model_optimized.onnx"

print("[Preprocessing] Starting Shape Inference and Graph Optimization...")
preprocess_onnx_with_transformer(fp32_model, opt_model)
print("[Preprocessing] Completed all preprocessing steps\n")

# ───────────────────────────────────────────────────────────────────────────────
# 4. Dynamic Weight-Only Quantization (QOperator 방식)
# ───────────────────────────────────────────────────────────────────────────────
print("[Quantization] Starting dynamic weight-only INT8 quantization...")
os.makedirs("onnx_bert_int8", exist_ok=True)
quantize_dynamic(
    model_input=opt_model,
    model_output="onnx_bert_int8/model.onnx",
    weight_type=WEIGHT_TYPE,
    per_channel=False
)
print("[Quantization] Completed dynamic weight-only quantization\n")

# ───────────────────────────────────────────────────────────────────────────────
# 5. Inference & Profiling (SessionOptions 튜닝 포함)
# ───────────────────────────────────────────────────────────────────────────────
def evaluate_tuned(onnx_dir, run_profile=RUN_PROFILE):
    """
    1) InferenceSession 생성 (스레드/메모리/프로파일링 옵션 포함)
    2) Warm-up
    3) 실제 추론 및 시간 측정
    4) 정확도(Accuracy) 계산
    5) 프로파일링 파일 경로 반환
    6) 모델 파일 크기 반환
    """
    paths = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    sess_opts = SessionOptions()

    # 5.1 Profiling 활성화
    if run_profile:
        sess_opts.enable_profiling = True
        sess_opts.profile_file_prefix = f"{safe_model}_{onnx_dir}_{timestamp}"

    # 5.2 스레드 및 메모리 옵션
    sess_opts.inter_op_num_threads  = 1     # 노드 간 병렬 스레드 수
    sess_opts.intra_op_num_threads  = 8     # 노드 내부 병렬 스레드 수
    sess_opts.enable_cpu_mem_arena  = True  # CPU 메모리 어레나 활성화
    sess_opts.enable_mem_pattern    = True  # 메모리 패턴 최적화 활성화
    sess_opts.execution_mode        = ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level    = 2     # WARNING (0=VERBOSE, 1=INFO, 2=WARNING)

    # 5.1 InferenceSession 생성
    session = InferenceSession(paths[0], sess_opts)
    inp_names = [inp.name for inp in session.get_inputs()]

    # 5.2 Warm-up
    print(f"[Evaluate] Warming up for {WARMUP_STEPS} steps on {onnx_dir}...")
    for ex in dataset.select(range(min(WARMUP_STEPS, len(dataset)))):
        feed = {n: np.expand_dims(ex[n], 0) for n in inp_names}
        session.run(None, feed)

    # 5.3 실제 추론 및 시간 측정
    total_t = 0.0
    for ex in tqdm(dataset, desc=f"[Evaluate] Running inference on {onnx_dir}"):
        feed = {n: np.expand_dims(ex[n], 0) for n in inp_names}
        start = time.time()
        session.run(None, feed)
        total_t += time.time() - start

    # 5.4 Accuracy 계산
    metric = eval_lib.load("glue", "mrpc")
    for ex in dataset:
        feed = {n: np.expand_dims(ex[n], 0) for n in inp_names}
        outputs = session.run(None, feed)
        logits = outputs[0]
        pred   = np.argmax(logits, axis=-1).astype(int).item()
        metric.add_batch(predictions=[pred], references=[int(ex["label"])])
    acc = metric.compute()["accuracy"]

    # 5.5 Profiling 파일 경로 수집
    prof_path = None
    if run_profile:
        prof_path = session.end_profiling()
        print(f"[Profile] Data saved to: {prof_path}")

    # 5.6 모델 파일 크기 (MB 단위)
    size_mb = os.path.getsize(paths[0]) / 1e6

    return total_t, size_mb, acc, prof_path

# ───────────────────────────────────────────────────────────────────────────────
# 6. FP32 vs Quant 모델 비교 및 출력
# ───────────────────────────────────────────────────────────────────────────────
results = {}
for name, path in [("FP32", "onnx_bert_fp32"), ("Quant", "onnx_bert_int8")]:
    print(f"[Compare] Evaluating {name} model...")
    total_t, size_mb, acc, prof_path = evaluate_tuned(path)
    results[name] = {
        "time": total_t,
        "size": size_mb,
        "acc":  acc,
        "profile": prof_path
    }
    print(f"[Compare] {name} → time: {total_t:.3f}s | size: {size_mb:.3f}MB | acc: {acc:.4f}\n")

print("# ────────────────────────────────────────────────────")
print("# Final Results")
print("# ────────────────────────────────────────────────────")
for name in results:
    print(f"{name} → time: {results[name]['time']:.3f}s | "
          f"size: {results[name]['size']:.3f}MB | "
          f"acc: {results[name]['acc']:.4f}")
print()

# ───────────────────────────────────────────────────────────────────────────────
# 7. Quantization Op 비율 계산 (Graph 분석)
# ───────────────────────────────────────────────────────────────────────────────
def quant_op_ratio(onnx_path):
    model = onnx.load(onnx_path)
    ops   = [node.op_type for node in model.graph.node]
    total = len(ops)
    quant_ops = [
        op for op in ops if any(q in op for q in [
            "DynamicQuantizeLinear", "DynamicQuantizeMatMul", "MatMulIntegerToFloat", "QLinearGemm"
        ])
    ]
    ratio = (len(quant_ops) / total * 100) if total else 0.0
    return len(quant_ops), total, ratio

for name, model_path in [
    ("FP32", "onnx_bert_fp32/model.onnx"),
    ("Quant", "onnx_bert_int8/model.onnx")
]:
    qn, tot, pct = quant_op_ratio(model_path)
    print(f"[Graph] {name} → {qn}/{tot} QuantOps ({pct:.2f}%)")

# 스크립트 종료
import sys
sys.exit(0)
