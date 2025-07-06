import json, argparse, sys, collections, re
import onnx
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Profiling JSON Analyzer + Quant Operator Breakdown")
    p.add_argument("json_file", help="ONNX Runtime profiling json")
    p.add_argument("--onnx", required=True, help="동일 세션 ONNX 모델 경로")
    return p.parse_args()

def load_events(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def find_dequant_nodes(events):
    # 1) Kernel Event Count
    node_ev = [
        ev for ev in events
        if ev.get("cat")=="Node"
           and ev.get("dur",0)>0
           and ev.get("name","").endswith("_kernel_time")
    ]
    cast_deq = set()
    mul_deq  = set()
    # 2) Sequential Scan : MatMulInteger → Cast → Mul → Mul
    for i in range(len(node_ev)-3):
        if node_ev[i]["args"].get("op_name") == "MatMulInteger":
            e1, e2, e3 = node_ev[i+1], node_ev[i+2], node_ev[i+3]
            if (e1["args"].get("op_name")=="Cast"
                and e2["args"].get("op_name")=="Mul"
                and e3["args"].get("op_name")=="Mul"):
                proto1 = e1["name"][:-len("_kernel_time")]
                proto2 = e2["name"][:-len("_kernel_time")]
                proto3 = e3["name"][:-len("_kernel_time")]
                cast_deq.add(proto1)
                mul_deq.update({proto2, proto3})
    return cast_deq, mul_deq

def analyze(events, cast_deq, mul_deq):
    # A) Node Events
    node_events = [
        ev for ev in events
        if ev.get("cat")=="Node"
           and ev.get("dur",0)>0
           and ev.get("name","").endswith("_kernel_time")
    ]

    # B) Op_name Based Count/ Kernel time
    op_cnt, op_time = collections.Counter(), collections.Counter()
    # proto_name → op_name mapping
    name2op = {}
    for ev in node_events:
        op = ev.get("args",{}).get("op_name", ev.get("name"))
        dur = ev.get("dur",0)
        op_cnt[op]  += 1
        op_time[op] += dur
        proto = ev["name"][:-len("_kernel_time")]
        name2op[proto] = op

    tot_evt, tot_time = sum(op_cnt.values()), sum(op_time.values())

    # Operator Count & Time Ratio
    print("\n# Operator Count & Time Ratio (Node Events Only)")
    print(f"{'Name':30s} {'Cnt':>6} {'Cnt%':>6}  {'Time(s)':>8} {'Time%':>6}")
    for op, cnt in op_cnt.most_common():
        print(f"{op:30s} {cnt:6d} {cnt/tot_evt*100:6.2f}%  "
              f"{op_time[op]/1e6:8.3f} {op_time[op]/tot_time*100:6.2f}%")

    # C) Dequant Count / Kernel Time 
    deq_cast_count = 0
    deq_cast_time  = 0
    deq_mul_count  = 0
    deq_mul_time   = 0
    for ev in node_events:
        proto = ev["name"][:-len("_kernel_time")]
        if proto in cast_deq:
            deq_cast_count += 1
            deq_cast_time  += ev["dur"]
        if proto in mul_deq:
            deq_mul_count += 1
            deq_mul_time  += ev["dur"]

    # Grouped Operator Ratios 
    quant_pat = re.compile(
        r'^(DynamicQuantizeLinear|'
        r'MatMulInteger|'
    )
    # Quant Ops Without Dequant Ops Count/ Kernel Time 
    base_cnt_q = sum(op_cnt[o] for o in op_cnt if quant_pat.match(o))
    base_time_q= sum(op_time[o] for o in op_time if quant_pat.match(o))
    # Qaunt Ops with Dequant Ops Count / Kernel Time 
    cnt_q = base_cnt_q + deq_cast_count + deq_mul_count
    time_q = base_time_q+ deq_cast_time  + deq_mul_time

    cnt_n = tot_evt - cnt_q
    time_n= tot_time - time_q

    key_ops = ['Add','Mul','Div','MatMul']
    cnt_k = sum(op_cnt[o] for o in key_ops if o in op_cnt)
    time_k= sum(op_time[o] for o in key_ops if o in op_time)
    cnt_r = cnt_n - cnt_k
    time_r= time_n - time_k

    print("\n# Grouped Operator Ratios")
    def pr(label, c, t):
        print(f"{label:30s} {c:6d} {c/tot_evt*100:6.2f}%  "
              f"{t/1e6:8.3f} {t/tot_time*100:6.2f}%")
    pr("Quant Ops",          cnt_q,  time_q)
    pr("Non-Quant Ops",      cnt_n,  time_n)
    pr("Add,Mul,Div,MatMul", cnt_k,  time_k)
    pr("Other Non-Quant",    cnt_r,  time_r)

    # ③ Quant / Non-Quant Ops List
    quant_ops = sorted({o for o in op_cnt if quant_pat.match(o)} 
                       | {name2op[p] for p in cast_deq|mul_deq if p in name2op})
    nonquant_ops = sorted(set(op_cnt) - set(quant_ops))

    print("\n# Quant Ops List:")
    print(", ".join(quant_ops) or "— No OPs —")
    print("\n# Non-Quant Ops List:")
    print(", ".join(nonquant_ops) or "— No OPs —")

    # Quant Operator Ratios 
    quant_nodes  = {'DynamicQuantizeLinear'}
    int8mm_nodes = {'MatMulInteger'}
    def sum_c(ops):
        return sum(op_cnt[o] for o in ops if o in op_cnt), \
               sum(op_time[o] for o in ops if o in op_time)

    cnt_q2, time_q2 = sum_c(quant_nodes)
    cnt_m2, time_m2 = sum_c(int8mm_nodes)
    # Quant(DynamicQuantizeLinear), INT8 MatMul(MatMulInteger), Dequant(Cast/Mul) 
    pr_map = [
        ("Quant (FP32→INT8)", cnt_q2, time_q2),
        ("INT8 MatMul",        cnt_m2, time_m2),
        ("Dequant Cast",       deq_cast_count,  deq_cast_time),
        ("Dequant Mul",        deq_mul_count,   deq_mul_time),
    ]
    print("\n# Quant Operator Ratios")
    for label, c, t in pr_map:
        print(f"{label:30s} {c:6d} {c/tot_evt*100:6.2f}%  "
              f"{t/1e6:8.3f} {t/tot_time*100:6.2f}%")

def main():
    args = parse_args()
    events = load_events(args.json_file)
    cast_deq, mul_deq = find_dequant_nodes(events)
    analyze(events, cast_deq, mul_deq)

if __name__=="__main__":
    main()
