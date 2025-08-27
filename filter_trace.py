import json
from collections import defaultdict


igno_cat_list = [
        'cpu_op', # cpu time
        'user_annotation', #cpu time
        #'kernel',
        #'cuda_runtime',
        ]

def analyze_trace(trace_file, threshold=200):
    with open(trace_file, "r") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", trace)  # 有些版本是 traceEvents

    stats = defaultdict(list)

    for e in events:
        if e.get("ph") == "X" and "dur" in e and  e['cat'] not in igno_cat_list:
            dur = e["dur"] /1000 # us -- ms
            if dur > threshold:
                name = e.get('cat') + '$' + e.get("name", "unknown")
                if 'Profiler' in name:
                    continue
                stats[name].append(dur)

    print(f"Events with dur > {threshold} ms:\n")
    print(f"{'Category':20s}  {'Name':40s} {'Count':>8s} {'Avg dur(ms)':>15s} {'Max dur(ms)':>15s}")
    print("-" * 105)
    for name, durs in sorted(stats.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(durs)
        avg_dur = sum(durs) / count
        max_dur = max(durs)
        print(f"{name.split('$')[0]:20s} {name.split('$')[1][:40]:40s} {count:8d} {avg_dur:15.2f} {max_dur:15.2f}")

if __name__ == "__main__":
    analyze_trace("vita_8b_tp4_gbs8_190.json", threshold=50) # ms
