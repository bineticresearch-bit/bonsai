"""Analyze a Phase 1 experiment result and determine PASS/FAIL.

Usage: /opt/homebrew/bin/python3.12 experiments/01-frozen-baseline/analyze_result.py --result experiments/01-frozen-baseline/results/test_N_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VALIDATION_CRITERIA = {
    1: {"name": "baseline_recall", "metric": "recall_accuracy", "target": 0.85, "op": ">="},
    2: {"name": "temporal_kv_recall", "metric": "recall_accuracy", "target": 0.0, "op": ">="},
    3: {"name": "memory_projection", "metric": "projected_peak_10m_gb", "target": 64, "op": "<="},
    4: {"name": "address_routing", "metric": "address_lookup_accuracy", "target": 0.99, "op": ">="},
    5: {"name": "kv_injection_recall", "metric": "recall_accuracy", "target": 0.75, "op": ">="},
    6: {"name": "kv_injection_tree", "metric": "recall_accuracy", "target": 0.75, "op": ">="},
}


def check_criterion(result: dict, test_num: int) -> tuple[str, str]:
    criteria = VALIDATION_CRITERIA.get(test_num)
    if not criteria:
        return "UNKNOWN", "No criteria for this test"

    metric = criteria["metric"]
    target = criteria["target"]
    op = criteria["op"]

    value = result.get(metric)
    if value is None:
        return "FAIL", f"Metric '{metric}' not found in result"

    if op == ">=":
        passed = value >= target
    elif op == "<=":
        passed = value <= target
    elif op == ">":
        passed = value > target
    elif op == "<":
        passed = value < target
    else:
        passed = False

    status = "PASS" if passed else "FAIL"
    detail = f"{metric}={value}, target={op}{target}"
    return status, detail


def analyze(result_path: str) -> dict:
    result = json.loads(Path(result_path).read_text())
    test_num = result.get("test", 0)

    status, detail = check_criterion(result, test_num)
    criteria = VALIDATION_CRITERIA.get(test_num, {})

    analysis = {
        "test": test_num,
        "test_name": criteria.get("name", "unknown"),
        "status": status,
        "metric": criteria.get("metric", ""),
        "detail": detail,
        "source_result": Path(result_path).name,
        "elapsed": result.get("elapsed_seconds", 0),
        "timestamp": result.get("timestamp", ""),
    }

    # Extract key metrics based on test type
    if test_num == 1:
        analysis["baseline_recall"] = result.get("recall_accuracy")
        analysis["total_tokens"] = result.get("total_tokens")
    elif test_num == 2:
        analysis["temporal_recall"] = result.get("recall_accuracy")
        analysis["degradation"] = result.get("degradation_pct")
        analysis["baseline_recall"] = result.get("baseline_recall")
        analysis["temporal_improvement"] = result.get("temporal_improvement_pct")
        analysis["slots_created"] = result.get("slots_created")
        analysis["kv_bytes"] = result.get("kv_bytes")
    elif test_num == 3:
        analysis["projected_peak"] = result.get("projected_peak_10m_gb")
        analysis["weight_size"] = result.get("weight_size_gb")
        analysis["chunk_size"] = result.get("chunk_size_tokens")
        analysis["slots_10m"] = result.get("slots_for_10m_tokens")
    elif test_num == 4:
        analysis["address_accuracy"] = result.get("address_lookup_accuracy")
        analysis["roundtrip_accuracy"] = result.get("roundtrip_accuracy")
        analysis["slots_created"] = result.get("total_slots_created")
    elif test_num == 5:
        analysis["kv_recall"] = result.get("recall_accuracy")
        analysis["injected_seq_len"] = result.get("injected_seq_len")
        analysis["injected_layers"] = result.get("injected_layers")
        analysis["baseline_recall"] = result.get("baseline_recall")
    elif test_num == 6:
        analysis["kv_recall"] = result.get("recall_accuracy")
        analysis["injected_seq_len"] = result.get("injected_seq_len")
        analysis["chunks_processed"] = result.get("chunks_processed")
        analysis["tree_depth"] = result.get("tree_depth")

    return analysis


def write_analysis_md(analysis: dict, output_path: str):
    md = f"""# Analysis: Test {analysis['test']} — {analysis['test_name']}

## Result

**Status: {analysis['status']}**

{detail_line(analysis)}

## Metrics

| Metric | Value |
|--------|-------|
| Test | {analysis['test']} |
| Name | {analysis['test_name']} |
| Elapsed | {analysis.get('elapsed', 0)}s |
| Timestamp | {analysis.get('timestamp', '')} |

## Details

{json.dumps({k: v for k, v in analysis.items() if k not in ('test', 'test_name', 'status', 'metric', 'detail', 'source_result', 'elapsed', 'timestamp')}, indent=2)}
"""
    Path(output_path).write_text(md)


def detail_line(a: dict) -> str:
    return f"- **{a.get('metric', '')}**: {a.get('detail', '')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="Path to test result JSON")
    args = parser.parse_args()

    analysis = analyze(args.result)

    out_dir = Path(args.result).parent
    out_path = str(out_dir / f"analysis_{analysis['test']}.md")
    write_analysis_md(analysis, out_path)

    print("=" * 50)
    print(f"Analysis: Test {analysis['test']} — {analysis['test_name']}")
    print("=" * 50)
    print(f"Status: {analysis['status']}")
    print(f"Detail: {analysis['detail']}")
    print(f"Written to: {out_path}")
    return 0 if analysis["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
