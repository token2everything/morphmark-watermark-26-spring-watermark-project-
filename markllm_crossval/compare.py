"""对比我们的实现与 MarkLLM 官方实现的评测结果"""

import json
from pathlib import Path


OUR_RESULTS = Path("outputs/results")
MARKLLM_RESULTS = Path("outputs/markllm_results")


def load_ours(name: str) -> dict:
    for p in OUR_RESULTS.glob(f"{name}_c4_*.json"):
        return json.load(p.open())
    return {}


def load_markllm(name: str) -> dict:
    for p in MARKLLM_RESULTS.glob(f"markllm_{name}_c4_*.json"):
        return json.load(p.open())
    return {}


def compare():
    methods = ["KGW"]  # MarkLLM v0.1.5 only has KGW, no MorphMark
    morphmark_variants = ["morphmark_exp", "morphmark_linear", "morphmark_log"]

    print(f"\n{'='*80}")
    print("Cross-Validation: Ours vs MarkLLM (Official)")
    print(f"{'='*80}")
    print(f"{'Method':<20s} {'Source':<12s} {'TPR@1%':>8s} {'AUC':>8s} {'Z_w_mean':>10s} {'Z_uw_mean':>10s}")
    print("-" * 72)

    for method in methods:
        ours = load_ours(method.lower())
        theirs = load_markllm(method)

        if ours:
            ours_tpr = ours.get("tpr_at_fpr_0.01", "?")
            ours_auc = ours.get("auc_roc", "?")
            ours_zw = ours.get("z_w_mean", "?")
            ours_zuw = ours.get("z_uw_mean", "?")
            print(f"{method:<20s} {'Ours':<12s} {str(ours_tpr):>8s} {str(ours_auc):>8s} {str(ours_zw):>10s} {str(ours_zuw):>10s}")

        if theirs:
            th_tpr = theirs.get("tpr_at_fpr_0.01", "?")
            th_auc = theirs.get("auc_roc", "?")
            th_zw = theirs.get("z_w_mean", "?")
            th_zuw = theirs.get("z_uw_mean", "?")
            print(f"{method:<20s} {'MarkLLM':<12s} {str(th_tpr):>8s} {str(th_auc):>8s} {str(th_zw):>10s} {str(th_zuw):>10s}")

            if ours and isinstance(ours_tpr, (int, float)) and isinstance(th_tpr, (int, float)):
                diff = abs(ours_tpr - th_tpr)
                flag = "⚠️  >5%" if diff > 0.05 else "✓"
                print(f"  → TPR diff: {diff:.4f} {flag}")
            if ours and isinstance(ours_auc, (int, float)) and isinstance(th_auc, (int, float)):
                diff = abs(ours_auc - th_auc)
                flag = "⚠️  >5%" if diff > 0.05 else "✓"
                print(f"  → AUC diff: {diff:.4f} {flag}")
        print()

    # Also show our MorphMark results (no MarkLLM to compare against)
    print(f"{'─'*80}")
    print("Our MorphMark results (no MarkLLM reference — MarkLLM v0.1.5 lacks MorphMark):")
    print(f"{'Method':<20s} {'Source':<12s} {'TPR@1%':>8s} {'AUC':>8s} {'Z_w_mean':>10s} {'Z_uw_mean':>10s}")
    print("-" * 72)
    for method in morphmark_variants:
        ours = load_ours(method)
        if ours:
            ours_tpr = ours.get("tpr_at_fpr_0.01", "?")
            ours_auc = ours.get("auc_roc", "?")
            ours_zw = ours.get("z_w_mean", "?")
            ours_zuw = ours.get("z_uw_mean", "?")
            label = method.replace("morphmark_", "MM_")
            print(f"{label:<20s} {'Ours':<12s} {str(ours_tpr):>8s} {str(ours_auc):>8s} {str(ours_zw):>10s} {str(ours_zuw):>10s}")

    print(f"{'='*80}")


if __name__ == "__main__":
    compare()
