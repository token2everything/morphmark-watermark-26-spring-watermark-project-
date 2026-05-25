"""EWD 熵加权检测验证：在已有生成文本上对比 EWD vs 标准 z-score"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.ewd import ewd_detect
from src.evaluation.metrics import compute_tpr_at_fpr, compute_auc_roc

RESULT_FILES = {
    "kgw": "outputs/results/kgw_c4_50.json",
    "morphmark_exp": "outputs/results/morphmark_exp_c4_50.json",
    "morphmark_linear": "outputs/results/morphmark_linear_c4_50.json",
    "morphmark_log": "outputs/results/morphmark_log_c4_50.json",
}


def load_results(method: str) -> dict:
    path = Path(RESULT_FILES[method])
    if not path.exists():
        raise FileNotFoundError(f"No results for {method}: {path}")
    with open(path) as f:
        return json.load(f)


def compute_ewd_for_texts(
    model, tokenizer, texts: list[str], key: int, gamma: float, vocab_size: int
) -> list[dict]:
    """对文本列表运行 EWD 检测。"""
    results = []
    for text in tqdm(texts, desc="EWD detect", leave=False):
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        r = ewd_detect(model, encoded, key, gamma, vocab_size, prompt_len=0)
        results.append(r)
    return results


def run_ewd_validation():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    vocab_size = model.config.vocab_size
    key, gamma = 15485863, 0.5

    all_summary = []

    for method_name in RESULT_FILES:
        print(f"\n{'='*60}")
        print(f"EWD Validation: {method_name}")
        print(f"{'='*60}")

        data = load_results(method_name)
        texts_uw = [r["text"] for r in data["results_uw"]]
        texts_w = [r["text"] for r in data["results_w"]]
        z_std_uw = np.array([r["z_score"] for r in data["results_uw"]])
        z_std_w = np.array([r["z_score"] for r in data["results_w"]])

        print("Running EWD on unwatermarked texts...")
        ewd_uw = compute_ewd_for_texts(model, tokenizer, texts_uw, key, gamma, vocab_size)
        z_ewd_uw = np.array([r["z_score_ewd"] for r in ewd_uw])
        h_uw = np.array([r["mean_entropy"] for r in ewd_uw])

        print("Running EWD on watermarked texts...")
        ewd_w = compute_ewd_for_texts(model, tokenizer, texts_w, key, gamma, vocab_size)
        z_ewd_w = np.array([r["z_score_ewd"] for r in ewd_w])
        h_w = np.array([r["mean_entropy"] for r in ewd_w])

        # Comparison metrics
        tpr_std = compute_tpr_at_fpr(z_std_w, z_std_uw, target_fpr=0.01)
        tpr_ewd = compute_tpr_at_fpr(z_ewd_w, z_ewd_uw, target_fpr=0.01)
        auc_std = compute_auc_roc(z_std_w, z_std_uw)
        auc_ewd = compute_auc_roc(z_ewd_w, z_ewd_uw)

        summary = {
            "method": method_name,
            "mean_entropy_uw": float(h_uw.mean()),
            "mean_entropy_w": float(h_w.mean()),
            "std": {
                "z_uw_mean": float(z_std_uw.mean()),
                "z_w_mean": float(z_std_w.mean()),
                "tpr_at_1pct": float(tpr_std),
                "auc_roc": float(auc_std),
            },
            "ewd": {
                "z_uw_mean": float(z_ewd_uw.mean()),
                "z_w_mean": float(z_ewd_w.mean()),
                "tpr_at_1pct": float(tpr_ewd),
                "auc_roc": float(auc_ewd),
            },
            "delta_tpr": float(tpr_ewd - tpr_std),
            "delta_auc": float(auc_ewd - auc_std),
        }

        print(f"  Mean entropy: uw={h_uw.mean():.3f}, w={h_w.mean():.3f}")
        print(f"  Standard:  z_uw={z_std_uw.mean():.2f}, z_w={z_std_w.mean():.2f}, TPR@1%={tpr_std:.4f}, AUC={auc_std:.4f}")
        print(f"  EWD:       z_uw={z_ewd_uw.mean():.2f}, z_w={z_ewd_w.mean():.2f}, TPR@1%={tpr_ewd:.4f}, AUC={auc_ewd:.4f}")
        print(f"  Δ: TPR={tpr_ewd - tpr_std:+.4f}, AUC={auc_ewd - auc_std:+.4f}")
        all_summary.append(summary)

    # Overall comparison
    print(f"\n{'='*70}")
    print("EWD vs Standard Detection — Summary")
    print(f"{'='*70}")
    print(f"{'Method':<20s} {'Std Z_w':>10s} {'EWD Z_w':>10s} {'Std TPR':>10s} {'EWD TPR':>10s} {'ΔTPR':>10s}")
    print("-" * 75)
    for s in all_summary:
        print(f"{s['method']:<20s} {s['std']['z_w_mean']:10.2f} {s['ewd']['z_w_mean']:10.2f} "
              f"{s['std']['tpr_at_1pct']:10.4f} {s['ewd']['tpr_at_1pct']:10.4f} "
              f"{s['delta_tpr']:+10.4f}")

    # Save
    out_path = Path("outputs/ablation/ewd_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_ewd_validation()
