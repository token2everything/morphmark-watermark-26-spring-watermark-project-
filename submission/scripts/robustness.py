"""鲁棒性测试：对已生成的水印文本施加攻击，测量检测退化"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.watermarking.kgw import kgw_detect
from src.watermarking.morphmark import morphmark_detect
from src.evaluation.robustness import wordnet_synonym_attack, random_deletion_attack


def load_generated_texts(method: str) -> tuple[list[str], list[str], dict]:
    """加载已生成的水印文本和参数。"""
    path = Path(f"outputs/results/{method}_c4_50.json")
    if not path.exists():
        raise FileNotFoundError(f"No results for {method}: {path}")
    with open(path) as f:
        data = json.load(f)

    texts_w = [r["text"] for r in data["results_w"]]
    z_scores_orig = [r["z_score"] for r in data["results_w"]]
    return texts_w, z_scores_orig, data


def detect_texts(texts: list[str], tokenizer, vocab_size: int, params: dict, detect_fn) -> list[float]:
    """对受损文本重新检测 z-score。"""
    z_scores = []
    for text in tqdm(texts, desc="Detecting", leave=False):
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        r = detect_fn(
            encoded, params["key"], params["gamma"], vocab_size,
            prefix_length=params.get("prefix_length", 1), prompt_len=0,
        )
        z_scores.append(r["z_score"])
    return z_scores


def run_robustness_test(method: str):
    """运行完整的鲁棒性测试套件。"""
    print(f"\n{'='*60}")
    print(f"Robustness Test: {method}")
    print(f"{'='*60}")

    texts_w, z_orig, data = load_generated_texts(method)
    params = {
        "key": 15485863, "gamma": 0.5, "prefix_length": 1,
    }

    # Dispatch detection function
    if method.startswith("morphmark"):
        detect_fn = morphmark_detect
    else:
        detect_fn = kgw_detect

    print("Loading tokenizer for detection...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    vocab_size = 50272  # OPT-1.3B model vocab size

    print(f"\nOriginal (no attack): mean z = {np.mean(z_orig):.3f}")

    results = []

    # WordNet synonym attack at varying strengths
    for ratio in [0.1, 0.3, 0.5, 0.7]:
        print(f"\n--- WordNet synonym attack (ratio={ratio}) ---")
        attacked = []
        for text in tqdm(texts_w, desc=f"WN attack {ratio}", leave=False):
            try:
                attacked.append(wordnet_synonym_attack(text, ratio=ratio))
            except Exception:
                attacked.append(text)  # fallback on NLTK errors
        z_scores = detect_texts(attacked, tokenizer, vocab_size, params, detect_fn)
        z_arr = np.array(z_scores)
        tpr_at_4 = np.mean(z_arr > 4.0)
        results.append({
            "attack": "wordnet_synonym", "ratio": ratio,
            "z_mean": float(z_arr.mean()), "z_std": float(z_arr.std()),
            "z_median": float(np.median(z_arr)),
            "tpr_at_z4": float(tpr_at_4),
            "z_retention": float(z_arr.mean() / np.mean(z_orig)),
        })
        print(f"  Z: {z_arr.mean():.2f} ± {z_arr.std():.2f}, "
              f"TPR@z>4: {tpr_at_4:.2%}, "
              f"Z retention: {results[-1]['z_retention']:.2%}")

    # Random deletion attack at varying strengths
    for ratio in [0.1, 0.3, 0.5]:
        print(f"\n--- Random deletion attack (ratio={ratio}) ---")
        attacked = []
        for text in tqdm(texts_w, desc=f"Del attack {ratio}", leave=False):
            attacked.append(random_deletion_attack(text, ratio=ratio))
        z_scores = detect_texts(attacked, tokenizer, vocab_size, params, detect_fn)
        z_arr = np.array(z_scores)
        tpr_at_4 = np.mean(z_arr > 4.0)
        results.append({
            "attack": "random_deletion", "ratio": ratio,
            "z_mean": float(z_arr.mean()), "z_std": float(z_arr.std()),
            "z_median": float(np.median(z_arr)),
            "tpr_at_z4": float(tpr_at_4),
            "z_retention": float(z_arr.mean() / np.mean(z_orig)),
        })
        print(f"  Z: {z_arr.mean():.2f} ± {z_arr.std():.2f}, "
              f"TPR@z>4: {tpr_at_4:.2%}, "
              f"Z retention: {results[-1]['z_retention']:.2%}")

    # Summary table
    print(f"\n{'─'*70}")
    print(f"Robustness Summary: {method}")
    print(f"{'Attack':<25s} {'Ratio':>6s} {'Z_mean':>8s} {'TPR@z>4':>8s} {'Z_ret':>8s}")
    print("-" * 60)
    for r in results:
        print(f"{r['attack']:<25s} {r['ratio']:6.1f} {r['z_mean']:8.2f} "
              f"{r['tpr_at_z4']:7.1%} {r['z_retention']:7.1%}")

    # Save
    out_dir = Path("outputs/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"robustness_{method}.json"
    with open(out_file, "w") as f:
        json.dump({"method": method, "original_z_mean": float(np.mean(z_orig)),
                   "results": results}, f, indent=2)
    print(f"\nSaved to {out_file}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="morphmark_exp,morphmark_linear,morphmark_log")
    args = parser.parse_args()

    for method in args.methods.split(","):
        try:
            run_robustness_test(method)
        except FileNotFoundError as e:
            print(f"Skip {method}: {e}")


if __name__ == "__main__":
    main()
