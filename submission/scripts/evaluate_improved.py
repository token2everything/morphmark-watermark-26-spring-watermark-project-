"""评测改进方法 vs 原版 MorphMark"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.watermarking.morphmark import MorphMarkLogitsProcessor, morphmark_detect
from src.watermarking.extensions import (
    MorphMarkFloorLogitsProcessor,
    EntropyAdaptiveMorphMarkLogitsProcessor,
)
from src.watermarking.adaptive import ExpStrength
from src.data.loader import load_dataset_by_name
from src.evaluation.metrics import compute_tpr_at_fpr, compute_best_f1, compute_auc_roc
from src.evaluation.quality import compute_ppl_simple


def run_comparison(n_samples: int = 50, variants: str = "all"):
    """Run head-to-head comparison of improvement variants vs vanilla."""
    print("Loading dataset...")
    prompts = load_dataset_by_name("c4", n_samples=n_samples, seed=42)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    vocab_size = model.config.vocab_size

    gen_kwargs = {
        "max_new_tokens": 200, "do_sample": True,
        "temperature": 0.7, "top_p": 0.9,
    }

    all_methods = {
        "MorphMark_exp (vanilla)": MorphMarkLogitsProcessor(
            adaptive=ExpStrength(p0=0.15, k=1.30),
            key=15485863, gamma=0.5, vocab_size=vocab_size,
        ),
        "MorphMarkFloor (δ=0.5)": MorphMarkFloorLogitsProcessor(
            adaptive=ExpStrength(p0=0.15, k=1.30),
            key=15485863, gamma=0.5, vocab_size=vocab_size,
            delta_floor=0.5,
        ),
        "MorphMarkFloor (δ=1.0)": MorphMarkFloorLogitsProcessor(
            adaptive=ExpStrength(p0=0.15, k=1.30),
            key=15485863, gamma=0.5, vocab_size=vocab_size,
            delta_floor=1.0,
        ),
        "EntropyAdaptive (α=0.3)": EntropyAdaptiveMorphMarkLogitsProcessor(
            adaptive=ExpStrength(p0=0.15, k=1.30),
            key=15485863, gamma=0.5, vocab_size=vocab_size,
            p0_base=0.15, alpha=0.3,
        ),
        "EntropyAdaptive (α=0.5)": EntropyAdaptiveMorphMarkLogitsProcessor(
            adaptive=ExpStrength(p0=0.15, k=1.30),
            key=15485863, gamma=0.5, vocab_size=vocab_size,
            p0_base=0.15, alpha=0.5,
        ),
    }

    if variants == "all":
        methods = all_methods
    elif variants == "floor":
        methods = {k: v for k, v in all_methods.items() if "Floor" in k or "vanilla" in k}
    elif variants == "entropy":
        methods = {k: v for k, v in all_methods.items() if "Entropy" in k or "vanilla" in k}
    else:
        # Comma-separated substrings to match
        selected = set(variants.split(","))
        methods = {k: v for k, v in all_methods.items()
                   if any(s.lower() in k.lower() for s in selected)}

    all_results = {}

    for method_name, processor in methods.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")

        results_uw = []
        results_w = []
        texts_uw = []
        texts_w = []

        for i, prompt in enumerate(tqdm(prompts, desc=method_name)):
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=512).to(model.device)
            prompt_len = encoded.input_ids.shape[1]

            # Unwatermarked
            torch.manual_seed(42 + i)
            out_uw = model.generate(**encoded, **gen_kwargs)
            text_uw = tokenizer.decode(out_uw[0], skip_special_tokens=True)
            tokens_uw = out_uw[0].cpu()
            texts_uw.append(text_uw)

            # Watermarked
            torch.manual_seed(42 + i)
            out_w = model.generate(
                **encoded, **gen_kwargs,
                logits_processor=LogitsProcessorList([processor]),
            )
            text_w = tokenizer.decode(out_w[0], skip_special_tokens=True)
            tokens_w = out_w[0].cpu()
            texts_w.append(text_w)

            r_uw = morphmark_detect(tokens_uw, 15485863, 0.5, vocab_size, prompt_len=prompt_len)
            r_w = morphmark_detect(tokens_w, 15485863, 0.5, vocab_size, prompt_len=prompt_len)

            results_uw.append(r_uw)
            results_w.append(r_w)

        z_uw = np.array([r["z_score"] for r in results_uw])
        z_w = np.array([r["z_score"] for r in results_w])

        ppl_uw = compute_ppl_simple(model, tokenizer, texts_uw, device=model.device)
        ppl_w = compute_ppl_simple(model, tokenizer, texts_w, device=model.device)
        ppl_uw_mean = float(np.mean([p for p in ppl_uw if not np.isnan(p)]))
        ppl_w_mean = float(np.mean([p for p in ppl_w if not np.isnan(p)]))

        tpr_1 = compute_tpr_at_fpr(z_w, z_uw, target_fpr=0.01)
        auc = compute_auc_roc(z_w, z_uw)
        f1_info = compute_best_f1(z_w, z_uw)

        result = {
            "method": method_name,
            "n_samples": n_samples,
            "z_uw_mean": float(z_uw.mean()),
            "z_uw_std": float(z_uw.std()),
            "z_w_mean": float(z_w.mean()),
            "z_w_std": float(z_w.std()),
            "ppl_uw_mean": ppl_uw_mean,
            "ppl_w_mean": ppl_w_mean,
            "ppl_delta_pct": (ppl_w_mean - ppl_uw_mean) / ppl_uw_mean * 100,
            "tpr_at_fpr_0.01": tpr_1,
            "auc_roc": auc,
            "best_f1": f1_info["best_f1"],
        }

        print(f"  Z_uw: {z_uw.mean():.3f} ± {z_uw.std():.3f}")
        print(f"  Z_w:  {z_w.mean():.3f} ± {z_w.std():.3f}")
        print(f"  PPL_uw: {ppl_uw_mean:.2f}")
        print(f"  PPL_w:  {ppl_w_mean:.2f} (Δ={result['ppl_delta_pct']:.1f}%)")
        print(f"  TPR@1%: {tpr_1:.4f}")
        print(f"  AUC:    {auc:.4f}")
        print(f"  Best F1: {f1_info['best_f1']:.4f}")

        all_results[method_name] = result

    # Comparison table
    print(f"\n{'='*80}")
    print("Improvement Comparison")
    print(f"{'='*80}")
    print(f"{'Method':<30s} {'PPL_w':>8s} {'ΔPPL%':>8s} {'TPR@1%':>8s} {'AUC':>8s} {'Z_w':>8s}")
    print("-" * 70)
    for name, r in all_results.items():
        short = name.split("(")[0].strip()[:28]
        print(f"{short:<30s} {r['ppl_w_mean']:8.2f} {r['ppl_delta_pct']:7.1f}% "
              f"{r['tpr_at_fpr_0.01']:8.4f} {r['auc_roc']:8.4f} {r['z_w_mean']:8.2f}")

    # Save
    out_path = Path(f"outputs/ablation/improvement_{variants}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--variants", type=str, default="entropy",
                       help="Which variants: all, floor, entropy, or comma-separated names")
    args = parser.parse_args()
    run_comparison(args.n_samples, args.variants)
