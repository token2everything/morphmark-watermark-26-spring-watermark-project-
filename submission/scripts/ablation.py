"""消融实验：参数扫参，收集 TPR 和 PPL"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.watermarking.kgw import KGWLogitsProcessor, kgw_detect
from src.watermarking.morphmark import MorphMarkLogitsProcessor, morphmark_detect
from src.watermarking.adaptive import create_adaptive_strength
from src.data.loader import load_dataset_by_name
from src.evaluation.metrics import compute_tpr_at_fpr, compute_auc_roc
from src.evaluation.quality import compute_ppl_simple


def build_processor(watermark_type: str, vocab_size: int, params: dict):
    """根据参数构建 LogitsProcessor。"""
    wm_type = params.get("type", watermark_type)
    key = params.get("key", 15485863)
    gamma = params.get("gamma", 0.5)
    prefix_length = params.get("prefix_length", 1)

    if wm_type == "kgw":
        return KGWLogitsProcessor(
            key=key, gamma=gamma, delta=params.get("delta", 2.0),
            vocab_size=vocab_size, prefix_length=prefix_length,
        )
    elif wm_type.startswith("morphmark"):
        adaptive = create_adaptive_strength(
            wm_type,  # full name like "morphmark_exp"
            p0=params.get("p0", 0.15),
            k=params.get("k", 1.30),
            epsilon=params.get("epsilon", 1e-10),
        )
        return MorphMarkLogitsProcessor(
            adaptive=adaptive, key=key, gamma=gamma,
            vocab_size=vocab_size, prefix_length=prefix_length,
        )
    else:
        raise ValueError(f"Unknown watermark type: {wm_type}")


def run_single_ablation(
    model, tokenizer, prompts: list[str],
    watermark_type: str, params: dict, base_seed: int = 42,
):
    """运行单组消融实验。"""
    vocab_size = model.config.vocab_size
    processor = build_processor(watermark_type, vocab_size, params)
    device = model.device

    gen_kwargs = {
        "max_new_tokens": 200, "do_sample": True,
        "temperature": 0.7, "top_p": 0.9,
    }

    results_uw = []
    results_w = []
    texts_uw = []
    texts_w = []

    for i, prompt in enumerate(tqdm(prompts, desc=f"Ablation {watermark_type}", leave=False)):
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=512).to(device)
        prompt_len = encoded.input_ids.shape[1]

        # Unwatermarked
        torch.manual_seed(base_seed + i)
        out_uw = model.generate(**encoded, **gen_kwargs)
        text_uw = tokenizer.decode(out_uw[0], skip_special_tokens=True)
        tokens_uw = out_uw[0].cpu()
        texts_uw.append(text_uw)

        # Watermarked
        torch.manual_seed(base_seed + i)
        out_w = model.generate(
            **encoded, **gen_kwargs,
            logits_processor=LogitsProcessorList([processor]),
        )
        text_w = tokenizer.decode(out_w[0], skip_special_tokens=True)
        tokens_w = out_w[0].cpu()
        texts_w.append(text_w)

        # Detect
        detect_fn = morphmark_detect if watermark_type.startswith("morphmark") else kgw_detect
        r_uw = detect_fn(tokens_uw, params.get("key", 15485863), params.get("gamma", 0.5),
                        vocab_size, prefix_length=params.get("prefix_length", 1),
                        prompt_len=prompt_len)
        r_w = detect_fn(tokens_w, params.get("key", 15485863), params.get("gamma", 0.5),
                       vocab_size, prefix_length=params.get("prefix_length", 1),
                       prompt_len=prompt_len)

        results_uw.append(r_uw)
        results_w.append(r_w)

    z_uw = np.array([r["z_score"] for r in results_uw])
    z_w = np.array([r["z_score"] for r in results_w])

    ppl_uw_vals = compute_ppl_simple(model, tokenizer, texts_uw, device=device)
    ppl_w_vals = compute_ppl_simple(model, tokenizer, texts_w, device=device)
    ppl_uw_mean = float(np.mean([p for p in ppl_uw_vals if not np.isnan(p)]))
    ppl_w_mean = float(np.mean([p for p in ppl_w_vals if not np.isnan(p)]))

    tpr_1 = compute_tpr_at_fpr(z_w, z_uw, target_fpr=0.01)
    auc = compute_auc_roc(z_w, z_uw)

    result = {
        "params": params,
        "watermark_type": watermark_type,
        "n_samples": len(prompts),
        "tpr_at_fpr_0.01": tpr_1,
        "auc_roc": auc,
        "ppl_uw_mean": ppl_uw_mean,
        "ppl_w_mean": ppl_w_mean,
        "ppl_delta_pct": (ppl_w_mean - ppl_uw_mean) / ppl_uw_mean * 100,
        "z_uw_mean": float(z_uw.mean()),
        "z_uw_std": float(z_uw.std()),
        "z_w_mean": float(z_w.mean()),
        "z_w_std": float(z_w.std()),
    }
    return result


def run_ablation_suite(
    model, tokenizer, prompts: list[str],
    suite_name: str, base_params: dict,
    param_name: str, param_values: list,
):
    """扫参：固定其他参数，只变一个。"""
    print(f"\n{'='*60}")
    print(f"Ablation: {suite_name} — sweeping {param_name}")
    print(f"Values: {param_values}")
    print(f"{'='*60}")

    all_results = []
    for val in param_values:
        params = dict(base_params)
        params[param_name] = val
        label = f"{param_name}={val}"
        print(f"\n--- {label} ---")
        res = run_single_ablation(model, tokenizer, prompts,
                                  params["type"], params)
        res["label"] = label
        all_results.append(res)
        print(f"  TPR@1%={res['tpr_at_fpr_0.01']:.4f}, AUC={res['auc_roc']:.4f}, "
              f"PPL_w={res['ppl_w_mean']:.2f}, ΔPPL={res['ppl_delta_pct']:.1f}%")

    # Summary
    print(f"\n{'─'*60}")
    print(f"Summary: {suite_name}")
    print(f"{'Param':<15s} {'TPR@1%':>8s} {'AUC':>8s} {'PPL_w':>8s} {'ΔPPL%':>8s}")
    print("-" * 55)
    for res in all_results:
        p = res["params"][param_name]
        print(f"{str(p):<15s} {res['tpr_at_fpr_0.01']:8.4f} {res['auc_roc']:8.4f} "
              f"{res['ppl_w_mean']:8.2f} {res['ppl_delta_pct']:7.1f}%")

    return all_results


def save_results(results: list, path: Path):
    """增量保存结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--suite", type=str, default="all",
                       help="Which suite: gamma_kgw, gamma_mm, delta, p0, k, all")
    parser.add_argument("--output", type=str, default="outputs/ablation/results.json")
    args = parser.parse_args()
    out_path = Path(args.output)

    # Load existing results if any
    all_results = []
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
        completed_suites = {r.get("suite") for r in all_results}
        print(f"Loaded {len(all_results)} existing results from {out_path}")
    else:
        completed_suites = set()

    print("Loading dataset...")
    prompts = load_dataset_by_name("c4", n_samples=args.n_samples, seed=42)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

    # --- gamma sweep (KGW) ---
    suite_name = "kgw_gamma"
    if args.suite in ("gamma_kgw", "all") and suite_name not in completed_suites:
        base = {"type": "kgw", "key": 15485863, "gamma": 0.5, "delta": 2.0}
        results = run_ablation_suite(
            model, tokenizer, prompts,
            "KGW gamma", base, "gamma",
            [0.1, 0.3, 0.5, 0.7, 0.9],
        )
        for r in results:
            r["suite"] = suite_name
        all_results.extend(results)
        save_results(all_results, out_path)

    # --- gamma sweep (MorphMark_exp) ---
    suite_name = "mm_gamma"
    if args.suite in ("gamma_mm", "all") and suite_name not in completed_suites:
        base = {"type": "morphmark_exp", "key": 15485863, "gamma": 0.5,
                "p0": 0.15, "k": 1.30}
        results = run_ablation_suite(
            model, tokenizer, prompts,
            "MorphMark_exp gamma", base, "gamma",
            [0.1, 0.3, 0.5, 0.7, 0.9],
        )
        for r in results:
            r["suite"] = suite_name
        all_results.extend(results)
        save_results(all_results, out_path)

    # --- delta sweep (KGW) ---
    suite_name = "kgw_delta"
    if args.suite in ("delta", "all") and suite_name not in completed_suites:
        base = {"type": "kgw", "key": 15485863, "gamma": 0.5, "delta": 2.0}
        results = run_ablation_suite(
            model, tokenizer, prompts,
            "KGW delta", base, "delta",
            [0.5, 1.0, 2.0, 3.0, 5.0],
        )
        for r in results:
            r["suite"] = suite_name
        all_results.extend(results)
        save_results(all_results, out_path)

    # --- p0 sweep (MorphMark_exp) ---
    suite_name = "mm_p0"
    if args.suite in ("p0", "all") and suite_name not in completed_suites:
        base = {"type": "morphmark_exp", "key": 15485863, "gamma": 0.5,
                "p0": 0.15, "k": 1.30}
        results = run_ablation_suite(
            model, tokenizer, prompts,
            "MorphMark_exp p0", base, "p0",
            [0.0, 0.05, 0.10, 0.15, 0.25, 0.35],
        )
        for r in results:
            r["suite"] = suite_name
        all_results.extend(results)
        save_results(all_results, out_path)

    # --- k sweep (MorphMark_exp) ---
    suite_name = "mm_k"
    if args.suite in ("k", "all") and suite_name not in completed_suites:
        base = {"type": "morphmark_exp", "key": 15485863, "gamma": 0.5,
                "p0": 0.15, "k": 1.30}
        results = run_ablation_suite(
            model, tokenizer, prompts,
            "MorphMark_exp k", base, "k",
            [0.5, 1.0, 1.5, 2.0, 3.0],
        )
        for r in results:
            r["suite"] = suite_name
        all_results.extend(results)
        save_results(all_results, out_path)

    print(f"\nAll ablation results saved to {out_path}")
    print(f"Total: {len(all_results)} runs")


if __name__ == "__main__":
    main()
