"""全流程评测：生成 → 检测 → 指标

对每个 prompt：
1. 生成无水印文本（建立零分布）
2. 生成水印文本
3. 检测并计算 z-score
4. 汇总计算 TPR@FPR, Best F1, AUC
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, ExperimentConfig
from src.watermarking.kgw import KGWLogitsProcessor, kgw_detect
from src.watermarking.morphmark import MorphMarkLogitsProcessor, morphmark_detect
from src.watermarking.adaptive import create_adaptive_strength
from src.data.loader import load_dataset_by_name
from src.evaluation.metrics import compute_tpr_at_fpr, compute_best_f1, compute_auc_roc
from src.evaluation.quality import compute_ppl_simple


def build_processor(config: ExperimentConfig, vocab_size: int):
    wm = config.watermark
    if wm.type == "kgw":
        return KGWLogitsProcessor(
            key=wm.key, gamma=wm.gamma, delta=wm.delta,
            vocab_size=vocab_size, prefix_length=wm.prefix_length,
        )
    elif wm.type.startswith("morphmark"):
        adaptive = create_adaptive_strength(wm.type, p0=wm.p0, k=wm.k,
                                            epsilon=wm.epsilon)
        return MorphMarkLogitsProcessor(
            adaptive=adaptive, key=wm.key, gamma=wm.gamma,
            vocab_size=vocab_size, prefix_length=wm.prefix_length,
        )
    else:
        raise ValueError(f"Unknown watermark type: {wm.type}")


def run_evaluation(config_path: str, n_samples: int | None = None):
    config = load_config(config_path)
    dataset_name = config.data.dataset
    n = n_samples or config.data.num_samples

    print(f"Loading dataset: {dataset_name} ({n} samples)")
    prompts = load_dataset_by_name(dataset_name, n_samples=n, seed=config.data.seed)

    print(f"Loading model: {config.model.name}")
    dtype = getattr(torch, config.model.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name, torch_dtype=dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    vocab_size = model.config.vocab_size

    processor = build_processor(config, vocab_size)
    wm = config.watermark

    gen_kwargs = {
        "max_new_tokens": config.generation.max_new_tokens,
        "do_sample": config.generation.do_sample,
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
        "no_repeat_ngram_size": config.generation.no_repeat_ngram_size,
    }
    if config.generation.top_k > 0:
        gen_kwargs["top_k"] = config.generation.top_k

    results_uw = []
    results_w = []

    for i, prompt in enumerate(tqdm(prompts, desc="Evaluating")):
        encoded = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=512).to(model.device)
        prompt_len = encoded.input_ids.shape[1]

        # 无水印
        torch.manual_seed(config.data.seed + i)
        out_uw = model.generate(**encoded, **gen_kwargs)
        text_uw = tokenizer.decode(out_uw[0], skip_special_tokens=True)
        tokens_uw = out_uw[0].cpu()

        # 水印
        torch.manual_seed(config.data.seed + i)
        out_w = model.generate(
            **encoded,
            **gen_kwargs,
            logits_processor=LogitsProcessorList([processor]),
        )
        text_w = tokenizer.decode(out_w[0], skip_special_tokens=True)
        tokens_w = out_w[0].cpu()

        # 检测
        r_uw = _detect(tokens_uw, wm, vocab_size, prompt_len)
        r_w = _detect(tokens_w, wm, vocab_size, prompt_len)

        results_uw.append({
            "prompt": prompt[:100],
            "text": text_uw,
            "z_score": r_uw["z_score"],
            "green_count": r_uw["green_count"],
            "total_scored": r_uw["total_scored"],
        })
        results_w.append({
            "prompt": prompt[:100],
            "text": text_w,
            "z_score": r_w["z_score"],
            "green_count": r_w["green_count"],
            "total_scored": r_w["total_scored"],
        })

    z_uw = np.array([r["z_score"] for r in results_uw])
    z_w = np.array([r["z_score"] for r in results_w])

    # PPL 计算
    texts_uw = [r["text"] for r in results_uw]
    texts_w = [r["text"] for r in results_w]
    ppl_uw = compute_ppl_simple(model, tokenizer, texts_uw, device=config.model.device)
    ppl_w = compute_ppl_simple(model, tokenizer, texts_w, device=config.model.device)
    ppl_uw_mean = float(np.mean([p for p in ppl_uw if not np.isnan(p)]))
    ppl_w_mean = float(np.mean([p for p in ppl_w if not np.isnan(p)]))

    tpr_1 = compute_tpr_at_fpr(z_w, z_uw, target_fpr=0.01)
    f1_info = compute_best_f1(z_w, z_uw)
    auc = compute_auc_roc(z_w, z_uw)

    print(f"\n{'='*60}")
    print(f"Results: {config.watermark.type} on {dataset_name} ({n} samples)")
    print(f"{'='*60}")
    print(f"Z-score (unwatermarked): mean={z_uw.mean():.3f}, std={z_uw.std():.3f}")
    print(f"Z-score (watermarked):   mean={z_w.mean():.3f}, std={z_w.std():.3f}")
    print(f"PPL (unwatermarked):     mean={ppl_uw_mean:.2f}")
    print(f"PPL (watermarked):       mean={ppl_w_mean:.2f}")
    print(f"Median abs gap: {np.median(z_w - z_uw):.3f}")
    print(f"---")
    print(f"TPR@FPR=1%:  {tpr_1:.4f}")
    print(f"Best F1:     {f1_info['best_f1']:.4f} (thresh={f1_info['best_threshold']:.2f})")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Green ratio (uw): {np.mean([r['green_count']/max(r['total_scored'],1) for r in results_uw]):.3f}")
    print(f"Green ratio (w):  {np.mean([r['green_count']/max(r['total_scored'],1) for r in results_w]):.3f}")

    # 保存
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{config.watermark.type}_{dataset_name}_{n}.json"

    output_data = {
        "config": config_path,
        "watermark_type": config.watermark.type,
        "dataset": dataset_name,
        "n_samples": n,
        "tpr_at_fpr_0.01": tpr_1,
        "best_f1": f1_info["best_f1"],
        "best_f1_threshold": f1_info["best_threshold"],
        "auc_roc": auc,
        "ppl_uw_mean": ppl_uw_mean,
        "ppl_w_mean": ppl_w_mean,
        "z_uw_mean": float(z_uw.mean()),
        "z_uw_std": float(z_uw.std()),
        "z_w_mean": float(z_w.mean()),
        "z_w_std": float(z_w.std()),
        "green_ratio_uw": float(np.mean([r['green_count']/max(r['total_scored'],1) for r in results_uw])),
        "green_ratio_w": float(np.mean([r['green_count']/max(r['total_scored'],1) for r in results_w])),
        "results_uw": results_uw,
        "results_w": results_w,
    }
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nSaved to {out_file}")

    return output_data


def _detect(tokens, wm_config, vocab_size, prompt_len):
    if wm_config.type.startswith("morphmark"):
        return morphmark_detect(tokens, wm_config.key, wm_config.gamma,
                               vocab_size, prefix_length=wm_config.prefix_length,
                               prompt_len=prompt_len)
    else:
        return kgw_detect(tokens, wm_config.key, wm_config.gamma,
                         vocab_size, prefix_length=wm_config.prefix_length,
                         prompt_len=prompt_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(args.config, args.n_samples)


if __name__ == "__main__":
    main()
