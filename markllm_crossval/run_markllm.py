"""用 MarkLLM 官方实现运行 KGW，生成对比结果"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset


def run_markllm_eval(method: str, config_path: str, n_samples: int = 50):
    """Run MarkLLM evaluation for a given method."""
    print(f"\n{'='*60}")
    print(f"MarkLLM: {method}")
    print(f"{'='*60}")

    ds = load_dataset("allenai/c4", "realnewslike", split="validation", streaming=True)
    ds = ds.shuffle(seed=42)
    prompts = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        words = item["text"].split()
        prompts.append(" ".join(words[:60]))

    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    hf_config = AutoConfig.from_pretrained("facebook/opt-1.3b")

    transformers_config = TransformersConfig(
        model=model, tokenizer=tokenizer,
        vocab_size=hf_config.vocab_size,
        device="cuda",
        max_new_tokens=200, do_sample=True,
        temperature=0.7, top_p=0.9,
        no_repeat_ngram_size=4,
    )

    watermark = AutoWatermark.load(
        algorithm_name=method,
        algorithm_config=config_path,
        transformers_config=transformers_config,
    )

    results_uw = []
    results_w = []

    for i, prompt in enumerate(tqdm(prompts, desc=f"MarkLLM {method}")):
        # Unwatermarked
        torch.manual_seed(42 + i)
        text_uw = watermark.generate_unwatermarked_text(prompt)

        # Watermarked
        torch.manual_seed(42 + i)
        text_w = watermark.generate_watermarked_text(prompt)

        # Detect
        det_uw = watermark.detect_watermark(text_uw)
        det_w = watermark.detect_watermark(text_w)
        z_uw = det_uw["score"] if isinstance(det_uw, dict) else det_uw[1]
        z_w = det_w["score"] if isinstance(det_w, dict) else det_w[1]

        results_uw.append({"text": text_uw, "z_score": z_uw})
        results_w.append({"text": text_w, "z_score": z_w})

    z_uw = np.array([r["z_score"] for r in results_uw])
    z_w = np.array([r["z_score"] for r in results_w])

    # Compute TPR@1%
    n_uw = len(z_uw)
    k = max(1, int(np.ceil(0.01 * n_uw)))
    threshold = np.partition(z_uw, -k)[-k]
    tpr = np.mean(z_w > threshold)

    # AUC
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.ones(len(z_w)), np.zeros(len(z_uw))])
    scores = np.concatenate([z_w, z_uw])
    auc = roc_auc_score(labels, scores)

    out = {
        "method": f"MarkLLM_{method}",
        "n_samples": n_samples,
        "z_uw_mean": float(z_uw.mean()),
        "z_uw_std": float(z_uw.std()),
        "z_w_mean": float(z_w.mean()),
        "z_w_std": float(z_w.std()),
        "tpr_at_fpr_0.01": float(tpr),
        "auc_roc": float(auc),
    }
    print(f"Z (uw): {z_uw.mean():.3f} ± {z_uw.std():.3f}")
    print(f"Z (w):  {z_w.mean():.3f} ± {z_w.std():.3f}")
    print(f"TPR@1%: {tpr:.4f}")
    print(f"AUC:    {auc:.4f}")

    out_dir = Path("outputs/markllm_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"markllm_{method}_c4_{n_samples}.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {out_file}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()
    run_markllm_eval(args.method, args.config, args.n_samples)
