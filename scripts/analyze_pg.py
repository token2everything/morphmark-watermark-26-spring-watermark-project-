"""PG 分布分析：收集 MorphMark 每 token 的 PG 和 r 值，生成可视化"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.watermarking.morphmark import MorphMarkLogitsProcessor
from src.watermarking.adaptive import ExpStrength, LinearStrength, LogStrength
from src.data.loader import load_dataset_by_name


ADAPTIVE_CLASSES = {
    "morphmark_exp": lambda: ExpStrength(p0=0.15, k=1.30),
    "morphmark_linear": lambda: LinearStrength(p0=0.15, k=1.55),
    "morphmark_log": lambda: LogStrength(p0=0.15, k=2.15),
}

KEY = 15485863
GAMMA = 0.5


def run_pg_collection(method: str, prompts: list[str], model, tokenizer, vocab_size: int):
    """运行一次生成并收集每 token 的 PG/r 值。"""
    adaptive = ADAPTIVE_CLASSES[method]()
    processor = MorphMarkLogitsProcessor(
        adaptive=adaptive, key=KEY, gamma=GAMMA,
        vocab_size=vocab_size, prefix_length=1, record_pg=True,
    )

    gen_kwargs = {
        "max_new_tokens": 200, "do_sample": True,
        "temperature": 0.7, "top_p": 0.9,
    }

    all_records = []
    for i, prompt in enumerate(tqdm(prompts, desc=f"PG collection {method}")):
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=512).to(model.device)
        processor.pg_records = []  # reset

        torch.manual_seed(42 + i)
        model.generate(
            **encoded, **gen_kwargs,
            logits_processor=LogitsProcessorList([processor]),
        )
        all_records.extend(processor.pg_records)

    return all_records


def analyze(records: list[dict], method: str):
    """分析 PG 记录并打印统计信息。"""
    pgs = np.array([r["pg"] for r in records])
    rs = np.array([r["r"] for r in records])
    positions = np.array([r["position"] for r in records])

    print(f"\n{'='*60}")
    print(f"PG Analysis: {method} ({len(records)} tokens)")
    print(f"{'='*60}")
    print(f"PG: mean={pgs.mean():.4f}, median={np.median(pgs):.4f}, "
          f"std={pgs.std():.4f}, min={pgs.min():.4f}, max={pgs.max():.4f}")

    # PG 分布直方图 bins
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
    print(f"\nPG histogram:")
    for i in range(len(bins)-1):
        count = np.sum((pgs >= bins[i]) & (pgs < bins[i+1]))
        pct = count / len(pgs) * 100
        bar = "█" * int(pct)
        print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f}): {count:5d} ({pct:5.1f}%) {bar}")

    # r 统计
    rs_active = rs[rs > 1e-9]
    print(f"\nr: mean={rs.mean():.4f}, median={np.median(rs):.4f}, "
          f"active_ratio={len(rs_active)/len(rs)*100:.1f}%")
    if len(rs_active) > 0:
        print(f"r (active only): mean={rs_active.mean():.4f}, "
              f"min={rs_active.min():.4f}, max={rs_active.max():.4f}")

    # PG vs position: 按位置分桶
    pos_bins = [(0, 50), (50, 100), (100, 150), (150, 200)]
    print(f"\nPG by position range:")
    for lo, hi in pos_bins:
        mask = (positions >= lo) & (positions < hi)
        if mask.sum() > 0:
            print(f"  pos [{lo:3d}-{hi:3d}): PG={pgs[mask].mean():.4f}, "
                  f"r={rs[mask].mean():.4f}, count={mask.sum()}")

    # PG vs r scatter summary
    pg_bins_r = [0, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    print(f"\nAverage r by PG range:")
    for i in range(len(pg_bins_r)-1):
        mask = (pgs >= pg_bins_r[i]) & (pgs < pg_bins_r[i+1])
        if mask.sum() > 0:
            print(f"  PG [{pg_bins_r[i]:.2f}-{pg_bins_r[i+1]:.2f}): "
                  f"r={rs[mask].mean():.4f}, count={mask.sum()}")

    return {"pg": pgs.tolist(), "r": rs.tolist(), "position": positions.tolist()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--methods", type=str, default="morphmark_exp,morphmark_linear,morphmark_log")
    args = parser.parse_args()

    prompts = load_dataset_by_name("c4", n_samples=args.n_samples, seed=42)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    vocab_size = model.config.vocab_size

    all_data = {}
    for method in args.methods.split(","):
        records = run_pg_collection(method, prompts, model, tokenizer, vocab_size)
        data = analyze(records, method)
        all_data[method] = data

    # Save
    out_path = Path("outputs/figures/pg_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved PG data to {out_path}")


if __name__ == "__main__":
    main()
