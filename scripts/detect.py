"""对文本文件检测水印"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.watermarking.kgw import kgw_detect
from src.watermarking.morphmark import morphmark_detect
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                       help="JSONL file with generated texts")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(config.model.name)
    vocab_size = hf_config.vocab_size
    wm = config.watermark

    with open(args.input) as f:
        lines = [json.loads(line) for line in f]

    results = []
    for item in lines:
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        tokens = tokenizer(item["text"], return_tensors="pt",
                          add_special_tokens=False)["input_ids"][0]

        if wm.type.startswith("morphmark"):
            r = morphmark_detect(tokens, wm.key, config.detection.gamma, vocab_size,
                                prefix_length=wm.prefix_length,
                                prompt_len=item.get("prompt_len", 0))
        else:
            r = kgw_detect(tokens, wm.key, config.detection.gamma, vocab_size,
                          prefix_length=wm.prefix_length,
                          prompt_len=item.get("prompt_len", 0))
        r["text"] = item["text"][:100]
        results.append(r)

    z_scores = [r["z_score"] for r in results]
    print(f"\nSamples: {len(results)}")
    print(f"Z-score mean: {np.mean(z_scores):.3f}")
    print(f"Z-score std:  {np.std(z_scores):.3f}")
    print(f"Z-score min:  {np.min(z_scores):.3f}")
    print(f"Z-score max:  {np.max(z_scores):.3f}")
    detected = sum(r["is_watermarked"] for r in results)
    print(f"Watermarked (z>4): {detected}/{len(results)} ({100*detected/len(results):.1f}%)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps({k: v for k, v in r.items()
                                   if k != "green_flags"}) + "\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
