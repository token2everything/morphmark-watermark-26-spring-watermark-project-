"""按配置生成水印文本"""

import argparse
import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.watermarking.kgw import KGWLogitsProcessor
from src.watermarking.morphmark import MorphMarkLogitsProcessor
from src.watermarking.adaptive import create_adaptive_strength


def load_model_and_tokenizer(config):
    dtype = getattr(torch, config.model.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name, torch_dtype=dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    return model, tokenizer


def build_processor(config, vocab_size):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt; if not provided, use from dataset")
    parser.add_argument("--output", type=str, default="outputs/generated/result.jsonl")
    parser.add_argument("--watermark", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model, tokenizer = load_model_and_tokenizer(config)
    processor = build_processor(config, len(tokenizer))

    gen_kwargs = {
        "max_new_tokens": config.generation.max_new_tokens,
        "do_sample": config.generation.do_sample,
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
        "no_repeat_ngram_size": config.generation.no_repeat_ngram_size,
    }
    if config.generation.top_k > 0:
        gen_kwargs["top_k"] = config.generation.top_k

    results = []
    prompt = args.prompt or "The capital of France is"

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = encoded.input_ids.shape[1]

    torch.manual_seed(42)
    if args.watermark:
        gen_kwargs["logits_processor"] = LogitsProcessorList([processor])

    output = model.generate(**encoded, **gen_kwargs)
    tokens = output[0].cpu()
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    result = {
        "prompt": prompt,
        "text": text,
        "prompt_len": prompt_len,
        "total_len": len(tokens),
        "watermarked": args.watermark,
    }
    results.append(result)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(results)} sample(s), saved to {args.output}")


if __name__ == "__main__":
    main()
