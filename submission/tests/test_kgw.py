"""KGW 端到端测试"""

import sys
sys.path.insert(0, '.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from src.watermarking.kgw import KGWLogitsProcessor, kgw_detect

MODEL_NAME = "facebook/opt-1.3b"
KEY = 15485863
GAMMA = 0.5
DELTA = 2.0
PROMPT = "The capital of France is"
PROMPT_TOKENS = 5


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def test_watermark_detection():
    """水印文本应有显著高于无水印文本的 z-score。"""
    print("Loading model...")
    model, tokenizer = load_model()
    vocab_size = model.config.vocab_size

    processor = KGWLogitsProcessor(
        key=KEY, gamma=GAMMA, delta=DELTA, vocab_size=vocab_size
    )
    encoded = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    prompt_len = encoded.input_ids.shape[1]

    # 生成无水印文本
    torch.manual_seed(42)
    out_uw = model.generate(**encoded, max_new_tokens=50, do_sample=True,
                            temperature=0.7, logits_processor=None)
    tokens_uw = out_uw[0].cpu()

    # 生成水印文本
    torch.manual_seed(42)
    out_w = model.generate(**encoded, max_new_tokens=50, do_sample=True,
                           temperature=0.7,
                           logits_processor=LogitsProcessorList([processor]))
    tokens_w = out_w[0].cpu()

    # 检测
    r_uw = kgw_detect(tokens_uw, KEY, GAMMA, vocab_size, prompt_len=prompt_len)
    r_w = kgw_detect(tokens_w, KEY, GAMMA, vocab_size, prompt_len=prompt_len)

    print(f"Unwatermarked: z={r_uw['z_score']:.3f}, green={r_uw['green_count']}/{r_uw['total_scored']}")
    print(f"Watermarked:   z={r_w['z_score']:.3f}, green={r_w['green_count']}/{r_w['total_scored']}")

    assert r_w['z_score'] > r_uw['z_score'], "水印文本应有更高 z-score"
    assert r_w['green_count'] > r_w['total_scored'] * GAMMA, "水印文本的绿 token 比例应高于 gamma"
    print("✓ test_watermark_detection passed")


def test_detection_consistency():
    """同一文本检测两次应返回相同结果。"""
    print("Loading model...")
    model, tokenizer = load_model()
    vocab_size = model.config.vocab_size

    processor = KGWLogitsProcessor(
        key=KEY, gamma=GAMMA, delta=DELTA, vocab_size=vocab_size
    )
    encoded = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    prompt_len = encoded.input_ids.shape[1]

    torch.manual_seed(42)
    out_w = model.generate(**encoded, max_new_tokens=30, do_sample=True,
                           temperature=0.7,
                           logits_processor=LogitsProcessorList([processor]))
    tokens_w = out_w[0].cpu()

    r1 = kgw_detect(tokens_w, KEY, GAMMA, vocab_size, prompt_len=prompt_len)
    r2 = kgw_detect(tokens_w, KEY, GAMMA, vocab_size, prompt_len=prompt_len)

    assert r1['z_score'] == r2['z_score'], "两次检测结果必须相同"
    assert r1['green_count'] == r2['green_count'], "绿 token 数必须相同"
    print("✓ test_detection_consistency passed")


if __name__ == '__main__':
    test_watermark_detection()
    test_detection_consistency()
    print("\nAll KGW tests passed!")
