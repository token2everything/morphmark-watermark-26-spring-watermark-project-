"""检测模块测试"""

import sys
sys.path.insert(0, '.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from scipy.stats import norm

from src.detection.zscore import compute_z_score, z_score_to_pvalue, detect_single
from src.detection.ewd import ewd_detect, compute_token_entropies
from src.watermarking.kgw import KGWLogitsProcessor

MODEL_NAME = "facebook/opt-1.3b"
KEY = 15485863
GAMMA = 0.5


def test_zscore_basics():
    """测试 z-score 基本计算。"""
    # 完全符合期望：z = 0
    z = compute_z_score(50, 100, 0.5)
    assert abs(z) < 1e-6, f"z should be 0, got {z}"

    # 全部是绿的：z > 0
    z = compute_z_score(100, 100, 0.5)
    assert z > 5, f"z should be large, got {z}"

    # p-value
    p = z_score_to_pvalue(4.0)
    expected = norm.sf(4.0)
    assert abs(p - expected) < 1e-6
    print(f"z=4.0 → p={p:.6f} (expected={expected:.6f})")
    print("✓ test_zscore_basics passed")


def test_detect_single():
    """测试单序列检测。"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = 50272  # OPT vocab

    text = "The capital of France is Paris, a beautiful city with rich history."
    tokens = tokenizer(text, return_tensors="pt",
                      add_special_tokens=False)["input_ids"][0]

    r = detect_single(tokens, KEY, GAMMA, vocab_size)
    assert "z_score" in r
    assert "p_value" in r
    assert "green_count" in r
    assert r["total_scored"] == len(tokens) - 1
    print(f"z={r['z_score']:.3f}, green={r['green_count']}/{r['total_scored']}")
    print("✓ test_detect_single passed")


def test_ewd():
    """测试 EWD 熵加权检测。"""
    print("Loading model for EWD...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = model.config.vocab_size

    # 生成水印文本
    processor = KGWLogitsProcessor(key=KEY, gamma=GAMMA, delta=2.0,
                                   vocab_size=vocab_size)
    prompt = "The capital of France is"
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = enc.input_ids.shape[1]

    torch.manual_seed(42)
    out = model.generate(**enc, max_new_tokens=30, do_sample=True,
                         temperature=0.7,
                         logits_processor=LogitsProcessorList([processor]))
    tokens = out[0].cpu()

    # EWD 检测
    r = ewd_detect(model, tokens, KEY, GAMMA, vocab_size,
                   prompt_len=prompt_len)

    print(f"Standard z: {r['z_score_standard']:.3f}")
    print(f"EWD z:      {r['z_score_ewd']:.3f}")
    print(f"Mean entropy: {r['mean_entropy']:.4f}")
    print(f"Green: {r['green_count']}/{r['total_scored']}")

    # EWD z-score 应该 >= 标准 z-score（更敏感的检测）
    # 注意：EWD 的 z 可能比标准 z 高或低，取决于熵分布
    # 对于 watermarked text，EWD 通常提供更强的信号
    assert r["z_score_ewd"] > 0, "EWD should detect the watermark"

    # 测试熵计算
    entropies = compute_token_entropies(model, tokens)
    assert len(entropies) == len(tokens)
    for h in entropies:
        assert h >= 0, f"熵应为非负: {h}"
        assert h < 12, f"熵应 < ln(50272) ≈ 10.8: {h}"
    print(f"Entropies: min={min(entropies):.3f}, max={max(entropies):.3f}")
    print("✓ test_ewd passed")


if __name__ == '__main__':
    test_zscore_basics()
    test_detect_single()
    test_ewd()
    print("\nAll detection tests passed!")
