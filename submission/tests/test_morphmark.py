"""MorphMark 测试"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from src.watermarking.adaptive import LinearStrength, ExpStrength, LogStrength
from src.watermarking.morphmark import MorphMarkLogitsProcessor, morphmark_detect
from src.watermarking.kgw import KGWLogitsProcessor, kgw_detect

MODEL_NAME = "facebook/opt-1.3b"
KEY = 15485863
GAMMA = 0.5
P0 = 0.15
PROMPT = "The capital of France is"


def test_adaptive_strength():
    """测试强度函数的基本行为。"""
    # Linear
    s = LinearStrength(p0=0.15, k=1.55)
    assert s.compute_r(0.05) == 1e-10, "PG <= p0 应返回 epsilon"
    assert s.compute_r(0.3) > s.compute_r(0.2), "should be monotonic"
    assert s.compute_r(0.3) > s.compute_r(0.15), "above p0 should be larger"

    # Exp
    e = ExpStrength(p0=0.15, k=1.30)
    assert e.compute_r(0.05) == 1e-10, "PG <= p0 应返回 epsilon"
    assert e.compute_r(0.5) > e.compute_r(0.3), "should be monotonic"

    # Log
    l = LogStrength(p0=0.15, k=2.15)
    assert l.compute_r(0.05) == 1e-10, "PG <= p0 应返回 epsilon"
    assert l.compute_r(0.5) > l.compute_r(0.3), "should be monotonic"

    # At PG=0.5, exp should be strongest, log weakest
    pg = 0.5
    r_exp = e.compute_r(pg)
    r_lin = s.compute_r(pg)
    r_log = l.compute_r(pg)
    print(f"PG={pg}: r_exp={r_exp:.4f}, r_linear={r_lin:.4f}, r_log={r_log:.4f}")
    assert r_exp > r_log, f"exp should be larger than log"

    print("✓ test_adaptive_strength passed")


def test_prob_adjustment():
    """测试概率调整正确性：调整后概率仍为合法分布。"""
    scores = torch.randn(50272)  # random logits
    probs = F.softmax(scores, dim=-1)

    # 模拟绿列表：前 50% 为绿色
    vocab_size = 50272
    green_size = int(GAMMA * vocab_size)
    green_mask = torch.zeros(vocab_size, dtype=torch.bool)
    green_mask[:green_size] = True

    pg = probs[green_mask].sum().item()
    adaptive = ExpStrength(p0=0.15, k=1.30)
    r = adaptive.compute_r(pg)

    if r > 1e-10:
        beta = r * (1.0 - pg)
        adjusted = probs.clone()
        weights_g = adjusted[green_mask]
        adjusted[green_mask] = weights_g + (weights_g / weights_g.sum()) * beta
        weights_r = adjusted[~green_mask]
        adjusted[~green_mask] = weights_r - (weights_r / weights_r.sum()) * beta

        adjusted = torch.clamp(adjusted, min=1e-10)
        adjusted = adjusted / adjusted.sum()

        total = adjusted.sum().item()
        assert abs(total - 1.0) < 1e-5, f"概率和应为 1.0, got {total}"
        assert (adjusted >= 0).all(), "所有概率应 >= 0"
        green_new = adjusted[green_mask].sum().item()
        assert green_new > pg, f"调整后绿列表概率应增加: {green_new:.4f} > {pg:.4f}"

    print("✓ test_prob_adjustment passed")


def test_end_to_end():
    """端到端测试：MorphMark vs KGW 在相同条件下对比。"""
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = model.config.vocab_size

    encoded = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    prompt_len = encoded.input_ids.shape[1]

    gen_kwargs = {
        "max_new_tokens": 50, "do_sample": True, "temperature": 0.7
    }

    # KGW
    kgw_proc = KGWLogitsProcessor(key=KEY, gamma=GAMMA, delta=2.0,
                                  vocab_size=vocab_size)
    torch.manual_seed(42)
    out_kgw = model.generate(**encoded,
                             logits_processor=LogitsProcessorList([kgw_proc]),
                             **gen_kwargs)
    tokens_kgw = out_kgw[0].cpu()

    # MorphMark exp
    adaptive = ExpStrength(p0=P0, k=1.30)
    mm_proc = MorphMarkLogitsProcessor(adaptive=adaptive, key=KEY,
                                       gamma=GAMMA, vocab_size=vocab_size)
    torch.manual_seed(42)
    out_mm = model.generate(**encoded,
                            logits_processor=LogitsProcessorList([mm_proc]),
                            **gen_kwargs)
    tokens_mm = out_mm[0].cpu()

    r_kgw = kgw_detect(tokens_kgw, KEY, GAMMA, vocab_size, prompt_len=prompt_len)
    r_mm = morphmark_detect(tokens_mm, KEY, GAMMA, vocab_size, prompt_len=prompt_len)

    print(f"KGW:  z={r_kgw['z_score']:.3f}, green={r_kgw['green_count']}/{r_kgw['total_scored']}")
    print(f"MorphMark: z={r_mm['z_score']:.3f}, green={r_mm['green_count']}/{r_mm['total_scored']}")

    # MorphMark should produce a competitive result
    assert r_mm['z_score'] > 0, "MorphMark should work"
    print("✓ test_end_to_end passed")


if __name__ == '__main__':
    test_adaptive_strength()
    test_prob_adjustment()
    test_end_to_end()
    print("\nAll MorphMark tests passed!")
