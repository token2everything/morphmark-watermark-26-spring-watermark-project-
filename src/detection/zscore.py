"""标准 z-score 水印检测"""

import torch
from scipy.stats import norm
from typing import Union

from ..watermarking.hash_split import compute_green_list


def compute_z_score(green_count: int, total_count: int, gamma: float) -> float:
    """计算 z-score 统计量。

    z = (|S|_G - γ·|T|) / sqrt(|T|·γ·(1-γ))
    """
    from math import sqrt
    expected = gamma * total_count
    denom = sqrt(total_count * gamma * (1 - gamma))
    if denom == 0:
        return 0.0
    return (green_count - expected) / denom


def z_score_to_pvalue(z: float) -> float:
    """单侧检验：z-score 转 p-value。"""
    return float(norm.sf(z))


def detect_single(
    token_ids: torch.Tensor,
    key: int,
    gamma: float,
    vocab_size: int,
    prefix_length: int = 1,
    prompt_len: int = 0,
) -> dict:
    """检测单个序列中的水印。

    Returns:
        dict with z_score, green_count, total_scored, p_value, is_watermarked
    """
    start = max(prefix_length, prompt_len)
    num_scored = len(token_ids) - start
    if num_scored < 1:
        return {"z_score": 0.0, "green_count": 0, "total_scored": 0,
                "p_value": 1.0, "is_watermarked": False}

    green_count = 0
    for i in range(start, len(token_ids)):
        prev_token = token_ids[i - 1].item()
        curr_token = token_ids[i].item()
        green_ids = compute_green_list(prev_token, key, gamma, vocab_size)
        if curr_token in green_ids:
            green_count += 1

    z = compute_z_score(green_count, num_scored, gamma)
    p_val = z_score_to_pvalue(z)

    return {
        "z_score": z,
        "green_count": green_count,
        "total_scored": num_scored,
        "p_value": p_val,
        "is_watermarked": z > 4.0,
    }


def detect_batch(
    token_ids_list: list[torch.Tensor],
    prompt_lens: list[int],
    key: int,
    gamma: float,
    vocab_size: int,
    prefix_length: int = 1,
) -> list[dict]:
    """批量检测。"""
    results = []
    for token_ids, plen in zip(token_ids_list, prompt_lens):
        r = detect_single(token_ids, key, gamma, vocab_size,
                          prefix_length=prefix_length, prompt_len=plen)
        results.append(r)
    return results
