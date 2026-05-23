"""EWD 熵加权水印检测 (Lu et al. 2024)

不同于标准 z-score 给每个 token 等权重，EWD 根据每步生成时的
模型不确定度（熵）加权：高熵 token 携带更多水印信号。

加权 z-score:
  z_ewd = (sum_{i∈G} H_i - γ·sum_{i∈all} H_i) / sqrt(γ·(1-γ)·sum_{i∈all} H_i²)
"""

import torch
import torch.nn.functional as F
from math import sqrt
from typing import List

from ..watermarking.hash_split import compute_green_list


def compute_token_entropies(
    model,
    input_ids: torch.Tensor,
) -> List[float]:
    """对序列中的每个 token 位置计算生成时的熵。

    运行一次完整序列的 forward pass，从 logits 提取每位置的
    概率分布并计算熵：H_i = -sum_j p_j(i) * log(p_j(i))

    注意：logits[t] 是对 token[t] 的预测分布（基于 tokens[0:t]）。

    Returns:
        每位置的熵值列表，长度 = seq_len
    """
    device = next(model.parameters()).device
    if input_ids.device != device:
        input_ids = input_ids.to(device)

    model.eval()
    with torch.no_grad():
        output = model(torch.unsqueeze(input_ids, 0))
        logits = output.logits[0]  # (seq_len, vocab_size)

    entropies = []
    for t in range(logits.shape[0]):
        probs = F.softmax(logits[t].float(), dim=-1)  # fp32 for stability
        log_probs = torch.log(probs + 1e-10)
        h = -(probs * log_probs).sum().item()
        entropies.append(h)

    model.train()
    return entropies


def ewd_detect(
    model,
    token_ids: torch.Tensor,
    key: int,
    gamma: float,
    vocab_size: int,
    prefix_length: int = 1,
    prompt_len: int = 0,
) -> dict:
    """EWD 熵加权检测。

    Args:
        model: 原始生成模型（用于计算熵）
        token_ids: 完整序列 (prompt + generated)
        key: 私钥
        gamma: 绿列表比例
        vocab_size: 词表大小
        prefix_length: hash 上下文窗口
        prompt_len: prompt 长度（不参与评分）

    Returns:
        dict with z_score_ewd, z_score_standard, green_count, ...
    """
    start = max(prefix_length, prompt_len)
    num_scored = len(token_ids) - start
    if num_scored < 1:
        return {"z_score_ewd": 0.0, "z_score_standard": 0.0,
                "green_count": 0, "total_scored": 0,
                "is_watermarked": False}

    # 计算每位置的熵
    entropies = compute_token_entropies(model, token_ids)

    green_count = 0
    weighted_green = 0.0
    weighted_total = 0.0
    weighted_sq_total = 0.0

    for i in range(start, len(token_ids)):
        prev_token = token_ids[i - 1].item()
        curr_token = token_ids[i].item()
        green_ids = compute_green_list(prev_token, key, gamma, vocab_size)

        h = max(entropies[i], 1e-10)
        weighted_total += h
        weighted_sq_total += h * h

        if curr_token in green_ids:
            green_count += 1
            weighted_green += h

    # 标准 z-score
    z_standard = _compute_z_score(green_count, num_scored, gamma)

    # EWD 加权 z-score
    expected_weighted = gamma * weighted_total
    denom = sqrt(gamma * (1 - gamma) * weighted_sq_total)
    if denom == 0:
        z_ewd = 0.0
    else:
        z_ewd = (weighted_green - expected_weighted) / denom

    return {
        "z_score_ewd": z_ewd,
        "z_score_standard": z_standard,
        "green_count": green_count,
        "total_scored": num_scored,
        "is_watermarked": z_ewd > 4.0,
        "mean_entropy": float(weighted_total / num_scored) if num_scored > 0 else 0.0,
    }


def _compute_z_score(green_count: int, total: int, gamma: float) -> float:
    expected = gamma * total
    denom = sqrt(total * gamma * (1 - gamma))
    if denom == 0:
        return 0.0
    return (green_count - expected) / denom
