"""MorphMark 自适应水印算法 (Wang et al. 2025)

与 KGW 的关键区别：不是加固定 delta，而是根据绿列表累计概率 PG
自适应调整水印强度 r = φ(PG)，概率层面重分布。

公式 (6):
  p̂_i = p_i + (p_i/P_G) · r · (1-P_G)    for green
  p̂_i = p_i - (p_i/(1-P_G)) · r · (1-P_G)  for red
"""

import math
from math import sqrt
from typing import List

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from .hash_split import compute_green_list
from .adaptive import AdaptiveStrength, create_adaptive_strength

FLOOR = 1e-10


class MorphMarkLogitsProcessor(LogitsProcessor):
    """MorphMark 水印的 HuggingFace LogitsProcessor。

    每步生成时：
    1. 软最大化 scores 得到 probs
    2. 计算 PG = sum(probs[green])
    3. 自适应强度 r = adaptive.compute_r(PG)
    4. 按公式(6) 在概率层面调整
    5. clamp + renormalize + log → 返回 logits
    """

    def __init__(
        self,
        adaptive: AdaptiveStrength,
        key: int,
        gamma: float,
        vocab_size: int,
        prefix_length: int = 1,
        record_pg: bool = False,
    ):
        self.adaptive = adaptive
        self.key = key
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.prefix_length = prefix_length
        self.record_pg = record_pg
        self.pg_records: list[dict] = []  # 每 token 的 (pg, r, position)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.prefix_length:
            return scores

        real_vocab = scores.shape[-1]

        for b_idx in range(input_ids.shape[0]):
            prev_token = input_ids[b_idx, -1].item()
            green_ids = compute_green_list(
                prev_token, self.key, self.gamma, real_vocab
            )
            green_ids_tensor = torch.tensor(list(green_ids), dtype=torch.long, device=scores.device)
            green_mask = torch.zeros(real_vocab, dtype=torch.bool, device=scores.device)
            green_mask[green_ids_tensor] = True

            probs = F.softmax(scores[b_idx], dim=-1)
            pg = probs[green_mask].sum().item()

            r = self.adaptive.compute_r(pg)

            if self.record_pg:
                self.pg_records.append({
                    "position": input_ids.shape[-1],
                    "pg": pg,
                    "r": r,
                    "green_size": len(green_ids),
                })

            if r <= FLOOR * 10:
                continue  # 水印强度可忽略，跳过调整

            # 公式 (6): 概率层面调整
            beta = r * (1.0 - pg)
            adjusted = probs.clone()

            if green_mask.any():
                weights_g = adjusted[green_mask]
                normalized_g = weights_g / (weights_g.sum() + FLOOR)
                adjusted[green_mask] = weights_g + normalized_g * beta

            red_mask = ~green_mask
            if red_mask.any():
                weights_r = adjusted[red_mask]
                normalized_r = weights_r / (weights_r.sum() + FLOOR)
                adjusted[red_mask] = weights_r - normalized_r * beta

            # 数值安全：clamp 到正数，重归一化，转 log
            adjusted = torch.clamp(adjusted, min=FLOOR)
            adjusted = adjusted / adjusted.sum()
            scores[b_idx] = torch.log(adjusted)

        return scores


# ---- 检测 ----

def morphmark_detect(
    token_ids: torch.Tensor,
    key: int,
    gamma: float,
    vocab_size: int,
    prefix_length: int = 1,
    prompt_len: int = 0,
) -> dict:
    """检测 MorphMark 水印（与 KGW 检测相同）。

    因为 MorphMark 也使用相同的红绿列表机制，
    只是水印强度自适应而非固定，检测逻辑不变。
    """
    start = max(prefix_length, prompt_len)
    num_scored = len(token_ids) - start
    if num_scored < 1:
        return {"z_score": 0.0, "green_count": 0, "total_scored": 0,
                "is_watermarked": False, "green_flags": []}

    green_count = 0
    green_flags = [-1] * prefix_length

    for i in range(start, len(token_ids)):
        prev_token = token_ids[i - 1].item()
        curr_token = token_ids[i].item()
        green_ids = compute_green_list(prev_token, key, gamma, vocab_size)
        if curr_token in green_ids:
            green_count += 1
            green_flags.append(1)
        else:
            green_flags.append(0)

    z_score = _compute_z_score(green_count, num_scored, gamma)
    return {
        "z_score": z_score,
        "green_count": green_count,
        "total_scored": num_scored,
        "is_watermarked": z_score > 4.0,
        "green_flags": green_flags,
    }


def _compute_z_score(green_count: int, total: int, gamma: float) -> float:
    expected = gamma * total
    denom = sqrt(total * gamma * (1 - gamma))
    if denom == 0:
        return 0.0
    return (green_count - expected) / denom
