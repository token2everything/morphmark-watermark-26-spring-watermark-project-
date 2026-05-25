"""KGW 水印算法 (Kirchenbauer et al. 2023)

KGW 是红绿列表水印的奠基方法。在每步生成时：
1. 用前一 token + 私钥确定性分割词表
2. 绿 token logits 加固定 delta
3. 生成后通过 z-score 检测
"""

from math import sqrt
from typing import List, Tuple

import torch
from transformers import LogitsProcessor

from .hash_split import compute_green_list


# ---- LogitsProcessor ----

class KGWLogitsProcessor(LogitsProcessor):
    """KGW 水印的 HuggingFace LogitsProcessor。

    在 __call__ 中将绿列表 token 的 logits 加上固定 delta。
    """

    def __init__(
        self,
        key: int,
        gamma: float,
        delta: float,
        vocab_size: int,
        prefix_length: int = 1,
    ):
        self.key = key
        self.gamma = gamma
        self.delta = delta
        self.vocab_size = vocab_size
        self.prefix_length = prefix_length

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
            scores[b_idx, green_mask] += self.delta

        return scores


# ---- 检测 ----

def kgw_detect(
    token_ids: torch.Tensor,
    key: int,
    gamma: float,
    vocab_size: int,
    prefix_length: int = 1,
    prompt_len: int = 0,
) -> dict:
    """对 token 序列进行 KGW 水印检测。

    对每个位置 i (>= prefix_length)：
        - 用 token_ids[i-1] 重建绿列表
        - 检查 token_ids[i] 是否在绿列表中
    然后计算 z-score。

    Args:
        token_ids: 完整的 token ID 序列 (prompt + generated)
        key: 私钥
        gamma: 绿列表比例
        vocab_size: 词表大小
        prefix_length: hash 上下文窗口
        prompt_len: prompt 长度（这些 token 不参与 z-score 计算）

    Returns:
        dict with z_score, green_count, total_scored, is_watermarked
    """
    num_scored = len(token_ids) - max(prefix_length, prompt_len)
    if num_scored < 1:
        # 序列太短，无法有效检测
        num_scored = len(token_ids) - prefix_length
        start_pos = prefix_length
    else:
        start_pos = max(prefix_length, prompt_len)

    if num_scored < 1:
        return {"z_score": 0.0, "green_count": 0, "total_scored": 0,
                "is_watermarked": False, "green_flags": []}

    green_count = 0
    green_flags = [-1] * prefix_length  # 前缀不可评分

    for i in range(start_pos, len(token_ids)):
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
    """计算标准 z-score。"""
    expected = gamma * total
    denom = sqrt(total * gamma * (1 - gamma))
    if denom == 0:
        return 0.0
    return (green_count - expected) / denom
