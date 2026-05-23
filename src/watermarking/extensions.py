"""MorphMark 改进实验

基于 Phase 7a-d 分析结果:
- ~20% token PG < p0, 完全无水印 → 浪费检测能力
- 高 PG token r≈1.0, 已最大化 → 改进空间在低 PG 区
- 假阳性主要来自长文本 + 自然高绿词比例

改进 1: MorphMarkFloor — 低 PG 时加最小 KGW delta 作为底线
改进 2: EntropyAdaptive — 动态调整 p0 基于 token 熵估计
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from .hash_split import compute_green_list
from .adaptive import AdaptiveStrength, ExpStrength, create_adaptive_strength

FLOOR = 1e-10


class MorphMarkFloorLogitsProcessor(LogitsProcessor):
    """MorphMark + KGW floor: 低 PG token 至少获得最小 delta。

    问题: 原始 MorphMark 对 PG < p0 的 token 完全不加 watermark (~20% token)
    改进: 即使 r ≈ 0, 也给绿 token 加一个最小 delta (默认 0.5)
    效果: 每 token 都参与检测, 提升低熵场景的 z-score
    开销: PPL 可能微增, 但 delta_floor 很小
    """

    def __init__(
        self,
        adaptive: AdaptiveStrength,
        key: int,
        gamma: float,
        vocab_size: int,
        prefix_length: int = 1,
        delta_floor: float = 0.5,
    ):
        self.adaptive = adaptive
        self.key = key
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.prefix_length = prefix_length
        self.delta_floor = delta_floor

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

            if r <= FLOOR * 10:
                # KGW floor: apply minimum delta to green tokens
                if self.delta_floor > 0:
                    scores[b_idx, green_mask] += self.delta_floor
                continue

            # MorphMark full adjustment (same as original)
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

            adjusted = torch.clamp(adjusted, min=FLOOR)
            adjusted = adjusted / adjusted.sum()
            scores[b_idx] = torch.log(adjusted)

        return scores


class EntropyAdaptiveMorphMarkLogitsProcessor(LogitsProcessor):
    """熵自适应的 MorphMark: p0 随 token 熵估计动态调整。

    观察: 高熵位置 PG 大 → 可以用更低的 p0 更激进加水印
         低熵位置 PG 小 → 用更高的 p0 更保守减少 PPL 影响

    实现: p0_effective = p0_base * (1 - α * (H_token - H_min) / (H_max - H_min))
         其中 H_token 从 logits 的熵估计
         高熵 → p0_effective 低 → 更容易触发水印
    """

    def __init__(
        self,
        adaptive: AdaptiveStrength,
        key: int,
        gamma: float,
        vocab_size: int,
        prefix_length: int = 1,
        p0_base: float = 0.15,
        alpha: float = 0.5,  # 自适应强度 (0=固定p0, 1=全自适应)
    ):
        self.adaptive = adaptive
        self.key = key
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.prefix_length = prefix_length
        self.p0_base = p0_base
        self.alpha = alpha

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

            # Estimate token entropy from logits
            log_probs = F.log_softmax(scores[b_idx], dim=-1)
            entropy = -(probs * log_probs).sum().item()
            # Normalize: typical entropy range for OPT-1.3B is [0, ~12] (max = ln(50272) ≈ 10.8)
            max_entropy = math.log(real_vocab)
            norm_entropy = min(entropy / max_entropy, 1.0)

            # High entropy → lower effective p0 → more aggressive watermarking
            p0_effective = self.p0_base * (1.0 - self.alpha * norm_entropy)
            p0_effective = max(0.01, min(p0_effective, 0.5))

            # Adjust adaptive's p0 temporarily
            original_p0 = self.adaptive.p0
            self.adaptive.p0 = p0_effective
            r = self.adaptive.compute_r(pg)
            self.adaptive.p0 = original_p0

            if r <= FLOOR * 10:
                continue

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

            adjusted = torch.clamp(adjusted, min=FLOOR)
            adjusted = adjusted / adjusted.sum()
            scores[b_idx] = torch.log(adjusted)

        return scores
