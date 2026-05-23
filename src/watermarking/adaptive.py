"""自适应水印强度函数 φ(PG)

根据 Theorem 1，最优水印强度 r 与 PG 正相关。
三种增长函数：linear, exp, log。
"""

import math
from typing import Protocol


class AdaptiveStrength(Protocol):
    """自适应强度接口。"""
    def compute_r(self, pg: float) -> float:
        ...


class LinearStrength:
    """z(x) = k * x"""

    def __init__(self, p0: float, k: float, epsilon: float = 1e-10):
        self.p0 = p0
        self.k = k
        self.epsilon = epsilon

    def compute_r(self, pg: float) -> float:
        if pg <= self.p0:
            return self.epsilon
        r = self.k * pg
        return min(r, 1.0 - self.epsilon)


class ExpStrength:
    """z(x) = k * (exp(x) - 1)"""

    def __init__(self, p0: float, k: float, epsilon: float = 1e-10):
        self.p0 = p0
        self.k = k
        self.epsilon = epsilon

    def compute_r(self, pg: float) -> float:
        if pg <= self.p0:
            return self.epsilon
        r = self.k * (math.exp(pg) - 1)
        return min(r, 1.0 - self.epsilon)


class LogStrength:
    """z(x) = ln(k * x + 1)"""

    def __init__(self, p0: float, k: float, epsilon: float = 1e-10):
        self.p0 = p0
        self.k = k
        self.epsilon = epsilon

    def compute_r(self, pg: float) -> float:
        if pg <= self.p0:
            return self.epsilon
        r = math.log(self.k * pg + 1)
        return min(r, 1.0 - self.epsilon)


def create_adaptive_strength(
    wm_type: str, p0: float, k: float, epsilon: float = 1e-10
) -> AdaptiveStrength:
    """工厂函数：根据配置类型创建相应的强度函数。"""
    if wm_type == "morphmark_linear":
        return LinearStrength(p0, k, epsilon)
    elif wm_type == "morphmark_exp":
        return ExpStrength(p0, k, epsilon)
    elif wm_type == "morphmark_log":
        return LogStrength(p0, k, epsilon)
    else:
        raise ValueError(f"Unknown morphmark type: {wm_type}")
