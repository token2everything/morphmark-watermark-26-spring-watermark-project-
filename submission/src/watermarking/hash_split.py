"""绿/红列表分割 —— 核心基础设施

这是整个水印系统中最关键的文件。生成和检测都必须调用同一个 compute_green_list 函数。
"""
import hashlib
import random
from typing import Set


def compute_green_list(
    prev_token: int,
    private_key: int,
    gamma: float,
    vocab_size: int
) -> Set[int]:
    """依据前一 token 和私钥确定性分割词表。

    算法:
        1. seed = SHA-256(prev_token || private_key)
        2. rng = random.Random(seed)
        3. shuffle vocab indices
        4. 返回前 gamma*|V| 个作为绿列表

    Args:
        prev_token: 前一 token 的 ID
        private_key: 私钥
        gamma: 绿列表比例，默认 0.5
        vocab_size: 词表大小

    Returns:
        绿列表中 token ID 的集合
    """
    raw = f"{prev_token}_{private_key}".encode()
    digest = hashlib.sha256(raw).hexdigest()
    seed = int(digest[:16], 16)

    rng = random.Random(seed)
    indices = list(range(vocab_size))
    rng.shuffle(indices)

    green_size = int(gamma * vocab_size)
    return set(indices[:green_size])
