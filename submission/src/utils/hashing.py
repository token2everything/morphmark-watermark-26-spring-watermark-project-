"""基于 SHA-256 的哈希随机种子生成器"""

import hashlib


def generate_seed(prev_token: int, private_key: int) -> int:
    """为绿/红列表分割生成确定性随机种子。"""
    raw = f"{prev_token}_{private_key}".encode()
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:16], 16)
