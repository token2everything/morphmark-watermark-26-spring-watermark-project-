"""测试 hash_split —— 核心基础设施"""

import sys
sys.path.insert(0, '.')

from src.watermarking.hash_split import compute_green_list

VOCAB_SIZE = 50265
GAMMA = 0.5
KEY = 15485863


def test_deterministic():
    """相同输入产生相同绿列表。"""
    g1 = compute_green_list(42, KEY, GAMMA, VOCAB_SIZE)
    g2 = compute_green_list(42, KEY, GAMMA, VOCAB_SIZE)
    assert g1 == g2, "相同输入应产生相同绿列表"


def test_green_size():
    """绿列表大小 = gamma * vocab_size。"""
    g = compute_green_list(100, KEY, GAMMA, VOCAB_SIZE)
    expected = int(GAMMA * VOCAB_SIZE)
    assert abs(len(g) - expected) <= 1, f"绿列表大小 {len(g)} 与期望 {expected} 不符"


def test_different_prev_token():
    """不同 prev_token 产生不同绿列表。"""
    g1 = compute_green_list(10, KEY, GAMMA, VOCAB_SIZE)
    g2 = compute_green_list(20, KEY, GAMMA, VOCAB_SIZE)
    overlap = len(g1 & g2)
    # 两个随机集合应有大约 GAMMA 的重叠率
    assert overlap < len(g1), "不同 prev_token 不应产生完全相同的绿列表"


def test_different_key():
    """不同 key 产生不同绿列表。"""
    g1 = compute_green_list(100, 12345, GAMMA, VOCAB_SIZE)
    g2 = compute_green_list(100, 67890, GAMMA, VOCAB_SIZE)
    overlap = len(g1 & g2)
    assert overlap < len(g1), "不同 key 不应产生完全相同的绿列表"


def test_disjoint_and_cover():
    """绿列表内的 ID 都在词表范围内。"""
    g = compute_green_list(7, KEY, GAMMA, VOCAB_SIZE)
    for token_id in g:
        assert 0 <= token_id < VOCAB_SIZE, f"token id {token_id} 超出词表范围"


if __name__ == '__main__':
    test_deterministic()
    print("✓ test_deterministic passed")
    test_green_size()
    print("✓ test_green_size passed")
    test_different_prev_token()
    print("✓ test_different_prev_token passed")
    test_different_key()
    print("✓ test_different_key passed")
    test_disjoint_and_cover()
    print("✓ test_disjoint_and_cover passed")
    print("\nAll hash_split tests passed!")
