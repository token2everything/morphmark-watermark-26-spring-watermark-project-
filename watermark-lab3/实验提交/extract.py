"""
DCT 数字水印提取程序
从含水印图像中提取 "复旦大学"
"""
import numpy as np
from PIL import Image
from scipy.fftpack import dct

# ========== 参数（与嵌入程序保持一致）==========
BLOCK_SIZE = 8
SEED = 2026
EMBED_ZIGZAG = 58

ZIGZAG = np.array([
    [ 0,  1,  5,  6, 14, 15, 27, 28],
    [ 2,  4,  7, 13, 16, 26, 29, 42],
    [ 3,  8, 12, 17, 25, 30, 41, 43],
    [ 9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63],
])

def zigzag_to_xy(idx):
    mask = ZIGZAG == idx
    pos = np.argwhere(mask)
    return pos[0][0], pos[0][1]

def bits_to_text(bits):
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        bytes_list.append(byte)
    return bytes(bytes_list).decode('utf-8')

def extract_watermark():
    # 1. 读取含水印图像
    img = Image.open("lenna_watermarked.bmp")
    arr = np.array(img, dtype=np.float64)
    H, W, _ = arr.shape
    blue = arr[:, :, 2]

    num_blocks_h = H // BLOCK_SIZE
    num_blocks_w = W // BLOCK_SIZE
    total_blocks = num_blocks_h * num_blocks_w

    er, ec = zigzag_to_xy(EMBED_ZIGZAG)

    # 2. 先提取前16位（长度信息）
    rng = np.random.RandomState(SEED)
    block_indices_16 = rng.permutation(total_blocks)[:16]

    length_bits = []
    for block_idx in block_indices_16:
        bi = block_idx // num_blocks_w
        bj = block_idx % num_blocks_w
        y0, x0 = bi * BLOCK_SIZE, bj * BLOCK_SIZE

        block = blue[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE]
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        if dct_block[er, ec] > 0:
            length_bits.append(1)
        else:
            length_bits.append(0)

    # 解码长度
    data_len = 0
    for b in length_bits:
        data_len = (data_len << 1) | b
    print(f"水印数据长度: {data_len} bits")

    total_bits = 16 + data_len

    # 3. 提取全部比特
    rng = np.random.RandomState(SEED)
    block_indices = rng.permutation(total_blocks)[:total_bits]

    all_bits = []
    for block_idx in block_indices:
        bi = block_idx // num_blocks_w
        bj = block_idx % num_blocks_w
        y0, x0 = bi * BLOCK_SIZE, bj * BLOCK_SIZE

        block = blue[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE]
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        if dct_block[er, ec] > 0:
            all_bits.append(1)
        else:
            all_bits.append(0)

    # 4. 解码水印文本
    payload_bits = all_bits[16:16 + data_len]
    watermark_text = bits_to_text(payload_bits)
    print(f"提取的水印: {watermark_text}")
    return watermark_text

if __name__ == '__main__':
    extract_watermark()
