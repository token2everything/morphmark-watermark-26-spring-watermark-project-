
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct

# ========== 参数 ==========
BLOCK_SIZE = 8
SEED = 2026                    # 随机种子（提取时需要相同种子）
ALPHA = 80                     # 嵌入强度（高频系数修改幅度）
EMBED_ZIGZAG = 58              # 嵌入位置（zigzag序号0~63，58属于高频）

# ========== zigzag扫描顺序 ==========
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
    """zigzag序号 → (row, col)"""
    mask = ZIGZAG == idx
    pos = np.argwhere(mask)
    return pos[0][0], pos[0][1]

def text_to_bits(text):
    """将文本转为比特列表"""
    data = text.encode('utf-8')
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_text(bits):
    """将比特列表还原为文本"""
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        bytes_list.append(byte)
    return bytes(bytes_list).decode('utf-8')

def embed_watermark():
    # 1. 读取载体图像
    img = Image.open("Lenna.bmp")
    arr = np.array(img, dtype=np.float64)
    H, W, C = arr.shape
    print(f"图像尺寸: {H}×{W}, 通道数: {C}")

    # 2. 在水印比特前加长度头（16位），方便提取
    watermark_text = "复旦大学"
    payload_bits = text_to_bits(watermark_text)
    length_bits = []
    length = len(payload_bits)
    for i in range(15, -1, -1):
        length_bits.append((length >> i) & 1)
    all_bits = length_bits + payload_bits
    print(f"水印文本: {watermark_text}")
    print(f"水印比特: {all_bits}")
    print(f"总比特数: {len(all_bits)} (16位长度 + {len(payload_bits)}位数据)")

    # 3. 随机选择嵌入块
    num_blocks_h = H // BLOCK_SIZE
    num_blocks_w = W // BLOCK_SIZE
    total_blocks = num_blocks_h * num_blocks_w
    print(f"分块: {num_blocks_h}×{num_blocks_w} = {total_blocks}块")

    rng = np.random.RandomState(SEED)
    block_indices = rng.permutation(total_blocks)[:len(all_bits)]

    # 4. 嵌入位置 (zigzag坐标→块内像素坐标)
    er, ec = zigzag_to_xy(EMBED_ZIGZAG)
    print(f"嵌入位置: DCT块内 ({er}, {ec}), zigzag序号 {EMBED_ZIGZAG} (高频)")

    # 5. 在蓝色通道嵌入水印
    blue = arr[:, :, 2].copy()

    for bit_idx, block_idx in enumerate(block_indices):
        bi = block_idx // num_blocks_w          # 块行
        bj = block_idx % num_blocks_w           # 块列
        y0, x0 = bi * BLOCK_SIZE, bj * BLOCK_SIZE

        # 取8×8块，做DCT
        block = blue[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE]
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        # 在高频位置嵌入
        bit = all_bits[bit_idx]
        if bit == 1:
            dct_block[er, ec] = ALPHA
        else:
            dct_block[er, ec] = -ALPHA

        # IDCT还原
        idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        blue[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE] = idct_block

    # 6. 裁切到合法范围并保存
    arr[:, :, 2] = blue
    arr = np.clip(arr, 0, 255)
    result = Image.fromarray(arr.astype(np.uint8))
    result.save("lenna_watermarked.bmp")
    print("\n嵌入完成！保存为 lenna_watermarked.bmp")

    # 计算PSNR
    original = np.array(Image.open("Lenna.bmp"), dtype=np.float64)
    mse = np.mean((arr - original) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    print(f"PSNR: {psnr:.2f} dB")

if __name__ == '__main__':
    embed_watermark()
