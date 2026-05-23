"""数据集加载器：C4, WMT16, CNN-DailyMail, HumanEval"""

from typing import List
from datasets import load_dataset


def load_c4_subset(
    n_samples: int = 400,
    prompt_tokens: int = 30,
    seed: int = 42,
) -> List[str]:
    """从 C4 realnewslike 子集加载 prompts。

    取每条文本的前 prompt_tokens 个 token 作为 prompt。
    """
    ds = load_dataset("c4", "realnewslike", split="validation", streaming=True,
                     trust_remote_code=True)
    ds = ds.shuffle(seed=seed)
    prompts = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        text = item["text"]
        # 简单截断——后续 tokenizer 会精确控制
        words = text.split()
        prompt = " ".join(words[:prompt_tokens * 2])  # 粗略估计
        prompts.append(prompt)
    return prompts


def load_wmt16(
    n_samples: int = 400,
    seed: int = 42,
) -> List[tuple]:
    """加载 WMT16 DE-EN 测试集。返回 (de, en) 对。"""
    ds = load_dataset("wmt16", "de-en", split="test", streaming=True)
    ds = ds.shuffle(seed=seed)
    pairs = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        pairs.append((item["translation"]["de"], item["translation"]["en"]))
    return pairs


def load_cnn_dailymail(
    n_samples: int = 400,
    seed: int = 42,
    global_prompt: str = "Please summarize the following article: ",
) -> List[tuple]:
    """加载 CNN/DailyMail 测试集。返回 (article, highlights) 对。"""
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
    ds = ds.shuffle(seed=seed)
    pairs = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        prompt = f"{global_prompt}{item['article']}"
        pairs.append((prompt, item["highlights"]))
    return pairs


def load_dataset_by_name(name: str, **kwargs) -> List[str] | List[tuple]:
    """根据名称加载数据集。"""
    if name == "c4":
        return load_c4_subset(**kwargs)
    elif name == "wmt16":
        return load_wmt16(**kwargs)
    elif name == "cnn_dailymail":
        return load_cnn_dailymail(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
