"""鲁棒性攻击实现：WordNet 同义词替换、随机删除、GPT 改写"""

import random
import re
import nltk
from nltk.corpus import wordnet


def _ensure_nltk():
    try:
        wordnet.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)


def _get_pos(word: str):
    """获取单词的简单 POS 标签映射到 WordNet。"""
    tag = nltk.pos_tag([word])[0][1][0].upper() if nltk.pos_tag([word]) else None
    return {"N": wordnet.NOUN, "V": wordnet.VERB,
            "J": wordnet.ADJ, "R": wordnet.ADV}.get(tag)


def wordnet_synonym_attack(text: str, ratio: float = 0.3) -> str:
    """WordNet 同义词替换攻击。

    随机替换 ratio 比例的可替换词为同义词。

    Args:
        text: 原始文本
        ratio: 替换比例 (0.0 ~ 1.0)

    Returns:
        被攻击的文本
    """
    _ensure_nltk()
    words = text.split()
    if not words:
        return text

    n_replace = max(1, int(len(words) * ratio))
    indices = random.sample(range(len(words)), min(n_replace, len(words)))

    result = list(words)
    for i in indices:
        synsets = wordnet.synsets(words[i])
        synonyms = []
        for s in synsets:
            for lemma in s.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != words[i].lower():
                    synonyms.append(name)
        if synonyms:
            result[i] = random.choice(synonyms)

    return " ".join(result)


def random_deletion_attack(text: str, ratio: float = 0.3) -> str:
    """随机删除攻击。

    随机删除 ratio 比例的单词。
    """
    words = text.split()
    if not words:
        return text

    keep_count = max(1, int(len(words) * (1 - ratio)))
    kept = random.sample(words, keep_count)
    return " ".join(kept)


def gpt_paraphrase_attack(text: str, api_key: str, model: str = "gpt-3.5-turbo") -> str:
    """GPT 改写攻击（需要 OpenAI API key）。

    Args:
        text: 原始文本
        api_key: OpenAI API key
        model: GPT 模型

    Returns:
        改写后的文本
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Please rewrite the following text (Only return the rewritten text): {text}"
        }],
        temperature=0.7,
    )
    return response.choices[0].message.content
