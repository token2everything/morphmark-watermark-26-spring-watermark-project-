"""文本质量评估：PPL"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = 1024,
    stride: int = 512,
) -> list[float]:
    """使用 HuggingFace 计算每条文本的 Perplexity。

    Args:
        model: 参考模型
        tokenizer: 对应 tokenizer
        texts: 文本列表
        max_length: 滑动窗口最大长度
        stride: 滑动步长

    Returns:
        每条文本的 PPL 值
    """
    ppls = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=max_length)
            input_ids = encodings["input_ids"].to(model.device)
            seq_len = input_ids.size(1)

            nlls = []
            prev_end = 0
            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                if end <= prev_end:
                    break
                trg_len = end - prev_end
                chunk = input_ids[:, begin:end]
                target = chunk[:, -trg_len:]

                output = model(chunk)
                logits = output.logits[:, -trg_len - 1: -1, :]
                if target.size(1) > 0 and logits.size(1) > 0:
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target.reshape(-1),
                        reduction="sum",
                    )
                    nlls.append(loss.item())
                prev_end = end

            if nlls:
                ppl = torch.exp(torch.tensor(sum(nlls) / sum(
                    (target != tokenizer.pad_token_id).sum().item()
                    if tokenizer.pad_token_id is not None
                    else target.numel()
                    for target in [input_ids]
                ))).item()
                ppls.append(ppl)
            else:
                ppls.append(float("inf"))
    model.train()
    return ppls


def compute_ppl_simple(
    model, tokenizer, texts: list[str], device: str = "cuda",
) -> list[float]:
    """简化 PPL 计算：对整个序列做一次 forward pass，用 teacher forcing 算 loss。

    更快但要求文本长度在模型上下文窗口内。
    """
    ppls = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
            output = model(**enc, labels=enc["input_ids"])
            loss = output.loss
            if loss is not None:
                ppls.append(torch.exp(loss).item())
            else:
                ppls.append(float("nan"))
    model.train()
    return ppls
