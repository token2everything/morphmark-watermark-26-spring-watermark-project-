"""水印检测指标：TPR@FPR, Best F1, AUC, Bootstrap CI"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_tpr_at_fpr(
    z_scores_wm: np.ndarray,
    z_scores_uw: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """在指定 FPR 下计算 TPR。

    从未水印分布计算阈值：使得 FPR <= target_fpr，
    然后在此阈值下计算水印样本的 TPR。

    Args:
        z_scores_wm: 水印文本的 z-scores
        z_scores_uw: 未水印文本的 z-scores
        target_fpr: 目标假阳性率

    Returns:
        TPR at target FPR
    """
    n_uw = len(z_scores_uw)
    k = int(np.ceil(target_fpr * n_uw))
    if k > 0:
        threshold = np.partition(z_scores_uw, -k)[-k]
    else:
        threshold = float("inf")

    tpr = np.mean(z_scores_wm > threshold)
    return float(tpr)


def compute_best_f1(
    z_scores_wm: np.ndarray,
    z_scores_uw: np.ndarray,
) -> dict:
    """扫描所有可能阈值，找到最优 F1 分数。

    Returns:
        dict with best_f1, best_threshold, best_tpr, best_fpr
    """
    all_scores = np.concatenate([z_scores_wm, z_scores_uw])
    labels = np.concatenate([
        np.ones(len(z_scores_wm)),
        np.zeros(len(z_scores_uw)),
    ])

    sorted_idx = np.argsort(all_scores)
    sorted_scores = all_scores[sorted_idx]
    sorted_labels = labels[sorted_idx]

    best_f1 = 0.0
    best_threshold = 0.0
    best_tpr = 0.0
    best_fpr = 0.0

    n_pos = len(z_scores_wm)
    n_neg = len(z_scores_uw)

    tp = n_pos  # start with all classified as positive
    fp = n_neg

    for i in range(len(sorted_scores) - 1, -1, -1):
        if sorted_labels[i] == 1:
            tp -= 1
        else:
            fp -= 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_pos if n_pos > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = sorted_scores[i] if i < len(sorted_scores) else sorted_scores[-1]
            best_tpr = recall
            best_fpr = fp / n_neg if n_neg > 0 else 0.0

    return {
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "best_tpr": best_tpr,
        "best_fpr": best_fpr,
    }


def compute_auc_roc(
    z_scores_wm: np.ndarray,
    z_scores_uw: np.ndarray,
) -> float:
    """计算 AUC-ROC。"""
    labels = np.concatenate([
        np.ones(len(z_scores_wm)),
        np.zeros(len(z_scores_uw)),
    ])
    scores = np.concatenate([z_scores_wm, z_scores_uw])
    return float(roc_auc_score(labels, scores))


def bootstrap_ci(
    data: np.ndarray,
    stat_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple:
    """Bootstrap 置信区间。

    Args:
        data: 输入数据
        stat_fn: 统计函数，接受数据返回标量
        n_bootstrap: 重采样次数
        alpha: 显著性水平

    Returns:
        (lower, upper, estimate)
    """
    estimate = stat_fn(data)
    bootstraps = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstraps.append(stat_fn(sample))
    lower = np.percentile(bootstraps, 100 * alpha / 2)
    upper = np.percentile(bootstraps, 100 * (1 - alpha / 2))
    return lower, upper, estimate
