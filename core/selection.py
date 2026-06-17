"""
core/selection.py — 选股与权重
"""

import numpy as np


def select_top_stocks(scores: dict, panel, top_k: int = 5,
                      drawdown_threshold: float = -0.08,
                      momentum_threshold: float = 0.0) -> tuple[list, list]:
    """精度门控 + 波动率过滤 + 分数排序 top-k。

    Args:
        scores: {stock_id: lgb_score}
        panel: MultiIndex(date, stock_id) 特征面板
        top_k: 选股数
        drawdown_threshold: 20日最大回撤阈值（<-8% 不选）
        momentum_threshold: 5日动量阈值（<0 不选）

    Returns:
        (selected_ids, scores_of_selected)
    """
    # 按分数排序
    sorted_stocks = sorted(scores.items(), key=lambda x: -x[1])

    # ── 精度门控 ──
    passed = []
    for sid, scv in sorted_stocks:
        try:
            sd = panel.xs(sid, level="stock_id")
            ret_5d = sd['pctChg'].iloc[-5:].mean() / 100.0
            dd = (sd['close'] / sd['close'].rolling(20).max() - 1.0).iloc[-20:].min()
        except Exception:
            ret_5d, dd = 0, -0.01

        if ret_5d > momentum_threshold and dd > drawdown_threshold:
            passed.append((sid, scv))

    # ── 门控不足时用全市场补 ──
    if len(passed) < top_k:
        existing = set(s for s, _ in passed)
        for sid, scv in sorted_stocks:
            if sid in existing:
                continue
            passed.append((sid, scv))
            existing.add(sid)
            if len(passed) >= top_k:
                break

    # ── 波动率过滤 ──
    pool_vols = []
    for sid, _ in sorted_stocks[:40]:
        try:
            pool_vols.append(
                panel.xs(sid, level="stock_id")['pctChg'].iloc[-20:].std() / 100.0)
        except Exception:
            pool_vols.append(0.03)
    vol_med = np.median(pool_vols) if pool_vols else 0.03

    filtered = []
    for sid, scv in passed:
        try:
            v = panel.xs(sid, level="stock_id")['pctChg'].iloc[-20:].std() / 100.0
        except Exception:
            v = 0.03
        if v <= vol_med * 1.2:
            filtered.append((sid, scv))

    # ── Top-k ──
    selected = filtered[:top_k]

    sel_ids = [s for s, _ in selected]
    sel_sc = [v for _, v in selected]

    return sel_ids, sel_sc


def compute_weights(selected_ids: list, scores: list,
                    method: str = "equal") -> list[float]:
    """权重分配。

    Args:
        method: "equal" — 等权; "score" — 分数加权（cap 0.5）

    Returns:
        权重列表
    """
    n = len(selected_ids)
    if n == 0:
        return []

    if method == "equal":
        return [1.0 / n] * n

    if method == "score":
        raw = np.array([max(0, s) for s in scores], dtype=float)
        if raw.sum() <= 1e-12:
            raw = np.ones(n)
        weights = raw / raw.sum()
        weights = np.clip(weights, None, 0.5)
        weights = weights / weights.sum()
        return list(weights)

    return [1.0 / n] * n
