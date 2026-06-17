"""评分与选股模块 — 校准 + 纳什均衡 + 选股过滤器 + 权重分配"""
import numpy as np
import torch
from pathlib import Path
from core.utils import _norm, _industry_col, _get_stock_embeddings
from config import (MODEL_DIR, DEVICE,
    GATE_DD_THRESHOLD, GATE_DD_MIN, GATE_DD_MAX, GATE_DD_CONSERVATIVE,
    TAIL_REVERSAL_BONUS, GAP_PENALTY_STRENGTH, BETA_PENALTY_STRENGTH,
    CROWDED_PENALTY_STRENGTH, TURNOVER_BONUS,
    VOL_MULT_DEFAULT, VOL_MULT_DIVERGE, VOL_MULT_TIGHTEN_RATIO,
    MAX_PER_INDUSTRY_DEFAULT, MAX_PER_INDUSTRY_DIVERGE,
    SOFT_CLUSTER_COS_THRESHOLD, CASH_BUFFER_LOW, CASH_BUFFER_HIGH,
    CONFIDENCE_VAR_THRESHOLD, CONFIDENCE_MAX_CASH, CONFIDENCE_CASH_SCALE,
    MACRO_DOWN_WEIGHT_OVERRIDE, TAIL_RISK_THRESHOLD, TAIL_CROWDING_THRESHOLD,
    NASH_LAM_DEFAULT, NASH_LAM_MIN, NASH_LAM_MAX, NASH_LAM_DIVERGE_SCALE,
    NASH_LAM_CONSERVATIVE_ADD, NASH_LAM_TAIL_ADD, NASH_POOL_SMALL_RATIO, NASH_PR_SCALE)

from core.utils import _norm


def calibrate_scores(scores, targets, mask=None):
    """
    Isotonic regression 校准：将模型分数映射到单调的"预期收益"尺度。

    为什么需要：
      - LGBM 输出 log-odds，CatBoost 输出 YetiRank，NN 输出 softmax logits
      - 不同模型的分数分布差异大，直接归一化+线性加权丢失了"置信度"信息
      - 校准后所有模型分数在统一的"预期收益"尺度上可比

    Args:
        scores: (N,) 模型原始分数
        targets: (N,) 真实收益
        mask: (N,) 有效样本掩码

    Returns:
        calibrated: (N,) 校准后的分数
    """
    from sklearn.isotonic import IsotonicRegression
    if mask is not None:
        valid = mask > 0.5
        s, t = scores[valid], targets[valid]
    else:
        s, t = scores, targets

    if len(s) < 20 or s.std() < 1e-8:
        return scores  # 样本太少或分数无变化，跳过

    # Isotonic: 保序回归，分数→收益
    ir = IsotonicRegression(out_of_bounds='clip', y_min=t.min(), y_max=t.max())
    try:
        calibrated = ir.fit_transform(s, t)
    except Exception:
        return scores  # 拟合失败则返回原始分数

    if mask is not None:
        result = scores.copy()
        result[valid] = calibrated
        return result
    return calibrated


def calibrate_multi_model(models_scores, targets, mask=None):
    """
    多模型校准：对每个模型的分数分别做 isotonic 校准。

    Args:
        models_scores: dict {model_name: scores_array}
        targets: (N,) 真实收益
        mask: (N,) 有效样本掩码

    Returns:
        calibrated: dict {model_name: calibrated_scores}
    """
    calibrated = {}
    for name, sc in models_scores.items():
        calibrated[name] = calibrate_scores(sc, targets, mask)
    return calibrated


# ── Embedding cache: 4处复用，避免重复加载模型 ──
_EMBEDDING_CACHE = None  # {'embeddings': ndarray (300, d_model), 'n_features': int}

def apply_turnover_penalty(scores, stock_ids, prev_selection=None, bonus=0.02):
    """
    换手惩罚/惯性奖励：上周入选的股票若本周仍在Top15，给小幅加分。
    减少不必要的换手，降低交易摩擦。
    """
    if prev_selection is None or len(prev_selection) == 0:
        return scores
    scores = np.array(scores, dtype=float)
    n = len(stock_ids)
    rank = np.argsort(np.argsort(scores)) / n  # 0-1 排名
    for i, sid in enumerate(stock_ids):
        if sid in prev_selection and rank[i] > 0.5:  # 仍在Top50%
            scores[i] += bonus  # 小幅加分
    return scores


def nash_equilibrium_selection(candidate_ids, candidate_scores, stock_idx_map, panel,
                                X_panel=None, lam=None, n_features=57):
    """
    在候选股中求解纯策略纳什均衡。
    效用 = 平均分数 - λ × 平均内部相似度

    lam = None 时自动确定：λ = mean_sim × 0.5（候选越拥挤 λ 越大）
    """
    n = len(candidate_ids)
    if n < 5:
        return candidate_ids, candidate_scores

    # NN embedding 相似度 — 复用缓存
    emb_cache = _get_stock_embeddings(X_panel, stock_idx_map)
    if emb_cache['embeddings'] is None:
        return candidate_ids, candidate_scores  # 模型不可用则跳过
    emb = emb_cache['embeddings']
    idxs = np.array([stock_idx_map[s] for s in candidate_ids])
    e = emb[idxs]
    e_n = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    sim = e_n @ e_n.T

    # ── 自动确定 λ ──
    n_pool = len(candidate_ids)
    if lam is None:
        upper_tri = np.triu(sim, k=1)
        mean_sim = upper_tri.sum() / max(n_pool * (n_pool - 1) / 2, 1)
        # λ = mean_sim × 0.5, 候选越拥挤 λ 越大, clamp [0.05, 0.8]
        lam = np.clip(mean_sim * 0.5, 0.05, 0.8)
        # 池子小时自动降低(原有逻辑)
        if n_pool <= 10:
            lam = lam * 0.6
        print(f"    Auto λ={lam:.3f} (mean_sim={mean_sim:.3f}, pool={n_pool})")
    else:
        if n_pool <= 10:
            lam = lam * 0.6

    sc = np.array(candidate_scores)

    def util(sel):
        if len(sel) < 2: return sc[sel].mean()
        return sc[sel].mean() - lam * np.mean(sim[np.ix_(sel, sel)])

    selected = list(np.argsort(sc)[-5:])
    for _ in range(100):
        cur = util(selected)
        best_dev = None
        for r in [i for i in range(n) if i not in selected]:
            for si in range(len(selected)):
                new_set = [x for x in selected if x != selected[si]] + [r]
                if util(new_set) > cur + 1e-6:
                    cur = util(new_set)
                    best_dev = (r, selected[si])
        if best_dev is None:
            break
        r, s = best_dev
        selected = [x for x in selected if x != s] + [r]

    return [candidate_ids[i] for i in selected], list(sc[selected])


# =============================================================================
# 选股与权重（精度门控 + 波动率过滤 + 行业分散）
# =============================================================================

def legacy_select_and_weight(pool_ids, pool_sc, pool, valid, scores, stock_ids,
                             panel, args, n_features=57, stock_idx_map=None,
                             X_lt=None, gate_dd=-0.08, nash_lam=0.25,
                             macro_risk=0.0, conservative=False):
    """v9 行业分散 + 交易性过滤 + 等权 20%。"""
    topk = min(args.topk, 5)
    n_stocks = len(stock_ids)

    # ── 收集每只股票的持仓特征 ──
    ret_1d = np.zeros(n_stocks)
    industries = ['Unknown'] * n_stocks
    for i, sid in enumerate(stock_ids):
        try:
            sd = panel.xs(sid, level="stock_id")
            ret_1d[i] = sd['pctChg'].iloc[-1] / 100.0 if len(sd) > 0 else 0
            ind_col = _industry_col(panel)
            industries[i] = str(sd[ind_col].iloc[-1]) if ind_col in sd.columns else 'Unknown'
        except:
            pass

    # ── 1. 交易性过滤：剔除涨跌停附近（买不到/有风险）──
    tradeable = (ret_1d < 0.09) & (ret_1d > -0.09)
    valid_pool = [i for i in valid if tradeable[i]]
    if len(valid_pool) < 5:
        valid_pool = list(valid)
        print(f"  [v9] Trade filter too strict → fallback ({len(valid_pool)} stocks)")

    # ── 2. 按分数降序，强制行业分散（同行业≤2只）──
    pool_scored = [(i, scores[i], industries[i]) for i in valid_pool]
    pool_scored.sort(key=lambda x: -x[1])

    sel_ids, ind_count = [], {}
    for idx, sc, ind in pool_scored:
        if ind_count.get(ind, 0) < 2:
            sel_ids.append(stock_ids[idx])
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(sel_ids) >= topk:
            break

    # ── 3. 不够 topk → 放开行业限制补齐 ──
    if len(sel_ids) < topk:
        for idx, sc, ind in pool_scored:
            sid = stock_ids[idx]
            if sid not in sel_ids:
                sel_ids.append(sid)
            if len(sel_ids) >= topk:
                break

    # ── 4. 等权 20% ──
    weights = [1.0 / len(sel_ids)] * len(sel_ids)

    print(f"  [v9] {len(sel_ids)} stocks, max 2/industry, equal-weight")
    return sel_ids, weights


# =============================================================================
# 主流程
# =============================================================================

