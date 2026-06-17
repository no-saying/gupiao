"""风险检测模块 — 宏观门控 + HMM市场状态 + 极端周检测 + Beta惩罚"""
import numpy as np
import pandas as pd
from config import (TAIL_RISK_THRESHOLD, TAIL_CROWDING_THRESHOLD, TAIL_DISP_SCALE,
    TAIL_DISP_WEIGHT, TAIL_CROWDING_WEIGHT, TAIL_MACRO_WEIGHT, TAIL_CROWDING_SCALE,
    HMM_N_STATES, HMM_SEQ_LEN, HMM_BEAR_SHIFT, HMM_BULL_SHIFT,
    MACRO_RISK_FACTOR_BASE, MACRO_RISK_HMM_SCALE)
from core.utils import _norm, _industry_col
from core.ensemble import AUX_MODEL_DIR


def _compute_macro_gate(panel, stock_ids):
    """宏观环境检测：下行风险等级 0-1。

    综合判断：指数趋势、量价关系、MA20偏离。
    返回 0（正常）~ 1（极端下行）。
    """
    try:
        first_stock = stock_ids[0]
        sd = panel.xs(first_stock, level='stock_id')
        close = sd['close'].astype(float)
        vol = sd['volume'].astype(float)

        # 等权市场收益（采样30只避免太慢）
        n_sample = min(30, len(stock_ids))
        all_ret = []
        for sid in stock_ids[:n_sample]:
            try:
                all_ret.append(panel.xs(sid, level='stock_id')['pctChg'].astype(float) / 100.0)
            except Exception:
                pass
        if not all_ret:
            return 0.0
        mkt_ret = pd.concat(all_ret, axis=1).mean(axis=1)

        ret_5d = float(mkt_ret.iloc[-5:].mean())
        ret_20d = float(mkt_ret.iloc[-20:].sum())
        vol_5d = float(vol.iloc[-5:].mean())
        vol_20d_avg = float(vol.iloc[-20:].mean())
        vol_ratio = vol_5d / (vol_20d_avg + 1e-12)
        last_close = float(close.iloc[-1])
        ma20 = float(close.iloc[-20:].mean())

        down_risk = 0.0
        if ret_5d < -0.01:   # 近5日下跌
            down_risk += 0.25
        if ret_5d < -0.03:   # 近5日大跌
            down_risk += 0.20
        if ret_20d < -0.03:  # 近20日趋势向下
            down_risk += 0.20
        if last_close < ma20:  # 跌破MA20
            down_risk += 0.15
        if ret_5d < 0 and vol_ratio > 1.2:  # 放量下跌
            down_risk += 0.20

        return float(np.clip(down_risk, 0.0, 1.0))
    except Exception:
        return 0.0


def _detect_tail_signals(X_lt, lgbm_scores, stock_ids, panel):
    """极端周检测：拥挤度 + pinball 离散度 + 宏观环境 + Beta 惩罚。

    返回 dict:
      - crowding: 0-1，AUX/SW 模型 top 集中度
      - pinball_disp: pinball 分数截面标准差
      - tail_risk: 0-1，综合极端周概率
      - reversal_bonus: (300,) 数组，LGBM↑pinball↓ 的股票加分
      - macro_risk: 0-1，宏观下行风险
      - beta_penalty: (300,) 数组，高Beta股的分数惩罚
    """
    # 1. Pinball 模型分数 + 离散度
    pinball_path = AUX_MODEL_DIR / "portfolio_model_pinball.pt"
    pinball_scores = np.zeros(300)
    if pinball_path.exists():
        try:
            pinball_scores = load_predict(pinball_path, X_lt, 80)
        except Exception:
            pass
    pinball_disp = float(np.std(pinball_scores)) if pinball_scores.any() else 0.0
    disp_level = np.clip(pinball_disp / 0.04, 0.0, 1.0)

    # 2. 拥挤度：pinball top-20 行业集中度
    crowding = 0.0
    dominant_sector = None  # 初始化
    # 优先用 SW L2 行业名（字符串），回退到 industry 编码
    industry_col = _industry_col(panel)
    if industry_col in panel.columns and pinball_scores.any():
        try:
            top20_idx = np.argsort(pinball_scores)[-20:][::-1]
            top20_stocks = [stock_ids[i] for i in top20_idx]
            sectors = []
            for sid in top20_stocks:
                try:
                    ind = str(panel.xs(sid, level='stock_id')[_industry_col(panel)].iloc[-1])
                    sectors.append(ind[:3] if industry_col == 'l2_name' else ind)
                except Exception:
                    sectors.append('???')
            from collections import Counter
            sector_counts = Counter(sectors)
            top_sector_pct = max(sector_counts.values()) / 20.0
            crowding = np.clip((top_sector_pct - 0.4) / 0.4, 0.0, 1.0)
            dominant_sector = sector_counts.most_common(1)[0][0] if sector_counts else None
        except Exception:
            pass

    # 拥挤行业动量惩罚：拥挤度高时，对拥挤行业的股票施加分数惩罚
    crowded_penalty = np.zeros(300)
    if crowding > 0.5 and dominant_sector and 'industry' in panel.columns:
        for i, sid in enumerate(stock_ids):
            try:
                ind = str(panel.xs(sid, level='stock_id')[_industry_col(panel)].iloc[-1])
                if ind[:3] == dominant_sector:
                    crowded_penalty[i] = crowding  # 拥挤度越高惩罚越重
            except Exception:
                pass

    # 3. 宏观下行风险
    macro_risk = _compute_macro_gate(panel, stock_ids)

    # 4. 反转候选股：LGBM 高分但 pinball 低分
    reversal_bonus = np.zeros(300)
    if lgbm_scores and pinball_scores.any():
        lgbm_arr = np.array([lgbm_scores.get(s, -999) for s in stock_ids])
        lgbm_rank = _norm(lgbm_arr)
        pb_rank = _norm(pinball_scores)
        reversal_bonus = np.clip(lgbm_rank - pb_rank, 0, 1)

    # 5. Beta 惩罚：计算每只股票的近似 Beta（与市场收益的相关性×波动率比）
    beta_penalty = np.zeros(300)
    try:
        n_sample = min(30, len(stock_ids))
        mkt_rets = []
        for sid in stock_ids[:n_sample]:
            try:
                mkt_rets.append(panel.xs(sid, level='stock_id')['pctChg'].astype(float) / 100.0)
            except Exception:
                pass
        if mkt_rets:
            mkt_ret = pd.concat(mkt_rets, axis=1).mean(axis=1)
            mkt_vol = float(mkt_ret.iloc[-60:].std())
            for i, sid in enumerate(stock_ids):
                try:
                    sd = panel.xs(sid, level='stock_id')
                    stk_ret = sd['pctChg'].astype(float).iloc[-60:] / 100.0
                    common = mkt_ret.iloc[-60:].index.intersection(stk_ret.index)
                    if len(common) < 20:
                        continue
                    stk_vol = float(stk_ret.loc[common].std())
                    corr = float(stk_ret.loc[common].corr(mkt_ret.loc[common]))
                    beta = max(0, (stk_vol / (mkt_vol + 1e-12)) * max(0, corr))
                    # Beta > 1.2 在极端周是风险
                    beta_penalty[i] = np.clip((beta - 0.8) / 0.8, 0.0, 1.0) if beta > 0.8 else 0.0
                except Exception:
                    pass
    except Exception:
        pass

    # 6. 综合极端周概率（宏观风险权重更高）
    tail_risk = np.clip(TAIL_DISP_WEIGHT * disp_level + TAIL_CROWDING_WEIGHT * crowding + TAIL_MACRO_WEIGHT * macro_risk, 0.0, 1.0)

    return {
        'crowding': crowding,
        'pinball_disp': pinball_disp,
        'disp_level': disp_level,
        'tail_risk': tail_risk,
        'macro_risk': macro_risk,
        'reversal_bonus': reversal_bonus,
        'beta_penalty': beta_penalty,
        'crowded_penalty': crowded_penalty,
        'dominant_sector': dominant_sector or '',
    }


NEW_FEATS = {'gap_ratio', 'close_pos', 'intraday_range', 'streak_up', 'streak_down', 'vol_ret_corr_10d'}
EXCLUDE = {'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
           'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM', 'pbMRQ',
           'market_ret', 'stock_id', 'vol_ma5', 'vol_ma20',
           'turn_ma5', 'turn_ma20', 'industry', 'amplitude', 'change',
           'roe_ttm', 'np_margin', 'gp_margin', 'eps_ttm',
           'current_ratio', 'debt_to_asset', 'profit_yoy', 'equity_yoy', 'nn_score',
           # Tushare raw columns (derived features start with mf_/margin_/north_/chip_/holder_/limit_)
           'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
           'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
           'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
           'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
           'net_mf_vol', 'net_mf_amount',
           'rzye', 'rqye', 'rzmre', 'rqyl', 'rzche', 'rqchl', 'rqmcl', 'rzrqye',
           'north_hold_vol', 'north_hold_ratio',
           'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv',
           'turnover_rate', 'turnover_rate_f', 'volume_ratio',
           'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct',
           'weight_avg', 'winner_rate',
           'holder_num',
           'limit_up_times', 'limit_down_times', 'broken_board_times',
           'industry_l1', 'industry_l3',
           'l1_code', 'l1_name', 'l2_code', 'l3_code', 'l3_name',
           'pe', 'pb', 'ps', 'dv_ratio',
           }


# =============================================================================
# LightGBM Ranker
# =============================================================================

def compute_market_regime(panel, n_states=3, seq_len=63):
    """
    HMM 隐马尔可夫模型市场状态识别。
    基于等权市场收益率序列，识别牛/震荡/熊三种状态。
    返回: (regime_id, regime_name, transition_prob)
    """
    from hmmlearn import hmm
    # 提取等权市场日收益率
    try:
        first_stock = panel.index.get_level_values('stock_id').unique()[0]
        sd = panel.xs(first_stock, level='stock_id')
        ret = sd['pctChg'].iloc[-seq_len:].values / 100.0
        ret = np.nan_to_num(ret, nan=0.0)
        ret = ret.reshape(-1, 1)

        # 训练 3 状态 HMM
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag',
                                 n_iter=100, random_state=42)
        model.fit(ret)
        states = model.predict(ret)

        # 按均值排序: 0=熊, 1=震荡, 2=牛 (假设收益率均值递增)
        state_means = [np.mean(ret[states == s]) for s in range(n_states)]
        order = np.argsort(state_means)
        state_map = {order[0]: 0, order[1]: 1, order[2]: 2}  # 0=熊, 1=震荡, 2=牛
        current_state = state_map[states[-1]]

        # 转移概率
        trans_prob = model.transmat_[states[-1], :]
        next_probs = {0: trans_prob[order[0]], 1: trans_prob[order[1]], 2: trans_prob[order[2]]}

        regime_names = {0: "熊市", 1: "震荡", 2: "牛市"}
        return {
            "id": current_state,
            "name": regime_names[current_state],
            "next_bear": next_probs[0],
            "next_bull": next_probs[2],
            "volatility": float(np.std(ret)),
        }
    except Exception as e:
        return {"id": 1, "name": "震荡", "next_bear": 0.0, "next_bull": 0.0, "volatility": 0.0}


def dynamic_moe_blend(lgbm_scores, nn_scores, regime, base_lgbm_w=0.7):
    """动态混合专家 (MoE): 根据市场状态动态调整融合权重。"""
    if nn_scores is None:
        return lgbm_scores
    regime_id = regime["id"]
    nn_w = 0.40 if regime_id == 2 else 0.15 if regime_id == 0 else 0.30
    lgbm_w = 1.0 - nn_w
    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx - mn > 1e-9 else np.zeros_like(s)
    blend = lgbm_w * norm(lgbm_scores) + nn_w * norm(nn_scores)
    print(f"  MoE: {regime['name']} → LGBM={lgbm_w:.2f} NN={nn_w:.2f}")
    return blend


def hmm_forward_predict(regime_info):
    """
    HMM 前向预测：用转移矩阵预测下周最可能的市场状态。
    返回调整因子: 熊市概率高→收紧beta_penalty, 牛市概率高→放松。
    """
    next_bear = regime_info.get("next_bear", 0.33)
    next_bull = regime_info.get("next_bull", 0.33)
    # 预期风险 = 熊市概率 - 牛市概率 (范围 -1 到 1)
    expected_risk = next_bear - next_bull
    # 缩放为调节因子: 高风险→1.3(用力收紧), 低风险→0.7(放松)
    risk_factor = 1.0 + 0.3 * np.clip(expected_risk, -1, 1)
    if abs(expected_risk) > 0.3:
        direction = "收紧防御" if expected_risk > 0 else "放松保护"
        print(f"  [HMM Forward] P(熊)={next_bear:.0%} P(牛)={next_bull:.0%} → {direction} ×{risk_factor:.2f}")
    return risk_factor


# =============================================================================
# 纳什均衡选股
# =============================================================================

