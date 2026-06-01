"""
Master Controller — LightGBM 主力 + NN 辅助 + 纳什博弈

使用方法:
  python controller.py                         # LGBM+NN 融合 (0.149)
  python controller.py --game                  # + 纳什均衡多样化
  python controller.py --game 0.3              # 自定义竞争强度
  python controller.py --ensemble lgbm         # LightGBM 单独
  python controller.py --ensemble compare      # 对比所有策略
"""

import argparse, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import MODEL_DIR, DEVICE, SUBMISSION_PATH
warnings.filterwarnings("ignore")

# GP最优权重 (NN ensemble)
GP_WEIGHTS = {
    "g791_f1": 0.261, "g791_f2": 0.000, "g791_f3": 0.128, "g791_f4": 0.261,
    "seed42_f1": 0.032, "seed42_f2": 0.000, "seed42_f3": 0.000, "seed42_f4": 0.248,
    "g787_f1": 0.000, "g787_f2": 0.000, "g787_f3": 0.000, "g787_f4": 0.070,
}
NEW_FEATS = {'gap_ratio', 'close_pos', 'intraday_range', 'streak_up', 'streak_down', 'vol_ret_corr_10d'}
EXCLUDE = {'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
           'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM', 'pbMRQ',
           'market_ret', 'stock_id', 'vol_ma5', 'vol_ma20',
           'turn_ma5', 'turn_ma20', 'industry', 'amplitude', 'change',
           'roe_ttm', 'np_margin', 'gp_margin', 'eps_ttm',
           'current_ratio', 'debt_to_asset', 'profit_yoy', 'equity_yoy', 'nn_score'}


# =============================================================================
# LightGBM Ranker
# =============================================================================

def build_lgbm_data(panel):
    feature_cols = [c for c in panel.columns if c not in EXCLUDE and c not in ('date', 'stock_id', 'index_name')]
    df = panel.reset_index()[['date', 'stock_id', 'open'] + feature_cols].copy()
    df = df.sort_values(['date', 'stock_id'])
    df['target'] = df.groupby('stock_id')['open'].transform(lambda x: x.shift(-4) / x - 1)
    df = df.dropna(subset=['target']).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df['rank_label'] = df.groupby('date')['target'].transform(
        lambda x: pd.qcut(x.rank(method='first'), q=10, labels=False, duplicates='drop'))
    for col in feature_cols:
        df[col] = df[col].fillna(df.groupby('date')[col].transform('median'))
    return df, feature_cols


def compute_rankic(pred_df, pred_col='lgb_score', target_col='target'):
    """截面 RankIC：Spearman 相关系数按日平均。"""
    ics = []
    for _, day in pred_df.groupby('date'):
        if day[pred_col].nunique() <= 1 or day[target_col].nunique() <= 1:
            continue
        ic = day[pred_col].corr(day[target_col], method='spearman')
        if pd.notna(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def _pct_rank_per_group(scores, groups):
    """按 group 计算百分位排名 [0,1]，用于 DoubleEnsemble 误差加权。"""
    ranks = np.empty(len(scores), dtype=np.float32)
    start = 0
    for g in groups:
        end = start + g
        s = scores[start:end]
        ranks[start:end] = (s.argsort().argsort().astype(float) + 1) / max(g, 1)
        start = end
    return ranks


def train_lgbm_ranker(df, feature_cols, model_path=None,
                      double_ensemble: bool = False,
                      de_n_models: int = 6,
                      de_sub_feature_ratio: float = 0.7):
    import lightgbm as lgb
    if model_path and model_path.exists() and not double_ensemble:
        model = lgb.Booster(model_file=str(model_path))
        print("  Loaded cached LGBM model")
        return model, df[feature_cols].mean(), df[feature_cols].std().replace(0, 1), None

    dates = sorted(df['date'].unique())
    train_df = df[df['date'].isin(dates[:-5])].copy()
    val_dates = dates[-5:]
    val_df = df[df['date'].isin(val_dates)].copy()
    fm = train_df[feature_cols].mean()
    fs = train_df[feature_cols].std().replace(0, 1)
    X_tr = (train_df[feature_cols].values - fm.values) / fs.values
    y_tr = train_df['rank_label'].values.astype(int)
    grp = train_df.groupby('date').size().values
    print(f"  Training LGBM ({len(train_df)} rows, {len(grp)} dates)...")

    # ── M0: 全特征训练 ──
    print(f"  [DE] M0: {len(feature_cols)} features, uniform weights")
    model = lgb.LGBMRanker(
        objective='lambdarank', boosting_type='gbdt',
        n_estimators=500, num_leaves=63, learning_rate=0.05,
        min_child_samples=20, reg_lambda=0.1, reg_alpha=0.1,
        subsample=0.8, colsample_bytree=0.8,
        label_gain=[i for i in range(10)], verbose=-1, random_state=42)
    model.fit(X_tr, y_tr, group=grp, eval_metric=['ndcg'], callbacks=[lgb.log_evaluation(0)])

    # ── DoubleEnsemble: 70% 特征子采样 + M0 误差加权 ──
    de_models = [model]  # M0 is first
    if double_ensemble and de_n_models > 1:
        # 获取 M0 在训练集上的预测
        m0_train_pred = model.predict(X_tr)
        train_grp_sizes = grp
        m0_ranks = _pct_rank_per_group(m0_train_pred, train_grp_sizes)
        # 训练 label 的百分位 (rank_label 是 0-9 分位数)
        label_ranks = np.clip(y_tr.astype(float) / 9.0, 0, 1)
        # 排序误差 = |M0百分位 - label百分位|
        errors = np.abs(m0_ranks - label_ranks) + 1e-6

        n_feats = len(feature_cols)
        rng = np.random.RandomState(42)

        for i in range(1, de_n_models):
            # 70% 随机特征子集
            k = max(1, int(n_feats * de_sub_feature_ratio))
            feat_idx = rng.choice(n_feats, size=k, replace=False)
            sub_feats = [feature_cols[j] for j in sorted(feat_idx)]

            # 按 M0 误差重加权样本
            sample_w = (errors / errors.mean()).astype(np.float32)

            # 子集数据
            X_sub = (train_df[sub_feats].values - fm[sub_feats].values) / fs[sub_feats].values
            grp_sub = grp  # group 不变（同一批日期）

            print(f"  [DE] M{i}: {len(sub_feats)} features, error-weighted")
            sub_model = lgb.LGBMRanker(
                objective='lambdarank', boosting_type='gbdt',
                n_estimators=500, num_leaves=63, learning_rate=0.05,
                min_child_samples=20, reg_lambda=0.1, reg_alpha=0.1,
                subsample=0.8, colsample_bytree=0.8,
                label_gain=[i for i in range(10)], verbose=-1, random_state=42 + i)
            sub_model.fit(X_sub, y_tr, group=grp_sub,
                          sample_weight=sample_w,
                          eval_metric=['ndcg'], callbacks=[lgb.log_evaluation(0)])
            de_models.append((sub_model, feat_idx))

        print(f"  [DE] Trained {len(de_models)} models ({de_n_models-1} subsampled + 1 full)")

    # 验证集 RankIC (用 M0 或 DE 融合分数)
    if double_ensemble and len(de_models) > 1:
        val_X = ((val_df[feature_cols].values - fm.values) / fs.values).astype(np.float32)
        val_grp_sizes = val_df.groupby('date').size().values
        val_ranks = np.zeros(len(val_df))
        val_ranks += _pct_rank_per_group(model.predict(val_X), val_grp_sizes)  # M0
        for sub_model, feat_idx in de_models[1:]:
            sub_feats = [feature_cols[j] for j in sorted(feat_idx)]
            val_X_sub = ((val_df[sub_feats].values - fm[sub_feats].values) / fs[sub_feats].values).astype(np.float32)
            val_ranks += _pct_rank_per_group(sub_model.predict(val_X_sub), val_grp_sizes)
        val_pred = val_ranks / len(de_models)
    else:
        val_pred = model.predict(((val_df[feature_cols].values - fm.values) / fs.values).astype(np.float32))

    val_df = val_df.copy()
    val_df['lgb_score'] = val_pred
    rankic = compute_rankic(val_df, 'lgb_score', 'target')
    print(f"  Val RankIC (last 5 days): {rankic:.4f}")

    if model_path and not double_ensemble:
        model.booster_.save_model(str(model_path))

    if double_ensemble:
        return de_models, fm, fs, rankic, feature_cols
    return model, fm, fs, rankic



def predict_lgbm(model, fm, fs, df, feature_cols, date):
    day = df[df['date'] == date].copy()
    if len(day) == 0: return None
    # 如果是 DoubleEnsemble 模型列表
    if isinstance(model, list):
        n_models = len(model)
        groups = [len(day)]  # 单日，group_size = 全部股票
        rank_sum = np.zeros(len(day), dtype=np.float64)
        for i, item in enumerate(model):
            if i == 0:
                # M0: 全特征
                X = (day[feature_cols].values - fm.values) / fs.values
                rank_sum += _pct_rank_per_group(item.predict(X), groups)
            else:
                sub_model, feat_idx = item
                sub_feats = [feature_cols[j] for j in sorted(feat_idx)]
                X_sub = (day[sub_feats].values - fm[sub_feats].values) / fs[sub_feats].values
                rank_sum += _pct_rank_per_group(sub_model.predict(X_sub), groups)
        day['lgb_score'] = (rank_sum / n_models).astype(np.float32)
    else:
        X = (day[feature_cols].values - fm.values) / fs.values
        day['lgb_score'] = model.predict(X)
    return day


# =============================================================================
# NN 模型加载
# =============================================================================

def load_predict(path, X_np, n_features=None):
    from model import PortfolioPredictor
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    # 优先使用 checkpoint 中的特征数（兼容新旧模型）
    n_features = cfg.get("n_features", n_features or 57)
    model = PortfolioPredictor(n_features=n_features,
        d_model=cfg.get("d_model", 128), n_transformer_layers=cfg.get("n_transformer_layers", 2),
        n_gru_layers=cfg.get("n_gru_layers", 2), d_ff=cfg.get("d_ff", 256),
        use_attention=cfg.get("use_attention", False),
        use_market_gate=cfg.get("use_market_gate", False))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE); model.eval()
    fm = np.array(ckpt["feat_mean"]).reshape(1,1,1,-1)
    fs = np.array(ckpt["feat_std"]).reshape(1,1,1,-1)
    X_n = (X_np.astype(np.float32) - fm) / fs
    X_t = torch.from_numpy(X_n).to(DEVICE, dtype=torch.float32)
    m_t = torch.ones(1, X_np.shape[1], device=DEVICE)
    with torch.no_grad():
        s, _ = model(X_t, m_t)
    return s.squeeze(0).cpu().numpy()


# =============================================================================
# HMM 市场状态 + 动态 MoE 融合
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
    """
    动态混合专家 (MoE): 根据市场状态动态调整 LGBM/NN 融合权重。
    牛市→NN权重高(捕捉趋势)，熊市→LGBM权重高(截面排序稳健)，震荡→均衡。
    """
    if nn_scores is None:
        return lgbm_scores

    regime_id = regime["id"]
    # 牛市: NN 0.4, LGBM 0.6; 熊市: NN 0.15, LGBM 0.85; 震荡: 默认
    if regime_id == 2:  # 牛市
        nn_w = 0.40
    elif regime_id == 0:  # 熊市
        nn_w = 0.15
    else:
        nn_w = 0.30

    lgbm_w = 1.0 - nn_w

    def norm(s):
        mn, mx = s.min(), s.max()
        if mx - mn < 1e-9: return np.zeros_like(s)
        return (s - mn) / (mx - mn)

    blend = lgbm_w * norm(lgbm_scores) + nn_w * norm(nn_scores)
    print(f"  MoE: {regime['name']} → LGBM={lgbm_w:.2f} NN={nn_w:.2f}")
    return blend


# =============================================================================
# 纳什均衡选股
# =============================================================================

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

    # NN embedding 相似度
    from model import PortfolioPredictor
    seed_model = "portfolio_model_g791_v2_fold1.pt" if n_features >= 57 else "portfolio_model_g791_fold1.pt"
    ckpt = torch.load(MODEL_DIR / seed_model, map_location=DEVICE, weights_only=False)
    fm_n = np.array(ckpt["feat_mean"]).reshape(1,1,1,-1)
    fs_n = np.array(ckpt["feat_std"]).reshape(1,1,1,-1)
    X_n = (X_panel.astype(np.float32) - fm_n) / fs_n
    X_t = torch.from_numpy(X_n).to(DEVICE, dtype=torch.float32)
    m_t = torch.ones(1, 300, device=DEVICE)

    model_e = PortfolioPredictor(n_features=n_features,
            d_model=ckpt["config"].get("d_model",128),
            n_transformer_layers=ckpt["config"].get("n_transformer_layers",2),
            n_gru_layers=ckpt["config"].get("n_gru_layers",2),
            d_ff=ckpt["config"].get("d_ff",256),
            use_market_gate=ckpt["config"].get("use_market_gate", False))
    model_e.load_state_dict(ckpt["model_state_dict"])
    model_e.to(DEVICE); model_e.eval()
    with torch.no_grad():
        emb = model_e.encode_stocks(X_t, m_t).squeeze(0).cpu().numpy()
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
# BDC 风格组合构建流水线
# =============================================================================

def bdc_extract_candidate_features(pool_ids, pool_sc, panel):
    """从 panel 中提取候选股的风险特征。"""
    candidates = []
    for sid, sc in zip(pool_ids, pool_sc):
        try:
            sd = panel.xs(sid, level="stock_id")
            vol_20d = sd['pctChg'].iloc[-20:].std() / 100.0
            vol_5d = sd['pctChg'].iloc[-5:].std() / 100.0
            turn = float(sd['turn'].iloc[-1]) if 'turn' in sd.columns else 0.5
            turn_ma5 = float(sd['turn'].iloc[-5:].mean()) if 'turn' in sd.columns else 0.5
            turn_ratio = turn / max(turn_ma5, 1e-9) if turn_ma5 > 1e-9 else 1.0
            ampl = float(sd['amplitude'].iloc[-1]) if 'amplitude' in sd.columns else 1.0
            ampl_ma5 = float(sd['amplitude'].iloc[-5:].mean()) if 'amplitude' in sd.columns else 1.0
            ampl_ratio = ampl / max(ampl_ma5, 1e-9) if ampl_ma5 > 1e-9 else 1.0
            ret_5d = float(sd['pctChg'].iloc[-5:].mean()) / 100.0
            dd = float((sd['close'] / sd['close'].rolling(20).max() - 1.0).iloc[-20:].min())
        except Exception:
            continue
        candidates.append({
            'stock_id': sid, 'score': sc,
            'volatility_20d': vol_20d, 'volatility_5d': vol_5d,
            'turnover_rate': turn, 'turnover_ratio_10d': turn_ratio,
            'amplitude_ratio_5d': ampl_ratio,
            'ret_5d': ret_5d, 'max_dd_20d': dd,
        })
    return candidates


def bdc_risk_filter(cand_df, top_k):
    """百分比位风险过滤 —— 波动率/换手率/振幅百分位阈值，带 Fallback。"""
    df = cand_df.copy()
    vol20_th = df['volatility_20d'].quantile(0.85)
    vol5_th = df['volatility_5d'].quantile(0.85)
    turn_low = df['turnover_rate'].quantile(0.03)
    turn_high = df['turnover_rate'].quantile(0.97)
    turn_ratio_th = df['turnover_ratio_10d'].quantile(0.95)

    filtered = df[
        (df['volatility_20d'] <= vol20_th) &
        (df['volatility_5d'] <= vol5_th) &
        (df['turnover_rate'] >= turn_low) &
        (df['turnover_rate'] <= turn_high) &
        (df['turnover_ratio_10d'] <= turn_ratio_th)
    ].copy()

    # Fallback: 过滤后不足 top_k 则退回全部候选
    if len(filtered) < top_k:
        filtered = df.copy()
        filtered['_fallback'] = True
    else:
        filtered['_fallback'] = False

    return filtered


def bdc_rerank_with_risk(filtered_df, risk_penalty_weight=0.30):
    """风险调整排序：pred_rank - λ × risk_penalty"""
    df = filtered_df.copy()
    df['pred_rank_pct'] = df['score'].rank(pct=True)
    df['risk_vol20_pct'] = df['volatility_20d'].rank(pct=True)
    df['risk_vol5_pct'] = df['volatility_5d'].rank(pct=True)
    df['risk_turnover_pct'] = df['turnover_ratio_10d'].rank(pct=True)
    df['risk_amplitude_pct'] = df['amplitude_ratio_5d'].rank(pct=True)
    df['risk_penalty'] = (
        0.40 * df['risk_vol20_pct'] +
        0.25 * df['risk_vol5_pct'] +
        0.20 * df['risk_turnover_pct'] +
        0.15 * df['risk_amplitude_pct']
    )
    df['selection_score'] = df['pred_rank_pct'] - risk_penalty_weight * df['risk_penalty']
    return df


def bdc_build_weights(selected, top_k, max_single_weight=0.5):
    """预测比例加权 + 单票权重上限 + 超额再分配。"""
    raw = selected['score'].clip(lower=0.0)
    if raw.sum() <= 1e-12:
        raw = pd.Series(1.0, index=selected.index)
    invested = len(selected) / top_k
    weights = invested * raw / raw.sum()

    if max_single_weight < 1.0 and not weights.empty:
        capped = weights.clip(upper=max_single_weight)
        excess = max(0.0, float(weights.sum() - capped.sum()))
        if excess > 1e-12:
            room = (max_single_weight - capped).clip(lower=0.0)
            total_room = float(room.sum())
            if total_room > 1e-12:
                capped += room / total_room * min(excess, total_room)
                capped = capped.clip(upper=max_single_weight)
        weights = capped
    return list(weights)


def bdc_select_and_weight(pool_ids, pool_sc, panel, top_k=5, risk_penalty=0.30,
                           hybrid=False, precision_ids=None, psc=None):
    """完整的 BDC 风格选股管线。返回 (sel_ids, weights) 或 (None, None)。

    hybrid=True 时: 先走旧管线精度门控 → BDC 风险调整排序 → BDC 权重
    hybrid=False: 纯 BDC 管线
    """
    if hybrid and precision_ids is not None and psc is not None:
        # 混合模式: 在精度门控之后做 BDC 风格排序和加权
        if len(precision_ids) < 1:
            return None, None
        candidates = bdc_extract_candidate_features(precision_ids, psc, panel)
    else:
        candidates = bdc_extract_candidate_features(pool_ids, pool_sc, panel)

    if len(candidates) < max(3, min(top_k, 5)):
        return None, None

    cand_df = pd.DataFrame(candidates)
    filtered = bdc_risk_filter(cand_df, top_k)
    scored = bdc_rerank_with_risk(filtered, risk_penalty_weight=risk_penalty)
    selected = scored.nlargest(top_k, 'selection_score')

    weights = bdc_build_weights(selected, top_k)
    return list(selected['stock_id']), weights


# =============================================================================
# 旧版选股（原有逻辑，封装为函数方便对比）
# =============================================================================

def legacy_select_and_weight(pool_ids, pool_sc, pool, valid, scores, stock_ids,
                             panel, args, n_features=57, stock_idx_map=None,
                             X_lt=None):
    """原有的精度门控+波动率过滤+选股权重逻辑。"""
    # ── 精度门控 + 超买过滤 ──
    precision_ids, psc = [], []
    for idx, sid in enumerate(pool_ids):
        scv = pool_sc[idx]
        try:
            sd = panel.xs(sid, level="stock_id")
            ret_5d = sd['pctChg'].iloc[-5:].mean() / 100.0
            dd = (sd['close'] / sd['close'].rolling(20).max() - 1.0).iloc[-20:].min()
        except:
            ret_5d, dd = 0, -0.01

        # 超买过滤 (RSI>75 或 10日涨幅>20%)
        if getattr(args, 'no_overbought', False):
            try:
                rsi_val = float(sd['rsi_14'].iloc[-1]) if 'rsi_14' in sd.columns else 50
                ret_10d = sd['pctChg'].iloc[-10:].sum() / 100.0
                if rsi_val > 75 or ret_10d > 0.20:
                    print(f"    overbought skip {sid}: RSI={rsi_val:.0f} ret_10d={ret_10d:.1%}")
                    continue
            except:
                pass

        if ret_5d > 0 and dd > -0.08:
            precision_ids.append(sid)
            psc.append(scv)

    # ── 纳什均衡（放在门控之后，在约10~15只里做多样化）──
    # 池子太小(<7)跳过；池子适中(7~10)放松阈值；池子够大(>10)正常触发
    if stock_idx_map is not None and X_lt is not None and len(precision_ids) >= 7:
        lam = max(0, min(1, args.game)) if args.game is not None else None
        # 用 X_lt 的实际特征数
        actual_n_features = X_lt.shape[-1] if hasattr(X_lt, 'shape') else n_features
        print(f"  Nash eq (λ={lam:.3f}, gate-pool={len(precision_ids)}, feats={actual_n_features})...")
        try:
            precision_ids, psc = nash_equilibrium_selection(
                precision_ids, psc, stock_idx_map, panel, X_lt, lam,
                n_features=actual_n_features)
        except Exception as e:
            print(f"  [WARN] Nash failed: {e}")

    # 波动率过滤
    pool_vols = []
    for sid in pool_ids:
        try:
            pool_vols.append(panel.xs(sid, level="stock_id")['pctChg'].iloc[-20:].std() / 100.0)
        except:
            pool_vols.append(0.03)
    vol_med = np.median(pool_vols)

    final_ids, fsc = [], []
    for sid, scv in zip(precision_ids, psc):
        try:
            v = panel.xs(sid, level="stock_id")['pctChg'].iloc[-20:].std() / 100.0
        except:
            v = 0.03
        if v <= vol_med * 1.2:
            final_ids.append(sid)
            fsc.append(scv)

    # ── 市场状态门控（20日均线方向） ──
    if getattr(args, 'market_state', False):
        try:
            mid_date = str(panel.index.get_level_values('date').unique()[-20])
            market_20d_ret = panel.xs(panel.index.get_level_values('stock_id').unique()[0], level='stock_id')['pctChg'].iloc[-20:].sum() / 100.0
            market_5d_ret = panel.xs(panel.index.get_level_values('stock_id').unique()[0], level='stock_id')['pctChg'].iloc[-5:].mean() / 100.0
            # 防御板块关键词
            defensive_kw = ['煤炭', '石油', '公用', '银行', '电力', '交运', '钢铁', '建筑']
            offensive_kw = ['半导体', 'AI', '通信', '软件', '计算机', '电子', '传媒', '军工']

            if market_20d_ret < -0.03 and market_5d_ret < 0:
                # 市场进入下行趋势: 仅保留防御板块
                def_filtered_ids, def_filtered_sc = [], []
                for sid, scv in zip(final_ids, fsc):
                    try:
                        ind = panel.xs(sid, level='stock_id')['industry'].iloc[-1] if 'industry' in panel.columns else ''
                        if any(kw in str(ind) for kw in defensive_kw):
                            def_filtered_ids.append(sid)
                            def_filtered_sc.append(scv)
                    except:
                        pass
                if len(def_filtered_ids) >= 2:
                    old_n = len(final_ids)
                    final_ids, fsc = def_filtered_ids, def_filtered_sc
                    print(f"  Market DOWN: switched to defensive ({old_n}→{len(final_ids)} stocks)")
                elif len(def_filtered_ids) >= 1:
                    # 混合防御+原池
                    orig = list(zip(final_ids, fsc))
                    def_set = set(def_filtered_ids)
                    kept = [(s, c) for s, c in orig if s in def_set]
                    for s, c in orig:
                        if len(kept) >= 5: break
                        if s not in def_set:
                            kept.append((s, c))
                    final_ids = [s for s, c in kept]
                    fsc = [c for s, c in kept]
                    print(f"  Market DOWN: mixed defensive+original ({len(final_ids)} stocks)")
        except Exception as e:
            print(f"  [warn] market_state skipped: {e}")

    # ── 行业分散（同行业最多 N 只） ──
    if getattr(args, 'diverse_industry', None) is not None and 'industry' in panel.columns:
        max_per_ind = args.diverse_industry
        sel_ordered = sorted(zip(final_ids, fsc), key=lambda x: -x[1])
        ind_count = {}
        diverse_ids, diverse_sc = [], []
        for sid, scv in sel_ordered:
            try:
                ind = str(panel.xs(sid, level='stock_id')['industry'].iloc[-1])
            except:
                ind = 'Unknown'
            if ind_count.get(ind, 0) < max_per_ind:
                ind_count[ind] = ind_count.get(ind, 0) + 1
                diverse_ids.append(sid)
                diverse_sc.append(scv)
        if len(diverse_ids) >= max(2, min(args.topk, 5) // 2):
            rej = len(final_ids) - len(diverse_ids)
            if rej > 0:
                print(f"  Industry diverse: {rej} replaced (max {max_per_ind}/industry)")
            final_ids, fsc = diverse_ids, diverse_sc

    # 选 top K
    topk = min(args.topk, 5)
    if len(final_ids) >= topk:
        order = np.argsort(fsc)[-topk:]
        sel_ids = [final_ids[i] for i in order]
        sel_sc = [fsc[i] for i in order]
    elif len(final_ids) >= 2:
        sel_ids, sel_sc = final_ids, fsc
    else:
        # fallback
        ti = valid[np.argsort(scores[valid])[-topk:]]
        sel_ids = [stock_ids[i] for i in ti]
        sel_sc = scores[ti]

    sel_sc = np.array(sel_sc)

    # ── 置信度建模（模型集成方差 → 减仓） ──
    if getattr(args, 'confidence', False) and stock_idx_map is not None and X_lt is not None:
        try:
            n_models_avail = sum(1 for w in GP_WEIGHTS.values() if w > 0)
            if n_models_avail >= 4:
                all_preds = []
                for name, w in GP_WEIGHTS.items():
                    if w > 0:
                        seed, fold = name.split("_f")
                        p = load_predict(MODEL_DIR / f"portfolio_model_{seed}_v2_fold{fold}.pt", X_lt, 57)
                        all_preds.append(p)
                all_preds = np.array(all_preds)
                idxs = [stock_idx_map[s] for s in sel_ids]
                var_per_stock = np.var(all_preds[:, idxs], axis=0)
                mean_var = float(np.mean(var_per_stock))
                # 方差 > 0.05 表示模型间分歧大 → 减仓
                if mean_var > 0.05:
                    cash_pct = min(0.3, mean_var * 2)
                    print(f"  Confidence: var={mean_var:.4f} → {cash_pct*100:.0f}% cash")
                    sel_sc = sel_sc * (1.0 - cash_pct)
        except Exception as e:
            print(f"  [warn] confidence: {e}")

    # 加权 (HRP 优先于其他方法)
    if getattr(args, 'hrp', False):
        # 层次风险平价 (HRP): 基于 embedding 聚类树分配权重
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            if stock_idx_map is not None and X_lt is not None:
                from model import PortfolioPredictor
                ckpt = torch.load(MODEL_DIR / "portfolio_model_g791_v2_fold1.pt", map_location='cpu', weights_only=False)
                fm_n = np.array(ckpt["feat_mean"]).reshape(1,1,1,-1)
                fs_n = np.array(ckpt["feat_std"]).reshape(1,1,1,-1)
                X_n = (X_lt.astype(np.float32) - fm_n) / fs_n
                X_t = torch.from_numpy(X_n).to(DEVICE, dtype=torch.float32)
                m_t = torch.ones(1, 300, device=DEVICE)
                model_e = PortfolioPredictor(n_features=X_lt.shape[-1],
                    d_model=ckpt["config"].get("d_model",128),
                    use_market_gate=ckpt["config"].get("use_market_gate", False))
                model_e.load_state_dict(ckpt["model_state_dict"])
                model_e.to(DEVICE); model_e.eval()
                with torch.no_grad():
                    emb = model_e.encode_stocks(X_t, m_t).squeeze(0).cpu().numpy()
                idxs = np.array([stock_idx_map[s] for s in sel_ids])
                e = emb[idxs]
                # 距离矩阵 → 层次聚类 → HRP 权重
                corr = np.corrcoef(e)
                dist = np.sqrt(2 * (1 - np.clip(corr, -1, 1)))
                dist = np.nan_to_num(dist, nan=0.0)
                dist = (dist + dist.T) / 2  # 确保对称
                links = linkage(squareform(dist), method='ward')
                # 沿聚类树逆序分配: 方差大的子簇权重低
                from scipy.cluster.hierarchy import leaves_list
                order = leaves_list(links)
                # 计算波动率倒数作为降序权重
                hrp_vols = []
                for sid in sel_ids:
                    try:
                        v = float(panel.xs(sid, level="stock_id")["pctChg"].std()) / 100.0
                    except:
                        v = 0.03
                    hrp_vols.append(max(v, 0.005))
                inv_vols = np.array([1.0 / v for v in hrp_vols])
                weights = np.zeros(len(sel_ids))
                for i, idx in enumerate(order):
                    weights[idx] = inv_vols[i] if len(inv_vols) > i else 1.0
                weights = weights / weights.sum()
                print(f"  HRP weights: {['{:.3f}'.format(w) for w in weights]}")
            else:
                weights = np.ones(len(sel_ids)) / len(sel_ids)
        except Exception as e:
            print(f"  [warn] HRP failed: {e}, fallback equal")
            weights = np.ones(len(sel_ids)) / len(sel_ids)
    elif args.weight == "equal":
        weights = np.ones(len(sel_ids)) / len(sel_ids)
    elif args.weight == "softmax":
        x = np.array(sel_sc) - max(sel_sc)
        weights = np.exp(x / 0.3) / np.exp(x / 0.3).sum()
    elif args.weight == "inv_vol":
        vols = []
        for sid in sel_ids:
            try:
                v = float(panel.xs(sid, level="stock_id")["pctChg"].std()) / 100.0
            except:
                v = 0.03
            vols.append(max(v, 0.005))
        inv = 1.0 / np.array(vols)
        weights = inv / inv.sum()
    elif args.weight == "bdc":
        # BDC 风格：预测比例加权 + 单票权重上限
        raw = np.clip(np.array(sel_sc, dtype=float), 0, None)
        if raw.sum() <= 1e-12:
            raw = np.ones_like(raw)
        invested = len(sel_ids) / min(topk, 5)
        weights_arr = invested * raw / raw.sum()
        cap = 0.5
        if cap < 1.0:
            capped = np.clip(weights_arr, None, cap)
            excess = max(0.0, float(weights_arr.sum() - capped.sum()))
            if excess > 1e-12:
                room = np.clip(cap - capped, 0, None)
                total_room = float(room.sum())
                if total_room > 1e-12:
                    capped += room / total_room * min(excess, total_room)
                    capped = np.clip(capped, None, cap)
            weights_arr = capped
        weights = list(weights_arr)
    else:
        weights = np.ones(len(sel_ids)) / len(sel_ids)

    return sel_ids, list(weights)


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Controller — LGBM + NN + 纳什")
    parser.add_argument("--ensemble", type=str, default="lgbm-nn",
        choices=["lgbm", "lgbm-nn", "gp", "compare"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--weight", type=str, default="bdc", choices=["equal", "softmax", "inv_vol", "bdc"])
    parser.add_argument("--output", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--game", type=float, nargs="?", const=None, default=None,
                        help="纳什均衡λ (默认: 自动根据拥挤度确定, 如--game 0.25)")
    parser.add_argument("--multi-day", type=int, default=1, choices=[1,2,3,4,5],
                        help="多日 Rolling 预测")
    parser.add_argument("--cash-buffer", type=float, default=None,
                        help="自适应现金仓位(默认关闭)")
    parser.add_argument("--center", action="store_true",
                        help="零均值居中")
    parser.add_argument("--no-overbought", action="store_true",
                        help="超买过滤: RSI>75或10日涨幅超20pct排除")
    parser.add_argument("--diverse-industry", type=int, default=None,
                        help="行业分散: 同行业最多N只")
    parser.add_argument("--smart", action="store_true",
                        help="智能选股: 板块轮动+拥挤度+顶点分析+宏观过滤")
    parser.add_argument("--max-score", action="store_true",
                        help="极限分数模式: softmax+无过滤器+game智能调节")
    parser.add_argument("--moe", action="store_true",
                        help="动态MoE融合: HMM市场状态自适应LGBM/NN权重")
    parser.add_argument("--confidence", action="store_true",
                        help="置信度建模: 模型集成方差高时自动减仓")
    parser.add_argument("--hrp", action="store_true",
                        help="层次风险平价: 基于NN embedding聚类分配权重")

    args = parser.parse_args()

    from data_loader import get_official_stock_ids, build_panel_from_official
    from features import engineer_features, make_window_samples
    from features_alpha158 import add_alpha158_features, ALPHA158_NEW_COLS
    from score_self import calculate_predict_weight_score

    print("=" * 60)
    stock_ids = get_official_stock_ids()
    panel = build_panel_from_official(stock_ids)
    panel = engineer_features(panel, stock_ids)
    panel = add_alpha158_features(panel)
    stock_idx_map = {s: i for i, s in enumerate(stock_ids)}

    use_v2 = args.ensemble in ("gp", "lgbm-nn", "compare")
    is_lgbm = args.ensemble in ("lgbm", "lgbm-nn", "compare")

    # LightGBM 分数
    if is_lgbm:
        print("Building LightGBM data...")
        lgbm_df, feat_cols = build_lgbm_data(panel)
        latest_date = sorted(lgbm_df['date'].unique())[-1]
        model_path = MODEL_DIR / "lgbm_ranker.txt"
        if args.no_cache and model_path.exists(): model_path.unlink()
        lgbm_result = train_lgbm_ranker(lgbm_df, feat_cols, model_path)
        if False:  # double_ensemble disabled
            lgbm_model, lgbm_fm, lgbm_fs, lgbm_rankic, _ = lgbm_result
        else:
            lgbm_model, lgbm_fm, lgbm_fs, lgbm_rankic = lgbm_result
        lgbm_day = predict_lgbm(lgbm_model, lgbm_fm, lgbm_fs, lgbm_df, feat_cols, latest_date)
        lgbm_scores = lgbm_day.set_index('stock_id')['lgb_score'].to_dict()

    # NN 面板
    if use_v2:
        X_all, _, mask, dates = make_window_samples(panel, stock_ids, normalize=False)
    else:
        drop = [c for c in NEW_FEATS if c in panel.columns]
        p2 = panel.drop(columns=drop) if drop else panel
        X_all, _, mask, dates = make_window_samples(p2, stock_ids, normalize=False)

    X_lt = X_all[-1:]; m_lt = mask[-1]
    print(f"  Features: {X_all.shape[-1]}, Latest: {dates[-1]}")

    # NN 分数计算
    nn_scores = None
    if args.ensemble == "gp":
        scores = np.zeros(300)
        print("  GP v2 ensemble (12 NN)...")
        for name, w in GP_WEIGHTS.items():
            if w > 0:
                seed, fold = name.split("_f")
                scores += w * load_predict(MODEL_DIR / f"portfolio_model_{seed}_v2_fold{fold}.pt", X_lt, 57)
        nn_scores = scores
    elif args.ensemble == "lgbm-nn":
        n_days = max(1, min(args.multi_day, len(dates)))
        day_weights = {1: [1.0], 2: [0.6, 0.4], 3: [0.5, 0.3, 0.2],
                       4: [0.4, 0.3, 0.2, 0.1], 5: [0.4, 0.25, 0.15, 0.12, 0.08]}
        dw = np.array(day_weights.get(n_days, [1.0])[:n_days])
        dw = dw / dw.sum()
        def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
        ensemble = np.zeros(300)
        lgbm_available_dates = sorted(lgbm_df['date'].unique())
        for i in range(n_days):
            # LGBM 用 lgbm_df 的实际日期（dates 可能含无 target 的未来日）
            lgbm_date = lgbm_available_dates[-1 - i] if len(lgbm_available_dates) > i else lgbm_available_dates[0]
            date_str = dates[-1 - i]
            # NN 分数
            X_day = X_all[-1-i:-i] if i > 0 else X_all[-1:]
            gp_day = np.zeros(300)
            for name, w in GP_WEIGHTS.items():
                if w > 0:
                    seed, fold = name.split("_f")
                    gp_day += w * load_predict(MODEL_DIR / f"portfolio_model_{seed}_v2_fold{fold}.pt", X_day, 57)
            lgbm_day = predict_lgbm(lgbm_model, lgbm_fm, lgbm_fs, lgbm_df, feat_cols, lgbm_date)
            lgbm_day_arr = np.zeros(300)
            if lgbm_day is not None:
                lgbm_dict = lgbm_day.set_index('stock_id')['lgb_score'].to_dict()
                lgbm_day_arr = np.array([lgbm_dict.get(s, 0) for s in stock_ids])
            # 融合 (支持动态 MoE)
            if args.moe:
                regime = compute_market_regime(panel)
                blend = dynamic_moe_blend(lgbm_day_arr, gp_day, regime)
            else:
                blend = 0.7 * norm(lgbm_day_arr) + 0.3 * norm(gp_day)
            ensemble += dw[i] * blend
            if n_days > 1:
                print(f"    Day {i+1}({date_str}): w={dw[i]:.2f}")
        nn_scores = ensemble
        if n_days > 1:
            print(f"  {n_days}-day rolling ensemble (weights={dict(enumerate(dw))})")
        elif not args.moe:
            print("  LightGBM(0.7) + NN GP(0.3) blend")

    # Compare 模式
    if args.ensemble == "compare":
        print("\n" + "=" * 60); print("COMPARISON"); print("=" * 60)
        df_test = panel.reset_index()[['date', 'stock_id', 'open']]
        last_dates = sorted(df_test['date'].unique())[-5:]
        test_data = df_test[df_test['date'].isin(last_dates)].copy()
        test_data = test_data.rename(columns={'stock_id': '股票代码'})
        test_data['股票代码'] = test_data['股票代码'].astype(str).str.zfill(6)
        def eval_top5(sd, label):
            ids = sorted(sd, key=sd.get, reverse=True)[:args.topk]
            w = np.ones(len(ids))/len(ids)
            out = pd.DataFrame({"股票代码": ids, "权重": w})
            out['股票代码'] = out['股票代码'].astype(str).str.zfill(6)
            sc = calculate_predict_weight_score(out, test_data)
            print(f"  {label:40s}: {sc:.6f}")

        eval_top5(lgbm_scores, "LightGBM Ranker")
        gp_scores = np.zeros(300)
        for name, w in GP_WEIGHTS.items():
            if w > 0:
                seed, fold = name.split("_f")
                gp_scores += w * load_predict(MODEL_DIR / f"portfolio_model_{seed}_v2_fold{fold}.pt", X_lt, 57)
        eval_top5({s: gp_scores[i] for i, s in enumerate(stock_ids)}, "NN GP v2")
        lgbm_arr = np.array([lgbm_scores.get(s, 0) for s in stock_ids])
        def norm2(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
        blend = 0.7 * norm2(lgbm_arr) + 0.3 * norm2(gp_scores)
        eval_top5({s: blend[i] for i, s in enumerate(stock_ids)}, "LGBM(0.7)+NN(0.3)")
        # old models
        drop = [c for c in NEW_FEATS if c in panel.columns]
        p_old = panel.drop(columns=drop) if drop else panel
        X_old, _, _, _ = make_window_samples(p_old, stock_ids, normalize=False)
        X_ol = X_old[-1:]
        s_f1 = load_predict(MODEL_DIR / "portfolio_model_g791_fold1.pt", X_ol, 51)
        eval_top5({s: s_f1[i] for i, s in enumerate(stock_ids)}, "old fold1")
        s_g = load_predict(MODEL_DIR / "portfolio_model_g791.pt", X_ol, 51)
        s_s = load_predict(MODEL_DIR / "portfolio_model_seed42.pt", X_ol, 51)
        s_h = load_predict(MODEL_DIR / "portfolio_model_g787.pt", X_ol, 51)
        o3 = 0.6*s_g + 0.15*s_s + 0.25*s_h
        eval_top5({s: o3[i] for i, s in enumerate(stock_ids)}, "old 3model")
        return

    # ── 最终分数 ──
    if nn_scores is not None:
        scores = nn_scores
    elif is_lgbm:
        scores = np.array([lgbm_scores.get(s, -999) for s in stock_ids])
    else:
        print("[ERROR] No scores"); return

    # ── 零均值居中（选股前消除系统性偏差） ──
    if args.center:
        valid_center = np.where(m_lt > 0.5)[0]
        if len(valid_center) > 0:
            scores = scores.copy()
            scores[valid_center] = scores[valid_center] - scores[valid_center].mean()
            print(f"  Zero-sum centering applied (mean before: {scores[valid_center].mean():.6f})")

    # ── 极限分数模式: 宏观+赛道拥挤度+顶点博弈 ──
    if getattr(args, 'max_score', False):
        print(f"\n  {'='*50}")
        print(f"  极限分数模式 — 市场分析 + 博弈论")
        print(f"  {'='*50}")
        try:
            sample_stock = stock_ids[0]
            sd = panel.xs(sample_stock, level="stock_id")
            market_20d = sd['pctChg'].iloc[-20:].sum() / 100.0
            market_5d = sd['pctChg'].iloc[-5:].mean() / 100.0
            regime = "牛市" if market_20d > 0.03 and market_5d > 0 else "震荡" if market_20d > -0.03 else "熊市"
            print(f"  宏观: {regime} (20d={market_20d:+.1%}, 5d={market_5d:+.1%})")
        except:
            pass
        # 赛道拥挤度
        if 'industry' in panel.columns:
            ind_counts = {}
            for sid in pool_ids if 'pool_ids' in dir() else []:
                try:
                    ind = str(panel.xs(sid, level='stock_id')['industry'].iloc[-1])
                    ind_counts[ind] = ind_counts.get(ind, 0) + 1
                except:
                    pass
            if ind_counts:
                top_sectors = sorted(ind_counts.items(), key=lambda x: -x[1])[:5]
                print(f"  赛道拥挤(top40):")
                for ind, cnt in top_sectors:
                    pct = cnt / len(pool_ids) * 100
                    print(f"    {'⚠' if pct > 25 else ' '} {ind}: {cnt}只({pct:.0f}%)")
                # 超拥挤→自动加强博弈
                if any(cnt/len(pool_ids) > 0.25 for _, cnt in top_sectors):
                    if args.game is not None and args.game < 0.5:
                        args.game = min(0.5, args.game * 1.5)
                        print(f"  博弈升级: λ→{args.game:.2f}(赛道拥挤)")
        print()

    # ── 候选池 ──
    valid = np.where(m_lt > 0.5)[0]
    pool = valid[np.argsort(scores[valid])[-40:]]
    pool_ids = [stock_ids[i] for i in pool]
    pool_sc = [scores[i] for i in pool]

    # ── BDC 管线 vs 旧管线 ──
    if False:  # BDC 管线已移除
        if args.bdc == "hybrid":
            print("  Using HYBRID pipeline (legacy precision gate + BDC rerank + BDC weights)...")
            # 先做旧管线的精度门控
            precision_ids, psc = [], []
            for idx, sid in enumerate(pool_ids):
                scv = pool_sc[idx]
                try:
                    sd = panel.xs(sid, level="stock_id")
                    ret_5d = sd['pctChg'].iloc[-5:].mean() / 100.0
                    dd = (sd['close'] / sd['close'].rolling(20).max() - 1.0).iloc[-20:].min()
                except:
                    ret_5d, dd = 0, -0.01
                if ret_5d > 0 and dd > -0.08:
                    precision_ids.append(sid)
                    psc.append(scv)
            print(f"    Gate: {len(pool_ids)}→{len(precision_ids)} passed momentum/drawdown filter")
            sel_ids, weights = bdc_select_and_weight(
                pool_ids, pool_sc, panel,
                top_k=min(args.topk, 5), risk_penalty=args.risk_penalty,
                hybrid=True, precision_ids=precision_ids, psc=psc,
            )
            if sel_ids is None or len(sel_ids) < 1:
                print("  [WARN] Hybrid pipeline returned no candidates, falling back to legacy")
                sel_ids, weights = legacy_select_and_weight(
                    pool_ids, pool_sc, pool, valid, scores, stock_ids, panel, args,
                    stock_idx_map=stock_idx_map, X_lt=X_lt)
        else:
            print("  Using BDC-style pipeline (risk-filter + risk-adjusted rerank + pred-weights)...")
            sel_ids, weights = bdc_select_and_weight(
                pool_ids, pool_sc, panel,
                top_k=min(args.topk, 5), risk_penalty=args.risk_penalty,
            )
            if sel_ids is None or len(sel_ids) < 1:
                print("  [WARN] BDC pipeline returned no candidates, falling back to legacy")
                legacy_ids, legacy_weights = legacy_select_and_weight(
                    pool_ids, pool_sc, pool, valid, scores, stock_ids, panel, args,
                    stock_idx_map=stock_idx_map, X_lt=X_lt)
                sel_ids, weights = legacy_ids, legacy_weights
    else:
        sel_ids, weights = legacy_select_and_weight(
            pool_ids, pool_sc, pool, valid, scores, stock_ids, panel, args,
            stock_idx_map=stock_idx_map, X_lt=X_lt)

    # ── 自适应现金仓位 ──（选股后、输出前）
    if args.cash_buffer is not None and args.cash_buffer > 0 and len(weights) > 0:
        try:
            top_scores = scores[valid]
            n_compare = min(10, len(top_scores))
            if len(top_scores) >= n_compare:
                signal_strength = float(top_scores[-1] - top_scores[-n_compare])
            elif len(top_scores) >= 5:
                signal_strength = float(top_scores[-1] - top_scores[-5])
            else:
                signal_strength = float(top_scores[-1] - top_scores[len(top_scores)//2])

            buffer_threshold = args.cash_buffer * float(np.std(top_scores)) if len(top_scores) > 1 else 0.05
            cash_pct = 0.0
            if signal_strength < buffer_threshold:
                cash_pct = 0.15
            if signal_strength < buffer_threshold * 0.5:
                cash_pct = 0.30

            if cash_pct > 0:
                weights = [w * (1.0 - cash_pct) for w in weights]
                print(f"  Cash buffer: {cash_pct*100:.0f}% (strength={signal_strength:.6f})")
        except Exception as e:
            print(f"  [warn] cash_buffer skipped: {e}")

    # ── 输出 ──
    print(f"\nSelected {len(sel_ids)} stocks:")
    for sid, w in zip(sel_ids, weights):
        try:
            sd = panel.xs(sid, level="stock_id")
            mom = sd['pctChg'].iloc[-5:].mean()
            vol = sd['pctChg'].iloc[-20:].std()
        except:
            mom, vol = 0, 3
        print(f"  {sid}: w={w:.3f} mom={mom:+.2f}% vol={vol:.1f}%")

    df_out = pd.DataFrame({"stock_id": [str(s).zfill(6) for s in sel_ids], "weight": weights})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    import subprocess
    r = subprocess.run(["python", "score_self.py"], capture_output=True, text=True)
    for line in r.stdout.strip().split("\n"):
        if "score" in line.lower() or "得分" in line:
            print(f"  {line}")
    try:
        score = float(r.stdout.strip().split("\n")[-1].split(": ")[-1])
    except: score = -999
    w_str = ", ".join([f"{w:.3f}" for w in weights])
    label = f"{args.ensemble} game={args.game}" if args.game is not None else args.ensemble
    print(f"\n  SCORE = {score:.6f}  [{w_str}]  {label}")

    # ── 自动记录日志 ──
    _log_path = Path(args.output).parent / "run_log.csv"
    _log_exists = _log_path.exists()
    import csv
    try:
        with open(_log_path, "a", newline="") as _f:
            _w = csv.writer(_f)
            if not _log_exists:
                _w.writerow(["timestamp","score","ensemble","weight","game",
                             "multi_day","stocks","n"])
            _w.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{score:.6f}", args.ensemble, args.weight,
                args.game if args.game is not None else "",
                getattr(args, 'multi_day', 1),
                " ".join([str(s) for s in sel_ids]),
                len(sel_ids),
            ])
        print(f"  [log] {_log_path.name}")
    except Exception as _e:
        print(f"  [log] warn: {_e}")

    # ── BDC vs 旧管线对比 ── (已移除)
    if False:
        bdc_label = {"pure": "BDC纯管线", "hybrid": "BDC混合管线"}.get(args.bdc, "BDC管线")
        print(f"\n  --- 自动对比: {bdc_label} vs 旧管线 ---")
        import shutil, os
        bdc_result_path = args.output
        legacy_result_path = str(Path(args.output).parent / "result_legacy_compare.csv")

        # 计算旧管线结果
        legacy_ids, legacy_weights = legacy_select_and_weight(
            pool_ids, pool_sc, pool, valid, scores, stock_ids, panel, args,
            stock_idx_map=stock_idx_map, X_lt=X_lt)
        legacy_out = pd.DataFrame({"股票代码": [str(s).zfill(6) for s in legacy_ids], "权重": legacy_weights})
        legacy_out.to_csv(legacy_result_path, index=False)

        # 用旧管线结果替换、算分、恢复
        shutil.copy2(bdc_result_path, bdc_result_path + ".bdc_backup")
        shutil.copy2(legacy_result_path, bdc_result_path)
        r2 = subprocess.run(["python", "score_self.py"], capture_output=True, text=True)
        try:
            legacy_score = float(r2.stdout.strip().split("\n")[-1].split(": ")[-1])
        except:
            legacy_score = -999
        shutil.copy2(bdc_result_path + ".bdc_backup", bdc_result_path)
        os.remove(bdc_result_path + ".bdc_backup")
        os.remove(legacy_result_path)

        lw_str = ", ".join([f"{w:.3f}" for w in legacy_weights])
        delta = score - legacy_score
        sign = "+" if delta > 0 else ""
        winner = "✅ BDC 更好" if delta > 0 else "⚠️ 旧管线更好" if delta < 0 else "➡️ 持平"
        print(f"  ╔══════════════════════════════════════════════╗")
        print(f"  ║  BDC  管线: {score:.6f}  [{w_str}]")
        print(f"  ║  旧管线:   {legacy_score:.6f}  [{lw_str}]")
        print(f"  ║  Δ = {sign}{delta:.6f}   {winner}")
        print(f"  ╚══════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
