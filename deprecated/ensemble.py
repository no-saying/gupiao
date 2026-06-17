"""集成模块 — LGBM/CatBoost/NN 训练 + 预测 + 融合"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from core.utils import _resolve_model_path, _norm
from config import (MODEL_DIR, DEVICE, NN_GP_SHARPE_BLEND, AUX_MODEL_WEIGHTS,
    LGBM_CATBOOST_BLEND, LGBM_GAP, LGBM_VAL_N, LGBM_PURGE_DAYS, LGBM_PURGE_WINDOW,
    LGBM_N_FOLDS, LGBM_MIN_TREES, LGBM_BLIND_TREES_FALLBACK,
    LGBM_LEAVES, LGBM_LR, LGBM_MIN_CHILD, LGBM_REG_LAMBDA, LGBM_REG_ALPHA,
    LGBM_SUBSAMPLE, LGBM_COLSAMPLE, LGBM_TIME_DECAY_HALF, LGBM_NN_BASE,
    DE_N_MODELS, DE_SUB_FEATURE_RATIO, DE_KF_N_ESTIMATORS, DE_KF_LEAVES, DE_KF_LR, DE_KF_EARLY_STOP,
    CATBOOST_ITERATIONS, CATBOOST_LR, CATBOOST_DEPTH, CATBOOST_EARLY_STOP, CATBOOST_CAT_FEATURES)

GP_WEIGHTS = {
    "g791_f1": 0.33, "g791_f2": 0.00, "g791_f3": 0.17, "g791_f4": 0.33,
    "seed42_f1": 0.00, "seed42_f2": 0.00, "seed42_f3": 0.00, "seed42_f4": 0.17,
}
# Sharpe 损失模型 — 保留12个提供多样性（权重低但独立优化目标不同）
SHARPE_WEIGHTS = {
    "g791_1": 0.083, "g791_2": 0.083, "g791_3": 0.083, "g791_4": 0.083,
    "seed42_1": 0.083, "seed42_2": 0.083, "seed42_3": 0.083, "seed42_4": 0.083,
}
# GP + Sharpe 等权融合
NN_BLEND_GP_W = NN_GP_SHARPE_BLEND[0]     # listnet 模型权重
NN_BLEND_SHARPE_W = NN_GP_SHARPE_BLEND[1]  # sharpe 模型权重
# 辅助模型 (pinball=极端涨幅, top1=最强股, focal=top5%概率)
AUX_MODELS = AUX_MODEL_WEIGHTS.copy()
# 短窗口"当前市场专家" — 3种子 pinball + 1 focal 等权
SHORT_WIN_MODELS = {
    "sw60_pinball": 1.0,         # seed 42, Sharpe 2.62
    "sw60_pinball_s791": 1.0,    # seed 791, Sharpe 1.80
    "sw60_pinball_s787": 1.0,    # seed 787, Sharpe 2.46
    "sw60_focal": 0.6,           # seed 791, Sharpe 1.39
}
AUX_MODEL_DIR = MODEL_DIR

def _load_nn_ensemble(X, n_features=None, include_sharpe=True, version="v3"):
    if n_features is None:
        n_features = X.shape[-1]
    """加载 GP (listnet) + Sharpe 模型，返回融合后的 NN 分数。

    版本自动选择：优先加载 v3 模型，不存在则回落 v2。
    """
    gp = np.zeros(300)
    for name, w in GP_WEIGHTS.items():
        if w > 0:
            seed, fold = name.split("_f")
            path = _resolve_model_path(seed, version, fold)
            if path.exists():
                gp += w * load_predict(path, X, n_features)
    if not include_sharpe:
        return gp
    sp = np.zeros(300)
    for name, w in SHARPE_WEIGHTS.items():
        if w > 0:
            seed, fold = name.split("_")
            path = _resolve_model_path(seed, "sharpe", fold)
            if path.exists():
                sp += w * load_predict(path, X, n_features)
    # PCC + Sortino 模型（额外多样性）
    if include_sharpe:
        pcc_path = MODEL_DIR / "portfolio_model_pcc.pt"
        sortino_path = MODEL_DIR / "portfolio_model_sortino_s101.pt"
        pc = np.zeros(300)
        st = np.zeros(300)
        has_pcc = False; has_sortino = False
        if pcc_path.exists():
            try:
                pc = load_predict(pcc_path, X, n_features); has_pcc = True
            except Exception:
                pass
        if sortino_path.exists():
            try:
                st = load_predict(sortino_path, X, n_features); has_sortino = True
            except Exception:
                pass
        if has_pcc and has_sortino:
            return 0.44 * _norm(gp) + 0.44 * _norm(sp) + 0.06 * _norm(pc) + 0.06 * _norm(st)
        elif has_pcc:
            return 0.46 * _norm(gp) + 0.46 * _norm(sp) + 0.08 * _norm(pc)
        elif has_sortino:
            return 0.47 * _norm(gp) + 0.47 * _norm(sp) + 0.06 * _norm(st)
    return NN_BLEND_GP_W * _norm(gp) + NN_BLEND_SHARPE_W * _norm(sp)


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

def build_lgbm_data(panel):
    feature_cols = [c for c in panel.columns if c not in EXCLUDE and c not in ('date', 'stock_id', 'index_name')]
    df = panel.reset_index()[['date', 'stock_id', 'open'] + feature_cols].copy()
    df = df.sort_values(['date', 'stock_id'])
    # 用 adj_factor 修正收益率计算（处理分红/拆股导致的复权断裂）
    # v7 修复：买入价从 open[T] 改为 open[T+1]，消灭 T 日数据穿越
    # 旧: target = open[T+4]/open[T] - 1  ← 模型知道T日收盘后才"买入"T日开盘
    # 新: target = open[T+5]/open[T+1] - 1 ← 模型只能以T+1开盘买入
    if 'adj_factor' in df.columns:
        df['adj_factor'] = df['adj_factor'].fillna(1.0)
        df['open_adj'] = df['open'] * df['adj_factor']
        df['target'] = df.groupby('stock_id')['open_adj'].transform(
            lambda x: x.shift(-5) / x.shift(-1) - 1)
        df = df.drop(columns=['open_adj'])
    else:
        df['target'] = df.groupby('stock_id')['open'].transform(
            lambda x: x.shift(-5) / x.shift(-1) - 1)
    df = df.dropna(subset=['target']).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    # 极端 target 由截面 Rank 化自动处理（rank 对异常值天然鲁棒），不额外裁剪

    # ── v7: 特征截面 Rank 化（去牛熊偏差）──
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
                and c not in ('open', 'adj_factor', 'target', 'ret_1d', 'vol_20d')]
    for col in num_cols:
        df[col] = df.groupby('date')[col].rank(pct=True)
        df[col] = df[col].fillna(0.5)
    feature_cols = num_cols
    print(f"  [v7] Cross-sectional ranked {len(feature_cols)} features")

    # ── v7.3: 风险调整标签 ──
    # 计算过去 20 日波动率，target = 未来收益 / 波动率（夏普化）
    # ── V9: 行业中性化 Target ──
    # 1. 去极值 [-15%, +20%]
    df['target'] = df['target'].clip(-0.15, 0.20)
    # 2. 行业内 Z-Score：同行业同天，收益比行业均值强几个标准差
    ind_col = 'l2_name' if 'l2_name' in df.columns else ('industry' if 'industry' in df.columns else None)
    if ind_col:
        df['target'] = df.groupby(['date', ind_col])['target'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8) if len(x) >= 3 else 0.0)
        df['target'] = df['target'].fillna(0).clip(-5, 5)
        print(f"  [v9] Industry-neutral target (by {ind_col}): return in std above industry mean")
    else:
        print(f"  [v9] No industry column, using raw clipped target")

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
    """
    时间序列划分（零泄漏 + Purged K-Fold 稳定性）：
      Train:  [start ... T-25]    绝大部分数据
      Val:    [T-25 ... T-5]      20天验证（够大，不受单周扰动）
      Gap:    [T-5 ... T-1]       5天缓冲
      Predict: T                  最新一天

    Purged K-Fold：在近半年数据上做3折，均值确定最优树数。
    然后全量盲跑（不早停），避免验证集过小导致的保守模式。
    """
    import lightgbm as lgb

    dates = sorted(df['date'].unique())
    n = len(dates)
    GAP = LGBM_GAP
    VAL_N = LGBM_VAL_N

    if n < GAP + VAL_N + 100:
        VAL_N = max(5, n // 10)
        print(f"  [WARN] Not enough data, VAL_N reduced to {VAL_N}")

    train_dates = dates[:n - GAP - VAL_N]
    val_dates = dates[n - GAP - VAL_N:n - GAP]
    gap_dates = dates[n - GAP:]
    pred_date = dates[-1]

    if len(train_dates) < 100:
        raise RuntimeError(f"Not enough training data: {len(train_dates)} dates < 100")

    train_df = df[df['date'].isin(train_dates)].copy()
    val_df = df[df['date'].isin(val_dates)].copy()
    print(f"  Split: train[{train_dates[0].date()}~{train_dates[-1].date()}]={len(train_dates)}d "
          f"| val[{val_dates[0].date()}~{val_dates[-1].date()}]={len(val_dates)}d "
          f"| gap+pred={len(gap_dates)}d (预测日={pred_date.date()})")

    fm = train_df[feature_cols].mean()
    fs = train_df[feature_cols].std().replace(0, 1)
    X_tr = (train_df[feature_cols].values - fm.values) / fs.values
    y_tr = train_df['target'].values.astype(np.float32)
    grp = train_df.groupby('date').size().values
    n_sample = len(train_df)
    print(f"  Training ({n_sample} rows, {len(grp)} dates)...")

    # ── 时间衰减权重 ──
    latest_train = pd.Timestamp(train_dates[-1])
    train_dt = pd.to_datetime(train_df['date'])
    days_ago = (latest_train - train_dt).dt.days.values.astype(float)
    sample_w = np.exp(-np.log(2) * days_ago / LGBM_TIME_DECAY_HALF)
    sample_w = sample_w / sample_w.mean()
    print(f"  Time decay: half_life={LGBM_TIME_DECAY_HALF}d, w=[{sample_w.min():.2f}, {sample_w.max():.2f}]")

    X_es = ((val_df[feature_cols].values - fm.values) / fs.values).astype(np.float32)
    y_es = val_df['target'].values.astype(np.float32)
    grp_es = val_df.groupby('date').size().values
    print(f"  Val set: {len(val_dates)} days, {len(val_df)} rows")

    # ── Purged K-Fold：近半年3折，均值确定最优树数 ──
    # Purged K-Fold
    recent_cutoff = n - min(LGBM_PURGE_WINDOW, n // 2)
    recent_dates = dates[recent_cutoff:n - GAP - VAL_N]
    kf_optimal_trees = []
    if len(recent_dates) >= 80:
        N_FOLDS = LGBM_N_FOLDS
        fold_size = len(recent_dates) // N_FOLDS
        purge = LGBM_PURGE_DAYS
        for k in range(N_FOLDS):
            fold_val_start = recent_cutoff + k * fold_size
            fold_val_end = min(fold_val_start + fold_size, n - GAP - VAL_N - purge)
            fold_train_end = max(0, fold_val_start - purge)
            if fold_val_end - fold_val_start < 10:
                continue
            fold_train = df[df['date'].isin(dates[:fold_train_end])]
            fold_val = df[df['date'].isin(dates[fold_val_start:fold_val_end])]
            if len(fold_train) < 500 or len(fold_val) < 50:
                continue
            fm_k = fold_train[feature_cols].mean()
            fs_k = fold_train[feature_cols].std().replace(0, 1)
            Xk_tr = ((fold_train[feature_cols].values - fm_k.values) / fs_k.values).astype(np.float32)
            yk_tr = fold_train['target'].values.astype(np.float32)
            gk_tr = fold_train.groupby('date').size().values
            Xk_es = ((fold_val[feature_cols].values - fm_k.values) / fs_k.values).astype(np.float32)
            yk_es = fold_val['target'].values.astype(np.float32)
            gk_es = fold_val.groupby('date').size().values
            kf_model = lgb.LGBMRegressor(
                objective='huber', boosting_type='gbdt', alpha=1.5, reg_alpha=1.0, reg_lambda=5.0,
                n_estimators=DE_KF_N_ESTIMATORS, num_leaves=DE_KF_LEAVES, learning_rate=DE_KF_LR,
                min_child_samples=LGBM_MIN_CHILD, subsample=LGBM_SUBSAMPLE, colsample_bytree=LGBM_COLSAMPLE,
                verbose=-1, random_state=42 + k)
            kf_model.fit(Xk_tr, yk_tr, eval_set=[(Xk_es, yk_es)],
                         callbacks=[lgb.early_stopping(DE_KF_EARLY_STOP, verbose=False), lgb.log_evaluation(0)])
            kf_optimal_trees.append(max(50, kf_model.best_iteration_ or 400))
        if kf_optimal_trees:
            BLIND_TREES = max(LGBM_MIN_TREES, int(np.median(kf_optimal_trees)))
            print(f"  [Purged K-Fold] {N_FOLDS} folds, optimal trees: {kf_optimal_trees}, median(保底{LGBM_MIN_TREES})={BLIND_TREES}")
        else:
            BLIND_TREES = LGBM_BLIND_TREES_FALLBACK
    else:
        BLIND_TREES = LGBM_BLIND_TREES_FALLBACK
        print(f"  [Purged K-Fold] Not enough recent data, blind trees={BLIND_TREES}")

    # ── 全量盲跑（不早停，用K-Fold确定的最优树数）──
    print(f"  [M0] {len(feature_cols)} features, blind {BLIND_TREES} trees (Purged K-Fold)")
    model = lgb.LGBMRegressor(
        objective='huber', boosting_type='gbdt', alpha=1.5, reg_alpha=1.0, reg_lambda=5.0,
        n_estimators=BLIND_TREES, num_leaves=68, learning_rate=0.0196,
        min_child_samples=42, reg_lambda=0.0022, reg_alpha=0.0054,
        subsample=0.738, colsample_bytree=0.939,
        verbose=-1, random_state=42)
    model.fit(X_tr, y_tr, sample_weight=sample_w,
              eval_set=[(X_es, y_es)],
              callbacks=[lgb.log_evaluation(0)])
    n_trees = BLIND_TREES
    conservative = n_trees < 250  # 只在极端少时触发
    if conservative:
        print(f"  [LGBM] {n_trees} trees → CONSERVATIVE mode")
    else:
        print(f"  [LGBM] Trained {n_trees} trees (Purged K-Fold blend)")

    # ── DoubleEnsemble: 70% 特征子采样 + M0 误差加权 ──
    de_models = [model]  # M0 is first
    if double_ensemble and de_n_models > 1:
        # 获取 M0 在训练集上的预测
        m0_train_pred = model.predict(X_tr)
        train_grp_sizes = grp
        m0_ranks = _pct_rank_per_group(m0_train_pred, train_grp_sizes)
        # y_tr 是行业中性化连续值，转为 [0,1]
        label_ranks = np.clip((y_tr.astype(float) - y_tr.min()) / (y_tr.max() - y_tr.min() + 1e-8), 0, 1)
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
            sub_model = lgb.LGBMRegressor(
                objective='lambdarank', boosting_type='gbdt', metric='ndcg', eval_at=[5],
                n_estimators=500, num_leaves=68, learning_rate=0.0196,
                min_child_samples=42, reg_lambda=0.0022, reg_alpha=0.0054,
                subsample=0.738, colsample_bytree=0.939,
                verbose=-1, random_state=42 + i)
            sub_model.fit(X_sub, y_tr, sample_weight=sample_w)
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
        return de_models, fm, fs, rankic, feature_cols, conservative
    return model, fm, fs, rankic, conservative


# walk_forward_validate 已移至 validate.py



def predict_lgbm(model, fm, fs, df, feature_cols, date):
    day = df[df['date'] == date].copy()
    if len(day) == 0: return None
    # 如果是 DoubleEnsemble 模型列表
    if isinstance(model, list):
        n_models = len(model)
        score_sum = np.zeros(len(day), dtype=np.float64)
        for i, item in enumerate(model):
            if i == 0:
                # M0: 全特征
                X = (day[feature_cols].values - fm.values) / fs.values
                score_sum += item.predict(X)
            else:
                sub_model, feat_idx = item
                sub_feats = [feature_cols[j] for j in sorted(feat_idx)]
                X_sub = (day[sub_feats].values - fm[sub_feats].values) / fs[sub_feats].values
                score_sum += sub_model.predict(X_sub)
        day['lgb_score'] = (score_sum / n_models).astype(np.float32)
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
        use_market_gate=cfg.get("use_market_gate", False), use_gat=cfg.get("use_gat", False))
    # strict=False: 兼容旧 checkpoint（缺少 score_scale/score_norm 等新字段）
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE); model.eval()
    fm = np.array(ckpt["feat_mean"]).reshape(-1)
    fs = np.array(ckpt["feat_std"]).reshape(-1)
    saved_n = len(fm)
    cur_n = X_np.shape[-1]
    # 对齐：按 checkpoint 中保存的 feature_cols 顺序匹配
    saved_cols = ckpt.get("feature_cols", None)
    if saved_cols is not None and cur_n != saved_n:
        # 尝试按列名对齐（比截断/补零更可靠）
        try:
            import pickle
            from config import PROCESSED_DIR
            with open(PROCESSED_DIR / "samples.pkl", "rb") as f:
                proc = pickle.load(f)
            cur_cols = proc.get("feature_cols", None)
            if cur_cols is not None and len(cur_cols) == cur_n:
                # 找到当前列在 saved 列中的位置
                saved_idx = [saved_cols.index(c) if c in saved_cols else -1
                            for c in cur_cols]
                aligned = np.zeros((*X_np.shape[:-1], saved_n), dtype=np.float32)
                for i, si in enumerate(saved_idx):
                    if si >= 0:
                        aligned[..., si] = X_np[..., i]
                X_np = aligned
            else:
                # 回退：截断或补零
                if cur_n > saved_n:
                    X_np = X_np[..., :saved_n]
                else:
                    X_np = np.pad(X_np, ((0,0),(0,0),(0,0),(0,saved_n-cur_n)))
        except Exception:
            if cur_n > saved_n:
                X_np = X_np[..., :saved_n]
            else:
                X_np = np.pad(X_np, ((0,0),(0,0),(0,0),(0,saved_n-cur_n)))
    fm = fm.reshape(1, 1, 1, -1)
    fs = fs.reshape(1, 1, 1, -1)
    X_n = (X_np.astype(np.float32) - fm) / fs
    X_t = torch.from_numpy(X_n).to(DEVICE, dtype=torch.float32)
    m_t = torch.ones(1, X_np.shape[1], device=DEVICE)
    with torch.no_grad():
        s, _ = model(X_t, m_t)
    return s.squeeze(0).cpu().numpy()


# =============================================================================
# HMM 市场状态 + 动态 MoE 融合
# =============================================================================

def train_catboost_ranker(df, feature_cols, lgbm_scores=None):
    """
    CatBoost Ranker: 利用 SW 行业作为高基数类别特征。
    CatBoost 原生支持 categorical features，处理 312 类 SW L3 行业远超 LGBM。
    与 LGBM 等权融合。
    """
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError:
        print("  [CatBoost] not installed, skip")
        return None

    dates = sorted(df['date'].unique())
    n = len(dates)
    GAP, VAL_N = 5, 20
    train_dates = dates[:n - GAP - VAL_N]
    val_dates = dates[n - GAP - VAL_N:n - GAP]

    train_df = df[df['date'].isin(train_dates)].copy()
    val_df = df[df['date'].isin(val_dates)].copy()

    # 数值特征 + 多类别特征（SW行业+日历，CatBoost原生处理）
    num_feats = [c for c in feature_cols if c not in ('industry', 'industry_l1', 'industry_l3',
                'l1_name', 'l2_name', 'l3_name',
                'wday_0', 'wday_1', 'wday_2', 'wday_3',
                'is_month_end', 'is_cny_before', 'is_cny_after')]
    cat_feats = [c for c in CATBOOST_CAT_FEATURES if c in df.columns]
    if not cat_feats:
        cat_feats = ['l2_name'] if 'l2_name' in df.columns else (['industry'] if 'industry' in df.columns else [])

    # 合并数值和分类特征
    all_feats = num_feats + cat_feats
    X_tr = train_df[all_feats].copy()
    for c in num_feats:
        if c in X_tr.columns: X_tr[c] = X_tr[c].fillna(0)
    for c in cat_feats:
        if c in X_tr.columns:
            if X_tr[c].dtype == np.float32 or X_tr[c].dtype == np.float64:
                X_tr[c] = X_tr[c].fillna(0).astype(int)
            else:
                X_tr[c] = X_tr[c].fillna("未知")
    y_tr = train_df['target'].values.astype(np.float32)
    grp_tr = train_df.groupby('date').size().values

    X_val = val_df[all_feats].copy()
    for c in num_feats:
        if c in X_val.columns: X_val[c] = X_val[c].fillna(0)
    for c in cat_feats:
        if c in X_val.columns:
            if X_val[c].dtype == np.float32 or X_val[c].dtype == np.float64:
                X_val[c] = X_val[c].fillna(0).astype(int)
            else:
                X_val[c] = X_val[c].fillna("未知")
    y_val = val_df['target'].values.astype(np.float32)
    grp_val = val_df.groupby('date').size().values

    print(f"  [CatBoost] Training ({len(X_tr)} rows, {len(all_feats)} features, cat={cat_feats})...")
    train_pool = Pool(X_tr, y_tr, group_id=np.repeat(np.arange(len(grp_tr)), grp_tr),
                       cat_features=cat_feats)
    val_pool = Pool(X_val, y_val, group_id=np.repeat(np.arange(len(grp_val)), grp_val),
                     cat_features=cat_feats)

    model = CatBoostRanker(
        iterations=CATBOOST_ITERATIONS, learning_rate=CATBOOST_LR, depth=CATBOOST_DEPTH,
        loss_function='YetiRank', random_seed=42,
        early_stopping_rounds=CATBOOST_EARLY_STOP, verbose=False)
    model.fit(train_pool, eval_set=val_pool)
    print(f"  [CatBoost] Trained {model.tree_count_} trees")
    return model


