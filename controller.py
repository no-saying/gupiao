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


def train_lgbm_ranker(df, feature_cols, model_path=None):
    import lightgbm as lgb
    if model_path and model_path.exists():
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

    model = lgb.LGBMRanker(
        objective='lambdarank', boosting_type='gbdt',
        n_estimators=500, num_leaves=63, learning_rate=0.05,
        min_child_samples=20, reg_lambda=0.1, reg_alpha=0.1,
        subsample=0.8, colsample_bytree=0.8,
        label_gain=[i for i in range(10)], verbose=-1, random_state=42)
    model.fit(X_tr, y_tr, group=grp, eval_metric=['ndcg'], callbacks=[lgb.log_evaluation(0)])

    # 验证集 RankIC
    val_pred = model.predict(((val_df[feature_cols].values - fm.values) / fs.values).astype(np.float32))
    val_df = val_df.copy()
    val_df['lgb_score'] = val_pred
    rankic = compute_rankic(val_df, 'lgb_score', 'target')
    print(f"  Val RankIC (last 5 days): {rankic:.4f}")

    if model_path:
        model.booster_.save_model(str(model_path))
    return model, fm, fs, rankic



def predict_lgbm(model, fm, fs, df, feature_cols, date):
    day = df[df['date'] == date].copy()
    if len(day) == 0: return None
    X = (day[feature_cols].values - fm.values) / fs.values
    day['lgb_score'] = model.predict(X)
    return day


# =============================================================================
# NN 模型加载
# =============================================================================

def load_predict(path, X_np, n_features=57):
    from model import PortfolioPredictor
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    model = PortfolioPredictor(n_features=n_features,
        d_model=cfg.get("d_model", 128), n_transformer_layers=cfg.get("n_transformer_layers", 2),
        n_gru_layers=cfg.get("n_gru_layers", 2), d_ff=cfg.get("d_ff", 256),
        use_attention=cfg.get("use_attention", False))
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
# 纳什均衡选股
# =============================================================================

def _compute_spectral_similarity(candidate_ids, panel, lookback=60):
    """
    用 FFT 频谱替代 NN embedding 算相似度。
    对每只股票取最近 lookback 天收益率 → FFT → 功率谱归一化 → 余弦相似度。
    """
    n = len(candidate_ids)
    spectra = []
    for sid in candidate_ids:
        try:
            sd = panel.xs(sid, level="stock_id")
            ret = sd['pctChg'].iloc[-lookback:].values / 100.0
            ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
            if len(ret) < lookback:
                ret = np.pad(ret, (0, lookback - len(ret)), 'constant')
            # Hann 窗减少频谱泄漏
            window = np.hanning(lookback)
            ret_w = ret * window
            # FFT 功率谱
            spectrum = np.abs(np.fft.rfft(ret_w)) ** 2
            # 归一化
            s_norm = spectrum / (np.linalg.norm(spectrum) + 1e-12)
            spectra.append(s_norm)
        except Exception:
            spectra.append(np.zeros(lookback // 2 + 1))
    if not spectra:
        return np.eye(n)
    mat = np.array(spectra)
    sim = mat @ mat.T  # 余弦相似度
    return sim


def nash_equilibrium_selection(candidate_ids, candidate_scores, stock_idx_map, panel,
                                X_panel=None, lam=0.25, n_features=57, mode='embedding'):
    """
    在候选股中求解纯策略纳什均衡。
    效用 = 平均分数 - λ × 平均内部相似度

    mode='embedding': 用 NN embedding 余弦相似度（原版）
    mode='spectral':  用 FFT 频谱功率谱余弦相似度（无需模型）
    """
    n = len(candidate_ids)
    if n < 5:
        return candidate_ids, candidate_scores

    if mode == 'spectral':
        sim = _compute_spectral_similarity(candidate_ids, panel)
    else:
        # 原版：NN embedding 相似度
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
            d_ff=ckpt["config"].get("d_ff",256))
        model_e.load_state_dict(ckpt["model_state_dict"])
        model_e.to(DEVICE); model_e.eval()
        with torch.no_grad():
            emb = model_e.encode_stocks(X_t, m_t).squeeze(0).cpu().numpy()
        idxs = np.array([stock_idx_map[s] for s in candidate_ids])
        e = emb[idxs]
        e_n = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        sim = e_n @ e_n.T

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
                             X_lt=None, nash_mode="embedding"):
    """原有的精度门控+波动率过滤+选股权重逻辑。"""
    # 精度门控
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

    # ── 纳什均衡（放在门控之后，在约10~15只里做多样化）──
    # 池子太小(<7)跳过；池子适中(7~10)放松阈值；池子够大(>10)正常触发
    if (args.game is not None and stock_idx_map is not None
            and X_lt is not None and len(precision_ids) >= 7):
        lam = max(0, min(1, args.game))
        if len(precision_ids) <= 10:
            lam = lam * 0.6  # 小池子降低竞争强度，保留更多动量股
        # 用 X_lt 的实际特征数
        actual_n_features = X_lt.shape[-1] if hasattr(X_lt, 'shape') else n_features
        print(f"  Nash eq (λ={lam:.3f}, gate-pool={len(precision_ids)}, feats={actual_n_features})...")
        try:
            precision_ids, psc = nash_equilibrium_selection(
                precision_ids, psc, stock_idx_map, panel, X_lt, lam,
                n_features=actual_n_features, mode=nash_mode)
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

    # 加权
    if args.weight == "equal":
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
    parser = argparse.ArgumentParser(description="Controller — LGBM + NN + 博弈")
    parser.add_argument("--ensemble", type=str, default="lgbm-nn",
        choices=["lgbm", "lgbm-nn", "gp", "fold1", "old-tscv", "old-3model", "compare"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--weight", type=str, default="bdc", choices=["equal", "softmax", "inv_vol", "bdc"])
    parser.add_argument("--output", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--game", type=float, nargs="?", const=0.25, default=None,
                        help="纳什均衡多样化(λ,默认0.25)。在精度门控后对动量候选股做多样化选股")
    parser.add_argument("--nash-mode", type=str, default="embedding",
                        choices=["embedding", "spectral"],
                        help="纳什相似度: 'embedding'(NN,默认) 或 'spectral'(FFT频谱)")
    parser.add_argument("--multi-day", type=int, default=1, choices=[1,2,3,4,5],
                        help="多日 Rolling 预测(默认1=仅最新日, 3=最近3日加权)")

    parser.add_argument("--bdc", type=str, nargs="?", const="pure", default=None,
                        choices=["pure", "hybrid"],
                        help="BDC 管线: 'pure'=纯BDC, 'hybrid'=精度门控+BDC排序加权")
    parser.add_argument("--risk-penalty", type=float, default=0.30,
                        help="BDC管线: 风险惩罚权重 (默认0.30)")

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
        lgbm_model, lgbm_fm, lgbm_fs, lgbm_rankic = train_lgbm_ranker(lgbm_df, feat_cols, model_path)
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
    elif args.ensemble == "fold1":
        nn_scores = load_predict(MODEL_DIR / "portfolio_model_g791_fold1.pt", X_lt, 51)
    elif args.ensemble == "old-tscv":
        nn_scores = np.mean([load_predict(MODEL_DIR / f"portfolio_model_g791_fold{k}.pt", X_lt, 51) for k in range(1,5)], axis=0)
    elif args.ensemble == "old-3model":
        s_g = load_predict(MODEL_DIR / "portfolio_model_g791.pt", X_lt, 51)
        s_s = load_predict(MODEL_DIR / "portfolio_model_seed42.pt", X_lt, 51)
        s_h = load_predict(MODEL_DIR / "portfolio_model_g787.pt", X_lt, 51)
        nn_scores = 0.6 * s_g + 0.15 * s_s + 0.25 * s_h
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
            # 融合
            blend = 0.7 * norm(lgbm_day_arr) + 0.3 * norm(gp_day)
            ensemble += dw[i] * blend
            if n_days > 1:
                print(f"    Day {i+1}({date_str}): w={dw[i]:.2f}")
        nn_scores = ensemble
        if n_days > 1:
            print(f"  {n_days}-day rolling ensemble (weights={dict(enumerate(dw))})")
        else:
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

    # ── 候选池 ──
    valid = np.where(m_lt > 0.5)[0]
    pool = valid[np.argsort(scores[valid])[-40:]]
    pool_ids = [stock_ids[i] for i in pool]
    pool_sc = [scores[i] for i in pool]

    # ── BDC 管线 vs 旧管线 ──
    if args.bdc:
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
                    stock_idx_map=stock_idx_map, X_lt=X_lt, nash_mode=args.nash_mode)
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
                    stock_idx_map=stock_idx_map, X_lt=X_lt, nash_mode=args.nash_mode)
                sel_ids, weights = legacy_ids, legacy_weights
    else:
        sel_ids, weights = legacy_select_and_weight(
            pool_ids, pool_sc, pool, valid, scores, stock_ids, panel, args,
            stock_idx_map=stock_idx_map, X_lt=X_lt, nash_mode=args.nash_mode)

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
                             "nash_mode","bdc","multi_day","stocks","n"])
            _w.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{score:.6f}", args.ensemble, args.weight,
                args.game if args.game is not None else "",
                getattr(args, 'nash_mode', ''),
                args.bdc if args.bdc else "",
                getattr(args, 'multi_day', 1),
                " ".join([str(s) for s in sel_ids]),
                len(sel_ids),
            ])
        print(f"  [log] {_log_path.name}")
    except Exception as _e:
        print(f"  [log] warn: {_e}")

    # ── BDC vs 旧管线对比 ──
    if args.bdc:
        bdc_label = {"pure": "BDC纯管线", "hybrid": "BDC混合管线"}.get(args.bdc, "BDC管线")
        print(f"\n  --- 自动对比: {bdc_label} vs 旧管线 ---")
        import shutil, os
        bdc_result_path = args.output
        legacy_result_path = str(Path(args.output).parent / "result_legacy_compare.csv")

        # 计算旧管线结果
        legacy_ids, legacy_weights = legacy_select_and_weight(
            pool_ids, pool_sc, pool, valid, scores, stock_ids, panel, args,
            stock_idx_map=stock_idx_map, X_lt=X_lt, nash_mode=args.nash_mode)
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
