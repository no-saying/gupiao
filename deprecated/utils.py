"""工具模块 — Embedding 缓存 + 小工具函数"""
import numpy as np
import torch
from pathlib import Path
from config import MODEL_DIR, DEVICE


def _norm(s):
    """Min-max normalize to [0,1]."""
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def _industry_col(panel):
    """返回行业列名，优先SW L2，回退到 industry 编码。"""
    return 'l2_name' if 'l2_name' in panel.columns else 'industry'


def _resolve_model_path(seed, version, fold):
    """自动选择最新版本模型：v3 > v2 > 无后缀"""
    for ver in [version, "v2", ""]:
        path = MODEL_DIR / f"portfolio_model_{seed}_{ver}_fold{fold}.pt" if ver else MODEL_DIR / f"portfolio_model_{seed}_fold{fold}.pt"
        if path.exists():
            return path
    return MODEL_DIR / f"portfolio_model_{seed}_{version}_fold{fold}.pt"


_EMBEDDING_CACHE = None  # {'embeddings': ndarray (300, d_model), 'n_features': int}

def _get_stock_embeddings(X_lt, stock_idx_map=None, force_refresh=False):
    """计算 NN stock embeddings，每次运行仅计算一次。

    被以下模块共用（原先各自独立加载同一个模型 4-5 次）：
      - Nash 均衡 (nash_equilibrium_selection)
      - 隐式软聚类 (soft-cluster dedup)
      - HRP 权重 (hierarchical risk parity)
      - PR 特征值自适应 λ

    Returns: dict with keys 'embeddings' (300, d_model) and 'n_features'
    """
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is not None and not force_refresh:
        return _EMBEDDING_CACHE

    model_path = _resolve_model_path("g791", "v3", 1)
    if not model_path.exists():
        print("  [EMBED] Model not found, embedding cache disabled")
        _EMBEDDING_CACHE = {'embeddings': None, 'n_features': 0}
        return _EMBEDDING_CACHE

    from model import PortfolioPredictor
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    saved_n = cfg.get("n_features", 57)
    cur_n = X_lt.shape[-1]

    # 标准化：与 load_predict 一致
    fm = np.array(ckpt["feat_mean"]).reshape(-1)
    fs = np.array(ckpt["feat_std"]).reshape(-1)
    X_aligned = X_lt.astype(np.float32)
    if cur_n != saved_n:
        if cur_n > saved_n:
            X_aligned = X_aligned[..., :saved_n]
        else:
            X_aligned = np.pad(X_aligned, ((0,0),(0,0),(0,0),(0,saved_n-cur_n)))
        # 对齐 fm/fs
        if len(fm) > X_aligned.shape[-1]:
            fm = fm[:X_aligned.shape[-1]]
            fs = fs[:X_aligned.shape[-1]]
    fm = fm.reshape(1, 1, 1, -1)
    fs = fs.reshape(1, 1, 1, -1)

    X_n = (X_aligned - fm) / fs
    X_t = torch.from_numpy(X_n).to(DEVICE, dtype=torch.float32)
    m_t = torch.ones(1, X_aligned.shape[1], device=DEVICE)

    model = PortfolioPredictor(
        n_features=saved_n,
        d_model=cfg.get("d_model", 128),
        n_transformer_layers=cfg.get("n_transformer_layers", 2),
        n_gru_layers=cfg.get("n_gru_layers", 2),
        d_ff=cfg.get("d_ff", 256),
        use_market_gate=cfg.get("use_market_gate", False))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        emb = model.encode_stocks(X_t, m_t).squeeze(0).cpu().numpy()

    _EMBEDDING_CACHE = {'embeddings': emb, 'n_features': saved_n}
    print(f"  [EMBED] Cached ({emb.shape[0]} stocks × {emb.shape[1]} dims, reuses: Nash/Cluster/HRP/PR-λ)")
    return _EMBEDDING_CACHE


