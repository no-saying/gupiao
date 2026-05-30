"""Prediction script — reads test.csv, loads model, outputs result.csv.

Weight allocation methods:
  - equal:        1/N (baseline)
  - softmax:      exp(score/T) / sum(exp(score/T))
  - inv_vol:      1/vol / sum(1/vol)  (risk parity)
  - score_vol:    score / vol  (return-risk balanced, recommended)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, OUTPUT_DIR, DEVICE, MAX_STOCKS, TOP_K_CANDIDATES
from data_loader import load_panel_from_csv
from featurework import engineer_features, make_window_samples
from model import PortfolioPredictor


# ---------------------------------------------------------------------------
# Diverse selection
# ---------------------------------------------------------------------------

def select_diverse_portfolio(
    scores: np.ndarray,
    stock_ids: list[str],
    embeddings: torch.Tensor | None = None,
    max_stocks: int = MAX_STOCKS,
    top_k: int = TOP_K_CANDIDATES,
) -> list[int]:
    """Greedy diverse top-K selection. Returns indices."""
    order = np.argsort(scores)[::-1]
    candidates = list(order[:min(top_k, len(order))])

    if embeddings is not None and len(candidates) > 1:
        selected = [candidates.pop(0)]
        while len(selected) < max_stocks and candidates:
            best_score, best_idx = -np.inf, None
            for idx in candidates:
                sim = max(float(torch.cosine_similarity(
                    embeddings[idx].unsqueeze(0), embeddings[s].unsqueeze(0), dim=-1,
                )) for s in selected)
                adj = scores[idx] - sim * scores[idx] * 0.5
                if adj > best_score:
                    best_score, best_idx = adj, idx
            if best_idx is None:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)
    else:
        selected = candidates[:max_stocks]
    return selected


# ---------------------------------------------------------------------------
# Weight allocation
# ---------------------------------------------------------------------------

def compute_weights(
    method: str,
    indices: list[int],
    scores: np.ndarray,
    panel: pd.DataFrame,
    stock_ids: list[str],
    temperature: float = 0.5,
) -> np.ndarray:
    """Allocate weights to selected stocks.

    Methods:
      equal     — 1/N, simplest
      softmax   — score-based via softmax
      inv_vol   — inverse volatility (risk parity)
      score_vol — score / volatility (return-risk balanced, recommended)
    """
    k = len(indices)
    if k == 0:
        return np.array([])

    if method == "equal":
        return np.ones(k) / k

    sel_scores = scores[indices]
    sel_ids = [stock_ids[i] for i in indices]

    if method == "softmax":
        x = np.exp((sel_scores - sel_scores.max()) / temperature)
        w = x / x.sum()

    elif method == "inv_vol":
        vols = _get_volatilities(sel_ids, panel)
        inv = 1.0 / np.clip(vols, 0.005, None)
        w = inv / inv.sum()

    elif method == "score_vol":
        vols = _get_volatilities(sel_ids, panel)
        # score in [0, 1] range via min-max, then divide by vol
        s_norm = (sel_scores - sel_scores.min()) / (sel_scores.max() - sel_scores.min() + 1e-9)
        raw = s_norm / np.clip(vols, 0.005, None)
        w = raw / raw.sum()

    else:
        w = np.ones(k) / k

    return w


def _get_volatilities(sel_ids: list[str], panel: pd.DataFrame) -> np.ndarray:
    """Extract historical daily volatility for each selected stock."""
    vols = []
    for sid in sel_ids:
        try:
            sd = panel.xs(sid, level="stock_id")
            vol = float((sd["pctChg"] / 100.0).std())
        except Exception:
            vol = 0.03  # default ~3% daily vol
        vols.append(max(vol, 0.002))
    return np.array(vols)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/app/data/test.csv")
    p.add_argument("--train-data", type=str, default="/app/data/train.csv")
    p.add_argument("--model-dir", type=str, default="/app/model")
    p.add_argument("--output", type=str, default="/app/output/result.csv")
    p.add_argument("--seed", type=int, default=789)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    model_path = Path(args.model_dir) / "portfolio_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    nf = ckpt["config"]["n_features"]
    stock_ids = ckpt.get("stock_ids")

    model = PortfolioPredictor(n_features=nf).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[test] Loaded model: {sum(p.numel() for p in model.parameters()):,} params")

    # Load data (need full feature set from train data)
    panel = load_panel_from_csv(args.train_data)
    if stock_ids is None:
        stock_ids = sorted(panel.index.get_level_values("stock_id").unique())
    panel = engineer_features(panel)

    # Build latest sample
    X, y, mask, dates = make_window_samples(panel, stock_ids)
    if len(X) == 0:
        raise RuntimeError("No valid samples")

    xb = torch.from_numpy(X[-1:]).to(DEVICE)
    mb = torch.from_numpy(mask[-1:]).to(DEVICE)

    with torch.no_grad():
        scores, _ = model(xb, mb)
        emb = model.encode_stocks(xb, mb)

    s = scores.squeeze(0).cpu().numpy()
    e = emb.squeeze(0).cpu()
    valid_n = int(mask[-1].sum())
    print(f"[test] Date: {dates[-1]}  Valid: {valid_n}/{len(stock_ids)}")

    # Select stocks
    indices = select_diverse_portfolio(s, stock_ids, e)

    # Allocate weights (score_vol = return-risk balanced)
    w = compute_weights("score_vol", indices, s, panel, stock_ids)

    sel = [(stock_ids[i], float(w[j])) for j, i in enumerate(indices)]
    df = pd.DataFrame(sel, columns=["stock_id", "weight"])
    print(f"[test] Weights: " + ", ".join(f"{sid}={wt:.3f}" for sid, wt in sel))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[test] -> {args.output}")


if __name__ == "__main__":
    main()
