"""Prediction script — reads test.csv, loads model, outputs result.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, OUTPUT_DIR, DEVICE, MAX_STOCKS, TOP_K_CANDIDATES
from data_loader import load_panel_from_csv, load_test_csv
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
) -> list[tuple[str, float]]:
    """Greedy diverse top-K selection."""
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

    w = np.ones(len(selected)) / len(selected)
    return [(stock_ids[i], float(w)) for i, w in zip(selected, w)]


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

    sel = select_diverse_portfolio(s, stock_ids, e)
    df = pd.DataFrame(sel, columns=["stock_id", "weight"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[test] -> {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
