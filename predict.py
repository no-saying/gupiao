"""Generate submission CSV from a trained model."""

from __future__ import annotations

import argparse, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    MODEL_DIR, PROCESSED_DIR, DEVICE, SUBMISSION_PATH,
    MAX_STOCKS, TOP_K_CANDIDATES, START_DATE, END_DATE,
)
from data_loader import fetch_csi300_stocks, build_panel
from features import engineer_features, make_window_samples
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
    """Greedy diverse selection using embedding cosine distance."""
    n = len(stock_ids)
    order = np.argsort(scores)[::-1]
    candidates = list(order[:min(top_k, n)])

    if embeddings is not None and len(candidates) > 1:
        selected = [candidates.pop(0)]
        while len(selected) < max_stocks and candidates:
            best_score, best_idx = -np.inf, None
            for idx in candidates:
                sim = max(
                    float(torch.cosine_similarity(
                        embeddings[idx].unsqueeze(0),
                        embeddings[s].unsqueeze(0), dim=-1,
                    )) for s in selected
                )
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
# Generate
# ---------------------------------------------------------------------------

def generate_submission(
    model: PortfolioPredictor,
    stock_ids: list[str],
    panel: pd.DataFrame,
    output_path: Path = SUBMISSION_PATH,
) -> None:
    model.eval()
    model.to(DEVICE)

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
    print(f"[predict] Date: {dates[-1]}  Valid: {int(mask[-1].sum())}/{len(stock_ids)}")

    sel = select_diverse_portfolio(s, stock_ids, e)
    df = pd.DataFrame(sel, columns=["stock_id", "weight"])
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[predict] -> {output_path}")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=str(MODEL_DIR / "portfolio_model.pt"))
    p.add_argument("--output", type=str, default=str(SUBMISSION_PATH))
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location=DEVICE, weights_only=False)
    nf = ckpt["config"]["n_features"]
    stock_ids = ckpt.get("stock_ids") or fetch_csi300_stocks()

    model = PortfolioPredictor(n_features=nf)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[predict] Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    panel = build_panel(stock_ids, START_DATE, END_DATE)
    panel = engineer_features(panel, stock_ids)
    generate_submission(model, stock_ids, panel, Path(args.output))


if __name__ == "__main__":
    main()
