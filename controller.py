"""Ensemble controller — multi-model integration with weight optimization."""

from __future__ import annotations

import argparse, pickle, subprocess, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, PROCESSED_DIR, DEVICE, SUBMISSION_PATH, START_DATE, END_DATE
from data_loader import fetch_csi300_stocks, build_panel
from features import engineer_features, make_window_samples, get_norm_stats
from model import PortfolioPredictor
from predict import select_diverse_portfolio

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "baseline":  {"path": MODEL_DIR / "portfolio_model.pt", "features": 34},
    "seed42":    {"path": MODEL_DIR / "portfolio_model_seed42.pt", "features": 52},
    "seed123":   {"path": MODEL_DIR / "portfolio_model_seed123.pt", "features": 52},
    "seed456":   {"path": MODEL_DIR / "portfolio_model_seed456.pt", "features": 52},
    "seed789":   {"path": MODEL_DIR / "portfolio_model_seed789.pt", "features": 52},
    "seed999":   {"path": MODEL_DIR / "portfolio_model_seed999.pt", "features": 52},
    "g787":      {"path": MODEL_DIR / "portfolio_model_g787.pt", "features": 52},
    "g789":      {"path": MODEL_DIR / "portfolio_model_g789.pt", "features": 52},
    "g791":      {"path": MODEL_DIR / "portfolio_model_g791.pt", "features": 52},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(entry: dict, n_features: int | None = None) -> tuple | None:
    path = entry["path"]
    if not path.exists():
        return None

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    nf = n_features or entry["features"]
    model = PortfolioPredictor(n_features=nf).to(DEVICE)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        sd = model.state_dict()
        for k, v in ckpt["model_state_dict"].items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v
        model.load_state_dict(sd)
    model.eval()
    return model, ckpt.get("feat_mean"), ckpt.get("feat_std")


def _stock_stats(sel_ids, panel):
    stats = {}
    for sid in sel_ids:
        try:
            sd = panel.xs(sid, level="stock_id")
            ret = sd["pctChg"] / 100.0
            stats[sid] = {
                "vol": float(ret.std()),
                "sharpe": float(ret.mean() / ret.std()) if ret.std() > 0 else 0,
            }
        except Exception:
            stats[sid] = {"vol": 0.03, "sharpe": 0}
    return stats


def _compute_weights(method, scores, stats=None, temperature=0.5):
    n = len(scores)
    if method == "equal":
        return np.ones(n) / n
    if method == "softmax":
        x = np.exp((scores - scores.max()) / temperature)
        return x / x.sum()
    if method == "sharpe" and stats:
        w = np.array([max(stats[s]["sharpe"], 0) + 0.01 for s in stats])
        return w / w.sum()
    return np.ones(n) / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="seed789,seed456")
    p.add_argument("--weight", type=str, default="sharpe",
                   choices=["equal", "softmax", "sharpe"])
    p.add_argument("--output", type=str, default=str(SUBMISSION_PATH))
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()

    # Data
    print("=" * 60)
    pp = PROCESSED_DIR / "samples.pkl"
    if pp.exists():
        with open(pp, "rb") as f:
            data = pickle.load(f)
        stock_ids = data["stock_ids"]
        panel = build_panel(stock_ids, START_DATE, END_DATE)
        panel = engineer_features(panel, stock_ids)
    else:
        stock_ids = fetch_csi300_stocks()
        panel = build_panel(stock_ids, START_DATE, END_DATE)
        panel = engineer_features(panel, stock_ids)

    # Load models
    names = [m.strip() for m in args.models.split(",")]
    models = []
    print(f"Loading {len(names)} models ...")
    for name in names:
        entry = MODEL_REGISTRY.get(name)
        if entry is None:
            continue
        r = _load_model(entry)
        if r:
            models.append((name, *r))
            print(f"  {name}")

    if not models:
        sys.exit(1)

    # Predict
    X, y, mask, dates = make_window_samples(panel, stock_ids)
    xb = torch.from_numpy(X[-1:]).to(DEVICE)
    mb = torch.from_numpy(mask[-1:]).to(DEVICE)

    all_scores, all_embs = [], []
    for name, model, fm, fs in models:
        with torch.no_grad():
            s, _ = model(xb, mb)
            e = model.encode_stocks(xb, mb)
        all_scores.append(s.squeeze(0).cpu().numpy())
        all_embs.append(e.squeeze(0).cpu())

    avg_scores = np.mean(all_scores, axis=0)
    avg_emb = torch.stack(all_embs).mean(dim=0)

    sel = select_diverse_portfolio(avg_scores, stock_ids, avg_emb)
    sel_ids = [s[0] for s in sel]
    sel_scores = np.array([avg_scores[stock_ids.index(s)] for s in sel_ids])
    stats = _stock_stats(sel_ids, panel)

    print(f"\nSelected: {', '.join(sel_ids)}")

    # Weight optimization
    if args.compare:
        methods = ["equal", "softmax", "sharpe"]
        best_score, best_w, best_method = -999, None, None
        for method in methods:
            w = _compute_weights(method, sel_scores, stats)
            pd.DataFrame({"stock_id": sel_ids, "weight": w}).to_csv("output/result.csv", index=False)
            r = subprocess.run(["python", "score_self.py"], capture_output=True, text=True)
            score = float(r.stdout.strip().split(": ")[-1]) if ": " in r.stdout else -999
            print(f"  {method:10s} {w} -> {score:.6f}")
            if score > best_score:
                best_score, best_w, best_method = score, w, method
        print(f"\n  Best: {best_method} ({best_score:.6f})")
        w = best_w
    else:
        w = _compute_weights(args.weight, sel_scores, stats)
        pd.DataFrame({"stock_id": sel_ids, "weight": w}).to_csv("output/result.csv", index=False)
        r = subprocess.run(["python", "score_self.py"], capture_output=True, text=True)
        print(f"\n  {args.weight}: {w} -> {r.stdout.strip().split(': ')[-1]}")

    df = pd.DataFrame({"stock_id": sel_ids, "weight": w})
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
