"""Training script — reads train.csv, trains model, saves to model/."""

from __future__ import annotations

import argparse, pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    MODEL_DIR, OUTPUT_DIR, DEVICE, D_MODEL,
    BATCH_SIZE, N_EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP,
    LR_PATIENCE, EARLY_STOP_PATIENCE, VAL_RATIO, TEST_RATIO, MAX_STOCKS,
)
from data_loader import load_panel_from_csv, build_event_mask
from featurework import engineer_features, make_window_samples
from model import PortfolioPredictor, listnet_loss, lambdarank_loss, pairwise_ranking_loss

LOSS_FN = {"listnet": listnet_loss, "lambdarank": lambdarank_loss, "pairwise": pairwise_ranking_loss}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_dataloaders(X, y, mask):
    n = len(X)
    ts = int(n * (1 - TEST_RATIO))
    vs = int(n * (1 - TEST_RATIO - VAL_RATIO))

    def _dl(a, b, c, shuffle=False):
        ds = TensorDataset(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (_dl(X[:vs], y[:vs], mask[:vs], shuffle=True),
            _dl(X[vs:ts], y[vs:ts], mask[vs:ts]),
            _dl(X[ts:], y[ts:], mask[ts:]))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    tl, correct, total = 0.0, 0.0, 0.0
    ns = 0
    for xb, yb, mb in loader:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        s, _ = model(xb, mb)
        tl += loss_fn(s, yb, mb).item() * xb.size(0)
        ns += xb.size(0)
        for i in range(xb.size(0)):
            vi = (mb[i] > 0.5).nonzero(as_tuple=True)[0]
            if len(vi) < 2:
                continue
            sv, tv = s[i][vi], yb[i][vi]
            sd = sv.unsqueeze(1) - sv.unsqueeze(0)
            td = tv.unsqueeze(1) - tv.unsqueeze(0)
            ok = (torch.sign(sd) == torch.sign(td)).float()
            vp = (td.abs() > 1e-6).float()
            correct += (ok * vp).sum().item()
            total += vp.sum().item()
    return {"loss": tl / max(ns, 1), "pair_acc": correct / max(total, 1)}


def _portfolio_ret(scores, y_true, mask_vec, k=MAX_STOCKS):
    vi = np.where(mask_vec > 0.5)[0]
    if len(vi) == 0:
        return 0.0
    top = vi[np.argsort(scores[vi])[::-1][:min(k, len(vi))]]
    return float(np.mean(y_true[top]))


@torch.no_grad()
def backtest(model, loader):
    model.eval()
    rets = []
    for xb, yb, mb in loader:
        s, _ = model(xb.to(DEVICE), mb.to(DEVICE))
        for i in range(len(xb)):
            rets.append(_portfolio_ret(s[i].cpu().numpy(), yb[i].numpy(), mb[i].numpy()))
    arr = np.array(rets)
    return {"mean": float(np.mean(arr)), "sharpe": float(np.mean(arr) / (np.std(arr) + 1e-9)),
            "win_rate": float((arr > 0).mean())}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, tl, vl, epochs=N_EPOCHS, lr=LR, loss_name="listnet"):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=LR_PATIENCE)
    fn = LOSS_FN[loss_name]
    best_loss = float("inf")
    best_w = None
    patience = 0

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(tl, desc=f"Epoch {ep:3d}")
        for xb, yb, mb in pbar:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            s, _ = model(xb, mb)
            loss = fn(s, yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{train_loss / max(pbar.n, 1):.4f}")

        vm = evaluate(model, vl, fn)
        print(f"  train={train_loss/len(tl):.4f}  val={vm['loss']:.4f}  pair_acc={vm['pair_acc']:.3f}")
        sch.step(vm["loss"])

        if vm["loss"] < best_loss:
            best_loss = vm["loss"]
            best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at {ep}")
                break

    model.load_state_dict(best_w)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/app/data/train.csv")
    p.add_argument("--model-dir", type=str, default="/app/model")
    p.add_argument("--output", type=str, default="/app/output/result.csv")
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--loss", choices=list(LOSS_FN), default="listnet")
    p.add_argument("--seed", type=int, default=789)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    print("=" * 60)
    panel = load_panel_from_csv(args.data)
    stock_ids = sorted(panel.index.get_level_values("stock_id").unique())
    panel = engineer_features(panel)
    X, y, mask, dates = make_window_samples(panel, stock_ids)
    nf = X.shape[-1]
    print(f"  X={X.shape}  features={nf}")

    # Train
    tl, vl, sl = make_dataloaders(X, y, mask)
    model = PortfolioPredictor(n_features=nf)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  Device: {DEVICE}")
    model = train_model(model, tl, vl, epochs=args.epochs, loss_name=args.loss)

    # Eval
    tm = evaluate(model, sl, LOSS_FN[args.loss])
    bt = backtest(model, sl)
    print(f"\n  Test: loss={tm['loss']:.4f}  pair_acc={tm['pair_acc']:.3f}")
    print(f"  Backtest: mean={bt['mean']:+.6f}  sharpe={bt['sharpe']:+.4f}  win={bt['win_rate']:.1%}")

    # Save
    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "stock_ids": stock_ids,
        "config": {"n_features": nf, "d_model": D_MODEL},
    }, out_dir / "portfolio_model.pt")
    print(f"  Saved to {out_dir / 'portfolio_model.pt'}")


if __name__ == "__main__":
    main()
