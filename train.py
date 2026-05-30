"""Training script for the attention-based stock ranking model."""

from __future__ import annotations

import argparse, pickle, shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    MODEL_DIR, PROCESSED_DIR, RAW_DIR, DEVICE, D_MODEL,
    BATCH_SIZE, N_EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP,
    LR_PATIENCE, EARLY_STOP_PATIENCE, VAL_RATIO, TEST_RATIO,
    MAX_STOCKS, START_DATE, END_DATE,
)
from data_loader import fetch_csi300_stocks, build_panel
from features import engineer_features, make_window_samples
from model import PortfolioPredictor, listnet_loss, lambdarank_loss, pairwise_ranking_loss


LOSS_FN = {
    "listnet": listnet_loss,
    "lambdarank": lambdarank_loss,
    "pairwise": pairwise_ranking_loss,
}


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

def make_dataloaders(
    X: np.ndarray, y: np.ndarray, mask: np.ndarray,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Chronological train/val/test split."""
    n = len(X)
    ts = int(n * (1 - test_ratio))
    vs = int(n * (1 - test_ratio - val_ratio))

    def _dl(a, b, c, shuffle=False):
        ds = TensorDataset(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    t = _dl(X[:vs], y[:vs], mask[:vs], shuffle=True)
    v = _dl(X[vs:ts], y[vs:ts], mask[vs:ts])
    s = _dl(X[ts:], y[ts:], mask[ts:])
    print(f"[train] Split: {len(t.dataset)} / {len(v.dataset)} / {len(s.dataset)}")
    return t, v, s


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: PortfolioPredictor, loader: DataLoader, loss_fn) -> dict[str, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0.0, 0.0
    n_samples = 0

    for xb, yb, mb in loader:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        scores, _ = model(xb, mb)
        total_loss += loss_fn(scores, yb, mb).item() * xb.size(0)
        n_samples += xb.size(0)

        for i in range(xb.size(0)):
            vi = (mb[i] > 0.5).nonzero(as_tuple=True)[0]
            if len(vi) < 2:
                continue
            sv, tv = scores[i][vi], yb[i][vi]
            sd = sv.unsqueeze(1) - sv.unsqueeze(0)
            td = tv.unsqueeze(1) - tv.unsqueeze(0)
            ok = (torch.sign(sd) == torch.sign(td)).float()
            vp = (td.abs() > 1e-6).float()
            correct += (ok * vp).sum().item()
            total += vp.sum().item()

    return {"loss": total_loss / max(n_samples, 1), "pair_acc": correct / max(total, 1)}


def _portfolio_ret(scores, y_true, mask_vec, k=MAX_STOCKS):
    vi = np.where(mask_vec > 0.5)[0]
    if len(vi) == 0:
        return 0.0
    top = vi[np.argsort(scores[vi])[::-1][:min(k, len(vi))]]
    return float(np.mean(y_true[top]))


@torch.no_grad()
def backtest(model: PortfolioPredictor, loader: DataLoader) -> dict[str, float]:
    model.eval()
    rets = []
    for xb, yb, mb in loader:
        scores, _ = model(xb.to(DEVICE), mb.to(DEVICE))
        for i in range(len(xb)):
            rets.append(_portfolio_ret(scores[i].cpu().numpy(), yb[i].numpy(), mb[i].numpy()))
    arr = np.array(rets)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "sharpe": float(np.mean(arr) / (np.std(arr) + 1e-9)),
        "win_rate": float((arr > 0).mean()),
        "n": len(arr),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model, train_loader, val_loader,
    epochs=N_EPOCHS, lr=LR, wd=WEIGHT_DECAY, loss_name="listnet",
) -> PortfolioPredictor:
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=LR_PATIENCE)
    loss_fn = LOSS_FN[loss_name]

    best_loss = float("inf")
    best_w: dict | None = None
    patience = 0

    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep:3d}")
        for xb, yb, mb in pbar:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            scores, _ = model(xb, mb)
            loss = loss_fn(scores, yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl += loss.item()
            pbar.set_postfix(loss=f"{tl / max(pbar.n, 1):.4f}")

        vm = evaluate(model, val_loader, loss_fn)
        print(f"  train={tl/len(train_loader):.4f}  val={vm['loss']:.4f}  pair_acc={vm['pair_acc']:.3f}")
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
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--loss", choices=list(LOSS_FN), default="listnet")
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", type=str, default=str(MODEL_DIR / "portfolio_model.pt"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.download:
        shutil.rmtree(RAW_DIR, ignore_errors=True)
        RAW_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    stocks = fetch_csi300_stocks()
    panel = build_panel(stocks, START_DATE, END_DATE)
    panel = engineer_features(panel, stocks)

    X, y, mask, dates = make_window_samples(panel, stocks)
    nf = X.shape[-1]
    print(f"  X={X.shape}  features={nf}")

    with open(PROCESSED_DIR / "samples.pkl", "wb") as f:
        pickle.dump({"X": X, "y": y, "mask": mask, "dates": dates, "stock_ids": stocks}, f)

    tl, vl, sl = make_dataloaders(X, y, mask)
    model = PortfolioPredictor(n_features=nf)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  Device: {DEVICE}")

    model = train_model(model, tl, vl, epochs=args.epochs, lr=args.lr, loss_name=args.loss)
    tm = evaluate(model, sl, LOSS_FN[args.loss])
    bt = backtest(model, sl)
    print(f"\n  Test: loss={tm['loss']:.4f}  pair_acc={tm['pair_acc']:.3f}")
    print(f"  Backtest: mean={bt['mean']:+.6f}  sharpe={bt['sharpe']:+.4f}  win={bt['win_rate']:.1%}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "stock_ids": stocks,
        "config": {"n_features": nf, "d_model": D_MODEL},
    }, args.output)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
