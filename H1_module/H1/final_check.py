"""
final_check.py — Verify B1-P15-Top3 on 11 multi-window WF-CV.
Protocol: V8 Top15 pool -> LogisticRegression p_positive -> Top3 [0.4,0.3,0.3]
Must reproduce: mean~+0.0324, win~91%, min~-0.0363, sharpe~0.999
"""
import sys, os, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(BASE, '..'))
sys.path.insert(0, BASE)
from H1_dataset import load_data, add_label
from H1_features import add_v8_score, build_gate_features, get_gate_feature_names, cross_sectional_rank

WINDOWS_JSON = os.path.join(PROJ, 'data', 'multi_window', 'windows.json')
CTX_DIR = os.path.join(PROJ, 'data', 'multi_window')
FN = get_gate_feature_names()
POOL = 15; TOP_N = 3; W = np.array([0.4, 0.3, 0.3])

def score(stocks, wts, test):
    t = 0.0
    for s, wt in zip(stocks, wts):
        sd = test[test["股票代码"] == s].sort_values("日期")
        if len(sd) >= 2:
            t += wt * (float(sd.iloc[-1]["开盘"]) - float(sd.iloc[0]["开盘"])) / max(float(sd.iloc[0]["开盘"]), 1e-8)
    return t

with open(WINDOWS_JSON) as f:
    windows = json.load(f)

ref_rows = []; all_ret = []
print(f"{'='*90}")
print(f"final_check — B1-P15-Top3 [Pool={POOL}, W={W}]")
print(f"{'='*90}")
print(f"{'Week':<22s} {'Stock1':<8s} {'Stock2':<8s} {'Stock3':<8s} {'Return':>8s} {'p1':>6s} {'p2':>6s} {'p3':>6s}")
print("-"*75)

for win in windows:
    folder = win['folder']
    ctx = load_data(os.path.join(CTX_DIR, folder, 'context.csv'))
    ctx = add_label(ctx); ctx = add_v8_score(ctx)
    gf = build_gate_features(ctx); X_all = cross_sectional_rank(gf, FN)
    lm = ctx["label"].notna().values
    ld = ctx["日期"].max(); pm = (ctx["日期"] == ld).values
    test = load_data(os.path.join(CTX_DIR, folder, 'test.csv'))

    Xt = X_all.loc[lm, FN].values.astype(np.float64)
    yt = (ctx.loc[lm, "label"].values > 0).astype(int)
    m = LogisticRegression(max_iter=1000); m.fit(Xt, yt)

    today = ctx.loc[pm, ["股票代码", "v8_score"]].copy()
    today["pp"] = m.predict_proba(X_all.loc[pm, FN].values.astype(np.float64))[:, 1]
    pool = today.nlargest(POOL, "v8_score")
    final = pool.nlargest(TOP_N, "pp")
    stocks = final["股票代码"].tolist()
    ppos = final["pp"].round(4).tolist()
    ret = score(stocks, W, test)
    all_ret.append(ret)
    ref_rows.append({'week': folder, 'stock_1': stocks[0], 'stock_2': stocks[1], 'stock_3': stocks[2],
                     'return': round(ret, 6), 'p1': ppos[0], 'p2': ppos[1], 'p3': ppos[2]})
    print(f"  {folder:<20s} {stocks[0]:<8s} {stocks[1]:<8s} {stocks[2]:<8s} {ret:>+8.4f} {ppos[0]:>6.3f} {ppos[1]:>6.3f} {ppos[2]:>6.3f}")

rr = np.array(all_ret)
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Mean={np.mean(rr):+.4f}  Win={(rr>0).mean():.0%}  Min={np.min(rr):+.4f}  Max={np.max(rr):+.4f}  Sharpe={np.mean(rr)/(np.std(rr)+1e-8):+.3f}")
for month in ['2026-03', '2026-04', '2026-05']:
    idx = [i for i, w in enumerate(windows) if w['folder'].startswith(month)]
    rm = rr[idx]
    print(f"  {month} ({len(idx)}w): mean={np.mean(rm):+.4f} win={(rm>0).mean():.0%} min={np.min(rm):+.4f} max={np.max(rm):+.4f}")

ref_path = os.path.join(BASE, 'outputs', 'B1_P15_TOP3_REFERENCE.csv')
pd.DataFrame(ref_rows).to_csv(ref_path, index=False)
print(f"\n  Reference saved: {ref_path}")

# Verify against expected
expected = {'mean': 0.0324, 'win': 0.91, 'min': -0.0363, 'max': 0.0829, 'sharpe': 0.999}
actual = {'mean': np.mean(rr), 'win': (rr > 0).mean(), 'min': np.min(rr), 'max': np.max(rr), 'sharpe': np.mean(rr) / (np.std(rr) + 1e-8)}
ok = all(abs(actual[k] - expected[k]) < 0.005 for k in ['mean', 'win', 'min'])
ok &= abs(actual['sharpe'] - expected['sharpe']) < 0.05
print(f"  Consistency: {'PASS' if ok else 'FAIL — STOP, do not proceed to predict_final.py'}")
