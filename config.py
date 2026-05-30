"""Project configuration for the attention-based stock portfolio prediction model.

All hyperparameters, paths, and global settings are centralized here.
"""

from __future__ import annotations

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
TEMP_DIR = ROOT / "temp"
SUBMISSION_PATH = ROOT / "result.csv"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, TEMP_DIR):
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data window
# ---------------------------------------------------------------------------

START_DATE = "2021-06-01"       # reserved warm-up for feature computation
EFFECTIVE_START = "2022-01-01"  # first valid training sample date
END_DATE = "2026-05-30"

LOOKBACK_DAYS = 60               # trading days per input window
PREDICT_HORIZON = 4              # T+1 open -> T+5 open
STEP_DAYS = 1                    # stride between consecutive samples

# ---------------------------------------------------------------------------
# Event calendar — extreme windows excluded from training
# ---------------------------------------------------------------------------

EVENT_WINDOWS: list[tuple[str, str, str, str]] = [
    ("Russia-Ukraine",   "2022-02-24", "2022-03-09",
     "俄乌冲突爆发，全球避险，A股单月跌超8%"),
    ("Shanghai Lockdown", "2022-03-14", "2022-03-18",
     "上海封城恐慌，上证单周跌4.5%"),
    ("Lockdown Bottom",   "2022-04-25", "2022-04-29",
     "封城+人民币贬值，上证破3000"),
    ("H-Share Crash",     "2022-10-24", "2022-10-31",
     "港股A股联动暴跌，外资加速流出"),
    ("Reopen Frenzy",     "2022-11-28", "2022-12-09",
     "防疫转向初期，消费板块短期暴涨20%"),
    ("Stamp Duty Cut",    "2023-08-28", "2023-08-30",
     "印花税减半，政策底信号"),
    ("Quant Crash",       "2024-01-29", "2024-02-07",
     "雪球敲入+量化DMA强平，微盘单周跌超20%"),
    ("924 Stimulus",      "2024-09-24", "2024-10-08",
     "三部门联合发布会，沪指单周涨12.8%"),
    ("Tariff Shock",      "2025-04-07", "2025-04-10",
     "贸易争端升级关税冲击"),
]
EVENT_FILTER_MODE: str = "strict"  # "strict" | "lenient"

# ---------------------------------------------------------------------------
# Cross-market indices (macro features)
# ---------------------------------------------------------------------------

EXTRA_INDICES: dict[str, str] = {
    "sh.000016": "SSE50",           # 上证50
    "sh.000905": "CSI500",          # 中证500
    "sz.399006": "ChiNext",         # 创业板指
    "sh.000001": "SSE_Composite",   # 上证综指
}
EXTRA_INDEX_RET_WINDOWS: list[int] = [5, 10, 20]

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

MOMENTUM_WINDOWS = [5, 10, 20, 60]
VOLATILITY_WINDOWS = [5, 10, 20]
MA_WINDOWS = [5, 10, 20, 60]
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BETA_WINDOW = 60

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

D_MODEL = 128
N_HEADS = 8
N_GRU_LAYERS = 2
DROPOUT = 0.1
N_TRANSFORMER_LAYERS = 2
D_FF = 256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

BATCH_SIZE = 16
N_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
LR_PATIENCE = 10
EARLY_STOP_PATIENCE = 25
VAL_RATIO = 0.15
TEST_RATIO = 0.10
RANKING_MARGIN = 0.05

# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

MAX_STOCKS = 5
TOP_K_CANDIDATES = 30
TEMPERATURE = 0.5

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
