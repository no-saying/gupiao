"""
core/__init__.py — 精简 LGBM 管线统一导出
"""

from core.data      import build_lgbm_data, EXCLUDE_COLS
from core.features  import engineer_features
from core.alpha158  import add_alpha158_features
from core.model     import train_lgbm, predict_lgbm, compute_rankic
from core.selection import select_top_stocks, compute_weights
from core.validate  import walk_forward_validate
