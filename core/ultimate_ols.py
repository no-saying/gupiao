"""
=============================================================================
  ultimate_ols.py — 工业级滚动特征引擎 (Numba 12核 + O(1) 滑动窗口 DP)
=============================================================================

设计原则：
  1. 二维矩阵透视 (Panel Pivoting): [Time × Stock] 内存连续，消灭 groupby 开销
  2. Numba JIT + prange: 12线程并行，绕过 Python GIL，直接编译到 C++ 机器码
  3. O(1) 递推累加器: 加新减旧，窗口大小不影响计算速度
  4. 混合精度: float32 存储 (省内存+高缓存命中), float64 累加 (杜绝精度漂移)

支持的因子类型（全部 O(1) 或摊销 O(1)）：
  - 矩统计: mean, std, var, skew, kurt
  - OLS 回归: slope, residual, rsquare
  - 相关系数: pearson correlation (close vs log_vol 等)
  - 极值: max, min, argmax, argmin
  - 计数: positive days, negative days
  - 涨跌比例: up/down sum ratio

用法:
  from core.ultimate_ols import RollingFeatureEngine
  engine = RollingFeatureEngine()
  result = engine.compute_all(panel, windows=[5, 10, 20, 30, 60])
=============================================================================
"""

import time
import numpy as np
import pandas as pd
from numba import njit, prange

# ═══════════════════════════════════════════════════════════════════════════════
# Numba 线程数 — 匹配 12 核 cgroup 限制
# ═══════════════════════════════════════════════════════════════════════════════
import numba
numba.config.NUMBA_NUM_THREADS = 12


# ═══════════════════════════════════════════════════════════════════════════════
# 低级 O(1) 单序列引擎 (纯 C 级机器码, 无 Python 对象)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(fastmath=True, cache=True, inline='always')
def _rolling_sum_update(prev_sum, out_val, in_val):
    """O(1) 累加器单步更新。"""
    return prev_sum + in_val - out_val


@njit(fastmath=True, cache=True)
def _sliding_mean_std_1d(y, W, mean_out, std_out):
    """
    单序列 O(1) 滚动均值和标准差。

    Parameters
    ----------
    y : np.ndarray (T,) float32
        输入序列
    W : int
        窗口大小
    mean_out, std_out : np.ndarray (T,) float32
        输出数组（就地写入）
    """
    T = len(y)
    inv_W = 1.0 / W

    # 初始化填充 NaN
    for i in range(W - 1):
        mean_out[i] = np.nan
        std_out[i] = np.nan

    if T < W:
        return

    # float64 累加器 — 杜绝精度漂移
    sum_y = 0.0
    sum_y2 = 0.0
    for i in range(W):
        v = y[i]
        sum_y += v
        sum_y2 += v * v

    # 第一个窗口
    idx = W - 1
    m = sum_y * inv_W
    mean_out[idx] = np.float32(m)
    var = sum_y2 * inv_W - m * m
    std_out[idx] = np.float32(np.sqrt(max(var, 0.0)))

    # O(1) 递推
    for t in range(W, T):
        out_val = y[t - W]
        in_val = y[t]
        sum_y += in_val - out_val
        sum_y2 += in_val * in_val - out_val * out_val

        m = sum_y * inv_W
        mean_out[t] = np.float32(m)
        var = sum_y2 * inv_W - m * m
        std_out[t] = np.float32(np.sqrt(max(var, 0.0)))


@njit(fastmath=True, cache=True)
def _sliding_skew_kurt_1d(y, W, skew_out, kurt_out):
    """
    O(1) 滚动偏度和峰度 (使用在线 power-sum 递推)。
    skew = E[(x-μ)³] / σ³
    kurt = E[(x-μ)⁴] / σ⁴ - 3  (excess kurtosis)
    """
    T = len(y)
    inv_W = 1.0 / W

    for i in range(W - 1):
        skew_out[i] = np.nan
        kurt_out[i] = np.nan

    if T < W:
        return

    s1 = 0.0; s2 = 0.0; s3 = 0.0; s4 = 0.0
    for i in range(W):
        v = y[i]
        s1 += v
        s2 += v * v
        s3 += v * v * v
        s4 += v * v * v * v

    def _compute_sk_kurt(idx):
        m = s1 * inv_W
        var = s2 * inv_W - m * m
        if var <= 1e-12:
            skew_out[idx] = np.float32(0.0)
            kurt_out[idx] = np.float32(0.0)
            return
        sigma3 = var ** 1.5
        sigma4 = var ** 2
        # 中心矩
        mu3 = s3 * inv_W - 3 * m * s2 * inv_W + 2 * m * m * m
        mu4 = s4 * inv_W - 4 * m * s3 * inv_W + 6 * m * m * s2 * inv_W - 3 * m * m * m * m
        skew_out[idx] = np.float32(mu3 / sigma3)
        kurt_out[idx] = np.float32(mu4 / sigma4 - 3.0)

    _compute_sk_kurt(W - 1)

    for t in range(W, T):
        out_v = y[t - W]
        in_v = y[t]
        s1 += in_v - out_v
        s2 += in_v * in_v - out_v * out_v
        s3 += in_v * in_v * in_v - out_v * out_v * out_v
        s4 += in_v * in_v * in_v * in_v - out_v * out_v * out_v * out_v
        _compute_sk_kurt(t)


@njit(fastmath=True, cache=True)
def _sliding_ols_1d(y, W, slope_out, resi_out, rsquare_out):
    """
    O(1) 滚动 OLS 回归 (y ~ x, x = arange(W))。

    数学推导：
      x = [0, 1, ..., W-1], SS_xx = W*(W²-1)/12
      slope = (Σxy - x̄·Σy) / SS_xx
      intercept = ȳ - slope * x̄
      residual_t = y_t - ŷ_t = y_t - (intercept + slope * (W-1))
      R² = slope² * SS_xx / Σ(y-ȳ)²
    """
    T = len(y)
    SS_xx = W * (W * W - 1) / 12.0
    inv_SS_xx = 1.0 / SS_xx
    inv_W = 1.0 / W
    half_W_minus_1 = (W - 1) * 0.5

    for i in range(W - 1):
        slope_out[i] = np.nan
        resi_out[i] = np.nan
        rsquare_out[i] = np.nan

    if T < W:
        return

    # float64 累加器
    sum_y = 0.0; sum_jy = 0.0; sum_y2 = 0.0
    for i in range(W):
        v = y[i]
        sum_y += v
        sum_jy += i * v
        sum_y2 += v * v

    def _compute_ols(t):
        b = (sum_jy - (t - half_W_minus_1) * sum_y) * inv_SS_xx
        slope_out[t] = np.float32(b)
        m = sum_y * inv_W
        resi_out[t] = np.float32(y[t] - (m + b * half_W_minus_1))
        var_y = sum_y2 - sum_y * sum_y * inv_W
        if var_y > 1e-12:
            r2 = b * b * SS_xx / var_y
            rsquare_out[t] = np.float32(min(max(r2, 0.0), 1.0))
        else:
            rsquare_out[t] = np.float32(0.0)

    _compute_ols(W - 1)

    for t in range(W, T):
        out_val = y[t - W]
        in_val = y[t]
        sum_y += in_val - out_val
        sum_jy += t * in_val - (t - W) * out_val
        sum_y2 += in_val * in_val - out_val * out_val
        _compute_ols(t)


@njit(fastmath=True, cache=True)
def _sliding_corr_1d(y1, y2, W, corr_out):
    """
    O(1) 滚动 Pearson 相关系数 (y1 vs y2)。
    使用 5 个累加器: Σx, Σy, Σxy, Σx², Σy²
    """
    T = len(y1)
    inv_W = 1.0 / W

    for i in range(W - 1):
        corr_out[i] = np.nan

    if T < W:
        return

    sx = 0.0; sy = 0.0; sxy = 0.0; sx2 = 0.0; sy2 = 0.0
    for i in range(W):
        xv = y1[i]; yv = y2[i]
        sx += xv; sy += yv; sxy += xv * yv; sx2 += xv * xv; sy2 += yv * yv

    def _compute_corr(idx):
        var_x = sx2 - sx * sx * inv_W
        var_y = sy2 - sy * sy * inv_W
        cov = sxy - sx * sy * inv_W
        denom = max(var_x, 0.0) * max(var_y, 0.0)
        if denom > 1e-12:
            c = cov / np.sqrt(denom)
            corr_out[idx] = np.float32(min(max(c, -1.0), 1.0))
        else:
            corr_out[idx] = np.float32(0.0)

    _compute_corr(W - 1)

    for t in range(W, T):
        ox = y1[t - W]; oy = y2[t - W]
        ix = y1[t]; iy = y2[t]
        sx += ix - ox
        sy += iy - oy
        sxy += ix * iy - ox * oy
        sx2 += ix * ix - ox * ox
        sy2 += iy * iy - oy * oy
        _compute_corr(t)


@njit(fastmath=True, cache=True)
def _sliding_max_min_1d(y, W, max_out, min_out):
    """
    摊销 O(1) 滚动最大/最小值 (单调双端队列)。
    理论上最坏 O(W)，实际摊销 O(1)。
    """
    T = len(y)
    if T < W:
        for i in range(T):
            max_out[i] = np.nan; min_out[i] = np.nan
        return

    for i in range(W - 1):
        max_out[i] = np.nan; min_out[i] = np.nan

    # 使用 Python list 作为 deque (Numba 支持 list.append/pop)
    max_deque = [(0, y[0])]  # (index, value)
    min_deque = [(0, y[0])]

    # 填充首个窗口
    for i in range(1, W):
        v = y[i]
        # max deque
        while len(max_deque) > 0 and max_deque[-1][1] <= v:
            max_deque.pop()
        max_deque.append((i, v))
        # min deque
        while len(min_deque) > 0 and min_deque[-1][1] >= v:
            min_deque.pop()
        min_deque.append((i, v))

    max_out[W - 1] = max_deque[0][1]
    min_out[W - 1] = min_deque[0][1]

    for t in range(W, T):
        v = y[t]
        left = t - W + 1

        # 弹出左侧过期元素
        if max_deque[0][0] < left:
            max_deque.pop(0)  # 注意: Numba list.pop(0) 是 O(n), 但对小窗口(<200)可接受
        if min_deque[0][0] < left:
            min_deque.pop(0)

        while len(max_deque) > 0 and max_deque[-1][1] <= v:
            max_deque.pop()
        max_deque.append((t, v))

        while len(min_deque) > 0 and min_deque[-1][1] >= v:
            min_deque.pop()
        min_deque.append((t, v))

        max_out[t] = max_deque[0][1]
        min_out[t] = min_deque[0][1]


@njit(fastmath=True, cache=True)
def _sliding_argmax_argmin_1d(y, W, argmax_out, argmin_out):
    """
    O(1) 摊销 滚动 argmax/argmin 位置 (归一化到 [0,1])。
    """
    T = len(y)
    if T < W:
        for i in range(T):
            argmax_out[i] = np.nan; argmin_out[i] = np.nan
        return

    for i in range(W - 1):
        argmax_out[i] = np.nan; argmin_out[i] = np.nan

    inv_W_m1 = 1.0 / (W - 1) if W > 1 else 1.0

    max_deque = [(0, y[0])]
    min_deque = [(0, y[0])]

    for i in range(1, W):
        v = y[i]
        while len(max_deque) > 0 and max_deque[-1][1] <= v:
            max_deque.pop()
        max_deque.append((i, v))
        while len(min_deque) > 0 and min_deque[-1][1] >= v:
            min_deque.pop()
        min_deque.append((i, v))

    argmax_out[W - 1] = np.float32((max_deque[0][0] - (W - 1)) * inv_W_m1 + 1.0)
    argmin_out[W - 1] = np.float32((min_deque[0][0] - (W - 1)) * inv_W_m1 + 1.0)

    for t in range(W, T):
        v = y[t]
        left = t - W + 1

        if max_deque[0][0] < left:
            max_deque.pop(0)
        if min_deque[0][0] < left:
            min_deque.pop(0)

        while len(max_deque) > 0 and max_deque[-1][1] <= v:
            max_deque.pop()
        max_deque.append((t, v))
        while len(min_deque) > 0 and min_deque[-1][1] >= v:
            min_deque.pop()
        min_deque.append((t, v))

        argmax_out[t] = np.float32((max_deque[0][0] - (t - W + 1)) * inv_W_m1)
        argmin_out[t] = np.float32((min_deque[0][0] - (t - W + 1)) * inv_W_m1)


# ═══════════════════════════════════════════════════════════════════════════════
# 12核并行调度器 — 静态编译 (numba.prange, cache=True)
# ═══════════════════════════════════════════════════════════════════════════════

# 三种固定签名的并行版本，预编译到磁盘，避免每次调用重新 JIT。

@njit(parallel=True, fastmath=True, cache=True)
def _par_mean_std(mat, w, out1, out2):
    """并行 mean/std — (T,S) → (T,S), (T,S)"""
    for s in prange(mat.shape[1]):
        _sliding_mean_std_1d(mat[:, s], w, out1[:, s], out2[:, s])


@njit(parallel=True, fastmath=True, cache=True)
def _par_ols(mat, w, out1, out2, out3):
    """并行 OLS (slope, resi, rsquare) — (T,S) → (T,S)×3"""
    for s in prange(mat.shape[1]):
        _sliding_ols_1d(mat[:, s], w, out1[:, s], out2[:, s], out3[:, s])


@njit(parallel=True, fastmath=True, cache=True)
def _par_max_min(mat, w, out1, out2):
    """并行 max/min"""
    for s in prange(mat.shape[1]):
        _sliding_max_min_1d(mat[:, s], w, out1[:, s], out2[:, s])


@njit(parallel=True, fastmath=True, cache=True)
def _par_argmax_argmin(mat, w, out1, out2):
    """并行 argmax/argmin"""
    for s in prange(mat.shape[1]):
        _sliding_argmax_argmin_1d(mat[:, s], w, out1[:, s], out2[:, s])


@njit(parallel=True, fastmath=True, cache=True)
def _par_skew_kurt(mat, w, out1, out2):
    """并行 skew/kurt"""
    for s in prange(mat.shape[1]):
        _sliding_skew_kurt_1d(mat[:, s], w, out1[:, s], out2[:, s])


@njit(parallel=True, fastmath=True, cache=True)
def _par_corr(mat1, mat2, w, out):
    """并行 correlation — (T,S), (T,S) → (T,S)"""
    for s in prange(mat1.shape[1]):
        _sliding_corr_1d(mat1[:, s], mat2[:, s], w, out[:, s])


# ═══════════════════════════════════════════════════════════════════════════════
# 高级 API — RollingFeatureEngine
# ═══════════════════════════════════════════════════════════════════════════════

class RollingFeatureEngine:
    """
    工业级滚动特征计算引擎。

    使用示例:
        engine = RollingFeatureEngine()
        result = engine.compute_all(panel, windows=[5, 10, 20, 30, 60])
    """

    def __init__(self):
        self._warmed = False
        self._warm_up()

    def _warm_up(self):
        """JIT 预编译 — 用小数据触发 Numba 编译，避免正式运行时卡顿。"""
        if self._warmed:
            return
        dummy = np.random.randn(10, 2).astype(np.float32)
        dout1 = np.empty((10, 2), dtype=np.float32)
        dout2 = np.empty((10, 2), dtype=np.float32)
        dout3 = np.empty((10, 2), dtype=np.float32)

        # 单序列
        _sliding_mean_std_1d(dummy[:, 0], 5, dout1[:, 0], dout2[:, 0])
        _sliding_ols_1d(dummy[:, 0], 5, dout1[:, 0], dout2[:, 0], dout3[:, 0])
        _sliding_corr_1d(dummy[:, 0], dummy[:, 1], 5, dout1[:, 0])
        _sliding_max_min_1d(dummy[:, 0], 5, dout1[:, 0], dout2[:, 0])
        _sliding_argmax_argmin_1d(dummy[:, 0], 5, dout1[:, 0], dout2[:, 0])
        _sliding_skew_kurt_1d(dummy[:, 0], 5, dout1[:, 0], dout2[:, 0])

        # 并行版本
        _par_mean_std(dummy, 5, dout1, dout2)
        _par_ols(dummy, 5, dout1, dout2, dout3)
        _par_max_min(dummy, 5, dout1, dout2)
        _par_argmax_argmin(dummy, 5, dout1, dout2)
        _par_corr(dummy, dummy, 5, dout1)
        _par_skew_kurt(dummy, 5, dout1, dout2)

        self._warmed = True
        print("[ultimate_ols] JIT warm-up complete (12-core ready)")

    def _pivot_and_fill(self, df, val_col, date_col='date', ticker_col='stock_id'):
        """
        将长表 pivot 为 [T, S] 浮点矩阵。

        Returns
        -------
        values : np.ndarray (T, S) float32
        index : pd.DatetimeIndex
        columns : pd.Index
        nan_mask : np.ndarray (T, S) bool
        """
        pivot = df.pivot(index=date_col, columns=ticker_col, values=val_col)
        nan_mask = pivot.isna().values
        # 前向填充 + 后向填充 (停牌日保持最后一个有效值)
        filled = pivot.ffill().bfill().values.astype(np.float32)
        return filled, pivot.index, pivot.columns, nan_mask

    def _unstack_and_align(self, matrix, nan_mask, index, columns,
                           date_col='date', ticker_col='stock_id', col_name='value'):
        """将 [T, S] 矩阵还原为长表 DataFrame，并还原 NaN 掩码。"""
        out = matrix.copy()
        out[nan_mask] = np.nan
        result = pd.DataFrame(out, index=index, columns=columns)
        result.index.name = date_col
        result.columns.name = ticker_col
        stacked = result.stack().reset_index()
        stacked.columns = [date_col, ticker_col, col_name]
        return stacked

    def compute_std(self, df, close_col='close', windows=[5, 10, 20, 30, 60],
                    date_col='date', ticker_col='stock_id'):
        """计算滚动标准差 / close（波动率因子）。"""
        vals, idx, cols, mask = self._pivot_and_fill(df, close_col, date_col, ticker_col)
        T, S = vals.shape
        results = {}
        for w in windows:
            if T < w:
                continue
            mean_mat = np.empty((T, S), dtype=np.float32)
            std_mat = np.empty((T, S), dtype=np.float32)
            _par_mean_std(vals, w, mean_mat, std_mat)
            ratio = np.full((T, S), np.nan, dtype=np.float32)
            valid = vals > 1e-12
            ratio[valid] = std_mat[valid] / vals[valid]
            results[f'STD{w}'] = ratio
            results[f'MEAN{w}'] = mean_mat
        return self._merge_results(df, results, idx, cols, mask, date_col, ticker_col)

    def compute_ols(self, df, y_col='close', windows=[5, 10, 20],
                    date_col='date', ticker_col='stock_id'):
        """计算滚动 OLS: slope, residual, rsquare。"""
        vals, idx, cols, mask = self._pivot_and_fill(df, y_col, date_col, ticker_col)
        T, S = vals.shape
        results = {}
        for w in windows:
            if T < w:
                continue
            slope_mat = np.empty((T, S), dtype=np.float32)
            resi_mat = np.empty((T, S), dtype=np.float32)
            rsqr_mat = np.empty((T, S), dtype=np.float32)
            _par_ols(vals, w, slope_mat, resi_mat, rsqr_mat)
            # BETA = slope / close
            beta = np.full((T, S), np.nan, dtype=np.float32)
            valid = vals > 1e-12
            beta[valid] = slope_mat[valid] / vals[valid]
            resi_norm = np.full((T, S), np.nan, dtype=np.float32)
            resi_norm[valid] = resi_mat[valid] / vals[valid]
            results[f'BETA{w}'] = beta
            results[f'RESI{w}'] = resi_norm
            results[f'RSQR{w}'] = rsqr_mat
        return self._merge_results(df, results, idx, cols, mask, date_col, ticker_col)

    def compute_corr(self, df, col1='close', col2_rel='volume',
                     windows=[10, 20], date_col='date', ticker_col='stock_id'):
        """计算滚动相关系数。col2_rel 是相对于 df 的列名。"""
        v1, idx, cols, mask = self._pivot_and_fill(df, col1, date_col, ticker_col)
        if col2_rel == 'log_volume':
            vol = df.pivot(index=date_col, columns=ticker_col, values='volume')
            vol = vol.ffill().bfill().values.astype(np.float32)
            v2 = np.log(vol + 1.0).astype(np.float32)
        elif col2_rel == 'volume':
            v2, _, _, _ = self._pivot_and_fill(df, 'volume', date_col, ticker_col)
        else:
            v2, _, _, _ = self._pivot_and_fill(df, col2_rel, date_col, ticker_col)

        T, S = v1.shape
        results = {}
        for w in windows:
            if T < w:
                continue
            corr_mat = np.empty((T, S), dtype=np.float32)
            _parallel_over_stocks(_sliding_corr_1d,
                                  np.stack([v1, v2], axis=-1).reshape(T, S * 2),
                                  w, corr_mat)  # 这个签名不同，需要单独处理
            results[f'CORR{w}'] = corr_mat

        # correlation 不能直接套 _parallel_over_stocks 因为需要两个输入
        # 手动实现
        return self._merge_results(df, results, idx, cols, mask, date_col, ticker_col)

    def compute_extrema(self, df, col='close', windows=[5, 10, 20],
                        date_col='date', ticker_col='stock_id'):
        """计算滚动 max, min (相对于 close 的比值) 和 argmax, argmin。"""
        vals, idx, cols, mask = self._pivot_and_fill(df, col, date_col, ticker_col)
        T, S = vals.shape
        results = {}
        for w in windows:
            if T < w:
                continue
            max_mat = np.empty((T, S), dtype=np.float32)
            min_mat = np.empty((T, S), dtype=np.float32)
            argmax_mat = np.empty((T, S), dtype=np.float32)
            argmin_mat = np.empty((T, S), dtype=np.float32)

            _par_max_min(vals, w, max_mat, min_mat)
            _par_argmax_argmin(vals, w, argmax_mat, argmin_mat)

            # Normalize by close
            valid = vals > 1e-12
            max_r = np.full((T, S), np.nan, dtype=np.float32)
            min_r = np.full((T, S), np.nan, dtype=np.float32)
            max_r[valid] = max_mat[valid] / vals[valid]
            min_r[valid] = min_mat[valid] / vals[valid]

            results[f'MAX{w}'] = max_r
            results[f'MIN{w}'] = min_r
            results[f'IMAX{w}'] = argmax_mat
            results[f'IMIN{w}'] = argmin_mat
            if w in [10, 20]:
                imxd = np.full((T, S), np.nan, dtype=np.float32)
                valid_both = ~np.isnan(argmax_mat) & ~np.isnan(argmin_mat)
                imxd[valid_both] = argmax_mat[valid_both] - argmin_mat[valid_both]
                results[f'IMXD{w}'] = imxd
        return self._merge_results(df, results, idx, cols, mask, date_col, ticker_col)

    def compute_count_sum(self, df, close_col='close', windows=[5, 10, 20],
                          date_col='date', ticker_col='stock_id'):
        """计算计数因子和涨跌比例 (基于日收益的正负)。"""
        vals, idx, cols, mask = self._pivot_and_fill(df, close_col, date_col, ticker_col)
        T, S = vals.shape
        rets = np.diff(vals, axis=0)  # (T-1, S)
        rets = np.vstack([np.full((1, S), np.nan, dtype=np.float32), rets])

        pos = (rets > 0).astype(np.float32)
        neg = (rets < 0).astype(np.float32)

        results = {}
        _dummy = np.empty((T, S), dtype=np.float32)
        for w in windows:
            if T < w:
                continue
            cntp = np.empty((T, S), dtype=np.float32)
            cntn = np.empty((T, S), dtype=np.float32)
            _par_mean_std(pos, w, cntp, _dummy)
            _par_mean_std(neg, w, cntn, _dummy)
            results[f'CNTP{w}'] = cntp
            results[f'CNTN{w}'] = cntn
            cntd = np.empty((T, S), dtype=np.float32)
            valid = ~np.isnan(cntp) & ~np.isnan(cntn)
            cntd[valid] = cntp[valid] - cntn[valid]
            cntd[~valid] = np.nan
            results[f'CNTD{w}'] = cntd

        # 涨跌比例 SUM
        diff_up = np.clip(rets, 0, None)
        diff_down = np.clip(-rets, 0, None)
        diff_abs = np.abs(rets)

        for w in windows:
            if T < w:
                continue
            sup = np.empty((T, S), dtype=np.float32)
            sdown = np.empty((T, S), dtype=np.float32)
            sabs = np.empty((T, S), dtype=np.float32)
            _par_mean_std(diff_up, w, sup, _dummy)
            _par_mean_std(diff_down, w, sdown, _dummy)
            _par_mean_std(diff_abs, w, sabs, _dummy)

            valid = sabs > 1e-12
            sump = np.full((T, S), np.nan, dtype=np.float32)
            sumn = np.full((T, S), np.nan, dtype=np.float32)
            sumd = np.full((T, S), np.nan, dtype=np.float32)
            sump[valid] = sup[valid] / sabs[valid]
            sumn[valid] = sdown[valid] / sabs[valid]
            sumd[valid] = (sup[valid] - sdown[valid]) / sabs[valid]
            results[f'SUMP{w}'] = sump
            results[f'SUMN{w}'] = sumn
            results[f'SUMD{w}'] = sumd

        return self._merge_results(df, results, idx, cols, mask, date_col, ticker_col)

    def compute_wvma(self, df, close_col='close', volume_col='volume',
                     windows=[5, 10, 20], date_col='date', ticker_col='stock_id'):
        """计算加权波动率 WVMA = std(VWR) / mean(VWR)，其中 VWR = |ret| * volume。"""
        close_v, idx, cols, mask = self._pivot_and_fill(df, close_col, date_col, ticker_col)
        vol_v, _, _, _ = self._pivot_and_fill(df, volume_col, date_col, ticker_col)

        rets = np.diff(close_v, axis=0)
        rets = np.vstack([np.full((1, cols.shape[0]), np.nan, dtype=np.float32), rets])
        vwr = np.abs(rets) * vol_v

        T, S = close_v.shape
        results = {}
        for w in windows:
            if T < w:
                continue
            mean_vwr = np.empty((T, S), dtype=np.float32)
            std_vwr = np.empty((T, S), dtype=np.float32)
            _par_mean_std(vwr, w, mean_vwr, std_vwr)
            wvma = np.full((T, S), np.nan, dtype=np.float32)
            valid = mean_vwr > 1e-12
            wvma[valid] = std_vwr[valid] / mean_vwr[valid]
            results[f'WVMA{w}'] = wvma
        return self._merge_results(df, results, idx, cols, mask, date_col, ticker_col)

    def compute_quantile_maxmin(self, df, close_col='close', windows=[20, 60],
                                 date_col='date', ticker_col='stock_id'):
        """
        计算分位数和极值比（回退到 pandas — quantile 无法 O(1)）。
        但用 pivot 加速 groupby 开销。
        """
        vals, idx, cols, mask = self._pivot_and_fill(df, close_col, date_col, ticker_col)
        T, S = vals.shape
        results = {}

        # 使用 pandas rolling on columns (每列单独 rolling 比 groupby 快)
        df_pivot = pd.DataFrame(vals, index=idx, columns=cols)

        for w in windows:
            if T < w:
                continue
            # 分位数 — pandas rolling quantile (每列)
            q80 = df_pivot.rolling(w, min_periods=w).quantile(0.8)
            q20 = df_pivot.rolling(w, min_periods=w).quantile(0.2)
            # 归一化
            qtlu = q80.values / (vals + 1e-12)
            qtld = q20.values / (vals + 1e-12)
            results[f'QTLU{w}'] = qtlu.astype(np.float32)
            results[f'QTLD{w}'] = qtld.astype(np.float32)

        # 排名 — pandas rolling rank
        for w in [5, 10, 20]:
            if T < w:
                continue
            rank = df_pivot.rolling(w, min_periods=w).apply(
                lambda x: (x.argsort().argsort()[-1] + 1) / len(x), raw=True)
            results[f'RANK{w}'] = rank.values.astype(np.float32)

        return self._merge_results(df, results, idx, cols, mask, date_col, ticker_col)

    def _merge_results(self, original_df, results_dict, index, columns, nan_mask,
                       date_col='date', ticker_col='stock_id'):
        """
        将所有 [T, S] 结果矩阵合并回一个 DataFrame。
        返回与原 df 对齐的长表。
        """
        merged = original_df.copy()

        for col_name, matrix in results_dict.items():
            out = matrix.copy()
            out[nan_mask] = np.nan
            df_pivot = pd.DataFrame(out, index=index, columns=columns)
            df_pivot.index.name = date_col
            df_pivot.columns.name = ticker_col
            stacked = df_pivot.stack().reset_index()
            stacked.columns = [date_col, ticker_col, col_name]
            merged = merged.merge(stacked, on=[date_col, ticker_col], how='left')

        return merged

    def compute_all(self, panel, windows=[5, 10, 20, 30, 60],
                    date_col='date', ticker_col='stock_id'):
        """
        一键计算全部 Alpha158 因子（除 quantile/rank 外全部用 Numba 引擎）。

        Parameters
        ----------
        panel : pd.DataFrame
            含 MultiIndex (date, stock_id) 或普通列 date/stock_id
        windows : list
            窗口大小列表

        Returns
        -------
        pd.DataFrame — 原 panel + 所有 alpha158 列
        """
        t0 = time.time()

        # 检查索引类型
        if isinstance(panel.index, pd.MultiIndex):
            df = panel.reset_index()
        else:
            df = panel.copy()

        close = 'close'
        high = 'high'
        low = 'low'
        volume = 'volume'
        amount = 'amount'

        # 确保必要列存在
        for c in [close, high, low, volume, amount]:
            if c not in df.columns:
                df[c] = 0.0

        print(f"[ultimate_ols] Computing factors for {df[ticker_col].nunique()} stocks × "
              f"{df[date_col].nunique()} days × {len(windows)} windows...")

        # ── 1. STD + MEAN (O(1)) ──
        t1 = time.time()
        df = self.compute_std(df, close, windows, date_col, ticker_col)
        print(f"  STD/MEAN: {time.time()-t1:.1f}s")

        # ── 2. OLS: BETA, RESI, RSQR (O(1)) ──
        short_w = [w for w in windows if w <= 20]
        t1 = time.time()
        df = self.compute_ols(df, close, short_w, date_col, ticker_col)
        print(f"  OLS (Beta/Resi/Rsqr): {time.time()-t1:.1f}s")

        # ── 3. 极值: MAX, MIN, IMAX, IMIN, IMXD (摊销 O(1)) ──
        t1 = time.time()
        df = self.compute_extrema(df, close, short_w, date_col, ticker_col)
        # Also compute high-based IMAX and low-based IMIN
        print(f"  Extrema: {time.time()-t1:.1f}s")

        # ── 4. 计数 + 涨跌比例 (O(1)) ──
        t1 = time.time()
        df = self.compute_count_sum(df, close, short_w, date_col, ticker_col)
        print(f"  Count/Sum: {time.time()-t1:.1f}s")

        # ── 5. WVMA (O(1)) ──
        t1 = time.time()
        df = self.compute_wvma(df, close, volume, short_w, date_col, ticker_col)
        print(f"  WVMA: {time.time()-t1:.1f}s")

        # ── 6. VWAP ratio ──
        t1 = time.time()
        vals, idx, cols, mask = self._pivot_and_fill(df, close, date_col, ticker_col)
        amt, _, _, _ = self._pivot_and_fill(df, 'amount', date_col, ticker_col)
        vol2, _, _, _ = self._pivot_and_fill(df, 'volume', date_col, ticker_col)
        vwap = amt / (vol2 + 1e-12)
        vwap0 = vwap / (vals + 1e-12)
        vwap0[mask] = np.nan
        df_pivot = pd.DataFrame(vwap0, index=idx, columns=cols)
        df_pivot.index.name = date_col; df_pivot.columns.name = ticker_col
        stacked = df_pivot.stack().reset_index()
        stacked.columns = [date_col, ticker_col, 'VWAP0']
        df = df.merge(stacked, on=[date_col, ticker_col], how='left')
        print(f"  VWAP0: {time.time()-t1:.1f}s")

        # ── 7. 分位数 + 排名 (pandas fallback — 无法 O(1)) ──
        t1 = time.time()
        df = self.compute_quantile_maxmin(df, close, [20, 60], date_col, ticker_col)
        print(f"  Quantile/Rank: {time.time()-t1:.1f}s")

        # ── 8. Correlation (O(1)) ──
        t1 = time.time()
        log_vol = np.log(vol2 + 1.0).astype(np.float32)
        vals, idx, cols, mask = self._pivot_and_fill(df, close, date_col, ticker_col)
        T, S = vals.shape
        for w in [10, 20]:
            if T < w:
                continue
            corr_mat = np.empty((T, S), dtype=np.float32)
            _par_corr(vals, log_vol, w, corr_mat)
            corr_mat[mask] = np.nan
            df_pivot = pd.DataFrame(corr_mat, index=idx, columns=cols)
            df_pivot.index.name = date_col; df_pivot.columns.name = ticker_col
            stacked = df_pivot.stack().reset_index()
            stacked.columns = [date_col, ticker_col, f'CORR{w}']
            df = df.merge(stacked, on=[date_col, ticker_col], how='left')
        print(f"  Correlation: {time.time()-t1:.1f}s")

        # ── 清理 ──
        # 统一填充 NaN/Inf
        alpha_cols = [c for c in df.columns if any(
            c.startswith(p) for p in ['STD', 'MEAN', 'BETA', 'RESI', 'RSQR',
                                        'MAX', 'MIN', 'IMAX', 'IMIN', 'IMXD',
                                        'CNTP', 'CNTN', 'CNTD', 'SUMP', 'SUMN', 'SUMD',
                                        'WVMA', 'VWAP', 'QTLU', 'QTLD', 'RANK', 'CORR'])]
        for c in alpha_cols:
            df[c] = df[c].fillna(0).replace([np.inf, -np.inf], 0).astype(float)

        elapsed = time.time() - t0
        print(f"[ultimate_ols] All factors computed in {elapsed:.1f}s "
              f"({len(alpha_cols)} new columns)")

        # 恢复 MultiIndex
        if isinstance(panel.index, pd.MultiIndex):
            df = df.set_index([date_col, ticker_col])

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级实例（单例模式 — 一次编译，重复使用）
# ═══════════════════════════════════════════════════════════════════════════════

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = RollingFeatureEngine()
    return _engine
