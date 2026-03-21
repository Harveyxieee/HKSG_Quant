from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from collections import deque
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


FeatureMap = Dict[str, Dict[str, float]]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def stddev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def zscore(value: float, values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    sigma = stddev(values)
    return 0.0 if sigma <= 1e-12 else (value - avg) / sigma


def compute_return(prices: List[float], lookback: int) -> float:
    if len(prices) <= lookback:
        return 0.0
    base_price = prices[-lookback - 1]
    last_price = prices[-1]
    return last_price / base_price - 1.0 if base_price > 0 else 0.0


@dataclass
class MarketSnapshot:
    median_ret15: float = 0.0
    median_ret60: float = 0.0
    up_ratio_15: float = 0.0
    positive_score_ratio: float = 0.0
    avg_score: float = 0.0

@dataclass
class PortfolioRiskState:
    covariance_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    portfolio_volatility: float = 0.0
    average_correlation: float = 0.0
    market_regime: str = "neutral"
    risk_score: float = 0.0
    target_exposure: float = 0.0
    diversification_breakdown: bool = False
    data_frequency: str = "1m"
    raw_weights: Dict[str, float] = field(default_factory=dict)
    adjusted_weights: Dict[str, float] = field(default_factory=dict)
# =========================
# REGIME FILTER
# =========================
def _resample_price_frame(price_df: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    candidate = price_df.resample(rule).last() if rule else price_df.copy()
    candidate = candidate.sort_index().ffill().dropna(how="all")
    candidate = candidate.loc[:, candidate.notna().sum() >= 2]
    return candidate

def load_price_data(
    history: Dict[str, Deque[Dict[str, float]]],
    universe: List[str],
    frequency: str = "auto",
    min_periods: int = 30,
) -> Tuple[pd.DataFrame, str]:
    series_map: Dict[str, pd.Series] = {}
    for pair in universe:
        rows = list(history.get(pair, []))
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        if "price" not in frame or "ts" not in frame:
            continue
        frame = frame[["ts", "price"]].copy()
        frame["ts"] = pd.to_datetime(frame["ts"], unit="ms", errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame = frame.dropna(subset=["ts", "price"])
        if frame.empty:
            continue
        frame = frame.drop_duplicates(subset=["ts"]).sort_values("ts").set_index("ts")
        frame["price"] = frame["price"].where(frame["price"] > 0)
        series = frame["price"].dropna()
        if len(series) >= 2:
            series_map[pair] = series

    if not series_map:
        return pd.DataFrame(), "raw"

    price_df = pd.concat(series_map, axis=1).sort_index()
    if frequency == "auto":
        for label, rule in [("daily", "1D"), ("hourly", "1H"), ("raw", None)]:
            candidate = _resample_price_frame(price_df, rule)
            if len(candidate) >= min_periods:
                return candidate, label
        return _resample_price_frame(price_df, None), "raw"

    mapping = {"daily": "1D", "hourly": "1H", "raw": None}
    selected = _resample_price_frame(price_df, mapping.get(frequency, None))
    return selected, frequency if frequency in mapping else "raw"


def compute_returns(price_df: pd.DataFrame, method: str = "pct_change") -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    if method == "log":
        returns_df = np.log(price_df).diff()
    else:
        returns_df = price_df.pct_change(fill_method=None)
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns_df = returns_df.loc[:, returns_df.notna().sum() >= 2]
    return returns_df


def compute_cov_matrix(
    returns_df: pd.DataFrame,
    min_samples_per_asset: int = 30,
    min_periods_pairwise: int = 20,
) -> pd.DataFrame:
    if returns_df.empty:
        return pd.DataFrame()

    # 1) 先删掉几乎没数据的资产列
    valid_counts = returns_df.notna().sum(axis=0)
    keep_cols = valid_counts[valid_counts >= min_samples_per_asset].index.tolist()
    returns_df = returns_df[keep_cols]

    if returns_df.shape[1] == 0:
        return pd.DataFrame()

    # 2) 不再整表 dropna(any)，而是用 pairwise covariance
    cov = returns_df.cov(min_periods=min_periods_pairwise)

    # 3) 对角线（单资产方差）至少要有值
    for col in cov.columns:
        if pd.isna(cov.loc[col, col]):
            asset_var = returns_df[col].var()
            cov.loc[col, col] = asset_var if pd.notna(asset_var) else 0.0

    # 4) 非对角缺失先补 0，表示“协方差未知时不过度惩罚”
    cov = cov.fillna(0.0)

    return cov


def compute_portfolio_volatility(weights: Dict[str, float], cov_matrix: pd.DataFrame) -> float:
    if not weights or cov_matrix.empty:
        return 0.0
    common = [pair for pair in weights if pair in cov_matrix.index]
    if not common:
        return 0.0
    vector = np.array([weights[pair] for pair in common], dtype=float)
    sigma = cov_matrix.loc[common, common].to_numpy(dtype=float)
    variance = float(vector.T @ sigma @ vector)
    return math.sqrt(max(variance, 0.0))


def compute_average_correlation(
    returns_df: Optional[pd.DataFrame] = None,
    cov_matrix: Optional[pd.DataFrame] = None,
    window: int = 60,
    min_samples_per_asset: int = 30,
    min_periods_pairwise: int = 20,
) -> float:
    if returns_df is not None and not returns_df.empty:
        sample = returns_df.tail(window).dropna(how="all")
        if sample.empty:
            return 0.0

        valid_counts = sample.notna().sum(axis=0)
        keep_cols = valid_counts[valid_counts >= min_samples_per_asset].index.tolist()
        sample = sample[keep_cols]

        if sample.shape[1] < 2:
            return 0.0

        corr_matrix = sample.corr(min_periods=min_periods_pairwise)

    elif cov_matrix is not None and not cov_matrix.empty:
        diag = np.sqrt(np.maximum(np.diag(cov_matrix.to_numpy(dtype=float)), 0.0))
        denom = np.outer(diag, diag)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_values = np.divide(
                cov_matrix.to_numpy(dtype=float),
                denom,
                where=denom > 0,
            )
        corr_matrix = pd.DataFrame(
            corr_values,
            index=cov_matrix.index,
            columns=cov_matrix.columns,
        )
    else:
        return 0.0

    if corr_matrix.shape[0] < 2:
        return 0.0

    values = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr_matrix.iloc[i, j]
            if pd.notna(v) and np.isfinite(v):
                values.append(float(v))

    return float(sum(values) / len(values)) if values else 0.0


def check_diversification_breakdown(average_correlation: float, threshold: float) -> bool:
    return average_correlation >= threshold


def detect_market_regime(
    portfolio_volatility: float,
    average_correlation: float,
    cfg: Any,
    diversification_breakdown: bool = False,
) -> Tuple[str, float]:
    vol_score = 0.0
    corr_score = 0.0
    if cfg.risk_off_portfolio_vol_threshold > cfg.risk_on_portfolio_vol_threshold:
        vol_score = clamp(
            (portfolio_volatility - cfg.risk_on_portfolio_vol_threshold)
            / (cfg.risk_off_portfolio_vol_threshold - cfg.risk_on_portfolio_vol_threshold),
            0.0,
            1.0,
        )
    if cfg.risk_off_correlation_threshold > cfg.risk_on_correlation_threshold:
        corr_score = clamp(
            (average_correlation - cfg.risk_on_correlation_threshold)
            / (cfg.risk_off_correlation_threshold - cfg.risk_on_correlation_threshold),
            0.0,
            1.0,
        )
    total_weight = max(cfg.risk_vol_score_weight + cfg.risk_corr_score_weight, 1e-9)
    risk_score = (
        cfg.risk_vol_score_weight * vol_score
        + cfg.risk_corr_score_weight * corr_score
    ) / total_weight

    if (
        diversification_breakdown
        or portfolio_volatility >= cfg.risk_off_portfolio_vol_threshold
        or average_correlation >= cfg.risk_off_correlation_threshold
    ):
        return "risk_off", max(risk_score, 0.85)
    if (
        portfolio_volatility <= cfg.risk_on_portfolio_vol_threshold
        and average_correlation <= cfg.risk_on_correlation_threshold
    ):
        return "risk_on", min(risk_score, 0.25)
    return "neutral", clamp(risk_score, 0.25, 0.85)


def adjust_exposure_by_risk(
    base_exposure: float,
    market_regime: str,
    risk_score: float,
    cfg: Any,
    diversification_breakdown: bool = False,
    portfolio_volatility: float = 0.0,
) -> float:
    regime_multiplier = {
        "risk_on": cfg.risk_on_exposure_multiplier,
        "neutral": cfg.neutral_exposure_multiplier,
        "risk_off": cfg.risk_off_exposure_multiplier,
    }.get(market_regime, cfg.neutral_exposure_multiplier)
    target_exposure = base_exposure * regime_multiplier
    if cfg.enable_volatility_targeting and portfolio_volatility > 1e-12:
        vol_scale = clamp(
            cfg.target_portfolio_volatility / portfolio_volatility,
            cfg.min_vol_target_scale,
            cfg.max_vol_target_scale,
        )
        target_exposure = min(target_exposure, base_exposure * vol_scale)
    if diversification_breakdown:
        target_exposure *= cfg.diversification_breakdown_exposure_multiplier
    return clamp(target_exposure, 0.0, base_exposure)


def adjust_weights_by_risk(
    weights: Dict[str, float],
    features: FeatureMap,
    returns_df: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    market_regime: str,
    diversification_breakdown: bool,
    cfg: Any,
) -> Dict[str, float]:
    if not weights:
        return {}

    common = [pair for pair in weights if pair in cov_matrix.index]
    regime_strength = {
        "risk_on": cfg.risk_on_weight_penalty_scale,
        "neutral": cfg.neutral_weight_penalty_scale,
        "risk_off": cfg.risk_off_weight_penalty_scale,
    }.get(market_regime, cfg.neutral_weight_penalty_scale)

    asset_volatility: Dict[str, float] = {}
    asset_correlation: Dict[str, float] = {}
    if common:
        diag = np.sqrt(np.maximum(np.diag(cov_matrix.loc[common, common].to_numpy(dtype=float)), 0.0))
        asset_volatility = {pair: float(diag[idx]) for idx, pair in enumerate(common)}
        corr_sample = returns_df.tail(cfg.risk_cov_window).dropna(how="all")
        corr_sample = corr_sample.loc[:, [pair for pair in common if pair in corr_sample.columns]]

        if not corr_sample.empty and corr_sample.shape[1] >= 2:
            valid_counts = corr_sample.notna().sum(axis=0)
            keep_cols = valid_counts[valid_counts >= max(10, cfg.risk_cov_window // 3)].index.tolist()
            corr_sample = corr_sample[keep_cols]

            if corr_sample.shape[1] >= 2:
                corr_matrix = corr_sample.corr(min_periods=max(8, cfg.risk_cov_window // 4))
                for pair in corr_matrix.columns:
                    others = corr_matrix.loc[pair, corr_matrix.columns != pair]
                    others = others[np.isfinite(others)]
                    asset_correlation[pair] = float(others.mean()) if len(others) else 0.0

    positive_vols = [value for value in asset_volatility.values() if value > 0]
    median_volatility = median(positive_vols) if positive_vols else 0.0
    core_assets = {item.strip().upper() for item in cfg.diversification_core_assets.split(",") if item.strip()}

    scaled_weights: Dict[str, float] = {}
    for pair, weight in weights.items():
        multiplier = 1.0
        asset_vol = asset_volatility.get(pair, 0.0)
        asset_corr = asset_correlation.get(pair, 0.0)

        if median_volatility > 1e-12 and asset_vol > median_volatility:
            vol_penalty = (asset_vol / median_volatility) - 1.0
            multiplier /= 1.0 + regime_strength * cfg.risk_weight_vol_penalty * vol_penalty

        if asset_corr > cfg.risk_on_correlation_threshold:
            corr_penalty = asset_corr - cfg.risk_on_correlation_threshold
            multiplier /= 1.0 + regime_strength * cfg.risk_weight_corr_penalty * corr_penalty

        if diversification_breakdown:
            base_symbol = pair.split("/")[0].upper()
            if core_assets and base_symbol in core_assets:
                multiplier *= cfg.diversification_core_asset_multiplier
            else:
                multiplier *= cfg.diversification_alt_weight_multiplier

        if features.get(pair, {}).get("vol60", 0.0) > cfg.vol_cap * cfg.high_vol_feature_cutoff_multiplier:
            multiplier *= cfg.high_vol_feature_weight_multiplier

        scaled_weights[pair] = max(weight * multiplier, 0.0)

    total_scaled = sum(scaled_weights.values())
    total_raw = sum(weights.values())
    if total_scaled <= 1e-12 or total_raw <= 1e-12:
        return {}
    return {
        pair: value / total_scaled * total_raw
        for pair, value in scaled_weights.items()
        if value > 0
    }


class RegimeFilter:
    def __init__(self, symbols: List[str], btc_symbol: str = "BTC/USD", maxlen: int = 200):
        self.symbols = list(symbols)
        self.btc_symbol = btc_symbol
        self.maxlen = maxlen
        self.price_history = {s: deque(maxlen=maxlen) for s in symbols}
        self.current_regime = None

    def sync_symbols(self, symbols: List[str]) -> None:
        """
        universe 变化时保留已有 symbol 的历史，只增删映射，不整对象重建。
        """
        new_symbols = list(symbols)
        new_set = set(new_symbols)
        old_set = set(self.price_history.keys())

        # 新增 symbol：创建新 deque
        for symbol in new_symbols:
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.maxlen)

        # 删除不再需要的 symbol
        for symbol in list(old_set - new_set):
            self.price_history.pop(symbol, None)

        # 保持迭代顺序与当前 universe 一致
        self.symbols = new_symbols
        self.price_history = {symbol: self.price_history[symbol] for symbol in self.symbols}

    def update_market_data(self, tickers: Dict[str, Dict[str, float]]) -> None:
        for symbol in self.symbols:
            if symbol in tickers:
                price = tickers[symbol].get("LastPrice", 0)
                if price > 0:
                    self.price_history[symbol].append(float(price))

    def _to_df(self) -> pd.DataFrame:
        non_empty = {
            k: list(v)
            for k, v in self.price_history.items()
            if len(v) > 0
        }
        if not non_empty:
            return pd.DataFrame()
        df = pd.DataFrame(non_empty)
        return df.ffill().dropna(how="all")

    def detect_regime(self) -> Dict[str, float]:
        price_df = self._to_df()

        if self.btc_symbol not in price_df or len(price_df[self.btc_symbol]) < 10:
            self.current_regime = {"regime": "neutral", "risk_multiplier": 0.5}
            return self.current_regime

        returns_df = price_df.pct_change(fill_method=None)
        btc_price = price_df[self.btc_symbol]
        ma = btc_price.rolling(min(50, len(btc_price))).mean()
        if len(btc_price) < 10 or np.isnan(ma.iloc[-1]):
            self.current_regime = {"regime": "neutral", "risk_multiplier": 0.5}
            return self.current_regime

        btc_trend = btc_price.iloc[-1] / ma.iloc[-1] - 1.0

        btc_returns = returns_df[self.btc_symbol]
        vol = btc_returns.rolling(min(20, len(btc_returns))).std()
        vol_window = vol.dropna()
        if len(vol_window) < 20:
            vol_pct = 0.5
        else:
            current = vol.iloc[-1]
            vol_pct = float((vol_window < current).mean())

        latest_returns = returns_df.iloc[-1].dropna()
        if len(latest_returns) == 0:
            return self.current_regime or {"regime": "neutral", "risk_multiplier": 0.5}
        breadth = float((latest_returns > 0).mean())

        score = 0
        if btc_trend > 0.01:
            score += 2
        elif btc_trend > 0.002:
            score += 1
        elif btc_trend < -0.01:
            score -= 2
        elif btc_trend < -0.002:
            score -= 1

        if breadth > 0.6:
            score += 1
        if breadth < 0.3:
            score -= 1
        if float(latest_returns.mean()) > 0:
            score += 1
        if vol_pct > 0.95:
            score -= 2
        elif vol_pct > 0.8:
            score -= 1

        prev = self.current_regime["regime"] if self.current_regime else "neutral"
        if vol_pct > 0.97:
            regime = "panic"
        elif prev == "trend":
            regime = "trend" if score >= 1 else "neutral"
        elif prev == "range":
            regime = "range" if score <= 0 else "neutral"
        else:
            if score >= 2:
                regime = "trend"
            elif score <= -1:
                regime = "range"
            else:
                regime = "neutral"

        mapping = {"trend": 1.0, "neutral": 0.6, "range": 0.5, "panic": 0.2}
        self.current_regime = {"regime": regime, "risk_multiplier": mapping[regime]}
        return self.current_regime


class MuModelWrapper:
    def __init__(
        self,
        model_path: Optional[str],
        meta_path: Optional[str],
        required: bool = False,
    ):
        self.model = None
        self.feature_names: List[str] = []
        self.feature_defaults: Dict[str, float] = {}
        self.ready = False
        self.error: Optional[str] = None
        self.model_path = model_path
        self.meta_path = meta_path
        self.required = required
        self._load()

    def _load(self) -> None:
        if not self.model_path or not self.meta_path:
            self.error = "MU_MODEL_PATH or MU_MODEL_META_PATH missing"
            if self.required:
                raise RuntimeError(self.error)
            return

        if not Path(self.model_path).exists() or not Path(self.meta_path).exists():
            self.error = (
                f"mu model or metadata file not found: "
                f"model_path={self.model_path}, meta_path={self.meta_path}"
            )
            if self.required:
                raise RuntimeError(self.error)
            return

        try:
            from xgboost import XGBRegressor

            model = XGBRegressor()
            model.load_model(self.model_path)

            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            feature_names = meta.get("feature_names")
            if not feature_names:
                feature_names = meta.get("feature_columns", [])

            self.feature_names = list(feature_names)

            feature_defaults = meta.get("feature_defaults")
            if not feature_defaults:
                feature_defaults = {name: 0.0 for name in self.feature_names}

            self.feature_defaults = dict(feature_defaults)

            if not self.feature_names:
                raise ValueError("feature_names/feature_columns missing in metadata")

            self.model = model
            self.ready = True
            self.error = None

        except Exception as exc:
            self.error = f"failed to load MU model: {exc}"
            self.ready = False
            if self.required:
                raise RuntimeError(self.error) from exc

    def predict(self, rows: List[Dict[str, float]]) -> List[float]:
        if not self.ready or self.model is None:
            if self.required:
                raise RuntimeError(
                    f"MU model unavailable during predict: {self.error or 'unknown error'}"
                )
            return [0.0 for _ in rows]

        import pandas as pd

        frame_rows = []
        for row in rows:
            data = {
                name: row.get(name, self.feature_defaults.get(name, 0.0))
                for name in self.feature_names
            }
            frame_rows.append(data)

        X = pd.DataFrame(frame_rows, columns=self.feature_names)
        preds = self.model.predict(X)
        return [float(x) for x in preds]


class MomentumStrategy:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.regime_filter: Optional[RegimeFilter] = None
        self.mu_weight = float(os.getenv("MU_BLEND_WEIGHT", "0.15"))
        self.fixed_weight = float(os.getenv("FIXED_BLEND_WEIGHT", str(1.0 - self.mu_weight)))
        repo_root = Path(__file__).resolve().parent
        default_model_path = repo_root / "artifacts" / "mu_xgb_model.json"
        default_meta_path = repo_root / "artifacts" / "mu_xgb_model.meta.json"

        self.mu_model_path = os.getenv("MU_MODEL_PATH", str(default_model_path))
        self.mu_model_meta_path = os.getenv("MU_MODEL_META_PATH", str(default_meta_path))

        self.mu_model_required = (
                os.getenv("MU_MODEL_REQUIRED", "true").strip().lower() == "true"
        )

        logger.info("MU model path=%s", self.mu_model_path)
        logger.info("MU meta path=%s", self.mu_model_meta_path)
        logger.info("MU model required=%s", self.mu_model_required)

        self.mu_model = MuModelWrapper(
            self.mu_model_path,
            self.mu_model_meta_path,
            required=self.mu_model_required,
        )

        # --- turnover reduction / less aggressive sell ---
        self.rank_retention_buffer = 0.18
        self.holding_bonus_floor = 0.22

        # anti-chase for new entries
        self.pump_chase_cutoff = 0.035
        self.pullback_entry_floor = -0.025
        self.range_keep_exposure = 0.30

    def trend_efficiency(self, prices: List[float], lookback: int) -> float:
        if len(prices) <= lookback:
            return 0.0
        net_move = abs(compute_return(prices, lookback))
        path_move = 0.0
        for offset in range(lookback):
            current = prices[-1 - offset]
            previous = prices[-2 - offset]
            if previous > 0:
                path_move += abs(current / previous - 1.0)
        return net_move / path_move if path_move > 1e-12 else 0.0

    def capped_inverse_vol_weights(self, strengths: List[Tuple[str, float]]) -> Dict[str, float]:
        remaining = self.cfg.target_gross_exposure
        pending = {pair: strength for pair, strength in strengths if strength > 0}
        weights: Dict[str, float] = {}
        while pending and remaining > 1e-12:
            total_strength = sum(pending.values())
            if total_strength <= 0:
                break
            capped_pairs: List[str] = []
            for pair, strength in pending.items():
                proposed = remaining * (strength / total_strength)
                if proposed >= self.cfg.max_single_weight:
                    weights[pair] = self.cfg.max_single_weight
                    remaining -= self.cfg.max_single_weight
                    capped_pairs.append(pair)
            if not capped_pairs:
                for pair, strength in pending.items():
                    weights[pair] = remaining * (strength / total_strength)
                break
            for pair in capped_pairs:
                pending.pop(pair, None)
        return {pair: weight for pair, weight in weights.items() if weight > 0}

    def _base_feature_block(
        self,
        history: Dict[str, Deque[Dict[str, float]]],
        trade_pairs: Dict[str, Dict[str, Any]],
    ) -> FeatureMap:
        features: FeatureMap = {}
        eligible_pairs: List[str] = []
        for pair, series in history.items():
            if pair not in trade_pairs or len(series) < self.cfg.min_history:
                continue
            prices = [entry["price"] for entry in series]
            price = prices[-1]
            returns_1m: List[float] = []
            max_lookback = min(60, len(prices) - 1)
            for offset in range(1, max_lookback + 1):
                current = prices[-offset]
                previous = prices[-offset - 1]
                if previous > 0:
                    returns_1m.append(current / previous - 1.0)
            if len(prices) < 20:
                continue

            ma20 = mean(prices[-20:])
            ma60 = mean(prices[-60:]) if len(prices) >= 60 else mean(prices)
            high20 = max(prices[-20:])
            low20 = min(prices[-20:])
            vol20 = stddev(returns_1m[:20]) if len(returns_1m) >= 20 else stddev(returns_1m)
            vol60 = stddev(returns_1m)
            bid = series[-1]["bid"]
            ask = series[-1]["ask"]
            spread = 0.0
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                if mid > 0:
                    spread = (ask - bid) / mid
            volume_values = [float(entry.get("unit_trade_value", 0.0)) for entry in list(series)[-20:]]
            current_volume = float(series[-1].get("unit_trade_value", 0.0))
            volume_z20 = zscore(current_volume, volume_values) if len(volume_values) >= 2 else 0.0

            quote_volume_values = [float(entry.get("quote_volume", entry.get("unit_trade_value", 0.0))) for entry in
                                   list(series)[-20:]]
            current_quote_volume = float(series[-1].get("quote_volume", series[-1].get("unit_trade_value", 0.0)))
            quote_volume_z20 = zscore(current_quote_volume, quote_volume_values) if len(
                quote_volume_values) >= 2 else 0.0

            # 当前 Roostoo history 里没有独立 quote_volume / trades 字段，
            # 暂时用 unit_trade_value 代理 quote_volume，trades 先设为 0
            quote_volume_z20 = volume_z20
            trades_z20 = 0.0

            feature = {
                "price": price,
                "ret1": compute_return(prices, 1) if len(prices) > 1 else 0.0,
                "ret3": compute_return(prices, 3) if len(prices) > 3 else 0.0,
                "ret5": compute_return(prices, 5) if len(prices) > 5 else 0.0,
                "ret15": compute_return(prices, 15) if len(prices) > 15 else 0.0,
                "ret30": compute_return(prices, 30) if len(prices) > 30 else 0.0,
                "ret60": compute_return(prices, 60) if len(prices) > 60 else 0.0,
                "dist_ma20": price / ma20 - 1.0 if ma20 > 0 else 0.0,
                "dist_ma60": price / ma60 - 1.0 if ma60 > 0 else 0.0,
                "vol20": vol20,
                "vol60": vol60,
                "trend_ratio60": compute_return(prices, 60) / max(vol60 * math.sqrt(max(len(returns_1m), 1)),
                                                                  1e-9) if len(prices) > 60 else 0.0,
                "efficiency20": self.trend_efficiency(prices, 20) if len(prices) >= 3 else 0.0,
                "range_position20": 0.5 if high20 <= low20 else clamp((price - low20) / (high20 - low20), 0.0, 1.0),
                "pullback20": price / high20 - 1.0 if high20 > 0 else 0.0,
                "spread": spread,
                "change_24h": float(series[-1].get("change_24h", 0.0)),
                "unit_trade_value": current_volume,
                "quote_volume": current_quote_volume,
                "volume_z20": volume_z20,
                "quote_volume_z20": quote_volume_z20,
            }
            features[pair] = feature
            if spread <= self.cfg.spread_threshold and current_volume >= self.cfg.min_24h_dollar_vol:
                eligible_pairs.append(pair)
        return {pair: features[pair] for pair in eligible_pairs}

    def compute_features(
        self,
        history: Dict[str, Deque[Dict[str, float]]],
        trade_pairs: Dict[str, Dict[str, Any]],
    ) -> FeatureMap:
        features = self._base_feature_block(history, trade_pairs)
        if not features:
            return {}

        pairs = list(features.keys())
        ret1_values = [features[p]["ret1"] for p in pairs]
        ret3_values = [features[p]["ret3"] for p in pairs]
        ret5_values = [features[p]["ret5"] for p in pairs]
        ret15_values = [features[p]["ret15"] for p in pairs]
        ret30_values = [features[p]["ret30"] for p in pairs]
        ret60_values = [features[p]["ret60"] for p in pairs]
        dist20_values = [features[p]["dist_ma20"] for p in pairs]
        dist60_values = [features[p]["dist_ma60"] for p in pairs]
        vol20_values = [features[p]["vol20"] for p in pairs]
        vol60_values = [features[p]["vol60"] for p in pairs]
        trend_values = [features[p]["trend_ratio60"] for p in pairs]
        efficiency_values = [features[p]["efficiency20"] for p in pairs]
        range_values = [features[p]["range_position20"] for p in pairs]
        pullback_values = [features[p]["pullback20"] for p in pairs]
        volumez_values = [features[p]["volume_z20"] for p in pairs]
        quote_volumez_values = [features[p]["quote_volume_z20"] for p in pairs]

        inference_rows: List[Dict[str, float]] = []
        inference_pairs: List[str] = []

        for pair in pairs:
            feature = features[pair]

            # fixed score (original prior-driven engine)
            fixed_score = (
                0.18 * zscore(feature["ret5"], ret5_values)
                + 0.22 * zscore(feature["ret15"], ret15_values)
                + 0.20 * zscore(feature["ret30"], ret30_values)
                + 0.18 * zscore(feature["ret60"], ret60_values)
                + 0.10 * zscore(feature["trend_ratio60"], trend_values)
                + 0.08 * zscore(feature["efficiency20"], efficiency_values)
                + 0.06 * zscore(feature["range_position20"], range_values)
                + 0.10 * zscore(-feature["dist_ma20"], [-v for v in dist20_values])
                + 0.12 * (-zscore(feature["vol60"], vol60_values))
            )
            if feature["dist_ma20"] > self.cfg.max_pump_distance:
                fixed_score -= 0.20 + 2.5 * (feature["dist_ma20"] - self.cfg.max_pump_distance)
            if feature["spread"] > self.cfg.spread_threshold * 0.7:
                fixed_score -= 0.15 * (feature["spread"] / max(self.cfg.spread_threshold, 1e-9))
            if feature["ret5"] < 0 and feature["pullback20"] < -0.03:
                fixed_score -= 0.12
            if feature["ret60"] < 0 and feature["ret5"] > 0.02:
                fixed_score -= 0.10
            feature["fixed_score"] = fixed_score

            # relative features for universal mu inference
            row = {
                "ret_1": feature["ret1"],
                "ret_3": feature["ret3"],
                "ret_5": feature["ret5"],
                "ret_15": feature["ret15"],
                "ret_30": feature["ret30"],
                "ret_60": feature["ret60"],
                "dist_ma20": feature["dist_ma20"],
                "dist_ma60": feature["dist_ma60"],
                "vol20": feature["vol20"],
                "vol60": feature["vol60"],
                "range_pos20": feature["range_position20"],
                "pullback20": feature["pullback20"],
                "volume_z20": feature["volume_z20"],
                "quote_volume_z20": feature["quote_volume_z20"],
                "hour": 0.0,
                "day": 0.0,

                "ret_1_cs_z": zscore(feature["ret1"], ret1_values),
                "ret_3_cs_z": zscore(feature["ret3"], ret3_values),
                "ret_5_cs_z": zscore(feature["ret5"], ret5_values),
                "ret_15_cs_z": zscore(feature["ret15"], ret15_values),
                "ret_30_cs_z": zscore(feature["ret30"], ret30_values),
                "ret_60_cs_z": zscore(feature["ret60"], ret60_values),
                "dist_ma20_cs_z": zscore(feature["dist_ma20"], dist20_values),
                "dist_ma60_cs_z": zscore(feature["dist_ma60"], dist60_values),
                "vol20_cs_z": zscore(feature["vol20"], vol20_values),
                "vol60_cs_z": zscore(feature["vol60"], vol60_values),
                "range_pos20_cs_z": zscore(feature["range_position20"], range_values),
                "pullback20_cs_z": zscore(feature["pullback20"], pullback_values),
                "volume_z20_cs_z": zscore(feature["volume_z20"], volumez_values),
                "quote_volume_z20_cs_z": zscore(feature["quote_volume_z20"], quote_volumez_values),
            }
            inference_pairs.append(pair)
            inference_rows.append(row)

        preds = self.mu_model.predict(inference_rows)
        mu_values = preds if preds else [0.0 for _ in inference_rows]
        mu_z_values = [zscore(v, mu_values) for v in mu_values] if len(mu_values) >= 2 else [0.0 for _ in mu_values]

        for i, pair in enumerate(inference_pairs):
            features[pair]["pred_mu"] = mu_values[i]
            features[pair]["pred_mu_z"] = mu_z_values[i]
            features[pair]["score"] = self.fixed_weight * features[pair]["fixed_score"] + self.mu_weight * features[pair]["pred_mu_z"]
        return features

    def market_snapshot(self, features: FeatureMap) -> MarketSnapshot:
        if not features:
            return MarketSnapshot()
        ret15_values = [feature["ret15"] for feature in features.values()]
        ret60_values = [feature["ret60"] for feature in features.values()]
        scores = [feature["score"] for feature in features.values()]
        return MarketSnapshot(
            median_ret15=median(ret15_values),
            median_ret60=median(ret60_values),
            up_ratio_15=sum(1 for value in ret15_values if value > 0) / len(ret15_values),
            positive_score_ratio=sum(1 for value in scores if value > 0) / len(scores),
            avg_score=mean(scores),
        )

    def risk_on(self, snapshot: MarketSnapshot, prev_risk_on: bool) -> bool:
        if prev_risk_on:
            return (
                snapshot.median_ret60 > self.cfg.regime_exit_median_60m_threshold
                and snapshot.up_ratio_15 >= self.cfg.regime_exit_up_ratio_threshold
                and snapshot.positive_score_ratio >= self.cfg.regime_exit_positive_score_ratio_threshold
            )
        return (
            snapshot.median_ret60 > self.cfg.market_median_60m_threshold
            and snapshot.up_ratio_15 >= self.cfg.market_up_ratio_threshold
            and snapshot.positive_score_ratio >= self.cfg.market_positive_score_ratio_threshold
        )

    def _select_ranked_with_retention(
        self,
        ranked: List[Tuple[str, Dict[str, float], float, float]],
        held_pairs: set[str],
    ) -> List[Tuple[str, Dict[str, float], float, float]]:
        if not ranked:
            return []

        ranked.sort(
            key=lambda item: (
                item[2],
                item[1].get("pred_mu", 0.0),
                item[1]["trend_ratio60"],
                item[1]["ret15"],
            ),
            reverse=True,
        )

        if len(ranked) <= self.cfg.top_n:
            return ranked

        selected = ranked[: self.cfg.top_n]
        selected_pairs = {item[0] for item in selected}
        cutoff_score = selected[-1][2]

        for item in ranked[self.cfg.top_n:]:
            pair, feature, ranking_score, threshold = item
            if pair not in held_pairs:
                continue
            if pair in selected_pairs:
                continue

            # 已有持仓只要没明显掉出 cutoff，就允许保留
            if ranking_score >= cutoff_score - self.rank_retention_buffer:
                selected.append(item)
                selected_pairs.add(pair)

        return selected

    def target_weights(
            self,
            features: FeatureMap,
            risk_on: bool,
            positions: Dict[str, float],
    ) -> Dict[str, float]:
        if not risk_on or not features:
            return {}

        held_pairs = set(positions)
        ranked: List[Tuple[str, Dict[str, float], float, float]] = []

        for pair, feature in features.items():
            is_held = pair in held_pairs
            threshold = self.cfg.exit_score_threshold if is_held else self.cfg.entry_score_threshold
            score = feature["score"]

            if score < threshold:
                continue

            # --- anti-chase for new entries only ---
            if not is_held:
                dist_ma20 = feature.get("dist_ma20", 0.0)
                pullback20 = feature.get("pullback20", 0.0)
                ret5 = feature.get("ret5", 0.0)
                ret15 = feature.get("ret15", 0.0)

                # 涨太快、离均线太远：不追
                if dist_ma20 > self.pump_chase_cutoff:
                    continue
                if ret5 > self.pump_chase_cutoff or ret15 > self.pump_chase_cutoff * 1.5:
                    continue

                # 回撤太深也不接
                if pullback20 < self.pullback_entry_floor:
                    continue

            # --- holding retention bonus ---
            holding_bonus = 0.0
            if is_held:
                holding_bonus = max(self.cfg.holding_score_bonus, self.holding_bonus_floor)

            ranking_score = score + holding_bonus

            # 对新仓再加一点“别追涨”的软惩罚，而不是只靠硬过滤
            if not is_held:
                if feature.get("dist_ma20", 0.0) > self.pump_chase_cutoff * 0.6:
                    ranking_score -= 0.12
                if feature.get("ret5", 0.0) > self.pump_chase_cutoff * 0.6:
                    ranking_score -= 0.10
                if feature.get("range_position20", 0.5) > 0.92:
                    ranking_score -= 0.08

            ranked.append((pair, feature, ranking_score, threshold))

        ranked = self._select_ranked_with_retention(ranked, held_pairs)
        if not ranked:
            return {}

        strengths: List[Tuple[str, float]] = []
        for pair, feature, ranking_score, threshold in ranked:
            vol = clamp(feature["vol60"], self.cfg.vol_floor, self.cfg.vol_cap)

            liquidity_multiplier = clamp(
                math.log1p(feature["unit_trade_value"] / max(self.cfg.min_24h_dollar_vol, 1.0)),
                0.75,
                1.35,
            )

            quality_multiplier = clamp(
                1.0
                + max(feature["trend_ratio60"], 0.0) * 0.08
                + feature["efficiency20"] * 0.20
                + max(feature.get("pred_mu_z", 0.0), 0.0) * 0.05,
                0.80,
                1.55,
            )

            # 已持仓再给一点强度保护，减少被 trim / replace
            if pair in held_pairs:
                quality_multiplier *= 1.08

            edge = max(ranking_score - threshold + 0.20, 0.05)
            strengths.append((pair, (edge * liquidity_multiplier * quality_multiplier) / vol))

        return self.capped_inverse_vol_weights(strengths)

    def _position_weight_proxy(
        self,
        positions: Dict[str, float],
        features: FeatureMap,
        history: Dict[str, Deque[Dict[str, float]]],
    ) -> Dict[str, float]:
        """
        用于在没有新 target，或者需要给风险层一个“现有仓位结构”输入时，
        构造一个稳定的权重代理。
        """
        if not positions:
            return {}

        strengths: List[Tuple[str, float]] = []
        for pair, quantity in positions.items():
            if quantity <= 0:
                continue

            feature = features.get(pair)
            if feature is not None:
                vol = clamp(feature.get("vol60", self.cfg.vol_floor), self.cfg.vol_floor, self.cfg.vol_cap)
                score = max(feature.get("score", 0.0), 0.0)
                strength = max(0.05, (0.30 + score) / vol)
            else:
                # 没 feature 时退化成等权强度
                strength = 1.0

            strengths.append((pair, strength))

        if not strengths:
            return {}

        return self.capped_inverse_vol_weights(strengths)

    def _range_keep_weights(
        self,
        positions: Dict[str, float],
        features: FeatureMap,
        history: Dict[str, Deque[Dict[str, float]]],
    ) -> Dict[str, float]:
        """
        range 环境下不直接清空，而是给现有持仓一个较小总暴露的保留权重。
        """
        base = self._position_weight_proxy(positions, features, history)
        if not base:
            return {}

        total = sum(base.values())
        if total <= 1e-12:
            return {}

        target_total = min(self.cfg.target_gross_exposure, self.range_keep_exposure)
        return {
            pair: weight / total * target_total
            for pair, weight in base.items()
            if weight > 0
        }

    def evaluate_portfolio_risk(
            self,
            history: Dict[str, Deque[Dict[str, float]]],
            trade_pairs: Dict[str, Dict[str, Any]],
            features: FeatureMap,
            raw_weights: Dict[str, float],
            positions: Dict[str, float],
    ) -> PortfolioRiskState:
        universe = sorted(pair for pair in trade_pairs if pair in history)

        price_df, data_frequency = load_price_data(
            history=history,
            universe=universe,
            frequency=self.cfg.risk_data_frequency,
            min_periods=self.cfg.risk_min_periods,
        )
        returns_df = compute_returns(price_df, method=self.cfg.risk_return_method)

        # 样本太少时，直接给一个保守但可运行的默认风险状态
        if returns_df.shape[0] < max(20, self.cfg.risk_cov_window // 2):
            return PortfolioRiskState(
                covariance_matrix={},
                portfolio_volatility=0.0,
                average_correlation=0.0,
                market_regime="neutral",
                risk_score=0.5,
                target_exposure=self.cfg.neutral_exposure_multiplier * self.cfg.target_gross_exposure,
                diversification_breakdown=False,
                data_frequency=data_frequency,
                raw_weights=raw_weights,
                adjusted_weights=raw_weights,
            )

        cov_matrix = compute_cov_matrix(
            returns_df,
            min_samples_per_asset=max(20, self.cfg.risk_cov_window // 2),
            min_periods_pairwise=max(10, self.cfg.risk_cov_window // 3),
        )
        logger.info(
            "Risk matrix: returns_shape=%s cov_shape=%s",
            tuple(returns_df.shape) if not returns_df.empty else (0, 0),
            tuple(cov_matrix.shape) if not cov_matrix.empty else (0, 0),
        )

        weight_proxy = raw_weights or self._position_weight_proxy(positions, features, history)
        if not weight_proxy and not cov_matrix.empty:
            equal_weight = self.cfg.target_gross_exposure / max(len(cov_matrix.columns), 1)
            weight_proxy = {pair: equal_weight for pair in cov_matrix.columns}

        portfolio_volatility = compute_portfolio_volatility(weight_proxy, cov_matrix)
        average_correlation = compute_average_correlation(
            returns_df=returns_df,
            cov_matrix=cov_matrix,
            window=self.cfg.risk_cov_window,
        )
        diversification_breakdown = check_diversification_breakdown(
            average_correlation=average_correlation,
            threshold=self.cfg.diversification_breakdown_corr_threshold,
        )
        market_regime, risk_score = detect_market_regime(
            portfolio_volatility=portfolio_volatility,
            average_correlation=average_correlation,
            cfg=self.cfg,
            diversification_breakdown=diversification_breakdown,
        )
        adjusted_weights = adjust_weights_by_risk(
            weights=raw_weights,
            features=features,
            returns_df=returns_df,
            cov_matrix=cov_matrix,
            market_regime=market_regime,
            diversification_breakdown=diversification_breakdown,
            cfg=self.cfg,
        )
        target_exposure = adjust_exposure_by_risk(
            base_exposure=self.cfg.target_gross_exposure,
            market_regime=market_regime,
            risk_score=risk_score,
            cfg=self.cfg,
            diversification_breakdown=diversification_breakdown,
            portfolio_volatility=portfolio_volatility,
        )

        return PortfolioRiskState(
            covariance_matrix=cov_matrix.round(8).to_dict() if not cov_matrix.empty else {},
            portfolio_volatility=portfolio_volatility,
            average_correlation=average_correlation,
            market_regime=market_regime,
            risk_score=risk_score,
            target_exposure=target_exposure,
            diversification_breakdown=diversification_breakdown,
            data_frequency=data_frequency,
            raw_weights=raw_weights,
            adjusted_weights=adjusted_weights,
        )

    def generate_signals(
            self,
            history: Dict[str, Deque[Dict[str, float]]],
            trade_pairs: Dict[str, Dict[str, Any]],
            positions: Dict[str, float],
            prev_risk_on: bool,
    ) -> Dict[str, Any]:
        symbols = sorted([
            pair for pair in trade_pairs
            if pair in history and len(history[pair]) > 0
        ])

        if self.regime_filter is None:
            logger.info("Initializing regime filter for %d symbols", len(symbols))
            self.regime_filter = RegimeFilter(symbols)
        else:
            old_symbols = set(self.regime_filter.price_history.keys())
            new_symbols = set(symbols)
            if old_symbols != new_symbols:
                added = sorted(new_symbols - old_symbols)
                removed = sorted(old_symbols - new_symbols)
                logger.info(
                    "Updating regime filter symbols. added=%s removed=%s total=%d",
                    added,
                    removed,
                    len(symbols),
                )
                self.regime_filter.sync_symbols(symbols)

        latest_tickers = {
            pair: {"LastPrice": history[pair][-1]["price"]}
            for pair in history
            if len(history[pair]) > 0
        }
        self.regime_filter.update_market_data(latest_tickers)
        regime = self.regime_filter.detect_regime()

        features = self.compute_features(history, trade_pairs)
        snapshot = self.market_snapshot(features)

        regime_name = regime["regime"]

        # 第一层：只做方向过滤，不做仓位缩放
        # trend / neutral 允许开新仓；range / panic 不开新仓
        allow_new_entries = (
                regime_name in ["trend", "neutral"]
                and self.risk_on(snapshot, prev_risk_on)
        )

        # 原始方向层目标权重：这里只表达“想买谁”，不表达最终总仓位
        raw_weights = self.target_weights(features, allow_new_entries, positions)

        # 如果处于 range / panic，不开新仓；
        # 但如果已经有持仓，则保留现有持仓作为风险层输入，让第二层决定缩多少仓
        if regime_name == "range":
            if positions:
                raw_weights = self._range_keep_weights(positions, features, history)
            else:
                raw_weights = {}
        elif regime_name == "panic":
            raw_weights = {}

        # 第二层：只负责风险覆盖与仓位缩放
        portfolio_risk = self.evaluate_portfolio_risk(
            history=history,
            trade_pairs=trade_pairs,
            features=features,
            raw_weights=raw_weights,
            positions=positions,
        )

        # 最终 risk_on 只表示“是否允许新增风险”
        # risk_off 时不新增；range / panic 时也不新增
        risk_on = allow_new_entries and portfolio_risk.market_regime != "risk_off"

        # 第二层先调结构，再统一缩放总 exposure
        final_weights = portfolio_risk.adjusted_weights or raw_weights
        if final_weights:
            total = sum(final_weights.values())
            if total > 0:
                final_weights = {
                    pair: weight / total * portfolio_risk.target_exposure
                    for pair, weight in final_weights.items()
                }
            else:
                final_weights = {}

        return {
            "features": features,
            "snapshot": snapshot,
            "risk_on": risk_on,
            "weights": final_weights,
            "pre_risk_weights": raw_weights,
            "portfolio_risk": {
                "covariance_matrix": portfolio_risk.covariance_matrix,
                "portfolio_volatility": portfolio_risk.portfolio_volatility,
                "average_correlation": portfolio_risk.average_correlation,
                "market_regime": portfolio_risk.market_regime,
                "risk_score": portfolio_risk.risk_score,
                "target_exposure": portfolio_risk.target_exposure,
                "diversification_breakdown": portfolio_risk.diversification_breakdown,
                "data_frequency": portfolio_risk.data_frequency,
                "raw_weights": portfolio_risk.raw_weights,
                "adjusted_weights": final_weights,
            },
            "regime": {
                **regime,
                "mu_ready": self.mu_model.ready,
                "mu_blend_weight": self.mu_weight,
                "fixed_blend_weight": self.fixed_weight,
                "mu_error": self.mu_model.error,
            },
        }