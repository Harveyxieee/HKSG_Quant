from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

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
    return ordered[mid] if len(ordered) % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2.0


def stddev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def zscore(value: float, values: List[float]) -> float:
    sigma = stddev(values)
    if sigma <= 1e-12:
        return 0.0
    return (value - mean(values)) / sigma


def compute_return(prices: List[float], lookback: int) -> float:
    if len(prices) <= lookback:
        return 0.0
    base_price = prices[-lookback - 1]
    last_price = prices[-1]
    return last_price / base_price - 1.0 if base_price > 0 else 0.0


def regression_r2(prices: List[float], window: int) -> float:
    if len(prices) < window or window < 3:
        return 0.0
    sample = prices[-window:]
    if any(price <= 0 for price in sample):
        return 0.0
    y = np.log(np.array(sample, dtype=float))
    x = np.arange(len(sample), dtype=float)
    x_centered = x - float(x.mean())
    y_centered = y - float(y.mean())
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-12:
        return 0.0
    slope = float(np.dot(x_centered, y_centered) / denom)
    intercept = float(y.mean()) - slope * float(x.mean())
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - float(y.mean())) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return clamp(1.0 - ss_res / ss_tot, 0.0, 1.0)


def volume_confirmation_score(prices: List[float], volumes: List[float], window: int) -> float:
    if len(prices) < window + 1 or len(volumes) < window:
        return 0.0
    up_volume = 0.0
    down_volume = 0.0
    signed_flow = 0.0
    total_volume = 0.0
    price_slice = prices[-(window + 1):]
    volume_slice = volumes[-window:]
    for idx in range(window):
        base_price = price_slice[idx]
        next_price = price_slice[idx + 1]
        volume = max(float(volume_slice[idx]), 0.0)
        if base_price <= 0 or volume <= 0:
            continue
        ret = next_price / base_price - 1.0
        total_volume += volume
        signed_flow += ret * volume
        if ret >= 0:
            up_volume += volume
        else:
            down_volume += volume
    if total_volume <= 1e-12:
        return 0.0
    flow_component = signed_flow / total_volume
    volume_bias = (up_volume - down_volume) / max(up_volume + down_volume, 1e-12)
    return clamp(flow_component * 25.0 + volume_bias, -1.0, 1.0)


def drawdown_recovery_score(prices: List[float], window: int) -> float:
    if len(prices) < window:
        return 0.0
    sample = prices[-window:]
    peak_idx = max(range(len(sample)), key=lambda idx: sample[idx])
    peak_price = sample[peak_idx]
    if peak_price <= 0 or peak_idx >= len(sample) - 1:
        return 0.0
    post_peak = sample[peak_idx:]
    trough_price = min(post_peak)
    current_price = sample[-1]
    drawdown = peak_price / trough_price - 1.0 if trough_price > 0 else 0.0
    if drawdown <= 1e-12:
        return 0.0
    return clamp((current_price - trough_price) / (peak_price - trough_price), 0.0, 1.0)


def orthogonalize_signal_maps(signal_map: Dict[str, Dict[str, float]], keys: List[str]) -> None:
    pairs = list(signal_map.keys())
    if len(pairs) < 3:
        for pair in pairs:
            for key in keys:
                signal_map[pair][f"{key}_ortho"] = signal_map[pair].get(key, 0.0)
        return
    matrix = np.array([[signal_map[pair].get(key, 0.0) for key in keys] for pair in pairs], dtype=float)
    for idx in range(matrix.shape[1]):
        sigma = float(np.std(matrix[:, idx]))
        matrix[:, idx] = 0.0 if sigma <= 1e-12 else (matrix[:, idx] - float(matrix[:, idx].mean())) / sigma
    ortho = matrix.copy()
    for idx in range(1, ortho.shape[1]):
        basis = ortho[:, :idx]
        target = matrix[:, idx]
        coeffs, *_ = np.linalg.lstsq(basis, target, rcond=None)
        residual = target - basis @ coeffs
        sigma = float(np.std(residual))
        ortho[:, idx] = 0.0 if sigma <= 1e-12 else (residual - float(residual.mean())) / sigma
    for row_idx, pair in enumerate(pairs):
        for col_idx, key in enumerate(keys):
            signal_map[pair][f"{key}_ortho"] = float(ortho[row_idx, col_idx])


@dataclass
class MarketSnapshot:
    median_ret15: float = 0.0
    median_ret60: float = 0.0
    up_ratio_15: float = 0.0
    positive_score_ratio: float = 0.0
    avg_score: float = 0.0
    avg_confidence: float = 0.0


class DirectionalRegimeFilter:
    def __init__(self, symbols: List[str], btc_symbol: str = "BTC/USD", maxlen: int = 240):
        self.symbols = list(symbols)
        self.btc_symbol = btc_symbol
        self.maxlen = maxlen
        self.price_history = {symbol: deque(maxlen=maxlen) for symbol in symbols}
        self.current_regime: Optional[Dict[str, float]] = None

    def sync_symbols(self, symbols: List[str]) -> None:
        new_symbols = list(symbols)
        for symbol in new_symbols:
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.maxlen)
        for symbol in list(self.price_history):
            if symbol not in new_symbols:
                self.price_history.pop(symbol, None)
        self.symbols = new_symbols
        self.price_history = {symbol: self.price_history[symbol] for symbol in self.symbols}

    def update_market_data(self, tickers: Dict[str, Dict[str, float]]) -> None:
        for symbol in self.symbols:
            if symbol in tickers:
                price = float(tickers[symbol].get("LastPrice", 0.0))
                if price > 0:
                    self.price_history[symbol].append(price)

    def detect_regime(self) -> Dict[str, float]:
        non_empty = {key: list(values) for key, values in self.price_history.items() if values}
        price_df = pd.DataFrame(non_empty).ffill().dropna(how="all") if non_empty else pd.DataFrame()
        if self.btc_symbol not in price_df or len(price_df[self.btc_symbol]) < 10:
            self.current_regime = {"regime": "neutral", "risk_multiplier": 0.6}
            return self.current_regime
        returns_df = price_df.pct_change(fill_method=None)
        btc_price = price_df[self.btc_symbol]
        ma = btc_price.rolling(min(50, len(btc_price))).mean()
        if np.isnan(ma.iloc[-1]):
            self.current_regime = {"regime": "neutral", "risk_multiplier": 0.6}
            return self.current_regime
        btc_trend = btc_price.iloc[-1] / ma.iloc[-1] - 1.0
        vol = returns_df[self.btc_symbol].rolling(min(20, len(returns_df))).std()
        vol_window = vol.dropna()
        vol_pct = 0.5 if len(vol_window) < 20 else float((vol_window < vol.iloc[-1]).mean())
        latest_returns = returns_df.iloc[-1].dropna()
        breadth = float((latest_returns > 0).mean()) if len(latest_returns) else 0.5
        score = 0
        score += 2 if btc_trend > 0.01 else 1 if btc_trend > 0.002 else -2 if btc_trend < -0.01 else -1 if btc_trend < -0.002 else 0
        score += 1 if breadth > 0.6 else -1 if breadth < 0.3 else 0
        score += 1 if len(latest_returns) and float(latest_returns.mean()) > 0 else 0
        score += -2 if vol_pct > 0.95 else -1 if vol_pct > 0.8 else 0
        prev = self.current_regime["regime"] if self.current_regime else "neutral"
        if vol_pct > 0.97:
            regime = "panic"
        elif prev == "trend":
            regime = "trend" if score >= 1 else "neutral"
        elif prev == "range":
            regime = "range" if score <= 0 else "neutral"
        else:
            regime = "trend" if score >= 2 else "range" if score <= -1 else "neutral"
        mapping = {"trend": 1.0, "neutral": 0.6, "range": 0.45, "panic": 0.2}
        self.current_regime = {"regime": regime, "risk_multiplier": mapping[regime]}
        return self.current_regime


class MuModelWrapper:
    def __init__(self, model_path: Optional[str], meta_path: Optional[str], required: bool = False):
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
            self.error = "MU model path missing"
            return
        if not Path(self.model_path).exists() or not Path(self.meta_path).exists():
            self.error = "MU model files missing"
            if self.required:
                raise RuntimeError(self.error)
            return
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor()
            model.load_model(self.model_path)
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.feature_names = list(meta.get("feature_names") or meta.get("feature_columns") or [])
            if not self.feature_names:
                raise ValueError("MU metadata missing feature names")
            self.feature_defaults = dict(meta.get("feature_defaults") or {name: 0.0 for name in self.feature_names})
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
                raise RuntimeError(self.error or "MU model unavailable")
            return [0.0 for _ in rows]
        frame = pd.DataFrame(
            [{name: row.get(name, self.feature_defaults.get(name, 0.0)) for name in self.feature_names} for row in rows],
            columns=self.feature_names,
        )
        preds = self.model.predict(frame)
        return [float(value) for value in preds]


class AlphaModel:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.mu_weight = float(os.getenv("MU_BLEND_WEIGHT", "0.15"))
        self.fixed_weight = float(os.getenv("FIXED_BLEND_WEIGHT", str(1.0 - self.mu_weight)))
        repo_root = Path(__file__).resolve().parent
        self.mu_model = MuModelWrapper(
            os.getenv("MU_MODEL_PATH", str(repo_root / "artifacts" / "mu_xgb_model.json")),
            os.getenv("MU_MODEL_META_PATH", str(repo_root / "artifacts" / "mu_xgb_model.meta.json")),
            required=os.getenv("MU_MODEL_REQUIRED", "false").strip().lower() == "true",
        )
        self.directional_filter = DirectionalRegimeFilter([])
        self.core_assets = {
            item.strip().upper()
            for item in getattr(self.cfg, "diversification_core_assets", "BTC,ETH").split(",")
            if item.strip()
        }
        self.preferred_assets = {
            item.strip().upper()
            for item in getattr(self.cfg, "preferred_assets", "").split(",")
            if item.strip()
        }
        self.blocked_assets = {
            item.strip().upper()
            for item in getattr(self.cfg, "blocked_assets", "").split(",")
            if item.strip()
        }

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

    def update_directional_regime(self, history: Dict[str, Deque[Dict[str, float]]], symbols: List[str]) -> Dict[str, float]:
        self.directional_filter.sync_symbols(symbols)
        latest_tickers = {pair: {"LastPrice": history[pair][-1]["price"]} for pair in history if len(history[pair]) > 0}
        self.directional_filter.update_market_data(latest_tickers)
        return self.directional_filter.detect_regime()

    def _base_feature_block(self, history: Dict[str, Deque[Dict[str, float]]], trade_pairs: Dict[str, Dict[str, Any]]) -> FeatureMap:
        features: FeatureMap = {}
        eligible_pairs: List[str] = []
        for pair, series in history.items():
            if pair not in trade_pairs or len(series) < self.cfg.min_history:
                continue
            base_symbol = pair.split("/")[0].upper()
            if base_symbol in self.blocked_assets:
                continue
            if self.preferred_assets and base_symbol not in self.preferred_assets:
                continue
            prices = [float(entry["price"]) for entry in series]
            if len(prices) < 20:
                continue
            returns_1m = []
            for offset in range(1, min(120, len(prices) - 1) + 1):
                current = prices[-offset]
                previous = prices[-offset - 1]
                if previous > 0:
                    returns_1m.append(current / previous - 1.0)
            price = prices[-1]
            ma20 = mean(prices[-20:])
            ma60 = mean(prices[-60:]) if len(prices) >= 60 else mean(prices)
            high20 = max(prices[-20:])
            low20 = min(prices[-20:])
            high15 = max(prices[-15:]) if len(prices) >= 15 else max(prices)
            low5 = min(prices[-5:]) if len(prices) >= 5 else min(prices)
            low15 = min(prices[-15:]) if len(prices) >= 15 else min(prices)
            high60 = max(prices[-60:]) if len(prices) >= 60 else max(prices)
            low60 = min(prices[-60:]) if len(prices) >= 60 else min(prices)
            bid = float(series[-1].get("bid", 0.0))
            ask = float(series[-1].get("ask", 0.0))
            spread = 0.0
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                spread = (ask - bid) / mid if mid > 0 else 0.0
            volume_values = [float(entry.get("unit_trade_value", 0.0)) for entry in list(series)[-20:]]
            current_volume = float(series[-1].get("unit_trade_value", 0.0))
            feature = {
                "price": price,
                "ret1": compute_return(prices, 1) if len(prices) > 1 else 0.0,
                "ret3": compute_return(prices, 3) if len(prices) > 3 else 0.0,
                "ret5": compute_return(prices, 5) if len(prices) > 5 else 0.0,
                "ret15": compute_return(prices, 15) if len(prices) > 15 else 0.0,
                "ret30": compute_return(prices, 30) if len(prices) > 30 else 0.0,
                "ret60": compute_return(prices, 60) if len(prices) > 60 else 0.0,
                "ret120": compute_return(prices, 120) if len(prices) > 120 else compute_return(prices, min(60, len(prices) - 1)),
                "dist_ma20": price / ma20 - 1.0 if ma20 > 0 else 0.0,
                "dist_ma60": price / ma60 - 1.0 if ma60 > 0 else 0.0,
                "vol20": stddev(returns_1m[:20]) if len(returns_1m) >= 20 else stddev(returns_1m),
                "vol60": stddev(returns_1m[:60]) if len(returns_1m) >= 60 else stddev(returns_1m),
                "trend_ratio60": compute_return(prices, 60) / max(stddev(returns_1m[:60]) * math.sqrt(max(min(len(returns_1m), 60), 1)), 1e-9) if len(prices) > 60 else 0.0,
                "efficiency20": self.trend_efficiency(prices, 20) if len(prices) >= 21 else 0.0,
                "range_position20": 0.5 if high20 <= low20 else clamp((price - low20) / (high20 - low20), 0.0, 1.0),
                "range_position60": 0.5 if high60 <= low60 else clamp((price - low60) / (high60 - low60), 0.0, 1.0),
                "pullback20": price / high20 - 1.0 if high20 > 0 else 0.0,
                "pullback60": price / high60 - 1.0 if high60 > 0 else 0.0,
                "pullback15": price / high15 - 1.0 if high15 > 0 else 0.0,
                "breakout20": price / high20 - 1.0 if high20 > 0 else 0.0,
                "breakout60": price / high60 - 1.0 if high60 > 0 else 0.0,
                "rebound_from_low5": price / low5 - 1.0 if low5 > 0 else 0.0,
                "rebound_from_low15": price / low15 - 1.0 if low15 > 0 else 0.0,
                "shock_drop_5m": price / max(prices[-min(5, len(prices)):]) - 1.0 if len(prices) >= 2 else 0.0,
                "spread": spread,
                "change_24h": float(series[-1].get("change_24h", 0.0)),
                "unit_trade_value": current_volume,
                "volume_z20": zscore(current_volume, volume_values) if len(volume_values) >= 2 else 0.0,
                "trend_stability20": regression_r2(prices, 20),
                "trend_stability60": regression_r2(prices, min(60, len(prices))),
                "volume_confirmation20": volume_confirmation_score(prices, volume_values, min(20, len(volume_values))),
                "recovery_after_drawdown20": drawdown_recovery_score(prices, 20),
            }
            features[pair] = feature
            min_volume_threshold = self.cfg.min_24h_dollar_vol
            if base_symbol not in self.core_assets:
                min_volume_threshold = max(min_volume_threshold, getattr(self.cfg, "liquid_asset_volume_threshold", min_volume_threshold))
            if spread <= self.cfg.spread_threshold and current_volume >= min_volume_threshold:
                eligible_pairs.append(pair)
        return {pair: features[pair] for pair in eligible_pairs}

    def compute_features(self, history: Dict[str, Deque[Dict[str, float]]], trade_pairs: Dict[str, Dict[str, Any]]) -> FeatureMap:
        features = self._base_feature_block(history, trade_pairs)
        if not features:
            return {}
        pairs = list(features.keys())
        btc_feature = features.get("BTC/USD")
        eth_feature = features.get("ETH/USD")
        btc_ret15 = btc_feature["ret15"] if btc_feature else median([features[p]["ret15"] for p in pairs])
        btc_ret60 = btc_feature["ret60"] if btc_feature else median([features[p]["ret60"] for p in pairs])
        eth_ret15 = eth_feature["ret15"] if eth_feature else btc_ret15
        eth_ret60 = eth_feature["ret60"] if eth_feature else btc_ret60
        sections = {key: [features[p][key] for p in pairs] for key in ["ret15", "ret30", "ret60", "ret120", "dist_ma20", "vol60", "breakout20", "breakout60", "pullback20", "volume_z20", "trend_stability20", "trend_stability60", "volume_confirmation20", "recovery_after_drawdown20"]}
        for pair in pairs:
            feature = features[pair]
            rs15 = 0.5 * (feature["ret15"] - btc_ret15) + 0.5 * (feature["ret15"] - eth_ret15)
            rs60 = 0.5 * (feature["ret60"] - btc_ret60) + 0.5 * (feature["ret60"] - eth_ret60)
            feature["relative_strength"] = 0.45 * rs15 + 0.55 * rs60
        sections["relative_strength"] = [features[p]["relative_strength"] for p in pairs]
        signal_map: Dict[str, Dict[str, float]] = {}
        inference_rows: List[Dict[str, float]] = []
        for pair in pairs:
            feature = features[pair]
            signal_map[pair] = {
                "multi_horizon_momentum": 0.16 * zscore(feature["ret15"], sections["ret15"]) + 0.18 * zscore(feature["ret30"], sections["ret30"]) + 0.24 * zscore(feature["ret60"], sections["ret60"]) + 0.30 * zscore(feature["ret120"], sections["ret120"]) + 0.12 * zscore(feature["relative_strength"], sections["relative_strength"]),
                "volatility_breakout": 0.45 * zscore(feature["breakout20"], sections["breakout20"]) + 0.30 * zscore(feature["breakout60"], sections["breakout60"]) + 0.25 * zscore(feature["volume_confirmation20"], sections["volume_confirmation20"]),
                "mean_reversion": 0.30 * zscore(-feature["dist_ma20"], [-v for v in sections["dist_ma20"]]) + 0.25 * zscore(feature["pullback20"], sections["pullback20"]) + 0.20 * zscore(feature["recovery_after_drawdown20"], sections["recovery_after_drawdown20"]) + 0.25 * zscore(feature["rebound_from_low15"], [features[p]["rebound_from_low15"] for p in pairs]),
                "trend_stability": 0.65 * zscore(feature["trend_stability20"], sections["trend_stability20"]) + 0.35 * zscore(feature["trend_stability60"], sections["trend_stability60"]),
                "volume_confirmation": 0.55 * zscore(feature["volume_confirmation20"], sections["volume_confirmation20"]) + 0.25 * zscore(feature["volume_z20"], sections["volume_z20"]) + 0.20 * zscore(feature["relative_strength"], sections["relative_strength"]),
            }
            inference_rows.append({
                "ret_1": feature["ret1"], "ret_3": feature["ret3"], "ret_5": feature["ret5"], "ret_15": feature["ret15"], "ret_30": feature["ret30"], "ret_60": feature["ret60"], "dist_ma20": feature["dist_ma20"], "dist_ma60": feature["dist_ma60"], "vol20": feature["vol20"], "vol60": feature["vol60"], "range_pos20": feature["range_position20"], "pullback20": feature["pullback20"], "volume_z20": feature["volume_z20"], "trend_stability20": feature["trend_stability20"], "relative_strength": feature["relative_strength"], "volume_confirmation20": feature["volume_confirmation20"], "recovery_after_drawdown20": feature["recovery_after_drawdown20"], "hour": 0.0, "day": 0.0,
            })
        orthogonalize_signal_maps(signal_map, ["multi_horizon_momentum", "volatility_breakout", "mean_reversion", "trend_stability", "volume_confirmation"])
        preds = self.mu_model.predict(inference_rows)
        pred_z = [zscore(value, preds) for value in preds] if len(preds) >= 2 else [0.0 for _ in preds]
        for idx, pair in enumerate(pairs):
            feature = features[pair]
            for key, value in signal_map[pair].items():
                feature[key] = value
            fixed_score = 0.38 * feature["multi_horizon_momentum_ortho"] + 0.22 * feature["volatility_breakout_ortho"] + 0.08 * feature["mean_reversion_ortho"] + 0.18 * feature["trend_stability_ortho"] + 0.14 * feature["volume_confirmation_ortho"]
            if feature["dist_ma20"] > getattr(self.cfg, "max_pump_distance", 0.05):
                fixed_score -= 0.20 + 2.0 * (feature["dist_ma20"] - getattr(self.cfg, "max_pump_distance", 0.05))
            if feature["spread"] > getattr(self.cfg, "spread_threshold", 0.006) * 0.7:
                fixed_score -= 0.15 * (feature["spread"] / max(getattr(self.cfg, "spread_threshold", 0.006), 1e-9))
            fixed_score += 0.06 * zscore(feature["relative_strength"], sections["relative_strength"])
            base_symbol = pair.split("/")[0].upper()
            if base_symbol in self.core_assets:
                fixed_score += 0.02
            elif self.preferred_assets and base_symbol not in self.preferred_assets:
                fixed_score -= 0.10
            feature["fixed_score"] = fixed_score
            feature["pred_mu"] = preds[idx] if idx < len(preds) else 0.0
            feature["pred_mu_z"] = pred_z[idx] if idx < len(pred_z) else 0.0
            feature["alpha_score"] = self.fixed_weight * fixed_score + self.mu_weight * feature["pred_mu_z"]
            agreement = 1.0 / (1.0 + stddev([feature["multi_horizon_momentum_ortho"], feature["volatility_breakout_ortho"], feature["mean_reversion_ortho"], feature["trend_stability_ortho"], feature["volume_confirmation_ortho"]]))
            liquidity_score = clamp(math.log1p(feature["unit_trade_value"] / max(getattr(self.cfg, "min_24h_dollar_vol", 120000.0), 1.0)), 0.0, 1.0)
            spread_score = clamp(1.0 - feature["spread"] / max(getattr(self.cfg, "spread_threshold", 0.006), 1e-9), 0.0, 1.0)
            feature["confidence"] = clamp(0.42 * min(abs(feature["alpha_score"]) / 2.5, 1.0) + 0.23 * agreement + 0.18 * liquidity_score + 0.09 * spread_score + 0.08 * clamp(feature["trend_stability20"], 0.0, 1.0), 0.0, 1.0)
            feature["score"] = feature["alpha_score"]
            base_symbol = pair.split("/")[0].upper()
            if base_symbol in self.core_assets:
                feature["asset_bucket"] = "core"
            elif feature["unit_trade_value"] >= getattr(self.cfg, "liquid_asset_volume_threshold", 400000.0):
                feature["asset_bucket"] = "liquid"
            else:
                feature["asset_bucket"] = "satellite"
            state = "neutral"
            if (
                feature.get("breakout20", 0.0) > 0.003
                and feature.get("ret15", 0.0) > 0.0
                and feature.get("trend_stability_ortho", 0.0) > 0.0
            ):
                state = "breakout_acceleration"
            elif (
                feature.get("ret60", 0.0) > 0.0
                and feature.get("trend_stability_ortho", 0.0) > 0.10
                and feature.get("multi_horizon_momentum_ortho", 0.0) > 0.0
            ):
                state = "trend_follow"
            elif (
                feature.get("shock_drop_5m", 0.0) <= getattr(self.cfg, "flash_crash_drop_threshold", -0.03)
                and feature.get("rebound_from_low5", 0.0) >= max(
                    getattr(self.cfg, "flash_crash_rebound_threshold", 0.008),
                    getattr(self.cfg, "recovery_rebound_confirmation", 0.008),
                )
                and feature.get("ret1", 0.0) > 0.0
                and feature.get("volume_confirmation20", 0.0) >= getattr(self.cfg, "recovery_volume_confirmation_floor", 0.0)
            ):
                state = "shock_rebound"
            elif (
                feature.get("pullback20", 0.0) <= getattr(self.cfg, "recovery_pullback_threshold", -0.025)
                and feature.get("rebound_from_low5", 0.0) >= getattr(self.cfg, "recovery_rebound_confirmation", 0.008)
                and feature.get("ret3", 0.0) > 0.0
            ):
                state = "rebound_confirmed"
            elif (
                feature.get("rebound_from_low15", 0.0) > 0.01
                and feature.get("ret3", 0.0) < 0.0
                and feature.get("pullback15", 0.0) < -0.01
            ):
                state = "failed_rebound"
            elif (
                abs(feature.get("ret15", 0.0)) < 0.004
                and abs(feature.get("dist_ma20", 0.0)) < 0.012
                and feature.get("trend_stability20", 0.0) < 0.25
            ):
                state = "range_chop"
            feature["setup_state"] = state
        return features

    def market_snapshot(self, features: FeatureMap) -> MarketSnapshot:
        if not features:
            return MarketSnapshot()
        ret15_values = [feature["ret15"] for feature in features.values()]
        ret60_values = [feature["ret60"] for feature in features.values()]
        scores = [feature["score"] for feature in features.values()]
        confidence = [feature.get("confidence", 0.0) for feature in features.values()]
        return MarketSnapshot(
            median_ret15=median(ret15_values),
            median_ret60=median(ret60_values),
            up_ratio_15=sum(1 for value in ret15_values if value > 0) / len(ret15_values),
            positive_score_ratio=sum(1 for value in scores if value > 0) / len(scores),
            avg_score=mean(scores),
            avg_confidence=mean(confidence),
        )

import math
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FeatureMap = Dict[str, Dict[str, float]]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class PortfolioRiskState:
    covariance_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    portfolio_volatility: float = 0.0
    average_correlation: float = 0.0
    market_regime: str = "neutral"
    risk_score: float = 0.0
    target_exposure: float = 0.0
    diversification_breakdown: bool = False
    correlation_shock: bool = False
    alpha_aggressiveness: float = 1.0
    drawdown_scale: float = 1.0
    data_frequency: str = "1m"
    raw_weights: Dict[str, float] = field(default_factory=dict)
    adjusted_weights: Dict[str, float] = field(default_factory=dict)


def _resample_price_frame(price_df: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    candidate = price_df.resample(rule).last() if rule else price_df.copy()
    candidate = candidate.sort_index().ffill().dropna(how="all")
    return candidate.loc[:, candidate.notna().sum() >= 2]


def load_price_data(history: Dict[str, Deque[Dict[str, float]]], universe: List[str], frequency: str = "auto", min_periods: int = 30) -> Tuple[pd.DataFrame, str]:
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
        for label, rule in [("daily", "1D"), ("hourly", "1h"), ("raw", None)]:
            candidate = _resample_price_frame(price_df, rule)
            if len(candidate) >= min_periods:
                return candidate, label
        return _resample_price_frame(price_df, None), "raw"
    mapping = {"daily": "1D", "hourly": "1h", "raw": None}
    return _resample_price_frame(price_df, mapping.get(frequency, None)), frequency if frequency in mapping else "raw"


def compute_returns(price_df: pd.DataFrame, method: str = "pct_change") -> pd.DataFrame:
    if price_df.empty:
        return pd.DataFrame()
    returns_df = np.log(price_df).diff() if method == "log" else price_df.pct_change(fill_method=None)
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return returns_df.loc[:, returns_df.notna().sum() >= 2]


def compute_cov_matrix(returns_df: pd.DataFrame, min_samples_per_asset: int = 30, min_periods_pairwise: int = 20, shrinkage: float = 0.25) -> pd.DataFrame:
    if returns_df.empty:
        return pd.DataFrame()
    valid_counts = returns_df.notna().sum(axis=0)
    keep_cols = valid_counts[valid_counts >= min_samples_per_asset].index.tolist()
    returns_df = returns_df[keep_cols]
    if returns_df.shape[1] == 0:
        return pd.DataFrame()
    cov = returns_df.cov(min_periods=min_periods_pairwise).fillna(0.0)
    cov_values = cov.to_numpy(dtype=float)
    shrunk = (1.0 - shrinkage) * cov_values + shrinkage * np.diag(np.diag(cov_values))
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


def compute_portfolio_volatility(weights: Dict[str, float], cov_matrix: pd.DataFrame) -> float:
    if not weights or cov_matrix.empty:
        return 0.0
    common = [pair for pair in weights if pair in cov_matrix.index]
    if not common:
        return 0.0
    vector = np.array([weights[pair] for pair in common], dtype=float)
    sigma = cov_matrix.loc[common, common].to_numpy(dtype=float)
    return math.sqrt(max(float(vector.T @ sigma @ vector), 0.0))


def compute_average_correlation(returns_df: Optional[pd.DataFrame] = None, cov_matrix: Optional[pd.DataFrame] = None, window: int = 60, min_samples_per_asset: int = 30, min_periods_pairwise: int = 20) -> float:
    if returns_df is not None and not returns_df.empty:
        sample = returns_df.tail(window).dropna(how="all")
        valid_counts = sample.notna().sum(axis=0)
        sample = sample[valid_counts[valid_counts >= min_samples_per_asset].index.tolist()]
        if sample.shape[1] < 2:
            return 0.0
        corr_matrix = sample.corr(min_periods=min_periods_pairwise)
    elif cov_matrix is not None and not cov_matrix.empty:
        diag = np.sqrt(np.maximum(np.diag(cov_matrix.to_numpy(dtype=float)), 0.0))
        denom = np.outer(diag, diag)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_values = np.divide(cov_matrix.to_numpy(dtype=float), denom, where=denom > 0)
        corr_matrix = pd.DataFrame(corr_values, index=cov_matrix.index, columns=cov_matrix.columns)
    else:
        return 0.0
    values = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            if pd.notna(value) and np.isfinite(value):
                values.append(float(value))
    return float(sum(values) / len(values)) if values else 0.0


class RiskModel:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.previous_average_correlation = 0.0

    def _build_risk_universe(self, history: Dict[str, Deque[Dict[str, float]]], trade_pairs: Dict[str, Dict[str, Any]], raw_weights: Dict[str, float], positions: Dict[str, float]) -> List[str]:
        focus_pairs: List[str] = []
        seen: set[str] = set()

        def add_pair(pair: str) -> None:
            if pair not in trade_pairs or pair not in history or pair in seen or len(history[pair]) <= 0:
                return
            seen.add(pair)
            focus_pairs.append(pair)

        for pair in raw_weights:
            add_pair(pair)
        for pair in positions:
            add_pair(pair)
        for pair in ("BTC/USD", "ETH/USD"):
            add_pair(pair)
        if len(focus_pairs) >= 2:
            return focus_pairs
        ranked_liquidity = sorted(
            (pair for pair in trade_pairs if pair in history and len(history[pair]) > 0),
            key=lambda pair: float(history[pair][-1].get("unit_trade_value", 0.0)),
            reverse=True,
        )
        for pair in ranked_liquidity:
            add_pair(pair)
            if len(focus_pairs) >= 8:
                break
        return focus_pairs

    def _drawdown_scale(self, current_drawdown: float) -> float:
        max_dd = max(getattr(self.cfg, "max_portfolio_drawdown", 0.10), 1e-9)
        floor = getattr(self.cfg, "drawdown_exposure_floor", 0.25)
        return clamp(1.0 - 0.85 * clamp(current_drawdown / max_dd, 0.0, 1.0), floor, 1.0)

    def _detect_market_regime(self, portfolio_volatility: float, average_correlation: float, diversification_breakdown: bool, correlation_shock: bool) -> Tuple[str, float]:
        risk_on_vol = getattr(self.cfg, "risk_on_portfolio_vol_threshold", 0.015)
        risk_off_vol = getattr(self.cfg, "risk_off_portfolio_vol_threshold", 0.035)
        risk_on_corr = getattr(self.cfg, "risk_on_correlation_threshold", 0.35)
        risk_off_corr = getattr(self.cfg, "risk_off_correlation_threshold", 0.65)
        vol_score = clamp((portfolio_volatility - risk_on_vol) / max(risk_off_vol - risk_on_vol, 1e-9), 0.0, 1.0)
        corr_score = clamp((average_correlation - risk_on_corr) / max(risk_off_corr - risk_on_corr, 1e-9), 0.0, 1.0)
        risk_score = 0.5 * vol_score + 0.5 * corr_score
        if diversification_breakdown or correlation_shock or portfolio_volatility >= risk_off_vol or average_correlation >= risk_off_corr:
            return "risk_off", max(risk_score, 0.85)
        if portfolio_volatility <= risk_on_vol and average_correlation <= risk_on_corr:
            return "risk_on", min(risk_score, 0.25)
        return "neutral", clamp(risk_score, 0.25, 0.85)

    def evaluate(self, history: Dict[str, Deque[Dict[str, float]]], trade_pairs: Dict[str, Dict[str, Any]], features: FeatureMap, raw_weights: Dict[str, float], positions: Dict[str, float], current_drawdown: float = 0.0) -> PortfolioRiskState:
        universe = self._build_risk_universe(history, trade_pairs, raw_weights, positions)
        risk_frequency = getattr(self.cfg, "risk_data_frequency", "hourly")
        if risk_frequency in ("auto", "raw"):
            risk_frequency = "hourly"
        price_df, data_frequency = load_price_data(history, universe, frequency=risk_frequency, min_periods=getattr(self.cfg, "risk_min_periods", 60))
        returns_df = compute_returns(price_df, method=getattr(self.cfg, "risk_return_method", "log"))
        conservative_exposure = getattr(self.cfg, "neutral_exposure_multiplier", 0.7) * getattr(self.cfg, "target_gross_exposure", 0.72)
        minimum_samples = max(20, getattr(self.cfg, "risk_cov_window", 60) // 2)
        if returns_df.shape[0] < minimum_samples:
            return PortfolioRiskState(target_exposure=conservative_exposure, risk_score=0.5, alpha_aggressiveness=0.85, drawdown_scale=self._drawdown_scale(current_drawdown), data_frequency=data_frequency, raw_weights=raw_weights)
        cov_matrix = compute_cov_matrix(returns_df, min_samples_per_asset=max(20, getattr(self.cfg, "risk_cov_window", 60) // 2), min_periods_pairwise=max(10, getattr(self.cfg, "risk_cov_window", 60) // 3), shrinkage=getattr(self.cfg, "covariance_shrinkage", 0.25))
        if cov_matrix.shape[0] < 2:
            return PortfolioRiskState(covariance_matrix=cov_matrix.round(8).to_dict() if not cov_matrix.empty else {}, target_exposure=conservative_exposure, risk_score=0.5, alpha_aggressiveness=0.85, drawdown_scale=self._drawdown_scale(current_drawdown), data_frequency=data_frequency, raw_weights=raw_weights)
        weight_proxy = {pair: weight for pair, weight in raw_weights.items() if pair in cov_matrix.columns and weight > 0}
        if not weight_proxy:
            positive_scores = sorted(((pair, max(features.get(pair, {}).get("score", 0.0), 0.0)) for pair in features if pair in cov_matrix.columns), key=lambda item: item[1], reverse=True)[: max(3, getattr(self.cfg, "top_n", 5))]
            total_score = sum(score for _, score in positive_scores)
            weight_proxy = {pair: score / total_score for pair, score in positive_scores if total_score > 1e-12 and score > 0}
            if not weight_proxy:
                equal_weight = 1.0 / max(len(cov_matrix.columns), 1)
                weight_proxy = {pair: equal_weight for pair in cov_matrix.columns}
        portfolio_volatility = compute_portfolio_volatility(weight_proxy, cov_matrix)
        average_correlation = compute_average_correlation(returns_df=returns_df, cov_matrix=cov_matrix, window=getattr(self.cfg, "risk_cov_window", 60), min_samples_per_asset=max(10, getattr(self.cfg, "risk_cov_window", 60) // 3), min_periods_pairwise=max(8, getattr(self.cfg, "risk_cov_window", 60) // 4))
        diversification_breakdown = average_correlation >= getattr(self.cfg, "diversification_breakdown_corr_threshold", 0.75)
        correlation_shock = self.previous_average_correlation > 0 and (average_correlation - self.previous_average_correlation) >= getattr(self.cfg, "correlation_shock_delta", 0.12)
        self.previous_average_correlation = average_correlation
        market_regime, risk_score = self._detect_market_regime(portfolio_volatility, average_correlation, diversification_breakdown, correlation_shock)
        drawdown_scale = self._drawdown_scale(current_drawdown)
        regime_multiplier = {"risk_on": getattr(self.cfg, "risk_on_exposure_multiplier", 1.0), "neutral": getattr(self.cfg, "neutral_exposure_multiplier", 0.7), "risk_off": getattr(self.cfg, "risk_off_exposure_multiplier", 0.35)}.get(market_regime, getattr(self.cfg, "neutral_exposure_multiplier", 0.7))
        target_exposure = getattr(self.cfg, "target_gross_exposure", 0.72) * regime_multiplier
        if getattr(self.cfg, "enable_volatility_targeting", True) and portfolio_volatility > 1e-12:
            vol_scale = clamp(getattr(self.cfg, "target_portfolio_volatility", 0.02) / portfolio_volatility, getattr(self.cfg, "min_vol_target_scale", 0.35), getattr(self.cfg, "max_vol_target_scale", 1.2))
            target_exposure = min(target_exposure, getattr(self.cfg, "target_gross_exposure", 0.72) * vol_scale)
        if market_regime == "risk_on":
            target_exposure = max(target_exposure, getattr(self.cfg, "target_gross_exposure", 0.72) * getattr(self.cfg, "risk_on_exposure_floor", 0.80))
        elif market_regime == "neutral":
            target_exposure = max(target_exposure, getattr(self.cfg, "target_gross_exposure", 0.72) * getattr(self.cfg, "neutral_exposure_floor", 0.45))
        if diversification_breakdown:
            target_exposure *= getattr(self.cfg, "diversification_breakdown_exposure_multiplier", 0.75)
        if correlation_shock:
            target_exposure *= getattr(self.cfg, "correlation_shock_exposure_multiplier", 0.70)
        target_exposure *= drawdown_scale
        alpha_aggressiveness = {"risk_on": 1.0, "neutral": 0.85, "risk_off": 0.60}.get(market_regime, 0.85)
        if correlation_shock:
            alpha_aggressiveness *= 0.85
        alpha_aggressiveness *= drawdown_scale
        return PortfolioRiskState(
            covariance_matrix=cov_matrix.round(8).to_dict() if not cov_matrix.empty else {},
            portfolio_volatility=portfolio_volatility,
            average_correlation=average_correlation,
            market_regime=market_regime,
            risk_score=risk_score,
            target_exposure=clamp(target_exposure, 0.0, getattr(self.cfg, "target_gross_exposure", 0.72)),
            diversification_breakdown=diversification_breakdown,
            correlation_shock=correlation_shock,
            alpha_aggressiveness=alpha_aggressiveness,
            drawdown_scale=drawdown_scale,
            data_frequency=data_frequency,
            raw_weights=raw_weights,
        )

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


FeatureMap = Dict[str, Dict[str, float]]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class PortfolioConstructor:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.rank_retention_buffer = 0.40
        self.holding_bonus_floor = 0.36
        self.range_keep_exposure = getattr(cfg, "range_keep_exposure", 0.10)
        self.pump_chase_cutoff = 0.035
        self.pullback_entry_floor = -0.020

    def _is_recovery_mode(self, positions: Dict[str, float], direction_regime: str, current_drawdown: float) -> bool:
        if not getattr(self.cfg, "enable_recovery_reentry", True):
            return False
        if positions:
            return False
        if direction_regime not in {"trend", "neutral"}:
            return False
        return current_drawdown >= max(getattr(self.cfg, "max_portfolio_drawdown", 0.10) * 0.50, 0.03)

    def _recovery_reentry_ready(self, feature: Dict[str, float]) -> bool:
        return (
            feature.get("pullback20", 0.0) <= getattr(self.cfg, "recovery_pullback_threshold", -0.025)
            and feature.get("rebound_from_low5", 0.0) >= getattr(self.cfg, "recovery_rebound_confirmation", 0.008)
            and feature.get("ret3", 0.0) > 0.0
            and feature.get("volume_confirmation_ortho", 0.0) >= getattr(self.cfg, "recovery_volume_confirmation_floor", 0.0)
            and feature.get("trend_stability_ortho", 0.0) >= getattr(self.cfg, "recovery_trend_stability_floor", -0.10)
        )

    def _style_score(self, feature: Dict[str, float], regime: str) -> float:
        momentum = 0.40 * feature.get("multi_horizon_momentum_ortho", 0.0) + 0.22 * feature.get("volatility_breakout_ortho", 0.0) + 0.18 * feature.get("trend_stability_ortho", 0.0) + 0.12 * feature.get("volume_confirmation_ortho", 0.0) + 0.08 * feature.get("pred_mu_z", 0.0)
        defensive = 0.30 * feature.get("mean_reversion_ortho", 0.0) + 0.24 * feature.get("trend_stability_ortho", 0.0) + 0.18 * feature.get("volume_confirmation_ortho", 0.0) + 0.16 * (-feature.get("vol60", 0.0)) + 0.12 * feature.get("recovery_after_drawdown20", 0.0)
        if regime == "risk_on":
            return 0.88 * momentum + 0.12 * defensive
        if regime == "risk_off":
            return 0.50 * momentum + 0.50 * defensive
        return 0.72 * momentum + 0.28 * defensive

    def _state_score_adjustment(self, feature: Dict[str, float]) -> float:
        state = str(feature.get("setup_state", "neutral"))
        return {
            "trend_follow": getattr(self.cfg, "trend_state_score_boost", 0.12),
            "breakout_acceleration": getattr(self.cfg, "breakout_state_score_boost", 0.08),
            "shock_rebound": getattr(self.cfg, "shock_rebound_probe_boost", 0.08),
            "rebound_confirmed": getattr(self.cfg, "rebound_state_score_boost", 0.03),
            "failed_rebound": -getattr(self.cfg, "failed_rebound_score_penalty", 0.20),
            "range_chop": -getattr(self.cfg, "range_chop_score_penalty", 0.12),
        }.get(state, 0.0)

    def _bucket_score_adjustment(self, feature: Dict[str, float], direction_regime: str) -> float:
        bucket = str(feature.get("asset_bucket", "satellite"))
        if bucket == "core":
            return 0.08
        if bucket == "liquid":
            return 0.03 if direction_regime in {"trend", "neutral"} else -0.01
        return -0.08 if direction_regime != "trend" else -0.03

    def _enforce_bucket_caps(self, weights: Dict[str, float], features: FeatureMap) -> Dict[str, float]:
        if not weights:
            return {}
        cap_map = {
            "core": getattr(self.cfg, "core_bucket_weight_cap", 0.55),
            "liquid": getattr(self.cfg, "liquid_bucket_weight_cap", 0.35),
            "satellite": getattr(self.cfg, "satellite_bucket_weight_cap", 0.20),
        }
        adjusted = dict(weights)
        for _ in range(len(adjusted) + 2):
            bucket_totals: Dict[str, float] = {}
            for pair, weight in adjusted.items():
                bucket = str(features.get(pair, {}).get("asset_bucket", "satellite"))
                bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + weight
            violating = {bucket: total for bucket, total in bucket_totals.items() if total > cap_map.get(bucket, 1.0) + 1e-12}
            if not violating:
                break
            spill = 0.0
            for bucket, total in violating.items():
                cap = cap_map.get(bucket, 1.0)
                scale = cap / max(total, 1e-12)
                for pair in list(adjusted):
                    if str(features.get(pair, {}).get("asset_bucket", "satellite")) == bucket:
                        old = adjusted[pair]
                        adjusted[pair] *= scale
                        spill += old - adjusted[pair]
            receivers = [pair for pair, weight in adjusted.items() if cap_map.get(str(features.get(pair, {}).get("asset_bucket", "satellite")), 1.0) - bucket_totals.get(str(features.get(pair, {}).get("asset_bucket", "satellite")), 0.0) > 1e-12 and weight > 0]
            receiver_total = sum(adjusted[pair] for pair in receivers)
            if spill <= 1e-12 or receiver_total <= 1e-12:
                break
            for pair in receivers:
                adjusted[pair] += spill * (adjusted[pair] / receiver_total)
        total = sum(adjusted.values())
        return {pair: weight / total for pair, weight in adjusted.items()} if total > 1e-12 else weights

    def _select_ranked_with_retention(self, ranked: List[Tuple[str, Dict[str, float], float, float]], held_pairs: set[str]) -> List[Tuple[str, Dict[str, float], float, float]]:
        if not ranked:
            return []
        ranked.sort(key=lambda item: (item[2], item[1].get("pred_mu", 0.0), item[1].get("ret15", 0.0)), reverse=True)
        target_slots = max(getattr(self.cfg, "top_n", 5), getattr(self.cfg, "portfolio_min_positions", 4))
        target_slots = min(target_slots, max(getattr(self.cfg, "portfolio_max_positions", 6), 1))
        if len(ranked) <= target_slots:
            return ranked
        selected = ranked[:target_slots]
        selected_pairs = {item[0] for item in selected}
        cutoff_score = selected[-1][2]
        for item in ranked[target_slots:]:
            pair, _, ranking_score, _ = item
            if pair in held_pairs and pair not in selected_pairs and ranking_score >= cutoff_score - self.rank_retention_buffer:
                selected.append(item)
                selected_pairs.add(pair)
        return selected

    def alpha_proxy_weights(self, features: FeatureMap, positions: Dict[str, float], direction_regime: str, current_drawdown: float = 0.0) -> Dict[str, float]:
        held_pairs = set(positions)
        ranked: List[Tuple[str, Dict[str, float], float, float]] = []
        recovery_mode = self._is_recovery_mode(positions, direction_regime, current_drawdown)
        regime_entry_multiplier = {
            "trend": 0.90,
            "neutral": 0.97,
            "range": 0.98,
            "panic": 9.99,
        }.get(direction_regime, 1.0)
        for pair, feature in features.items():
            is_held = pair in held_pairs
            threshold = getattr(self.cfg, "exit_score_threshold", 0.12) if is_held else getattr(self.cfg, "entry_score_threshold", 0.68) * regime_entry_multiplier
            if recovery_mode and not is_held:
                threshold = max(threshold - getattr(self.cfg, "recovery_entry_score_relaxation", 0.10), getattr(self.cfg, "exit_score_threshold", 0.12) + 0.10)
            style_score = self._style_score(feature, "neutral")
            combined_score = 0.70 * feature.get("score", 0.0) + 0.30 * style_score
            combined_score += self._state_score_adjustment(feature) + self._bucket_score_adjustment(feature, direction_regime)
            min_confidence = getattr(self.cfg, "hold_confidence_threshold", 0.30) if is_held else getattr(self.cfg, "entry_confidence_threshold", 0.55)
            if recovery_mode and not is_held:
                min_confidence = max(min_confidence - getattr(self.cfg, "recovery_entry_confidence_relaxation", 0.06), getattr(self.cfg, "hold_confidence_threshold", 0.30))
            if str(feature.get("asset_bucket", "satellite")) == "satellite" and direction_regime != "trend":
                min_confidence = max(min_confidence, 0.62)
            if feature.get("confidence", 0.0) < min_confidence:
                continue
            if combined_score < threshold:
                continue
            if not is_held:
                if feature.get("score", 0.0) <= 0.0:
                    continue
                if feature.get("dist_ma20", 0.0) > self.pump_chase_cutoff:
                    continue
                if feature.get("pullback20", 0.0) < self.pullback_entry_floor:
                    continue
                if recovery_mode and not self._recovery_reentry_ready(feature):
                    continue
                if direction_regime == "range":
                    if not getattr(self.cfg, "enable_range_entries", False):
                        continue
                    if feature.get("mean_reversion_ortho", 0.0) < getattr(self.cfg, "range_mean_reversion_floor", 0.00):
                        continue
                    if feature.get("trend_stability_ortho", 0.0) < getattr(self.cfg, "range_trend_stability_floor", -0.05):
                        continue
                    if feature.get("confidence", 0.0) < max(min_confidence, getattr(self.cfg, "range_entry_confidence_floor", 0.62)):
                        continue
            if direction_regime == "panic" and not is_held:
                continue
            ranking_score = combined_score + (max(getattr(self.cfg, "holding_score_bonus", 0.12), self.holding_bonus_floor) if is_held else 0.0)
            ranked.append((pair, feature, ranking_score, threshold))
        ranked = self._select_ranked_with_retention(ranked, held_pairs)
        strengths = {}
        for pair, _, ranking_score, threshold in ranked:
            strengths[pair] = max(ranking_score - threshold + 0.15, 0.02) * clamp(features[pair].get("confidence", 0.5), 0.2, 1.0)
        total = sum(strengths.values())
        if total <= 1e-12:
            return {}
        gross = getattr(self.cfg, "target_gross_exposure", 0.72)
        return {pair: value / total * gross for pair, value in strengths.items()}

    def _risk_parity(self, candidates: List[str], cov_matrix: pd.DataFrame, seed_strengths: Dict[str, float]) -> Dict[str, float]:
        if not candidates:
            return {}
        if cov_matrix.empty:
            total = sum(seed_strengths.values())
            return {pair: seed_strengths[pair] / total for pair in candidates} if total > 0 else {}
        sigma = cov_matrix.loc[candidates, candidates].to_numpy(dtype=float)
        weights = np.array([max(seed_strengths.get(pair, 1.0), 1e-6) for pair in candidates], dtype=float)
        weights /= weights.sum()
        for _ in range(80):
            marginal = sigma @ weights
            risk_contrib = weights * marginal
            total_var = float(weights.T @ marginal)
            if total_var <= 1e-12:
                break
            target = total_var / len(candidates)
            adjust = np.sqrt(target / np.maximum(risk_contrib, 1e-12))
            weights *= adjust
            weights = np.maximum(weights, 1e-8)
            weights /= weights.sum()
        return {pair: float(weights[idx]) for idx, pair in enumerate(candidates)}

    def _apply_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        if not weights:
            return {}
        max_single = getattr(self.cfg, "max_single_weight", 0.28)
        min_single = getattr(self.cfg, "min_effective_weight", 0.04)
        adjusted = dict(weights)
        for _ in range(len(adjusted) + 2):
            over = {pair: weight for pair, weight in adjusted.items() if weight > max_single}
            if not over:
                break
            excess = sum(weight - max_single for weight in over.values())
            under_pairs = [pair for pair, weight in adjusted.items() if weight < max_single]
            under_total = sum(adjusted[pair] for pair in under_pairs)
            for pair in over:
                adjusted[pair] = max_single
            if excess <= 1e-12 or under_total <= 1e-12:
                break
            for pair in under_pairs:
                adjusted[pair] += excess * (adjusted[pair] / under_total)
        total = sum(adjusted.values())
        normalized = {pair: weight / total for pair, weight in adjusted.items() if total > 1e-12 and weight > 0}
        filtered = {pair: weight for pair, weight in normalized.items() if weight >= min_single}
        filtered_total = sum(filtered.values())
        return {pair: weight / filtered_total for pair, weight in filtered.items()} if filtered_total > 1e-12 else normalized

    def construct(self, features: FeatureMap, positions: Dict[str, float], risk_state: PortfolioRiskState, direction_regime: str, current_drawdown: float = 0.0) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
        pre_risk_weights = self.alpha_proxy_weights(features, positions, direction_regime, current_drawdown=current_drawdown)
        if not pre_risk_weights:
            return {}, {}, {}
        held_pairs = set(positions)
        recovery_mode = self._is_recovery_mode(positions, direction_regime, current_drawdown)
        entry_modes: Dict[str, str] = {}
        candidates = list(pre_risk_weights.keys())
        if risk_state.market_regime == "risk_off":
            candidates = [pair for pair in candidates if pair in held_pairs]
        if direction_regime == "panic":
            return pre_risk_weights, {}, {}
        if not candidates:
            return pre_risk_weights, {}, {}
        if direction_regime == "range":
            if not getattr(self.cfg, "enable_range_entries", False):
                candidates = [pair for pair in candidates if pair in held_pairs]
                if not candidates:
                    return pre_risk_weights, {}, {}
            defensive_candidates = []
            for pair in candidates:
                feature = features[pair]
                defensive_score = (
                    0.45 * feature.get("mean_reversion_ortho", 0.0)
                    + 0.25 * feature.get("trend_stability_ortho", 0.0)
                    + 0.20 * feature.get("volume_confirmation_ortho", 0.0)
                    + 0.10 * feature.get("confidence", 0.0)
                )
                if defensive_score >= getattr(self.cfg, "range_defensive_score_threshold", 0.08):
                    defensive_candidates.append((pair, defensive_score))
            defensive_candidates.sort(key=lambda item: item[1], reverse=True)
            max_positions = max(1, min(getattr(self.cfg, "range_max_positions", 2), getattr(self.cfg, "top_n", 5)))
            candidates = [pair for pair, _ in defensive_candidates[:max_positions]]
            if not candidates:
                return pre_risk_weights, {}, {}
        elif recovery_mode:
            recovery_ranked = sorted(
                candidates,
                key=lambda pair: (
                    0.65 * features[pair].get("score", 0.0)
                    + 0.20 * self._style_score(features[pair], risk_state.market_regime)
                    + 0.15 * features[pair].get("confidence", 0.0)
                ),
                reverse=True,
            )
            candidates = recovery_ranked[: max(1, min(getattr(self.cfg, "recovery_max_positions", 2), getattr(self.cfg, "top_n", 5)))]
            if not candidates:
                return pre_risk_weights, {}, {}
            entry_modes = {pair: "recovery_reentry" for pair in candidates}
        cov_matrix = pd.DataFrame(risk_state.covariance_matrix)
        seed_strengths = {}
        for pair in candidates:
            feature = features[pair]
            style_score = self._style_score(feature, risk_state.market_regime)
            state_multiplier = 1.0 + self._state_score_adjustment(feature)
            bucket_multiplier = 1.04 if str(feature.get("asset_bucket", "satellite")) == "core" else 0.97 if str(feature.get("asset_bucket", "satellite")) == "satellite" else 1.0
            seed_strengths[pair] = max(0.02, (0.65 * max(feature.get("score", 0.0), -0.5) + 0.35 * style_score + 0.40) * clamp(feature.get("confidence", 0.5), 0.2, 1.0) * max(state_multiplier, 0.5) * bucket_multiplier)
        weights = self._risk_parity(candidates, cov_matrix, seed_strengths)
        if not weights:
            return pre_risk_weights, {}, {}
        if not cov_matrix.empty:
            corr = cov_matrix.loc[candidates, candidates].copy()
            diag = np.sqrt(np.maximum(np.diag(corr.to_numpy(dtype=float)), 0.0))
            denom = np.outer(diag, diag)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_values = np.divide(corr.to_numpy(dtype=float), denom, where=denom > 0)
            corr = pd.DataFrame(corr_values, index=candidates, columns=candidates)
            adjusted = {}
            core_assets = {item.strip().upper() for item in getattr(self.cfg, "diversification_core_assets", "BTC,ETH").split(",") if item.strip()}
            for pair, weight in weights.items():
                others = corr.loc[pair, corr.columns != pair]
                asset_corr = float(others[np.isfinite(others)].mean()) if len(others) else 0.0
                corr_penalty = max(asset_corr - getattr(self.cfg, "risk_on_correlation_threshold", 0.35), 0.0)
                vol_penalty = max(features[pair].get("vol60", 0.0) / max(getattr(self.cfg, "vol_cap", 0.08), 1e-9) - 0.5, 0.0)
                multiplier = 1.0 / (1.0 + 0.75 * corr_penalty + 0.15 * vol_penalty)
                if risk_state.diversification_breakdown:
                    base_symbol = pair.split("/")[0].upper()
                    multiplier *= getattr(self.cfg, "diversification_core_asset_multiplier", 1.15) if base_symbol in core_assets else getattr(self.cfg, "diversification_alt_weight_multiplier", 0.85)
                adjusted[pair] = weight * multiplier
            total_adjusted = sum(adjusted.values())
            weights = {pair: value / total_adjusted for pair, value in adjusted.items()} if total_adjusted > 1e-12 else {}
        weights = self._enforce_bucket_caps(weights, features)
        weights = self._apply_caps(weights)
        exposure = risk_state.target_exposure
        if direction_regime == "range":
            exposure = min(exposure, max(self.range_keep_exposure, getattr(self.cfg, "range_entry_exposure", 0.12)))
        elif recovery_mode:
            exposure = min(exposure, getattr(self.cfg, "recovery_reentry_exposure", 0.22))
        final_weights = {pair: weight * exposure for pair, weight in weights.items()}
        for pair in final_weights:
            entry_modes.setdefault(pair, "target_rebalance")
        return pre_risk_weights, final_weights, entry_modes


class MomentumStrategy:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.alpha_model = AlphaModel(cfg)
        self.risk_model = RiskModel(cfg)
        self.portfolio_constructor = PortfolioConstructor(cfg)

    def compute_features(self, history: Dict[str, Deque[Dict[str, float]]], trade_pairs: Dict[str, Dict[str, Any]]) -> FeatureMap:
        return self.alpha_model.compute_features(history, trade_pairs)

    def market_snapshot(self, features: FeatureMap) -> MarketSnapshot:
        return self.alpha_model.market_snapshot(features)

    def select_daily_activity_probe(self, features: FeatureMap, positions: Dict[str, float], direction_regime: str) -> Optional[str]:
        if direction_regime == "panic":
            return None
        candidates = []
        held_pairs = set(positions)
        for pair, feature in features.items():
            if pair in held_pairs:
                continue
            if str(feature.get("asset_bucket", "satellite")) == "satellite":
                continue
            if feature.get("confidence", 0.0) < getattr(self.cfg, "daily_activity_min_confidence", 0.60):
                continue
            if str(feature.get("setup_state", "neutral")) in {"failed_rebound", "range_chop"}:
                continue
            if feature.get("score", 0.0) <= 0.0:
                continue
            probe_score = (
                0.50 * feature.get("score", 0.0)
                + 0.20 * feature.get("confidence", 0.0)
                + 0.20 * self.portfolio_constructor._style_score(feature, "neutral")
                + 0.10 * feature.get("relative_strength", 0.0)
            )
            if str(feature.get("setup_state", "neutral")) == "shock_rebound":
                probe_score += getattr(self.cfg, "shock_rebound_probe_boost", 0.08)
            candidates.append((pair, probe_score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0] if candidates else None

    def generate_signals(
        self,
        history: Dict[str, Deque[Dict[str, float]]],
        trade_pairs: Dict[str, Dict[str, Any]],
        positions: Dict[str, float],
        prev_risk_on: bool,
        current_drawdown: float = 0.0,
    ) -> Dict[str, Any]:
        symbols = sorted([pair for pair in trade_pairs if pair in history and len(history[pair]) > 0])
        directional_regime = self.alpha_model.update_directional_regime(history, symbols)
        features = self.compute_features(history, trade_pairs)
        snapshot = self.market_snapshot(features)

        alpha_proxy = self.portfolio_constructor.alpha_proxy_weights(
            features=features,
            positions=positions,
            direction_regime=str(directional_regime["regime"]),
            current_drawdown=current_drawdown,
        )
        portfolio_risk = self.risk_model.evaluate(
            history=history,
            trade_pairs=trade_pairs,
            features=features,
            raw_weights=alpha_proxy,
            positions=positions,
            current_drawdown=current_drawdown,
        )
        pre_risk_weights, final_weights, entry_modes = self.portfolio_constructor.construct(
            features=features,
            positions=positions,
            risk_state=portfolio_risk,
            direction_regime=str(directional_regime["regime"]),
            current_drawdown=current_drawdown,
        )

        allow_directional_entries = directional_regime["regime"] in ["trend", "neutral"]
        positive_alpha_breadth = snapshot.positive_score_ratio >= getattr(self.cfg, "market_positive_score_ratio_threshold", 0.45) * 0.85
        risk_on = allow_directional_entries and positive_alpha_breadth and portfolio_risk.market_regime != "risk_off"
        if directional_regime["regime"] == "panic":
            final_weights = {}
            risk_on = False

        portfolio_risk.adjusted_weights = final_weights
        daily_activity_probe = self.select_daily_activity_probe(
            features=features,
            positions=positions,
            direction_regime=str(directional_regime["regime"]),
        )
        return {
            "features": features,
            "snapshot": snapshot,
            "risk_on": risk_on,
            "weights": final_weights,
            "pre_risk_weights": pre_risk_weights,
            "entry_modes": entry_modes,
            "daily_activity_probe": daily_activity_probe,
            "portfolio_risk": {
                "covariance_matrix": portfolio_risk.covariance_matrix,
                "portfolio_volatility": portfolio_risk.portfolio_volatility,
                "average_correlation": portfolio_risk.average_correlation,
                "market_regime": portfolio_risk.market_regime,
                "risk_score": portfolio_risk.risk_score,
                "target_exposure": portfolio_risk.target_exposure,
                "diversification_breakdown": portfolio_risk.diversification_breakdown,
                "correlation_shock": portfolio_risk.correlation_shock,
                "alpha_aggressiveness": portfolio_risk.alpha_aggressiveness,
                "drawdown_scale": portfolio_risk.drawdown_scale,
                "data_frequency": portfolio_risk.data_frequency,
                "raw_weights": portfolio_risk.raw_weights,
                "adjusted_weights": final_weights,
            },
            "regime": {
                **directional_regime,
                "mu_ready": self.alpha_model.mu_model.ready,
                "mu_blend_weight": self.alpha_model.mu_weight,
                "fixed_blend_weight": self.alpha_model.fixed_weight,
                "mu_error": self.alpha_model.mu_model.error,
            },
        }
