from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

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

# =========================
# REGIME FILTER（内嵌版）
# =========================
from collections import deque
import numpy as np
import pandas as pd

class RegimeFilter:
    def __init__(self, symbols, btc_symbol="BTC/USD", maxlen=200):
        self.symbols = symbols
        self.btc_symbol = btc_symbol
        self.price_history = {s: deque(maxlen=maxlen) for s in symbols}
        self.current_regime = None

    def update_market_data(self, tickers):
        for symbol in self.symbols:
            if symbol in tickers:
                price = tickers[symbol].get("LastPrice", 0)
                if price > 0:
                    self.price_history[symbol].append(float(price))

    def _to_df(self):
        df = pd.DataFrame({k: list(v) for k, v in self.price_history.items()})
        return df.ffill().dropna(how="all")

    def detect_regime(self):
        price_df = self._to_df()

        if self.btc_symbol not in price_df or len(price_df[self.btc_symbol]) < 10:
            self.current_regime = {
                "regime": "neutral",
                "risk_multiplier": 0.5,
            }
            return self.current_regime

        returns_df = price_df.pct_change()

        # ===== BTC TREND =====
        btc_price = price_df[self.btc_symbol]
        ma = btc_price.rolling(min(50, len(btc_price))).mean()
        if len(btc_price) < 10 or np.isnan(ma.iloc[-1]):
            self.current_regime = {
                "regime": "neutral",
                "risk_multiplier": 0.5,
            }
            return self.current_regime
        btc_trend = btc_price.iloc[-1] / ma.iloc[-1] - 1

        # ===== VOL =====
        btc_returns = returns_df[self.btc_symbol]
        vol = btc_returns.rolling(min(20, len(btc_returns))).std()
        vol_window = vol.dropna()

        if len(vol_window) < 20:
            vol_pct = 0.5
        else:
            current = vol.iloc[-1]
            vol_pct = (vol_window < current).mean()

        # ===== BREADTH =====
        latest_returns = returns_df.iloc[-1].dropna()
        if len(latest_returns) == 0:
            return self.current_regime or {"regime": "neutral", "risk_multiplier": 0.5}
        breadth = (latest_returns > 0).mean()

        # ===== SCORE =====
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

        if latest_returns.mean() > 0:
            score += 1

        if vol_pct > 0.95:
            score -= 2
        elif vol_pct > 0.8:
            score -= 1

        prev = self.current_regime["regime"] if self.current_regime else "neutral"

        prev = self.current_regime["regime"] if self.current_regime else "neutral"

        # ===== PANIC 优先级最高 =====
        if vol_pct > 0.97:
            regime = "panic"

        # ===== TREND（带滞后）=====
        elif prev == "trend":
            if score >= 1:  # 👈 从2降到1（更容易维持）
                regime = "trend"
            else:
                regime = "neutral"

        # ===== RANGE（带滞后）=====
        elif prev == "range":
            if score <= 0:  # 👈 从-1放宽到0
                regime = "range"
            else:
                regime = "neutral"

        # ===== 新进入状态（严格）=====
        else:
            if score >= 2:
                regime = "trend"
            elif score <= -1:
                regime = "range"
            else:
                regime = "neutral"

        mapping = {
            "trend": 1.0,
            "neutral": 0.6,
            "range": 0.5,
            "panic": 0.2,
        }

        self.current_regime = {
            "regime": regime,
            "risk_multiplier": mapping[regime],
        }

        return self.current_regime

class MomentumStrategy:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.regime_filter = None

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

    def compute_features(
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
            if len(prices) < 2:
                continue
            price = prices[-1]
            returns_1m = []
            max_lookback = min(60, len(prices) - 1)

            for offset in range(1, max_lookback + 1):
                current = prices[-offset]
                previous = prices[-offset - 1]
                if previous > 0:
                    returns_1m.append(current / previous - 1.0)

            if len(prices) < 20:
                continue
            ma20 = mean(prices[-20:])
            high20 = max(prices[-20:])
            low20 = min(prices[-20:])
            vol60 = stddev(returns_1m)
            bid = series[-1]["bid"]
            ask = series[-1]["ask"]
            spread = 0.0
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                if mid > 0:
                    spread = (ask - bid) / mid
            features[pair] = {
                "price": price,
                "ret5": compute_return(prices, 5),
                "ret15": compute_return(prices, 15),
                "ret30": compute_return(prices, 30),
                "ret60": compute_return(prices, 60),
                "dist_ma20": price / ma20 - 1.0 if ma20 > 0 else 0.0,
                "vol60": vol60,
                "trend_ratio60": compute_return(prices, 60) / max(vol60 * math.sqrt(max(len(returns_1m), 1)), 1e-9),
                "efficiency20": self.trend_efficiency(prices, 20),
                "range_position20": 0.5 if high20 <= low20 else clamp((price - low20) / (high20 - low20), 0.0, 1.0),
                "pullback20": price / high20 - 1.0 if high20 > 0 else 0.0,
                "spread": spread,
                "change_24h": series[-1]["change_24h"],
                "unit_trade_value": series[-1]["unit_trade_value"],
            }
            if spread <= self.cfg.spread_threshold and series[-1]["unit_trade_value"] >= self.cfg.min_24h_dollar_vol:
                eligible_pairs.append(pair)
        if not eligible_pairs:
            return {}

        ret5_values = [features[pair]["ret5"] for pair in eligible_pairs]
        ret15_values = [features[pair]["ret15"] for pair in eligible_pairs]
        ret30_values = [features[pair]["ret30"] for pair in eligible_pairs]
        ret60_values = [features[pair]["ret60"] for pair in eligible_pairs]
        dist_values = [features[pair]["dist_ma20"] for pair in eligible_pairs]
        vol_values = [features[pair]["vol60"] for pair in eligible_pairs]
        trend_values = [features[pair]["trend_ratio60"] for pair in eligible_pairs]
        efficiency_values = [features[pair]["efficiency20"] for pair in eligible_pairs]
        range_values = [features[pair]["range_position20"] for pair in eligible_pairs]

        for pair in eligible_pairs:
            feature = features[pair]
            score = (
                0.18 * zscore(feature["ret5"], ret5_values)
                + 0.22 * zscore(feature["ret15"], ret15_values)
                + 0.20 * zscore(feature["ret30"], ret30_values)
                + 0.18 * zscore(feature["ret60"], ret60_values)
                + 0.10 * zscore(feature["trend_ratio60"], trend_values)
                + 0.08 * zscore(feature["efficiency20"], efficiency_values)
                + 0.06 * zscore(feature["range_position20"], range_values)
                + 0.10 * zscore(-feature["dist_ma20"], [-value for value in dist_values])
                + 0.12 * (-zscore(feature["vol60"], vol_values))
            )
            if feature["dist_ma20"] > self.cfg.max_pump_distance:
                score -= 0.20 + 2.5 * (feature["dist_ma20"] - self.cfg.max_pump_distance)
            if feature["spread"] > self.cfg.spread_threshold * 0.7:
                score -= 0.15 * (feature["spread"] / max(self.cfg.spread_threshold, 1e-9))
            if feature["ret5"] < 0 and feature["pullback20"] < -0.03:
                score -= 0.12
            if feature["ret60"] < 0 and feature["ret5"] > 0.02:
                score -= 0.10
            feature["score"] = score
        return {pair: features[pair] for pair in eligible_pairs}

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
            threshold = self.cfg.exit_score_threshold if pair in held_pairs else self.cfg.entry_score_threshold
            if feature["score"] < threshold:
                continue
            ranking_score = feature["score"] + (self.cfg.holding_score_bonus if pair in held_pairs else 0.0)
            ranked.append((pair, feature, ranking_score, threshold))
        ranked.sort(key=lambda item: (item[2], item[1]["trend_ratio60"], item[1]["ret15"]), reverse=True)
        ranked = ranked[: self.cfg.top_n]
        if not ranked:
            return {}
        strengths: List[Tuple[str, float]] = []
        for pair, feature, ranking_score, threshold in ranked:
            vol = clamp(feature["vol60"], self.cfg.vol_floor, self.cfg.vol_cap)
            liquidity_multiplier = clamp(math.log1p(feature["unit_trade_value"] / max(self.cfg.min_24h_dollar_vol, 1.0)), 0.75, 1.35)
            quality_multiplier = clamp(
                1.0 + max(feature["trend_ratio60"], 0.0) * 0.08 + feature["efficiency20"] * 0.20,
                0.80,
                1.50,
            )
            edge = max(ranking_score - threshold + 0.20, 0.05)
            strengths.append((pair, (edge * liquidity_multiplier * quality_multiplier) / vol))
        return self.capped_inverse_vol_weights(strengths)

    def generate_signals(
        self,
        history: Dict[str, Deque[Dict[str, float]]],
        trade_pairs: Dict[str, Dict[str, Any]],
        positions: Dict[str, float],
        prev_risk_on:
        bool,
    ) -> Dict[str, Any]:
        # ===== 初始化 Regime =====
        if self.regime_filter is None:
            symbols = [pair for pair in history if len(history[pair]) > 0]
            self.regime_filter = RegimeFilter(symbols)

        # 用当前价格更新
        latest_tickers = {
            pair: {"LastPrice": history[pair][-1]["price"]}
            for pair in history
            if len(history[pair]) > 0
        }

        self.regime_filter.update_market_data(latest_tickers)
        regime = self.regime_filter.detect_regime()

        features = self.compute_features(history, trade_pairs)
        snapshot = self.market_snapshot(features)
        risk_on = regime["regime"] in ["trend", "neutral"]
        weights = self.target_weights(features, risk_on, positions)

        # range 不清仓
        regime_name = regime["regime"]

        if regime_name == "range" and positions:
            # 保留已有仓位，按小仓位分配
            n = len(positions)
            if n > 0:
                weights = {
                    p: min(self.cfg.max_single_weight, 0.3 / n)
                    for p in positions
                }
        # ===== Regime 控制仓位（核心）=====
        if weights:
            if regime_name == "range":
                multiplier = 1.0
            elif regime_name == "neutral":
                multiplier = 0.6
            elif regime_name == "trend":
                multiplier = 1.0
            else:  # panic
                multiplier = 0.0

            total = sum(weights.values())
            if total > 0:
                weights = {p: w / total * self.cfg.target_gross_exposure * multiplier for p, w in weights.items()}

        return {
            "features": features,
            "snapshot": snapshot,
            "risk_on": risk_on,
            "weights": weights,
            "regime": regime,
        }