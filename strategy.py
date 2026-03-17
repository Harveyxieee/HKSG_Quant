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


class MomentumStrategy:
    def __init__(self, cfg: Any):
        self.cfg = cfg

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
            price = prices[-1]
            returns_1m = []
            for offset in range(1, 61):
                current = prices[-offset]
                previous = prices[-offset - 1]
                if previous > 0:
                    returns_1m.append(current / previous - 1.0)
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
        prev_risk_on: bool,
    ) -> Dict[str, Any]:
        features = self.compute_features(history, trade_pairs)
        snapshot = self.market_snapshot(features)
        risk_on = self.risk_on(snapshot, prev_risk_on)
        weights = self.target_weights(features, risk_on, positions)
        return {
            "features": features,
            "snapshot": snapshot,
            "risk_on": risk_on,
            "weights": weights,
        }