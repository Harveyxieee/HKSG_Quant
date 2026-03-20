from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    risk_score: float = 0.5
    target_exposure: float = 0.0
    diversification_breakdown: bool = False
    data_frequency: str = "raw"
    raw_weights: Dict[str, float] = field(default_factory=dict)
    adjusted_weights: Dict[str, float] = field(default_factory=dict)


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


def compute_cov_matrix(returns_df: pd.DataFrame, window: int) -> pd.DataFrame:
    if returns_df.empty:
        return pd.DataFrame()
    sample = returns_df.tail(window).dropna(how="all")
    sample = sample.loc[:, sample.notna().sum() >= 2]
    sample = sample.dropna(axis=0, how="any")
    if len(sample) < 2 or sample.empty:
        return pd.DataFrame()
    return sample.cov()


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
) -> float:
    if returns_df is not None and not returns_df.empty:
        sample = returns_df.tail(window).dropna(how="all")
        sample = sample.loc[:, sample.notna().sum() >= 2]
        sample = sample.dropna(axis=0, how="any")
        if sample.shape[1] < 2 or len(sample) < 2:
            return 0.0
        corr_matrix = sample.corr()
    elif cov_matrix is not None and not cov_matrix.empty:
        diag = np.sqrt(np.maximum(np.diag(cov_matrix.to_numpy(dtype=float)), 0.0))
        denom = np.outer(diag, diag)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_values = np.divide(cov_matrix.to_numpy(dtype=float), denom, where=denom > 0)
        corr_matrix = pd.DataFrame(corr_values, index=cov_matrix.index, columns=cov_matrix.columns)
    else:
        return 0.0

    if corr_matrix.shape[0] < 2:
        return 0.0
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    values = corr_matrix.to_numpy(dtype=float)[mask]
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else 0.0


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
        corr_sample = corr_sample.dropna(axis=0, how="any")
        if corr_sample.shape[1] >= 2 and len(corr_sample) >= 2:
            corr_matrix = corr_sample.corr()
            for pair in corr_matrix.columns:
                others = corr_matrix.loc[pair, corr_matrix.columns != pair]
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

        btc_price = price_df[self.btc_symbol]
        ma = btc_price.rolling(min(50, len(btc_price))).mean()
        if len(btc_price) < 10 or np.isnan(ma.iloc[-1]):
            self.current_regime = {
                "regime": "neutral",
                "risk_multiplier": 0.5,
            }
            return self.current_regime
        btc_trend = btc_price.iloc[-1] / ma.iloc[-1] - 1

        btc_returns = returns_df[self.btc_symbol]
        vol = btc_returns.rolling(min(20, len(btc_returns))).std()
        vol_window = vol.dropna()
        if len(vol_window) < 20:
            vol_pct = 0.5
        else:
            current = vol.iloc[-1]
            vol_pct = (vol_window < current).mean()

        latest_returns = returns_df.iloc[-1].dropna()
        if len(latest_returns) == 0:
            return self.current_regime or {"regime": "neutral", "risk_multiplier": 0.5}
        breadth = (latest_returns > 0).mean()

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

    def _position_weight_proxy(
        self,
        positions: Dict[str, float],
        features: FeatureMap,
        history: Dict[str, Deque[Dict[str, float]]],
    ) -> Dict[str, float]:
        notional_map: Dict[str, float] = {}
        for pair, quantity in positions.items():
            price = features.get(pair, {}).get("price")
            if price is None and history.get(pair):
                price = float(history[pair][-1].get("price", 0.0))
            if price and price > 0:
                notional_map[pair] = quantity * price
        total = sum(notional_map.values())
        if total > 1e-12:
            return {pair: value / total * self.cfg.target_gross_exposure for pair, value in notional_map.items()}
        return {}

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
        cov_matrix = compute_cov_matrix(returns_df, window=self.cfg.risk_cov_window)

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
        if self.regime_filter is None:
            symbols = [pair for pair in history if len(history[pair]) > 0]
            self.regime_filter = RegimeFilter(symbols)

        latest_tickers = {
            pair: {"LastPrice": history[pair][-1]["price"]}
            for pair in history
            if len(history[pair]) > 0
        }
        self.regime_filter.update_market_data(latest_tickers)
        regime = self.regime_filter.detect_regime()

        features = self.compute_features(history, trade_pairs)
        snapshot = self.market_snapshot(features)
        base_risk_on = regime["regime"] in ["trend", "neutral"]
        raw_weights = self.target_weights(features, base_risk_on, positions)

        regime_name = regime["regime"]
        if regime_name == "range" and positions:
            n = len(positions)
            if n > 0:
                raw_weights = {
                    pair: min(self.cfg.max_single_weight, 0.3 / n)
                    for pair in positions
                }

        if raw_weights:
            if regime_name == "range":
                multiplier = 1.0
            elif regime_name == "neutral":
                multiplier = 0.6
            elif regime_name == "trend":
                multiplier = 1.0
            else:
                multiplier = 0.0
            total = sum(raw_weights.values())
            if total > 0:
                raw_weights = {
                    pair: weight / total * self.cfg.target_gross_exposure * multiplier
                    for pair, weight in raw_weights.items()
                }

        portfolio_risk = self.evaluate_portfolio_risk(
            history=history,
            trade_pairs=trade_pairs,
            features=features,
            raw_weights=raw_weights,
            positions=positions,
        )

        risk_on = base_risk_on and portfolio_risk.market_regime != "risk_off"
        final_weights = portfolio_risk.adjusted_weights or raw_weights
        if final_weights:
            total = sum(final_weights.values())
            if total > 0:
                final_weights = {
                    pair: weight / total * portfolio_risk.target_exposure
                    for pair, weight in final_weights.items()
                }

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
            "regime": regime,
        }
