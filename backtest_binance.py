from __future__ import annotations

import argparse
import io
import json
import math
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from bot import Config
from strategy import MomentumStrategy

BINANCE_DATA_BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
ROOSTOO_BASE_URL = "https://mock-api.roostoo.com"
AGG_TRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]
DEFAULT_FALLBACK_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]


@dataclass
class SimPositionMeta:
    pair: str
    quantity: float
    entry_price: float
    highest_price: float
    last_trade_ts: int
    last_signal_score: float
    last_reason: str = ""


@dataclass
class SimBacktestState:
    peak_equity: float = 0.0
    cooldown_until: Dict[str, int] = field(default_factory=dict)
    positions_meta: Dict[str, SimPositionMeta] = field(default_factory=dict)
    risk_on: bool = False
    portfolio_reentry_allowed_at: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Binance spot daily aggTrades data to backtest the current MomentumStrategy.",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbols or Roostoo pairs, e.g. BTCUSDT,ETHUSDT or BTC/USD,ETH/USD",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="How many trailing months of data to use. Default: 3",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Optional exact trailing days to use. If set, it overrides --months",
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Inclusive end date in YYYY-MM-DD. Default: today",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=None,
        help="Initial equity for the backtest. Default: Roostoo exchangeInfo InitialWallet.USD",
    )
    parser.add_argument(
        "--rebalance-minutes",
        type=int,
        default=60,
        help="Rebalance interval in minutes. Default: 60 to cap execution at most once every hour",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Per-side trading fee applied on turnover, e.g. 0.001 = 10bps. Default: 0.001",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/binance_cache",
        help="Directory used to cache downloaded Binance zip files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/backtests/binance",
        help="Directory where backtest reports will be written.",
    )
    parser.add_argument(
        "--roostoo-base-url",
        default=ROOSTOO_BASE_URL,
        help="Roostoo API base URL used to fetch exchangeInfo. Default: https://mock-api.roostoo.com",
    )
    return parser.parse_args()


def pair_from_symbol(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USD"
    raise ValueError(f"Only USDT spot symbols are supported for now: {symbol}")


def symbol_from_pair(pair: str) -> str:
    base, quote = pair.split("/", 1)
    if quote.upper() != "USD":
        raise ValueError(f"Only /USD Roostoo pairs are supported for Binance mapping: {pair}")
    return f"{base.upper()}USDT"


def normalize_timestamp_to_ms(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    if numeric.dropna().median() >= 1e15:
        return numeric / 1000.0
    return numeric


def date_range(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def fetch_daily_zip(symbol: str, day: date, cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{symbol}-aggTrades-{day.isoformat()}.zip"
    local_path = cache_dir / symbol / file_name
    if local_path.exists():
        return local_path.read_bytes()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BINANCE_DATA_BASE_URL}/{symbol}/{file_name}"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urlopen(url, timeout=180) as response:
                payload = response.read()
            local_path.write_bytes(payload)
            return payload
        except (TimeoutError, URLError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to download {url}")


def fetch_json(url: str, timeout: int = 30) -> Dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_roostoo_exchange_info(base_url: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v3/exchangeInfo"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return fetch_json(url, timeout=30)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch exchange info from {url}")


def resolve_roostoo_universe(symbols_arg: str, exchange_info: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    trade_pairs = exchange_info.get("TradePairs", {})
    if not isinstance(trade_pairs, dict):
        raise RuntimeError("Unexpected exchangeInfo.TradePairs format")

    allowed_pairs = {
        pair: rules
        for pair, rules in trade_pairs.items()
        if isinstance(rules, dict) and rules.get("CanTrade", True)
    }
    if not allowed_pairs:
        raise RuntimeError("No tradable pairs found in Roostoo exchangeInfo")

    requested = [item.strip().upper() for item in symbols_arg.split(",") if item.strip()]
    pair_to_symbol: Dict[str, str] = {}
    for item in requested:
        pair = item if "/" in item else pair_from_symbol(item)
        if pair not in allowed_pairs:
            continue
        pair_to_symbol[pair] = symbol_from_pair(pair)

    if not pair_to_symbol:
        for pair in sorted(allowed_pairs):
            try:
                pair_to_symbol[pair] = symbol_from_pair(pair)
            except ValueError:
                continue

    if not pair_to_symbol:
        pair_to_symbol = {
            pair_from_symbol(symbol): symbol
            for symbol in DEFAULT_FALLBACK_SYMBOLS
        }
        filtered_pairs = {pair: allowed_pairs.get(pair, {}) for pair in pair_to_symbol}
        return pair_to_symbol, filtered_pairs

    filtered_pairs = {pair: allowed_pairs[pair] for pair in pair_to_symbol if pair in allowed_pairs}
    return pair_to_symbol, filtered_pairs


def read_agg_trades_from_zip(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_names:
            return pd.DataFrame(columns=AGG_TRADE_COLUMNS)
        with archive.open(csv_names[0]) as handle:
            frame = pd.read_csv(handle, header=None, names=AGG_TRADE_COLUMNS)
    if not frame.empty and str(frame.iloc[0]["agg_trade_id"]).lower() == "agg_trade_id":
        frame = frame.iloc[1:].reset_index(drop=True)
    return frame


def load_symbol_minute_bars(symbol: str, start_date: date, end_date: date, cache_dir: Path) -> pd.DataFrame:
    daily_frames: List[pd.DataFrame] = []
    for day in date_range(start_date, end_date):
        try:
            payload = fetch_daily_zip(symbol, day, cache_dir)
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        except URLError:
            raise
        frame = read_agg_trades_from_zip(payload)
        if frame.empty:
            continue
        daily_frames.append(frame)

    if not daily_frames:
        return pd.DataFrame(columns=["price", "quote_value", "change_24h", "bid", "ask", "unit_trade_value"])

    trades = pd.concat(daily_frames, ignore_index=True)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
    trades["timestamp"] = normalize_timestamp_to_ms(trades["timestamp"])
    trades = trades.dropna(subset=["price", "quantity", "timestamp"])
    if trades.empty:
        return pd.DataFrame(columns=["price", "quote_value", "change_24h", "bid", "ask", "unit_trade_value"])

    trades["quote_value"] = trades["price"] * trades["quantity"]
    trades["ts"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True).dt.floor("min")

    minute = (
        trades.groupby("ts", sort=True)
        .agg(
            price=("price", "last"),
            quote_value=("quote_value", "sum"),
        )
        .sort_index()
    )

    full_index = pd.date_range(
        start=pd.Timestamp(start_date, tz=timezone.utc),
        end=pd.Timestamp(end_date + timedelta(days=1), tz=timezone.utc) - pd.Timedelta(minutes=1),
        freq="1min",
    )
    minute = minute.reindex(full_index)
    minute["price"] = minute["price"].ffill()
    minute["quote_value"] = minute["quote_value"].fillna(0.0)
    minute = minute.dropna(subset=["price"])
    minute["change_24h"] = minute["price"].pct_change(24 * 60).fillna(0.0)
    minute["unit_trade_value"] = minute["quote_value"].rolling(24 * 60, min_periods=1).sum()
    avg_minute_quote = (minute["unit_trade_value"] / (24 * 60)).clip(lower=1.0)
    realized_quote = minute["quote_value"].clip(lower=avg_minute_quote * 0.05)
    liquidity_pressure = (avg_minute_quote / realized_quote).pow(0.5).clip(lower=1.0, upper=6.0)
    spread = (0.0006 * liquidity_pressure).clip(lower=0.0006, upper=0.0040)
    minute["bid"] = minute["price"] * (1.0 - spread / 2.0)
    minute["ask"] = minute["price"] * (1.0 + spread / 2.0)
    return minute[["price", "quote_value", "change_24h", "bid", "ask", "unit_trade_value"]]


def load_market_data(symbols: List[str], start_date: date, end_date: date, cache_dir: Path) -> Dict[str, pd.DataFrame]:
    market_data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        bars = load_symbol_minute_bars(symbol, start_date, end_date, cache_dir)
        if not bars.empty:
            market_data[symbol] = bars
    return market_data


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def round_down(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def build_history_buffers(
    pair_to_symbol: Dict[str, str],
    cfg: Config,
) -> Tuple[Dict[str, Deque[Dict[str, float]]], Dict[str, str]]:
    history = {
        pair: deque(maxlen=cfg.lookback_minutes)
        for pair in pair_to_symbol
    }
    return history, pair_to_symbol


def calculate_metrics(equity_curve: pd.DataFrame, rebalance_returns: List[float]) -> Dict[str, float]:
    if equity_curve.empty:
        return {}

    equity = equity_curve["equity"]
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    minute_returns = equity.pct_change().dropna()
    periods_per_year = 365 * 24 * 60
    if len(minute_returns) >= 2 and minute_returns.std(ddof=1) > 0:
        sharpe = float(math.sqrt(periods_per_year) * minute_returns.mean() / minute_returns.std(ddof=1))
    else:
        sharpe = 0.0

    negative_returns = minute_returns[minute_returns < 0]
    if len(negative_returns) >= 2 and negative_returns.std(ddof=1) > 0:
        sortino = float(math.sqrt(periods_per_year) * minute_returns.mean() / negative_returns.std(ddof=1))
    else:
        sortino = 0.0

    elapsed_minutes = max(len(equity_curve) - 1, 1)
    annualized_return = float((equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / elapsed_minutes) - 1.0)
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    winning_rebalances = sum(1 for value in rebalance_returns if value > 0)
    win_rate = winning_rebalances / len(rebalance_returns) if rebalance_returns else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "rebalance_win_rate": float(win_rate),
    }


def score_total_backtest(
    metrics: Dict[str, float],
    executed_symbols: int,
    requested_symbols: int,
    rebalance_minutes: int,
) -> Dict[str, float]:
    total_return = float(metrics.get("total_return", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    sortino = float(metrics.get("sortino", 0.0))
    calmar = float(metrics.get("calmar", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))

    score_base = 60.0
    score_return = clamp(total_return / 0.20 * 20.0, -20.0, 20.0)
    risk_composite = 0.5 * sharpe + 0.3 * sortino + 0.2 * calmar
    score_risk_adjusted = clamp(risk_composite / 2.0 * 10.0, -10.0, 10.0)

    if max_drawdown >= 0:
        score_drawdown = 5.0
    elif max_drawdown >= -0.05:
        score_drawdown = 5.0
    elif max_drawdown <= -0.25:
        score_drawdown = -10.0
    else:
        score_drawdown = 5.0 - ((abs(max_drawdown) - 0.05) / 0.20) * 15.0
    score_drawdown = clamp(score_drawdown, -10.0, 5.0)

    universe_ratio = executed_symbols / requested_symbols if requested_symbols > 0 else 0.0
    score_validity = clamp(universe_ratio * 5.0, 0.0, 5.0)
    score_compliance = 5.0 if rebalance_minutes >= 60 else clamp(rebalance_minutes / 60.0 * 5.0, 0.0, 5.0)

    total_score = clamp(
        score_base + score_return + score_risk_adjusted + score_drawdown + score_validity + score_compliance,
        0.0,
        100.0,
    )
    return {
        "score_base": score_base,
        "score_return": score_return,
        "score_risk_adjusted": score_risk_adjusted,
        "score_drawdown": score_drawdown,
        "score_validity": score_validity,
        "score_compliance": score_compliance,
        "total_score": total_score,
    }


def score_trade_point(
    equity_value: float,
    initial_equity: float,
    running_peak: float,
    signal_weight: float,
    risk_on: bool,
    market_regime: str,
) -> Dict[str, float]:
    total_return = equity_value / initial_equity - 1.0 if initial_equity > 0 else 0.0
    current_drawdown = equity_value / running_peak - 1.0 if running_peak > 0 else 0.0

    score_base = 60.0
    score_return = clamp(total_return / 0.10 * 12.0, -12.0, 12.0)
    if current_drawdown >= 0:
        score_drawdown = 8.0
    elif current_drawdown <= -0.15:
        score_drawdown = -12.0
    else:
        score_drawdown = 8.0 - (abs(current_drawdown) / 0.15) * 20.0
    score_drawdown = clamp(score_drawdown, -12.0, 8.0)

    score_signal = clamp(signal_weight / 0.30 * 10.0, -10.0, 10.0)
    regime_bonus_map = {
        "risk_on": 5.0,
        "neutral": 2.0,
        "risk_off": -5.0,
    }
    score_regime = regime_bonus_map.get(market_regime, 0.0)
    if not risk_on:
        score_regime = min(score_regime, 0.0)

    point_score = clamp(
        score_base + score_return + score_drawdown + score_signal + score_regime,
        0.0,
        100.0,
    )
    return {
        "point_score": point_score,
        "point_score_return": score_return,
        "point_score_drawdown": score_drawdown,
        "point_score_signal": score_signal,
        "point_score_regime": score_regime,
        "point_portfolio_return": total_return,
        "point_portfolio_drawdown": current_drawdown,
    }


def mark_to_market_equity(cash: float, quantities: Dict[str, float], prices: Dict[str, float]) -> float:
    return cash + sum(
        quantity * prices[pair]
        for pair, quantity in quantities.items()
        if pair in prices and prices[pair] > 0
    )


def current_notional(quantities: Dict[str, float], prices: Dict[str, float]) -> Dict[str, float]:
    return {
        pair: quantity * prices[pair]
        for pair, quantity in quantities.items()
        if pair in prices and prices[pair] > 0
    }


def sync_position_meta(
    state: SimBacktestState,
    quantities: Dict[str, float],
    prices: Dict[str, float],
    ts_ms: int,
) -> None:
    live_pairs = set(quantities)
    for pair in list(state.positions_meta):
        if pair not in live_pairs:
            state.positions_meta.pop(pair, None)
    for pair, quantity in quantities.items():
        price = prices.get(pair, 0.0)
        meta = state.positions_meta.get(pair)
        if meta is None:
            state.positions_meta[pair] = SimPositionMeta(
                pair=pair,
                quantity=quantity,
                entry_price=price,
                highest_price=price,
                last_trade_ts=ts_ms,
                last_signal_score=0.0,
                last_reason="recovered_from_balance",
            )
        else:
            meta.quantity = quantity


def rebalance_notional_threshold(cfg: Config, equity: float) -> float:
    return max(cfg.min_rebalance_notional, equity * cfg.rebalance_threshold)


def quantity_for_notional(cfg: Config, trade_pairs: Dict[str, Dict[str, Any]], pair: str, notional_usd: float, price: float) -> float:
    rules = trade_pairs.get(pair, {})
    amount_precision = int(rules.get("AmountPrecision", 6))
    min_order = safe_float(rules.get("MiniOrder", 1.0))
    if price <= 0:
        return 0.0
    quantity = round_down(notional_usd / price, amount_precision)
    if quantity <= 0 or quantity * price < min_order:
        return 0.0
    return quantity


def quantity_for_fraction(
    cfg: Config,
    trade_pairs: Dict[str, Dict[str, Any]],
    pair: str,
    quantity: float,
    price: float,
    fraction: float,
) -> float:
    if quantity <= 0 or price <= 0:
        return 0.0
    fraction = clamp(fraction, 0.0, 1.0)
    if fraction <= 0:
        return 0.0
    if fraction >= 1.0:
        return quantity
    return min(quantity, quantity_for_notional(cfg, trade_pairs, pair, quantity * price * fraction, price))


def execution_price(side: str, market_row: Dict[str, float], quantity: float, reference_price: float) -> float:
    if quantity <= 0:
        return 0.0
    mid = float(market_row.get("price", reference_price))
    bid = float(market_row.get("bid", mid))
    ask = float(market_row.get("ask", mid))
    base_price = ask if side == "BUY" else bid
    if base_price <= 0:
        return 0.0

    notional = max(quantity * max(reference_price, 0.0), quantity * base_price)
    minute_liquidity = float(market_row.get("quote_value", 0.0))
    daily_liquidity = float(market_row.get("unit_trade_value", 0.0))
    fallback_liquidity = max(daily_liquidity / (24 * 60) * 0.25, 1.0)
    effective_liquidity = max(minute_liquidity, fallback_liquidity)
    participation = notional / effective_liquidity if effective_liquidity > 0 else 1.0
    impact = clamp(math.sqrt(max(participation, 0.0)) * 0.0005, 0.0, 0.0030)
    if side == "BUY":
        return base_price * (1.0 + impact)
    return max(base_price * (1.0 - impact), 1e-12)


def build_position_meta(state: SimBacktestState, pair: str, quantity: float, price: float, score: float, ts_ms: int) -> SimPositionMeta:
    meta = state.positions_meta.get(pair)
    if meta is None:
        meta = SimPositionMeta(
            pair=pair,
            quantity=quantity,
            entry_price=price,
            highest_price=price,
            last_trade_ts=ts_ms,
            last_signal_score=score,
            last_reason="recovered_from_balance",
        )
    meta.quantity = quantity
    meta.highest_price = max(meta.highest_price, price)
    meta.last_signal_score = score
    return meta


def exit_reasons(cfg: Config, pair: str, meta: SimPositionMeta, price: float, score: float, targets: Dict[str, float]) -> List[str]:
    reasons: List[str] = []
    if price <= meta.entry_price * (1 - cfg.per_position_stop_loss):
        reasons.append("stop_loss")
    if price <= meta.highest_price * (1 - cfg.per_position_trailing_stop):
        reasons.append("trailing_stop")
    if score < cfg.exit_score_threshold:
        reasons.append("score_decay")
    if pair not in targets:
        reasons.append("not_in_targets")
    return reasons


def set_cooldown(state: SimBacktestState, pair: str, ts_ms: int, minutes: int) -> None:
    state.cooldown_until[pair] = ts_ms + minutes * 60_000


def in_cooldown(state: SimBacktestState, pair: str, ts_ms: int) -> bool:
    return ts_ms < state.cooldown_until.get(pair, 0)


def unrealized_return(entry_price: float, price: float) -> float:
    if entry_price <= 0 or price <= 0:
        return 0.0
    return price / entry_price - 1.0


def run_backtest(
    cfg: Config,
    market_data: Dict[str, pd.DataFrame],
    pair_to_symbol: Dict[str, str],
    trade_pairs: Dict[str, Dict[str, Any]],
    initial_equity: float,
    rebalance_minutes: int,
    fee_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    strategy = MomentumStrategy(cfg)
    history, pair_to_symbol = build_history_buffers(pair_to_symbol, cfg)
    common_index = sorted(set().union(*(frame.index for frame in market_data.values())))

    cash = initial_equity
    quantities: Dict[str, float] = {}
    state = SimBacktestState(peak_equity=initial_equity, risk_on=False)
    equity_rows: List[Dict[str, float]] = []
    trade_rows: List[Dict[str, float]] = []
    rebalance_returns: List[float] = []
    last_rebalance_equity = initial_equity
    running_peak_equity = initial_equity

    for step, ts in enumerate(common_index):
        ts_ms = int(ts.timestamp() * 1000.0)
        current_prices: Dict[str, float] = {}
        current_market: Dict[str, Dict[str, float]] = {}
        for pair, symbol in pair_to_symbol.items():
            frame = market_data[symbol]
            if ts not in frame.index:
                continue
            row = frame.loc[ts]
            entry = {
                "ts": float(ts_ms),
                "price": float(row["price"]),
                "bid": float(row["bid"]),
                "ask": float(row["ask"]),
                "change_24h": float(row["change_24h"]),
                "unit_trade_value": float(row["unit_trade_value"]),
            }
            history[pair].append(entry)
            current_prices[pair] = float(row["price"])
            current_market[pair] = entry

        if not current_prices:
            continue

        equity_before_rebalance = mark_to_market_equity(cash, quantities, current_prices)
        state.peak_equity = max(state.peak_equity, equity_before_rebalance)
        sync_position_meta(state, quantities, current_prices, ts_ms)

        if step > 0 and step % rebalance_minutes == 0:
            skip_trim_pairs: set[str] = set()
            allow_new_buys = ts_ms >= state.portfolio_reentry_allowed_at
            drawdown = 1.0 - (equity_before_rebalance / state.peak_equity) if state.peak_equity > 0 else 0.0
            if drawdown >= cfg.max_portfolio_drawdown and quantities:
                allow_new_buys = False
                state.portfolio_reentry_allowed_at = max(
                    state.portfolio_reentry_allowed_at,
                    ts_ms + cfg.portfolio_drawdown_cooldown_minutes * 60_000,
                )
                for pair, quantity in list(quantities.items()):
                    price = current_prices.get(pair, 0.0)
                    market_row = current_market.get(pair)
                    if quantity <= 0 or price <= 0:
                        continue
                    sell_quantity = quantity_for_fraction(cfg, trade_pairs, pair, quantity, price, 0.50)
                    if sell_quantity <= 0:
                        continue
                    if market_row is None:
                        continue
                    executed_price = execution_price("SELL", market_row, sell_quantity, price)
                    notional = sell_quantity * executed_price
                    fee_paid = notional * fee_rate
                    cash += notional - fee_paid
                    remaining_quantity = quantities.get(pair, 0.0) - sell_quantity
                    if remaining_quantity > quantity * 1e-6:
                        quantities[pair] = remaining_quantity
                        meta = state.positions_meta.get(pair)
                        if meta:
                            meta.quantity = remaining_quantity
                    else:
                        quantities.pop(pair, None)
                        state.positions_meta.pop(pair, None)
                        set_cooldown(state, pair, ts_ms, cfg.cooldown_minutes)
                    skip_trim_pairs.add(pair)
                    trade_score = score_trade_point(
                        equity_value=mark_to_market_equity(cash, quantities, current_prices),
                        initial_equity=initial_equity,
                        running_peak=max(running_peak_equity, state.peak_equity),
                        signal_weight=0.0,
                        risk_on=state.risk_on,
                        market_regime="risk_off",
                    )
                    trade_rows.append(
                        {
                            "ts": ts.isoformat(),
                            "pair": pair,
                            "symbol": pair_to_symbol[pair],
                            "side": "SELL",
                            "price": executed_price,
                            "quantity": sell_quantity,
                            "notional": notional,
                            "turnover": notional,
                            "fee_paid": fee_paid,
                            "target_weight": 0.0,
                            "signal_score": -999.0,
                            "reason": "portfolio_drawdown",
                            "risk_on": int(bool(state.risk_on)),
                            "market_regime": "risk_off",
                            "signal_regime": "panic_exit",
                            **trade_score,
                        }
                    )
                sync_position_meta(state, quantities, current_prices, ts_ms)
                equity_before_rebalance = mark_to_market_equity(cash, quantities, current_prices)

            signals = strategy.generate_signals(
                history=history,
                trade_pairs=trade_pairs,
                positions=quantities.copy(),
                prev_risk_on=state.risk_on,
            )
            state.risk_on = bool(signals["risk_on"])
            target_weights = signals["weights"]
            features = signals["features"]
            signal_regime = str(signals["regime"]["regime"])
            market_regime = str(signals["portfolio_risk"]["market_regime"])

            if features:
                for pair, quantity in list(quantities.items()):
                    price = current_prices.get(pair, 0.0)
                    market_row = current_market.get(pair)
                    if price <= 0:
                        continue
                    score = features.get(pair, {}).get("score", -999.0)
                    meta = build_position_meta(state, pair, quantity, price, score, ts_ms)
                    reasons = exit_reasons(cfg, pair, meta, price, score, target_weights)
                    if reasons:
                        current_return = unrealized_return(meta.entry_price, price)
                        protected_profit = current_return >= cfg.profit_protect_threshold
                        if "stop_loss" in reasons or "trailing_stop" in reasons:
                            sell_fraction = 1.0
                        elif "not_in_targets" in reasons:
                            sell_fraction = (
                                cfg.profit_protect_not_in_targets_fraction
                                if protected_profit
                                else 0.30
                            )
                        else:
                            sell_fraction = (
                                cfg.profit_protect_score_decay_fraction
                                if protected_profit
                                else 0.25
                            )
                        sell_quantity = quantity_for_fraction(cfg, trade_pairs, pair, quantity, price, sell_fraction)
                        if sell_quantity <= 0:
                            state.positions_meta[pair] = meta
                            continue
                        if market_row is None:
                            state.positions_meta[pair] = meta
                            continue
                        executed_price = execution_price("SELL", market_row, sell_quantity, price)
                        notional = sell_quantity * executed_price
                        fee_paid = notional * fee_rate
                        cash += notional - fee_paid
                        remaining_quantity = quantities.get(pair, 0.0) - sell_quantity
                        if remaining_quantity > quantity * 1e-6:
                            quantities[pair] = remaining_quantity
                            meta.quantity = remaining_quantity
                            state.positions_meta[pair] = meta
                            if sell_fraction < 1.0:
                                skip_trim_pairs.add(pair)
                        else:
                            quantities.pop(pair, None)
                            state.positions_meta.pop(pair, None)
                            set_cooldown(state, pair, ts_ms, cfg.cooldown_minutes)
                        post_trade_equity = mark_to_market_equity(cash, quantities, current_prices)
                        trade_score = score_trade_point(
                            equity_value=post_trade_equity,
                            initial_equity=initial_equity,
                            running_peak=max(running_peak_equity, state.peak_equity, post_trade_equity),
                            signal_weight=0.0,
                            risk_on=bool(signals["risk_on"]),
                            market_regime=market_regime,
                        )
                        trade_rows.append(
                            {
                                "ts": ts.isoformat(),
                                "pair": pair,
                                "symbol": pair_to_symbol[pair],
                                "side": "SELL",
                                "price": executed_price,
                                "quantity": sell_quantity,
                                "notional": notional,
                                "turnover": notional,
                                "fee_paid": fee_paid,
                                "target_weight": target_weights.get(pair, 0.0),
                                "signal_score": score,
                                "reason": "+".join(reasons),
                                "risk_on": int(bool(signals["risk_on"])),
                                "market_regime": market_regime,
                                "signal_regime": signal_regime,
                                **trade_score,
                            }
                        )
                    else:
                        state.positions_meta[pair] = meta

                portfolio_equity = mark_to_market_equity(cash, quantities, current_prices)
                notional_map = current_notional(quantities, current_prices)
                threshold = rebalance_notional_threshold(cfg, portfolio_equity)
                for pair, quantity in list(quantities.items()):
                    if pair in skip_trim_pairs:
                        continue
                    price = current_prices.get(pair, 0.0)
                    market_row = current_market.get(pair)
                    if price <= 0:
                        continue
                    target_usd = portfolio_equity * target_weights.get(pair, 0.0)
                    current_usd = notional_map.get(pair, 0.0)
                    trim_usd = current_usd - target_usd
                    if trim_usd < threshold:
                        continue
                    sell_quantity = quantity if target_usd <= 0 else quantity_for_notional(cfg, trade_pairs, pair, trim_usd, price)
                    if sell_quantity <= 0:
                        continue
                    if market_row is None:
                        continue
                    score = features.get(pair, {}).get("score", -999.0)
                    reason = "target_exit_retry" if target_usd <= 0 else "target_trim"
                    executed_price = execution_price("SELL", market_row, sell_quantity, price)
                    notional = sell_quantity * executed_price
                    fee_paid = notional * fee_rate
                    cash += notional - fee_paid
                    remaining_quantity = quantities.get(pair, 0.0) - sell_quantity
                    if remaining_quantity > quantity * 1e-6:
                        quantities[pair] = remaining_quantity
                        meta = state.positions_meta.get(pair)
                        if meta:
                            meta.quantity = remaining_quantity
                    else:
                        quantities.pop(pair, None)
                        state.positions_meta.pop(pair, None)
                        if target_usd <= 0:
                            set_cooldown(state, pair, ts_ms, cfg.cooldown_minutes)
                    post_trade_equity = mark_to_market_equity(cash, quantities, current_prices)
                    trade_score = score_trade_point(
                        equity_value=post_trade_equity,
                        initial_equity=initial_equity,
                        running_peak=max(running_peak_equity, state.peak_equity, post_trade_equity),
                        signal_weight=target_weights.get(pair, 0.0),
                        risk_on=bool(signals["risk_on"]),
                        market_regime=market_regime,
                    )
                    trade_rows.append(
                        {
                            "ts": ts.isoformat(),
                            "pair": pair,
                            "symbol": pair_to_symbol[pair],
                            "side": "SELL",
                            "price": executed_price,
                            "quantity": sell_quantity,
                            "notional": notional,
                            "turnover": notional,
                            "fee_paid": fee_paid,
                            "target_weight": target_weights.get(pair, 0.0),
                            "signal_score": score,
                            "reason": reason,
                            "risk_on": int(bool(signals["risk_on"])),
                            "market_regime": market_regime,
                            "signal_regime": signal_regime,
                            **trade_score,
                        }
                    )

                portfolio_equity = mark_to_market_equity(cash, quantities, current_prices)
                usd_free = cash
                threshold = rebalance_notional_threshold(cfg, portfolio_equity)
                notional_map = current_notional(quantities, current_prices)
                for pair, weight in sorted(target_weights.items(), key=lambda item: item[1], reverse=True):
                    if not allow_new_buys:
                        break
                    if in_cooldown(state, pair, ts_ms):
                        continue
                    price = current_prices.get(pair, 0.0)
                    market_row = current_market.get(pair)
                    if price <= 0 or pair not in features:
                        continue
                    target_usd = portfolio_equity * weight
                    current_usd = notional_map.get(pair, 0.0)
                    diff_usd = target_usd - current_usd
                    if diff_usd < threshold:
                        continue
                    usable_cash = max(0.0, usd_free - portfolio_equity * cfg.cash_buffer)
                    if usable_cash <= 0:
                        continue
                    buy_usd = min(diff_usd, usable_cash)
                    buy_quantity = quantity_for_notional(cfg, trade_pairs, pair, buy_usd, price)
                    if buy_quantity <= 0:
                        continue
                    if market_row is None:
                        continue
                    executed_price = execution_price("BUY", market_row, buy_quantity, price)
                    notional = buy_quantity * executed_price
                    fee_paid = notional * fee_rate
                    total_cost = notional + fee_paid
                    if total_cost > cash + 1e-9:
                        continue
                    cash -= total_cost
                    quantities[pair] = quantities.get(pair, 0.0) + buy_quantity
                    score = features[pair]["score"]
                    meta = state.positions_meta.get(pair)
                    if meta is None:
                        state.positions_meta[pair] = SimPositionMeta(
                            pair=pair,
                            quantity=buy_quantity,
                            entry_price=executed_price,
                            highest_price=executed_price,
                            last_trade_ts=ts_ms,
                            last_signal_score=score,
                            last_reason="target_rebalance",
                        )
                    else:
                        previous_quantity = meta.quantity
                        new_quantity = previous_quantity + buy_quantity
                        if new_quantity > 0:
                            meta.entry_price = ((meta.entry_price * previous_quantity) + (executed_price * buy_quantity)) / new_quantity
                        meta.quantity = new_quantity
                        meta.highest_price = max(meta.highest_price, executed_price)
                        meta.last_trade_ts = ts_ms
                        meta.last_signal_score = score
                        meta.last_reason = "target_rebalance"
                    usd_free -= notional
                    notional_map[pair] = quantities[pair] * price
                    post_trade_equity = mark_to_market_equity(cash, quantities, current_prices)
                    trade_score = score_trade_point(
                        equity_value=post_trade_equity,
                        initial_equity=initial_equity,
                        running_peak=max(running_peak_equity, state.peak_equity, post_trade_equity),
                        signal_weight=weight,
                        risk_on=bool(signals["risk_on"]),
                        market_regime=market_regime,
                    )
                    trade_rows.append(
                        {
                            "ts": ts.isoformat(),
                            "pair": pair,
                            "symbol": pair_to_symbol[pair],
                            "side": "BUY",
                            "price": executed_price,
                            "quantity": buy_quantity,
                            "notional": notional,
                            "turnover": notional,
                            "fee_paid": fee_paid,
                            "target_weight": weight,
                            "signal_score": score,
                            "reason": "target_rebalance",
                            "risk_on": int(bool(signals["risk_on"])),
                            "market_regime": market_regime,
                            "signal_regime": signal_regime,
                            **trade_score,
                        }
                    )

            sync_position_meta(state, quantities, current_prices, ts_ms)
            post_trade_equity = mark_to_market_equity(cash, quantities, current_prices)
            if last_rebalance_equity > 0:
                rebalance_returns.append(post_trade_equity / last_rebalance_equity - 1.0)
            last_rebalance_equity = post_trade_equity
            equity_before_rebalance = post_trade_equity

        marked_equity = mark_to_market_equity(cash, quantities, current_prices)
        running_peak_equity = max(running_peak_equity, marked_equity)
        equity_rows.append(
            {
                "ts": ts.isoformat(),
                "equity": marked_equity,
                "cash": cash,
                "gross_exposure": sum(current_notional(quantities, current_prices).values()) / marked_equity if marked_equity > 0 else 0.0,
                "positions": len(quantities),
            }
        )

    equity_curve = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    metrics = calculate_metrics(equity_curve, rebalance_returns)
    metrics["start_equity"] = float(initial_equity)
    metrics["end_equity"] = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else float(initial_equity)
    metrics["trade_count"] = float(len(trades))
    metrics["symbols"] = float(len(market_data))
    return equity_curve, trades, metrics


def summarize_run(
    symbols: List[str],
    roostoo_pairs: List[str],
    start_date: date,
    end_date: date,
    cfg: Config,
    metrics: Dict[str, float],
    scorecard: Dict[str, float],
) -> pd.DataFrame:
    rows = [
        {"metric": "symbols", "value": ",".join(symbols)},
        {"metric": "roostoo_pairs", "value": ",".join(roostoo_pairs)},
        {"metric": "start_date", "value": start_date.isoformat()},
        {"metric": "end_date", "value": end_date.isoformat()},
        {"metric": "poll_minutes", "value": cfg.poll_seconds / 60.0},
        {"metric": "lookback_minutes", "value": cfg.lookback_minutes},
        {"metric": "top_n", "value": cfg.top_n},
    ]
    for key, value in metrics.items():
        rows.append({"metric": key, "value": value})
    for key, value in scorecard.items():
        rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows)


def save_equity_plot(equity_curve: pd.DataFrame, output_path: Path) -> None:
    if equity_curve.empty:
        return

    plot_frame = equity_curve.copy()
    plot_frame["ts"] = pd.to_datetime(plot_frame["ts"], utc=True, errors="coerce")
    plot_frame = plot_frame.dropna(subset=["ts"]).copy()
    if plot_frame.empty:
        return

    plot_frame["position_value"] = plot_frame["equity"] - plot_frame["cash"]
    series = {
        "Total Equity": ("#16324f", plot_frame["equity"].astype(float).tolist()),
        "Cash": ("#2a9d8f", plot_frame["cash"].astype(float).tolist()),
        "Position Value": ("#e76f51", plot_frame["position_value"].astype(float).tolist()),
    }

    values = [value for _label, (_color, data) in series.items() for value in data]
    if not values:
        return

    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0

    width = 1400
    height = 760
    left = 90
    right = 30
    top = 50
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    count = max(len(plot_frame) - 1, 1)

    def x_pos(index: int) -> float:
        return left + plot_width * index / count

    def y_pos(value: float) -> float:
        scaled = (value - min_value) / (max_value - min_value)
        return top + plot_height * (1.0 - scaled)

    grid_lines = []
    y_labels = []
    for idx in range(6):
        value = min_value + (max_value - min_value) * idx / 5
        y = y_pos(value)
        grid_lines.append(f"<line x1='{left}' y1='{y:.2f}' x2='{width - right}' y2='{y:.2f}' stroke='#d9dee5' stroke-width='1' />")
        y_labels.append(f"<text x='{left - 10}' y='{y + 4:.2f}' font-size='12' text-anchor='end' fill='#425466'>{value:,.0f}</text>")

    x_labels = []
    for idx in range(5):
        point_idx = round((len(plot_frame) - 1) * idx / 4) if len(plot_frame) > 1 else 0
        x = x_pos(point_idx)
        ts_label = plot_frame.iloc[point_idx]["ts"].strftime("%Y-%m-%d")
        x_labels.append(f"<text x='{x:.2f}' y='{height - 28}' font-size='12' text-anchor='middle' fill='#425466'>{ts_label}</text>")

    paths = []
    legend = []
    legend_y = 24
    legend_x = left
    offset = 0
    for label, (color, data) in series.items():
        points = " ".join(f"{x_pos(idx):.2f},{y_pos(value):.2f}" for idx, value in enumerate(data))
        paths.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{points}' />")
        legend.append(f"<line x1='{legend_x + offset}' y1='{legend_y}' x2='{legend_x + offset + 18}' y2='{legend_y}' stroke='{color}' stroke-width='3' />")
        legend.append(f"<text x='{legend_x + offset + 24}' y='{legend_y + 4}' font-size='13' fill='#1f2933'>{label}</text>")
        offset += 180

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='#ffffff' />
<text x='{left}' y='28' font-size='20' font-weight='700' fill='#0f172a'>Backtest Asset Curve</text>
<text x='{left}' y='46' font-size='12' fill='#51606f'>Total equity = cash + position value</text>
{''.join(grid_lines)}
<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='#6b7280' stroke-width='1.2' />
<line x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}' stroke='#6b7280' stroke-width='1.2' />
{''.join(paths)}
{''.join(y_labels)}
{''.join(x_labels)}
{''.join(legend)}
</svg>"""
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    end_date = date.fromisoformat(args.end_date)
    if args.days is not None:
        start_date = end_date - timedelta(days=max(args.days - 1, 0))
    else:
        start_date = end_date - timedelta(days=max(args.months * 31, 1))

    cfg = Config.from_env()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exchange_info = fetch_roostoo_exchange_info(args.roostoo_base_url)
    pair_to_symbol, trade_pairs = resolve_roostoo_universe(args.symbols, exchange_info)
    if not pair_to_symbol:
        raise SystemExit("No Roostoo tradable pairs could be resolved to Binance symbols.")

    symbols = list(pair_to_symbol.values())
    market_data = load_market_data(symbols, start_date, end_date, cache_dir)
    if not market_data:
        raise SystemExit("No Binance aggTrades data could be loaded for the selected symbols/date range.")

    available_pair_to_symbol = {
        pair: symbol
        for pair, symbol in pair_to_symbol.items()
        if symbol in market_data
    }
    available_trade_pairs = {
        pair: trade_pairs.get(pair, {})
        for pair in available_pair_to_symbol
    }
    if not available_pair_to_symbol:
        raise SystemExit("Resolved Roostoo pairs did not have matching Binance aggTrades data.")

    initial_equity = args.initial_equity
    if initial_equity is None:
        initial_equity = float(exchange_info.get("InitialWallet", {}).get("USD", 50_000.0))

    equity_curve, trades, metrics = run_backtest(
        cfg=cfg,
        market_data=market_data,
        pair_to_symbol=available_pair_to_symbol,
        trade_pairs=available_trade_pairs,
        initial_equity=initial_equity,
        rebalance_minutes=args.rebalance_minutes,
        fee_rate=args.fee_rate,
    )
    scorecard = score_total_backtest(
        metrics=metrics,
        executed_symbols=len(available_pair_to_symbol),
        requested_symbols=max(len(pair_to_symbol), 1),
        rebalance_minutes=args.rebalance_minutes,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_run(
        symbols=list(available_pair_to_symbol.values()),
        roostoo_pairs=list(available_pair_to_symbol.keys()),
        start_date=start_date,
        end_date=end_date,
        cfg=cfg,
        metrics=metrics,
        scorecard=scorecard,
    )
    summary_path = run_dir / f"summary_{stamp}.csv"
    equity_path = run_dir / f"equity_curve_{stamp}.csv"
    trades_path = run_dir / f"trades_{stamp}.csv"
    equity_plot_path = run_dir / f"equity_curve_{stamp}.svg"

    summary.to_csv(summary_path, index=False)
    equity_curve.to_csv(equity_path, index=False)
    trades.to_csv(trades_path, index=False)
    save_equity_plot(equity_curve, equity_plot_path)

    print("Backtest completed.")
    print(f"Roostoo pairs: {', '.join(available_pair_to_symbol)}")
    print(f"Binance symbols: {', '.join(available_pair_to_symbol.values())}")
    print(f"Date range: {start_date} -> {end_date}")
    print(f"Initial equity: {initial_equity:.2f}")
    print(f"Final equity: {metrics.get('end_equity', initial_equity):.2f}")
    print(f"Total return: {metrics.get('total_return', 0.0):.2%}")
    print(f"Annualized return: {metrics.get('annualized_return', 0.0):.2%}")
    print(f"Sharpe: {metrics.get('sharpe', 0.0):.3f}")
    print(f"Sortino: {metrics.get('sortino', 0.0):.3f}")
    print(f"Calmar: {metrics.get('calmar', 0.0):.3f}")
    print(f"Max drawdown: {metrics.get('max_drawdown', 0.0):.2%}")
    print(f"Trades: {int(metrics.get('trade_count', 0.0))}")
    print(f"Total score: {scorecard.get('total_score', 0.0):.1f}/100")
    print(f"Summary CSV: {summary_path}")
    print(f"Equity CSV: {equity_path}")
    print(f"Trades CSV: {trades_path}")
    print(f"Equity Plot: {equity_plot_path}")
    print()
    print("Note: this backtest now aligns the universe and initial wallet to Roostoo exchangeInfo,")
    print("maps Roostoo /USD pairs to Binance USDT spot aggTrades, and enforces max 1 rebalance every hour by default.")


if __name__ == "__main__":
    main()
