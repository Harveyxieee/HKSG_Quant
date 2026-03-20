from __future__ import annotations

import argparse
import io
import json
import math
import time
import zipfile
from collections import deque
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
        default=30,
        help="Rebalance interval in minutes. Default: 30 to cap execution at one trade every half hour",
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
    minute["bid"] = minute["price"]
    minute["ask"] = minute["price"]
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
    score_compliance = 5.0 if rebalance_minutes >= 30 else clamp(rebalance_minutes / 30.0 * 5.0, 0.0, 5.0)

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
    prev_risk_on = False
    equity_rows: List[Dict[str, float]] = []
    trade_rows: List[Dict[str, float]] = []
    rebalance_returns: List[float] = []
    last_rebalance_equity = initial_equity
    running_peak_equity = initial_equity

    for step, ts in enumerate(common_index):
        current_prices: Dict[str, float] = {}
        for pair, symbol in pair_to_symbol.items():
            frame = market_data[symbol]
            if ts not in frame.index:
                continue
            row = frame.loc[ts]
            entry = {
                "ts": float(ts.timestamp() * 1000.0),
                "price": float(row["price"]),
                "bid": float(row["bid"]),
                "ask": float(row["ask"]),
                "change_24h": float(row["change_24h"]),
                "unit_trade_value": float(row["unit_trade_value"]),
            }
            history[pair].append(entry)
            current_prices[pair] = float(row["price"])

        if not current_prices:
            continue

        equity_before_rebalance = cash + sum(
            quantity * current_prices[pair]
            for pair, quantity in quantities.items()
            if pair in current_prices
        )

        if step > 0 and step % rebalance_minutes == 0:
            signals = strategy.generate_signals(
                history=history,
                trade_pairs=trade_pairs,
                positions=quantities.copy(),
                prev_risk_on=prev_risk_on,
            )
            prev_risk_on = bool(signals["risk_on"])
            target_weights = signals["weights"]

            target_values: Dict[str, float] = {}
            all_pairs = set(quantities) | set(target_weights)
            for pair in all_pairs:
                price = current_prices.get(pair)
                if not price or price <= 0:
                    continue
                target_value = equity_before_rebalance * target_weights.get(pair, 0.0)
                target_values[pair] = target_value

            candidate_trades: List[Dict[str, float | str]] = []
            for pair in all_pairs:
                price = current_prices.get(pair)
                if not price or price <= 0:
                    continue
                current_value = quantities.get(pair, 0.0) * price
                target_value = target_values.get(pair, 0.0)
                delta_value = target_value - current_value
                if abs(delta_value) < 1e-8:
                    continue
                candidate_trades.append(
                    {
                        "pair": pair,
                        "price": price,
                        "current_value": current_value,
                        "target_value": target_value,
                        "delta_value": delta_value,
                    }
                )

            sell_candidates = [item for item in candidate_trades if float(item["delta_value"]) < 0]
            buy_candidates = [item for item in candidate_trades if float(item["delta_value"]) > 0]
            ordered = sorted(
                sell_candidates,
                key=lambda item: abs(float(item["delta_value"])),
                reverse=True,
            ) + sorted(
                buy_candidates,
                key=lambda item: abs(float(item["delta_value"])),
                reverse=True,
            )

            executed_notional = 0.0
            executed_fee = 0.0
            executed_pair = None
            executed_side = None
            executed_quantity = 0.0
            executed_target_weight = 0.0
            for candidate in ordered:
                pair = str(candidate["pair"])
                price = float(candidate["price"])
                delta_value = float(candidate["delta_value"])
                current_value = float(candidate["current_value"])

                if delta_value < 0:
                    sell_value = min(abs(delta_value), current_value)
                    if sell_value <= 0:
                        continue
                    sell_quantity = min(quantities.get(pair, 0.0), sell_value / price)
                    if sell_quantity <= 0:
                        continue
                    executed_notional = sell_quantity * price
                    executed_fee = executed_notional * fee_rate
                    cash += executed_notional - executed_fee
                    remaining_quantity = quantities.get(pair, 0.0) - sell_quantity
                    if remaining_quantity > 1e-12:
                        quantities[pair] = remaining_quantity
                    else:
                        quantities.pop(pair, None)
                    executed_pair = pair
                    executed_side = "SELL"
                    executed_quantity = sell_quantity
                    executed_target_weight = target_weights.get(pair, 0.0)
                    break

                affordable_value = cash / (1.0 + fee_rate) if fee_rate >= 0 else cash
                buy_value = min(delta_value, max(affordable_value, 0.0))
                if buy_value <= 0:
                    continue
                buy_quantity = buy_value / price
                if buy_quantity <= 0:
                    continue
                executed_notional = buy_quantity * price
                executed_fee = executed_notional * fee_rate
                cash -= executed_notional + executed_fee
                quantities[pair] = quantities.get(pair, 0.0) + buy_quantity
                executed_pair = pair
                executed_side = "BUY"
                executed_quantity = buy_quantity
                executed_target_weight = target_weights.get(pair, 0.0)
                break

            post_trade_equity = cash + sum(
                quantity * current_prices[pair]
                for pair, quantity in quantities.items()
                if pair in current_prices
            )
            if executed_pair is not None:
                trade_score = score_trade_point(
                    equity_value=post_trade_equity,
                    initial_equity=initial_equity,
                    running_peak=max(running_peak_equity, post_trade_equity),
                    signal_weight=executed_target_weight,
                    risk_on=bool(signals["risk_on"]),
                    market_regime=str(signals["portfolio_risk"]["market_regime"]),
                )
                trade_rows.append(
                    {
                        "ts": ts.isoformat(),
                        "pair": executed_pair,
                        "symbol": pair_to_symbol[executed_pair],
                        "side": executed_side,
                        "price": current_prices[executed_pair],
                        "quantity": executed_quantity,
                        "notional": executed_notional,
                        "turnover": executed_notional,
                        "fee_paid": executed_fee,
                        "target_weight": executed_target_weight,
                        "risk_on": int(bool(signals["risk_on"])),
                        "market_regime": signals["portfolio_risk"]["market_regime"],
                        "signal_regime": signals["regime"]["regime"],
                        **trade_score,
                    }
                )
                running_peak_equity = max(running_peak_equity, post_trade_equity)

            if last_rebalance_equity > 0:
                rebalance_returns.append(post_trade_equity / last_rebalance_equity - 1.0)
            last_rebalance_equity = post_trade_equity
            equity_before_rebalance = post_trade_equity

        marked_equity = cash + sum(
            quantity * current_prices[pair]
            for pair, quantity in quantities.items()
            if pair in current_prices
        )
        running_peak_equity = max(running_peak_equity, marked_equity)
        equity_rows.append(
            {
                "ts": ts.isoformat(),
                "equity": marked_equity,
                "cash": cash,
                "gross_exposure": sum(
                    quantity * current_prices[pair]
                    for pair, quantity in quantities.items()
                    if pair in current_prices
                ) / marked_equity if marked_equity > 0 else 0.0,
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
    summary = summarize_run(
        symbols=list(available_pair_to_symbol.values()),
        roostoo_pairs=list(available_pair_to_symbol.keys()),
        start_date=start_date,
        end_date=end_date,
        cfg=cfg,
        metrics=metrics,
        scorecard=scorecard,
    )
    summary_path = output_dir / f"summary_{stamp}.csv"
    equity_path = output_dir / f"equity_curve_{stamp}.csv"
    trades_path = output_dir / f"trades_{stamp}.csv"

    summary.to_csv(summary_path, index=False)
    equity_curve.to_csv(equity_path, index=False)
    trades.to_csv(trades_path, index=False)

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
    print()
    print("Note: this backtest now aligns the universe and initial wallet to Roostoo exchangeInfo,")
    print("maps Roostoo /USD pairs to Binance USDT spot aggTrades, and enforces max 1 trade every 30 minutes by default.")


if __name__ == "__main__":
    main()
