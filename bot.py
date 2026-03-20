#!/usr/bin/env python3
from __future__ import annotations

import atexit
import csv
import json
import logging
import math
import os
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from api import RoostooClient, UnknownOrderStateError
from strategy import MomentumStrategy


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default).lower()).strip().lower() == "true"


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    api_secret: str
    bot_name: str
    poll_seconds: int
    lookback_minutes: int
    min_history: int
    data_dir: Path
    log_dir: Path
    state_file: Path
    history_file: Path
    lock_file: Path
    app_log: Path
    request_log_csv: Path
    trade_log_csv: Path
    portfolio_log_csv: Path
    signal_log_csv: Path
    top_n: int
    target_gross_exposure: float
    max_single_weight: float
    cash_buffer: float
    entry_score_threshold: float
    exit_score_threshold: float
    rebalance_threshold: float
    min_rebalance_notional: float
    spread_threshold: float
    min_24h_dollar_vol: float
    max_pump_distance: float
    market_median_60m_threshold: float
    market_up_ratio_threshold: float
    regime_exit_median_60m_threshold: float
    regime_exit_up_ratio_threshold: float
    market_positive_score_ratio_threshold: float
    regime_exit_positive_score_ratio_threshold: float
    vol_floor: float
    vol_cap: float
    holding_score_bonus: float
    risk_cov_window: int
    risk_data_frequency: str
    risk_return_method: str
    risk_min_periods: int
    risk_on_portfolio_vol_threshold: float
    risk_off_portfolio_vol_threshold: float
    risk_on_correlation_threshold: float
    risk_off_correlation_threshold: float
    diversification_breakdown_corr_threshold: float
    risk_on_exposure_multiplier: float
    neutral_exposure_multiplier: float
    risk_off_exposure_multiplier: float
    enable_volatility_targeting: bool
    target_portfolio_volatility: float
    min_vol_target_scale: float
    max_vol_target_scale: float
    risk_vol_score_weight: float
    risk_corr_score_weight: float
    risk_on_weight_penalty_scale: float
    neutral_weight_penalty_scale: float
    risk_off_weight_penalty_scale: float
    risk_weight_vol_penalty: float
    risk_weight_corr_penalty: float
    diversification_alt_weight_multiplier: float
    diversification_core_asset_multiplier: float
    diversification_breakdown_exposure_multiplier: float
    diversification_core_assets: str
    high_vol_feature_cutoff_multiplier: float
    high_vol_feature_weight_multiplier: float
    per_position_stop_loss: float
    per_position_trailing_stop: float
    max_portfolio_drawdown: float
    cooldown_minutes: int
    request_timeout: int
    max_retries: int
    retry_sleep_seconds: float
    order_failure_pause_seconds: int
    loop_error_backoff_cap_seconds: int
    cancel_all_on_start: bool
    dry_run: bool

    @classmethod
    def from_env(cls) -> "Config":
        data_dir = Path(os.getenv("BOT_DATA_DIR", "./data"))
        log_dir = data_dir / "logs"
        return cls(
            base_url=os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com").rstrip("/"),
            api_key=os.getenv("ROOSTOO_API_KEY", "YZJpIGyMhk1efgdP8qZM0LTZZ2eFJh35ovWByPgSG73XS5OeWruM8XygCqHypBK7"),
            api_secret=os.getenv("ROOSTOO_API_SECRET", "PTtlKF7Vwed7MfiUCz6G6PySVDH5zP8Vjz4lmIYuxQyrjq5EvseYAJV3jAhVnYyK"),
            bot_name=os.getenv("BOT_NAME", "roostoo_prelim_bot"),
            poll_seconds=int(os.getenv("POLL_SECONDS", "60")),
            lookback_minutes=int(os.getenv("LOOKBACK_MINUTES", "360")),
            min_history=int(os.getenv("MIN_HISTORY", "10")),
            data_dir=data_dir,
            log_dir=log_dir,
            state_file=data_dir / "state.json",
            history_file=data_dir / "history.json",
            lock_file=data_dir / "bot.lock",
            app_log=log_dir / "bot.log",
            request_log_csv=log_dir / "requests.csv",
            trade_log_csv=log_dir / "trades.csv",
            portfolio_log_csv=log_dir / "portfolio.csv",
            signal_log_csv=log_dir / "signals.csv",
            top_n=int(os.getenv("TOP_N", "3")),
            target_gross_exposure=float(os.getenv("TARGET_GROSS_EXPOSURE", "0.78")),
            max_single_weight=float(os.getenv("MAX_SINGLE_WEIGHT", "0.28")),
            cash_buffer=float(os.getenv("CASH_BUFFER", "0.20")),
            entry_score_threshold=float(os.getenv("ENTRY_SCORE_THRESHOLD", "0.60")),
            exit_score_threshold=float(os.getenv("EXIT_SCORE_THRESHOLD", "0.12")),
            rebalance_threshold=float(os.getenv("REBALANCE_THRESHOLD", "0.03")),
            min_rebalance_notional=float(os.getenv("MIN_REBALANCE_NOTIONAL", "25")),
            spread_threshold=float(os.getenv("SPREAD_THRESHOLD", "0.006")),
            min_24h_dollar_vol=float(os.getenv("MIN_24H_DOLLAR_VOL", "50000")),
            max_pump_distance=float(os.getenv("MAX_PUMP_DISTANCE", "0.05")),
            market_median_60m_threshold=float(os.getenv("MARKET_MEDIAN_60M_THRESHOLD", "0.0005")),
            market_up_ratio_threshold=float(os.getenv("MARKET_UP_RATIO_THRESHOLD", "0.52")),
            regime_exit_median_60m_threshold=float(os.getenv("REGIME_EXIT_MEDIAN_60M_THRESHOLD", "0.0005")),
            regime_exit_up_ratio_threshold=float(os.getenv("REGIME_EXIT_UP_RATIO_THRESHOLD", "0.48")),
            market_positive_score_ratio_threshold=float(os.getenv("MARKET_POSITIVE_SCORE_RATIO_THRESHOLD", "0.45")),
            regime_exit_positive_score_ratio_threshold=float(os.getenv("REGIME_EXIT_POSITIVE_SCORE_RATIO_THRESHOLD", "0.42")),
            vol_floor=float(os.getenv("VOL_FLOOR", "0.004")),
            vol_cap=float(os.getenv("VOL_CAP", "0.08")),
            holding_score_bonus=float(os.getenv("HOLDING_SCORE_BONUS", "0.08")),
            risk_cov_window=int(os.getenv("RISK_COV_WINDOW", "60")),
            risk_data_frequency=os.getenv("RISK_DATA_FREQUENCY", "auto").strip().lower(),
            risk_return_method=os.getenv("RISK_RETURN_METHOD", "pct_change").strip().lower(),
            risk_min_periods=int(os.getenv("RISK_MIN_PERIODS", "30")),
            risk_on_portfolio_vol_threshold=float(os.getenv("RISK_ON_PORTFOLIO_VOL_THRESHOLD", "0.03")),
            risk_off_portfolio_vol_threshold=float(os.getenv("RISK_OFF_PORTFOLIO_VOL_THRESHOLD", "0.07")),
            risk_on_correlation_threshold=float(os.getenv("RISK_ON_CORRELATION_THRESHOLD", "0.35")),
            risk_off_correlation_threshold=float(os.getenv("RISK_OFF_CORRELATION_THRESHOLD", "0.65")),
            diversification_breakdown_corr_threshold=float(os.getenv("DIVERSIFICATION_BREAKDOWN_CORR_THRESHOLD", "0.72")),
            risk_on_exposure_multiplier=float(os.getenv("RISK_ON_EXPOSURE_MULTIPLIER", "1.0")),
            neutral_exposure_multiplier=float(os.getenv("NEUTRAL_EXPOSURE_MULTIPLIER", "0.75")),
            risk_off_exposure_multiplier=float(os.getenv("RISK_OFF_EXPOSURE_MULTIPLIER", "0.35")),
            enable_volatility_targeting=env_bool("ENABLE_VOLATILITY_TARGETING", False),
            target_portfolio_volatility=float(os.getenv("TARGET_PORTFOLIO_VOLATILITY", "0.04")),
            min_vol_target_scale=float(os.getenv("MIN_VOL_TARGET_SCALE", "0.25")),
            max_vol_target_scale=float(os.getenv("MAX_VOL_TARGET_SCALE", "1.0")),
            risk_vol_score_weight=float(os.getenv("RISK_VOL_SCORE_WEIGHT", "0.5")),
            risk_corr_score_weight=float(os.getenv("RISK_CORR_SCORE_WEIGHT", "0.5")),
            risk_on_weight_penalty_scale=float(os.getenv("RISK_ON_WEIGHT_PENALTY_SCALE", "0.25")),
            neutral_weight_penalty_scale=float(os.getenv("NEUTRAL_WEIGHT_PENALTY_SCALE", "0.75")),
            risk_off_weight_penalty_scale=float(os.getenv("RISK_OFF_WEIGHT_PENALTY_SCALE", "1.25")),
            risk_weight_vol_penalty=float(os.getenv("RISK_WEIGHT_VOL_PENALTY", "1.0")),
            risk_weight_corr_penalty=float(os.getenv("RISK_WEIGHT_CORR_PENALTY", "1.0")),
            diversification_alt_weight_multiplier=float(os.getenv("DIVERSIFICATION_ALT_WEIGHT_MULTIPLIER", "0.45")),
            diversification_core_asset_multiplier=float(os.getenv("DIVERSIFICATION_CORE_ASSET_MULTIPLIER", "1.1")),
            diversification_breakdown_exposure_multiplier=float(os.getenv("DIVERSIFICATION_BREAKDOWN_EXPOSURE_MULTIPLIER", "0.75")),
            diversification_core_assets=os.getenv("DIVERSIFICATION_CORE_ASSETS", "BTC,ETH"),
            high_vol_feature_cutoff_multiplier=float(os.getenv("HIGH_VOL_FEATURE_CUTOFF_MULTIPLIER", "1.15")),
            high_vol_feature_weight_multiplier=float(os.getenv("HIGH_VOL_FEATURE_WEIGHT_MULTIPLIER", "0.8")),
            per_position_stop_loss=float(os.getenv("PER_POSITION_STOP_LOSS", "0.035")),
            per_position_trailing_stop=float(os.getenv("PER_POSITION_TRAILING_STOP", "0.045")),
            max_portfolio_drawdown=float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "0.10")),
            cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "15")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "5")),
            max_retries=int(os.getenv("MAX_RETRIES", "1")),
            retry_sleep_seconds=float(os.getenv("RETRY_SLEEP_SECONDS", "1.5")),
            order_failure_pause_seconds=int(os.getenv("ORDER_FAILURE_PAUSE_SECONDS", "180")),
            loop_error_backoff_cap_seconds=int(os.getenv("LOOP_ERROR_BACKOFF_CAP_SECONDS", "900")),
            cancel_all_on_start=env_bool("CANCEL_ALL_ON_START", False),
            dry_run=env_bool("DRY_RUN", True),
        )


CFG = Config.from_env()
CFG.data_dir.mkdir(parents=True, exist_ok=True)
CFG.log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(CFG.app_log), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("roostoo_bot")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def now_ms() -> int:
    return int(time.time() * 1000)


def round_down(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def append_csv(path: Path, headers: List[str], row: Dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json_atomic(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class InstanceLockError(RuntimeError):
    pass


@dataclass
class PositionMeta:
    pair: str
    quantity: float
    entry_price: float
    highest_price: float
    last_trade_ts: int
    last_signal_score: float
    last_reason: str = ""


@dataclass
class PortfolioSnapshot:
    wallet: Dict[str, Dict[str, float]]
    positions: Dict[str, float]
    equity: float
    usd_free: float
    current_notional: Dict[str, float] = field(default_factory=dict)
    drawdown: float = 0.0


@dataclass
class BotState:
    peak_equity: float = 0.0
    cooldown_until: Dict[str, int] = field(default_factory=dict)
    positions_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    start_ts: int = 0
    risk_on: bool = False

    def __post_init__(self) -> None:
        if self.start_ts == 0:
            self.start_ts = now_ms()


def load_state() -> BotState:
    if not CFG.state_file.exists():
        return BotState()
    try:
        with CFG.state_file.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return BotState(
            peak_equity=safe_float(raw.get("peak_equity")),
            cooldown_until={key: int(value) for key, value in raw.get("cooldown_until", {}).items()},
            positions_meta=dict(raw.get("positions_meta", {})),
            start_ts=int(raw.get("start_ts", now_ms())),
            risk_on=bool(raw.get("risk_on", False)),
        )
    except Exception as exc:
        logger.warning("Failed to load state.json: %s", exc)
        return BotState()


def save_state(state: BotState) -> None:
    write_json_atomic(CFG.state_file, asdict(state))


def load_history() -> Dict[str, Deque[Dict[str, float]]]:
    history: Dict[str, Deque[Dict[str, float]]] = defaultdict(lambda: deque(maxlen=CFG.lookback_minutes))
    if not CFG.history_file.exists():
        return history
    try:
        with CFG.history_file.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            return history
        for pair, rows in raw.items():
            if not isinstance(rows, list):
                continue
            restored = deque(maxlen=CFG.lookback_minutes)
            for row in rows[-CFG.lookback_minutes:]:
                if not isinstance(row, dict):
                    continue
                price = safe_float(row.get("price"))
                if price <= 0:
                    continue
                restored.append(
                    {
                        "ts": safe_float(row.get("ts")),
                        "price": price,
                        "bid": safe_float(row.get("bid")),
                        "ask": safe_float(row.get("ask")),
                        "change_24h": safe_float(row.get("change_24h")),
                        "unit_trade_value": safe_float(row.get("unit_trade_value")),
                    }
                )
            if restored:
                history[pair] = restored
    except Exception as exc:
        logger.warning("Failed to load history.json: %s", exc)
    return history


def save_history(history: Dict[str, Deque[Dict[str, float]]]) -> None:
    payload = {pair: list(series) for pair, series in history.items() if series}
    write_json_atomic(CFG.history_file, payload)


class RoostooMomentumBot:
    def __init__(self, cfg: Config, client: RoostooClient):
        self.cfg = cfg
        self.client = client
        self.strategy = MomentumStrategy(cfg)
        self.state = load_state()
        self.exchange_info = self.client.exchange_info()
        self.trade_pairs: Dict[str, Dict[str, Any]] = self.exchange_info.get("TradePairs", {})
        self.history = load_history()
        self._stop_requested = False
        self._pause_until_ms = 0
        self._consecutive_loop_failures = 0
        self._has_lock = False
        self.last_rebalance_ts = 0
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)
        self.acquire_instance_lock()
        atexit.register(self.shutdown)

    def _handle_stop(self, *_args: Any) -> None:
        self._stop_requested = True
        logger.info("Stop signal received.")

    def acquire_instance_lock(self) -> None:
        if self.cfg.lock_file.exists():
            try:
                with self.cfg.lock_file.open("r", encoding="utf-8") as handle:
                    lock_data = json.load(handle)
            except Exception:
                lock_data = {}
            lock_pid = int(lock_data.get("pid", 0))
            if process_is_alive(lock_pid):
                raise InstanceLockError(f"Bot lock already held by pid {lock_pid}: {self.cfg.lock_file}")
            try:
                self.cfg.lock_file.unlink()
                logger.warning("Removed stale bot lock file: %s", self.cfg.lock_file)
            except FileNotFoundError:
                pass
        try:
            fd = os.open(str(self.cfg.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise InstanceLockError(f"Bot lock already exists: {self.cfg.lock_file}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "bot_name": self.cfg.bot_name, "started_at": now_ms()}, handle, indent=2)
        self._has_lock = True

    def release_instance_lock(self) -> None:
        if not self._has_lock:
            return
        self._has_lock = False
        try:
            if self.cfg.lock_file.exists():
                self.cfg.lock_file.unlink()
        except Exception as exc:
            logger.warning("Failed to release bot lock: %s", exc)

    def shutdown(self) -> None:
        try:
            self.persist_runtime_state()
        except Exception:
            pass
        self.release_instance_lock()
        self.client.close()

    def interruptible_sleep(self, seconds: float) -> None:
        end_at = time.time() + max(0.0, seconds)
        while not self._stop_requested:
            remaining = end_at - time.time()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))

    def register_loop_failure(self, exc: Exception) -> None:
        self._consecutive_loop_failures += 1
        self.client.reset_session()
        capped_power = min(self._consecutive_loop_failures - 1, 5)
        backoff_seconds = min(
            self.cfg.loop_error_backoff_cap_seconds,
            max(self.cfg.poll_seconds, 5) * (2 ** capped_power),
        )
        self._pause_until_ms = max(self._pause_until_ms, now_ms() + int(backoff_seconds * 1000))
        logger.warning(
            "Loop failure #%s. Entering %.1fs backoff. Error=%s",
            self._consecutive_loop_failures,
            backoff_seconds,
            exc,
        )

    def clear_loop_failures(self) -> None:
        self._consecutive_loop_failures = 0
        self._pause_until_ms = 0

    def bootstrap(self) -> None:
        logger.info("Server time synced: %s", self.client.sync_time())
        logger.info("Restored %s pairs of local history.", len(self.history))
        if self.cfg.cancel_all_on_start:
            try:
                logger.info("Cancelling all pending orders on startup.")
                logger.info("Cancel response: %s", self.client.cancel_order())
            except Exception as exc:
                logger.warning("Cancel-all failed: %s", exc)
        logger.info("Pending count: %s", self.client.pending_count())

    def persist_runtime_state(self) -> None:
        save_state(self.state)
        save_history(self.history)

    def fetch_all_tickers(self) -> Dict[str, Any]:
        response = self.client.ticker()
        if not response.get("Success", False):
            raise RuntimeError(f"Ticker fetch failed: {response}")
        data = response.get("Data", {})
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected ticker data: {response}")
        return data

    def update_history(self, tickers: Dict[str, Any]) -> None:
        ts = float(now_ms())
        for pair, ticker in tickers.items():
            price = safe_float(ticker.get("LastPrice"))
            if price <= 0 or pair not in self.trade_pairs:
                continue
            self.history[pair].append(
                {
                    "ts": ts,
                    "price": price,
                    "bid": safe_float(ticker.get("MaxBid")),
                    "ask": safe_float(ticker.get("MinAsk")),
                    "change_24h": safe_float(ticker.get("Change")),
                    "unit_trade_value": safe_float(ticker.get("UnitTradeValue")),
                }
            )

    def get_wallet(self) -> Dict[str, Dict[str, float]]:
        response = self.client.balance()
        if not response.get("Success", False):
            raise RuntimeError(f"Balance fetch failed: {response}")
        wallet: Dict[str, Dict[str, float]] = {}
        for coin, value in response.get("SpotWallet", {}).items():
            wallet[coin] = {"Free": safe_float(value.get("Free")), "Lock": safe_float(value.get("Lock"))}
        return wallet

    def realized_positions(self, wallet: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        positions: Dict[str, float] = {}
        for coin, balance in wallet.items():
            if coin == "USD":
                continue
            quantity = balance["Free"] + balance["Lock"]
            pair = f"{coin}/USD"
            if quantity > 0 and pair in self.trade_pairs:
                positions[pair] = quantity
        return positions

    def pair_price(self, pair: str, tickers: Dict[str, Any]) -> float:
        return safe_float(tickers.get(pair, {}).get("LastPrice"))

    def mark_to_market_equity(self, wallet: Dict[str, Dict[str, float]], tickers: Dict[str, Any]) -> float:
        total_equity = 0.0
        for coin, balance in wallet.items():
            quantity = balance["Free"] + balance["Lock"]
            if quantity <= 0:
                continue
            if coin == "USD":
                total_equity += quantity
            else:
                total_equity += quantity * self.pair_price(f"{coin}/USD", tickers)
        return total_equity

    def build_portfolio_snapshot(self, tickers: Dict[str, Any]) -> PortfolioSnapshot:
        wallet = self.get_wallet()
        positions = self.realized_positions(wallet)
        equity = self.mark_to_market_equity(wallet, tickers)
        current_notional = {pair: quantity * self.pair_price(pair, tickers) for pair, quantity in positions.items()}
        return PortfolioSnapshot(
            wallet=wallet,
            positions=positions,
            equity=equity,
            usd_free=wallet.get("USD", {}).get("Free", 0.0),
            current_notional=current_notional,
        )

    def position_meta(self, pair: str) -> Optional[PositionMeta]:
        raw = self.state.positions_meta.get(pair)
        return PositionMeta(**raw) if raw else None

    def set_position_meta(self, meta: PositionMeta) -> None:
        self.state.positions_meta[meta.pair] = asdict(meta)

    def remove_position_meta(self, pair: str) -> None:
        self.state.positions_meta.pop(pair, None)

    def sync_position_meta(self, positions: Dict[str, float], tickers: Dict[str, Any]) -> None:
        live_pairs = set(positions)
        for pair in list(self.state.positions_meta):
            if pair not in live_pairs:
                self.remove_position_meta(pair)
        for pair, quantity in positions.items():
            meta = self.position_meta(pair)
            if meta is None:
                price = self.pair_price(pair, tickers)
                meta = PositionMeta(
                    pair=pair,
                    quantity=quantity,
                    entry_price=price,
                    highest_price=price,
                    last_trade_ts=now_ms(),
                    last_signal_score=0.0,
                    last_reason="recovered_from_balance",
                )
            else:
                meta.quantity = quantity
            self.set_position_meta(meta)

    def capture_portfolio_state(self, portfolio: PortfolioSnapshot, tickers: Dict[str, Any]) -> None:
        if self.state.peak_equity <= 0:
            self.state.peak_equity = portfolio.equity
        self.state.peak_equity = max(self.state.peak_equity, portfolio.equity)
        if self.state.peak_equity > 0:
            portfolio.drawdown = 1.0 - (portfolio.equity / self.state.peak_equity)
        self.log_portfolio(portfolio.wallet, portfolio.positions, portfolio.equity, portfolio.drawdown)
        self.sync_position_meta(portfolio.positions, tickers)
        logger.info(
            "Equity=%.2f Peak=%.2f Drawdown=%.2f%% Positions=%s",
            portfolio.equity,
            self.state.peak_equity,
            portfolio.drawdown * 100,
            portfolio.positions,
        )

    def in_cooldown(self, pair: str) -> bool:
        return now_ms() < self.state.cooldown_until.get(pair, 0)

    def set_cooldown(self, pair: str, minutes: int) -> None:
        self.state.cooldown_until[pair] = now_ms() + minutes * 60_000

    def rebalance_notional_threshold(self, equity: float) -> float:
        return max(self.cfg.min_rebalance_notional, equity * self.cfg.rebalance_threshold)

    def quantity_for_notional(self, pair: str, notional_usd: float, price: float) -> float:
        rules = self.trade_pairs[pair]
        amount_precision = int(rules.get("AmountPrecision", 6))
        min_order = safe_float(rules.get("MiniOrder", 1.0))
        if price <= 0:
            return 0.0
        quantity = round_down(notional_usd / price, amount_precision)
        if quantity <= 0 or quantity * price < min_order:
            return 0.0
        return quantity

    def log_trade(
        self,
        pair: str,
        side: str,
        quantity: float,
        reason: str,
        expected_price: float,
        response: Dict[str, Any],
        signal_score: float,
    ) -> None:
        detail = response.get("OrderDetail", {})
        append_csv(
            self.cfg.trade_log_csv,
            [
                "ts",
                "pair",
                "side",
                "quantity",
                "reason",
                "signal_score",
                "expected_price",
                "order_id",
                "status",
                "role",
                "filled_qty",
                "filled_avg_price",
                "commission_percent",
                "success",
                "errmsg",
            ],
            {
                "ts": now_ms(),
                "pair": pair,
                "side": side,
                "quantity": quantity,
                "reason": reason,
                "signal_score": signal_score,
                "expected_price": expected_price,
                "order_id": detail.get("OrderID", ""),
                "status": detail.get("Status", ""),
                "role": detail.get("Role", ""),
                "filled_qty": detail.get("FilledQuantity", detail.get("Quantity", "")),
                "filled_avg_price": detail.get("FilledAverPrice", detail.get("Price", "")),
                "commission_percent": detail.get("CommissionPercent", ""),
                "success": response.get("Success", False),
                "errmsg": response.get("ErrMsg", ""),
            },
        )

    def log_portfolio(self, wallet: Dict[str, Dict[str, float]], positions: Dict[str, float], equity: float, drawdown: float) -> None:
        append_csv(
            self.cfg.portfolio_log_csv,
            ["ts", "equity", "drawdown", "usd_free", "usd_lock", "positions_json"],
            {
                "ts": now_ms(),
                "equity": equity,
                "drawdown": drawdown,
                "usd_free": wallet.get("USD", {}).get("Free", 0.0),
                "usd_lock": wallet.get("USD", {}).get("Lock", 0.0),
                "positions_json": json.dumps(positions, sort_keys=True),
            },
        )

    def log_signal_snapshot(self, signals: Dict[str, Any]) -> None:
        top_signals = sorted(
            [{**{"pair": pair}, **feature} for pair, feature in signals["features"].items()],
            key=lambda item: item["score"],
            reverse=True,
        )[:10]
        append_csv(
            self.cfg.signal_log_csv,
            ["ts", "event", "detail"],
            {
                "ts": now_ms(),
                "event": "signal_snapshot",
                "detail": json.dumps(
                    {
                        "risk_on": signals["risk_on"],
                        "market": asdict(signals["snapshot"]),
                        "portfolio_risk": signals.get("portfolio_risk", {}),
                        "pre_risk_weights": signals.get("pre_risk_weights", {}),
                        "top": top_signals,
                        "targets": signals["weights"],
                    },
                    ensure_ascii=False,
                )[:3000],
            },
        )

    def submit_market_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        reason: str,
        signal_score: float,
        expected_price: float,
    ) -> bool:
        if quantity <= 0:
            return False
        if self.cfg.dry_run:
            fake_response = {
                "Success": True,
                "ErrMsg": "",
                "OrderDetail": {
                    "OrderID": f"dryrun-{now_ms()}",
                    "Status": "FILLED",
                    "Role": "TAKER",
                    "Quantity": quantity,
                    "FilledQuantity": quantity,
                    "FilledAverPrice": expected_price,
                    "CommissionPercent": 0.001,
                },
            }
            self.log_trade(pair, side, quantity, reason + "|dry_run", expected_price, fake_response, signal_score)
            logger.info("[DRY_RUN] %s %s qty=%.8f reason=%s", side, pair, quantity, reason)
            return True
        try:
            response = self.client.place_market_order(pair, side, quantity)
        except UnknownOrderStateError as exc:
            failure_response = {"Success": False, "ErrMsg": str(exc), "OrderDetail": {}}
            self.log_trade(pair, side, quantity, reason + "|unknown_state", expected_price, failure_response, signal_score)
            self._pause_until_ms = max(self._pause_until_ms, now_ms() + self.cfg.order_failure_pause_seconds * 1000)
            logger.error(
                "Order state unknown for %s %s qty=%.8f. Trading paused for %ss.",
                side,
                pair,
                quantity,
                self.cfg.order_failure_pause_seconds,
            )
            raise
        self.log_trade(pair, side, quantity, reason, expected_price, response, signal_score)
        if not response.get("Success", False):
            logger.warning("Order failed: %s", response)
            return False
        logger.info("Order OK: %s %s qty=%.8f reason=%s", side, pair, quantity, reason)
        return True

    def build_position_meta(self, pair: str, quantity: float, price: float, score: float) -> PositionMeta:
        meta = self.position_meta(pair)
        if meta is None:
            meta = PositionMeta(
                pair=pair,
                quantity=quantity,
                entry_price=price,
                highest_price=price,
                last_trade_ts=now_ms(),
                last_signal_score=score,
                last_reason="recovered_from_balance",
            )
        meta.quantity = quantity
        meta.highest_price = max(meta.highest_price, price)
        meta.last_signal_score = score
        return meta

    def exit_reasons(self, pair: str, meta: PositionMeta, price: float, score: float, targets: Dict[str, float]) -> List[str]:
        reasons: List[str] = []
        if price <= meta.entry_price * (1 - self.cfg.per_position_stop_loss):
            reasons.append("stop_loss")
        if price <= meta.highest_price * (1 - self.cfg.per_position_trailing_stop):
            reasons.append("trailing_stop")
        if score < self.cfg.exit_score_threshold:
            reasons.append("score_decay")
        if pair not in targets:
            reasons.append("not_in_targets")
        return reasons

    def manage_existing_positions(self, positions: Dict[str, float], tickers: Dict[str, Any], features: Dict[str, Dict[str, float]], targets: Dict[str, float]) -> None:
        for pair, quantity in list(positions.items()):
            price = self.pair_price(pair, tickers)
            if price <= 0:
                continue
            score = features.get(pair, {}).get("score", -999.0)
            meta = self.build_position_meta(pair, quantity, price, score)
            reasons = self.exit_reasons(pair, meta, price, score, targets)
            if reasons:
                if self.submit_market_order(pair, "SELL", quantity, "+".join(reasons), score, price):
                    self.remove_position_meta(pair)
                    self.set_cooldown(pair, self.cfg.cooldown_minutes)
                continue
            self.set_position_meta(meta)

    def trim_positions(
        self,
        portfolio: PortfolioSnapshot,
        tickers: Dict[str, Any],
        features: Dict[str, Dict[str, float]],
        targets: Dict[str, float],
        rebalance_threshold: float,
    ) -> None:
        for pair, quantity in list(portfolio.positions.items()):
            price = self.pair_price(pair, tickers)
            if price <= 0:
                continue
            target_usd = portfolio.equity * targets.get(pair, 0.0)
            current_usd = portfolio.current_notional.get(pair, 0.0)
            trim_usd = current_usd - target_usd
            if trim_usd < rebalance_threshold:
                continue
            sell_quantity = quantity if target_usd <= 0 else self.quantity_for_notional(pair, trim_usd, price)
            if sell_quantity <= 0:
                continue
            score = features.get(pair, {}).get("score", -999.0)
            reason = "target_exit_retry" if target_usd <= 0 else "target_trim"
            if self.submit_market_order(pair, "SELL", sell_quantity, reason, score, price):
                if sell_quantity >= quantity * 0.999999:
                    self.remove_position_meta(pair)
                    if target_usd <= 0:
                        self.set_cooldown(pair, self.cfg.cooldown_minutes)

    def record_buy_fill(self, pair: str, quantity: float, price: float, score: float) -> None:
        meta = self.position_meta(pair)
        if meta is None:
            self.set_position_meta(
                PositionMeta(
                    pair=pair,
                    quantity=quantity,
                    entry_price=price,
                    highest_price=price,
                    last_trade_ts=now_ms(),
                    last_signal_score=score,
                    last_reason="target_rebalance",
                )
            )
            return
        previous_quantity = meta.quantity
        new_quantity = previous_quantity + quantity
        if new_quantity > 0:
            meta.entry_price = ((meta.entry_price * previous_quantity) + (price * quantity)) / new_quantity
        meta.quantity = new_quantity
        meta.highest_price = max(meta.highest_price, price)
        meta.last_trade_ts = now_ms()
        meta.last_signal_score = score
        meta.last_reason = "target_rebalance"
        self.set_position_meta(meta)

    def add_target_positions(
        self,
        portfolio: PortfolioSnapshot,
        tickers: Dict[str, Any],
        features: Dict[str, Dict[str, float]],
        targets: Dict[str, float],
        rebalance_threshold: float,
    ) -> None:
        for pair, weight in sorted(targets.items(), key=lambda item: item[1], reverse=True):
            if self.in_cooldown(pair):
                continue
            price = self.pair_price(pair, tickers)
            if price <= 0:
                continue
            target_usd = portfolio.equity * weight
            current_usd = portfolio.current_notional.get(pair, 0.0)
            diff_usd = target_usd - current_usd
            if diff_usd < rebalance_threshold:
                continue
            usable_cash = max(0.0, portfolio.usd_free - portfolio.equity * self.cfg.cash_buffer)
            if usable_cash <= 0:
                continue
            buy_usd = min(diff_usd, usable_cash)
            quantity = self.quantity_for_notional(pair, buy_usd, price)
            if quantity <= 0:
                continue
            score = features[pair]["score"]
            if self.submit_market_order(pair, "BUY", quantity, "target_rebalance", score, price):
                self.record_buy_fill(pair, quantity, price, score)
                portfolio.usd_free -= quantity * price

    def exit_all_positions(self, positions: Dict[str, float], tickers: Dict[str, Any], reason: str) -> None:
        for pair, quantity in positions.items():
            if quantity <= 0:
                continue
            price = self.pair_price(pair, tickers)
            self.submit_market_order(pair, "SELL", quantity, reason, -999.0, price)
            self.remove_position_meta(pair)
            self.set_cooldown(pair, self.cfg.cooldown_minutes)

    def rebalance_once(self) -> None:
        now = time.time()

        # cooldown
        cooldown_seconds = 300  # 5分钟（推荐）

        if now - self.last_rebalance_ts < cooldown_seconds:
            logger.info("Cooldown active, skipping rebalance.")
            return
        tickers = self.fetch_all_tickers()
        self.update_history(tickers)

        portfolio = self.build_portfolio_snapshot(tickers)
        self.capture_portfolio_state(portfolio, tickers)
        if portfolio.drawdown >= self.cfg.max_portfolio_drawdown:
            logger.warning("Portfolio kill switch triggered.")
            self.exit_all_positions(portfolio.positions, tickers, "portfolio_drawdown")
            self.persist_runtime_state()
            self.last_rebalance_ts = now
            return

        signals = self.strategy.generate_signals(
            history=self.history,
            trade_pairs=self.trade_pairs,
            positions=portfolio.positions,
            prev_risk_on=self.state.risk_on,
        )
        self.state.risk_on = signals["risk_on"]
        self.log_signal_snapshot(signals)
        logger.info(
            "Regime=%s risk=%.2f targets=%s",
            signals["regime"]["regime"],
            signals["regime"]["risk_multiplier"],
            signals["weights"]
        )

        if not signals["features"]:
            logger.info("Not enough history yet.")
            self.persist_runtime_state()
            self.last_rebalance_ts = now
            return

        self.manage_existing_positions(portfolio.positions, tickers, signals["features"], signals["weights"])

        portfolio = self.build_portfolio_snapshot(tickers)
        self.sync_position_meta(portfolio.positions, tickers)
        rebalance_threshold = self.rebalance_notional_threshold(portfolio.equity)
        self.trim_positions(portfolio, tickers, signals["features"], signals["weights"], rebalance_threshold)

        portfolio = self.build_portfolio_snapshot(tickers)
        self.sync_position_meta(portfolio.positions, tickers)
        self.add_target_positions(portfolio, tickers, signals["features"], signals["weights"], rebalance_threshold)
        self.persist_runtime_state()
        self.last_rebalance_ts = now

    def run_forever(self) -> None:
        self.bootstrap()
        logger.info("Starting bot '%s'. DRY_RUN=%s", self.cfg.bot_name, self.cfg.dry_run)
        while not self._stop_requested:
            pause_remaining = max(0.0, (self._pause_until_ms - now_ms()) / 1000.0)
            if pause_remaining > 0:
                logger.info("Circuit breaker active. Sleeping %.2fs", pause_remaining)
                self.interruptible_sleep(pause_remaining)
                continue
            loop_started_at = time.time()
            try:
                self.rebalance_once()
                self.clear_loop_failures()
            except Exception as exc:
                logger.exception("Main loop error: %s", exc)
                self.register_loop_failure(exc)
            try:
                self.persist_runtime_state()
            except Exception as exc:
                logger.exception("State persistence error: %s", exc)
            sleep_for = max(5.0, self.cfg.poll_seconds - (time.time() - loop_started_at))
            logger.info("Sleeping %.2fs", sleep_for)
            self.interruptible_sleep(sleep_for)
        try:
            self.persist_runtime_state()
        except Exception as exc:
            logger.exception("Final state persistence error: %s", exc)
        self.release_instance_lock()
        self.client.close()
        logger.info("Bot stopped cleanly.")


def main() -> None:
    if not CFG.dry_run and (not CFG.api_key or not CFG.api_secret):
        raise SystemExit("Missing ROOSTOO_API_KEY / ROOSTOO_API_SECRET.")
    client = RoostooClient(CFG)
    try:
        bot = RoostooMomentumBot(CFG, client)
        bot.run_forever()
    except InstanceLockError as exc:
        client.close()
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
