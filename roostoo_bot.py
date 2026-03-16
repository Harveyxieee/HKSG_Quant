#!/usr/bin/env python3
from __future__ import annotations

import atexit
import csv
import hashlib
import hmac
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
from typing import Any, Deque, Dict, List, Optional, Tuple

import requests

FeatureMap = Dict[str, Dict[str, float]]


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
            api_key=os.getenv("ROOSTOO_API_KEY", ""),
            api_secret=os.getenv("ROOSTOO_API_SECRET", ""),
            bot_name=os.getenv("BOT_NAME", "roostoo_prelim_bot"),
            poll_seconds=int(os.getenv("POLL_SECONDS", "60")),
            lookback_minutes=int(os.getenv("LOOKBACK_MINUTES", "360")),
            min_history=int(os.getenv("MIN_HISTORY", "75")),
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
            market_median_60m_threshold=float(os.getenv("MARKET_MEDIAN_60M_THRESHOLD", "0.0015")),
            market_up_ratio_threshold=float(os.getenv("MARKET_UP_RATIO_THRESHOLD", "0.55")),
            regime_exit_median_60m_threshold=float(os.getenv("REGIME_EXIT_MEDIAN_60M_THRESHOLD", "0.0005")),
            regime_exit_up_ratio_threshold=float(os.getenv("REGIME_EXIT_UP_RATIO_THRESHOLD", "0.48")),
            market_positive_score_ratio_threshold=float(os.getenv("MARKET_POSITIVE_SCORE_RATIO_THRESHOLD", "0.50")),
            regime_exit_positive_score_ratio_threshold=float(os.getenv("REGIME_EXIT_POSITIVE_SCORE_RATIO_THRESHOLD", "0.42")),
            vol_floor=float(os.getenv("VOL_FLOOR", "0.004")),
            vol_cap=float(os.getenv("VOL_CAP", "0.08")),
            holding_score_bonus=float(os.getenv("HOLDING_SCORE_BONUS", "0.08")),
            per_position_stop_loss=float(os.getenv("PER_POSITION_STOP_LOSS", "0.035")),
            per_position_trailing_stop=float(os.getenv("PER_POSITION_TRAILING_STOP", "0.045")),
            max_portfolio_drawdown=float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "0.10")),
            cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "15")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "20")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_sleep_seconds=float(os.getenv("RETRY_SLEEP_SECONDS", "1.5")),
            order_failure_pause_seconds=int(os.getenv("ORDER_FAILURE_PAUSE_SECONDS", "180")),
            loop_error_backoff_cap_seconds=int(os.getenv("LOOP_ERROR_BACKOFF_CAP_SECONDS", "900")),
            cancel_all_on_start=env_bool("CANCEL_ALL_ON_START", False),
            dry_run=env_bool("DRY_RUN", False),
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


def round_down(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def compute_return(prices: List[float], lookback: int) -> float:
    if len(prices) <= lookback:
        return 0.0
    base_price = prices[-lookback - 1]
    last_price = prices[-1]
    return last_price / base_price - 1.0 if base_price > 0 else 0.0


def append_csv(path: Path, headers: List[str], row: Dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def sha256_json(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


class UnknownOrderStateError(RuntimeError):
    pass


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
class MarketSnapshot:
    median_ret15: float = 0.0
    median_ret60: float = 0.0
    up_ratio_15: float = 0.0
    positive_score_ratio: float = 0.0
    avg_score: float = 0.0


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


class RoostooClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.time_offset_ms = 0
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": f"{self.cfg.bot_name}/1.0"})
        return session

    def reset_session(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
        self.session = self._build_session()

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def timestamp_ms(self) -> int:
        return now_ms() + self.time_offset_ms

    def sync_time(self) -> int:
        response = self.server_time()
        server_time = int(safe_float(response.get("ServerTime"), now_ms()))
        self.time_offset_ms = server_time - now_ms()
        return server_time

    def _sign(self, params: Dict[str, Any]) -> str:
        items = sorted((key, str(value)) for key, value in params.items())
        payload = "&".join(f"{key}={value}" for key, value in items)
        return hmac.new(self.cfg.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _log_request(self, method: str, path: str, params: Dict[str, Any], ok: bool, response_json: Any) -> None:
        append_csv(
            CFG.request_log_csv,
            ["ts", "method", "path", "params_sha256", "ok", "response_snippet"],
            {
                "ts": now_ms(),
                "method": method,
                "path": path,
                "params_sha256": sha256_json(params),
                "ok": ok,
                "response_snippet": json.dumps(response_json, ensure_ascii=False)[:800],
            },
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        is_post: bool = False,
        retry_safe: bool = True,
        ambiguous_error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = dict(params or {})
        headers: Dict[str, str] = {}
        if signed or path == "/v3/ticker":
            params["timestamp"] = self.timestamp_ms()
        if signed:
            if not self.cfg.api_key or not self.cfg.api_secret:
                raise RuntimeError("Signed endpoint requested but ROOSTOO_API_KEY / ROOSTOO_API_SECRET missing")
            headers["RST-API-KEY"] = self.cfg.api_key
            headers["MSG-SIGNATURE"] = self._sign(params)
        if is_post:
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        url = f"{self.cfg.base_url}{path}"
        last_exc: Optional[Exception] = None
        max_attempts = self.cfg.max_retries if retry_safe else 1
        for attempt in range(1, max_attempts + 1):
            resp = None
            try:
                if is_post:
                    resp = self.session.post(url, data=params, headers=headers, timeout=self.cfg.request_timeout)
                else:
                    resp = self.session.get(url, params=params, headers=headers, timeout=self.cfg.request_timeout)
                resp.raise_for_status()
                data = resp.json()
                self._log_request(method, path, params, True, data)
                return data
            except Exception as exc:
                last_exc = exc
                try:
                    snippet = resp.text[:800] if resp is not None else str(exc)
                except Exception:
                    snippet = str(exc)
                self._log_request(method, path, params, False, {"error": snippet})
                logger.warning("Request failed (%s %s) attempt %s/%s: %s", method, path, attempt, max_attempts, exc)
                if "timestamp" in snippet.lower():
                    try:
                        self.sync_time()
                    except Exception:
                        pass
                if retry_safe and attempt < max_attempts:
                    self.reset_session()
                    time.sleep(self.cfg.retry_sleep_seconds * (2 ** (attempt - 1)))
        if ambiguous_error_message is not None:
            raise UnknownOrderStateError(f"{ambiguous_error_message}: {last_exc}")
        raise RuntimeError(f"Request failed after retries: {method} {path} -> {last_exc}")

    def server_time(self) -> Dict[str, Any]:
        return self._request("GET", "/v3/serverTime")

    def exchange_info(self) -> Dict[str, Any]:
        return self._request("GET", "/v3/exchangeInfo")

    def ticker(self, pair: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if pair:
            params["pair"] = pair
        return self._request("GET", "/v3/ticker", params=params)

    def balance(self) -> Dict[str, Any]:
        return self._request("GET", "/v3/balance", signed=True)

    def pending_count(self) -> Dict[str, Any]:
        return self._request("GET", "/v3/pending_count", signed=True)

    def place_market_order(self, pair: str, side: str, quantity: float) -> Dict[str, Any]:
        payload = {"pair": pair, "side": side, "type": "MARKET", "quantity": quantity}
        return self._request(
            "POST",
            "/v3/place_order",
            params=payload,
            signed=True,
            is_post=True,
            retry_safe=False,
            ambiguous_error_message=f"Order submission status unknown for {side} {pair}",
        )

    def cancel_order(self, pair: Optional[str] = None, order_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if pair:
            payload["pair"] = pair
        if order_id:
            payload["order_id"] = order_id
        return self._request("POST", "/v3/cancel_order", params=payload, signed=True, is_post=True)


class RoostooMomentumBot:
    def __init__(self, cfg: Config, client: RoostooClient):
        self.cfg = cfg
        self.client = client
        self.state = load_state()
        self.exchange_info = self.client.exchange_info()
        self.trade_pairs: Dict[str, Dict[str, Any]] = self.exchange_info.get("TradePairs", {})
        self.history = load_history()
        self._stop_requested = False
        self._pause_until_ms = 0
        self._consecutive_loop_failures = 0
        self._has_lock = False
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
        for coin, value in response.get("Wallet", {}).items():
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

    def compute_features(self) -> FeatureMap:
        features: FeatureMap = {}
        eligible_pairs: List[str] = []
        for pair, series in self.history.items():
            if pair not in self.trade_pairs or len(series) < self.cfg.min_history:
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

    def risk_on(self, snapshot: MarketSnapshot) -> bool:
        if self.state.risk_on:
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

    def target_weights(self, features: FeatureMap, risk_on: bool, positions: Dict[str, float]) -> Dict[str, float]:
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

    def position_meta(self, pair: str) -> Optional[PositionMeta]:
        raw = self.state.positions_meta.get(pair)
        return PositionMeta(**raw) if raw else None

    def set_position_meta(self, meta: PositionMeta) -> None:
        self.state.positions_meta[meta.pair] = asdict(meta)

    def remove_position_meta(self, pair: str) -> None:
        self.state.positions_meta.pop(pair, None)

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

    def log_signal_snapshot(self, risk_on: bool, snapshot: MarketSnapshot, features: FeatureMap, targets: Dict[str, float]) -> None:
        top_signals = sorted(
            [{**{"pair": pair}, **feature} for pair, feature in features.items()],
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
                    {"risk_on": risk_on, "market": asdict(snapshot), "top": top_signals, "targets": targets},
                    ensure_ascii=False,
                )[:3000],
            },
        )

    def log_regime(self, risk_on: bool, snapshot: MarketSnapshot) -> None:
        logger.info(
            "Regime risk_on=%s median15=%.4f median60=%.4f breadth=%.2f positive_scores=%.2f avg_score=%.3f",
            risk_on,
            snapshot.median_ret15,
            snapshot.median_ret60,
            snapshot.up_ratio_15,
            snapshot.positive_score_ratio,
            snapshot.avg_score,
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

    def manage_existing_positions(self, positions: Dict[str, float], tickers: Dict[str, Any], features: FeatureMap, targets: Dict[str, float]) -> None:
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
        features: FeatureMap,
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
        features: FeatureMap,
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
        tickers = self.fetch_all_tickers()
        self.update_history(tickers)

        portfolio = self.build_portfolio_snapshot(tickers)
        self.capture_portfolio_state(portfolio, tickers)
        if portfolio.drawdown >= self.cfg.max_portfolio_drawdown:
            logger.warning("Portfolio kill switch triggered.")
            self.exit_all_positions(portfolio.positions, tickers, "portfolio_drawdown")
            self.persist_runtime_state()
            return

        features = self.compute_features()
        if not features:
            logger.info("Not enough history yet.")
            self.persist_runtime_state()
            return

        snapshot = self.market_snapshot(features)
        market_is_risk_on = self.risk_on(snapshot)
        self.state.risk_on = market_is_risk_on
        targets = self.target_weights(features, market_is_risk_on, portfolio.positions)
        self.log_regime(market_is_risk_on, snapshot)
        self.log_signal_snapshot(market_is_risk_on, snapshot, features, targets)

        self.manage_existing_positions(portfolio.positions, tickers, features, targets)

        portfolio = self.build_portfolio_snapshot(tickers)
        self.sync_position_meta(portfolio.positions, tickers)
        rebalance_threshold = self.rebalance_notional_threshold(portfolio.equity)
        self.trim_positions(portfolio, tickers, features, targets, rebalance_threshold)

        portfolio = self.build_portfolio_snapshot(tickers)
        self.sync_position_meta(portfolio.positions, tickers)
        self.add_target_positions(portfolio, tickers, features, targets, rebalance_threshold)
        self.persist_runtime_state()

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
