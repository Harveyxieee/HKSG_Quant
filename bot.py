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
    rebalance_minutes: int
    risk_check_minutes: int
    target_gross_exposure: float
    max_single_weight: float
    min_effective_weight: float
    cash_buffer: float
    entry_score_threshold: float
    exit_score_threshold: float
    entry_confidence_threshold: float
    hold_confidence_threshold: float
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
    risk_data_frequency: str
    risk_min_periods: int
    risk_return_method: str
    risk_cov_window: int
    risk_on_portfolio_vol_threshold: float
    risk_off_portfolio_vol_threshold: float
    risk_on_correlation_threshold: float
    risk_off_correlation_threshold: float
    risk_vol_score_weight: float
    risk_corr_score_weight: float
    risk_on_exposure_multiplier: float
    neutral_exposure_multiplier: float
    risk_off_exposure_multiplier: float
    risk_on_exposure_floor: float
    neutral_exposure_floor: float
    enable_volatility_targeting: bool
    target_portfolio_volatility: float
    min_vol_target_scale: float
    max_vol_target_scale: float
    diversification_breakdown_corr_threshold: float
    diversification_breakdown_exposure_multiplier: float
    risk_on_weight_penalty_scale: float
    neutral_weight_penalty_scale: float
    risk_off_weight_penalty_scale: float
    risk_weight_vol_penalty: float
    risk_weight_corr_penalty: float
    diversification_alt_weight_multiplier: float
    diversification_core_asset_multiplier: float
    diversification_core_assets: str
    high_vol_feature_cutoff_multiplier: float
    high_vol_feature_weight_multiplier: float
    per_position_stop_loss: float
    per_position_trailing_stop: float
    max_portfolio_drawdown: float
    cooldown_minutes: int
    min_hold_minutes: int
    portfolio_drawdown_cooldown_minutes: int
    portfolio_drawdown_cooldown_floor_minutes: int
    portfolio_drawdown_sell_floor: float
    portfolio_drawdown_sell_cap: float
    profit_protect_threshold: float
    profit_protect_score_decay_fraction: float
    profit_protect_not_in_targets_fraction: float
    startup_warmup_minutes: int
    max_data_delay_minutes: int
    min_fresh_points_after_start: int
    min_fresh_span_minutes: int
    request_timeout: int
    max_retries: int
    retry_sleep_seconds: float
    order_failure_pause_seconds: int
    loop_error_backoff_cap_seconds: int
    cancel_all_on_start: bool
    dry_run: bool
    max_turnover_per_rebalance: float
    range_entry_exposure: float
    range_keep_exposure: float
    range_max_positions: int
    range_turnover_multiplier: float
    enable_range_entries: bool
    range_entry_confidence_floor: float
    range_mean_reversion_floor: float
    range_trend_stability_floor: float
    range_defensive_score_threshold: float
    enable_recovery_reentry: bool
    recovery_entry_score_relaxation: float
    recovery_entry_confidence_relaxation: float
    recovery_reentry_exposure: float
    recovery_max_positions: int
    empty_book_reentry_minutes: int
    recovery_rebound_confirmation: float
    recovery_pullback_threshold: float
    recovery_volume_confirmation_floor: float
    recovery_trend_stability_floor: float
    recovery_min_hold_multiplier: float
    recovery_exit_score_grace: float
    recovery_not_in_targets_confirm_bars: int
    recovery_daily_entry_limit: int
    recovery_entry_cooldown_minutes: int
    block_same_day_soft_exit: bool
    portfolio_min_positions: int
    portfolio_max_positions: int
    liquid_asset_volume_threshold: float
    satellite_asset_volume_threshold: float
    core_bucket_weight_cap: float
    liquid_bucket_weight_cap: float
    satellite_bucket_weight_cap: float
    trend_state_score_boost: float
    breakout_state_score_boost: float
    rebound_state_score_boost: float
    failed_rebound_score_penalty: float
    range_chop_score_penalty: float
    daily_activity_enabled: bool
    daily_activity_probe_exposure: float
    daily_activity_min_confidence: float
    daily_activity_entry_cooldown_minutes: int
    daily_soft_trade_limit: int
    daily_new_entry_limit: int
    flash_crash_lookback_minutes: int
    flash_crash_drop_threshold: float
    flash_crash_rebound_threshold: float
    flash_crash_sell_fraction: float
    shock_rebound_probe_boost: float

    @classmethod
    def from_env(cls) -> "Config":
        data_dir = Path(os.getenv("BOT_DATA_DIR", "./data"))
        log_dir = data_dir / "logs"
        return cls(
            base_url=os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com").rstrip("/"),
            api_key=os.getenv("ROOSTOO_API_KEY", "zx1oEdatzBZf1jklzkJ5F6zDf2RiNKvX3ONQchbPeYj6mxo1kSjUadOb9pzrKElw"),
            api_secret=os.getenv("ROOSTOO_API_SECRET", "kgRYJVTWrhpsD41vtE3z1aKNfOSaRayw99nSvQkLNjHMfV6O6z1wTDouPWZvmRAh"),
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
            top_n=int(os.getenv("TOP_N", "4")),
            rebalance_minutes=int(os.getenv("REBALANCE_MINUTES", "30")),
            risk_check_minutes=int(os.getenv("RISK_CHECK_MINUTES", "15")),
            target_gross_exposure=float(os.getenv("TARGET_GROSS_EXPOSURE", "0.72")),
            max_single_weight=float(os.getenv("MAX_SINGLE_WEIGHT", "0.28")),
            min_effective_weight=float(os.getenv("MIN_EFFECTIVE_WEIGHT", "0.07")),
            cash_buffer=float(os.getenv("CASH_BUFFER", "0.25")),
            entry_score_threshold=float(os.getenv("ENTRY_SCORE_THRESHOLD", "0.68")),
            exit_score_threshold=float(os.getenv("EXIT_SCORE_THRESHOLD", "0.12")),
            entry_confidence_threshold=float(os.getenv("ENTRY_CONFIDENCE_THRESHOLD", "0.55")),
            hold_confidence_threshold=float(os.getenv("HOLD_CONFIDENCE_THRESHOLD", "0.30")),
            rebalance_threshold=float(os.getenv("REBALANCE_THRESHOLD", "0.03")),
            min_rebalance_notional=float(os.getenv("MIN_REBALANCE_NOTIONAL", "25")),
            spread_threshold=float(os.getenv("SPREAD_THRESHOLD", "0.006")),
            min_24h_dollar_vol=float(os.getenv("MIN_24H_DOLLAR_VOL", "120000")),
            max_pump_distance=float(os.getenv("MAX_PUMP_DISTANCE", "0.05")),
            market_median_60m_threshold=float(os.getenv("MARKET_MEDIAN_60M_THRESHOLD", "0.0005")),
            market_up_ratio_threshold=float(os.getenv("MARKET_UP_RATIO_THRESHOLD", "0.52")),
            regime_exit_median_60m_threshold=float(os.getenv("REGIME_EXIT_MEDIAN_60M_THRESHOLD", "0.0005")),
            regime_exit_up_ratio_threshold=float(os.getenv("REGIME_EXIT_UP_RATIO_THRESHOLD", "0.48")),
            market_positive_score_ratio_threshold=float(os.getenv("MARKET_POSITIVE_SCORE_RATIO_THRESHOLD", "0.45")),
            regime_exit_positive_score_ratio_threshold=float(os.getenv("REGIME_EXIT_POSITIVE_SCORE_RATIO_THRESHOLD", "0.42")),
            vol_floor=float(os.getenv("VOL_FLOOR", "0.004")),
            vol_cap=float(os.getenv("VOL_CAP", "0.08")),
            holding_score_bonus=float(os.getenv("HOLDING_SCORE_BONUS", "0.12")),
            per_position_stop_loss=float(os.getenv("PER_POSITION_STOP_LOSS", "0.06")),
            risk_data_frequency=os.getenv("RISK_DATA_FREQUENCY", "raw").strip().lower(),
            risk_min_periods=int(os.getenv("RISK_MIN_PERIODS", "60")),
            risk_return_method=os.getenv("RISK_RETURN_METHOD", "log").strip().lower(),
            risk_cov_window=int(os.getenv("RISK_COV_WINDOW", "60")),
            risk_on_portfolio_vol_threshold=float(os.getenv("RISK_ON_PORTFOLIO_VOL_THRESHOLD", "0.015")),
            risk_off_portfolio_vol_threshold=float(os.getenv("RISK_OFF_PORTFOLIO_VOL_THRESHOLD", "0.035")),
            risk_on_correlation_threshold=float(os.getenv("RISK_ON_CORRELATION_THRESHOLD", "0.35")),
            risk_off_correlation_threshold=float(os.getenv("RISK_OFF_CORRELATION_THRESHOLD", "0.65")),
            risk_vol_score_weight=float(os.getenv("RISK_VOL_SCORE_WEIGHT", "0.5")),
            risk_corr_score_weight=float(os.getenv("RISK_CORR_SCORE_WEIGHT", "0.5")),
            risk_on_exposure_multiplier=float(os.getenv("RISK_ON_EXPOSURE_MULTIPLIER", "1.0")),
            neutral_exposure_multiplier=float(os.getenv("NEUTRAL_EXPOSURE_MULTIPLIER", "0.7")),
            risk_off_exposure_multiplier=float(os.getenv("RISK_OFF_EXPOSURE_MULTIPLIER", "0.35")),
            risk_on_exposure_floor=float(os.getenv("RISK_ON_EXPOSURE_FLOOR", "0.80")),
            neutral_exposure_floor=float(os.getenv("NEUTRAL_EXPOSURE_FLOOR", "0.45")),
            enable_volatility_targeting=env_bool("ENABLE_VOLATILITY_TARGETING", True),
            target_portfolio_volatility=float(os.getenv("TARGET_PORTFOLIO_VOLATILITY", "0.020")),
            min_vol_target_scale=float(os.getenv("MIN_VOL_TARGET_SCALE", "0.35")),
            max_vol_target_scale=float(os.getenv("MAX_VOL_TARGET_SCALE", "1.20")),
            diversification_breakdown_corr_threshold=float(
                os.getenv("DIVERSIFICATION_BREAKDOWN_CORR_THRESHOLD", "0.75")),
            diversification_breakdown_exposure_multiplier=float(
                os.getenv("DIVERSIFICATION_BREAKDOWN_EXPOSURE_MULTIPLIER", "0.75")),
            risk_on_weight_penalty_scale=float(os.getenv("RISK_ON_WEIGHT_PENALTY_SCALE", "0.10")),
            neutral_weight_penalty_scale=float(os.getenv("NEUTRAL_WEIGHT_PENALTY_SCALE", "0.25")),
            risk_off_weight_penalty_scale=float(os.getenv("RISK_OFF_WEIGHT_PENALTY_SCALE", "0.50")),
            risk_weight_vol_penalty=float(os.getenv("RISK_WEIGHT_VOL_PENALTY", "1.0")),
            risk_weight_corr_penalty=float(os.getenv("RISK_WEIGHT_CORR_PENALTY", "1.0")),
            diversification_alt_weight_multiplier=float(
                os.getenv("DIVERSIFICATION_ALT_WEIGHT_MULTIPLIER", "0.85")
            ),
            diversification_core_asset_multiplier=float(
                os.getenv("DIVERSIFICATION_CORE_ASSET_MULTIPLIER", "1.15")
            ),
            diversification_core_assets=os.getenv("DIVERSIFICATION_CORE_ASSETS", "BTC,ETH"),
            high_vol_feature_cutoff_multiplier=float(
                os.getenv("HIGH_VOL_FEATURE_CUTOFF_MULTIPLIER", "0.85")
            ),
            high_vol_feature_weight_multiplier=float(
                os.getenv("HIGH_VOL_FEATURE_WEIGHT_MULTIPLIER", "0.80")
            ),
            per_position_trailing_stop=float(os.getenv("PER_POSITION_TRAILING_STOP", "0.06")),
            max_portfolio_drawdown=float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "0.10")),
            cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "15")),
            min_hold_minutes=int(os.getenv("MIN_HOLD_MINUTES", "240")),
            portfolio_drawdown_cooldown_minutes=int(os.getenv("PORTFOLIO_DRAWDOWN_COOLDOWN_MINUTES", "1440")),
            portfolio_drawdown_cooldown_floor_minutes=int(os.getenv("PORTFOLIO_DRAWDOWN_COOLDOWN_FLOOR_MINUTES", "1440")),
            portfolio_drawdown_sell_floor=float(os.getenv("PORTFOLIO_DRAWDOWN_SELL_FLOOR", "0.50")),
            portfolio_drawdown_sell_cap=float(os.getenv("PORTFOLIO_DRAWDOWN_SELL_CAP", "0.50")),
            profit_protect_threshold=float(os.getenv("PROFIT_PROTECT_THRESHOLD", "0.10")),
            profit_protect_score_decay_fraction=float(os.getenv("PROFIT_PROTECT_SCORE_DECAY_FRACTION", "0.18")),
            profit_protect_not_in_targets_fraction=float(os.getenv("PROFIT_PROTECT_NOT_IN_TARGETS_FRACTION", "0.25")),
            startup_warmup_minutes=int(os.getenv("STARTUP_WARMUP_MINUTES", "30")),
            max_data_delay_minutes=int(os.getenv("MAX_DATA_DELAY_MINUTES", "2")),
            min_fresh_points_after_start=int(os.getenv("MIN_FRESH_POINTS_AFTER_START", "30")),
            min_fresh_span_minutes=int(os.getenv("MIN_FRESH_SPAN_MINUTES", "30")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "5")),
            max_retries=int(os.getenv("MAX_RETRIES", "1")),
            retry_sleep_seconds=float(os.getenv("RETRY_SLEEP_SECONDS", "1.5")),
            order_failure_pause_seconds=int(os.getenv("ORDER_FAILURE_PAUSE_SECONDS", "180")),
            loop_error_backoff_cap_seconds=int(os.getenv("LOOP_ERROR_BACKOFF_CAP_SECONDS", "900")),
            cancel_all_on_start=env_bool("CANCEL_ALL_ON_START", False),
            dry_run=env_bool("DRY_RUN", False),
            max_turnover_per_rebalance=float(os.getenv("MAX_TURNOVER_PER_REBALANCE", "0.60")),
            range_entry_exposure=float(os.getenv("RANGE_ENTRY_EXPOSURE", "0.12")),
            range_keep_exposure=float(os.getenv("RANGE_KEEP_EXPOSURE", "0.10")),
            range_max_positions=int(os.getenv("RANGE_MAX_POSITIONS", "2")),
            range_turnover_multiplier=float(os.getenv("RANGE_TURNOVER_MULTIPLIER", "0.35")),
            enable_range_entries=env_bool("ENABLE_RANGE_ENTRIES", False),
            range_entry_confidence_floor=float(os.getenv("RANGE_ENTRY_CONFIDENCE_FLOOR", "0.62")),
            range_mean_reversion_floor=float(os.getenv("RANGE_MEAN_REVERSION_FLOOR", "0.00")),
            range_trend_stability_floor=float(os.getenv("RANGE_TREND_STABILITY_FLOOR", "-0.05")),
            range_defensive_score_threshold=float(os.getenv("RANGE_DEFENSIVE_SCORE_THRESHOLD", "0.08")),
            enable_recovery_reentry=env_bool("ENABLE_RECOVERY_REENTRY", True),
            recovery_entry_score_relaxation=float(os.getenv("RECOVERY_ENTRY_SCORE_RELAXATION", "0.10")),
            recovery_entry_confidence_relaxation=float(os.getenv("RECOVERY_ENTRY_CONFIDENCE_RELAXATION", "0.06")),
            recovery_reentry_exposure=float(os.getenv("RECOVERY_REENTRY_EXPOSURE", "0.08")),
            recovery_max_positions=int(os.getenv("RECOVERY_MAX_POSITIONS", "2")),
            empty_book_reentry_minutes=int(os.getenv("EMPTY_BOOK_REENTRY_MINUTES", "120")),
            recovery_rebound_confirmation=float(os.getenv("RECOVERY_REBOUND_CONFIRMATION", "0.008")),
            recovery_pullback_threshold=float(os.getenv("RECOVERY_PULLBACK_THRESHOLD", "-0.025")),
            recovery_volume_confirmation_floor=float(os.getenv("RECOVERY_VOLUME_CONFIRMATION_FLOOR", "0.0")),
            recovery_trend_stability_floor=float(os.getenv("RECOVERY_TREND_STABILITY_FLOOR", "-0.10")),
            recovery_min_hold_multiplier=float(os.getenv("RECOVERY_MIN_HOLD_MULTIPLIER", "2.0")),
            recovery_exit_score_grace=float(os.getenv("RECOVERY_EXIT_SCORE_GRACE", "0.08")),
            recovery_not_in_targets_confirm_bars=int(os.getenv("RECOVERY_NOT_IN_TARGETS_CONFIRM_BARS", "6")),
            recovery_daily_entry_limit=int(os.getenv("RECOVERY_DAILY_ENTRY_LIMIT", "1")),
            recovery_entry_cooldown_minutes=int(os.getenv("RECOVERY_ENTRY_COOLDOWN_MINUTES", "720")),
            block_same_day_soft_exit=env_bool("BLOCK_SAME_DAY_SOFT_EXIT", False),
            portfolio_min_positions=int(os.getenv("PORTFOLIO_MIN_POSITIONS", "4")),
            portfolio_max_positions=int(os.getenv("PORTFOLIO_MAX_POSITIONS", "5")),
            liquid_asset_volume_threshold=float(os.getenv("LIQUID_ASSET_VOLUME_THRESHOLD", "400000")),
            satellite_asset_volume_threshold=float(os.getenv("SATELLITE_ASSET_VOLUME_THRESHOLD", "180000")),
            core_bucket_weight_cap=float(os.getenv("CORE_BUCKET_WEIGHT_CAP", "0.58")),
            liquid_bucket_weight_cap=float(os.getenv("LIQUID_BUCKET_WEIGHT_CAP", "0.32")),
            satellite_bucket_weight_cap=float(os.getenv("SATELLITE_BUCKET_WEIGHT_CAP", "0.10")),
            trend_state_score_boost=float(os.getenv("TREND_STATE_SCORE_BOOST", "0.08")),
            breakout_state_score_boost=float(os.getenv("BREAKOUT_STATE_SCORE_BOOST", "0.05")),
            rebound_state_score_boost=float(os.getenv("REBOUND_STATE_SCORE_BOOST", "0.02")),
            failed_rebound_score_penalty=float(os.getenv("FAILED_REBOUND_SCORE_PENALTY", "0.24")),
            range_chop_score_penalty=float(os.getenv("RANGE_CHOP_SCORE_PENALTY", "0.16")),
            daily_activity_enabled=env_bool("DAILY_ACTIVITY_ENABLED", True),
            daily_activity_probe_exposure=float(os.getenv("DAILY_ACTIVITY_PROBE_EXPOSURE", "0.03")),
            daily_activity_min_confidence=float(os.getenv("DAILY_ACTIVITY_MIN_CONFIDENCE", "0.55")),
            daily_activity_entry_cooldown_minutes=int(os.getenv("DAILY_ACTIVITY_ENTRY_COOLDOWN_MINUTES", "360")),
            daily_soft_trade_limit=int(os.getenv("DAILY_SOFT_TRADE_LIMIT", "8")),
            daily_new_entry_limit=int(os.getenv("DAILY_NEW_ENTRY_LIMIT", "3")),
            flash_crash_lookback_minutes=int(os.getenv("FLASH_CRASH_LOOKBACK_MINUTES", "5")),
            flash_crash_drop_threshold=float(os.getenv("FLASH_CRASH_DROP_THRESHOLD", "-0.03")),
            flash_crash_rebound_threshold=float(os.getenv("FLASH_CRASH_REBOUND_THRESHOLD", "0.008")),
            flash_crash_sell_fraction=float(os.getenv("FLASH_CRASH_SELL_FRACTION", "0.50")),
            shock_rebound_probe_boost=float(os.getenv("SHOCK_REBOUND_PROBE_BOOST", "0.08")),
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


def portfolio_drawdown_response(cfg: Config, drawdown: float) -> tuple[float, int]:
    max_dd = max(cfg.max_portfolio_drawdown, 1e-9)
    severity = clamp((drawdown - max_dd) / max_dd, 0.0, 1.0)
    sell_fraction = clamp(
        cfg.portfolio_drawdown_sell_floor
        + (cfg.portfolio_drawdown_sell_cap - cfg.portfolio_drawdown_sell_floor) * severity,
        cfg.portfolio_drawdown_sell_floor,
        cfg.portfolio_drawdown_sell_cap,
    )
    cooldown_minutes = int(round(
        cfg.portfolio_drawdown_cooldown_floor_minutes
        + (cfg.portfolio_drawdown_cooldown_minutes - cfg.portfolio_drawdown_cooldown_floor_minutes) * severity
    ))
    return sell_fraction, cooldown_minutes


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
    not_in_targets_bars: int = 0
    recovery_trade_day: str = ""
    entry_day: str = ""


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
    portfolio_reentry_allowed_at: int = 0
    recovery_entries_by_day: Dict[str, int] = field(default_factory=dict)
    trades_by_day: Dict[str, int] = field(default_factory=dict)
    soft_trades_by_day: Dict[str, int] = field(default_factory=dict)
    buy_trades_by_day: Dict[str, int] = field(default_factory=dict)

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
            portfolio_reentry_allowed_at=int(raw.get("portfolio_reentry_allowed_at", 0)),
            recovery_entries_by_day={key: int(value) for key, value in raw.get("recovery_entries_by_day", {}).items()},
            trades_by_day={key: int(value) for key, value in raw.get("trades_by_day", {}).items()},
            soft_trades_by_day={key: int(value) for key, value in raw.get("soft_trades_by_day", {}).items()},
            buy_trades_by_day={key: int(value) for key, value in raw.get("buy_trades_by_day", {}).items()},
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
                        "quote_volume": safe_float(row.get("quote_volume", row.get("unit_trade_value"))),
                        "base_volume": safe_float(row.get("base_volume")),
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
        self.last_risk_check_ts = 0
        self.session_start_ts = now_ms()
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
                    "quote_volume": safe_float(ticker.get("UnitTradeValue")),
                    "base_volume": safe_float(ticker.get("CoinTradeValue")),
                }
            )

    def freshness_reference_pair(self) -> Optional[str]:
        if "BTC/USD" in self.trade_pairs:
            return "BTC/USD"
        if self.trade_pairs:
            return sorted(self.trade_pairs.keys())[0]
        return None

    def latest_history_ts(self, pair: Optional[str] = None) -> int:
        if pair:
            series = self.history.get(pair)
            if series:
                return int(series[-1].get("ts", 0))
            return 0

        latest_ts = 0
        for series in self.history.values():
            if not series:
                continue
            latest_ts = max(latest_ts, int(series[-1].get("ts", 0)))
        return latest_ts

    def fresh_points_after_start(self, pair: str) -> List[Dict[str, float]]:
        series = self.history.get(pair, deque())
        return [row for row in series if int(row.get("ts", 0)) >= self.session_start_ts]

    def pair_freshness_status(self, pair: str) -> tuple[bool, str]:
        latest_ts = self.latest_history_ts(pair)
        if latest_ts <= 0:
            return False, f"{pair}:no_history"

        delay_ms = now_ms() - latest_ts
        if delay_ms > self.cfg.max_data_delay_minutes * 60_000:
            return False, (
                f"{pair}:stale_latest delay_min={delay_ms / 60_000:.1f} "
                f"limit={self.cfg.max_data_delay_minutes}"
            )

        fresh_rows = self.fresh_points_after_start(pair)
        if len(fresh_rows) < self.cfg.min_fresh_points_after_start:
            return False, (
                f"{pair}:insufficient_fresh_points count={len(fresh_rows)} "
                f"need={self.cfg.min_fresh_points_after_start}"
            )

        fresh_span_ms = int(fresh_rows[-1]["ts"] - fresh_rows[0]["ts"]) if len(fresh_rows) >= 2 else 0
        if fresh_span_ms < self.cfg.min_fresh_span_minutes * 60_000:
            return False, (
                f"{pair}:insufficient_fresh_span span_min={fresh_span_ms / 60_000:.1f} "
                f"need={self.cfg.min_fresh_span_minutes}"
            )

        return True, (
            f"{pair}:fresh ok latest_delay_min={delay_ms / 60_000:.1f} "
            f"fresh_points={len(fresh_rows)} span_min={fresh_span_ms / 60_000:.1f}"
        )

    def freshness_check_pairs(self) -> List[str]:
        preferred = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "SUI/USD"]
        pairs = [pair for pair in preferred if pair in self.trade_pairs]
        if len(pairs) >= 3:
            return pairs[:5]

        others = sorted(pair for pair in self.trade_pairs if pair not in pairs)
        return (pairs + others)[:5]

    def history_is_fresh_enough(self) -> tuple[bool, str]:
        check_pairs = self.freshness_check_pairs()
        if not check_pairs:
            return False, "no_check_pairs"

        ok_pairs = []
        bad_reasons = []

        for pair in check_pairs:
            ok, reason = self.pair_freshness_status(pair)
            if ok:
                ok_pairs.append(pair)
            else:
                bad_reasons.append(reason)

        ok_ratio = len(ok_pairs) / len(check_pairs)
        need_ok = min(3, len(check_pairs))

        if len(ok_pairs) >= need_ok or ok_ratio >= 0.6:
            return True, (
                f"fresh basket ok ok={len(ok_pairs)}/{len(check_pairs)} "
                f"pairs={ok_pairs}"
            )

        return False, (
            f"fresh basket failed ok={len(ok_pairs)}/{len(check_pairs)} "
            f"bad={bad_reasons[:3]}"
        )

    def get_wallet(self) -> Dict[str, Dict[str, float]]:
        response = self.client.balance()
        logger.info("RAW BALANCE RESPONSE: %s", response)
        if not response.get("Success", False):
            raise RuntimeError(f"Balance fetch failed: {response}")
        wallet: Dict[str, Dict[str, float]] = {}
        for coin, value in response.get("SpotWallet", {}).items():
            wallet[coin] = {"Free": safe_float(value.get("Free")), "Lock": safe_float(value.get("Lock"))}
        logger.info("PARSED WALLET: %s", wallet)
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

    def prune_recovery_entry_counters(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        self.state.recovery_entries_by_day = {
            day: count for day, count in self.state.recovery_entries_by_day.items() if day >= today
        }
        self.state.trades_by_day = {
            day: count for day, count in self.state.trades_by_day.items() if day >= today
        }
        self.state.soft_trades_by_day = {
            day: count for day, count in self.state.soft_trades_by_day.items() if day >= today
        }
        self.state.buy_trades_by_day = {
            day: count for day, count in self.state.buy_trades_by_day.items() if day >= today
        }

    def today_trade_count(self) -> int:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        return self.state.trades_by_day.get(today, 0)

    def today_soft_trade_count(self) -> int:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        return self.state.soft_trades_by_day.get(today, 0)

    def today_buy_trade_count(self) -> int:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        return self.state.buy_trades_by_day.get(today, 0)

    @staticmethod
    def is_hard_risk_reason(reason: str) -> bool:
        hard_tokens = ("stop_loss", "trailing_stop", "portfolio_drawdown", "flash_crash_defense")
        return any(token in reason for token in hard_tokens)

    def can_place_soft_trade(self) -> bool:
        limit = max(self.cfg.daily_soft_trade_limit, 0)
        return limit <= 0 or self.today_soft_trade_count() < limit

    def can_place_new_entry(self) -> bool:
        limit = max(self.cfg.daily_new_entry_limit, 0)
        return limit <= 0 or self.today_buy_trade_count() < limit

    def record_trade_activity(self, side: str, reason: str) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        self.state.trades_by_day[today] = self.state.trades_by_day.get(today, 0) + 1
        if side.upper() == "BUY":
            self.state.buy_trades_by_day[today] = self.state.buy_trades_by_day.get(today, 0) + 1
        if not self.is_hard_risk_reason(reason):
            self.state.soft_trades_by_day[today] = self.state.soft_trades_by_day.get(today, 0) + 1

    def can_place_recovery_entry(self, ts_ms: int) -> bool:
        day_key = time.strftime("%Y-%m-%d", time.gmtime(ts_ms / 1000.0))
        return self.state.recovery_entries_by_day.get(day_key, 0) < self.cfg.recovery_daily_entry_limit

    def record_recovery_entry(self, ts_ms: int) -> None:
        day_key = time.strftime("%Y-%m-%d", time.gmtime(ts_ms / 1000.0))
        self.state.recovery_entries_by_day[day_key] = self.state.recovery_entries_by_day.get(day_key, 0) + 1

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

    def normalize_order_quantity(self, pair: str, quantity: float, price: float) -> float:
        rules = self.trade_pairs[pair]
        amount_precision = int(rules.get("AmountPrecision", 6))
        min_order = safe_float(rules.get("MiniOrder", 1.0))

        if price <= 0 or quantity <= 0:
            return 0.0

        normalized = round_down(quantity, amount_precision)
        if normalized <= 0:
            return 0.0

        if normalized * price < min_order:
            return 0.0

        return normalized

    def sell_quantity_for_position(self, pair: str, quantity: float, price: float) -> float:
        """
        缁熶竴鎵€鏈?SELL 璺緞鐨勬暟閲忚鏁淬€?        浼樺厛鎸夋寔浠撳師濮嬫暟閲忓悜涓嬪彇鏁村埌浜ゆ槗绮惧害锛岄伩鍏嶈秴鍗栵紱
        鑻ヨ鏁村悗杈句笉鍒版渶灏忎笅鍗曢锛屽垯杩斿洖 0銆?        """
        return self.normalize_order_quantity(pair, quantity, price)

    def sell_quantity_for_fraction(self, pair: str, quantity: float, price: float, fraction: float) -> float:
        if price <= 0 or quantity <= 0 or fraction <= 0:
            return 0.0
        if fraction >= 0.999999:
            return self.sell_quantity_for_position(pair, quantity, price)
        return self.normalize_order_quantity(pair, quantity * fraction, price)

    def unrealized_return(self, entry_price: float, price: float) -> float:
        if entry_price <= 0 or price <= 0:
            return 0.0
        return price / entry_price - 1.0

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
        self.record_trade_activity(side, reason)
        logger.info("Order OK: %s %s qty=%.8f reason=%s", side, pair, quantity, reason)
        return True

    def build_position_meta(self, pair: str, quantity: float, price: float, score: float) -> PositionMeta:
        meta = self.position_meta(pair)
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if meta is None:
            meta = PositionMeta(
                pair=pair,
                quantity=quantity,
                entry_price=price,
                highest_price=price,
                last_trade_ts=now_ms(),
                last_signal_score=score,
                last_reason="recovered_from_balance",
                entry_day=today,
            )
        meta.quantity = quantity
        meta.highest_price = max(meta.highest_price, price)
        meta.last_signal_score = score
        return meta

    def update_not_in_targets_counter(
            self,
            meta: PositionMeta,
            pair: str,
            targets: Dict[str, float],
    ) -> int:
        """
        鍙湁褰?pair 涓嶅湪 targets 鎴栫洰鏍囨潈閲嶆瀬灏忕殑鏃跺€欙紝鎵嶇疮璁¤鏁般€?        涓€鏃﹂噸鏂板洖鍒?targets锛屽氨娓呴浂銆?        """
        target_weight = float(targets.get(pair, 0.0))
        if target_weight > 1e-12:
            meta.not_in_targets_bars = 0
        else:
            meta.not_in_targets_bars += 1
        return meta.not_in_targets_bars

    def not_in_targets_confirmed(
            self,
            meta: PositionMeta,
            confirm_bars: int = 3,
    ) -> bool:
        return meta.not_in_targets_bars >= confirm_bars

    def exit_reasons(
            self,
            pair: str,
            meta: PositionMeta,
            price: float,
            score: float,
            targets: Dict[str, float],
    ) -> List[str]:
        reasons: List[str] = []
        recovery_position = meta.last_reason == "recovery_reentry"
        min_hold_minutes = float(getattr(self.cfg, 'min_hold_minutes', 180))
        if recovery_position:
            min_hold_minutes *= float(getattr(self.cfg, "recovery_min_hold_multiplier", 2.0))
        min_hold_ms = int(min_hold_minutes * 60_000)
        allow_soft_exit = now_ms() - meta.last_trade_ts >= min_hold_ms
        same_day_entry = meta.entry_day == time.strftime("%Y-%m-%d", time.gmtime())
        if getattr(self.cfg, "block_same_day_soft_exit", True) and same_day_entry:
            allow_soft_exit = False

        # Price-based risk exits should always remain active.
        if price <= meta.entry_price * (1 - self.cfg.per_position_stop_loss):
            reasons.append("stop_loss")
        if price <= meta.highest_price * (1 - self.cfg.per_position_trailing_stop):
            reasons.append("trailing_stop")

        # Update out-of-target confirmation before considering soft exits.
        target_weight = float(targets.get(pair, 0.0))
        not_in_targets_count = self.update_not_in_targets_counter(meta, pair, targets)
        in_targets = target_weight > 1e-12
        confirm_bars = int(getattr(self.cfg, "recovery_not_in_targets_confirm_bars", 6)) if recovery_position else 3
        confirmed_not_in_targets = self.not_in_targets_confirmed(meta, confirm_bars=confirm_bars)

        # Soft exits wait for a minimum holding period to reduce fee drag.
        exit_threshold = self.cfg.exit_score_threshold
        if recovery_position:
            exit_threshold -= float(getattr(self.cfg, "recovery_exit_score_grace", 0.08))
        score_decay = allow_soft_exit and score < exit_threshold

        if in_targets:
            if score_decay:
                reasons.append("score_decay")
        else:
            if allow_soft_exit and confirmed_not_in_targets:
                if score_decay:
                    reasons.append("score_decay")
                reasons.append("not_in_targets")
            else:
                logger.info(
                    "Hold %s pending soft-exit confirmation: %d/%d bars hold_ok=%s score_decay=%s",
                    pair,
                    not_in_targets_count,
                    confirm_bars,
                    allow_soft_exit,
                    score_decay,
                )

        meta.last_reason = "+".join(reasons) if reasons else "hold"
        return reasons

    def manage_existing_positions(
            self,
            positions: Dict[str, float],
            tickers: Dict[str, Any],
            features: Dict[str, Dict[str, float]],
            targets: Dict[str, float],
    ) -> set[str]:
        skip_trim_pairs: set[str] = set()
        for pair, quantity in list(positions.items()):
            price = self.pair_price(pair, tickers)
            if price <= 0:
                continue

            feature = features.get(pair)
            fresh_ok, fresh_reason = self.pair_freshness_status(pair)

            # Without a fresh feature, keep only price-based protection.
            if feature is None or not fresh_ok:
                prev_meta = self.position_meta(pair)
                fallback_score = prev_meta.last_signal_score if prev_meta is not None else 0.0
                meta = self.build_position_meta(pair, quantity, price, fallback_score)

                reasons: List[str] = []
                if price <= meta.entry_price * (1 - self.cfg.per_position_stop_loss):
                    reasons.append("stop_loss")
                if price <= meta.highest_price * (1 - self.cfg.per_position_trailing_stop):
                    reasons.append("trailing_stop")

                if reasons:
                    reason_text = "+".join(reasons)
                    if not fresh_ok:
                        reason_text += f"+stale_data"

                    sell_quantity = self.sell_quantity_for_position(pair, quantity, price)
                    if sell_quantity <= 0:
                        logger.info(
                            "Skip SELL %s because normalized quantity is too small. raw_qty=%.12f",
                            pair,
                            quantity,
                        )
                        meta.last_reason = "sell_qty_too_small"
                        self.set_position_meta(meta)
                        continue

                    try:
                        ok = self.submit_market_order(
                            pair,
                            "SELL",
                            sell_quantity,
                            reason_text,
                            fallback_score,
                            price,
                        )
                    except UnknownOrderStateError as exc:
                        logger.exception(
                            "Exit SELL status unknown for %s. qty=%.12f err=%s",
                            pair,
                            sell_quantity,
                            exc,
                        )
                        meta.last_reason = "sell_unknown_state"
                        self.set_position_meta(meta)
                        continue
                    except Exception as exc:
                        logger.exception(
                            "Exit SELL failed for %s. qty=%.12f err=%s",
                            pair,
                            sell_quantity,
                            exc,
                        )
                        meta.last_reason = "sell_failed"
                        self.set_position_meta(meta)
                        continue

                    if ok:
                        if sell_quantity >= quantity * 0.999999:
                            self.remove_position_meta(pair)
                        self.set_cooldown(pair, self.cfg.cooldown_minutes)
                    else:
                        self.set_position_meta(meta)
                    continue

                meta.last_reason = "hold_stale" if not fresh_ok else "hold_no_feature"
                self.set_position_meta(meta)
                logger.info(
                    "Hold existing %s without score-based exit: feature=%s fresh=%s reason=%s",
                    pair,
                    feature is not None,
                    fresh_ok,
                    fresh_reason if not fresh_ok else "ok",
                )
                continue

            score = feature["score"]
            meta = self.build_position_meta(pair, quantity, price, score)
            reasons = self.exit_reasons(pair, meta, price, score, targets)

            if reasons:
                soft_only_reasons = [reason for reason in reasons if not self.is_hard_risk_reason(reason)]
                if soft_only_reasons and not self.can_place_soft_trade():
                    meta.last_reason = "hold_soft_trade_cap"
                    self.set_position_meta(meta)
                    continue
                unrealized_return = self.unrealized_return(meta.entry_price, price)
                protected_profit = unrealized_return >= self.cfg.profit_protect_threshold
                if "stop_loss" in reasons or "trailing_stop" in reasons:
                    sell_fraction = 1.0
                elif "not_in_targets" in reasons:
                    sell_fraction = (
                        self.cfg.profit_protect_not_in_targets_fraction
                        if protected_profit
                        else 0.30
                    )
                else:
                    sell_fraction = (
                        self.cfg.profit_protect_score_decay_fraction
                        if protected_profit
                        else 0.25
                    )
                sell_quantity = self.sell_quantity_for_fraction(pair, quantity, price, sell_fraction)
                if sell_quantity <= 0:
                    logger.info(
                        "Skip SELL %s because normalized quantity is too small. raw_qty=%.12f",
                        pair,
                        quantity,
                    )
                    meta.last_reason = "sell_qty_too_small"
                    self.set_position_meta(meta)
                    continue

                try:
                    ok = self.submit_market_order(
                        pair,
                        "SELL",
                        sell_quantity,
                        "+".join(reasons),
                        score,
                        price,
                    )
                except UnknownOrderStateError as exc:
                    logger.exception(
                        "Exit SELL status unknown for %s. qty=%.12f err=%s",
                        pair,
                        sell_quantity,
                        exc,
                    )
                    meta.last_reason = "sell_unknown_state"
                    self.set_position_meta(meta)
                    continue
                except Exception as exc:
                    logger.exception(
                        "Exit SELL failed for %s. qty=%.12f err=%s",
                        pair,
                        sell_quantity,
                        exc,
                    )
                    meta.last_reason = "sell_failed"
                    self.set_position_meta(meta)
                    continue

                if ok:
                    if sell_quantity >= quantity * 0.999999:
                        self.remove_position_meta(pair)
                    elif sell_fraction < 1.0:
                        meta.quantity = quantity - sell_quantity
                        meta.last_trade_ts = now_ms()
                        meta.last_reason = "+".join(reasons)
                        self.set_position_meta(meta)
                        skip_trim_pairs.add(pair)
                    self.set_cooldown(pair, self.cfg.cooldown_minutes)
                else:
                    self.set_position_meta(meta)
                continue

            self.set_position_meta(meta)
        return skip_trim_pairs

    def trim_positions(
            self,
            portfolio: PortfolioSnapshot,
            tickers: Dict[str, Any],
            features: Dict[str, Dict[str, float]],
            targets: Dict[str, float],
            rebalance_threshold: float,
            skip_pairs: Optional[set[str]] = None,
    ) -> None:
        skip_pairs = skip_pairs or set()
        for pair, quantity in list(portfolio.positions.items()):
            if pair in skip_pairs:
                continue
            price = self.pair_price(pair, tickers)
            if price <= 0:
                continue
            meta = self.position_meta(pair)
            same_day_entry = meta is not None and meta.entry_day == time.strftime("%Y-%m-%d", time.gmtime())

            target_usd = portfolio.equity * targets.get(pair, 0.0)
            current_usd = portfolio.current_notional.get(pair, 0.0)
            trim_usd = current_usd - target_usd
            if trim_usd < rebalance_threshold:
                continue
            if getattr(self.cfg, "block_same_day_soft_exit", True) and same_day_entry and target_usd <= 0:
                continue

            if target_usd <= 0:
                sell_quantity = self.sell_quantity_for_position(pair, quantity, price)
                reason = "target_exit_retry"
            else:
                sell_quantity = self.quantity_for_notional(pair, trim_usd, price)
                sell_quantity = min(
                    sell_quantity,
                    self.sell_quantity_for_position(pair, quantity, price),
                )
                reason = "target_trim"

            if sell_quantity <= 0:
                continue
            if not self.can_place_soft_trade():
                continue

            score = features.get(pair, {}).get("score", -999.0)

            try:
                ok = self.submit_market_order(pair, "SELL", sell_quantity, reason, score, price)
            except UnknownOrderStateError as exc:
                logger.exception(
                    "Trim SELL status unknown for %s. qty=%.12f err=%s",
                    pair,
                    sell_quantity,
                    exc,
                )
                continue
            except Exception as exc:
                logger.exception(
                    "Trim SELL failed for %s. qty=%.12f err=%s",
                    pair,
                    sell_quantity,
                    exc,
                )
                continue

            if ok:
                if sell_quantity >= quantity * 0.999999:
                    self.remove_position_meta(pair)
                    if target_usd <= 0:
                        self.set_cooldown(pair, self.cfg.cooldown_minutes)

    def record_buy_fill(self, pair: str, quantity: float, price: float, score: float) -> None:
        self.record_buy_fill_with_reason(pair, quantity, price, score, "target_rebalance")

    def record_buy_fill_with_reason(self, pair: str, quantity: float, price: float, score: float, reason: str) -> None:
        meta = self.position_meta(pair)
        trade_day = time.strftime("%Y-%m-%d", time.gmtime())
        if meta is None:
            self.set_position_meta(
                PositionMeta(
                    pair=pair,
                    quantity=quantity,
                    entry_price=price,
                    highest_price=price,
                    last_trade_ts=now_ms(),
                    last_signal_score=score,
                    last_reason=reason,
                    recovery_trade_day=trade_day if reason == "recovery_reentry" else "",
                    entry_day=trade_day,
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
        meta.last_reason = reason
        meta.recovery_trade_day = trade_day if reason == "recovery_reentry" else meta.recovery_trade_day
        meta.entry_day = trade_day
        self.set_position_meta(meta)

    def add_target_positions(
            self,
            portfolio: PortfolioSnapshot,
            tickers: Dict[str, Any],
            features: Dict[str, Dict[str, float]],
            targets: Dict[str, float],
            entry_modes: Dict[str, str],
            rebalance_threshold: float,
    ) -> None:
        for pair, weight in sorted(targets.items(), key=lambda item: item[1], reverse=True):
            if self.in_cooldown(pair):
                continue

            fresh_ok, fresh_reason = self.pair_freshness_status(pair)
            if not fresh_ok:
                logger.info("Skip BUY %s due to stale pair data: %s", pair, fresh_reason)
                continue

            feature = features.get(pair)
            if feature is None:
                logger.info("Skip BUY %s due to missing feature row.", pair)
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

            score = feature["score"]
            entry_reason = entry_modes.get(pair, "target_rebalance")
            if not self.can_place_soft_trade():
                break
            if not self.can_place_new_entry():
                logger.info("Skip BUY %s because daily new-entry budget is exhausted.", pair)
                break
            if entry_reason == "recovery_reentry" and not self.can_place_recovery_entry(now_ms()):
                logger.info("Skip recovery BUY %s because daily recovery entry limit is reached.", pair)
                continue
            if self.submit_market_order(pair, "BUY", quantity, entry_reason, score, price):
                self.record_buy_fill_with_reason(pair, quantity, price, score, entry_reason)
                if entry_reason == "recovery_reentry":
                    self.record_recovery_entry(now_ms())
                    self.set_cooldown(pair, self.cfg.recovery_entry_cooldown_minutes)
                portfolio.usd_free -= quantity * price

    def maybe_place_daily_activity_probe(
            self,
            portfolio: PortfolioSnapshot,
            tickers: Dict[str, Any],
            features: Dict[str, Dict[str, float]],
            pair: Optional[str],
    ) -> None:
        if not self.cfg.daily_activity_enabled or not pair or self.today_trade_count() > 0:
            return
        if not self.can_place_soft_trade() or not self.can_place_new_entry():
            return
        if pair in portfolio.positions or self.in_cooldown(pair):
            return
        feature = features.get(pair)
        if feature is None:
            return
        price = self.pair_price(pair, tickers)
        if price <= 0:
            return
        usable_cash = max(0.0, portfolio.usd_free - portfolio.equity * self.cfg.cash_buffer)
        if usable_cash <= 0:
            return
        probe_usd = min(usable_cash, portfolio.equity * self.cfg.daily_activity_probe_exposure)
        quantity = self.quantity_for_notional(pair, probe_usd, price)
        if quantity <= 0:
            return
        score = feature["score"]
        if self.submit_market_order(pair, "BUY", quantity, "daily_activity_probe", score, price):
            self.record_buy_fill_with_reason(pair, quantity, price, score, "daily_activity_probe")
            self.set_cooldown(pair, self.cfg.daily_activity_entry_cooldown_minutes)
            portfolio.usd_free -= quantity * price

    def fast_shock_defense(
            self,
            portfolio: PortfolioSnapshot,
            tickers: Dict[str, Any],
    ) -> tuple[set[str], bool]:
        skip_trim_pairs: set[str] = set()
        triggered = False
        lookback = max(int(getattr(self.cfg, "flash_crash_lookback_minutes", 5)), 3)
        drop_threshold = float(getattr(self.cfg, "flash_crash_drop_threshold", -0.03))
        rebound_threshold = float(getattr(self.cfg, "flash_crash_rebound_threshold", 0.008))
        sell_fraction = float(getattr(self.cfg, "flash_crash_sell_fraction", 0.50))

        for pair, quantity in list(portfolio.positions.items()):
            series = self.history.get(pair)
            if quantity <= 0 or not series or len(series) < lookback:
                continue
            prices = [float(entry.get("price", 0.0)) for entry in list(series)[-lookback:] if float(entry.get("price", 0.0)) > 0]
            if len(prices) < 3:
                continue
            price = prices[-1]
            recent_peak = max(prices)
            recent_low = min(prices)
            drop_from_peak = price / recent_peak - 1.0 if recent_peak > 0 else 0.0
            rebound_from_low = price / recent_low - 1.0 if recent_low > 0 else 0.0
            if drop_from_peak > drop_threshold or rebound_from_low >= rebound_threshold:
                continue

            sell_quantity = self.sell_quantity_for_fraction(pair, quantity, price, sell_fraction)
            if sell_quantity <= 0:
                continue
            score = -999.0
            if self.submit_market_order(pair, "SELL", sell_quantity, "flash_crash_defense", score, price):
                skip_trim_pairs.add(pair)
                triggered = True
                if sell_quantity >= quantity * 0.999999:
                    self.remove_position_meta(pair)
                self.set_cooldown(pair, self.cfg.cooldown_minutes)

        return skip_trim_pairs, triggered

    def exit_all_positions(self, positions: Dict[str, float], tickers: Dict[str, Any], reason: str) -> None:
        for pair, quantity in list(positions.items()):
            if quantity <= 0:
                continue

            price = self.pair_price(pair, tickers)
            if price <= 0:
                logger.warning("Skip forced SELL %s due to invalid price.", pair)
                continue

            sell_quantity = self.sell_quantity_for_position(pair, quantity, price)
            if sell_quantity <= 0:
                logger.warning(
                    "Skip forced SELL %s because normalized quantity is too small. raw_qty=%.12f price=%.8f",
                    pair,
                    quantity,
                    price,
                )
                continue

            try:
                ok = self.submit_market_order(pair, "SELL", sell_quantity, reason, -999.0, price)
            except UnknownOrderStateError as exc:
                logger.exception(
                    "Forced SELL status unknown for %s. Continue exiting others. qty=%.12f err=%s",
                    pair,
                    sell_quantity,
                    exc,
                )
                continue
            except Exception as exc:
                logger.exception(
                    "Forced SELL failed for %s. Continue exiting others. qty=%.12f err=%s",
                    pair,
                    sell_quantity,
                    exc,
                )
                continue

            if ok:
                # 鍙湁褰撴湰娆″崠鍗曞熀鏈瓑浜庡彲鍗栨寔浠撴椂锛屾墠绉婚櫎 meta
                if sell_quantity >= quantity * 0.999999:
                    self.remove_position_meta(pair)
                self.set_cooldown(pair, self.cfg.cooldown_minutes)

    def reduce_positions(
            self,
            positions: Dict[str, float],
            tickers: Dict[str, Any],
            reason: str,
            sell_fraction: float,
    ) -> set[str]:
        reduced_pairs: set[str] = set()
        for pair, quantity in list(positions.items()):
            if quantity <= 0:
                continue

            price = self.pair_price(pair, tickers)
            if price <= 0:
                logger.warning("Skip risk reduction SELL %s due to invalid price.", pair)
                continue

            sell_quantity = self.sell_quantity_for_fraction(pair, quantity, price, sell_fraction)
            if sell_quantity <= 0:
                continue

            try:
                ok = self.submit_market_order(pair, "SELL", sell_quantity, reason, -999.0, price)
            except UnknownOrderStateError as exc:
                logger.exception(
                    "Risk reduction SELL status unknown for %s. qty=%.12f err=%s",
                    pair,
                    sell_quantity,
                    exc,
                )
                continue
            except Exception as exc:
                logger.exception(
                    "Risk reduction SELL failed for %s. qty=%.12f err=%s",
                    pair,
                    sell_quantity,
                    exc,
                )
                continue

            if ok:
                if sell_quantity >= quantity * 0.999999:
                    self.remove_position_meta(pair)
                else:
                    meta = self.position_meta(pair)
                    if meta is not None:
                        meta.quantity = quantity - sell_quantity
                        meta.last_trade_ts = now_ms()
                        meta.last_reason = reason
                        self.set_position_meta(meta)
                        reduced_pairs.add(pair)
                self.set_cooldown(pair, self.cfg.cooldown_minutes)
        return reduced_pairs

    def rebalance_once(self) -> None:
        now = time.time()

        # 1. 姣忓垎閽熼兘鏇存柊甯傚満鏁版嵁
        tickers = self.fetch_all_tickers()
        self.update_history(tickers)

        portfolio = self.build_portfolio_snapshot(tickers)
        self.capture_portfolio_state(portfolio, tickers)
        self.prune_recovery_entry_counters()

        skip_trim_pairs: set[str] = set()
        allow_new_entries = now_ms() >= self.state.portfolio_reentry_allowed_at
        shock_skip_pairs, shock_triggered = self.fast_shock_defense(portfolio, tickers)
        if shock_skip_pairs:
            skip_trim_pairs |= shock_skip_pairs
            allow_new_entries = False
            self.state.portfolio_reentry_allowed_at = max(
                self.state.portfolio_reentry_allowed_at,
                now_ms() + self.cfg.cooldown_minutes * 60_000,
            )
            portfolio = self.build_portfolio_snapshot(tickers)
            self.sync_position_meta(portfolio.positions, tickers)
            if shock_triggered:
                logger.warning("Fast shock defense triggered for pairs: %s", sorted(shock_skip_pairs))

        if portfolio.drawdown >= self.cfg.max_portfolio_drawdown and portfolio.positions:
            dd_sell_fraction, dd_cooldown_minutes = portfolio_drawdown_response(self.cfg, portfolio.drawdown)
            logger.warning(
                "Portfolio drawdown throttle triggered. drawdown=%.4f sell_fraction=%.2f cooldown_minutes=%d",
                portfolio.drawdown,
                dd_sell_fraction,
                dd_cooldown_minutes,
            )
            skip_trim_pairs |= self.reduce_positions(
                portfolio.positions,
                tickers,
                "portfolio_drawdown",
                sell_fraction=dd_sell_fraction,
            )
            allow_new_entries = False
            self.state.portfolio_reentry_allowed_at = max(
                self.state.portfolio_reentry_allowed_at,
                now_ms() + dd_cooldown_minutes * 60_000,
            )
            portfolio = self.build_portfolio_snapshot(tickers)
            self.sync_position_meta(portfolio.positions, tickers)
            if not portfolio.positions:
                self.state.portfolio_reentry_allowed_at = min(
                    self.state.portfolio_reentry_allowed_at,
                    now_ms() + self.cfg.empty_book_reentry_minutes * 60_000,
                )

        signals = self.strategy.generate_signals(
            history=self.history,
            trade_pairs=self.trade_pairs,
            positions=portfolio.positions,
            prev_risk_on=self.state.risk_on,
            current_drawdown=portfolio.drawdown,
        )
        self.state.risk_on = signals["risk_on"]
        self.log_signal_snapshot(signals)

        logger.info(
            "Regime=%s risk_on=%s overlay_regime=%s overlay_score=%.2f "
            "target_exposure=%.2f port_vol=%.4f avg_corr=%.4f "
            "mu_ready=%s mu_error=%s mu_blend=%s fixed_blend=%s "
            "pre=%s final=%s",
            signals["regime"]["regime"],
            signals["risk_on"],
            signals["portfolio_risk"]["market_regime"],
            signals["portfolio_risk"]["risk_score"],
            signals["portfolio_risk"]["target_exposure"],
            signals["portfolio_risk"]["portfolio_volatility"],
            signals["portfolio_risk"]["average_correlation"],
            signals["regime"].get("mu_ready"),
            signals["regime"].get("mu_error"),
            signals["regime"].get("mu_blend_weight"),
            signals["regime"].get("fixed_blend_weight"),
            signals.get("pre_risk_weights", {}),
            signals.get("weights", {}),
        )

        sample_pairs = list(signals.get("features", {}).items())[:5]
        sample_mu = {
            pair: {
                "pred_mu": feat.get("pred_mu"),
                "pred_mu_z": feat.get("pred_mu_z"),
                "score": feat.get("score"),
            }
            for pair, feat in sample_pairs
        }
        logger.info("MU sample=%s", sample_mu)

        if not signals["features"]:
            logger.info("Not enough history yet.")
            self.persist_runtime_state()
            return

        # warm up
        warmup_remaining_ms = (
            self.cfg.startup_warmup_minutes * 60_000
            - (now_ms() - self.session_start_ts)
        )
        if warmup_remaining_ms > 0:
            logger.info(
                "Startup warmup active, skip orders only. Remaining %.1f min",
                warmup_remaining_ms / 60_000,
            )
            self.persist_runtime_state()
            return

        # freshness gate
        fresh_ok, fresh_reason = self.history_is_fresh_enough()
        if not fresh_ok:
            logger.info("Freshness gate active, skip orders only. %s", fresh_reason)
            self.persist_runtime_state()
            return

        full_rebalance_due = (now - self.last_rebalance_ts) >= self.cfg.rebalance_minutes * 60
        risk_check_due = (now - self.last_risk_check_ts) >= self.cfg.risk_check_minutes * 60
        if not full_rebalance_due and not risk_check_due:
            logger.info("No risk check or full rebalance due yet.")
            self.persist_runtime_state()
            return

        # 3. 鐪熸涓嬪崟
        skip_trim_pairs |= self.manage_existing_positions(
            portfolio.positions,
            tickers,
            signals["features"],
            signals["weights"],
        )

        if full_rebalance_due:
            portfolio = self.build_portfolio_snapshot(tickers)
            self.sync_position_meta(portfolio.positions, tickers)
            rebalance_threshold = self.rebalance_notional_threshold(portfolio.equity)
            self.trim_positions(
                portfolio,
                tickers,
                signals["features"],
                signals["weights"],
                rebalance_threshold,
                skip_pairs=skip_trim_pairs,
            )

            portfolio = self.build_portfolio_snapshot(tickers)
            self.sync_position_meta(portfolio.positions, tickers)
            if allow_new_entries:
                self.add_target_positions(
                    portfolio,
                    tickers,
                    signals["features"],
                    signals["weights"],
                    signals.get("entry_modes", {}),
                    rebalance_threshold,
                )
            else:
                remaining_ms = max(0, self.state.portfolio_reentry_allowed_at - now_ms())
                logger.info(
                    "Skip new buys while portfolio drawdown cooldown is active. Remaining %.1f hours",
                    remaining_ms / 3_600_000,
                )
            portfolio = self.build_portfolio_snapshot(tickers)
            self.maybe_place_daily_activity_probe(
                portfolio,
                tickers,
                signals["features"],
                signals.get("daily_activity_probe"),
            )
            self.last_rebalance_ts = now
        else:
            logger.info("Risk-check cycle only: managed exits without full rebalance.")

        self.persist_runtime_state()
        self.last_risk_check_ts = now

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

