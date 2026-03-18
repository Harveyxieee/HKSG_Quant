from __future__ import annotations

import csv
import hashlib
import hmac
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class UnknownOrderStateError(RuntimeError):
    pass


class RoostooClient:
    def __init__(self, cfg: Any):
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

    @staticmethod
    def _append_csv(path: Path, headers: list[str], row: Dict[str, Any]) -> None:
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _sha256_json(payload: Dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self.time_offset_ms

    def sync_time(self) -> int:
        response = self.server_time()
        server_time = int(float(response.get("ServerTime", int(time.time() * 1000))))
        self.time_offset_ms = server_time - int(time.time() * 1000)
        return server_time

    def _sign(self, params: Dict[str, Any]) -> str:
        items = sorted((key, str(value)) for key, value in params.items())
        payload = "&".join(f"{key}={value}" for key, value in items)
        return hmac.new(self.cfg.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _log_request(self, method: str, path: str, params: Dict[str, Any], ok: bool, response_json: Any) -> None:
        self._append_csv(
            self.cfg.request_log_csv,
            ["ts", "method", "path", "params_sha256", "ok", "response_snippet"],
            {
                "ts": int(time.time() * 1000),
                "method": method,
                "path": path,
                "params_sha256": self._sha256_json(params),
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

