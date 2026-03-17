import requests
import time
import hmac
import hashlib

# ==============================
# CONFIG
# ==============================

BASE_URL = "https://mock-api.roostoo.com"

API_KEY = "YZJpIGyMhk1efgdP8qZM0LTZZ2eFJh35ovWByPgSG73XS5OeWruM8XygCqHypBK7"
SECRET_KEY = "PTtlKF7Vwed7MfiUCz6G6PySVDH5zP8Vjz4lmIYuxQyrjq5EvseYAJV3jAhVnYyK"

TIMEOUT = 1.5
MAX_RETRY = 3


# ==============================
# CORE REQUEST ENGINE
# ==============================

def _request(method, url, params=None, headers=None, data=None):
    """
    Robust request handler with retry + timeout
    """
    for attempt in range(MAX_RETRY):
        try:
            if method == "GET":
                res = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            else:
                res = requests.post(url, data=data, headers=headers, timeout=TIMEOUT)

            res.raise_for_status()
            return res.json()

        except requests.exceptions.Timeout:
            print(f"[Timeout] {url} | retry {attempt+1}")

        except requests.exceptions.RequestException as e:
            print(f"[Request Error] {url} | {e}")
            return None

        time.sleep(1)

    print(f"[FAILED] {url}")
    return None


# ==============================
# SIGNATURE
# ==============================

def _get_timestamp():
    return str(int(time.time() * 1000))


def _get_signed_headers(payload: dict = {}):
    payload['timestamp'] = _get_timestamp()

    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

    signature = hmac.new(
        SECRET_KEY.encode(),
        total_params.encode(),
        hashlib.sha256
    ).hexdigest()

    headers = {
        'RST-API-KEY': API_KEY,
        'MSG-SIGNATURE': signature,
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    return headers, payload, total_params


# ==============================
# PUBLIC API
# ==============================

def check_server_time():
    url = f"{BASE_URL}/v3/serverTime"
    return _request("GET", url)


def get_exchange_info():
    url = f"{BASE_URL}/v3/exchangeInfo"
    return _request("GET", url)


def get_ticker(pair=None):
    url = f"{BASE_URL}/v3/ticker"
    params = {'timestamp': _get_timestamp()}

    if pair:
        params['pair'] = pair

    return _request("GET", url, params=params)


# ==============================
# SIGNED API
# ==============================

def get_balance():
    url = f"{BASE_URL}/v3/balance"
    headers, payload, _ = _get_signed_headers({})
    return _request("GET", url, params=payload, headers=headers)


def get_pending_count():
    url = f"{BASE_URL}/v3/pending_count"
    headers, payload, _ = _get_signed_headers({})
    return _request("GET", url, params=payload, headers=headers)


def place_order(pair_or_coin, side, quantity, price=None, order_type=None):
    url = f"{BASE_URL}/v3/place_order"

    pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin

    if order_type is None:
        order_type = "LIMIT" if price else "MARKET"

    payload = {
        'pair': pair,
        'side': side.upper(),
        'type': order_type.upper(),
        'quantity': str(quantity)
    }

    if order_type == "LIMIT":
        payload['price'] = str(price)

    headers, _, total_params = _get_signed_headers(payload)

    return _request("POST", url, headers=headers, data=total_params)


def query_order(order_id=None, pair=None, pending_only=None):
    url = f"{BASE_URL}/v3/query_order"

    payload = {}

    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:
        payload['pair'] = pair
        if pending_only is not None:
            payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'

    headers, _, total_params = _get_signed_headers(payload)

    return _request("POST", url, headers=headers, data=total_params)


def cancel_order(order_id=None, pair=None):
    url = f"{BASE_URL}/v3/cancel_order"

    payload = {}

    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:
        payload['pair'] = pair

    headers, _, total_params = _get_signed_headers(payload)

    return _request("POST", url, headers=headers, data=total_params)


# ==============================
# SAFE WRAPPERS (关键)
# ==============================

def safe_get_ticker(pair):
    data = get_ticker(pair)
    if data is None:
        print("Ticker unavailable")
        return None

    return data.get("Data", {}).get(pair, None)


def safe_get_price(pair):
    ticker = safe_get_ticker(pair)
    if ticker is None:
        return None

    return float(ticker.get("LastPrice", 0))


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    print("\n--- Server Time ---")
    print(check_server_time())

    print("\n--- Exchange Info ---")
    info = get_exchange_info()
    if info:
        print(list(info.get("TradePairs", {}).keys()))
    else:
        print("Exchange API down")

    print("\n--- BTC Price ---")
    price = safe_get_price("BTC/USD")
    print(price)

    print("\n--- Balance ---")
    print(get_balance())