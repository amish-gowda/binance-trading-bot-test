import os
import sys
import time
import hmac
import hashlib
import logging
import argparse
import json
from urllib.parse import urlencode
from pathlib import Path
import unittest

import requests
from dotenv import load_dotenv


load_dotenv()

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "bot.log"

logger = logging.getLogger("BasicBot")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


class BinanceFuturesREST:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://testnet.binancefuture.com"):
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret are required")
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _encode_params(self, params: dict) -> str:
        items = []
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, bool):
                v = "true" if v else "false"
            else:
                v = str(v)
            items.append((k, v))
        return urlencode(items, doseq=True)

    def _sign(self, params: dict) -> dict:
        ps = dict(params)
        query_string = self._encode_params(ps)
        signature = hmac.new(self.api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        ps["signature"] = signature
        return ps

    def _safe_json(self, r: requests.Response):
        try:
            return r.json()
        except ValueError:
            return {"raw_text": r.text}

    def _request(self, method: str, path: str, params: dict = None, signed: bool = False):
        url = f"{self.base_url}{path}"
        params = params or {}
        try:
            if signed:
                params.update({"timestamp": int(time.time() * 1000)})
                signed_params = self._sign(params)
                if method.upper() == "GET":
                    logger.debug(f"REQUEST -> {method} {url} params={params}")
                    r = self.session.request(method, url, params=signed_params, timeout=15)
                else:
                    body = self._encode_params(signed_params)
                    headers = {"Content-Type": "application/x-www-form-urlencoded"}
                    logger.debug(f"REQUEST -> {method} {url} body={body}")
                    r = self.session.request(method, url, data=body, headers=headers, timeout=15)
            else:
                if method.upper() == "GET":
                    logger.debug(f"REQUEST -> {method} {url} params={params}")
                    r = self.session.request(method, url, params=params, timeout=15)
                else:
                    headers = {"Content-Type": "application/json"}
                    logger.debug(f"REQUEST -> {method} {url} json={params}")
                    r = self.session.request(method, url, json=params, headers=headers, timeout=15)

            logger.debug(f"RESPONSE [{r.status_code}] -> {r.text}")
            r.raise_for_status()
            return self._safe_json(r)
        except requests.HTTPError as e:
            text = None
            try:
                text = e.response.text if e.response is not None else None
            except Exception:
                text = None
            logger.error(f"HTTP error for {method} {url}: {e}; body={text}")
            return {"error": str(e), "response_text": text}
        except requests.RequestException as e:
            logger.error(f"Request exception for {method} {url}: {e}")
            return {"error": str(e)}

    def ping(self):
        return self._request("GET", "/fapi/v1/ping")

    def time(self):
        return self._request("GET", "/fapi/v1/time")

    def create_order(self, symbol: str, side: str, type_: str, quantity: float = None, price: float = None,
                     timeInForce: str = None, stopPrice: float = None, reduceOnly: bool = False, closePosition: bool = False,
                     positionSide: str = None, newClientOrderId: str = None, workingType: str = None, priceProtect: bool = False):
        path = "/fapi/v1/order"
        params = {"symbol": symbol, "side": side.upper(), "type": type_.upper()}
        if quantity is not None:
            params["quantity"] = float(quantity)
        if price is not None:
            params["price"] = float(price)
        if timeInForce is not None:
            params["timeInForce"] = timeInForce
        if stopPrice is not None:
            params["stopPrice"] = float(stopPrice)
        if reduceOnly:
            params["reduceOnly"] = True
        if closePosition:
            params["closePosition"] = True
        if positionSide is not None:
            params["positionSide"] = positionSide
        if newClientOrderId is not None:
            params["newClientOrderId"] = newClientOrderId
        if workingType is not None:
            params["workingType"] = workingType
        if priceProtect:
            params["priceProtect"] = True
        return self._request("POST", path, params=params, signed=True)

    def get_order(self, symbol: str, orderId: int = None, origClientOrderId: str = None):
        path = "/fapi/v1/order"
        params = {"symbol": symbol}
        if orderId is not None:
            params["orderId"] = int(orderId)
        if origClientOrderId is not None:
            params["origClientOrderId"] = origClientOrderId
        return self._request("GET", path, params=params, signed=True)


class BasicBot:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://testnet.binancefuture.com"):
        self.client = BinanceFuturesREST(api_key, api_secret, base_url=base_url)

    def ping(self):
        return self.client.ping()

    def place_market_order(self, symbol: str, side: str, quantity: float):
        logger.info(f"Placing MARKET order: {side} {quantity} {symbol}")
        return self.client.create_order(symbol=symbol, side=side, type_='MARKET', quantity=quantity)

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float, timeInForce: str = 'GTC'):
        logger.info(f"Placing LIMIT order: {side} {quantity} {symbol} @ {price} {timeInForce}")
        return self.client.create_order(symbol=symbol, side=side, type_='LIMIT', quantity=quantity, price=price, timeInForce=timeInForce)

    def place_stop_limit(self, symbol: str, side: str, quantity: float, price: float, stopPrice: float, timeInForce: str = 'GTC'):
        logger.info(f"Placing STOP_LIMIT order: {side} {quantity} {symbol} stopPrice={stopPrice} price={price}")
        return self.client.create_order(symbol=symbol, side=side, type_='STOP', quantity=quantity, price=price, stopPrice=stopPrice, timeInForce=timeInForce)


def validate_symbol(symbol: str) -> str:
    if not symbol or len(symbol) < 4:
        raise argparse.ArgumentTypeError("symbol must be a valid market symbol like BTCUSDT")
    return symbol.upper()


def positive_float(value):
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}")
    if f <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return f


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Simplified Binance Futures Trading Bot (Testnet)")
    parser.add_argument("--api-key", default=os.getenv("BINANCE_API_KEY"), help="Binance API key (or set BINANCE_API_KEY env var)")
    parser.add_argument("--api-secret", default=os.getenv("BINANCE_API_SECRET"), help="Binance API secret (or set BINANCE_API_SECRET env var)")
    parser.add_argument("--base-url", default=os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com"), help="Base URL for Futures Testnet")
    parser.add_argument("--symbol", type=validate_symbol, help="Trading symbol, e.g., BTCUSDT")
    parser.add_argument("--side", choices=["BUY", "SELL"], help="Order side")
    parser.add_argument("--type", dest="ordertype", choices=["MARKET", "LIMIT", "STOP_LIMIT"], help="Order type")
    parser.add_argument("--quantity", type=positive_float, help="Order quantity (in contract base units)")
    parser.add_argument("--price", type=positive_float, help="Limit price")
    parser.add_argument("--stopPrice", type=positive_float, help="Stop price for STOP_LIMIT")
    parser.add_argument("--timeInForce", default="GTC", choices=["GTC", "IOC", "FOK"], help="Time in force for LIMIT orders")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually send orders; just print what would be sent")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in tests and exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.run_tests:
        run_tests()
        return
    if not args.api_key or not args.api_secret:
        logger.error("API key and secret are required. Provide via CLI or environment variables.")
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables or pass --api-key and --api-secret.")
        return
    if not (args.symbol and args.side and args.ordertype and args.quantity):
        logger.info("Insufficient order parameters provided. Use --run-tests to execute tests or pass order parameters to place an order.")
        print("Example: --symbol BTCUSDT --side BUY --type MARKET --quantity 0.001")
        return
    bot = BasicBot(args.api_key, args.api_secret, base_url=args.base_url)
    ping_res = bot.ping()
    logger.info(f"Ping result: {ping_res}")
    if args.dry_run:
        logger.info("Dry-run mode: no orders will be placed. Showing payload below.")
    try:
        if args.ordertype == 'MARKET':
            if args.dry_run:
                logger.info(f"DRY: MARKET {args.side} {args.quantity} {args.symbol}")
            else:
                res = bot.place_market_order(args.symbol, args.side, args.quantity)
                print(json.dumps(res, indent=2))
        elif args.ordertype == 'LIMIT':
            if args.price is None:
                logger.error("LIMIT orders require --price")
                print("Provide --price for LIMIT orders")
                return
            if args.dry_run:
                logger.info(f"DRY: LIMIT {args.side} {args.quantity} {args.symbol} @ {args.price} {args.timeInForce}")
            else:
                res = bot.place_limit_order(args.symbol, args.side, args.quantity, args.price, timeInForce=args.timeInForce)
                print(json.dumps(res, indent=2))
        elif args.ordertype == 'STOP_LIMIT':
            if args.price is None or args.stopPrice is None:
                logger.error("STOP_LIMIT orders require --price and --stopPrice")
                print("Provide --price and --stopPrice for STOP_LIMIT orders")
                return
            if args.dry_run:
                logger.info(f"DRY: STOP_LIMIT {args.side} {args.quantity} {args.symbol} stopPrice={args.stopPrice} price={args.price}")
            else:
                res = bot.place_stop_limit(args.symbol, args.side, args.quantity, args.price, stopPrice=args.stopPrice, timeInForce=args.timeInForce)
                print(json.dumps(res, indent=2))
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")


class BasicTests(unittest.TestCase):
    def test_positive_float_valid(self):
        self.assertEqual(positive_float("1.5"), 1.5)

    def test_positive_float_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_float("-1")

    def test_encode_params_order(self):
        client = BinanceFuturesREST("k", "s")
        params = {"b": 2, "a": 1}
        encoded = client._encode_params(params)
        self.assertTrue(encoded.startswith("a=1&b=2"))

    def test_sign_adds_signature(self):
        client = BinanceFuturesREST("k", "secret")
        params = {"a": 1}
        signed = client._sign(params)
        self.assertIn("signature", signed)

    def test_parse_args_no_args(self):
        args = parse_args([])
        self.assertFalse(args.run_tests)


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(BasicTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("No arguments provided. Running built-in tests. Use --run-tests to run tests explicitly or provide order parameters to place an order.")
        run_tests()
    else:
        main()
