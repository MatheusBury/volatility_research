from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv

warnings.filterwarnings("ignore")

CALL_LETTERS = set("ABCDE")
PUT_LETTERS = set("FGHIJ")
PREFIX_MAP = {"PETR4": "PETR", "VALE3": "VALE", "ITUB4": "ITUB"}
RISK_FREE_RATE = 0.1475
MIN_D1_BARS = 10
MAX_IV = 5.0
DTE_MIN = 15
DTE_MAX = 365


class IVCollector:
    def __init__(self):
        self._connected = False
        self._option_metadata: list[dict[str, Any]] = []
        self._options_data: dict[str, pd.DataFrame] = {}
        self._underlying_data: dict[str, pd.DataFrame] = {}

    def connect(self) -> bool:
        if mt5.initialize():
            self._connected = True
            return True
        return False

    def _get_option_symbols(self, stock: str) -> list[Any]:
        prefix = PREFIX_MAP.get(stock)
        if prefix is None:
            return []
        all_syms = mt5.symbols_get()
        if all_syms is None:
            return []
        return [s for s in all_syms if s.name.startswith(prefix) and s.option_mode > 0]

    def _extract_chain(self, stock: str) -> list[dict[str, Any]]:
        opts = self._get_option_symbols(stock)
        prefix = PREFIX_MAP.get(stock, "")
        underlying_info = mt5.symbol_info(stock)
        underlying_price = float(underlying_info.bid) if underlying_info else 0.0

        results: list[dict[str, Any]] = []
        for s in opts:
            name = s.name
            series_letter = name[len(prefix)]
            opt_type = "C" if series_letter in CALL_LETTERS else "P" if series_letter in PUT_LETTERS else "?"
            exp_time = s.expiration_time
            if exp_time is None:
                continue
            exp_dt = datetime.fromtimestamp(exp_time)
            dte = (exp_dt - datetime.now()).days
            if dte < DTE_MIN or dte > DTE_MAX:
                continue
            strike = float(s.option_strike) if s.option_strike else 0.0
            if strike <= 0:
                continue
            results.append({
                "stock": stock,
                "symbol": name,
                "option_type": opt_type,
                "strike": strike,
                "expiration_time": exp_time,
                "expiration_date": exp_dt,
                "dte": dte,
                "bid": float(s.bid) if s.bid else 0.0,
                "ask": float(s.ask) if s.ask else 0.0,
                "last": float(s.last) if s.last else 0.0,
                "volume": float(s.session_volume or 0),
                "underlying_price": underlying_price,
                "series_letter": series_letter,
            })
        return results

    def get_atm_iv(self, stock: str, date: pd.Timestamp) -> Optional[dict[str, Any]]:
        if not self._connected:
            return None
        prefix = PREFIX_MAP.get(stock)
        if prefix is None:
            return None

        opts = self._extract_chain(stock)
        puts = [o for o in opts if o["option_type"] == "P"]
        if not puts:
            return None

        underlying_info = mt5.symbol_info(stock)
        spot = float(underlying_info.bid) if underlying_info else 0.0
        if spot <= 0:
            return None

        for p in puts:
            p["atm_dist"] = abs(p["strike"] - spot)
        atm_put = min(puts, key=lambda x: x["atm_dist"])

        rates = mt5.copy_rates_from_pos(atm_put["symbol"], mt5.TIMEFRAME_D1, 0, 5000)
        if rates is None or len(rates) < MIN_D1_BARS:
            return None
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")

        date_naive = date.tz_localize(None) if date.tz is not None else date
        date_only = date_naive.date()
        mask = df["timestamp"].dt.date == date_only
        row = df[mask]
        if row.empty:
            return None

        option_price = float(row.iloc[0]["close"])
        underlying_df = self._get_underlying_daily(stock)
        if underlying_df is None:
            return None
        und_mask = underlying_df["timestamp"].dt.date == date_only
        und_row = underlying_df[und_mask]
        if und_row.empty:
            return None
        underlying_price = float(und_row.iloc[0]["close"])
        if underlying_price <= 0:
            return None

        tte_years = (atm_put["expiration_date"] - date_naive).days / 365.0
        if tte_years < DTE_MIN / 365.0:
            return None

        try:
            iv = bs_iv(option_price, underlying_price, atm_put["strike"], tte_years, RISK_FREE_RATE, "p")
        except Exception:
            return None

        if iv is None or iv <= 0 or iv >= MAX_IV:
            return None

        return {
            "iv": float(iv),
            "strike": atm_put["strike"],
            "opt_type": "P",
            "dte": int(tte_years * 365),
            "option_price": option_price,
            "underlying_price": underlying_price,
            "symbol": atm_put["symbol"],
        }

    def _get_underlying_daily(self, stock: str) -> Optional[pd.DataFrame]:
        if stock in self._underlying_data:
            return self._underlying_data[stock]
        rates = mt5.copy_rates_from_pos(stock, mt5.TIMEFRAME_D1, 0, 5000)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        self._underlying_data[stock] = df
        return df

    def get_iv_timeseries(
        self, stock: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        if not self._connected:
            return pd.DataFrame()

        # Get ALL option symbols (no DTE filter — we'll filter historically)
        opts = self._get_option_symbols(stock)
        prefix = PREFIX_MAP.get(stock, "")

        puts = []
        for s in opts:
            sl = s.name[len(prefix)]
            if sl in PUT_LETTERS and s.expiration_time and s.option_strike:
                puts.append({
                    "symbol": s.name,
                    "strike": float(s.option_strike),
                    "expiration_date": datetime.fromtimestamp(s.expiration_time),
                })

        if not puts:
            return pd.DataFrame()

        underlying_df = self._get_underlying_daily(stock)
        if underlying_df is None:
            return pd.DataFrame()

        und = underlying_df.copy()
        und["date"] = und["timestamp"].dt.date

        start_naive = start_date.tz_localize(None) if start_date.tz is not None else start_date
        end_naive = end_date.tz_localize(None) if end_date.tz is not None else end_date

        records: list[dict[str, Any]] = []
        seen_dates: set = set()

        # Sort puts by volume (desc) to prefer liquid options
        puts_sorted = sorted(puts, key=lambda p: -p["strike"])

        for put in puts_sorted:
            rates = mt5.copy_rates_from_pos(put["symbol"], mt5.TIMEFRAME_D1, 0, 5000)
            if rates is None or len(rates) < MIN_D1_BARS:
                continue
            df = pd.DataFrame(rates)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            df["date"] = df["timestamp"].dt.date
            df = df[(df["timestamp"] >= start_naive) & (df["timestamp"] <= end_naive)]
            if df.empty:
                continue

            for _, row in df.iterrows():
                d = row["date"]
                if d in seen_dates:
                    continue

                tte_years = (put["expiration_date"] - row["timestamp"]).days / 365.0
                if tte_years < DTE_MIN / 365.0 or tte_years > DTE_MAX / 365.0:
                    continue

                option_price = float(row["close"])
                if option_price <= 0:
                    continue

                underlying_row = und[und["date"] == d]
                if underlying_row.empty:
                    continue
                underlying_price = float(underlying_row.iloc[0]["close"])
                if underlying_price <= 0:
                    continue

                moneyness = put["strike"] / underlying_price
                if moneyness < 0.7 or moneyness > 1.3:
                    continue

                intrinsic = max(put["strike"] - underlying_price, 0)
                time_value = option_price - intrinsic
                if time_value < 0 or time_value > max(put["strike"], underlying_price) * 0.5:
                    continue
                if option_price > put["strike"]:
                    continue

                try:
                    iv = bs_iv(option_price, underlying_price, put["strike"], tte_years, RISK_FREE_RATE, "p")
                except Exception:
                    continue

                if iv is not None and 0 < iv < MAX_IV:
                    records.append({
                        "date": row["timestamp"],
                        "stock": stock,
                        "symbol": put["symbol"],
                        "iv": float(iv),
                        "strike": put["strike"],
                        "dte": int(tte_years * 365),
                        "option_price": option_price,
                        "underlying_price": underlying_price,
                        "moneyness": moneyness,
                    })
                    seen_dates.add(d)

            if len(records) > 0:
                break

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records)
        result = result.sort_values("date").reset_index(drop=True)
        return result

    def get_bid_ask_spread(self, option_symbol: str) -> tuple[float, float]:
        info = mt5.symbol_info(option_symbol)
        if info is None:
            return 0.0, 0.0
        return float(info.bid or 0.0), float(info.ask or 0.0)

    def disconnect(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False
