import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta
import numpy as np
import json
import time
import random

try:
    from bs4 import BeautifulSoup as BS4
    BS4OK = True
except ImportError:
    BS4OK = False

try:
    import twstock
    TWCODES = twstock.codes
    USETWSTOCK = True
except ImportError:
    TWCODES = {}
    USETWSTOCK = False


st.set_page_config(
    page_title="AI 精簡股票分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI 精簡股票分析系統")
st.divider()


GOODINFO_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8",
    "Referer": "https://goodinfo.tw",
}


def finmind_headers(token: str):
    return {"Authorization": f"Bearer {token}"}


@st.cache_data(ttl=3600)
def get_tw_stock_display_name(stock_code: str, finmind_token: str = "") -> dict:
    stock_code = stock_code.strip()
    result = {"name": stock_code, "source": "unknown", "industry": "", "market": ""}

    if finmind_token:
        try:
            resp = requests.get(
                "https://api.finmindtrade.com/api/v4/data",
                params={
                    "dataset": "TaiwanStockInfo",
                    "data_id": stock_code,
                    "token": finmind_token,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                j = resp.json()
                if j.get("status") == 200 and j.get("data"):
                    row = j["data"][0]
                    name = str(row.get("stock_name", "")).strip()
                    if name and name != stock_code:
                        result.update({
                            "name": name,
                            "source": "finmind",
                            "industry": row.get("industry_category", "") or "",
                            "market": row.get("type", "") or "",
                        })
                        return result
        except Exception:
            pass

    if USETWSTOCK:
        try:
            info = TWCODES.get(stock_code)
            if info:
                name = getattr(info, "name", "") or ""
                if name and name != stock_code:
                    result.update({
                        "name": name,
                        "source": "twstock",
                        "industry": getattr(info, "group", "") or "",
                        "market": getattr(info, "market", "") or "",
                    })
        except Exception:
            pass

    return result


@st.cache_data(ttl=1800)
def get_tw_stock_price(symbol: str, apikey: str, start_date, end_date):
    try:
        resp = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={
                "dataset": "TaiwanStockPrice",
                "data_id": symbol,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "token": apikey,
            },
            timeout=20,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("status") != 200 or not result.get("data"):
            return None
        df = pd.DataFrame(result["data"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"max": "high", "min": "low", "TradingVolume": "volume"})
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        return df[["date", "open", "high", "low", "close", "volume"]]
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_us_stock_price(symbol: str, apikey: str, start_date, end_date):
    try:
        url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
        resp = requests.get(
            url,
            params={
                "symbol": symbol,
                "apikey": apikey,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_analyst_targets(symbol: str, apikey: str):
    try:
        resp = requests.get(
            "https://financialmodelingprep.com/stable/price-target",
            params={"symbol": symbol, "apikey": apikey, "limit": 20},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return pd.DataFrame(data)
    except Exception:
        pass

    try:
        resp = requests.get(
            "https://financialmodelingprep.com/stable/price-target-consensus",
            params={"symbol": symbol, "apikey": apikey},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def calc_basic_stats(df: pd.DataFrame):
    if df is None or df.empty:
        return {}
    close = df["close"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)
    out = {
        "latest_close": float(close.iloc[-1]),
        "change": float(close.iloc[-1] - close.iloc[0]),
        "change_pct": float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100) if close.iloc[0] else 0,
        "high_52w": float(df["high"].max()) if "high" in df.columns else float(close.max()),
        "low_52w": float(df["low"].min()) if "low" in df.columns else float(close.min()),
        "avg_volume": float(vol.mean()) if len(vol) else 0,
    }
    return out


def calculate_bull_signals(df: pd.DataFrame):
    if df is None or df.empty:
        return {"signals": [], "total_score": 0, "conclusion": "資料不足", "conclusion_level": "error"}

    signals = []
    score = 0

    def add(name, status, desc, pts):
        nonlocal score
        signals.append({"name": name, "status": status, "desc": desc, "score": pts})
        score += pts

    if "MA5" in df.columns and "MA20" in df.columns:
        if df["close"].iloc[-1] > df["MA5"].iloc[-1] > df["MA20"].iloc[-1]:
            add("均線多頭", "green", "MA5 > MA20", 15)
        elif df["close"].iloc[-1] > df["MA20"].iloc[-1]:
            add("均線偏多", "yellow", "站上 MA20", 8)
        else:
            add("均線偏弱", "red", "未站上 MA20", 0)
    else:
        add("均線", "red", "缺少均線", 0)

    if "RSI14" in df.columns:
        r = df["RSI14"].iloc[-1]
        if 50 <= r <= 70:
            add("RSI", "green", f"RSI={r:.1f}", 10)
        elif 40 <= r < 50:
            add("RSI", "yellow", f"RSI={r:.1f}", 5)
        else:
            add("RSI", "red", f"RSI={r:.1f}", 0)

    if "OBV" in df.columns:
        if len(df)
