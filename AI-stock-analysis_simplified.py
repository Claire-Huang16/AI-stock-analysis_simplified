import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta
import json
import numpy as np
import time
import random
try:
    from bs4 import BeautifulSoup as _BS4
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

# twstock：台股公司名稱本地查詢（T03 整合）
try:
    import twstock as _twstock
    _TW_CODES = _twstock.codes
    _USE_TWSTOCK = True
except ImportError:
    _TW_CODES = {}
    _USE_TWSTOCK = False

# 設置頁面配置
st.set_page_config(
    page_title="AI 股票趨勢分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主標題
st.title("📈 AI 股票趨勢分析系統")
st.divider()



def get_us_stock_data(symbol, api_key, start_date, end_date):
    try:
        url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
        params = {
            'symbol': symbol,
            'apikey': api_key,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            st.error(f"無法獲取股票 {symbol} 的數據，請檢查股票代碼是否正確。")
            return None
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"FMP API 請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"美股數據處理錯誤：{str(e)}")
        return None


# ─────────────────────────────────────────────
# 台股輔助函數
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _finmind_headers(token):
    """FinMind v4 正確驗證方式：Authorization Bearer header"""
    return {"Authorization": f"Bearer {token}"}


@st.cache_data(ttl=3600)
def get_tw_company_profile(stock_code):
    """台股公司名稱與產業（twstock 本地資料庫，不需網路）"""
    profile = {"stockCode": stock_code, "companyName": stock_code,
               "industry": "", "market": "台股"}
    if _USE_TWSTOCK:
        info = _TW_CODES.get(stock_code)
        if info:
            profile["companyName"] = info.name
            profile["industry"]    = getattr(info, "group", "")
            profile["market"]      = getattr(info, "market", "台股")
    return profile


@st.cache_data(ttl=1800)
def _goodinfo_get(url, params=None, timeout=20):
    """GET goodinfo，回傳 (html, error)，帶 1 秒延遲避免限速"""
    time.sleep(1)
    try:
        r = requests.get(url, params=params, headers=GOODINFO_HEADERS, timeout=timeout)
        r.encoding = "utf-8"
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        if not r.text.strip():
            return None, "回傳空頁面"
        return r.text, None
    except requests.exceptions.Timeout:
        return None, "請求逾時，請稍後再試"
    except requests.exceptions.ConnectionError:
        return None, "無法連線至 goodinfo.tw"
    except Exception as e:
        return None, str(e)


def get_tw_stock_price(symbol, api_key, start_date, end_date):
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            'dataset': 'TaiwanStockPrice',
            'data_id': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'token': api_key
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        result = response.json()
        if result.get('status') != 200:
            msg = result.get('msg', '未知錯誤')
            st.error(f"FinMind API 錯誤：{msg}")
            return None
        data = result.get('data', [])
        if not data:
            st.error(f"無法獲取台股 {symbol} 的數據，請確認代碼是否正確。")
            return None
        df = pd.DataFrame(data)
        required_cols = ['date', 'open', 'max', 'min', 'close', 'Trading_Volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"FinMind 回傳欄位缺失：{missing}")
            return None
        df = df.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'})
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"FinMind API 請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"台股數據處理錯誤：{str(e)}")
        return None


def get_tw_margin_trading(symbol, api_key, start_date, end_date):
    """
    融資融券餘額 — FinMind TaiwanStockMarginPurchaseShortSale

    FinMind 官方實際欄位名稱（v4 API）：
      MarginPurchaseTodayBalance  ← 融資餘額（張）★ 修正
      ShortSaleTodayBalance       ← 融券餘額（張）★ 修正
      MarginPurchaseBuy           ← 融資買進
      MarginPurchaseSell          ← 融資賣出
      ShortSaleBuy                ← 融券買進(回補)
      ShortSaleSell               ← 融券賣出
    統一重命名為 MarginPurchaseRemaining / ShortSaleRemaining 供後續顯示邏輯使用。
    """
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        _end   = datetime.now()
        _start = _end - timedelta(days=35)
        params = {
            'dataset':    'TaiwanStockMarginPurchaseShortSale',
            'data_id':    symbol,
            'start_date': _start.strftime('%Y-%m-%d'),
            'end_date':   _end.strftime('%Y-%m-%d'),
            'token':      api_key
        }
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        result = response.json()

        if result.get('status') != 200:
            return None
        if not result.get('data'):
            return None

        df = pd.DataFrame(result['data'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        col_lower = {c.lower(): c for c in df.columns}

        def _find(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
                if c.lower() in col_lower:
                    return col_lower[c.lower()]
            return None

        # ★ 修正：優先使用 FinMind 官方正確欄位名 TodayBalance
        margin_col = _find([
            'MarginPurchaseTodayBalance',   # FinMind v4 官方欄位 ★
            'MarginPurchaseRemaining',       # 相容舊命名
            'MarginPurchaseToday',
            'margin_purchase_remaining',
            'FundingRemaining',
        ])
        short_col = _find([
            'ShortSaleTodayBalance',         # FinMind v4 官方欄位 ★
            'ShortSaleRemaining',            # 相容舊命名
            'ShortSaleToday',
            'short_sale_remaining',
            'ShortRemaining',
        ])
        # 買進/賣出欄位（散戶籌碼比計算用）
        margin_buy_col  = _find(['MarginPurchaseBuy',  'marginpurchasebuy'])
        margin_sell_col = _find(['MarginPurchaseSell', 'marginpurchasesell'])
        short_buy_col   = _find(['ShortSaleBuy',       'shortsalebuy'])
        short_sell_col  = _find(['ShortSaleSell',      'shortsalesell'])

        if margin_col is None and short_col is None:
            return None

        # 統一重命名為固定欄位名，後續顯示邏輯不需改動
        rename_map = {}
        if margin_col and margin_col != 'MarginPurchaseRemaining':
            rename_map[margin_col] = 'MarginPurchaseRemaining'
        if short_col and short_col != 'ShortSaleRemaining':
            rename_map[short_col] = 'ShortSaleRemaining'
        if margin_buy_col  and margin_buy_col  != 'MarginPurchaseBuy':
            rename_map[margin_buy_col]  = 'MarginPurchaseBuy'
        if margin_sell_col and margin_sell_col != 'MarginPurchaseSell':
            rename_map[margin_sell_col] = 'MarginPurchaseSell'
        if short_buy_col   and short_buy_col   != 'ShortSaleBuy':
            rename_map[short_buy_col]   = 'ShortSaleBuy'
        if short_sell_col  and short_sell_col  != 'ShortSaleSell':
            rename_map[short_sell_col]  = 'ShortSaleSell'
        if rename_map:
            df = df.rename(columns=rename_map)

        for col in ['MarginPurchaseRemaining', 'ShortSaleRemaining',
                    'MarginPurchaseBuy', 'MarginPurchaseSell',
                    'ShortSaleBuy', 'ShortSaleSell']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception:
        return None


def get_tw_institutional(symbol, api_key, start_date, end_date):
    """
    三大法人買賣超 — FinMind TaiwanStockInstitutionalInvestorsBuySell
    修正：
    - 欄位名稱大小寫容錯（Buy/buy、Sell/sell、buy_volume 等）
    - 固定查近30天，不受K線日期範圍限制
    - 保留 buy_volume / sell_volume（千股）欄位供後續籌碼分析用
    """
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        # 固定查近30天（不受K線日期範圍影響）
        _end   = datetime.now()
        _start = _end - timedelta(days=35)   # 多抓5天保留交易日空缺
        params = {
            'dataset':    'TaiwanStockInstitutionalInvestorsBuySell',
            'data_id':    symbol,
            'start_date': _start.strftime('%Y-%m-%d'),
            'end_date':   _end.strftime('%Y-%m-%d'),
            'token':      api_key
        }
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        result = response.json()

        # FinMind 付費 token 失效或 status 非 200
        if result.get('status') != 200:
            return None
        if not result.get('data'):
            return None

        df = pd.DataFrame(result['data'])
        df['date'] = pd.to_datetime(df['date'])

        # ── 欄位名稱容錯（FinMind 不同版本欄位大小寫不一）──
        col_lower = {c.lower(): c for c in df.columns}

        def _find(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
                if c.lower() in col_lower:
                    return col_lower[c.lower()]
            return None

        buy_col  = _find(['buy', 'Buy', 'buy_volume',  'BuyVolume',  'buy_amount'])
        sell_col = _find(['sell','Sell','sell_volume', 'SellVolume', 'sell_amount'])

        if buy_col is None or sell_col is None:
            # 嘗試從所有數值欄中找（通常第3、4欄是buy/sell）
            num_cols = df.select_dtypes(include='number').columns.tolist()
            num_cols = [c for c in num_cols if 'date' not in c.lower()]
            if len(num_cols) >= 2:
                buy_col, sell_col = num_cols[0], num_cols[1]
            else:
                return None

        df = df.rename(columns={buy_col: 'buy', sell_col: 'sell'})
        df['buy']  = pd.to_numeric(df['buy'],  errors='coerce').fillna(0)
        df['sell'] = pd.to_numeric(df['sell'], errors='coerce').fillna(0)
        df['net']  = df['buy'] - df['sell']

        # 保留 name 欄（法人類別）
        name_col = _find(['name', 'Name', 'institutional_investors'])
        if name_col and name_col != 'name':
            df = df.rename(columns={name_col: 'name'})
        elif 'name' not in df.columns:
            df['name'] = 'Unknown'

        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception:
        return None


def get_tw_broker_trading(symbol, api_key, date_str=None):
    """
    獲取台股券商分點進出明細。

    ⚠️ 重要說明：
    - FinMind TaiwanStockTradingDailyReport 是「sponsor 付費方案」專屬功能
    - 免費 / backer 方案完全無法使用
    - 本函數改用 TWSE 官方網站 (bsr.twse.com.tw) 爬取，完全免費

    來源：TWSE 買賣日報表
      上市: https://bsr.twse.com.tw/bshtm/bsContent.aspx
      上櫃: https://bsr.tpex.org.tw/bshtm/bsContent.aspx

    Args:
        symbol   : 台股代碼，如 '2330'
        api_key  : FinMind API Token（若有 sponsor 方案則嘗試，否則直接用 TWSE）
        date_str : 查詢日期 'YYYY-MM-DD'，None=今天

    Returns:
        dict 或 None
        {
          'date'       : '2026-04-10',
          'buy_df'     : DataFrame (broker_name, buy, sell, net, ratio),
          'sell_df'    : DataFrame (broker_name, buy, sell, net, ratio),
          'total_buy'  : int,
          'total_sell' : int,
          'source'     : str,   # 資料來源說明
        }
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    twse_date = date_str.replace('-', '')  # YYYYMMDD

    # ── 嘗試 FinMind（僅 sponsor 方案有效）──
    if api_key and api_key.strip():
        try:
            url = "https://api.finmindtrade.com/api/v4/data"
            params = {
                'dataset': 'TaiwanStockTradingDailyReport',
                'data_id': symbol,
                'start_date': date_str,
                'end_date':   date_str,
                'token': api_key
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                res = resp.json()
                # status=40x 或 data 為空時跳過
                if res.get('status') == 200 and res.get('data'):
                    raw = pd.DataFrame(res['data'])
                    result = _parse_broker_df(raw, date_str, 'FinMind')
                    if result:
                        return result
        except Exception:
            pass

    # ── TWSE 上市股票爬取（免費）──
    # 上市代碼通常是 4 碼純數字，≤ 6 碼；上櫃多為 5 碼或含字母
    headers_web = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
        'Referer': 'https://bsr.twse.com.tw/bshtm/',
    }

    # 先試上市 (TWSE)
    for base_url, market_label in [
        ("https://bsr.twse.com.tw/bshtm/bsContent.aspx", "上市"),
        ("https://bsr.tpex.org.tw/bshtm/bsContent.aspx", "上櫃"),
    ]:
        try:
            url = f"{base_url}?v=t&BHID=&StockNo={symbol}&StartDate={twse_date}&EndDate={twse_date}"
            resp = requests.get(url, headers=headers_web, timeout=20)
            resp.raise_for_status()

            # TWSE 回 big5 編碼
            try:
                resp.encoding = 'big5'
                html_text = resp.text
            except Exception:
                html_text = resp.content.decode('big5', errors='replace')

            # 檢查是否有實際資料（無資料時頁面會很短或包含特定文字）
            if len(html_text) < 500:
                continue
            if '查無資料' in html_text or 'no data' in html_text.lower():
                continue

            # 解析 HTML 表格
            try:
                tables = pd.read_html(html_text, header=0, flavor='lxml')
            except Exception:
                try:
                    tables = pd.read_html(html_text, header=0)
                except Exception:
                    continue

            if not tables:
                continue

            # 找到包含券商買賣資料的表格（通常欄位含「買進」「賣出」）
            target_df = None
            for t in tables:
                col_str = ' '.join(str(c) for c in t.columns)
                if '買進' in col_str and '賣出' in col_str:
                    target_df = t
                    break
                # 有些版本用英文或無標頭，嘗試看資料列數
                if len(t) > 5 and len(t.columns) >= 4:
                    target_df = t
                    break

            if target_df is None or target_df.empty:
                continue

            result = _parse_broker_df(target_df, date_str, f'TWSE {market_label}')
            if result:
                return result

        except Exception:
            continue

    return None


def _parse_broker_df(raw_df, date_str, source_label):
    """
    通用解析函數：將各來源的原始 DataFrame 轉為標準格式
    """
    try:
        df = raw_df.copy()

        # 自動偵測欄位名稱
        col_map = {}
        for c in df.columns:
            cs = str(c).strip().replace('\u3000', '').replace(' ', '')
            if any(k in cs for k in ['券商', 'broker', 'Broker', '分點', '名稱', '證券商']):
                col_map[c] = 'broker_name'
            elif '買進' in cs or cs in ('buy', 'Buy'):
                col_map[c] = 'buy'
            elif '賣出' in cs or cs in ('sell', 'Sell'):
                col_map[c] = 'sell'

        if col_map:
            df = df.rename(columns=col_map)

        # 若沒偵測到，嘗試位置推斷（第0欄=名稱, 第1欄=買進, 第2欄=賣出）
        if 'buy' not in df.columns and len(df.columns) >= 3:
            cols = list(df.columns)
            df = df.rename(columns={cols[0]: 'broker_name', cols[1]: 'buy', cols[2]: 'sell'})

        if 'buy' not in df.columns or 'sell' not in df.columns:
            return None

        if 'broker_name' not in df.columns:
            df['broker_name'] = '未知券商'

        # 清理數值（移除千分位逗號）
        def clean_num(v):
            try:
                return int(str(v).replace(',', '').replace('，', '').strip())
            except Exception:
                return 0

        df['buy']  = df['buy'].apply(clean_num)
        df['sell'] = df['sell'].apply(clean_num)
        df['net']  = df['buy'] - df['sell']

        # 過濾合計/小計列
        df = df[~df['broker_name'].astype(str).str.contains(
            r'合計|小計|平均|Total|total|^\s*$', na=False, regex=True
        )]
        df = df[df['buy'] + df['sell'] > 0]  # 過濾零值列

        if df.empty:
            return None

        total_buy  = int(df['buy'].sum())
        total_sell = int(df['sell'].sum())
        total_vol  = total_buy + total_sell

        # 估成交比重（百分比）
        df['ratio'] = df.apply(
            lambda r: round((r['buy'] + r['sell']) / total_vol * 100, 2) if total_vol > 0 else 0.0,
            axis=1
        )

        buy_df  = df[df['net'] > 0].sort_values('net', ascending=False).head(20).reset_index(drop=True)
        sell_df = df[df['net'] < 0].copy()
        sell_df['net'] = sell_df['net'].abs()
        sell_df = sell_df.sort_values('net', ascending=False).head(20).reset_index(drop=True)

        if buy_df.empty and sell_df.empty:
            return None

        return {
            'date':       date_str,
            'buy_df':     buy_df[['broker_name', 'buy', 'sell', 'net', 'ratio']],
            'sell_df':    sell_df[['broker_name', 'buy', 'sell', 'net', 'ratio']],
            'total_buy':  total_buy,
            'total_sell': total_sell,
            'source':     source_label,
        }

    except Exception:
        return None



def get_analyst_targets(symbol, api_key):
    result = {'targets': None, 'consensus': None}
    try:
        url = "https://financialmodelingprep.com/stable/price-target"
        params = {'symbol': symbol, 'apikey': api_key, 'limit': 20}
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            keep = [c for c in ['publishedDate', 'analystCompany', 'analystName',
                                 'priceTarget', 'priceWhenPosted', 'newsTitle'] if c in df.columns]
            if keep:
                df = df[keep].copy()
                df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
                df = df.sort_values('publishedDate', ascending=False).head(20)
                result['targets'] = df
    except Exception:
        pass

    try:
        url2 = "https://financialmodelingprep.com/stable/price-target-consensus"
        params2 = {'symbol': symbol, 'apikey': api_key}
        resp2 = requests.get(url2, params=params2, timeout=20)
        resp2.raise_for_status()
        data2 = resp2.json()
        if data2 and isinstance(data2, list) and len(data2) > 0:
            result['consensus'] = data2[0]
        elif data2 and isinstance(data2, dict):
            result['consensus'] = data2
    except Exception:
        pass

    return result if (result['targets'] is not None or result['consensus'] is not None) else None


# ─────────────────────────────────────────────
# 法人目標價 — 額外爬蟲來源
# ─────────────────────────────────────────────

def get_finviz_targets(symbol):
    """
    美股 Finviz 法人目標價與評等爬蟲（無需 API Key，免費公開頁面）。
    URL: https://finviz.com/quote.ashx?t={symbol}
    回傳 (DataFrame, error_msg)
    DataFrame 欄位：券商／機構名稱 | 評等 | 評等異動 | 目標價 | 更新日期
    """
    if not _BS4_OK:
        return pd.DataFrame(), "bs4 未安裝，無法爬取 Finviz"
    url = f"https://finviz.com/quote.ashx?t={symbol.upper()}"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finviz.com/",
    }
    try:
        time.sleep(random.uniform(0.5, 1.2))
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return pd.DataFrame(), f"Finviz HTTP {resp.status_code}"
        soup = _BS4(resp.text, "html.parser")
        # 找 id="news-table" 上方的 analyst ratings 表格（class 含 ratings-outer-table 或 fullview-ratings-outer）
        table = soup.find("table", class_="js-table-ratings") or                 soup.find("table", {"id": "ratings_outer_table"}) or                 soup.find("table", class_="fullview-ratings-outer")
        if table is None:
            # 備用：找包含多個 Rating 欄位的表格
            for t in soup.find_all("table"):
                text = t.get_text()
                if "Price Target" in text and "Action" in text:
                    table = t
                    break
        if table is None:
            return pd.DataFrame(), "Finviz：找不到目標價表格（頁面結構可能已變更）"

        rows = table.find_all("tr")
        records = []
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) >= 4:
                records.append(cols[:5] if len(cols) >= 5 else cols + [''] * (5 - len(cols)))

        if not records:
            return pd.DataFrame(), "Finviz：解析結果為空"

        df = pd.DataFrame(records)
        # Finviz 欄位順序通常是：Date | Broker | Action | Rating | Target
        if df.shape[1] >= 5:
            df.columns = ['更新日期', '券商／機構名稱', '評等異動', '評等', '目標價'] +                          [f'col{i}' for i in range(5, df.shape[1])]
            df = df[['券商／機構名稱', '評等', '評等異動', '目標價', '更新日期']]
        elif df.shape[1] == 4:
            df.columns = ['更新日期', '券商／機構名稱', '評等', '目標價']
            df['評等異動'] = '—'
            df = df[['券商／機構名稱', '評等', '評等異動', '目標價', '更新日期']]
        else:
            return pd.DataFrame(), f"Finviz：欄位數不符（{df.shape[1]}欄）"

        df = df[df['券商／機構名稱'].str.strip() != ''].reset_index(drop=True)
        return df, None
    except requests.exceptions.Timeout:
        return pd.DataFrame(), "Finviz 請求逾時"
    except requests.exceptions.ConnectionError:
        return pd.DataFrame(), "無法連線至 finviz.com"
    except Exception as e:
        return pd.DataFrame(), str(e)


def get_anue_targets(symbol, stock_name=None):
    """
    台股鉅亨網法人評等爬蟲。
    搜尋 URL: https://news.cnyes.com/news/cat/tw_stock_target_price?keyword={keyword}
    回傳 (DataFrame, error_msg)
    DataFrame 欄位：券商／機構名稱 | 目標價 | 評等 | 更新日期 | 來源摘要
    """
    if not _BS4_OK:
        return pd.DataFrame(), "bs4 未安裝，無法爬取鉅亨網"
    keyword = f"{symbol} {stock_name}".strip() if stock_name else symbol
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "zh-TW,zh;q=0.9",
        "Referer": "https://news.cnyes.com/",
    }
    try:
        time.sleep(random.uniform(0.8, 1.5))
        # 鉅亨網新聞搜尋 API
        api_url = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock_target_price"
        params = {"limit": 30, "page": 1, "keyword": keyword}
        resp = requests.get(api_url, headers=headers, params=params, timeout=20)
        if resp.status_code != 200:
            # 備用：一般新聞搜尋
            api_url2 = f"https://api.cnyes.com/media/api/v1/newslist/search?keyword={keyword}&limit=20"
            resp = requests.get(api_url2, headers=headers, timeout=20)
            if resp.status_code != 200:
                return pd.DataFrame(), f"鉅亨網 HTTP {resp.status_code}"
        data = resp.json()
        items = (data.get("data", {}).get("items") or
                 data.get("items") or
                 data.get("data") or [])
        if not items or not isinstance(items, list):
            return pd.DataFrame(), "鉅亨網：無搜尋結果"

        records = []
        for item in items[:20]:
            title    = item.get("title", "")
            summary  = item.get("summary", item.get("content", ""))[:60]
            pub_at   = item.get("publishAt") or item.get("publish_at") or ""
            if pub_at:
                try:
                    pub_at = datetime.fromtimestamp(int(pub_at)).strftime("%Y-%m-%d")
                except Exception:
                    pub_at = str(pub_at)[:10]
            # 從標題擷取目標價（正則）
            import re
            price_match = re.search(r'目標(?:價|價位)[：:]\s*([NT$]*[\d,\.]+)', title)
            target_price = price_match.group(1) if price_match else "—"
            # 擷取評等
            rating_map = {"買進": "買進", "買入": "買進", "強烈買進": "強烈買進",
                          "中立": "中立", "持有": "持有", "賣出": "賣出", "減碼": "減碼",
                          "增加持股": "增持", "優於大盤": "優於大盤", "Outperform": "優於大盤"}
            rating = "—"
            for k, v in rating_map.items():
                if k in title:
                    rating = v
                    break
            # 擷取機構名稱（常見格式：「XXX證券」「XX投信」）
            inst_match = re.search(r'([^\s，,、]{2,6}(?:證券|投行|投信|投顧|銀行|Capital|Securities|Research))', title)
            institution = inst_match.group(1) if inst_match else "鉅亨網"
            records.append({
                '券商／機構名稱': institution,
                '目標價': target_price,
                '評等': rating,
                '更新日期': pub_at,
                '來源摘要': title[:40],
            })

        if not records:
            return pd.DataFrame(), "鉅亨網：無法解析目標價資料"
        df = pd.DataFrame(records)
        return df, None
    except requests.exceptions.Timeout:
        return pd.DataFrame(), "鉅亨網請求逾時"
    except requests.exceptions.ConnectionError:
        return pd.DataFrame(), "無法連線至鉅亨網"
    except Exception as e:
        return pd.DataFrame(), str(e)


def get_goodinfo_stock_rating(symbol):
    """
    GoodInfo 個股法人評等（StockRating.asp）爬蟲。
    回傳 (DataFrame, error_msg)
    DataFrame 欄位：券商／機構名稱 | 評等 | 目標價 | 更新日期
    """
    if not _BS4_OK:
        return pd.DataFrame(), "bs4 未安裝，無法爬取 GoodInfo"
    html, err = _goodinfo_get(
        "https://goodinfo.tw/tw/StockRating.asp",
        params={"STOCK_ID": symbol}
    )
    if err:
        return pd.DataFrame(), f"GoodInfo：{err}"
    try:
        soup = _BS4(html, "html.parser")
        # 找含有評等資料的主要表格
        target_div = (soup.find("div", id="divDetail") or
                      soup.find("div", id="divRating") or
                      soup.find("div", id="txtStockListData"))
        parse_html = target_div.prettify() if target_div else html
        dfs = pd.read_html(parse_html)
        dfs = [df for df in dfs if df.shape[0] > 1 and df.shape[1] >= 3]
        if not dfs:
            return pd.DataFrame(), "GoodInfo：找不到評等表格"
        df = dfs[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(str(c) for c in col if "Unnamed" not in str(c)).strip()
                          for col in df.columns]
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"GoodInfo 評等解析失敗：{str(e)}"


def get_analyst_targets_ai(symbol, openai_api_key, market='us', stock_name=None):
    """
    使用 OpenAI gpt-4o-mini + web_search_preview 搜尋近一個月法人目標價新聞。
    使用 Responses API（支援 web_search_preview 工具）。
    返回：dict { 'table': DataFrame or None, 'search_date': str }
    """
    import json

    today = datetime.now()
    search_date = today.strftime('%Y-%m-%d')
    yyyy_mm = today.strftime('%Y年%m月') if market == 'tw' else today.strftime('%Y-%m')

    if market == 'tw':
        name_part = f"{symbol} {stock_name}" if stock_name else symbol
        query = f"{name_part} 目標價 法人 {yyyy_mm}"
    else:
        query = f"{symbol} price target analyst {yyyy_mm}"

    system_msg = """你是一位專業的股票研究助理，負責從網路新聞中彙整法人目標價資訊。
請搜尋近一個月內各大券商、投行對指定股票的目標價報導，並以 JSON 格式回傳結構化資料。

回傳格式（僅回傳純 JSON，不含 markdown）：
{
  "targets": [
    {
      "institution": "券商/機構名稱",
      "target_price": "目標價（含幣別，如 NT$250 或 $185）",
      "rating": "評等（買入/中立/賣出/增持/買進 等，若無則填 N/A）",
      "date": "發布日期 YYYY-MM-DD（若不確定填 N/A）",
      "summary": "簡短說明依據（15字以內）"
    }
  ],
  "note": "若無資料則填 '近期無法人目標價資料'"
}

若找不到任何目標價資料，targets 請回傳空陣列 []。"""

    try:
        import openai
        client = openai.OpenAI(api_key=openai_api_key)

        # ── 方法 1：gpt-4o-search-preview（OpenAI 原生網路搜尋模型）──
        # 此模型內建即時網路搜尋，直接以 chat.completions 呼叫
        try:
            response = client.chat.completions.create(
                model="gpt-4o-search-preview",
                web_search_options={},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": f"請搜尋近一個月內的新聞，彙整以下查詢的法人目標價資料：{query}"}
                ],
                max_tokens=1500,
            )
            content = response.choices[0].message.content or ""

        except Exception:
            # ── 方法 2：Responses API + web_search_preview（相容舊版 SDK）──
            try:
                resp2 = client.responses.create(
                    model="gpt-4o-mini",
                    tools=[{"type": "web_search_preview"}],
                    input=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": f"請搜尋並彙整：{query}"}
                    ],
                    max_output_tokens=1500,
                )
                content = ""
                for block in resp2.output:
                    if hasattr(block, 'content'):
                        for c in block.content:
                            if hasattr(c, 'text'):
                                content += c.text
                    elif hasattr(block, 'text'):
                        content += block.text
            except Exception:
                # ── 方法 3：純 Chat Completions 降級（無即時搜尋）──
                resp3 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": (
                            f"根據你的訓練知識，請彙整 {query} 的近期法人目標價。"
                            f"若訓練資料截止後無法得知，請在 note 欄位說明，"
                            f"targets 回傳空陣列。"
                        )}
                    ],
                    max_tokens=1500,
                    temperature=0.2,
                )
                content = resp3.choices[0].message.content or ""

        # ── 解析 JSON ──────────────────────────────────────────
        content = content.strip()

        # 移除 markdown code fence
        if "```" in content:
            parts = content.split("```")
            for p in parts:
                p = p.strip().lstrip("json").lstrip("JSON").strip()
                if p.startswith("{"):
                    content = p
                    break

        # 若含前置說明文字，擷取第一個 { ... } 區塊
        if not content.startswith("{"):
            import re as _re
            m = _re.search(r'\{[\s\S]*\}', content)
            if m:
                content = m.group(0)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {'table': None, 'search_date': search_date,
                    'error': f'JSON 解析失敗，原始回應前200字：{content[:200]}'}

        targets = data.get("targets", [])

        if not targets:
            return {'table': None, 'search_date': search_date}

        df = pd.DataFrame(targets)
        # 統一欄位名稱（不論 AI 回傳英文或中文 key）
        col_map = {
            'institution': '券商／機構名稱',
            'target_price': '目標價',
            'rating': '評等',
            'date': '更新日期',
            'summary': '來源摘要',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        expected = ['券商／機構名稱', '目標價', '評等', '更新日期', '來源摘要']
        for col in expected:
            if col not in df.columns:
                df[col] = '—'
        df = df[expected]
        return {'table': df, 'search_date': search_date}

    except Exception as e:
        return {'table': None, 'search_date': search_date, 'error': str(e)}



# ─────────────────────────────────────────────
# 通用數據處理函數
# ─────────────────────────────────────────────

def filter_by_date_range(df, start_date, end_date):
    if df is None:
        return None
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    return df.loc[mask].copy().reset_index(drop=True)


def get_moving_averages(df):
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    for period in [5, 10, 20, 60]:
        df[f'MA{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
    return df


def calculate_rsi(df, period=14):
    try:
        if df is None or len(df) < period:
            df = df.copy() if df is not None else pd.DataFrame()
            df[f'RSI{period}'] = 50
            return df
        df = df.copy()
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        df[f'RSI{period}'] = rsi
        return df
    except Exception:
        if df is not None:
            df = df.copy()
            df[f'RSI{period}'] = 50
        return df


def calculate_advanced_indicators(df):
    if df is None or len(df) < 30:
        return df
    df = df.copy()

    # MACD (12, 26, 9)
    try:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_DIF']  = ema12 - ema26
        df['MACD_LINE'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = (df['MACD_DIF'] - df['MACD_LINE']) * 2
        dif_prev  = df['MACD_DIF'].shift(1)
        line_prev = df['MACD_LINE'].shift(1)
        df['MACD_GOLDEN'] = (dif_prev < line_prev) & (df['MACD_DIF'] > df['MACD_LINE'])
    except Exception:
        pass

    # 布林通道 BB (20, 2SD)
    try:
        df['BB_MID']   = df['close'].rolling(20).mean()
        bb_std         = df['close'].rolling(20).std()
        df['BB_UPPER'] = df['BB_MID'] + 2 * bb_std
        df['BB_LOWER'] = df['BB_MID'] - 2 * bb_std
        df['BB_WIDTH'] = df['BB_UPPER'] - df['BB_LOWER']
    except Exception:
        pass

    # OBV
    try:
        direction = np.sign(df['close'].diff())
        direction.iloc[0] = 0
        df['OBV'] = (direction * df['volume']).cumsum()
    except Exception:
        pass

    # DMI (14日)
    try:
        high  = df['high']
        low   = df['low']
        close = df['close']
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low  - prev_close).abs()
        TR  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move   = high - prev_high
        down_move = prev_low - low
        plus_dm_arr  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm_arr = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_s  = pd.Series(plus_dm_arr,  index=df.index)
        minus_dm_s = pd.Series(minus_dm_arr, index=df.index)

        ATR14    = TR.ewm(com=13, adjust=False).mean()
        plus_di  = 100 * plus_dm_s.ewm(com=13, adjust=False).mean() / ATR14
        minus_di = 100 * minus_dm_s.ewm(com=13, adjust=False).mean() / ATR14
        DX       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        ADX      = DX.ewm(com=13, adjust=False).mean()

        df['DMI_PLUS']   = plus_di
        df['DMI_MINUS']  = minus_di
        df['DMI_ADX']    = ADX

        plus_prev  = df['DMI_PLUS'].shift(1)
        minus_prev = df['DMI_MINUS'].shift(1)
        df['DMI_GOLDEN'] = (plus_prev < minus_prev) & (df['DMI_PLUS'] > df['DMI_MINUS'])
    except Exception:
        pass

    # KD 隨機指標（Stochastic %K/%D，9日）
    try:
        period_kd = 9
        low_min  = df['low'].rolling(period_kd).min()
        high_max = df['high'].rolling(period_kd).max()
        rsv = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
        rsv = rsv.fillna(50)
        df['KD_K'] = rsv.ewm(com=2, adjust=False).mean()   # %K（3日EMA平滑）
        df['KD_D'] = df['KD_K'].ewm(com=2, adjust=False).mean()  # %D
        k_prev = df['KD_K'].shift(1)
        d_prev = df['KD_D'].shift(1)
        df['KD_GOLDEN'] = (k_prev < d_prev) & (df['KD_K'] > df['KD_D'])
        df['KD_DEAD']   = (k_prev > d_prev) & (df['KD_K'] < df['KD_D'])
    except Exception:
        pass

    # ATR（平均真實範圍，14日）— 供 BIAS / 波動度使用
    try:
        high  = df['high']
        low   = df['low']
        prev_close = df['close'].shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low  - prev_close).abs()
        TR  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR14'] = TR.ewm(com=13, adjust=False).mean()
    except Exception:
        pass

    # CCI（商品通道指標，20日）
    try:
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_ma  = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['CCI20'] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))
    except Exception:
        pass

    # MFI（資金流量指數，14日）
    try:
        tp  = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']
        pos_mf = rmf.where(tp > tp.shift(1), 0.0)
        neg_mf = rmf.where(tp < tp.shift(1), 0.0)
        pos_sum = pos_mf.rolling(14).sum()
        neg_sum = neg_mf.rolling(14).sum()
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        df['MFI14'] = 100 - (100 / (1 + mfr))
    except Exception:
        pass

    # Williams %R（14日）
    try:
        hw = df['high'].rolling(14).max()
        lw = df['low'].rolling(14).min()
        df['WILLR14'] = -100 * (hw - df['close']) / (hw - lw).replace(0, np.nan)
    except Exception:
        pass

    # Aroon（25日）
    try:
        aw = 25
        df['AROON_UP']   = df['high'].rolling(aw + 1).apply(
            lambda x: 100 * x.argmax() / aw, raw=True)
        df['AROON_DOWN'] = df['low'].rolling(aw + 1).apply(
            lambda x: 100 * x.argmin() / aw, raw=True)
        df['AROON_OSC']  = df['AROON_UP'] - df['AROON_DOWN']
    except Exception:
        pass

    # BIAS 乖離率（MA20）
    try:
        if 'MA20' not in df.columns:
            df['MA20'] = df['close'].rolling(20).mean()
        df['BIAS20'] = (df['close'] - df['MA20']) / df['MA20'] * 100
    except Exception:
        pass

    # VWAP（累積）
    try:
        tp  = (df['high'] + df['low'] + df['close']) / 3
        cum_vol = df['volume'].cumsum()
        cum_tpv = (tp * df['volume']).cumsum()
        df['VWAP'] = cum_tpv / cum_vol.replace(0, np.nan)
    except Exception:
        pass

    return df


def calculate_bull_signals(df):
    signals = []

    def add(name, status, desc):
        score = 12.5 if status == 'green' else (6.0 if status == 'yellow' else 0.0)
        signals.append({'name': name, 'status': status, 'desc': desc, 'score': score})

    # 1. MACD 轉正
    try:
        golden_recent = df['MACD_GOLDEN'].tail(5).any() if 'MACD_GOLDEN' in df.columns else False
        macd_pos = df['MACD_LINE'].iloc[-1] > 0 if 'MACD_LINE' in df.columns else False
        dif_pos  = df['MACD_DIF'].iloc[-1] > 0  if 'MACD_DIF'  in df.columns else False
        if golden_recent or (macd_pos and dif_pos):
            add('MACD 轉正', 'green', '最近5日金叉 或 MACD/DIF均>0')
        elif macd_pos:
            add('MACD 轉正', 'yellow', 'MACD線剛轉正，無金叉確認')
        else:
            add('MACD 轉正', 'red', 'MACD線<0，偏空')
    except Exception:
        add('MACD 轉正', 'red', '無法計算')

    # 2. BB 突破中軌
    try:
        if 'BB_MID' in df.columns:
            c   = df['close'].iloc[-1]
            mid = df['BB_MID'].iloc[-1]
            w1  = df['BB_WIDTH'].iloc[-1]
            w0  = df['BB_WIDTH'].iloc[-2] if len(df) > 1 else w1
            if c > mid and w1 > w0:
                add('BB 突破中軌', 'green', f'收盤>{mid:.2f}且通道擴張')
            elif c > mid:
                add('BB 突破中軌', 'yellow', f'收盤>{mid:.2f}，通道未擴張')
            else:
                add('BB 突破中軌', 'red', f'收盤<BB中軌({mid:.2f})')
        else:
            add('BB 突破中軌', 'red', '無法計算')
    except Exception:
        add('BB 突破中軌', 'red', '無法計算')

    # 3. BB 壓縮後突破
    try:
        if 'BB_WIDTH' in df.columns:
            w_mean = df['BB_WIDTH'].mean()
            compressed = df['BB_WIDTH'].tail(20).min() < w_mean * 0.5
            broken_up  = df['close'].iloc[-1] > df['BB_MID'].iloc[-1]
            if compressed and broken_up:
                add('BB 壓縮突破', 'green', '近期曾壓縮且已突破中軌')
            elif compressed:
                add('BB 壓縮突破', 'yellow', '通道壓縮中，尚未突破')
            else:
                add('BB 壓縮突破', 'red', '通道未壓縮')
        else:
            add('BB 壓縮突破', 'red', '無法計算')
    except Exception:
        add('BB 壓縮突破', 'red', '無法計算')

    # 4. OBV 資金流入
    try:
        if 'OBV' in df.columns:
            obv_last     = df['OBV'].iloc[-1]
            obv_prev_max = df['OBV'].iloc[:-1].max() if len(df) > 1 else obv_last
            new_high  = obv_last > obv_prev_max
            vol_surge = df['volume'].tail(5).mean() > df['volume'].mean() * 1.2
            if new_high and vol_surge:
                add('OBV 資金流入', 'green', 'OBV新高且近5日量放大>120%')
            elif new_high:
                add('OBV 資金流入', 'yellow', 'OBV新高，但量未明顯放大')
            else:
                add('OBV 資金流入', 'red', 'OBV未創新高')
        else:
            add('OBV 資金流入', 'red', '無法計算')
    except Exception:
        add('OBV 資金流入', 'red', '無法計算')

    # 5. RSI 動量
    try:
        rsi_col = next((c for c in df.columns if c.startswith('RSI')), None)
        if rsi_col:
            rv = df[rsi_col].iloc[-1]
            if 50 <= rv <= 70:
                add('RSI 動量', 'green', f'RSI={rv:.1f}，多頭動能區間')
            elif rv > 70:
                add('RSI 動量', 'yellow', f'RSI={rv:.1f}，超買警示')
            elif 40 <= rv < 50:
                add('RSI 動量', 'yellow', f'RSI={rv:.1f}，動能偏弱')
            else:
                add('RSI 動量', 'red', f'RSI={rv:.1f}，超賣區間')
        else:
            add('RSI 動量', 'red', '無法計算')
    except Exception:
        add('RSI 動量', 'red', '無法計算')

    # 6. DMI 多頭趨勢
    try:
        if 'DMI_PLUS' in df.columns:
            pv = df['DMI_PLUS'].iloc[-1]
            mv = df['DMI_MINUS'].iloc[-1]
            av = df['DMI_ADX'].iloc[-1]
            if pv > mv and av > 25:
                add('DMI 多頭趨勢', 'green', f'+DI({pv:.1f})>-DI({mv:.1f})，ADX={av:.1f}>25')
            elif pv > mv:
                add('DMI 多頭趨勢', 'yellow', f'+DI>{mv:.1f}，但ADX={av:.1f}≤25')
            else:
                add('DMI 多頭趨勢', 'red', f'-DI({mv:.1f})≥+DI({pv:.1f})，偏空')
        else:
            add('DMI 多頭趨勢', 'red', '無法計算')
    except Exception:
        add('DMI 多頭趨勢', 'red', '無法計算')

    # 7. DMI 黃金交叉
    try:
        if 'DMI_GOLDEN' in df.columns:
            if df['DMI_GOLDEN'].tail(5).any():
                add('DMI 黃金交叉', 'green', '近5日+DI上穿-DI，交叉確認')
            elif 'DMI_PLUS' in df.columns and df['DMI_PLUS'].iloc[-1] > df['DMI_MINUS'].iloc[-1]:
                add('DMI 黃金交叉', 'yellow', '+DI>-DI，但無近期交叉')
            else:
                add('DMI 黃金交叉', 'red', '無黃金交叉，-DI主導')
        else:
            add('DMI 黃金交叉', 'red', '無法計算')
    except Exception:
        add('DMI 黃金交叉', 'red', '無法計算')

    # 8. 均線多頭排列
    try:
        ma_cols = ['MA5', 'MA10', 'MA20', 'MA60']
        if all(c in df.columns for c in ma_cols):
            v5, v10, v20, v60 = [df[c].iloc[-1] for c in ma_cols]
            if v5 > v10 > v20 > v60:
                add('均線多頭排列', 'green', 'MA5>MA10>MA20>MA60 完全多頭')
            elif v5 > v20:
                add('均線多頭排列', 'yellow', 'MA5>MA20，部分多頭')
            else:
                add('均線多頭排列', 'red', '均線未呈多頭排列')
        else:
            add('均線多頭排列', 'red', '均線資料不足')
    except Exception:
        add('均線多頭排列', 'red', '無法計算')

    # 9. KD 金叉
    try:
        if 'KD_K' in df.columns and 'KD_D' in df.columns:
            kv = df['KD_K'].iloc[-1]
            dv = df['KD_D'].iloc[-1]
            golden_kd = df['KD_GOLDEN'].tail(5).any() if 'KD_GOLDEN' in df.columns else False
            if golden_kd and kv < 80:
                add('KD 金叉', 'green', f'近5日K線金叉，K={kv:.1f}，D={dv:.1f}')
            elif kv > dv and kv < 80:
                add('KD 金叉', 'yellow', f'K({kv:.1f})>D({dv:.1f})，無近期金叉')
            elif kv >= 80:
                add('KD 金叉', 'yellow', f'KD高檔({kv:.1f})，超買警示')
            else:
                add('KD 金叉', 'red', f'K({kv:.1f})<D({dv:.1f})，偏空')
        else:
            add('KD 金叉', 'red', '無法計算')
    except Exception:
        add('KD 金叉', 'red', '無法計算')

    # 10. MFI 資金流入
    try:
        if 'MFI14' in df.columns:
            mv = df['MFI14'].iloc[-1]
            if 50 <= mv <= 80:
                add('MFI 資金強度', 'green', f'MFI={mv:.1f}，資金健康流入')
            elif mv > 80:
                add('MFI 資金強度', 'yellow', f'MFI={mv:.1f}，超買注意')
            elif 30 <= mv < 50:
                add('MFI 資金強度', 'yellow', f'MFI={mv:.1f}，資金偏弱')
            else:
                add('MFI 資金強度', 'red', f'MFI={mv:.1f}，資金明顯流出')
        else:
            add('MFI 資金強度', 'red', '無法計算')
    except Exception:
        add('MFI 資金強度', 'red', '無法計算')

    # 11. CCI 趨勢動能
    try:
        if 'CCI20' in df.columns:
            cv    = df['CCI20'].iloc[-1]
            cv_p  = df['CCI20'].iloc[-2] if len(df) > 1 else cv
            if cv > 100:
                add('CCI 動能', 'green', f'CCI={cv:.1f}，強勢突破區間')
            elif 0 < cv <= 100 and cv > cv_p:
                add('CCI 動能', 'green', f'CCI={cv:.1f}，由負轉正且上升')
            elif -100 <= cv <= 0:
                add('CCI 動能', 'yellow', f'CCI={cv:.1f}，弱勢整理中')
            else:
                add('CCI 動能', 'red', f'CCI={cv:.1f}，超賣偏空')
        else:
            add('CCI 動能', 'red', '無法計算')
    except Exception:
        add('CCI 動能', 'red', '無法計算')

    # 12. BIAS 乖離率
    try:
        if 'BIAS20' in df.columns:
            bv = df['BIAS20'].iloc[-1]
            if 0 < bv <= 6:
                add('BIAS 乖離率', 'green', f'BIAS={bv:.2f}%，正乖離適中')
            elif bv > 6:
                add('BIAS 乖離率', 'yellow', f'BIAS={bv:.2f}%，正乖離過大，回調風險')
            elif -3 <= bv <= 0:
                add('BIAS 乖離率', 'yellow', f'BIAS={bv:.2f}%，輕微負乖離')
            else:
                add('BIAS 乖離率', 'red', f'BIAS={bv:.2f}%，負乖離過深')
        else:
            add('BIAS 乖離率', 'red', '無法計算')
    except Exception:
        add('BIAS 乖離率', 'red', '無法計算')

    n_signals   = len(signals)
    total_score = sum(s['score'] for s in signals)
    # 歸一化到100分（不論訊號總數）
    max_score   = n_signals * 12.5
    total_score_norm = round(total_score / max_score * 100) if max_score > 0 else 0
    green_count = sum(1 for s in signals if s['status'] == 'green')

    if total_score_norm >= 70:
        conclusion = f"🟢 多頭訊號確認（{green_count}/{n_signals}項綠燈）—— 技術面偏多，條件符合"
        conclusion_level = 'success'
    elif total_score_norm >= 40:
        conclusion = f"🟡 訊號混合（{green_count}/{n_signals}項綠燈）—— 部分多頭條件成立，需審慎觀察"
        conclusion_level = 'warning'
    else:
        conclusion = f"🔴 條件不符（{green_count}/{n_signals}項綠燈）—— 多頭訊號不足，技術面偏弱"
        conclusion_level = 'error'

    return {
        'signals':           signals,
        'total_score':       total_score_norm,
        'conclusion':        conclusion,
        'conclusion_level':  conclusion_level
    }




# ─────────────────────────────────────────────────────────────────────────────
# 朱家泓趨勢線系統 — 多頭確認 / 趨勢轉換 / 進出場判斷
# 核心規則來源：《趨勢線》教材（轉折波 + 多空確認 + 盤整突破）
# ─────────────────────────────────────────────────────────────────────────────

def calculate_zhu_trend_system(df):
    """
    實作朱家泓趨勢線系統，產生 0–100 分的趨勢評分與進出場建議。

    評分邏輯（共 8 大項，每項滿分依重要性加權）：
      1. 均線排列結構       15分
      2. 正 / 負價區         12分
      3. 短線轉折波方向      15分（頭頭高底底高 / 頭頭低底底低）
      4. 多頭 / 空頭確認     18分（突破前高 / 跌破前低）
      5. 趨勢轉換偵測        10分（空轉多 / 多轉空）
      6. 盤整突破訊號        12分（橫盤後突破上緣）
      7. 成交量確認           8分（突破量 > 均量）
      8. 月線方向             10分（長線多空）

    總分 ≥ 50 → 進場（做多），< 50 → 出場 / 觀望。
    """

    result = {
        'score': 0,
        'max_score': 100,
        'items': [],         # [{name, score, max, status, desc}]
        'trend': 'neutral',  # 'bull' / 'bear' / 'neutral'
        'transition': None,  # '空轉多' / '多轉空' / None
        'action': 'hold',    # 'enter_long' / 'exit' / 'hold'
        'action_color': 'yellow',
        'action_label': '⚪ 觀望',
        'consolidation_breakout': False,
        'detail': {},
    }

    def add_item(name, score, max_score, status, desc):
        result['items'].append({
            'name': name, 'score': score, 'max': max_score,
            'status': status, 'desc': desc
        })
        result['score'] += score

    close  = df['close']
    high   = df['high']
    low    = df['low']
    n      = len(df)

    if n < 30:
        result['action'] = 'hold'
        result['action_label'] = '⚪ 資料不足'
        return result

    c     = close.iloc[-1]
    c_arr = close.values

    # ─── 1. 均線排列結構（15分）───────────────────────────────
    try:
        ma5  = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1] if n >= 60 else None

        if ma60 is not None:
            if c > ma5 > ma10 > ma20 > ma60:
                add_item('均線排列', 15, 15, 'green', '完全多頭排列 MA5>MA10>MA20>MA60')
            elif c > ma5 > ma20:
                add_item('均線排列', 8, 15, 'yellow', 'MA5>MA20，部分多頭，MA60未確認')
            elif c < ma5 < ma10 < ma20:
                add_item('均線排列', 0, 15, 'red', '完全空頭排列，均線壓制')
            else:
                add_item('均線排列', 4, 15, 'yellow', '均線交錯，趨勢不明')
        else:
            if c > ma5 > ma10 > ma20:
                add_item('均線排列', 12, 15, 'green', '短中期多頭排列（MA60資料不足）')
            elif c > ma20:
                add_item('均線排列', 6, 15, 'yellow', '股價在MA20上方')
            else:
                add_item('均線排列', 0, 15, 'red', '股價在MA20下方')
        result['detail']['ma5']  = ma5
        result['detail']['ma10'] = ma10
        result['detail']['ma20'] = ma20
        result['detail']['ma60'] = ma60
    except Exception:
        add_item('均線排列', 0, 15, 'red', '無法計算')

    # ─── 2. 正 / 負價區（5日均線上下）（12分）────────────────
    try:
        ma5_s = close.rolling(5).mean()
        above_ma5 = (close > ma5_s).tail(3).sum()   # 近3日在5均上方天數
        if above_ma5 == 3:
            add_item('正/負價區', 12, 12, 'green', '近3日全在5均上方（正價區）')
        elif above_ma5 >= 2:
            add_item('正/負價區', 7,  12, 'yellow', f'近3日{above_ma5}日在5均上方，正價區偏弱')
        elif above_ma5 == 1:
            add_item('正/負價區', 3,  12, 'yellow', '剛突破5均，確認中')
        else:
            add_item('正/負價區', 0,  12, 'red',    '近3日全在5均下方（負價區）')
        result['detail']['above_ma5_days'] = int(above_ma5)
    except Exception:
        add_item('正/負價區', 0, 12, 'red', '無法計算')

    # ─── 3. 短線轉折波方向（頭頭高底底高 vs 頭頭低底底低）（15分）
    # 邏輯：取近20根K線，找出局部高低點，判斷趨勢方向
    try:
        window = min(40, n)
        h_arr  = high.values[-window:]
        l_arr  = low.values[-window:]
        c_sub  = c_arr[-window:]

        # 找局部高點（前後各2根比較）
        local_highs = []
        local_lows  = []
        for i in range(2, len(h_arr) - 2):
            if h_arr[i] >= max(h_arr[i-2:i]) and h_arr[i] >= max(h_arr[i+1:i+3]):
                local_highs.append((i, h_arr[i]))
            if l_arr[i] <= min(l_arr[i-2:i]) and l_arr[i] <= min(l_arr[i+1:i+3]):
                local_lows.append((i, l_arr[i]))

        # 取最近 3 個高低點判斷方向
        bull_wave = False
        bear_wave = False
        wave_desc = '轉折波方向不明'

        if len(local_highs) >= 2 and len(local_lows) >= 2:
            h_vals = [v for _, v in local_highs[-3:]]
            l_vals = [v for _, v in local_lows[-3:]]
            h_rising = all(h_vals[i] < h_vals[i+1] for i in range(len(h_vals)-1))
            l_rising = all(l_vals[i] < l_vals[i+1] for i in range(len(l_vals)-1))
            h_falling= all(h_vals[i] > h_vals[i+1] for i in range(len(h_vals)-1))
            l_falling= all(l_vals[i] > l_vals[i+1] for i in range(len(l_vals)-1))

            if h_rising and l_rising:
                bull_wave  = True
                wave_desc  = f'頭頭高({h_vals[-1]:.2f})底底高({l_vals[-1]:.2f})，多頭轉折波確認'
            elif h_falling and l_falling:
                bear_wave  = True
                wave_desc  = f'頭頭低({h_vals[-1]:.2f})底底低({l_vals[-1]:.2f})，空頭轉折波確認'
            elif l_rising:
                wave_desc  = f'底底高({l_vals[-1]:.2f})，下方支撐增強'
            elif h_falling:
                wave_desc  = f'頭頭低({h_vals[-1]:.2f})，上方壓力增強'

        result['detail']['bull_wave'] = bull_wave
        result['detail']['bear_wave'] = bear_wave

        if bull_wave:
            add_item('轉折波方向', 15, 15, 'green',  wave_desc)
        elif not bear_wave and not bull_wave:
            add_item('轉折波方向', 6,  15, 'yellow', wave_desc)
        else:
            add_item('轉折波方向', 0,  15, 'red',    wave_desc)
    except Exception:
        add_item('轉折波方向', 0, 15, 'red', '無法計算')

    # ─── 4. 多頭確認 / 空頭確認（18分）────────────────────────
    # 多頭確認：收盤突破近期高點（20日最高），且5均向上
    # 空頭確認：收盤跌破近期低點（20日最低），且5均向下
    try:
        look = min(20, n - 1)
        recent_high = high.iloc[-look-1:-1].max()
        recent_low  = low.iloc[-look-1:-1].min()
        ma5_dir     = close.rolling(5).mean().diff(3).iloc[-1]  # MA5 3日斜率

        bull_confirm = c > recent_high and ma5_dir > 0
        bear_confirm = c < recent_low  and ma5_dir < 0

        result['detail']['recent_high'] = recent_high
        result['detail']['recent_low']  = recent_low
        result['detail']['ma5_slope']   = ma5_dir

        if bull_confirm:
            add_item('多/空頭確認', 18, 18, 'green',
                     f'突破近{look}日高點{recent_high:.2f}，MA5向上 → 多頭確認')
        elif c > recent_high:
            add_item('多/空頭確認', 10, 18, 'yellow',
                     f'突破近期高點{recent_high:.2f}，但MA5尚未轉向')
        elif bear_confirm:
            add_item('多/空頭確認', 0,  18, 'red',
                     f'跌破近{look}日低點{recent_low:.2f}，MA5向下 → 空頭確認')
        elif c < recent_low:
            add_item('多/空頭確認', 0,  18, 'red',
                     f'跌破近期低點{recent_low:.2f}')
        else:
            add_item('多/空頭確認', 5,  18, 'yellow',
                     f'在近期區間內（高:{recent_high:.2f} 低:{recent_low:.2f}）')
    except Exception:
        add_item('多/空頭確認', 0, 18, 'red', '無法計算')

    # ─── 5. 趨勢轉換偵測（10分）────────────────────────────────
    # 空轉多：近5日有大量長紅K突破前高；多轉空：近5日有大量長黑K跌破前低
    try:
        avg_vol = df['volume'].mean()
        recent5 = df.tail(5)
        transition = None
        trans_score = 4
        trans_desc  = '無明顯趨勢轉換訊號'
        trans_status= 'yellow'

        for _, row in recent5.iterrows():
            body     = abs(row['close'] - row['open'])
            body_pct = body / row['open'] * 100 if row['open'] > 0 else 0
            vol_surge= row['volume'] > avg_vol * 1.5
            red_k    = row['close'] > row['open'] and body_pct > 2 and vol_surge
            black_k  = row['close'] < row['open'] and body_pct > 2 and vol_surge

            if red_k and row['close'] > recent_high:
                transition    = '空轉多'
                trans_score   = 10
                trans_desc    = f"大量長紅K突破前高{recent_high:.2f}，空轉多訊號"
                trans_status  = 'green'
                break
            elif black_k and row['close'] < recent_low:
                transition    = '多轉空'
                trans_score   = 0
                trans_desc    = f"大量長黑K跌破前低{recent_low:.2f}，多轉空訊號"
                trans_status  = 'red'
                break

        result['transition'] = transition
        add_item('趨勢轉換', trans_score, 10, trans_status, trans_desc)
    except Exception:
        add_item('趨勢轉換', 0, 10, 'red', '無法計算')

    # ─── 6. 盤整突破訊號（12分）───────────────────────────────
    # 判斷近10根K線是否為窄幅橫盤，且最新K線突破橫盤上緣
    try:
        consolidation_window = min(10, n - 1)
        recent10_high = high.iloc[-consolidation_window-1:-1]
        recent10_low  = low.iloc[-consolidation_window-1:-1]
        range_pct     = (recent10_high.max() - recent10_low.min()) / recent10_low.min() * 100

        is_consolidated = range_pct < 8.0    # 振幅 < 8% 視為橫盤
        breakout_up     = c > recent10_high.max()
        breakout_down   = c < recent10_low.min()

        result['consolidation_breakout'] = is_consolidated and breakout_up
        result['detail']['consolidation_range_pct'] = range_pct

        if is_consolidated and breakout_up:
            add_item('盤整突破', 12, 12, 'green',
                     f'近{consolidation_window}根振幅{range_pct:.1f}%盤整後向上突破')
        elif is_consolidated and breakout_down:
            add_item('盤整突破', 0,  12, 'red',
                     f'近{consolidation_window}根振幅{range_pct:.1f}%盤整後向下跌破')
        elif is_consolidated:
            add_item('盤整突破', 5,  12, 'yellow',
                     f'正處橫盤整理（振幅{range_pct:.1f}%），等待突破方向')
        else:
            add_item('盤整突破', 3,  12, 'yellow',
                     f'非橫盤狀態（振幅{range_pct:.1f}%），正常趨勢行進中')
    except Exception:
        add_item('盤整突破', 0, 12, 'red', '無法計算')

    # ─── 7. 成交量確認（8分）──────────────────────────────────
    try:
        avg_vol5  = df['volume'].tail(5).mean()
        avg_volN  = df['volume'].mean()
        vol_ratio = avg_vol5 / avg_volN if avg_volN > 0 else 1.0
        result['detail']['vol_ratio'] = vol_ratio

        if vol_ratio >= 1.5 and c > close.shift(1).iloc[-1]:
            add_item('成交量確認', 8, 8, 'green',
                     f'近5日均量是全期{vol_ratio:.1f}倍，量增價漲')
        elif vol_ratio >= 1.2 and c > close.shift(1).iloc[-1]:
            add_item('成交量確認', 5, 8, 'yellow',
                     f'量溫和放大{vol_ratio:.1f}倍，量價偏多')
        elif vol_ratio < 0.7:
            add_item('成交量確認', 2, 8, 'yellow',
                     f'量能萎縮（近5日均量僅全期{vol_ratio:.1f}倍）')
        elif vol_ratio >= 1.5 and c < close.shift(1).iloc[-1]:
            add_item('成交量確認', 0, 8, 'red',
                     f'量大價跌（{vol_ratio:.1f}倍量），賣壓沉重')
        else:
            add_item('成交量確認', 3, 8, 'yellow',
                     f'量能正常（{vol_ratio:.1f}倍）')
    except Exception:
        add_item('成交量確認', 0, 8, 'red', '無法計算')

    # ─── 8. 月線方向（長線多空基礎）（10分）─────────────────────
    try:
        if n >= 20:
            ma20_now  = close.rolling(20).mean().iloc[-1]
            ma20_prev = close.rolling(20).mean().iloc[-6]  # 6日前斜率
            ma20_slope= (ma20_now - ma20_prev) / ma20_prev * 100

            if ma20_slope > 1.0 and c > ma20_now:
                add_item('月線方向', 10, 10, 'green',
                         f'MA20向上斜率{ma20_slope:.1f}%，股價在月線上')
            elif ma20_slope > 0 and c > ma20_now:
                add_item('月線方向', 7,  10, 'green',
                         f'MA20緩升，股價在月線上')
            elif ma20_slope < -1.0 and c < ma20_now:
                add_item('月線方向', 0,  10, 'red',
                         f'MA20向下斜率{ma20_slope:.1f}%，股價在月線下')
            elif c < ma20_now:
                add_item('月線方向', 2,  10, 'red',
                         f'股價在月線下方（斜率{ma20_slope:.1f}%）')
            else:
                add_item('月線方向', 5,  10, 'yellow',
                         f'MA20持平（斜率{ma20_slope:.1f}%）')
        else:
            add_item('月線方向', 3, 10, 'yellow', '資料不足20日')
    except Exception:
        add_item('月線方向', 0, 10, 'red', '無法計算')

    # ─── 彙整趨勢方向 ───────────────────────────────────────────
    score = result['score']
    bull_wave  = result['detail'].get('bull_wave', False)
    bear_wave  = result['detail'].get('bear_wave', False)

    if score >= 70:
        result['trend'] = 'bull'
    elif score <= 35:
        result['trend'] = 'bear'
    else:
        result['trend'] = 'neutral'

    # ─── 進出場判斷 ─────────────────────────────────────────────
    if score >= 80:
        result['action']        = 'enter_long'
        result['action_color']  = 'green'
        result['action_label']  = '🟢 強烈進場（信心高）'
    elif score >= 65:
        result['action']        = 'enter_long'
        result['action_color']  = 'green'
        result['action_label']  = '🟢 進場（信心中高）'
    elif score >= 50:
        result['action']        = 'enter_long'
        result['action_color']  = 'yellow'
        result['action_label']  = '🟡 可考慮進場（信心中等，謹慎）'
    elif score >= 35:
        result['action']        = 'exit'
        result['action_color']  = 'orange'
        result['action_label']  = '🟠 考慮出場 / 減碼'
    else:
        result['action']        = 'exit'
        result['action_color']  = 'red'
        result['action_label']  = '🔴 出場 / 觀望（偏空）'

    # 趨勢轉換強制覆蓋
    if result['transition'] == '空轉多' and score >= 50:
        result['action_label'] = '🚀 ' + result['action_label'] + ' ＋空轉多確認加持'
    elif result['transition'] == '多轉空':
        result['action']       = 'exit'
        result['action_label'] = '🔴 出場（多轉空訊號，優先出場）'

    return result


def display_zhu_trend_dashboard(zhu_result, symbol, currency_symbol='$'):
    """
    顯示朱家泓趨勢線系統的分析儀表板。
    獨立於現有多頭訊號儀表板，專注於趨勢 + 進出場判斷。
    """
    st.markdown("---")
    st.markdown("### 📐 趨勢線系統分析")

    score     = zhu_result['score']
    max_score = zhu_result['max_score']
    action    = zhu_result['action_label']
    trend     = zhu_result['trend']
    items     = zhu_result['items']
    transition= zhu_result['transition']

    # ── 頂部：分數 + 進出場判斷 ──────────────────────────────────
    col_score, col_action = st.columns([1, 2])

    with col_score:
        # 顏色漸層：紅→橙→黃→綠
        if score >= 65:
            bar_color = '#2ed573'
        elif score >= 50:
            bar_color = '#f9ca24'
        elif score >= 35:
            bar_color = '#ff7f50'
        else:
            bar_color = '#ff4757'

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            border: 2px solid {bar_color};
        ">
            <div style="font-size:13px;color:#aaa;margin-bottom:4px;">趨勢系統評分</div>
            <div style="font-size:52px;font-weight:900;color:{bar_color};line-height:1;">
                {score}
            </div>
            <div style="font-size:13px;color:#777;">/ {max_score} 分</div>
            <div style="margin-top:10px;">
                <div style="background:#333;border-radius:6px;height:8px;">
                    <div style="background:{bar_color};width:{score}%;height:8px;border-radius:6px;"></div>
                </div>
            </div>
            <div style="margin-top:10px;font-size:12px;color:#aaa;">
                {"🐂 多頭趨勢" if trend=="bull" else ("🐻 空頭趨勢" if trend=="bear" else "⚖️ 中性整理")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_action:
        # 進出場建議卡片
        border_c = '#2ed573' if 'green' in zhu_result['action_color'] or score >= 65 else                    ('#f9ca24' if score >= 50 else '#ff4757')

        # 進出場門檻說明
        if score >= 50:
            signal_text = f"✅ 評分 {score} ≥ 50，達進場門檻"
            signal_color= '#2ed573'
        else:
            signal_text = f"⛔ 評分 {score} < 50，未達進場門檻"
            signal_color= '#ff4757'

        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-radius:16px;padding:20px;
            border:2px solid {border_c};height:100%;
        ">
            <div style="font-size:13px;color:#aaa;margin-bottom:8px;">📍 進出場判斷</div>
            <div style="font-size:22px;font-weight:700;color:{border_c};margin-bottom:12px;">
                {action}
            </div>
            <div style="font-size:13px;color:{signal_color};margin-bottom:8px;">
                {signal_text}
            </div>
            <div style="font-size:12px;color:#888;border-top:1px solid #333;padding-top:8px;">
                50分以上進場 ｜ 50分以下出場<br>
                分數越高信心越強，反之亦然
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 趨勢轉換特別提示 ────────────────────────────────────────
    if transition:
        color = '#2ed573' if transition == '空轉多' else '#ff4757'
        icon  = '🚀' if transition == '空轉多' else '⚠️'
        st.markdown(f"""
        <div style="margin:12px 0;padding:12px 18px;
            background:linear-gradient(90deg,{color}22,transparent);
            border-left:4px solid {color};border-radius:0 8px 8px 0;font-size:14px;color:{color};">
            {icon} <b>趨勢轉換偵測：{transition}</b>
            &nbsp;—&nbsp;{next((i['desc'] for i in items if i['name']=='趨勢轉換'),'')}</div>
        """, unsafe_allow_html=True)

    if zhu_result.get('consolidation_breakout'):
        st.markdown("""
        <div style="margin:6px 0;padding:10px 18px;
            background:linear-gradient(90deg,#a29bfe22,transparent);
            border-left:4px solid #a29bfe;border-radius:0 8px 8px 0;font-size:13px;color:#a29bfe;">
            💡 <b>盤整突破訊號</b> — 橫盤整理後向上突破，量能配合效果更佳</div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # ── 8 項細項評分表格（動態行，每排 4 格）──────────────────
    status_emoji = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
    for row_start in range(0, len(items), 4):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(items):
                itm = items[idx]
                pct = itm['score'] / itm['max'] * 100 if itm['max'] > 0 else 0
                with col:
                    st.metric(
                        label=f"{status_emoji[itm['status']]} {itm['name']}",
                        value=f"{itm['score']:.0f} / {itm['max']:.0f}",
                        delta=itm['desc'],
                        delta_color='normal'
                    )

    # ── 說明區塊（折疊）─────────────────────────────────────────
    with st.expander("📖 評分規則說明（朱家泓趨勢線系統）", expanded=False):
        st.markdown("""
**系統核心理念**（來源：朱家泓《趨勢線》教材）

| 項目 | 滿分 | 核心判斷邏輯 |
|---|---|---|
| 均線排列 | 15 | MA5>MA10>MA20>MA60 = 完全多頭排列 |
| 正/負價區 | 12 | 股價在5日均線上方（正價區）= 多頭有利 |
| 轉折波方向 | 15 | 頭頭高底底高 = 多頭；頭頭低底底低 = 空頭 |
| 多/空頭確認 | 18 | 突破近20日高點且MA5向上 = 多頭確認 |
| 趨勢轉換 | 10 | 大量長紅K突破前高 = 空轉多；反之 = 多轉空 |
| 盤整突破 | 12 | 振幅<8%橫盤後向上突破 = 盤整突破訊號 |
| 成交量確認 | 8 | 近5日均量 > 全期均量1.5倍且價漲 |
| 月線方向 | 10 | MA20向上且股價在月線上方 |

**進出場判斷**：
- 🟢 **≥ 65分**：強信心，積極進場做多
- 🟡 **50–64分**：中等信心，謹慎進場
- 🟠 **35–49分**：考慮出場 / 減碼
- 🔴 **< 35分**：偏空，觀望或出場

> ⚠️ 以上為技術面分析，非投資建議，投資有風險，請依個人財務狀況審慎評估。
        """)

    st.caption(f"📅 趨勢系統評分時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")


def display_bull_dashboard(bull_signals, symbol):
    st.markdown("### 🚦 多頭訊號儀表板")
    emoji_map = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
    signals = bull_signals['signals']

    # 動態排版：每排 4 欄，自動換行
    for row_start in range(0, len(signals), 4):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(signals):
                s = signals[idx]
                with col:
                    st.metric(
                        label=f"{emoji_map[s['status']]} {s['name']}",
                        value=f"{s['score']:.0f} 分",
                        delta=s['desc'],
                        delta_color="normal"
                    )

    st.markdown("---")
    sc, dc = st.columns([1, 3])
    n_sig = len(signals)
    with sc:
        st.metric("整體評分", f"{bull_signals['total_score']:.0f} / 100",
                  delta=f"{sum(1 for s in signals if s['status']=='green')}/{n_sig}項綠燈")
    with dc:
        lvl = bull_signals['conclusion_level']
        if lvl == 'success':
            st.success(bull_signals['conclusion'])
        elif lvl == 'warning':
            st.warning(bull_signals['conclusion'])
        else:
            st.error(bull_signals['conclusion'])
        st.caption("⚠️ 此評分基於近期歷史數據的技術面統計，不構成任何投資建議。歷史表現不代表未來結果。")


# ─────────────────────────────────────────────────────────────────────────────
# K線分析系統 — 朱家泓《多空操作秘笈》K線型態辨識
# 來源：朱家泓-K線.docx 第3章 3-1～3-6
#   3-1 K線起源與基本概念 / 高低點壓力支撐
#   3-2 K線高檔反轉的3大型態訊號（變盤線／覆蓋／貫穿／吞噬／母子懷抱／夜星）
#   3-3 K線低檔反轉的3大型態訊號（變盤線／覆蓋／貫穿／吞噬／母子懷抱／晨星）
#   3-4 不同位置的大量紅黑K判讀（高檔出貨 vs 低檔進場／槌子倒槌）
#   3-5 不同K線組合的意義（上升三法／下降三法／連3紅／連3黑）
#   3-6 趨勢軌道線突破與跌破
#
# 台股慣例：紅K＝收盤>開盤（上漲），黑K＝收盤<開盤（下跌）
# ─────────────────────────────────────────────────────────────────────────────

def _kline_basic(row):
    """單根K線基礎結構：實體、影線、紅黑判斷"""
    o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    body_pct = body / rng
    upper_shadow = (h - max(o, c)) / rng
    lower_shadow = (min(o, c) - l) / rng
    is_red = c > o          # 紅K（上漲）
    is_black = c < o        # 黑K（下跌）
    is_flat = c == o
    return {
        'o': o, 'h': h, 'l': l, 'c': c, 'rng': rng, 'body': body,
        'body_pct': body_pct, 'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow,
        'is_red': is_red, 'is_black': is_black, 'is_flat': is_flat,
    }


def _classify_single_kline(k, long_body_thr=0.6, small_body_thr=0.15, shadow_long_thr=0.5):
    """
    辨識單根K線型態（依朱家泓圖表3-2-1 / 3-3-1 高低檔變盤K線分類）：
    長紅線/長黑線、十字線、紡錘線、槌子線、吊人線、倒槌線、倒T線、
    墓碑線、長T線（蜻蜓十字）
    """
    bp = k['body_pct']
    us = k['upper_shadow']
    ls = k['lower_shadow']

    # 長紅/長黑：實體佔比高
    if bp >= long_body_thr:
        return '長紅線' if k['is_red'] else ('長黑線' if k['is_black'] else '長十字線')

    # 十字線族：實體極小
    if bp <= 0.05:
        if us >= shadow_long_thr and ls >= shadow_long_thr:
            return '十字線'
        if ls >= shadow_long_thr and us <= 0.1:
            return '長T線'      # 蜻蜓十字（下影線長，開收盤同高點）
        if us >= shadow_long_thr and ls <= 0.1:
            return '墓碑線'      # 墓碑十字（上影線長，開收盤同低點）
        return '一字線' if (us <= 0.05 and ls <= 0.05) else '十字線'

    # 小實體 + 長下影線：槌子／吊人（依位置由外部判斷高低檔意義）
    if bp <= small_body_thr * 2 and ls >= shadow_long_thr and us <= 0.15:
        return '槌子線'   # 實體紅黑不重要，下影線長

    # 小實體 + 長上影線：倒槌／流星
    if bp <= small_body_thr * 2 and us >= shadow_long_thr and ls <= 0.15:
        return '倒槌線'

    # 紡錘線：小實體 + 上下影線都不算太長
    if bp <= small_body_thr:
        return '紡錘線'

    return '一般K線'


def _detect_two_candle_pattern(k1, k2):
    """
    辨識兩根K線組合型態（圖表3-2-x／3-3-x）：
    母子懷抱、長黑覆蓋(烏雲罩頂)、長紅覆蓋、長黑貫穿、長紅貫穿、長黑吞噬、長紅吞噬
    k1 = 前一根（母線）, k2 = 最新一根（子線）
    """
    patterns = []

    # ── 母子懷抱：k1為中長紅/黑實體，k2完全包覆在k1實體內 ──
    if k1['body_pct'] >= 0.4:
        k1_top, k1_bot = max(k1['o'], k1['c']), min(k1['o'], k1['c'])
        k2_top, k2_bot = max(k2['o'], k2['c']), min(k2['o'], k2['c'])
        if k2_top <= k1_top and k2_bot >= k1_bot:
            if k1['is_red']:
                patterns.append('母子懷抱(高檔不懷好意)' if False else '母子懷抱')
            elif k1['is_black']:
                patterns.append('母子懷抱')

    # ── 長黑覆蓋 / 烏雲罩頂：前紅後黑，黑K開盤穿入或高於前紅K高點，收盤深入前紅K實體 ──
    if k1['is_red'] and k2['is_black'] and k1['body_pct'] >= 0.4:
        k1_top, k1_bot = max(k1['o'], k1['c']), min(k1['o'], k1['c'])
        penetration = (k1_top - k2['c']) / (k1_top - k1_bot + 1e-9)
        if k2['o'] >= k1['c'] and 0.3 <= penetration <= 1.0:
            patterns.append('長黑覆蓋(烏雲罩頂)')

    # ── 長紅覆蓋：前黑後紅，紅K開盤低於前黑K低點或收盤，深入前黑K實體 ──
    if k1['is_black'] and k2['is_red'] and k1['body_pct'] >= 0.4:
        k1_top, k1_bot = max(k1['o'], k1['c']), min(k1['o'], k1['c'])
        penetration = (k2['c'] - k1_bot) / (k1_top - k1_bot + 1e-9)
        if k2['o'] <= k1['c'] and 0.3 <= penetration <= 1.0:
            patterns.append('長紅覆蓋')

    # ── 長黑貫穿：前紅後黑，黑K收盤跌破前紅K低點（一次貫穿前低）──
    if k1['is_red'] and k2['is_black'] and k1['body_pct'] >= 0.4:
        if k2['c'] < min(k1['o'], k1['c']):
            patterns.append('長黑貫穿')

    # ── 長紅貫穿：前黑後紅，紅K收盤突破前黑K高點（一次貫穿前高）──
    if k1['is_black'] and k2['is_red'] and k1['body_pct'] >= 0.4:
        if k2['c'] > max(k1['o'], k1['c']):
            patterns.append('長紅貫穿')

    # ── 長黑吞噬：黑K實體完全吞噬前一根紅K實體（含開高收低）──
    if k1['is_red'] and k2['is_black']:
        if k2['o'] >= k1['c'] and k2['c'] <= k1['o']:
            patterns.append('長黑吞噬')

    # ── 長紅吞噬：紅K實體完全吞噬前一根黑K實體 ──
    if k1['is_black'] and k2['is_red']:
        if k2['o'] <= k1['c'] and k2['c'] >= k1['o']:
            patterns.append('長紅吞噬')

    return patterns


def _detect_three_candle_pattern(k1, k2, k3):
    """
    辨識三根K線組合（晨星／夜星）（圖表3-2-x／3-3-x）
    k1=第一根, k2=中間變盤K, k3=確認K
    """
    patterns = []
    k1_top, k1_bot = max(k1['o'], k1['c']), min(k1['o'], k1['c'])
    k3_top, k3_bot = max(k3['o'], k3['c']), min(k3['o'], k3['c'])
    mid = (k1_top + k1_bot) / 2

    # 夜星：高檔長紅 + 跳空小實體（變盤線）+ 長黑收破前紅K中點以下
    if k1['is_red'] and k1['body_pct'] >= 0.4 and k2['body_pct'] <= 0.35:
        gapped_up = min(k2['o'], k2['c']) >= k1['c'] * 0.998
        if k3['is_black'] and k3['c'] < mid and gapped_up:
            patterns.append('夜星轉折')

    # 晨星：低檔長黑 + 跳空小實體（變盤線）+ 長紅收過前黑K中點以上
    if k1['is_black'] and k1['body_pct'] >= 0.4 and k2['body_pct'] <= 0.35:
        gapped_dn = max(k2['o'], k2['c']) <= k1['c'] * 1.002
        if k3['is_red'] and k3['c'] > mid and gapped_dn:
            patterns.append('晨星轉折')

    return patterns


def _detect_three_soldiers_crows(k_list):
    """
    辨識連續3黑／連續3紅（圖表3-5-22～3-5-25）
    k_list = 最近3根K線（由舊到新）
    """
    if len(k_list) < 3:
        return None
    if all(k['is_black'] for k in k_list):
        closes = [k['c'] for k in k_list]
        if closes[0] > closes[1] > closes[2]:
            return '下跌連3黑'
    if all(k['is_red'] for k in k_list):
        closes = [k['c'] for k in k_list]
        if closes[0] < closes[1] < closes[2]:
            return '上漲連3紅'
    return None


def _detect_rising_falling_three_method(k_list):
    """
    辨識上升三法／下降三法（圖表3-5-1～3-5-15）
    需要5根K線：中長K + 2~3根反向小K（不破前低/不過前高）+ 同向中長K創新高/新低
    k_list：最近5根（舊到新）
    """
    if len(k_list) < 5:
        return None
    k0, k_mid, k4 = k_list[0], k_list[1:-1], k_list[-1]

    # 上升三法：k0長紅，中間1~3根小黑/小紅未跌破k0低點，k4長紅創新高（突破k0高點）
    if k0['is_red'] and k0['body_pct'] >= 0.4:
        k0_low = min(k0['o'], k0['c'])
        k0_high = max(k0['o'], k0['c'])
        mid_ok = all(min(km['o'], km['c']) >= k0_low for km in k_mid) and all(km['body_pct'] <= 0.5 for km in k_mid)
        if mid_ok and k4['is_red'] and k4['c'] > k0_high:
            return '上升三法'

    # 下降三法：k0長黑，中間小紅/小黑未突破k0高點，k4長黑創新低（跌破k0低點）
    if k0['is_black'] and k0['body_pct'] >= 0.4:
        k0_low = min(k0['o'], k0['c'])
        k0_high = max(k0['o'], k0['c'])
        mid_ok = all(max(km['o'], km['c']) <= k0_high for km in k_mid) and all(km['body_pct'] <= 0.5 for km in k_mid)
        if mid_ok and k4['is_black'] and k4['c'] < k0_low:
            return '下降三法'

    return None


def calculate_kline_pattern_system(df):
    """
    朱家泓 K線型態分析系統 — 產生 0–100 分的K線結構評分。
    來源：《多空操作秘笈》第3章 K線高低檔反轉型態 + 大量紅黑K判讀 + 趨勢軌道。

    評分邏輯（共 6 大項）：
      1. 單根K線變盤訊號        15分（高檔出現黑K變盤線 / 低檔出現紅K變盤線）
      2. 兩根K線反轉組合        20分（覆蓋／貫穿／吞噬／母子懷抱，依高低檔位置判斷多空）
      3. 三根K線晨星/夜星       15分
      4. 連續K線型態            15分（上升三法/下降三法/連3紅/連3黑）
      5. 大量紅黑K位置判讀      20分（依朱家泓3-4：高檔大量紅K減分／低檔大量紅K加分）
      6. 趨勢軌道位置           15分（股價相對近期高低點壓力支撐位置）

    總分 ≥ 60 → K線結構偏多；< 40 → K線結構偏空。
    """
    result = {
        'score': 0,
        'max_score': 100,
        'items': [],
        'kline_trend': 'neutral',     # 'bull' / 'bear' / 'neutral'
        'patterns_found': [],          # 命中的型態名稱列表
        'latest_kline_type': None,     # 最新一根K線分類
        'position_context': None,      # '高檔' / '低檔' / '盤整' / '行進中'
        'action_label': '⚪ 觀望',
        'detail': {},
    }

    def add_item(name, score, max_score, status, desc):
        result['items'].append({'name': name, 'score': score, 'max': max_score, 'status': status, 'desc': desc})
        result['score'] += score

    n = len(df)
    if n < 10:
        add_item('資料檢查', 0, 100, 'red', '資料不足，無法進行K線型態分析')
        result['action_label'] = '⚪ 資料不足'
        return result

    # 準備最近K線結構（最多取最近10根供型態判讀）
    tail_n = min(10, n)
    recent = df.tail(tail_n).reset_index(drop=True)
    klines = [_kline_basic(recent.iloc[i]) for i in range(tail_n)]

    latest = klines[-1]
    prev1  = klines[-2] if tail_n >= 2 else None
    prev2  = klines[-3] if tail_n >= 3 else None

    # ── 判斷目前位置：高檔 / 低檔 / 盤整 / 行進中（依近20日相對位置）──
    look = min(20, n - 1)
    recent_high = df['high'].iloc[-look-1:-1].max() if n > look else df['high'].max()
    recent_low  = df['low'].iloc[-look-1:-1].min()  if n > look else df['low'].min()
    c_now = latest['c']
    rel_pos = (c_now - recent_low) / (recent_high - recent_low + 1e-9)  # 0=低檔, 1=高檔

    if rel_pos >= 0.75:
        position = '高檔'
    elif rel_pos <= 0.25:
        position = '低檔'
    else:
        position = '行進中'
    result['position_context'] = position
    result['detail']['rel_pos'] = round(rel_pos, 2)
    result['detail']['recent_high'] = recent_high
    result['detail']['recent_low']  = recent_low

    # ─── 1. 單根K線變盤訊號（15分）─────────────────────────────
    try:
        k_type = _classify_single_kline(latest)
        result['latest_kline_type'] = k_type
        avg_vol = df['volume'].tail(20).mean() if 'volume' in df.columns else None
        vol_now = df['volume'].iloc[-1] if 'volume' in df.columns else None
        vol_surge = bool(avg_vol and vol_now and vol_now > avg_vol * 1.3)

        # 高檔變盤線（看跌）：墓碑線、十字線、倒槌線、長黑線、吊人線型態
        bearish_reversal_k = k_type in ('墓碑線', '十字線', '倒槌線', '長黑線')
        # 低檔變盤線（看漲）：長T線、十字線、槌子線、長紅線
        bullish_reversal_k = k_type in ('長T線', '十字線', '槌子線', '長紅線')

        if position == '高檔' and bearish_reversal_k:
            score = 3 if k_type == '長黑線' and not vol_surge else (15 if vol_surge else 9)
            add_item('單根K線變盤訊號', score, 15, 'red' if score <= 6 else 'yellow',
                     f"高檔出現「{k_type}」{'（爆量，反轉訊號強）' if vol_surge else '，留意反轉風險'}")
        elif position == '低檔' and bullish_reversal_k:
            score = 3 if k_type == '長紅線' and not vol_surge else (15 if vol_surge else 9)
            add_item('單根K線變盤訊號', score, 15, 'green' if score >= 9 else 'yellow',
                     f"低檔出現「{k_type}」{'（爆量，止跌訊號強）' if vol_surge else '，留意止跌反彈'}")
        elif k_type == '長紅線':
            add_item('單根K線變盤訊號', 10, 15, 'green', f"當前為「{k_type}」，多方力道強勁")
        elif k_type == '長黑線':
            add_item('單根K線變盤訊號', 2, 15, 'red', f"當前為「{k_type}」，空方力道強勁")
        else:
            add_item('單根K線變盤訊號', 7, 15, 'yellow', f"當前為「{k_type}」，位置：{position}，無明顯變盤訊號")
        result['detail']['vol_surge'] = vol_surge
    except Exception:
        add_item('單根K線變盤訊號', 0, 15, 'red', '無法計算')

    # ─── 2. 兩根K線反轉組合（20分）─────────────────────────────
    try:
        two_patterns = _detect_two_candle_pattern(prev1, latest) if prev1 else []
        result['patterns_found'].extend(two_patterns)

        bearish_2k = {'長黑覆蓋(烏雲罩頂)', '長黑貫穿', '長黑吞噬'}
        bullish_2k = {'長紅覆蓋', '長紅貫穿', '長紅吞噬'}
        harami_set = {'母子懷抱'}

        hit_bear = [p for p in two_patterns if p in bearish_2k]
        hit_bull = [p for p in two_patterns if p in bullish_2k]
        hit_harami = [p for p in two_patterns if p in harami_set]

        if position == '高檔' and hit_bear:
            add_item('兩根K線反轉組合', 2, 20, 'red',
                     f"高檔出現「{'、'.join(hit_bear)}」，空頭轉折訊號強，留意反轉向下")
        elif position == '低檔' and hit_bull:
            add_item('兩根K線反轉組合', 20, 20, 'green',
                     f"低檔出現「{'、'.join(hit_bull)}」，多頭轉折訊號強，留意止跌反彈")
        elif position == '高檔' and hit_harami:
            add_item('兩根K線反轉組合', 5, 20, 'yellow',
                     "高檔出現「母子懷抱」，多空力道收斂，上漲力道減弱，注意是否轉折向下")
        elif position == '低檔' and hit_harami:
            add_item('兩根K線反轉組合', 14, 20, 'green',
                     "低檔出現「母子懷抱」，下跌力道收斂，留意止跌訊號（光明在望）")
        elif hit_bull:
            add_item('兩根K線反轉組合', 14, 20, 'green', f"出現「{'、'.join(hit_bull)}」，偏多訊號")
        elif hit_bear:
            add_item('兩根K線反轉組合', 4, 20, 'red', f"出現「{'、'.join(hit_bear)}」，偏空訊號")
        else:
            add_item('兩根K線反轉組合', 10, 20, 'yellow', '近2根K線無明顯反轉組合')
    except Exception:
        add_item('兩根K線反轉組合', 0, 20, 'red', '無法計算')

    # ─── 3. 三根K線晨星/夜星（15分）────────────────────────────
    try:
        three_patterns = []
        if prev2 is not None:
            three_patterns = _detect_three_candle_pattern(prev2, prev1, latest)
        result['patterns_found'].extend(three_patterns)

        if '夜星轉折' in three_patterns:
            if position == '高檔':
                add_item('晨星/夜星型態', 2, 15, 'red', "高檔出現「夜星轉折」，強力反轉向下訊號")
            else:
                add_item('晨星/夜星型態', 5, 15, 'yellow', "出現「夜星轉折」，留意轉弱")
        elif '晨星轉折' in three_patterns:
            if position == '低檔':
                add_item('晨星/夜星型態', 15, 15, 'green', "低檔出現「晨星轉折」，強力止跌反彈訊號")
            else:
                add_item('晨星/夜星型態', 10, 15, 'green', "出現「晨星轉折」，偏多訊號")
        else:
            add_item('晨星/夜星型態', 8, 15, 'yellow', '近期無晨星/夜星型態')
    except Exception:
        add_item('晨星/夜星型態', 0, 15, 'red', '無法計算')

    # ─── 4. 連續K線型態（上升三法/下降三法/連3紅/連3黑）（15分）──
    try:
        cont_patterns = []
        three_soldiers = _detect_three_soldiers_crows(klines[-3:]) if tail_n >= 3 else None
        if three_soldiers:
            cont_patterns.append(three_soldiers)
        method5 = _detect_rising_falling_three_method(klines[-5:]) if tail_n >= 5 else None
        if method5:
            cont_patterns.append(method5)
        result['patterns_found'].extend(cont_patterns)

        if '上升三法' in cont_patterns:
            add_item('連續K線型態', 15, 15, 'green', "出現「上升三法」，多頭中繼，趨勢延續看漲")
        elif '下降三法' in cont_patterns:
            add_item('連續K線型態', 0, 15, 'red', "出現「下降三法」，空頭中繼，趨勢延續看跌")
        elif '上漲連3紅' in cont_patterns:
            if position == '高檔':
                add_item('連續K線型態', 6, 15, 'yellow', "高檔連3紅，留意過熱拉回")
            else:
                add_item('連續K線型態', 12, 15, 'green', "出現「上漲連3紅」，多方動能轉強")
        elif '下跌連3黑' in cont_patterns:
            if position == '低檔':
                add_item('連續K線型態', 9, 15, 'yellow', "低檔連3黑，留意止跌訊號（KD背離注意反彈）")
            else:
                add_item('連續K線型態', 2, 15, 'red', "出現「下跌連3黑」，空方動能轉強")
        else:
            add_item('連續K線型態', 7, 15, 'yellow', '近期無明顯連續K線型態')
    except Exception:
        add_item('連續K線型態', 0, 15, 'red', '無法計算')

    # ─── 5. 大量紅黑K位置判讀（圖表3-4，20分）──────────────────
    try:
        avg_vol20 = df['volume'].tail(20).mean() if 'volume' in df.columns else None
        vol_now   = df['volume'].iloc[-1] if 'volume' in df.columns else None
        big_vol   = bool(avg_vol20 and vol_now and vol_now > avg_vol20 * 1.5)
        is_red    = latest['is_red']
        is_black  = latest['is_black']
        long_body = latest['body_pct'] >= 0.4

        if big_vol and is_red and long_body:
            if position == '高檔':
                add_item('大量紅黑K位置判讀', 4, 20, 'red',
                         "高檔大量長紅K——可能為主力出貨訊號，依朱家泓3-4法則『日線高檔大量紅K不能買』")
            elif position == '低檔':
                add_item('大量紅黑K位置判讀', 20, 20, 'green',
                         "低檔大量長紅K——空轉多訊號，依朱家泓3-4法則『空頭轉多頭第一次過前高的大量長紅K可買』")
            else:
                add_item('大量紅黑K位置判讀', 14, 20, 'green',
                         "行進中大量長紅K，多頭續漲或盤整突破訊號，留意是否站穩")
        elif big_vol and is_black and long_body:
            if position == '高檔':
                add_item('大量紅黑K位置判讀', 0, 20, 'red',
                         "高檔大量長黑K——轉折向下訊號強，依朱家泓3-2法則應留意出場")
            elif position == '低檔':
                add_item('大量紅黑K位置判讀', 10, 20, 'yellow',
                         "低檔大量長黑K——留意是否為主力誘空假跌破，須觀察次日是否止跌")
            else:
                add_item('大量紅黑K位置判讀', 6, 20, 'yellow',
                         "行進中大量長黑K，留意回檔或轉折風險")
        elif big_vol:
            add_item('大量紅黑K位置判讀', 10, 20, 'yellow', f"近期爆量但K線非長紅/長黑，方向待確認（位置：{position}）")
        else:
            add_item('大量紅黑K位置判讀', 10, 20, 'yellow', f"近期成交量無明顯異常放大（位置：{position}）")

        result['detail']['big_vol'] = big_vol
    except Exception:
        add_item('大量紅黑K位置判讀', 0, 20, 'red', '無法計算')

    # ─── 6. 趨勢軌道位置（壓力支撐，圖表3-1/3-6，15分）─────────
    try:
        dist_to_high = (recent_high - c_now) / c_now * 100 if c_now > 0 else 0
        dist_to_low  = (c_now - recent_low) / c_now * 100 if c_now > 0 else 0
        broke_high   = c_now > recent_high
        broke_low    = c_now < recent_low

        if broke_high:
            add_item('趨勢軌道位置', 15, 15, 'green', f"收盤突破近{look}日高點{recent_high:.2f}，上方壓力解除")
        elif broke_low:
            add_item('趨勢軌道位置', 0, 15, 'red', f"收盤跌破近{look}日低點{recent_low:.2f}，下方支撐失守")
        elif dist_to_high <= 3:
            add_item('趨勢軌道位置', 6, 15, 'yellow', f"接近近期高點壓力（距{dist_to_high:.1f}%），留意是否突破或拉回")
        elif dist_to_low <= 3:
            add_item('趨勢軌道位置', 9, 15, 'yellow', f"接近近期低點支撐（距{dist_to_low:.1f}%），留意是否止跌或破底")
        else:
            add_item('趨勢軌道位置', 8, 15, 'yellow', f"位於壓力({recent_high:.2f})與支撐({recent_low:.2f})區間中段")
    except Exception:
        add_item('趨勢軌道位置', 0, 15, 'red', '無法計算')

    # ─── 彙整 ───────────────────────────────────────────────────
    score = result['score']
    if score >= 60:
        result['kline_trend'] = 'bull'
    elif score <= 40:
        result['kline_trend'] = 'bear'
    else:
        result['kline_trend'] = 'neutral'

    if score >= 75:
        result['action_label'] = '🟢 K線結構強多頭'
    elif score >= 60:
        result['action_label'] = '🟢 K線結構偏多'
    elif score >= 40:
        result['action_label'] = '🟡 K線結構中性'
    elif score >= 25:
        result['action_label'] = '🟠 K線結構偏空'
    else:
        result['action_label'] = '🔴 K線結構強空頭'

    result['patterns_found'] = list(dict.fromkeys(result['patterns_found']))  # 去重保序
    return result


def display_kline_pattern_dashboard(kline_result, symbol):
    """顯示K線型態分析系統儀表板，風格與朱家泓趨勢線系統儀表板一致。"""
    st.markdown("---")
    st.markdown("### 🕯️ K線型態分析系統")
    st.caption("依據朱家泓《多空操作秘笈》第3章 K線高低檔反轉型態 + 大量紅黑K判讀法則")

    score = kline_result['score']
    max_score = kline_result['max_score']
    trend = kline_result['kline_trend']
    items = kline_result['items']
    patterns = kline_result['patterns_found']
    position = kline_result['position_context']
    k_type = kline_result['latest_kline_type']

    col_score, col_action = st.columns([1, 2])

    with col_score:
        if score >= 60:
            bar_color = '#2ed573'
        elif score >= 40:
            bar_color = '#f9ca24'
        elif score >= 25:
            bar_color = '#ff7f50'
        else:
            bar_color = '#ff4757'

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            border: 2px solid {bar_color};
        ">
            <div style="font-size:13px;color:#aaa;margin-bottom:4px;">K線結構評分</div>
            <div style="font-size:52px;font-weight:900;color:{bar_color};line-height:1;">
                {score}
            </div>
            <div style="font-size:13px;color:#777;">/ {max_score} 分</div>
            <div style="margin-top:10px;">
                <div style="background:#333;border-radius:6px;height:8px;">
                    <div style="background:{bar_color};width:{score}%;height:8px;border-radius:6px;"></div>
                </div>
            </div>
            <div style="margin-top:10px;font-size:12px;color:#aaa;">
                {"🐂 K線偏多" if trend=="bull" else ("🐻 K線偏空" if trend=="bear" else "⚖️ K線中性")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_action:
        border_c = '#2ed573' if score >= 60 else ('#f9ca24' if score >= 40 else '#ff4757')
        pattern_str = '、'.join(patterns) if patterns else '無明顯命中型態'

        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-radius:16px;padding:20px;
            border:2px solid {border_c};height:100%;
        ">
            <div style="font-size:13px;color:#aaa;margin-bottom:8px;">📍 K線結構判斷</div>
            <div style="font-size:22px;font-weight:700;color:{border_c};margin-bottom:12px;">
                {kline_result['action_label']}
            </div>
            <div style="font-size:13px;color:#ddd;margin-bottom:6px;">
                目前位置：<b>{position}</b>　最新K線：<b>{k_type}</b>
            </div>
            <div style="font-size:13px;color:#a29bfe;border-top:1px solid #333;padding-top:8px;">
                命中型態：{pattern_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # ── 6 項細項評分（每排3格）──
    status_emoji = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
    for row_start in range(0, len(items), 3):
        cols = st.columns(3)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(items):
                itm = items[idx]
                with col:
                    st.metric(
                        label=f"{status_emoji[itm['status']]} {itm['name']}",
                        value=f"{itm['score']:.0f} / {itm['max']:.0f}",
                        delta=itm['desc'],
                        delta_color='normal'
                    )

    with st.expander("📖 評分規則說明（K線型態分析系統）", expanded=False):
        st.markdown("""
**系統核心理念**（來源：朱家泓《多空操作秘笈》第3章）

| 項目 | 滿分 | 核心判斷邏輯 |
|---|---|---|
| 單根K線變盤訊號 | 15 | 高檔出現墓碑線/十字線/倒槌/長黑＝看跌；低檔出現長T線/十字線/槌子/長紅＝看漲 |
| 兩根K線反轉組合 | 20 | 覆蓋(烏雲罩頂)／貫穿／吞噬／母子懷抱，依高低檔位置判斷多空意義 |
| 晨星/夜星型態 | 15 | 三根K線組合：低檔晨星＝強力止跌；高檔夜星＝強力反轉向下 |
| 連續K線型態 | 15 | 上升三法/下降三法為中繼；連3紅/連3黑為動能訊號 |
| 大量紅黑K位置判讀 | 20 | 高檔大量長紅K不能買（主力出貨）；低檔大量長紅K可買（空轉多） |
| 趨勢軌道位置 | 15 | 股價相對近期高低點的壓力支撐位置，突破/跌破即時加減分 |

**判斷邏輯**：
- 🟢 **≥ 75分**：K線結構強多頭，型態與位置同時偏多
- 🟢 **60–74分**：K線結構偏多
- 🟡 **40–59分**：K線結構中性，無明顯方向
- 🟠 **25–39分**：K線結構偏空
- 🔴 **< 25分**：K線結構強空頭，留意風險

⚠️ 本系統為型態統計判斷，僅供參考，不構成投資建議。型態訊號需搭配成交量與趨勢確認。
        """)


# ─────────────────────────────────────────────
# 圖表函數
# ─────────────────────────────────────────────

def create_candlestick_chart(df, symbol, rsi_period, currency_symbol,

                              institutional_df=None, market='us', selected_mas=None):
    """
    v3 主K線多層圖：K線+BB+可切換MA / RSI / OBV / 成交量 / 三大法人（台股）
    美股5層，台股6層
    """
    if selected_mas is None:
        selected_mas = ['MA5', 'MA10', 'MA20', 'MA60']

    show_inst = market == 'tw' and institutional_df is not None and len(institutional_df) > 0

    if show_inst:
        rows = 7
        row_heights = [0.32, 0.11, 0.11, 0.10, 0.12, 0.12, 0.12]
        subplot_titles = (
            f'{symbol} K線 + 布林通道 + MA + VWAP',
            f'RSI ({rsi_period}日)',
            'OBV 量能指標',
            '成交量',
            '三大法人買賣超（張）',
            'KD 隨機指標（9日）',
            '（預留）'
        )
        chart_height = 1350
    else:
        rows = 6
        row_heights = [0.35, 0.14, 0.13, 0.12, 0.13, 0.13]
        subplot_titles = (
            f'{symbol} K線 + 布林通道 + MA + VWAP',
            f'RSI ({rsi_period}日)',
            'OBV 量能指標',
            '成交量',
            'KD 隨機指標（9日）',
            '（預留）'
        )
        chart_height = 1250

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )

    # ── Row 1：布林通道填色 + BB三軌 + K線 + 可切換MA ──

    # 1a. BB 上下軌填色
    if 'BB_UPPER' in df.columns and 'BB_LOWER' in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df['date'], df['date'][::-1]]),
            y=pd.concat([df['BB_UPPER'], df['BB_LOWER'][::-1]]),
            fill='toself',
            fillcolor='rgba(100,149,237,0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False, name='BB 通道填色',
            hoverinfo='skip'
        ), row=1, col=1)

    # 1b. BB 三軌
    if 'BB_UPPER' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['BB_UPPER'],
            mode='lines', name='BB上軌',
            line=dict(color='#e74c3c', dash='dash', width=1.2)
        ), row=1, col=1)
    if 'BB_MID' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['BB_MID'],
            mode='lines', name='BB中軌',
            line=dict(color='#1e90ff', width=2)
        ), row=1, col=1)
    if 'BB_LOWER' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['BB_LOWER'],
            mode='lines', name='BB下軌',
            line=dict(color='#2ecc71', dash='dash', width=1.2)
        ), row=1, col=1)

    # 1c. BB 壓縮期橙色矩形
    if 'BB_WIDTH' in df.columns:
        w_mean = df['BB_WIDTH'].mean()
        compressed = df[df['BB_WIDTH'] < w_mean * 0.5].copy()
        if len(compressed) > 0:
            compressed['gap'] = (compressed.index.to_series().diff() > 1)
            block_id = compressed['gap'].cumsum()
            for _, block in compressed.groupby(block_id):
                x0 = block['date'].iloc[0]
                x1 = block['date'].iloc[-1]
                fig.add_vrect(x0=x0, x1=x1,
                              fillcolor='rgba(255,165,0,0.12)',
                              line_width=0, row=1, col=1)

    # 1d. K線
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='K線圖',
        increasing_line_color='#ff4757',
        decreasing_line_color='#2ed573',
        increasing_fillcolor='#ff4757',
        decreasing_fillcolor='#2ed573'
    ), row=1, col=1)

    # 1e. 可切換MA
    ma_colors = {'MA5': '#ff6b6b', 'MA10': '#4ecdc4', 'MA20': '#45b7d1', 'MA60': '#96ceb4'}
    for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
        if ma in selected_mas and ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df[ma],
                mode='lines', name=ma,
                line=dict(color=ma_colors[ma], width=2)
            ), row=1, col=1)

    # 1f. VWAP（累積，灰色虛線）
    if 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['VWAP'],
            mode='lines', name='VWAP',
            line=dict(color='#f9ca24', width=1.5, dash='dot'),
            visible='legendonly'
        ), row=1, col=1)

    # ── Row 2：RSI ──
    rsi_col = f'RSI{rsi_period}'
    if rsi_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[rsi_col],
            mode='lines', name=f'RSI({rsi_period})',
            line=dict(color='#1e90ff', width=2)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[df['date'].iloc[0], df['date'].iloc[-1]], y=[70, 70],
            mode='lines', name='超買(70)',
            line=dict(color='red', dash='dash', width=1), showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[df['date'].iloc[0], df['date'].iloc[-1]], y=[30, 30],
            mode='lines', name='超賣(30)',
            line=dict(color='green', dash='dash', width=1), showlegend=False
        ), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,71,87,0.08)', line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor='rgba(46,213,115,0.08)', line_width=0, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    # ── Row 3：OBV ──
    if 'OBV' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['OBV'],
            mode='lines', name='OBV',
            line=dict(color='#9b59b6', width=2)
        ), row=3, col=1)

    # ── Row 4：成交量 ──
    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'],
        name='成交量',
        marker_color='#a55eea', opacity=0.6
    ), row=4, col=1)

    # ── Row 5（台股）：三大法人買賣超 ──
    if show_inst:
        inst_colors = {
            'Foreign_Investor': '#e74c3c',
            'Investment_Trust': '#3498db',
            'Dealer_self': '#2ecc71',
            'Dealer_Hedging': '#1abc9c',
            'Dealer': '#27ae60',
            'Total': '#f39c12'
        }
        inst_labels = {
            'Foreign_Investor': '外資',
            'Investment_Trust': '投信',
            'Dealer_self': '自營商(自行買賣)',
            'Dealer_Hedging': '自營商(避險)',
            'Dealer': '自營商合計',
            'Total': '三大法人合計'
        }
        for name_key, label in inst_labels.items():
            sub = institutional_df[institutional_df['name'] == name_key].copy()
            if sub.empty:
                continue
            base_color = inst_colors.get(name_key, '#e74c3c')
            colors = [base_color if v >= 0 else '#95a5a6' for v in sub['net']]
            fig.add_trace(go.Bar(
                x=sub['date'], y=sub['net'],
                name=label,
                marker_color=colors,
                opacity=0.75,
                visible='legendonly' if name_key == 'Total' else True
            ), row=5, col=1)

    # ── KD 隨機指標（台股 Row 6, 美股 Row 5）──
    kd_row = 6 if show_inst else 5
    if 'KD_K' in df.columns and 'KD_D' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['KD_K'],
            mode='lines', name='%K',
            line=dict(color='#ff6b35', width=2)
        ), row=kd_row, col=1)
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['KD_D'],
            mode='lines', name='%D',
            line=dict(color='#1e90ff', width=2)
        ), row=kd_row, col=1)
        # 超買/超賣基準線
        for lvl, clr in [(80, 'rgba(255,71,87,0.6)'), (20, 'rgba(46,213,115,0.6)')]:
            fig.add_hline(y=lvl, line_dash='dash', line_color=clr,
                          line_width=1, row=kd_row, col=1)
        fig.add_hrect(y0=80, y1=100, fillcolor='rgba(255,71,87,0.05)',
                      line_width=0, row=kd_row, col=1)
        fig.add_hrect(y0=0, y1=20, fillcolor='rgba(46,213,115,0.05)',
                      line_width=0, row=kd_row, col=1)
        # 金叉/死叉標記
        if 'KD_GOLDEN' in df.columns:
            gd = df[df['KD_GOLDEN']]
            if not gd.empty:
                fig.add_trace(go.Scatter(
                    x=gd['date'], y=gd['KD_K'],
                    mode='markers', name='KD金叉',
                    marker=dict(symbol='triangle-up', size=9, color='#f39c12')
                ), row=kd_row, col=1)
        if 'KD_DEAD' in df.columns:
            dd = df[df['KD_DEAD']]
            if not dd.empty:
                fig.add_trace(go.Scatter(
                    x=dd['date'], y=dd['KD_K'],
                    mode='markers', name='KD死叉',
                    marker=dict(symbol='triangle-down', size=9, color='#a29bfe')
                ), row=kd_row, col=1)
        fig.update_yaxes(range=[0, 100], row=kd_row, col=1)

    # ── 佈局更新 ──
    fig.update_layout(
        title=f'{symbol} 主K線圖（含布林通道、MA、RSI、OBV、KD）',
        height=chart_height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text=f"價格 ({currency_symbol})", row=1, col=1)
    fig.update_yaxes(title_text=f"RSI({rsi_period})", row=2, col=1)
    fig.update_yaxes(title_text="OBV", row=3, col=1)
    fig.update_yaxes(title_text="成交量", row=4, col=1)
    if show_inst:
        fig.update_yaxes(title_text="買賣超（張）", row=5, col=1)
    fig.update_yaxes(title_text="KD", row=kd_row, col=1)

    return fig


def display_institutional_table(institutional_df):
    """
    三大法人買賣超表格（v4.4 規格 Step 1-6）
    - 只保留 5 個 name（排除 Dealer 合計避免重複）
    - pivot 轉寬表，中文欄位，日期降序，最近 10 交易日
    - 最下方附 10日合計列
    - Total 直接使用 FinMind 官方合計，不自行加總
    """
    try:
        df = institutional_df.copy()

        # Step 2: 確保 net 欄位
        if 'net' not in df.columns:
            df['net'] = df['buy'] - df['sell']

        # Step 3: 篩選法人與日期
        target_names = ['Foreign_Investor', 'Investment_Trust',
                        'Dealer_self', 'Dealer_Hedging', 'Total']
        df = df[df['name'].isin(target_names)]
        top10_dates = sorted(df['date'].unique())[-10:]
        df = df[df['date'].isin(top10_dates)]

        # Step 4: pivot 轉寬表
        pivot = df.pivot_table(
            index='date',
            columns='name',
            values='net',
            aggfunc='sum'
        ).reset_index()

        pivot.rename(columns={
            'date':             '日期',
            'Foreign_Investor': '外資(張)',
            'Investment_Trust': '投信(張)',
            'Dealer_self':      '自營商-自行(張)',
            'Dealer_Hedging':   '自營商-避險(張)',
            'Total':            '三大法人合計(張)'
        }, inplace=True)

        pivot = pivot.sort_values('日期', ascending=False).reset_index(drop=True)
        pivot['日期'] = pd.to_datetime(pivot['日期']).dt.strftime('%Y-%m-%d')

        # Step 5: 10日合計列
        num_cols = [c for c in pivot.columns if c != '日期']
        sum_row = pivot[num_cols].sum(numeric_only=True).to_dict()
        sum_row['日期'] = '10日合計'
        pivot = pd.concat([pivot, pd.DataFrame([sum_row])], ignore_index=True)

        # 整數格式
        for col in num_cols:
            if col in pivot.columns:
                pivot[col] = pivot[col].apply(
                    lambda x: int(x) if pd.notna(x) and x != '' else 0
                )

        # Step 6: 顯示
        st.dataframe(pivot, use_container_width=True, hide_index=True)

    except Exception:
        try:
            simple = institutional_df.tail(40).copy()
            simple['date'] = simple['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(simple[['date', 'name', 'buy', 'sell', 'net']],
                         use_container_width=True, hide_index=True)
        except Exception:
            pass


def display_broker_table(broker_data, symbol, query_date=None):
    """
    券商分點進出明細 — 仿照富邦/元大分點查詢格式
    接收 get_tw_broker_trading() 回傳的 dict

    Args:
        broker_data: dict { date, buy_df, sell_df, total_buy, total_sell }
                     或舊格式 DataFrame（兼容）
        symbol: 股票代碼
        query_date: 日期字串（可由 broker_data['date'] 自動帶入）
    """
    # 兼容舊格式 DataFrame
    if isinstance(broker_data, pd.DataFrame):
        if broker_data.empty:
            st.info("ℹ️ 目前無券商分點資料（可能為非交易日或資料尚未更新）。")
            return
        total_vol = broker_data['buy'].sum() + broker_data['sell'].sum()
        buy_df  = broker_data[broker_data['net'] > 0].sort_values('net', ascending=False).head(20).reset_index(drop=True)
        sell_df = broker_data[broker_data['net'] < 0].copy()
        sell_df['net'] = sell_df['net'].abs()
        sell_df = sell_df.sort_values('net', ascending=False).head(20).reset_index(drop=True)
        def _ratio(r):
            return f"{(r['buy']+r['sell'])/total_vol*100:.2f}%" if total_vol > 0 else '—'
        buy_df['ratio']  = buy_df.apply(_ratio, axis=1)
        sell_df['ratio'] = sell_df.apply(_ratio, axis=1)
        total_buy  = int(broker_data['buy'].sum())
        total_sell = int(broker_data['sell'].sum())
        date_str   = query_date or datetime.now().strftime('%Y-%m-%d')
    elif isinstance(broker_data, dict):
        if not broker_data:
            st.info("ℹ️ 目前無券商分點資料（可能為非交易日或資料尚未更新）。")
            return
        buy_df     = broker_data.get('buy_df', pd.DataFrame())
        sell_df    = broker_data.get('sell_df', pd.DataFrame())
        total_buy  = broker_data.get('total_buy', 0)
        total_sell = broker_data.get('total_sell', 0)
        date_str   = broker_data.get('date', query_date or datetime.now().strftime('%Y-%m-%d'))
        # 確保有 ratio 欄位
        total_vol = total_buy + total_sell
        if 'ratio' not in buy_df.columns:
            def _ratio(r):
                return f"{(r['buy']+r['sell'])/total_vol*100:.2f}%" if total_vol > 0 else '—'
            if not buy_df.empty:  buy_df['ratio']  = buy_df.apply(_ratio, axis=1)
            if not sell_df.empty: sell_df['ratio'] = sell_df.apply(_ratio, axis=1)
        else:
            buy_df['ratio']  = buy_df['ratio'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x))
            sell_df['ratio'] = sell_df['ratio'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x))
    else:
        st.info("ℹ️ 目前無券商分點資料（可能為非交易日或資料尚未更新）。")
        return

    if buy_df.empty and sell_df.empty:
        st.info("ℹ️ 目前無券商分點資料（可能為非交易日或資料尚未更新）。")
        return

    try:
        total_vol  = total_buy + total_sell
        date_disp  = date_str.replace('-', '/')

        def fmt_num(v):
            try:    return f"{int(v):,}"
            except: return str(v)

        max_rows = max(len(buy_df), len(sell_df))

        html = f"""
<style>
.broker-wrap {{ overflow-x: auto; }}
.broker-table {{
    width: 100%; border-collapse: collapse;
    font-size: 13px;
    font-family: 'Microsoft JhengHei', '微軟正黑體', Arial, sans-serif;
}}
.broker-table th {{
    background-color: #d0e8f5; padding: 5px 10px;
    border: 1px solid #aac8d8; text-align: center;
    font-weight: bold; white-space: nowrap;
}}
.broker-table td {{
    padding: 4px 10px; border: 1px solid #dde;
    text-align: right; white-space: nowrap;
}}
.broker-table td.name-cell {{
    text-align: left; color: #1a5fa8; font-weight: 500; min-width: 110px;
}}
.broker-table tr:nth-child(even) {{ background-color: #f5faff; }}
.broker-table tr:hover {{ background-color: #e8f4ff; }}
.buy-net  {{ color: #cc0000; font-weight: bold; }}
.sell-net {{ color: #007700; font-weight: bold; }}
.total-row td {{
    font-weight: bold; background-color: #ddeedd !important;
    border-top: 2px solid #88aa88;
}}
.avg-row td {{
    font-weight: bold; background-color: #eef3ff !important;
}}
.divider-col {{
    width: 8px; min-width: 8px; background: #e8eef4; border: none !important; padding: 0 !important;
}}
.broker-header {{
    background: #e8f4f8; padding: 6px 14px; border-radius: 4px;
    font-size: 13px; margin-bottom: 8px; border-left: 4px solid #3498db;
}}
</style>

<div class="broker-header">
  <b>📊 {symbol} 券商分點 - 進出明細</b>&ensp;
  ｜&ensp;單位：張&ensp;
  ｜&ensp;最後更新日：{date_disp}&ensp;
  ｜&ensp;全體合計買進：{fmt_num(total_buy)}　賣出：{fmt_num(total_sell)}
</div>

<div class="broker-wrap">
<table class="broker-table">
  <thead>
    <tr>
      <th colspan="5" style="background:#fde8e8; color:#800000; font-size:14px;">🔴 買超券商（前20）</th>
      <th class="divider-col"></th>
      <th colspan="5" style="background:#e8fde8; color:#006600; font-size:14px;">🟢 賣超券商（前20）</th>
    </tr>
    <tr>
      <th>買超券商</th><th>買進</th><th>賣出</th><th style="color:#cc0000;">買超</th><th>估成交<br>比重</th>
      <th class="divider-col"></th>
      <th>賣超券商</th><th>買進</th><th>賣出</th><th style="color:#007700;">賣超</th><th>估成交<br>比重</th>
    </tr>
  </thead>
  <tbody>
"""
        for i in range(max_rows):
            if i < len(buy_df):
                b = buy_df.iloc[i]
                b_name  = str(b['broker_name'])[:14]
                b_ratio = str(b.get('ratio', '—'))
                buy_cells = (
                    f'<td class="name-cell">{b_name}</td>'
                    f'<td>{fmt_num(b["buy"])}</td>'
                    f'<td>{fmt_num(b["sell"])}</td>'
                    f'<td class="buy-net">{fmt_num(b["net"])}</td>'
                    f'<td>{b_ratio}</td>'
                )
            else:
                buy_cells = '<td></td><td></td><td></td><td></td><td></td>'

            if i < len(sell_df):
                s = sell_df.iloc[i]
                s_name  = str(s['broker_name'])[:14]
                s_ratio = str(s.get('ratio', '—'))
                sell_cells = (
                    f'<td class="name-cell">{s_name}</td>'
                    f'<td>{fmt_num(s["buy"])}</td>'
                    f'<td>{fmt_num(s["sell"])}</td>'
                    f'<td class="sell-net">{fmt_num(s["net"])}</td>'
                    f'<td>{s_ratio}</td>'
                )
            else:
                sell_cells = '<td></td><td></td><td></td><td></td><td></td>'

            html += f'    <tr>{buy_cells}<td class="divider-col"></td>{sell_cells}</tr>\n'

        # 合計列
        buy_net_total  = buy_df['net'].sum()  if not buy_df.empty  else 0
        sell_net_total = sell_df['net'].sum() if not sell_df.empty else 0
        buy_total_ratio_str  = f"{buy_net_total  / total_vol * 100:.2f}%" if total_vol > 0 else '—'
        sell_total_ratio_str = f"{sell_net_total / total_vol * 100:.2f}%" if total_vol > 0 else '—'

        # 計算平均買進成本（加權平均）
        def avg_cost(df_side, col):
            total = df_side[col].sum()
            if total > 0:
                return '—'  # 無價格資料時顯示 —
            return '—'

        html += f"""
    <tr class="total-row">
      <td class="name-cell">合計買超張數</td>
      <td colspan="2" style="text-align:center;">（上述{len(buy_df)}家）</td>
      <td class="buy-net">{fmt_num(buy_net_total)}</td>
      <td>{buy_total_ratio_str}</td>
      <td class="divider-col"></td>
      <td class="name-cell">合計賣超張數</td>
      <td colspan="2" style="text-align:center;">（上述{len(sell_df)}家）</td>
      <td class="sell-net">{fmt_num(sell_net_total)}</td>
      <td>{sell_total_ratio_str}</td>
    </tr>
  </tbody>
</table>
</div>
<div style="font-size:11px; color:#777; margin-top:6px; padding:0 4px;">
  【註1】合計買超或賣超，為上述家數之合計。&emsp;
  【註2】估成交比重 = 該券商(買進+賣出) ÷ 全體券商(買進+賣出)合計。
</div>
"""
        st.markdown(html, unsafe_allow_html=True)

    except Exception:
        try:
            # 降級顯示
            if not buy_df.empty:
                st.markdown("**🔴 買超券商**")
                st.dataframe(buy_df.head(20), use_container_width=True, hide_index=True)
            if not sell_df.empty:
                st.markdown("**🟢 賣超券商**")
                st.dataframe(sell_df.head(20), use_container_width=True, hide_index=True)
        except Exception:
            pass




def create_analyst_chart(analyst_data, symbol, currency_symbol, current_price):
    """法人目標價散點圖"""
    if analyst_data is None:
        return None
    targets_df = analyst_data.get('targets')
    consensus  = analyst_data.get('consensus')
    if targets_df is None or len(targets_df) == 0:
        return None
    try:
        df = targets_df.copy()
        df['priceTarget'] = pd.to_numeric(df['priceTarget'], errors='coerce')
        df = df.dropna(subset=['priceTarget'])
        if len(df) == 0:
            return None

        df['label'] = df.apply(lambda r: (
            r.get('analystCompany', '未知')[:20] + ' ' +
            (r['publishedDate'].strftime('%Y/%m/%d') if pd.notna(r['publishedDate']) else '')
        ), axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['publishedDate'],
            y=df['priceTarget'],
            mode='markers+text',
            marker=dict(size=10, color='#3498db', symbol='diamond', line=dict(color='white', width=1)),
            text=df['label'],
            textposition='top center',
            name='目標價',
            hovertemplate='%{text}<br>目標價：' + currency_symbol + '%{y:.2f}<extra></extra>'
        ))

        fig.add_hline(y=current_price, line_dash='dash', line_color='#e74c3c',
                      annotation_text=f'現價 {currency_symbol}{current_price:.2f}',
                      annotation_position='right')

        if consensus:
            target_mean = consensus.get('targetConsensus') or consensus.get('targetMean') or consensus.get('priceTarget')
            if target_mean:
                target_mean = float(target_mean)
                fig.add_hline(y=target_mean, line_dash='dot', line_color='#f39c12',
                              annotation_text=f'共識目標 {currency_symbol}{target_mean:.2f}',
                              annotation_position='left')

        avg_target = df['priceTarget'].mean()
        upside = ((avg_target - current_price) / current_price * 100) if current_price > 0 else 0
        title_extra = f'  |  平均目標 {currency_symbol}{avg_target:.2f}（{upside:+.1f}%）'

        fig.update_layout(
            title=f'{symbol} 法人目標價分布' + title_extra,
            height=450,
            template='plotly_white',
            yaxis_title=f'目標價（{currency_symbol}）',
            showlegend=False
        )
        return fig
    except Exception:
        return None



def generate_stock_evaluation(symbol, stock_data, openai_api_key, market='us',
                               zhu_result=None, bull_signals=None, kline_result=None, rsi_period=14,
                               institutional_df=None, margin_df=None,
                               weekly_df=None, financial_data=None,
                               analyst_data=None, insider_df=None, director_df=None):
    """
    使用 AI 自動填入朱家泓選股評量表格（圖表10-3-1）所有欄位，回傳 dict。
    提供豐富原始數值讓 AI 輸出如範例所示的具體描述（含價位、張數、KD值等）。
    """
    try:
        client = OpenAI(api_key=openai_api_key)
        currency    = "NT$" if market == 'tw' else "USD"
        market_desc = "台股" if market == 'tw' else "美股"
        rsi_col     = f'RSI{rsi_period}'

        d = stock_data
        # ── 近5根日K線 ──
        recent5 = d.tail(5)[['date','open','high','low','close','volume']].copy()
        recent5['date'] = recent5['date'].dt.strftime('%Y-%m-%d')
        recent5_str = recent5.to_string(index=False)

        close      = float(d['close'].iloc[-1])
        high_today = float(d['high'].iloc[-1])
        low_today  = float(d['low'].iloc[-1])

        def _v(col):  return float(d[col].iloc[-1])  if col in d.columns else None
        def _v2(col): return float(d[col].iloc[-2])  if col in d.columns and len(d)>=2 else None

        ma5=_v('MA5'); ma5p=_v2('MA5'); ma10=_v('MA10'); ma10p=_v2('MA10')
        ma20=_v('MA20'); ma20p=_v2('MA20'); ma60=_v('MA60'); ma60p=_v2('MA60')
        bb_u=_v('BB_UPPER'); bb_m=_v('BB_MID'); bb_l=_v('BB_LOWER')
        macd_dif=_v('MACD_DIF'); macd_dif_p=_v2('MACD_DIF')
        macd_line=_v('MACD_LINE'); macd_line_p=_v2('MACD_LINE')
        macd_hist=_v('MACD_HIST'); macd_hist_p=_v2('MACD_HIST')
        kd_k=_v('KD_K'); kd_d=_v('KD_D'); kd_k_p=_v2('KD_K'); kd_d_p=_v2('KD_D')
        kd_golden = bool(d['KD_GOLDEN'].tail(5).any()) if 'KD_GOLDEN' in d.columns else False
        kd_dead   = bool(d['KD_DEAD'].tail(5).any())   if 'KD_DEAD'   in d.columns else False
        rsi_val = _v(rsi_col)
        vol_today = float(d['volume'].iloc[-1]) if 'volume' in d.columns else 0
        vol_prev  = float(d['volume'].iloc[-2]) if 'volume' in d.columns and len(d)>=2 else 0
        vol_5avg  = float(d['volume'].tail(5).mean())  if 'volume' in d.columns else 1
        vol_20avg = float(d['volume'].tail(20).mean()) if 'volume' in d.columns else 1
        month_high = float(d['high'].tail(60).max()) if 'high' in d.columns else close
        month_low  = float(d['low'].tail(60).min())  if 'low'  in d.columns else close

        # ── 週線 ──
        w_close=w_ma20=w_ma60=w_ma5=w_ma10=None
        w_macd_dif=w_macd_dif_p=w_macd_line=w_macd_line_p=w_macd_hist=w_macd_hist_p=None
        w_kd_k=w_kd_d=w_kd_k_p=w_kd_d_p=None
        w_vol=w_vol_prev=None
        w_recent3 = "(週線數據不可用)"
        if weekly_df is not None and len(weekly_df) >= 2:
            def _wv(col):  return float(weekly_df[col].iloc[-1])  if col in weekly_df.columns else None
            def _wv2(col): return float(weekly_df[col].iloc[-2])  if col in weekly_df.columns else None
            w_close=_wv('close'); w_ma5=_wv('MA5'); w_ma10=_wv('MA10')
            w_ma20=_wv('MA20'); w_ma60=_wv('MA60')
            w_macd_dif=_wv('MACD_DIF'); w_macd_dif_p=_wv2('MACD_DIF')
            w_macd_line=_wv('MACD_LINE'); w_macd_line_p=_wv2('MACD_LINE')
            w_macd_hist=_wv('MACD_HIST'); w_macd_hist_p=_wv2('MACD_HIST')
            w_kd_k=_wv('KD_K'); w_kd_d=_wv('KD_D'); w_kd_k_p=_wv2('KD_K'); w_kd_d_p=_wv2('KD_D')
            w_vol=_wv('volume'); w_vol_prev=_wv2('volume')
            wr3 = weekly_df.tail(3)[['date','open','high','low','close','volume']].copy()
            wr3['date'] = pd.to_datetime(wr3['date']).dt.strftime('%Y-%m-%d')
            w_recent3 = wr3.to_string(index=False)

        # ── 融資融券 ──
        mg_rem=mg_chg=mg_buy=mg_sell=None
        ss_rem=ss_chg=ss_buy=ss_sell=mg_ratio=None
        if margin_df is not None and len(margin_df) >= 2:
            def _mg(cols):
                for c in cols:
                    if c in margin_df.columns: return float(margin_df[c].iloc[-1])
                return None
            def _mg2(cols):
                for c in cols:
                    if c in margin_df.columns: return float(margin_df[c].iloc[-2])
                return None
            REM_COLS = ['MarginPurchaseRemaining','MarginPurchaseTodayBalance','margin_purchase_remaining']
            SS_COLS  = ['ShortSaleRemaining','ShortSaleTodayBalance','short_sale_remaining']
            mg_rem = _mg(REM_COLS); mg_rem2 = _mg2(REM_COLS)
            mg_buy = _mg(['MarginPurchaseBuy','marginpurchasebuy'])
            mg_sell= _mg(['MarginPurchaseSell','marginpurchasesell'])
            ss_rem = _mg(SS_COLS);  ss_rem2 = _mg2(SS_COLS)
            ss_buy = _mg(['ShortSaleBuy','shortsalebuy'])
            ss_sell= _mg(['ShortSaleSell','shortsalesell'])
            for c in ['MarginPurchaseRatio','margin_ratio']:
                if c in margin_df.columns:
                    mg_ratio = float(margin_df[c].iloc[-1]); break
            if mg_rem is not None and mg_rem2 is not None: mg_chg = mg_rem - mg_rem2
            if ss_rem is not None and ss_rem2 is not None: ss_chg = ss_rem - ss_rem2

        # ── 三大法人 ──
        inst_lines = []
        if institutional_df is not None and len(institutional_df) > 0:
            try:
                it = institutional_df.tail(20)
                for kws, label in [
                    (['外資','Foreign','foreign'],'外資'),
                    (['投信','Investment_Trust','Investment Trust'],'投信'),
                    (['自營','Dealer','dealer'],'自營'),
                ]:
                    sub = None
                    if 'name' in it.columns:
                        for kw in kws:
                            mask = it['name'].astype(str).str.contains(kw, na=False)
                            if mask.any(): sub = it[mask]; break
                    if sub is not None and 'net' in sub.columns:
                        net_sum   = int(sub['net'].sum())
                        net_today = int(sub['net'].iloc[-1])
                        inst_lines.append(f"{label}：今日{'+' if net_today>=0 else ''}{net_today}張，近20日累計{'+' if net_sum>=0 else ''}{net_sum}張")
            except Exception:
                pass

        zhu_str  = ""
        if zhu_result:
            zhu_str = f"朱家泓評分：{zhu_result['score']}/100（{zhu_result['action_label']}），趨勢：{zhu_result['trend']}"
            if zhu_result.get('transition'): zhu_str += f"，{zhu_result['transition']}"
        bull_str = f"多頭訊號：{bull_signals['total_score']:.0f}/100 — {bull_signals['conclusion']}" if bull_signals else ""
        kline_str = ""
        if kline_result:
            kline_str = (f"K線型態評分：{kline_result['score']}/100（{kline_result['action_label']}），"
                         f"位置：{kline_result['position_context']}，最新K線：{kline_result['latest_kline_type']}")
            if kline_result.get('patterns_found'):
                kline_str += f"，命中型態：{'、'.join(kline_result['patterns_found'])}"

        # ── 輔助函式：安全格式化 ──
        def _f(v, fmt=".2f"): return format(v, fmt) if v is not None else "N/A"
        def _chg_str(chg): return (("增加" if chg>0 else "減少") + str(abs(int(chg))) + "張") if chg is not None else ""

        # ── 組裝數值摘要文字 ──
        lines_list = [
            f"【股票代碼】{symbol}（{market_desc}）　收盤：{currency}{_f(close)}",
            "",
            "【近5日K線】",
            recent5_str,
            "",
            "【日線均線】",
            f"MA5={_f(ma5)}（前{_f(ma5p)}）  MA10={_f(ma10)}（前{_f(ma10p)}）  MA20={_f(ma20)}（前{_f(ma20p)}）  MA60={_f(ma60)}（前{_f(ma60p)}）",
            f"布林上軌={_f(bb_u)}  中軌={_f(bb_m)}  下軌={_f(bb_l)}",
            f"收盤位置：{'高於' if ma5 and close>ma5 else '低於'}MA5  {'高於' if ma20 and close>ma20 else '低於'}MA20  {'高於' if ma60 and close>ma60 else '低於'}MA60",
            "",
            "【日線成交量】",
            f"今日={int(vol_today)}　前日={int(vol_prev)}　5日均={int(vol_5avg)}　20日均={int(vol_20avg)}　量比={vol_today/vol_20avg:.2f}倍",
            "",
            "【日線MACD(12,26,9)】",
            f"DIF={_f(macd_dif,'.4f')}（前{_f(macd_dif_p,'.4f')}）  SIGNAL={_f(macd_line,'.4f')}（前{_f(macd_line_p,'.4f')}）  HIST={_f(macd_hist,'.4f')}（前{_f(macd_hist_p,'.4f')}）",
            f"DIF在零軸{'上方' if macd_dif and macd_dif>0 else '下方'}　HIST{'擴大' if macd_hist and macd_hist_p and abs(macd_hist)>abs(macd_hist_p) else '縮小'}中",
            "",
            "【日線KD(9)】",
            f"K={_f(kd_k,'.1f')}（前{_f(kd_k_p,'.1f')}）  D={_f(kd_d,'.1f')}（前{_f(kd_d_p,'.1f')}）" +
            (" ⭐近5日KD金叉" if kd_golden else "") + (" ⚠️近5日KD死叉" if kd_dead else "") +
            (" 【高檔鈍化K>80】" if kd_k and kd_k>80 else " 【低檔鈍化K<20】" if kd_k and kd_k<20 else ""),
            f"RSI({rsi_period})={_f(rsi_val,'.2f')}",
            "",
            "【近3週K線】",
            w_recent3,
            "",
            "【週線均線】",
            f"週收={_f(w_close)}  週MA5={_f(w_ma5)}  週MA10={_f(w_ma10)}  週MA20={_f(w_ma20)}  週MA60={_f(w_ma60)}",
            f"週收 vs 週均：{'高於' if w_close and w_ma20 and w_close>w_ma20 else '低於'}週MA20  {'高於' if w_close and w_ma60 and w_close>w_ma60 else '低於'}週MA60",
            "",
            "【週線MACD】",
            f"DIF={_f(w_macd_dif,'.4f')}（前{_f(w_macd_dif_p,'.4f')}）  SIGNAL={_f(w_macd_line,'.4f')}（前{_f(w_macd_line_p,'.4f')}）  HIST={_f(w_macd_hist,'.4f')}",
            "",
            "【週線KD】",
            f"K={_f(w_kd_k,'.1f')}（前{_f(w_kd_k_p,'.1f')}）  D={_f(w_kd_d,'.1f')}（前{_f(w_kd_d_p,'.1f')}）",
            "",
            f"【近60日高低】最高={_f(month_high)}　最低={_f(month_low)}",
        ]

        if mg_rem is not None:
            lines_list += [
                "",
                "【融資融券（台股）】",
                f"融資餘額={int(mg_rem)}張（{_chg_str(mg_chg)}）　融資買進={int(mg_buy) if mg_buy else 'N/A'}張　融資賣出={int(mg_sell) if mg_sell else 'N/A'}張",
                f"融券餘額={int(ss_rem) if ss_rem else 'N/A'}張（{_chg_str(ss_chg)}）　融券賣出={int(ss_sell) if ss_sell else 'N/A'}張　融券回補={int(ss_buy) if ss_buy else 'N/A'}張",
                f"融資比率={_f(mg_ratio,'.2f') if mg_ratio else 'N/A'}%",
            ]
        if inst_lines:
            lines_list += ["", "【三大法人（台股）】"] + inst_lines
        if zhu_str:  lines_list += ["", zhu_str]
        if bull_str: lines_list.append(bull_str)
        if kline_str: lines_list.append(kline_str)

        data_block = "\n".join(lines_list)

        example_block = (
            "\n【填寫範例參考（安國8054，2023/9/15）】\n"
            "波型_月線=底部橫盤\n"
            "波型_週線=底底高、突破MA20、MA60\n"
            "波型_日線=多頭確認、突破MA20、MA60\n"
            "位置_週線=多頭回後上漲\n"
            "位置_日線=多頭確認，週前高盤整壓力\n"
            "K線_週=大量長紅K突破MA20、MA60\n"
            "K線_日=大量長紅K，收盤站上季線(MA60)\n"
            "均線_週=週MA20向上，股價站上週MA20\n"
            "均線_日=日均線4線多排向上，可中長線操作\n"
            "切線=切線多頭向上\n"
            "成交量_週=週量比前一週大3倍\n"
            "成交量_日=日線出大量\n"
            "指標_週MACD=週MACD在0軸之上向上，黃金交叉，紅柱\n"
            "指標_週KD=K59 > D40 多排向上\n"
            "指標_日MACD=日MACD在0軸之上向上，紅柱延長\n"
            "指標_日KD=K69 > D60 多排向上\n"
            "支撐_週=大量長紅低點37.45元；MA60為33.87元；MA20為33元\n"
            "支撐_日=大量長紅低點37.45元；MA20為33元\n"
            "壓力_週=38.3元、43元、48.55元、67.8元\n"
            "壓力_日=38.3元、43元、48.55元\n"
            "背離=無\n"
            "融資=增加589張，餘額4950張\n"
            "融券=增加95張，餘額144張\n"
            "融資比=2.91%\n"
            "法人買賣超=外資：連3天買超，今買135張\n"
            "型態=圓弧底多頭型態\n"
            "其他=今日爆大量長紅，明日遇壓容易震盪\n"
            "策略=1.明日遇壓，待突破壓力38.3元（週線多頭確認），開始做多。2.資金分配20張，10張短線守MA5操作，10張長線守MA20操作。\n"
        )

        system_msg = (
            "你是台灣資深股票技術分析師，精通朱家泓《多空操作秘笈》技術分析體系。\n"
            "任務：根據提供的完整數值數據，仿照範例填入選股評量表格每個欄位。\n"
            "要求：\n"
            "1. 所有欄位必須引用具體數值（價位到小數點、張數、KD值、倍數等），不可空洞\n"
            "2. 波型描述要判斷底部/中段/頂部的位置，以及是否突破均線\n"
            "3. 支撐/壓力必須給出具體價位（參考MA線、近期高低點、布林通道）\n"
            "4. 融資融券欄位必須包含「增加/減少X張，餘額X張」格式\n"
            "5. 策略欄位需包含：具體進出場條件（含突破哪條均線）、資金管理建議\n"
            "6. 以繁體中文輸出，直接輸出 JSON，不加 markdown 符號"
        )

        user_msg = (
            data_block + "\n\n" + example_block +
            "\n請仿照上方範例格式（具體數值、張數、價位），填入以下 JSON 結構，所有值為字串：\n"
            '{\n'
            '  "波型_月線": "",\n'
            '  "波型_週線": "",\n'
            '  "波型_日線": "",\n'
            '  "位置_週線": "",\n'
            '  "位置_日線": "",\n'
            '  "K線_週": "",\n'
            '  "K線_日": "",\n'
            '  "均線_週": "",\n'
            '  "均線_日": "",\n'
            '  "切線": "",\n'
            '  "成交量_週": "",\n'
            '  "成交量_日": "",\n'
            '  "指標_週MACD": "",\n'
            '  "指標_週KD": "",\n'
            '  "指標_日MACD": "",\n'
            '  "指標_日KD": "",\n'
            '  "支撐_週": "",\n'
            '  "支撐_日": "",\n'
            '  "壓力_週": "",\n'
            '  "壓力_日": "",\n'
            '  "背離": "",\n'
            '  "融資": "",\n'
            '  "融券": "",\n'
            '  "融資比": "",\n'
            '  "法人買賣超": "",\n'
            '  "型態": "",\n'
            '  "其他": "",\n'
            '  "策略": ""\n'
            '}'
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=1500,
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception as e:
        return {"_error": str(e)}


def generate_investment_advice(symbol, stock_data, openai_api_key, market='us', zhu_result=None,
                                bull_signals=None, kline_result=None, rsi_period=14,
                                institutional_df=None, margin_df=None,
                                financial_data=None, analyst_data=None,
                                insider_df=None, director_df=None):
    """
    獨立呼叫 AI，產生結構化的長線 / 短線投資建議卡片。
    與 generate_ai_insights 分開，避免篇幅膨脹，專注於可操作建議。
    """
    try:
        client = OpenAI(api_key=openai_api_key)

        currency    = "NT$" if market == 'tw' else "USD"
        market_desc = "台股" if market == 'tw' else "美股"
        rsi_col     = f'RSI{rsi_period}'
        latest_close  = stock_data['close'].iloc[-1]
        latest_rsi    = stock_data[rsi_col].iloc[-1] if rsi_col in stock_data.columns else 50
        ma20 = stock_data['MA20'].iloc[-1] if 'MA20' in stock_data.columns else None
        ma60 = stock_data['MA60'].iloc[-1] if 'MA60' in stock_data.columns else None
        bb_upper = stock_data['BB_UPPER'].iloc[-1] if 'BB_UPPER' in stock_data.columns else None
        bb_lower = stock_data['BB_LOWER'].iloc[-1] if 'BB_LOWER' in stock_data.columns else None

        # 基礎摘要
        summary_lines = [
            f"股票代碼：{symbol}（{market_desc}）",
            f"最新收盤：{currency} {latest_close:.2f}",
            f"RSI({rsi_period})：{latest_rsi:.2f}",
        ]
        if ma20: summary_lines.append(f"MA20：{ma20:.2f}")
        if ma60: summary_lines.append(f"MA60：{ma60:.2f}")
        if bb_upper and bb_lower:
            summary_lines.append(f"布林上軌：{bb_upper:.2f}  布林下軌：{bb_lower:.2f}")

        # ── 線型辨識輔助數值（朱家泓圖案判斷用）──
        try:
            _d = stock_data
            _close  = float(_d['close'].iloc[-1])
            _vol    = float(_d['volume'].iloc[-1])   if 'volume' in _d.columns else 0
            _vol20  = float(_d['volume'].tail(20).mean()) if 'volume' in _d.columns else 1
            _vol_ratio = round(_vol / _vol20, 2) if _vol20 > 0 else 0
            _ma5    = float(_d['MA5'].iloc[-1])  if 'MA5'  in _d.columns else None
            _ma20   = float(_d['MA20'].iloc[-1]) if 'MA20' in _d.columns else None
            _ma60   = float(_d['MA60'].iloc[-1]) if 'MA60' in _d.columns else None
            _kd_k   = float(_d['KD_K'].iloc[-1]) if 'KD_K' in _d.columns else None
            _kd_d   = float(_d['KD_D'].iloc[-1]) if 'KD_D' in _d.columns else None
            _kd_golden = bool(_d['KD_GOLDEN'].tail(5).any()) if 'KD_GOLDEN' in _d.columns else False
            _kd_dead   = bool(_d['KD_DEAD'].tail(5).any())   if 'KD_DEAD'   in _d.columns else False
            _macd_dif  = float(_d['MACD_DIF'].iloc[-1])  if 'MACD_DIF'  in _d.columns else None
            _macd_hist = float(_d['MACD_HIST'].iloc[-1]) if 'MACD_HIST' in _d.columns else None
            _macd_hist_p = float(_d['MACD_HIST'].iloc[-2]) if 'MACD_HIST' in _d.columns and len(_d)>=2 else None
            _hi_today = float(_d['high'].iloc[-1]) if 'high' in _d.columns else _close
            _lo_today = float(_d['low'].iloc[-1])  if 'low'  in _d.columns else _close
            _body_ratio = abs(_close - float(_d['open'].iloc[-1])) / (_hi_today - _lo_today + 0.001)
            _upper_shadow = (_hi_today - max(_close, float(_d['open'].iloc[-1]))) / (_hi_today - _lo_today + 0.001)
            _prev_hi5 = float(_d['high'].tail(5).iloc[:-1].max()) if 'high' in _d.columns else _close
            _min60  = float(_d['low'].tail(60).min())  if 'low'  in _d.columns else _close
            _max60  = float(_d['high'].tail(60).max()) if 'high' in _d.columns else _close

            pattern_data = {
                "收盤": round(_close, 2),
                "今日量/20日均量倍數": _vol_ratio,
                "今日實體比例(0~1)": round(_body_ratio, 2),
                "今日上影線比例(0~1)": round(_upper_shadow, 2),
                "近5日最高點": round(_prev_hi5, 2),
                "近60日最低": round(_min60, 2),
                "近60日最高": round(_max60, 2),
            }
            if _ma5:  pattern_data["MA5"] = round(_ma5, 2)
            if _ma20: pattern_data["MA20"] = round(_ma20, 2)
            if _ma60: pattern_data["MA60"] = round(_ma60, 2)
            if _kd_k: pattern_data["KD_K"] = round(_kd_k, 1)
            if _kd_d: pattern_data["KD_D"] = round(_kd_d, 1)
            pattern_data["KD近5日金叉"] = _kd_golden
            pattern_data["KD近5日死叉"] = _kd_dead
            if _macd_dif:  pattern_data["MACD_DIF"] = round(_macd_dif, 4)
            if _macd_hist: pattern_data["MACD_HIST"] = round(_macd_hist, 4)
            if _macd_hist_p: pattern_data["MACD_HIST前日"] = round(_macd_hist_p, 4)
            pattern_data["收盤高於MA5"]  = bool(_ma5  and _close > _ma5)
            pattern_data["收盤高於MA20"] = bool(_ma20 and _close > _ma20)
            pattern_data["收盤高於MA60"] = bool(_ma60 and _close > _ma60)
            user_prompt += f"\n### 朱家泓線型辨識輔助數值\n{json.dumps(pattern_data, ensure_ascii=False)}\n"
        except Exception:
            pass

        # 多頭訊號
        if bull_signals:
            summary_lines.append(
                f"多頭訊號評分：{bull_signals['total_score']:.0f}/100 — {bull_signals['conclusion']}"
            )

        # 朱家泓趨勢系統摘要
        if zhu_result:
            summary_lines.append(
                f"朱家泓趨勢系統評分：{zhu_result['score']}/100 — {zhu_result['action_label']}"
            )
            if zhu_result.get('transition'):
                summary_lines.append(f"趨勢轉換偵測：{zhu_result['transition']}")
            if zhu_result.get('consolidation_breakout'):
                summary_lines.append("盤整突破訊號：已確認")

        # K線型態分析系統摘要
        if kline_result:
            summary_lines.append(
                f"K線型態評分：{kline_result['score']}/100 — {kline_result['action_label']}"
                f"（位置：{kline_result['position_context']}，最新K線：{kline_result['latest_kline_type']}）"
            )
            if kline_result.get('patterns_found'):
                summary_lines.append(f"K線命中型態：{'、'.join(kline_result['patterns_found'])}")

        # 台股籌碼附加
        if market == 'tw':
            if institutional_df is not None and len(institutional_df) > 0:
                try:
                    inst_j = institutional_df.tail(10).to_json(orient='records', date_format='iso')
                    summary_lines.append(f"三大法人近10日：{inst_j}")
                except Exception:
                    pass
            if margin_df is not None and len(margin_df) > 0:
                try:
                    mg_j = margin_df.tail(5).to_json(orient='records', date_format='iso')
                    summary_lines.append(f"融資融券近5日：{mg_j}")
                except Exception:
                    pass
            if financial_data and financial_data.get('quarterly') is not None:
                try:
                    fq_j = financial_data['quarterly'].tail(4).to_json(orient='records', date_format='iso')
                    summary_lines.append(f"近4季財務數據：{fq_j}")
                except Exception:
                    pass
            if director_df is not None and isinstance(director_df, dict):
                try:
                    _dir = director_df.get("goodinfo") or director_df.get("mops")
                    if _dir is not None and len(_dir) > 0:
                        summary_lines.append(f"董監持股近況：{_dir.head(5).to_json(orient='records', date_format='iso')}")
                except Exception:
                    pass

        # 美股分析師目標價
        if analyst_data is not None:
            try:
                con = analyst_data.get('consensus')
                if con:
                    summary_lines.append(f"分析師共識：{json.dumps(con, ensure_ascii=False)}")
            except Exception:
                pass

        data_summary = "\n".join(summary_lines)

        system_msg = """你是一位資深股票分析師，擅長整合技術面、籌碼面與基本面，給出具體、結構化的投資建議。

重要規則：
- 使用繁體中文
- 只輸出格式化的投資建議，不要重複前面已有的技術分析
- 建議必須包含具體的觀察依據（例如：RSI數值、均線位置、法人動向）
- 嚴禁「保證獲利」等不當用語；須加上「以上為技術面分析參考，非投資建議，投資有風險」免責聲明
- 輸出格式嚴格按照下方範本
"""

        user_msg = f"""根據以下數據，給出{symbol}（{market_desc}）的長短線投資建議：

{data_summary}

請嚴格按照以下 Markdown 格式輸出（不要增加或刪除任何標題層級）：

---

## 🎯 投資建議摘要 — {symbol}

### ⚡ 短線操作建議（1–4週）

**方向**：[偏多 / 偏空 / 觀望]
**信心度**：[高 / 中 / 低]

**進場條件**：
- （條件1，需含具體技術數值）
- （條件2）

**出場 / 停損條件**：
- 停損：（具體數值或條件）
- 停利：（具體數值或條件）

**主要依據**：（2–3句說明，引用RSI/MACD/布林通道/成交量等具體數值）

---

### 📈 長線佈局建議（1–6個月）

**方向**：[偏多 / 偏空 / 持平觀察]
**信心度**：[高 / 中 / 低]

**佈局條件**：
- （條件1，需含具體技術或基本面數值）
- （條件2）

**關鍵觀察指標**：
- （指標1）
- （指標2）
- （指標3）

**主要依據**：（3–4句說明，整合技術面、籌碼面、基本面）

---

### ⚠️ 主要風險提示

- （風險1）
- （風險2）
- （風險3）

---

> 以上為技術面分析參考，非投資建議，投資有風險，請依個人財務狀況審慎評估。

---
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=1200,
            temperature=0.4
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ 投資建議生成失敗：{str(e)}"


# ─────────────────────────────────────────────
# AI 分析函數
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 側邊欄 UI
# ─────────────────────────────────────────────

st.sidebar.markdown("## 🔧 分析設定")
st.sidebar.divider()

market = st.sidebar.selectbox(
    "市場選擇",
    options=["台股 (TW)", "美股 (US)"],
    index=0,
    help="選擇要分析的市場"
)
is_tw = market == "台股 (TW)"

if is_tw:
    symbol = st.sidebar.text_input(
        "台股代碼",
        value="2330",
        help="輸入台股純數字代碼，例如：2330（台積電）、2317（鴻海）、0050（元大台灣50）"
    )
else:
    symbol = st.sidebar.text_input(
        "股票代碼",
        value="AAPL",
        help="輸入美股股票代碼，例如：AAPL, MSFT, GOOGL, TSLA, NVDA"
    )

if is_tw:
    finmind_api_key = st.sidebar.text_input("FinMind API Key", type="password",
                                             help="請輸入您的 FinMind API 金鑰（台股數據）")
    fmp_api_key = ""
else:
    fmp_api_key = st.sidebar.text_input("FMP API Key", type="password",
                                         help="請輸入您的 Financial Modeling Prep API 金鑰")
    finmind_api_key = ""

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password",
                                         help="請輸入您的 OpenAI API 金鑰")

default_start_date = datetime.now() - timedelta(days=365)
default_end_date   = datetime.now()

start_date = st.sidebar.date_input("起始日期", value=default_start_date)
end_date   = st.sidebar.date_input("結束日期", value=default_end_date)

rsi_period = st.sidebar.number_input(
    "RSI 計算天數", min_value=2, max_value=50, value=14, step=1,
    help="RSI 計算週期，預設 14 日，範圍 2–50"
)

# 移動平均線可切換顯示
selected_mas = st.sidebar.multiselect(
    "顯示移動平均線",
    options=["MA5", "MA10", "MA20", "MA60"],
    default=["MA5", "MA10", "MA20", "MA60"],
    help="選擇要在K線圖上顯示的移動平均線"
)

analyze_button = st.sidebar.button("🚀 開始分析", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📢 免責聲明

本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。

請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")


# ─────────────────────────────────────────────
# 主要分析邏輯
# ─────────────────────────────────────────────

if analyze_button:
    # ── 輸入驗證 ──
    if not symbol.strip():
        st.error("請輸入股票代碼")
    elif is_tw and not symbol.strip().isdigit():
        st.error("台股代碼請輸入純數字，例如：2330、0050")
    elif is_tw and not finmind_api_key.strip():
        st.error("請輸入 FinMind API Key")
    elif not is_tw and not fmp_api_key.strip():
        st.error("請輸入 FMP API Key")
    elif not openai_api_key.strip():
        st.error("請輸入 OpenAI API Key")
    elif start_date >= end_date:
        st.error("起始日期不能晚於或等於結束日期")
    else:
        market_key      = 'tw' if is_tw else 'us'
        currency_symbol = "NT$" if is_tw else "$"

        # ── Step 1: 獲取日K價格數據 ──
        spinner_price = "正在獲取台股價格數據..." if is_tw else "正在獲取美股價格數據..."
        with st.spinner(spinner_price):
            if is_tw:
                stock_data = get_tw_stock_price(symbol.strip(), finmind_api_key, start_date, end_date)
            else:
                stock_data = get_us_stock_data(symbol.upper(), fmp_api_key, start_date, end_date)

        if stock_data is not None and len(stock_data) > 0:
            st.success(f"成功獲取 {len(stock_data)} 筆交易數據")
            filtered_data = filter_by_date_range(stock_data, start_date, end_date)

            if filtered_data is not None and len(filtered_data) > 0:

                # ── Step 2: 計算技術指標 ──
                with st.spinner("正在計算技術指標（MA & RSI）..."):
                    data_with_ma = get_moving_averages(filtered_data)
                    data_with_indicators = calculate_rsi(data_with_ma, period=rsi_period)

                # ── Step 3: 計算進階指標 ──
                with st.spinner("正在計算進階技術指標（MACD / 布林通道 / OBV / DMI）..."):
                    data_with_indicators = calculate_advanced_indicators(data_with_indicators)

                # ── Step 5: 計算多頭訊號 ──
                bull_signals = calculate_bull_signals(data_with_indicators)
                zhu_result   = calculate_zhu_trend_system(data_with_indicators)
                kline_result = calculate_kline_pattern_system(data_with_indicators)

                # ── Step 6: 籌碼/附加數據 ──
                margin_df        = None
                institutional_df = None
                analyst_data     = None
                broker_df        = None
                broker_date      = None

                if is_tw:
                    with st.spinner("正在獲取台股籌碼數據（融資融券、三大法人）..."):
                        margin_df        = get_tw_margin_trading(symbol.strip(), finmind_api_key, start_date, end_date)
                        institutional_df = get_tw_institutional(symbol.strip(), finmind_api_key, start_date, end_date)

                    with st.spinner("正在從 TWSE 獲取券商分點進出明細..."):
                        # 往回找最近 5 個交易日（週末/假日無資料）
                        for days_back in range(0, 8):
                            _d = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                            # 跳過周末
                            _weekday = (datetime.now() - timedelta(days=days_back)).weekday()
                            if _weekday >= 5:  # 5=週六, 6=週日
                                continue
                            _result = get_tw_broker_trading(symbol.strip(), finmind_api_key, _d)
                            if _result is not None:
                                broker_df   = _result
                                broker_date = _result.get('date', _d)
                                break

                    status_parts = []
                    if margin_df is not None:         status_parts.append("融資融券")
                    if institutional_df is not None:   status_parts.append("三大法人")
                    if broker_df is not None:          status_parts.append(f"券商分點({broker_date})")

                    if status_parts:
                        st.success(f"成功獲取台股附加數據：{'、'.join(status_parts)}")
                    else:
                        st.warning("台股附加數據獲取失敗，將僅顯示技術面分析。")
                else:
                    with st.spinner("正在獲取法人目標價與評級..."):
                        analyst_data = get_analyst_targets(symbol.upper(), fmp_api_key)

                    if analyst_data is not None:
                        st.success("成功獲取附加數據：法人目標價")

                if data_with_indicators is not None:

                    rsi_col    = f'RSI{rsi_period}'
                    latest_rsi = data_with_indicators[rsi_col].iloc[-1] if rsi_col in data_with_indicators.columns else 50

                    # ── 顯示 5：基本統計4欄 ──
                    st.markdown("### 📈 基本統計資訊")
                    col1, col2, col3, col4 = st.columns(4)
                    s_price = data_with_indicators['close'].iloc[0]
                    e_price = data_with_indicators['close'].iloc[-1]
                    price_change_val = e_price - s_price
                    price_change_pct = (price_change_val / s_price) * 100
                    with col1:
                        st.metric("起始價格", f"{currency_symbol}{s_price:.2f}")
                    with col2:
                        st.metric("結束價格", f"{currency_symbol}{e_price:.2f}")
                    with col3:
                        st.metric("價格變化", f"{currency_symbol}{price_change_val:.2f}", f"{price_change_pct:.2f}%")
                    with col4:
                        rsi_delta = "超買🔴" if latest_rsi > 70 else ("超賣🟢" if latest_rsi < 30 else "中性🔵")
                        st.metric(f"最新 RSI ({rsi_period}日)", f"{latest_rsi:.2f}", rsi_delta)

                    # ── 顯示 9：法人目標價（多來源整合）──
                    st.markdown("### 🎯 法人目標價分析")

                    # ── 輔助函式：目標價統計 metrics ──
                    def _show_price_metrics(df, price_col, currency_sym):
                        prices_raw = []
                        for v in df[price_col]:
                            try:
                                num = float(str(v).replace('NT$','').replace('$','')
                                            .replace(',','').strip())
                                prices_raw.append(num)
                            except Exception:
                                pass
                        m1, m2, m3, m4 = st.columns(4)
                        with m1: st.metric("目標價最高", f"{currency_sym}{max(prices_raw):.2f}" if prices_raw else "—")
                        with m2: st.metric("目標價最低", f"{currency_sym}{min(prices_raw):.2f}" if prices_raw else "—")
                        with m3:
                            avg_p = sum(prices_raw)/len(prices_raw) if prices_raw else 0
                            st.metric("目標價均值", f"{currency_sym}{avg_p:.2f}" if prices_raw else "—")
                        with m4: st.metric("機構家數", f"{len(df)} 家")

                    # ── 決定 tabs 數量（美股 3 tabs / 台股 3 tabs）──
                    if is_tw:
                        _tw_sym   = symbol.strip()
                        _tw_name  = None
                        try:
                            _tw_profile = get_tw_company_profile(_tw_sym)
                            _tw_name    = _tw_profile.get("companyName", "")
                            if _tw_name == _tw_sym: _tw_name = None
                        except Exception:
                            pass

                        _tab_ai, _tab_anue, _tab_goodinfo = st.tabs([
                            "🤖 AI 新聞彙整", "📰 鉅亨網", "📊 GoodInfo 法人評等"
                        ])

                        # ── Tab A：AI 搜尋新聞 ──
                        with _tab_ai:
                            with st.spinner("AI 正在搜尋近一個月法人目標價新聞..."):
                                ai_targets = get_analyst_targets_ai(
                                    _tw_sym, openai_api_key,
                                    market=market_key, stock_name=_tw_name
                                )
                            st.caption(f"📅 資料搜尋時間：{ai_targets.get('search_date','')}，涵蓋近一個月新聞")
                            tgt_table = ai_targets.get('table')
                            if tgt_table is not None and len(tgt_table) > 0:
                                _show_price_metrics(tgt_table, '目標價', currency_symbol)
                                st.dataframe(tgt_table, use_container_width=True, hide_index=True)
                            else:
                                st.warning("⚠️ AI 搜尋未取得資料")
                                if ai_targets.get('error'):
                                    with st.expander("🔍 錯誤詳情"): st.code(ai_targets['error'])
                                st.markdown("🔗 [手動查詢鉅亨網目標價](https://cmnews.com.tw/report)")

                        # ── Tab B：鉅亨網爬蟲 ──
                        with _tab_anue:
                            with st.spinner("正在從鉅亨網取得法人評等資料..."):
                                anue_df, anue_err = get_anue_targets(_tw_sym, _tw_name)
                            if anue_err:
                                st.warning(f"⚠️ 鉅亨網：{anue_err}")
                                st.markdown("🔗 [前往鉅亨網搜尋](https://news.cnyes.com/news/cat/tw_stock_target_price)")
                            elif anue_df.empty:
                                st.info("ℹ️ 鉅亨網目前無此股票目標價新聞")
                            else:
                                if '目標價' in anue_df.columns:
                                    _show_price_metrics(anue_df, '目標價', currency_symbol)
                                st.dataframe(anue_df, use_container_width=True, hide_index=True)
                                st.caption("資料來源：鉅亨網（cnyes.com），標題擷取，僅供參考")
                                st.markdown(f"🔗 [前往鉅亨網查詢更多](https://news.cnyes.com/news/cat/tw_stock_target_price)")

                        # ── Tab C：GoodInfo 法人評等 ──
                        with _tab_goodinfo:
                            with st.spinner("正在從 GoodInfo 取得法人評等..."):
                                gi_df, gi_err = get_goodinfo_stock_rating(_tw_sym)
                            if gi_err:
                                st.warning(f"⚠️ GoodInfo：{gi_err}")
                                st.markdown(f"🔗 [GoodInfo 法人評等](https://goodinfo.tw/tw/StockRating.asp?STOCK_ID={_tw_sym})")
                            elif gi_df.empty:
                                st.info("ℹ️ GoodInfo 目前無此股票法人評等資料")
                                st.markdown(f"🔗 [GoodInfo 法人評等](https://goodinfo.tw/tw/StockRating.asp?STOCK_ID={_tw_sym})")
                            else:
                                st.dataframe(gi_df, use_container_width=True, hide_index=True)
                                st.caption(f"資料來源：GoodInfo.tw — 個股法人評等")
                                st.markdown(f"🔗 [GoodInfo 法人評等（完整頁面）](https://goodinfo.tw/tw/StockRating.asp?STOCK_ID={_tw_sym})")

                    else:
                        # 美股：3 tabs
                        _us_sym = symbol.upper()
                        _tab_ai, _tab_finviz, _tab_fmp = st.tabs([
                            "🤖 AI 新聞彙整", "📊 Finviz", "🏦 FMP（散點圖）"
                        ])

                        # ── Tab A：AI 搜尋新聞 ──
                        with _tab_ai:
                            with st.spinner("AI 正在搜尋近一個月法人目標價新聞..."):
                                ai_targets = get_analyst_targets_ai(
                                    _us_sym, openai_api_key, market=market_key
                                )
                            st.caption(f"📅 資料搜尋時間：{ai_targets.get('search_date','')}，涵蓋近一個月新聞")
                            tgt_table = ai_targets.get('table')
                            if tgt_table is not None and len(tgt_table) > 0:
                                _show_price_metrics(tgt_table, '目標價', currency_symbol)
                                st.dataframe(tgt_table, use_container_width=True, hide_index=True)
                            else:
                                st.warning("⚠️ AI 搜尋未取得資料")
                                if ai_targets.get('error'):
                                    with st.expander("🔍 錯誤詳情"): st.code(ai_targets['error'])
                                st.markdown(f"🔗 [手動查詢 Benzinga 目標價](https://www.benzinga.com/stock/{_us_sym.lower()}/analyst-ratings)")

                        # ── Tab B：Finviz 爬蟲 ──
                        with _tab_finviz:
                            with st.spinner("正在從 Finviz 取得法人評等..."):
                                fv_df, fv_err = get_finviz_targets(_us_sym)
                            if fv_err:
                                st.warning(f"⚠️ Finviz：{fv_err}")
                                st.markdown(f"🔗 [前往 Finviz 查詢](https://finviz.com/quote.ashx?t={_us_sym})")
                            elif fv_df.empty:
                                st.info("ℹ️ Finviz 目前無法人評等資料")
                                st.markdown(f"🔗 [前往 Finviz 查詢](https://finviz.com/quote.ashx?t={_us_sym})")
                            else:
                                if '目標價' in fv_df.columns:
                                    _show_price_metrics(fv_df, '目標價', currency_symbol)
                                st.dataframe(fv_df, use_container_width=True, hide_index=True)
                                st.caption("資料來源：Finviz.com — 法人評等與目標價紀錄（近期）")
                                st.markdown(f"🔗 [Finviz 完整頁面](https://finviz.com/quote.ashx?t={_us_sym})")

                        # ── Tab C：FMP 散點圖（原邏輯保留）──
                        with _tab_fmp:
                            if analyst_data is not None:
                                current_price = data_with_indicators['close'].iloc[-1]
                                analyst_fig = create_analyst_chart(
                                    analyst_data, _us_sym, currency_symbol, current_price)
                                if analyst_fig:
                                    st.plotly_chart(analyst_fig, use_container_width=True)
                                    con = analyst_data.get('consensus')
                                    if con:
                                        ec1, ec2, ec3, ec4 = st.columns(4)
                                        tgt_hi  = con.get('targetHigh') or con.get('targetHighPrice')
                                        tgt_lo  = con.get('targetLow')  or con.get('targetLowPrice')
                                        tgt_avg = con.get('targetConsensus') or con.get('targetMean') or con.get('priceTarget')
                                        rating  = con.get('consensus') or con.get('rating') or '—'
                                        with ec1: st.metric("目標價最高", f"{currency_symbol}{float(tgt_hi):.2f}" if tgt_hi else "—")
                                        with ec2: st.metric("目標價最低", f"{currency_symbol}{float(tgt_lo):.2f}" if tgt_lo else "—")
                                        with ec3: st.metric("目標價均值", f"{currency_symbol}{float(tgt_avg):.2f}" if tgt_avg else "—")
                                        with ec4: st.metric("共識評級", str(rating))
                                    # FMP 原始目標價表格
                                    if analyst_data.get('targets') is not None:
                                        with st.expander("📋 FMP 目標價明細", expanded=False):
                                            st.dataframe(analyst_data['targets'], use_container_width=True, hide_index=True)
                                else:
                                    st.info("ℹ️ FMP 目標價圖表無法生成")
                            else:
                                st.warning("⚠️ FMP 目標價資料未取得（請確認 FMP API Key）")
                                st.markdown(f"🔗 [FMP {_us_sym} 目標價](https://financialmodelingprep.com/financial-summary/{_us_sym})")

                    # ── 顯示 10：多頭訊號儀表板 ──
                    display_bull_dashboard(bull_signals, symbol.strip() if is_tw else symbol.upper())

                    # ── 顯示 1：主K線圖（圖1）──
                    st.markdown("### 📊 主K線圖（含布林通道、OBV、成交量）")
                    chart = create_candlestick_chart(
                        data_with_indicators,
                        symbol.upper() if not is_tw else symbol.strip(),
                        rsi_period,
                        currency_symbol,
                        institutional_df=institutional_df,
                        market=market_key,
                        selected_mas=selected_mas
                    )
                    st.plotly_chart(chart, use_container_width=True)

                    # ── 顯示 2：RSI 即時警告 ──
                    rsi_col = f'RSI{rsi_period}'
                    latest_rsi = data_with_indicators[rsi_col].iloc[-1] if rsi_col in data_with_indicators.columns else 50
                    if latest_rsi > 70:
                        st.warning(f"⚠️ RSI 超買警告：目前 RSI = **{latest_rsi:.2f}**，已進入超買區域（>70）。歷史數據顯示此區域價格波動風險較高。")
                    elif latest_rsi < 30:
                        st.success(f"📉 RSI 超賣提示：目前 RSI = **{latest_rsi:.2f}**，已進入超賣區域（<30）。歷史數據顯示此區域曾出現反彈現象，但不代表未來走勢。")
                    else:
                        st.info(f"📊 RSI 中性：目前 RSI = **{latest_rsi:.2f}**，位於中性區域（30–70）。")

                    # ── 顯示 9（台股）：籌碼面分析（T09規格第9項）──
                    if is_tw:
                        st.markdown("### 🧩 籌碼面分析（三大法人 ＋ 主力分點 ＋ 散戶籌碼比）")
                        st.caption("📡 資料來源：FinMind TaiwanStockInstitutionalInvestorsBuySell / TaiwanStockTradingDailyReport / TaiwanStockMarginPurchaseShortSale")

                        _chip_tab1, _chip_tab2, _chip_tab3, _chip_tab4 = st.tabs([
                            "🏦 三大法人30日", "🏢 主力分點進出", "📊 散戶籌碼比", "📈 走勢總圖"
                        ])

                        # ─── 頁籤1：三大法人30天詳細 ─────────────────
                        with _chip_tab1:
                            if institutional_df is not None and len(institutional_df) > 0:
                                _inst = institutional_df.copy()

                                # 法人名稱中文化
                                _name_map = {
                                    'Foreign_Investor':  '外資',
                                    'Investment_Trust':  '投信',
                                    'Dealer_self':       '自營商(自行)',
                                    'Dealer_Hedging':    '自營商(避險)',
                                    'Dealer':            '自營商',
                                    'Total':             '三大法人合計',
                                }
                                if 'name' in _inst.columns:
                                    _inst['法人'] = _inst['name'].map(_name_map).fillna(_inst['name'])

                                # ── 統計卡：外資/投信/自營累計買賣超 ──
                                _inst30 = _inst[_inst['date'] >= (datetime.now() - timedelta(days=30))]
                                _summ_rows = []
                                for _eng, _chn in [('Foreign_Investor','外資'),
                                                    ('Investment_Trust','投信'),
                                                    ('Dealer_self','自營商(自行)'),
                                                    ('Dealer_Hedging','自營商(避險)'),
                                                    ('Total','三大法人合計')]:
                                    _sub = _inst30[_inst30['name'] == _eng] if 'name' in _inst30.columns else pd.DataFrame()
                                    if not _sub.empty:
                                        _net_sum  = int(_sub['net'].sum())
                                        _buy_sum  = int(_sub['buy'].sum())
                                        _sell_sum = int(_sub['sell'].sum())
                                        # 連買/連賣天數
                                        _nets = _sub.sort_values('date')['net'].tolist()
                                        _streak = 0
                                        for _v in reversed(_nets):
                                            if _v > 0: _streak += 1
                                            else: break
                                        _streak_str = f"連買{_streak}天" if _streak >= 2 else (
                                                       f"連賣{abs(_streak)}天" if _streak < 0 else "")
                                        _summ_rows.append({
                                            '法人': _chn,
                                            '30日淨買超(張)': f"{_net_sum:+,}",
                                            '買進合計(張)': f"{_buy_sum:,}",
                                            '賣出合計(張)': f"{_sell_sum:,}",
                                            '連續狀態': _streak_str,
                                            '方向': '🟢 買超' if _net_sum > 0 else ('🔴 賣超' if _net_sum < 0 else '⚪ 持平'),
                                        })
                                if _summ_rows:
                                    _summ_df = pd.DataFrame(_summ_rows)
                                    st.dataframe(_summ_df, use_container_width=True, hide_index=True)

                                # ── 近10交易日逐日明細（pivot 表）──
                                st.markdown("##### 近10交易日逐日買賣超（張）")
                                display_institutional_table(institutional_df)

                                # ── 外資買超佔比分析 ──
                                if 'name' in _inst30.columns:
                                    _foreign = _inst30[_inst30['name'] == 'Foreign_Investor']
                                    if not _foreign.empty and 'volume' in data_with_indicators.columns:
                                        _avg_vol = data_with_indicators.tail(30)['volume'].mean()
                                        _foreign_net_avg = abs(_foreign['net']).mean()
                                        if _avg_vol > 0:
                                            _pct = _foreign_net_avg / _avg_vol * 100
                                            _label = "🟢 主動推升" if _pct > 20 else "⚪ 正常"
                                            st.info(f"外資買超佔成交量比重：約 **{_pct:.1f}%**（> 20% 視為主動推升）{_label}")
                            else:
                                st.warning("⚠️ 三大法人資料取得失敗，請確認 FinMind API Key 是否為付費 Backer 以上方案。")
                                st.markdown(f"🔗 [手動查詢三大法人](https://goodinfo.tw/tw/StockBuySaleByLegalPerson.asp?STOCK_ID={symbol.strip()})")

                        # ─── 頁籤2：主力券商分點 ─────────────────────
                        with _chip_tab2:
                            if broker_df is not None:
                                _src_broker = broker_df.get('source', 'TWSE') if isinstance(broker_df, dict) else 'TWSE'
                                st.caption(f"📡 資料來源：{_src_broker} | 查詢日期：{broker_date}")

                                # 主力特徵判斷
                                if isinstance(broker_df, dict):
                                    _buy_df  = broker_df.get('buy_df', pd.DataFrame())
                                    _sell_df = broker_df.get('sell_df', pd.DataFrame())
                                    _tb = broker_df.get('total_buy', 0)
                                    _ts = broker_df.get('total_sell', 0)
                                else:
                                    _buy_df  = broker_df[broker_df.get('net', 0) > 0] if 'net' in broker_df.columns else pd.DataFrame()
                                    _sell_df = broker_df[broker_df.get('net', 0) < 0] if 'net' in broker_df.columns else pd.DataFrame()
                                    _tb = int(broker_df['buy'].sum()) if 'buy' in broker_df.columns else 0
                                    _ts = int(broker_df['sell'].sum()) if 'sell' in broker_df.columns else 0

                                _c1, _c2, _c3 = st.columns(3)
                                with _c1: st.metric("主力買超分點數", f"{len(_buy_df)} 家")
                                with _c2: st.metric("主力賣超分點數", f"{len(_sell_df)} 家")
                                with _c3:
                                    _tv = _tb + _ts
                                    if not _buy_df.empty and _tv > 0 and 'buy' in _buy_df.columns and 'sell' in _buy_df.columns:
                                        _top1_vol = int(_buy_df.iloc[0]['buy']) + int(_buy_df.iloc[0]['sell']) if len(_buy_df) > 0 else 0
                                        _conc = _top1_vol / _tv * 100
                                        st.metric("主力最大分點集中度", f"{_conc:.1f}%",
                                                  delta="⚠️ 主力介入" if _conc > 10 else "⚪ 正常")

                                st.markdown("##### 主力淨買超前10大分點")
                                display_broker_table(broker_df, symbol.strip(), query_date=broker_date)
                            else:
                                st.warning("⚠️ 券商分點資料暫無（可能為非交易日或資料尚未更新）")
                                st.markdown(f"🔗 [GoodInfo 主力進出](https://goodinfo.tw/tw/StockBuySaleByBroker.asp?STOCK_ID={symbol.strip()})")

                        # ─── 頁籤3：散戶籌碼比 ───────────────────────
                        with _chip_tab3:
                            if margin_df is not None and len(margin_df) > 0:
                                _mg = margin_df.copy()
                                _latest = _mg.iloc[-1]
                                _prev30 = _mg.iloc[0]

                                _margin_now   = _latest.get('MarginPurchaseRemaining', 0) or 0
                                _margin_prev  = _prev30.get('MarginPurchaseRemaining', 0) or 0
                                _short_now    = _latest.get('ShortSaleRemaining', 0) or 0
                                _short_prev   = _prev30.get('ShortSaleRemaining', 0) or 0
                                _margin_chg   = _margin_now - _margin_prev
                                _short_chg    = _short_now  - _short_prev
                                _margin_chg_p = (_margin_chg / _margin_prev * 100) if _margin_prev > 0 else 0
                                _short_chg_p  = (_short_chg  / _short_prev  * 100) if _short_prev  > 0 else 0
                                _yuan_bi      = (_short_now / _margin_now * 100) if _margin_now > 0 else 0

                                # 散戶健康度判斷
                                _health_score = 0
                                if _margin_chg < 0: _health_score += 1   # 融資減：去槓桿，好
                                if _short_chg  > 0: _health_score += 1   # 融券增：軋空潛力，好
                                if _yuan_bi    > 15: _health_score += 1  # 券資比高：軋空題材
                                _health_label = {0:"🔴 籌碼不乾淨", 1:"🟡 中性", 2:"🟢 籌碼健康", 3:"🟢🟢 籌碼最乾淨"}[_health_score]

                                _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                                with _mc1:
                                    st.metric("融資餘額(張)", f"{int(_margin_now):,}",
                                              delta=f"{_margin_chg_p:+.1f}% 30日",
                                              delta_color="inverse")   # 融資減是好事，顏色反轉
                                with _mc2:
                                    st.metric("融券餘額(張)", f"{int(_short_now):,}",
                                              delta=f"{_short_chg_p:+.1f}% 30日",
                                              delta_color="normal")
                                with _mc3:
                                    st.metric("券資比", f"{_yuan_bi:.1f}%",
                                              delta="軋空題材" if _yuan_bi > 15 else "正常",
                                              delta_color="normal" if _yuan_bi > 15 else "off")
                                with _mc4:
                                    st.metric("散戶健康度", _health_label)

                                # 融資融券走勢圖
                                _fig_mg = go.Figure()
                                _fig_mg.add_trace(go.Scatter(
                                    x=_mg['date'], y=_mg['MarginPurchaseRemaining'],
                                    name='融資餘額(張)', line=dict(color='#E24B4A', width=1.5),
                                    hovertemplate='%{x|%Y-%m-%d}<br>融資：%{y:,}張<extra></extra>'
                                ))
                                if 'ShortSaleRemaining' in _mg.columns:
                                    _fig_mg.add_trace(go.Scatter(
                                        x=_mg['date'], y=_mg['ShortSaleRemaining'],
                                        name='融券餘額(張)', line=dict(color='#378ADD', width=1.5, dash='dot'),
                                        yaxis='y2',
                                        hovertemplate='%{x|%Y-%m-%d}<br>融券：%{y:,}張<extra></extra>'
                                    ))
                                _fig_mg.update_layout(
                                    title=f"{symbol.strip()} 融資融券餘額（近30天）",
                                    xaxis_title="日期", yaxis_title="融資餘額(張)",
                                    yaxis2=dict(title="融券餘額(張)", overlaying='y', side='right'),
                                    height=300, hovermode='x unified',
                                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                    legend=dict(orientation='h', y=1.05),
                                    margin=dict(l=50,r=50,t=50,b=40)
                                )
                                st.plotly_chart(_fig_mg, use_container_width=True)

                                # 明細表格
                                _mg_disp = _mg.tail(10)[['date'] + [c for c in [
                                    'MarginPurchaseRemaining','ShortSaleRemaining',
                                    'MarginPurchaseBuy','MarginPurchaseSell',
                                    'ShortSaleBuy','ShortSaleSell'
                                ] if c in _mg.columns]].copy()
                                _mg_disp.rename(columns={
                                    'date':'日期',
                                    'MarginPurchaseRemaining':'融資餘額',
                                    'ShortSaleRemaining':'融券餘額',
                                    'MarginPurchaseBuy':'融資買進',
                                    'MarginPurchaseSell':'融資賣出',
                                    'ShortSaleBuy':'融券買進(回補)',
                                    'ShortSaleSell':'融券賣出',
                                }, inplace=True)
                                _mg_disp['日期'] = pd.to_datetime(_mg_disp['日期']).dt.strftime('%Y-%m-%d')
                                st.dataframe(_mg_disp, use_container_width=True, hide_index=True)
                            else:
                                st.warning("⚠️ 融資融券資料取得失敗，請確認 FinMind API Key。")
                                st.markdown(f"🔗 [手動查詢融資融券](https://goodinfo.tw/tw/StockMarginTrading.asp?STOCK_ID={symbol.strip()})")

                        # ─── 頁籤4：三大法人走勢總圖 ─────────────────
                        with _chip_tab4:
                            if institutional_df is not None and len(institutional_df) > 0:
                                _inst_plot = institutional_df.copy()
                                _inst_plot['date'] = pd.to_datetime(_inst_plot['date'])

                                _fig_chip = go.Figure()
                                _color_map = {
                                    'Foreign_Investor':  ('#185FA5','外資'),
                                    'Investment_Trust':  ('#1D9E75','投信'),
                                    'Dealer_self':       ('#BA7517','自營商(自行)'),
                                }
                                for _eng, (_clr, _chn) in _color_map.items():
                                    _sub = _inst_plot[_inst_plot['name'] == _eng] if 'name' in _inst_plot.columns else pd.DataFrame()
                                    if not _sub.empty:
                                        _fig_chip.add_trace(go.Bar(
                                            x=_sub['date'], y=_sub['net'],
                                            name=_chn, marker_color=_clr, opacity=0.8,
                                            hovertemplate=f'{_chn} %{{x|%Y-%m-%d}}<br>買賣超：%{{y:,}}張<extra></extra>'
                                        ))
                                _fig_chip.update_layout(
                                    title=f"{symbol.strip()} 三大法人買賣超走勢（近30天）",
                                    barmode='group', xaxis_title='日期', yaxis_title='買賣超(張)',
                                    height=360, hovermode='x unified',
                                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                    legend=dict(orientation='h', y=1.05),
                                    margin=dict(l=50,r=20,t=50,b=40)
                                )
                                _fig_chip.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
                                st.plotly_chart(_fig_chip, use_container_width=True)
                            else:
                                st.info("ℹ️ 無三大法人資料可繪製走勢圖")


                    # ── 顯示 11：朱家泓趨勢線系統 ──
                    display_zhu_trend_dashboard(
                        zhu_result,
                        symbol.strip() if is_tw else symbol.upper(),
                        currency_symbol=currency_symbol
                    )

                    # ── 顯示 11b：K線型態分析系統 ──
                    display_kline_pattern_dashboard(
                        kline_result,
                        symbol.strip() if is_tw else symbol.upper()
                    )

                    # ── 顯示 16：選股評量表格（分析完成後自動 AI 填入）──
                    st.markdown("---")
                    st.markdown("### 📋 選股評量表格")
                    st.caption("依據朱家泓《多空操作秘笈》圖表10-3-1 — AI 自動填入，可手動微調後下載")

                    _eval_symbol = symbol.strip() if is_tw else symbol.upper()
                    _eval_key    = f"eval_{_eval_symbol}"

                    # 每次分析自動 AI 填入（以「股票+小時」作為快取鍵，避免重複呼叫）
                    _ts_now = f"{_eval_symbol}_{datetime.now().strftime('%Y%m%d%H')}"
                    if st.session_state.get("last_eval_ts") != _ts_now or not st.session_state.get(_eval_key):
                        _ai_result = generate_stock_evaluation(
                            symbol=_eval_symbol,
                            stock_data=data_with_indicators,
                            openai_api_key=openai_api_key,
                            market=market_key,
                            zhu_result=zhu_result,
                            bull_signals=bull_signals,
                            kline_result=kline_result,
                            rsi_period=rsi_period,
                            institutional_df=institutional_df if is_tw else None,
                            margin_df=margin_df if is_tw else None,
                            analyst_data=analyst_data,
                        )
                        if "_error" not in _ai_result:
                            st.session_state[_eval_key] = _ai_result
                            st.session_state["last_eval_ts"] = _ts_now
                        else:
                            st.session_state.setdefault(_eval_key, {})

                    with st.expander("📝 選股評量表格（點擊展開／收合）", expanded=True):
                        # 標題資訊列
                        _col_date, _col_biz, _col_cap = st.columns([1, 2, 1])
                        with _col_date:
                            _eval_date = st.date_input("日期", value=datetime.now().date(),
                                key=f"{_eval_key}_date")
                        with _col_biz:
                            _eval_biz = st.text_input("營業項目",
                                value=st.session_state[_eval_key].get("營業項目", ""),
                                key=f"{_eval_key}_biz")
                        with _col_cap:
                            _eval_cap = st.text_input("股本（億）",
                                value=st.session_state[_eval_key].get("股本", ""),
                                key=f"{_eval_key}_cap")

                        st.markdown("#### 基本面")
                        _eval_fundamental = st.text_area("基本面說明",
                            value=st.session_state[_eval_key].get("基本面", ""),
                            height=60, key=f"{_eval_key}_fundamental",
                            label_visibility="collapsed",
                            placeholder="輸入基本面重點（EPS、營收、產業趨勢等）...")

                        # 表格左右兩欄對應結構
                        _left_col, _right_col = st.columns(2)

                        with _left_col:
                            st.markdown("**波型（V）**")
                            _c1, _c2, _c3 = st.columns(3)
                            _eval_wave_m = _c1.text_input("月線", value=st.session_state[_eval_key].get("波型_月線",""), key=f"{_eval_key}_wave_m", placeholder="↑/↓/震盪")
                            _eval_wave_w = _c2.text_input("週線", value=st.session_state[_eval_key].get("波型_週線",""), key=f"{_eval_key}_wave_w", placeholder="↑/↓/震盪")
                            _eval_wave_d = _c3.text_input("日線", value=st.session_state[_eval_key].get("波型_日線",""), key=f"{_eval_key}_wave_d", placeholder="↑/↓/震盪")

                            st.markdown("**位置**")
                            _c1, _c2 = st.columns(2)
                            _eval_pos_w = _c1.text_input("週線位置", value=st.session_state[_eval_key].get("位置_週線",""), key=f"{_eval_key}_pos_w", placeholder="底/中/頂")
                            _eval_pos_d = _c2.text_input("日線位置", value=st.session_state[_eval_key].get("位置_日線",""), key=f"{_eval_key}_pos_d", placeholder="底/中/頂")

                            st.markdown("**K線（V）**")
                            _c1, _c2 = st.columns(2)
                            _eval_k_w = _c1.text_input("週K線", value=st.session_state[_eval_key].get("K線_週",""), key=f"{_eval_key}_k_w", placeholder="型態描述")
                            _eval_k_d = _c2.text_input("日K線", value=st.session_state[_eval_key].get("K線_日",""), key=f"{_eval_key}_k_d", placeholder="型態描述")

                            st.markdown("**均線（V）／切線（V）**")
                            _c1, _c2, _c3 = st.columns(3)
                            _eval_ma_w = _c1.text_input("週均線", value=st.session_state[_eval_key].get("均線_週",""), key=f"{_eval_key}_ma_w", placeholder="多頭/空頭")
                            _eval_ma_d = _c2.text_input("日均線", value=st.session_state[_eval_key].get("均線_日",""), key=f"{_eval_key}_ma_d", placeholder="多頭/空頭")
                            _eval_ma_cut = _c3.text_input("切線", value=st.session_state[_eval_key].get("切線",""), key=f"{_eval_key}_ma_cut", placeholder="向上/向下")

                            st.markdown("**成交量（V）**")
                            _c1, _c2 = st.columns(2)
                            _eval_vol_w = _c1.text_input("週量", value=st.session_state[_eval_key].get("成交量_週",""), key=f"{_eval_key}_vol_w", placeholder="放量/縮量")
                            _eval_vol_d = _c2.text_input("日量", value=st.session_state[_eval_key].get("成交量_日",""), key=f"{_eval_key}_vol_d", placeholder="放量/縮量")

                            st.markdown("**指標（V）**")
                            _c1, _c2, _c3, _c4 = st.columns(4)
                            _eval_ind_wmacd = _c1.text_input("週MACD", value=st.session_state[_eval_key].get("指標_週MACD",""), key=f"{_eval_key}_ind_wmacd", placeholder="金/死叉")
                            _eval_ind_wkd   = _c2.text_input("週KD",   value=st.session_state[_eval_key].get("指標_週KD",""),   key=f"{_eval_key}_ind_wkd",   placeholder="金/死叉")
                            _eval_ind_dmacd = _c3.text_input("日MACD", value=st.session_state[_eval_key].get("指標_日MACD",""), key=f"{_eval_key}_ind_dmacd", placeholder="金/死叉")
                            _eval_ind_dkd   = _c4.text_input("日KD",   value=st.session_state[_eval_key].get("指標_日KD",""),   key=f"{_eval_key}_ind_dkd",   placeholder="金/死叉")

                        with _right_col:
                            st.markdown("**支撐**")
                            _c1, _c2 = st.columns(2)
                            _eval_sup_w = _c1.text_input("週支撐", value=st.session_state[_eval_key].get("支撐_週",""), key=f"{_eval_key}_sup_w", placeholder="價位")
                            _eval_sup_d = _c2.text_input("日支撐", value=st.session_state[_eval_key].get("支撐_日",""), key=f"{_eval_key}_sup_d", placeholder="價位")

                            st.markdown("**壓力**")
                            _c1, _c2 = st.columns(2)
                            _eval_res_w = _c1.text_input("週壓力", value=st.session_state[_eval_key].get("壓力_週",""), key=f"{_eval_key}_res_w", placeholder="價位")
                            _eval_res_d = _c2.text_input("日壓力", value=st.session_state[_eval_key].get("壓力_日",""), key=f"{_eval_key}_res_d", placeholder="價位")

                            st.markdown("**背離**")
                            _eval_diverge = st.text_input("背離描述", value=st.session_state[_eval_key].get("背離",""), key=f"{_eval_key}_diverge", placeholder="RSI/MACD 頂背離或底背離")

                            st.markdown("**融資融券（V）**")
                            _c1, _c2, _c3 = st.columns(3)
                            _eval_margin   = _c1.text_input("融資",   value=st.session_state[_eval_key].get("融資",""),   key=f"{_eval_key}_margin",   placeholder="增/減")
                            _eval_short    = _c2.text_input("融券",   value=st.session_state[_eval_key].get("融券",""),   key=f"{_eval_key}_short",    placeholder="增/減")
                            _eval_mratio   = _c3.text_input("融資比", value=st.session_state[_eval_key].get("融資比",""), key=f"{_eval_key}_mratio",   placeholder="百分比")

                            st.markdown("**法人買賣超**")
                            _eval_inst = st.text_input("法人買賣超", value=st.session_state[_eval_key].get("法人買賣超",""), key=f"{_eval_key}_inst", placeholder="外資/投信/自營合計")

                            st.markdown("**型態**")
                            _eval_pattern = st.text_input("型態描述", value=st.session_state[_eval_key].get("型態",""), key=f"{_eval_key}_pattern", placeholder="頭肩底/雙底/杯柄/旗形...")

                            st.markdown("**其他**")
                            _eval_other = st.text_input("其他備註", value=st.session_state[_eval_key].get("其他",""), key=f"{_eval_key}_other", placeholder="財報、重大訊息、產業動態...")

                        # 策略欄位（全寬）
                        st.markdown("**📌 策略**")
                        _eval_strategy = st.text_area("操作策略",
                            value=st.session_state[_eval_key].get("策略", ""),
                            height=80, key=f"{_eval_key}_strategy",
                            label_visibility="collapsed",
                            placeholder="記錄進出場條件、停損停利設定、操作方向（做多/做空/觀望）...")

                        # ── 即時同步所有欄位到 session_state（無需按鈕，不觸發 rerun）──
                        _eval_snapshot = {
                            "日期": str(_eval_date), "營業項目": _eval_biz, "股本": _eval_cap,
                            "基本面": _eval_fundamental,
                            "波型_月線": _eval_wave_m, "波型_週線": _eval_wave_w, "波型_日線": _eval_wave_d,
                            "位置_週線": _eval_pos_w, "位置_日線": _eval_pos_d,
                            "K線_週": _eval_k_w, "K線_日": _eval_k_d,
                            "均線_週": _eval_ma_w, "均線_日": _eval_ma_d, "切線": _eval_ma_cut,
                            "成交量_週": _eval_vol_w, "成交量_日": _eval_vol_d,
                            "指標_週MACD": _eval_ind_wmacd, "指標_週KD": _eval_ind_wkd,
                            "指標_日MACD": _eval_ind_dmacd, "指標_日KD": _eval_ind_dkd,
                            "支撐_週": _eval_sup_w, "支撐_日": _eval_sup_d,
                            "壓力_週": _eval_res_w, "壓力_日": _eval_res_d,
                            "背離": _eval_diverge,
                            "融資": _eval_margin, "融券": _eval_short, "融資比": _eval_mratio,
                            "法人買賣超": _eval_inst, "型態": _eval_pattern, "其他": _eval_other,
                            "策略": _eval_strategy,
                        }
                        # 靜默同步，不觸發 rerun（值已在 widget key 裡，session_state 更新不影響畫面）
                        st.session_state[_eval_key].update(_eval_snapshot)

                        # ── 下載 CSV（download_button 不觸發 rerun，安全使用）──
                        st.markdown("---")
                        _eval_df  = pd.DataFrame([_eval_snapshot])
                        _eval_csv = _eval_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 下載評量表 CSV",
                            _eval_csv,
                            file_name=f"選股評量_{_eval_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key=f"{_eval_key}_download"
                        )

                    # ── 顯示 15：長短線投資建議 ──
                    st.markdown("---")
                    st.markdown("### 💡 長短線投資建議")
                    with st.spinner("AI 正在生成長短線投資建議..."):
                        investment_advice = generate_investment_advice(
                            symbol=symbol.strip() if is_tw else symbol.upper(),
                            stock_data=data_with_indicators,
                            openai_api_key=openai_api_key,
                            market=market_key,
                            bull_signals=bull_signals,
                            zhu_result=zhu_result,
                            kline_result=kline_result,
                            rsi_period=rsi_period,
                            institutional_df=institutional_df if is_tw else None,
                            margin_df=margin_df if is_tw else None,
                            analyst_data=analyst_data,
                        )
                    if investment_advice:
                        st.markdown(investment_advice)


                    st.success("✅ 分析完成！")
            else:
                st.warning("所選日期範圍內沒有交易數據，請調整日期範圍。")
        else:
            st.error("無法獲取股票數據，請檢查股票代碼和API金鑰。")


# ─────────────────────────────────────────────
# 初始歡迎頁面
# ─────────────────────────────────────────────

if not analyze_button:
    st.markdown("""
## 歡迎使用 AI 股票趨勢分析系統 👋

### 🚀 功能特色
- **雙市場支援**: 美股（FMP API）與台股（FinMind API），下拉選單一鍵切換
- **📈 基本統計資訊**: 起始/結束價格、漲跌幅、最新 RSI 一覽
- **🎯 法人目標價分析**: AI 新聞彙整 + 第三方數據源（鉅亨網／GoodInfo／Finviz／FMP）多來源整合
- **🚦 多頭訊號儀表板**: 8項指標燈號（🟢🟡🔴）+ 整體評分（0–100分）
- **📊 主K線圖**: 布林通道 + 可切換MA線 + OBV + 成交量
- **🧩 籌碼面分析（台股）**: 三大法人30日 + 主力分點進出 + 散戶籌碼比（融資融券）+ 走勢總圖
- **📐 趨勢線系統分析**: 朱家泓《趨勢線》教材規則，多頭確認/趨勢轉換/進出場判斷
- **🕯️ K線型態分析系統**: 朱家泓《多空操作秘笈》K線型態辨識
- **📋 選股評量表格**: 依據圖表10-3-1，AI 自動填入，可手動微調後下載 CSV
- **💡 長短線投資建議**: 整合技術/籌碼/法人面，獨立輸出短線（1–4週）與長線（1–6月）操作建議、進出場條件、停損停利與風險提示

### 📝 使用方法
1. 在左側選擇市場（美股 / 台股）
2. 輸入股票代碼（美股如：AAPL；台股純數字如：2330）
3. 輸入對應的 API 金鑰與 OpenAI API Key
4. 選擇分析的日期範圍（預設1年）、設定 RSI 計算天數
5. 選擇要顯示的移動平均線（MA5/MA10/MA20/MA60）
6. 點擊「🚀 開始分析」按鈕

### 🚦 多頭訊號儀表板（8項）
MACD轉正 / BB突破中軌 / BB壓縮突破 / OBV資金流入 / RSI動量 / DMI多頭趨勢 / DMI黃金交叉 / 均線多頭排列
- **🟢 綠燈（12.5分）** | **🟡 黃燈（6分）** | **🔴 紅燈（0分）**
- 評分 ≥70：多頭確認 | 40–69：訊號混合 | <40：條件不符

### 🗂️ 台股籌碼面分析
- **三大法人**: 外資/投信/自營商每日買賣超，中文欄位+30日彙總
- **主力分點**: TWSE 券商分點進出，主力買超/賣超分點與集中度
- **散戶籌碼比**: 融資/融券餘額、券資比、散戶健康度判斷

### 🔑 API 金鑰獲取
- **FMP API（美股）**: [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs)
- **FinMind API（台股）**: [FinMind Trade](https://finmindtrade.com/)（免費方案每日有請求次數限制）
- **OpenAI API**: [OpenAI Platform](https://platform.openai.com)

---
**開始您的技術分析之旅吧！** 📈
""")
