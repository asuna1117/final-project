import pandas as pd
import requests
import time
import re
import numpy as np
import yfinance as yf
import io
import contextlib
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import warnings
import json

from tej_client import TejClient

warnings.filterwarnings('ignore')

# --- 基礎設定 ---
list_url = "https://norway.twsthr.info/StockHoldersDividendTop.aspx?CID=0&Show=1"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stock_data_cache")      

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def _get_tej_client():
    try:
        client = TejClient()
        return client if client.configured else None
    except Exception:
        return None


def _quiet_yf_download(*args, **kwargs):
    """Silence yfinance console noise and return dataframe as-is."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return yf.download(*args, **kwargs)

# ==========================================
# 股價與爬蟲輔助功能
# ==========================================
def download_stock_price_history(stock_id, force_update=False):
    """下載單一股票 3 年歷史價格數據並存成 CSV (快取 12 小時)"""
    price_file = os.path.join(DATA_DIR, f"{stock_id}_price_history.csv")
    
    # 檢查快取是否有效
    if os.path.exists(price_file):
        file_age = time.time() - os.path.getmtime(price_file)
        if file_age <= 43200 and not force_update:
            try:
                df = pd.read_csv(price_file, index_col='Date', parse_dates=True)
                if not df.empty:
                    return df
            except:
                pass
    
    # 計算 3 年前的日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    
    # 嘗試兩種股票代碼後綴
    for suffix in [".TW", ".TWO"]:
        ticker = f"{stock_id}{suffix}"
        try:
            print(f"  ↓ 下載 {ticker} 價格歷史...", end=" ", flush=True)
            data = _quiet_yf_download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False
            )
            
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                    
                # 保留必要欄位
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                data.index.name = 'Date'
                data.to_csv(price_file, encoding='utf-8-sig')
                print(f"✓ 成功({len(data)}筆)")
                return data
        except Exception as e:
            print(f"✗ 失敗")
            continue
    
    print(f"✗ 無法取得")
    return None

def get_next_monday_open_price(stock_id, signal_date):
    """計算並從本地價格歷史取得下週一的開盤價"""
    try:
        signal_dt = datetime.strptime(str(signal_date), "%Y%m%d")
        days_until_monday = (7 - signal_dt.weekday()) % 7
        if days_until_monday == 0: days_until_monday = 7
        next_monday = signal_dt + timedelta(days=days_until_monday)

        # 先確保有下載該股票的價格歷史
        price_df = download_stock_price_history(stock_id)
        if price_df is None or price_df.empty:
            return np.nan
        
        # 從 DataFrame 中查找下週一的開盤價
        next_monday_str = next_monday.strftime("%Y-%m-%d")
        if next_monday_str in price_df.index.strftime("%Y-%m-%d").values:
            idx = price_df.index.strftime("%Y-%m-%d") == next_monday_str
            if idx.any():
                return float(price_df.loc[idx, 'Open'].iloc[0])
        
        return np.nan
    except Exception:
        return np.nan

def get_next_friday_close_price(stock_id, signal_date):
    """計算並從本地價格歷史取得下週五的收盤價，若無則往前查找最近交易日"""
    try:
        signal_dt = datetime.strptime(str(signal_date), "%Y%m%d")
        days_until_monday = (7 - signal_dt.weekday()) % 7
        if days_until_monday == 0: days_until_monday = 7
        next_monday = signal_dt + timedelta(days=days_until_monday)
        next_friday = next_monday + timedelta(days=4)

        # 先確保有下載該股票的價格歷史
        price_df = download_stock_price_history(stock_id)
        if price_df is None or price_df.empty:
            return np.nan
        
        # 先嘗試找下週五，若無則往前查找最多 5 個交易日
        # 但不能往前查超過進場日期 (signal_dt)
        for days_back in range(0, 5):
            target_date = next_friday - timedelta(days=days_back)

            # 若查到的日期早於進場日期，停止往前查找
            if target_date < signal_dt:
                break

            target_date_str = target_date.strftime("%Y-%m-%d")
            
            date_strs = price_df.index.strftime("%Y-%m-%d").values
            if target_date_str in date_strs:
                idx = price_df.index.strftime("%Y-%m-%d") == target_date_str
                if idx.any():
                    return float(price_df.loc[idx, 'Close'].iloc[0])

        return np.nan
    except Exception:
        return np.nan
    
def check_condition_e_with_yfinance(stock_id, signal_date, monday_open_price):
    """條件 E: 從本地價格歷史檢查下週二收盤 > 週一開盤，且週三、週四收盤連續走高。若無則往前查找最近交易日，但不超過進場日期。"""
    try:
        signal_dt = datetime.strptime(str(signal_date), "%Y%m%d")
    except ValueError:
        return False

    days_until_monday = (7 - signal_dt.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7

    next_monday = signal_dt + timedelta(days=days_until_monday)
    next_tuesday = next_monday + timedelta(days=1)
    next_wednesday = next_monday + timedelta(days=2)
    next_thursday = next_monday + timedelta(days=3)
    target_dates = [next_tuesday, next_wednesday, next_thursday]

    # 先確保有下載該股票的價格歷史
    price_df = download_stock_price_history(stock_id)
    if price_df is None or price_df.empty:
        return False
    
    # 轉換索引為日期字符串便於查詢
    price_df['Date_str'] = price_df.index.strftime("%Y-%m-%d")
    
    # 為每個目標日期查找收盤價（往前查找最多 5 天），但不能查到進場日前的價格
    closes = []
    for target_date in target_dates:
        found = False
        for days_back in range(0, 5):
            search_date = target_date - timedelta(days=days_back)

            # 不允許查到進場日期之前的價格
            if search_date < signal_dt:
                break

            search_date_str = search_date.strftime("%Y-%m-%d")
            
            if search_date_str in price_df['Date_str'].values:
                close_price = float(price_df[price_df['Date_str'] == search_date_str]['Close'].iloc[0])
                closes.append(close_price)
                found = True
                break
        
        if not found:
            return False  # 如果某一天怎麼都找不到，就視為條件不成立
    
    tue_close, wed_close, thu_close = closes
    
    # 條件 E: 下週二收盤 > 週一開盤，且週三、週四收盤連續走高
    return tue_close > monday_open_price and wed_close > tue_close and thu_close > wed_close


def _twse_t86_url(date_str):
    return (
        "https://www.twse.com.tw/rwd/zh/fund/T86"
        f"?date={date_str}&selectType=ALLBUT0999&response=json"
    )


def download_twse_foreign_buy_sell(target_date=None, force_update=False):
    """下載 TWSE 公開的三大法人資料，包含外資及陸資買賣超。

    回傳標準化 DataFrame，欄位至少包含：
    - 資料日期
    - 股票代號
    - 股票名稱
    - 外資及陸資買賣超股數
    - 投信買賣超股數
    - 自營商買賣超股數
    - 三大法人買賣超股數
    """
    if target_date is None:
        target_date = datetime.now()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y%m%d")

    if isinstance(target_date, datetime):
        date_str = target_date.strftime("%Y%m%d")
    else:
        date_str = str(target_date)

    cache_file = os.path.join(DATA_DIR, f"twse_t86_{date_str}.csv")
    if os.path.exists(cache_file) and not force_update:
        try:
            cached = pd.read_csv(cache_file)
            if not cached.empty:
                return cached
        except Exception:
            pass

    try:
        r = requests.get(_twse_t86_url(date_str), headers=headers, timeout=10)
        r.encoding = "utf-8"

        try:
            payload = r.json()
        except Exception:
            payload = json.loads(r.text)

        if str(payload.get("stat", "")).strip() not in {"OK", "ok"}:
            return pd.DataFrame()

        fields = payload.get("fields", [])
        rows = payload.get("data", [])
        if not fields or not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=fields)

        rename_map = {
            "證券代號": "股票代號",
            "證券名稱": "股票名稱",
            "外資及陸資買賣超股數": "外資買賣超",
            "外陸資買賣超股數(不含外資自營商)": "外資買賣超",
            "外陸資買進股數(不含外資自營商)": "外資買進股數",
            "外陸資賣出股數(不含外資自營商)": "外資賣出股數",
            "外資自營商買賣超股數": "外資自營商買賣超",
            "外資自營商買進股數": "外資自營商買進股數",
            "外資自營商賣出股數": "外資自營商賣出股數",
            "投信買賣超股數": "投信買賣超",
            "自營商買賣超股數": "自營商買賣超",
            "三大法人買賣超股數": "三大法人買賣超",
        }
        df = df.rename(columns=rename_map)

        if "股票代號" in df.columns:
            df["股票代號"] = df["股票代號"].astype(str).str.strip()
        if "股票名稱" in df.columns:
            df["股票名稱"] = df["股票名稱"].astype(str).str.strip()

        numeric_cols = [
            "外資買賣超",
            "外資買進股數",
            "外資賣出股數",
            "外資自營商買賣超",
            "外資自營商買進股數",
            "外資自營商賣出股數",
            "投信買賣超",
            "自營商買賣超",
            "三大法人買賣超",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("--", "", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["資料日期"] = date_str
        df = df.dropna(subset=["股票代號"]).reset_index(drop=True)
        df.to_csv(cache_file, index=False, encoding="utf-8-sig")
        return df
    except Exception:
        return pd.DataFrame()


def get_today_foreign_buy_sell(force_update=False):
    """取得今天或最近可用交易日的外資買賣超資料。"""
    today = datetime.now()

    for offset in range(0, 7):
        candidate = today - timedelta(days=offset)
        # 先嘗試每個曆日；若不是交易日，TWSE 會回傳非 OK
        df = download_twse_foreign_buy_sell(candidate, force_update=force_update)
        if df is not None and not df.empty:
            return df

    return pd.DataFrame()


def preload_twse_foreign_buy_sell(days_back=7, force_update=False):
    """先批次下載最近幾個交易日的 TWSE 外資買賣超 CSV，供後續離線判斷使用。"""
    today = datetime.now()
    loaded = []

    for offset in range(0, days_back + 1):
        candidate = today - timedelta(days=offset)
        date_str = candidate.strftime("%Y%m%d")
        cache_file = os.path.join(DATA_DIR, f"twse_t86_{date_str}.csv")

        if os.path.exists(cache_file) and not force_update:
            loaded.append(cache_file)
            continue

        df = download_twse_foreign_buy_sell(candidate, force_update=force_update)
        if df is not None and not df.empty:
            loaded.append(cache_file)

    return loaded


def _extract_float(value):
    try:
        if pd.isna(value):
            return np.nan
        text = str(value).replace(',', '').strip()
        if text in {'', '--', 'nan', 'None'}:
            return np.nan
        return float(text)
    except Exception:
        return np.nan


def _twse_margin_url(date_str):
    return (
        "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
        f"?date={date_str}&selectType=ALL&response=json"
    )


def download_twse_margin_data(target_date=None, force_update=False):
    """下載 TWSE 公開融資融券資料，回傳標準化 DataFrame。

    目前 TWSE 公開端點回傳的是市場層級的「信用交易統計」表。
    """
    if target_date is None:
        target_date = datetime.now()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y%m%d")

    if isinstance(target_date, datetime):
        date_str = target_date.strftime("%Y%m%d")
    else:
        date_str = str(target_date)

    cache_file = os.path.join(DATA_DIR, f"twse_margin_{date_str}.csv")
    if os.path.exists(cache_file) and not force_update:
        try:
            cached = pd.read_csv(cache_file)
            if not cached.empty:
                return cached
        except Exception:
            pass

    try:
        r = requests.get(_twse_margin_url(date_str), headers=headers, timeout=10)
        r.encoding = "utf-8"
        payload = r.json()
        if str(payload.get("stat", "")).strip() not in {"OK", "ok"}:
            return pd.DataFrame()

        tables = payload.get("tables", [])
        if not tables:
            return pd.DataFrame()

        table = tables[0]
        fields = table.get("fields", [])
        rows = table.get("data", [])
        if not fields or not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=fields)
        rename_map = {
            "項目": "項目",
            "買進": "買進",
            "賣出": "賣出",
            "現金(券)償還": "現金(券)償還",
            "前日餘額": "前日餘額",
            "今日餘額": "今日餘額",
        }
        df = df.rename(columns=rename_map)

        for col in ["買進", "賣出", "現金(券)償還", "前日餘額", "今日餘額"]:
            if col in df.columns:
                df[col] = df[col].apply(_extract_float)

        df["資料日期"] = date_str
        df.to_csv(cache_file, index=False, encoding="utf-8-sig")
        return df
    except Exception:
        return pd.DataFrame()


def preload_twse_margin_data(days_back=60, force_update=False):
    """先批次下載最近幾個交易日的 TWSE 融資融券 CSV，供後續離線判斷使用。"""
    today = datetime.now()
    loaded = []

    for offset in range(0, days_back + 1):
        candidate = today - timedelta(days=offset)
        date_str = candidate.strftime("%Y%m%d")
        cache_file = os.path.join(DATA_DIR, f"twse_margin_{date_str}.csv")

        if os.path.exists(cache_file) and not force_update:
            loaded.append(cache_file)
            continue

        df = download_twse_margin_data(candidate, force_update=force_update)
        if df is not None and not df.empty:
            loaded.append(cache_file)

    return loaded


def check_twse_foreign_consecutive_buy(stock_id, signal_date, consecutive_days=3, force_update=False):
    """檢查指定股票在 signal_date 往前的最近連續交易日是否連續外資買超。"""
    try:
        signal_dt = datetime.strptime(str(signal_date), "%Y%m%d")
    except Exception:
        return False

    stock_str = str(stock_id).strip()
    hit_days = 0

    for offset in range(0, 10):
        candidate = signal_dt - timedelta(days=offset)
        daily_df = download_twse_foreign_buy_sell(candidate, force_update=force_update)
        if daily_df is None or daily_df.empty:
            continue

        if "股票代號" not in daily_df.columns:
            continue

        row = daily_df[daily_df["股票代號"].astype(str).str.strip() == stock_str]
        if row.empty:
            continue

        foreign_net = _extract_float(row.iloc[0].get("外資買賣超"))
        if pd.isna(foreign_net) or foreign_net <= 0:
            return False

        hit_days += 1
        if hit_days >= consecutive_days:
            return True

    return False


def check_twse_margin_balance_low(stock_id, signal_date, lookback_days=60, percentile=20, force_update=False):
    """檢查市場融資餘額是否落在近 lookback_days 的低檔區間。

    這個公開端點目前是市場總量，不是個股，所以 stock_id 只保留相容介面。
    """
    try:
        signal_dt = datetime.strptime(str(signal_date), "%Y%m%d")
    except Exception:
        return False

    balances = []
    current_balance = np.nan

    for offset in range(0, lookback_days + 1):
        candidate = signal_dt - timedelta(days=offset)
        daily_df = download_twse_margin_data(candidate, force_update=force_update)
        if daily_df is None or daily_df.empty:
            continue

        row = daily_df[daily_df["項目"].astype(str).str.contains("融資", na=False)]
        if row.empty:
            continue

        balance = np.nan
        for key in ["今日餘額", "前日餘額", "買進", "賣出"]:
            if key in row.columns:
                balance = _extract_float(row.iloc[0].get(key))
                if not pd.isna(balance):
                    break

        if pd.isna(balance):
            continue

        balances.append(balance)
        if pd.isna(current_balance):
            current_balance = balance

    if len(balances) < 10 or pd.isna(current_balance):
        return False

    low_threshold = np.nanpercentile(balances, percentile)
    return current_balance <= low_threshold


def export_today_twse_foreign_buy_sell_csv(force_update=False):
    """下載今天或最近可用交易日的 TWSE 外資買賣超資料並匯出 CSV。"""
    df = get_today_foreign_buy_sell(force_update=force_update)
    if df is None or df.empty:
        return pd.DataFrame(), None

    date_value = str(df.iloc[0].get("資料日期", datetime.now().strftime("%Y%m%d")))
    export_file = os.path.join(DATA_DIR, f"twse_foreign_buy_sell_{date_value}.csv")
    df.to_csv(export_file, index=False, encoding="utf-8-sig")
    return df, export_file

# ==========================================
# 爬蟲引擎 (含 12 小時智慧快取)
# ==========================================
def get_stock_ids(url=list_url, force_update=False):
    """取得全市場股票代號清單"""
    list_file = os.path.join(DATA_DIR, "stock_list.txt")
    
    if os.path.exists(list_file):
        file_age = time.time() - os.path.getmtime(list_file)
        if file_age > 43200: force_update = True
            
    if not force_update and os.path.exists(list_file):
        with open(list_file, 'r', encoding='utf-8') as f: return [line.strip() for line in f.readlines()]
    try:
        r = requests.get(url, headers=headers)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'lxml')
        stock_ids = [match.group(1) for a in soup.find_all('a', href=True) if (match := re.search(r'STOCK=(\d{4,6})', a['href'])) and len(match.group(1)) == 4]
        stock_ids = sorted(list(set(stock_ids)))
        if stock_ids:
            with open(list_file, 'w', encoding='utf-8') as f:
                for sid in stock_ids: f.write(f"{sid}\n")
        return stock_ids
    except: return []

def get_individual_stock_data(stock_id, force_update=False):
    """取得單一股票籌碼資料 (具備防誤殺與黑名單假釋機制)"""
    file_path = os.path.join(DATA_DIR, f"{stock_id}.csv")
    
    # 檢查檔案是否超過 12 小時 (正常資料的快取)
    if os.path.exists(file_path):
        file_age = time.time() - os.path.getmtime(file_path)
        if file_age > 43200: 
            force_update = True 
            
    # 讀取本地快取
    if not force_update and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # 🌟 防誤殺機制：檢查黑名單是否超過「假釋期」(例如 30 天 = 2592000 秒)
            if '無資料' in df.columns:
                file_age = time.time() - os.path.getmtime(file_path)
                if file_age < 2592000:
                    return None  # 還在黑名單效期內，直接跳過
                else:
                    force_update = True  # 超過 30 天，強制重新爬取看看有沒有新資料
                    
            elif '>1000張百分比' in df.columns or '>400張百分比' in df.columns:
                if '股票代號' in df.columns:
                    df['股票代號'] = df['股票代號'].astype(str).str.zfill(4)
                if '資料日期' in df.columns:
                    df['資料日期'] = df['資料日期'].astype(str)
                return df
        except: 
            force_update = True

    url = f"https://norway.twsthr.info/StockHolders.aspx?stock={stock_id}"
    try:
        r = requests.get(url, headers=headers, timeout=5)
        
        # 🌟 防誤殺機制：如果網頁伺服器掛掉(不是200 OK)，直接回傳None，絕對不要存成黑名單！
        if r.status_code != 200:
            return None
            
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'lxml')
        data_rows = []
        for tr in soup.find_all('tr'):
            numbers = re.findall(r'\d+\.?\d*', tr.get_text(separator=' ').replace(',', ''))
            if len(numbers) >= 13 and re.match(r'^202\d{5}$', numbers[0]):
                try:
                    row = {
                        '資料日期': numbers[0], '總張數' : numbers[1], '總股東人數': numbers[2], 
                        '平均張數/人': numbers[3], '>1000張百分比': numbers[4], '>400張百分比': numbers[5], '收盤價': numbers[12]
                    }
                    if 0 < float(row['收盤價']) < 100000: data_rows.append(row)
                except: continue
        
        # 如果網頁正常，但真的解析不到資料，才更新黑名單檔案
        if not data_rows: 
            pd.DataFrame(columns=['無資料']).to_csv(file_path, index=False, encoding='utf-8-sig')
            return None
            
        df = pd.DataFrame(data_rows).drop_duplicates(subset=['資料日期'])
        for col in ['總張數','總股東人數', '平均張數/人', '>1000張百分比', '>400張百分比', '收盤價']: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        df.insert(0, '股票代號', str(stock_id).zfill(4))
        df = df.sort_values('資料日期', ascending=True).reset_index(drop=True)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return df
    except: 
        # 發生 Timeout 或是連線錯誤，直接回傳，不記錄黑名單
        return None


def get_tej_institutional_data(stock_id, start_date, end_date, force_update=False):
    """從 TEJ 取得三大法人資料，回傳標準化 DataFrame。"""
    client = _get_tej_client()
    if client is None:
        return pd.DataFrame()

    try:
        return client.get_institutional_trading(stock_id, start_date, end_date, force_update=force_update)
    except Exception:
        return pd.DataFrame()


def get_tej_margin_short_data(stock_id, start_date, end_date, force_update=False):
    """從 TEJ 取得融資融券資料，回傳標準化 DataFrame。"""
    client = _get_tej_client()
    if client is None:
        return pd.DataFrame()

    try:
        return client.get_margin_short(stock_id, start_date, end_date, force_update=force_update)
    except Exception:
        return pd.DataFrame()


def enrich_with_tej_features(stock_df, stock_id, force_update=False):
    """把 TEJ 的法人與融資融券資料合併到現有籌碼資料。"""
    if stock_df is None or stock_df.empty:
        return stock_df

    client = _get_tej_client()
    if client is None:
        return stock_df

    try:
        df = stock_df.copy()
        df["資料日期"] = df["資料日期"].astype(str)
        start_date = df["資料日期"].min()
        end_date = df["資料日期"].max()

        inst_df = get_tej_institutional_data(stock_id, start_date, end_date, force_update=force_update)
        margin_df = get_tej_margin_short_data(stock_id, start_date, end_date, force_update=force_update)

        if inst_df is not None and not inst_df.empty:
            inst_df = inst_df.copy()
            inst_df["資料日期"] = inst_df["資料日期"].astype(str)
            inst_df = inst_df.drop_duplicates(subset=["股票代號", "資料日期"])
            df = df.merge(inst_df, on=["股票代號", "資料日期"], how="left")

        if margin_df is not None and not margin_df.empty:
            margin_df = margin_df.copy()
            margin_df["資料日期"] = margin_df["資料日期"].astype(str)
            margin_df = margin_df.drop_duplicates(subset=["股票代號", "資料日期"])
            df = df.merge(margin_df, on=["股票代號", "資料日期"], how="left")

        return df.sort_values("資料日期", ascending=True).reset_index(drop=True)
    except Exception:
        return stock_df


def _run_twse_cache_preload():
    """Direct execution entrypoint: batch preloads recent TWSE CSV caches."""
    print("開始預載 TWSE 公開資料快取...")
    foreign_files = preload_twse_foreign_buy_sell(days_back=7, force_update=False)
    margin_files = preload_twse_margin_data(days_back=60, force_update=False)

    print(f"外資買賣超快取：{len(foreign_files)} 份")
    print(f"融資融券快取：{len(margin_files)} 份")

    if foreign_files:
        print("最近外資快取：")
        for path in foreign_files[-5:]:
            print(f" - {path}")

    if margin_files:
        print("最近融資快取：")
        for path in margin_files[-5:]:
            print(f" - {path}")


if __name__ == "__main__":
    _run_twse_cache_preload()