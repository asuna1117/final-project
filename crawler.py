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


def _quiet_yf_download(*args, **kwargs):
    """Silence yfinance console noise and return dataframe as-is."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return yf.download(*args, **kwargs)

# ==========================================
# 股價與爬蟲輔助功能
# ==========================================
def get_next_monday_open_price(stock_id, signal_date):
    """計算並抓取下週一的開盤價"""
    try:
        signal_dt = datetime.strptime(str(signal_date), "%Y%m%d")
        days_until_monday = (7 - signal_dt.weekday()) % 7
        if days_until_monday == 0: days_until_monday = 7
        next_monday = signal_dt + timedelta(days=days_until_monday)
        end_dt = next_monday + timedelta(days=14)

        for suffix in [".TW", ".TWO"]:
            ticker = f"{stock_id}{suffix}"
            data = _quiet_yf_download(ticker, start=next_monday.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False, auto_adjust=False, threads=False)
            if data is None or data.empty: continue
            
            open_series = data["Open"]
            if isinstance(open_series, pd.DataFrame): open_series = open_series.iloc[:, 0]
            open_series = open_series.dropna()

            if not open_series.empty: return float(open_series.iloc[0])
        return np.nan
    except Exception:
        return np.nan
    
def check_condition_e_with_yfinance(stock_id, signal_date, monday_open_price):
    """條件 E: 下週二收盤 > 週一開盤，且週三、週四收盤連續走高。"""
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
    expected_dates = [next_tuesday.date(), next_wednesday.date(), next_thursday.date()]

    for suffix in [".TW", ".TWO"]:
        ticker = f"{stock_id}{suffix}"
        data = _quiet_yf_download(
            ticker,
            start=next_tuesday.strftime("%Y-%m-%d"),
            end=(next_thursday + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if data is None or data.empty:
            continue

        close_series = data["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        close_series = close_series.dropna()

        daily_close = {pd.Timestamp(idx).date(): float(val) for idx, val in close_series.items()}
        if not all(day in daily_close for day in expected_dates):
            continue

        tue_close = daily_close[expected_dates[0]]
        wed_close = daily_close[expected_dates[1]]
        thu_close = daily_close[expected_dates[2]]

        return tue_close > monday_open_price and wed_close > tue_close and thu_close > wed_close

    return False

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
    """取得單一股票籌碼資料"""
    file_path = os.path.join(DATA_DIR, f"{stock_id}.csv")
    
    if os.path.exists(file_path):
        file_age = time.time() - os.path.getmtime(file_path)
        if file_age > 43200: 
            force_update = True 
            
    if not force_update and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if '>1000張百分比' in df.columns:
                df['股票代號'] = df['股票代號'].astype(str).str.zfill(4)
                df['資料日期'] = df['資料日期'].astype(str)
                return df
            force_update = True
        except: pass

    url = f"https://norway.twsthr.info/StockHolders.aspx?stock={stock_id}"
    try:
        r = requests.get(url, headers=headers)
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
        
        if not data_rows: return None
        df = pd.DataFrame(data_rows).drop_duplicates(subset=['資料日期'])
        for col in ['總張數','總股東人數', '平均張數/人', '>1000張百分比', '>400張百分比', '收盤價']: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        df.insert(0, '股票代號', str(stock_id).zfill(4))
        df = df.sort_values('資料日期', ascending=True).reset_index(drop=True)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return df
    except: return None