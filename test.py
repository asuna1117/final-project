import pandas as pd
import requests
import time
import re
import numpy as np
import yfinance as yf
from tabulate import tabulate
from bs4 import BeautifulSoup
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings('ignore')

# --- 設定 ---
list_url = "https://norway.twsthr.info/StockHoldersDividendTop.aspx?CID=0&Show=1"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stock_data_cache")      
INST_DIR = os.path.join(BASE_DIR, "inst_data_cache")       

for d in [DATA_DIR, INST_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def get_change_arrow(val):
    if val > 0: return f'△{val:.2f}'
    elif val < 0: return f'▽{abs(val):.2f}'
    else: return f'－{val:.2f}'

def get_next_monday_open_price(stock_id, signal_date):
    try:
        signal_dt = datetime.strptime(str(signal_date), "%Y%m%d")
        days_until_monday = (7 - signal_dt.weekday()) % 7
        if days_until_monday == 0: days_until_monday = 7
        next_monday = signal_dt + timedelta(days=days_until_monday)
        end_dt = next_monday + timedelta(days=14)

        for suffix in [".TW", ".TWO"]:
            ticker = f"{stock_id}{suffix}"
            data = yf.download(ticker, start=next_monday.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False, auto_adjust=False, threads=False)
            if data is None or data.empty: continue
            
            open_series = data["Open"]
            if isinstance(open_series, pd.DataFrame): open_series = open_series.iloc[:, 0]
            open_series = open_series.dropna()

            if not open_series.empty: return float(open_series.iloc[0])
        return np.nan
    except Exception:
        return np.nan

def get_institutional_data(stock_id, force_update=False):
    file_path = os.path.join(INST_DIR, f"{stock_id}_inst.csv")
    if not force_update and os.path.exists(file_path):
        try: return pd.read_csv(file_path)
        except Exception: pass

    print(f"   📊 抓取 {stock_id} 法人籌碼...", end=" ", flush=True)
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": str(stock_id), "start_date": "2023-01-01"}
    
    try:
        r = requests.get(url, params=parameter)
        data = r.json()
        if data.get("status") == 200 and data.get("data"):
            df = pd.DataFrame(data["data"])
            daily_df = df.groupby('date')[['buy', 'sell']].sum().reset_index()
            daily_df['net_buy_lots'] = (daily_df['buy'] - daily_df['sell']) / 1000
            daily_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            time.sleep(0.5)
            return daily_df
        else:
            print("(無法人資料)", end=" ")
            return pd.DataFrame()
    except Exception:
        print(f"(API失敗)", end=" ")
        return pd.DataFrame()

# ==========================================
# 核心功能：回測邏輯 (導入所有客製化參數)
# ==========================================
def backtest_squeeze_strategy(df_group, inst_df=None, enable_layer_3=False,
                              large_holder_tier='>400張百分比', continuous_weeks=4, 
                              min_growth=0.1, pop_decline_threshold=0.5,
                              corr_window=11, large_corr_thresh=0.6, 
                              retail_corr_thresh=-0.6, avg_corr_thresh=0.6):
    stock_id = df_group['股票代號'].iloc[0]
    df = df_group.sort_values('資料日期', ascending=True).reset_index(drop=True)
    trades = []
    
    min_length = max(corr_window + 2, continuous_weeks + 1)
    if len(df) < min_length: return []
    
    for i in range(corr_window, len(df)-1):
        if i - continuous_weeks < 0: continue
        
        # 1. 檢驗：大戶連買、平均張數連買、散戶減少
        weekly_growth_a = [((df.at[i-j, large_holder_tier] - df.at[i-j-1, large_holder_tier]) / df.at[i-j-1, large_holder_tier]) * 100 if df.at[i-j-1, large_holder_tier] > 0 else -np.inf for j in range(continuous_weeks)]
        is_continuous_buy = all(g > min_growth for g in weekly_growth_a)
        
        weekly_growth_b = [((df.at[i-j, '平均張數/人'] - df.at[i-j-1, '平均張數/人']) / df.at[i-j-1, '平均張數/人']) * 100 if df.at[i-j-1, '平均張數/人'] > 0 else -np.inf for j in range(continuous_weeks)]
        is_avg_per_person_continuous_up = all(g > min_growth for g in weekly_growth_b)
        
        pop_decline_pct = ((df.at[i-continuous_weeks, '總股東人數'] - df.at[i-1, '總股東人數']) / df.at[i-continuous_weeks, '總股東人數']) * 100

        if is_continuous_buy and is_avg_per_person_continuous_up and pop_decline_pct > pop_decline_threshold:
            
            # 2. 相關係數檢驗
            x_large = df.loc[i-corr_window:i, large_holder_tier].reset_index(drop=True)
            x_avg_per_person = df.loc[i-corr_window:i, '平均張數/人'].reset_index(drop=True)
            x_shareholders = df.loc[i-corr_window:i, '總股東人數'].reset_index(drop=True)
            y_next_close = df.loc[i-corr_window+1:i+1, '收盤價'].reset_index(drop=True)

            corr_val = x_large.corr(y_next_close)
            avg_corr_val = x_avg_per_person.corr(y_next_close)
            retail_corr_val = x_shareholders.corr(y_next_close)

            corr_val = 0.0 if pd.isna(corr_val) else corr_val
            avg_corr_val = 0.0 if pd.isna(avg_corr_val) else avg_corr_val
            retail_corr_val = 0.0 if pd.isna(retail_corr_val) else retail_corr_val

            if not (corr_val >= large_corr_thresh or avg_corr_val >= avg_corr_thresh or retail_corr_val <= retail_corr_thresh):
                continue

            # 3. 第三層條件切換邏輯
            condition_c_passed = True
            if enable_layer_3:
                condition_c_passed = False
                total_lots = df.at[i, '總張數']
                if inst_df is not None and not inst_df.empty:
                    signal_dt = datetime.strptime(str(df.at[i, '資料日期']), "%Y%m%d")
                    start_dt = signal_dt - timedelta(days=6)
                    mask = (inst_df['date'] >= start_dt.strftime("%Y-%m-%d")) & (inst_df['date'] <= signal_dt.strftime("%Y-%m-%d"))
                    week_inst_df = inst_df.loc[mask]
                    if not week_inst_df.empty:
                        cumulative_buy = week_inst_df['net_buy_lots'].sum()
                        if cumulative_buy > (total_lots * 0.005): condition_c_passed = True

            if not condition_c_passed: continue

            # 4. 記錄交易
            buy_price = get_next_monday_open_price(stock_id, df.at[i, '資料日期'])
            sell_price = df.at[i+1, '收盤價'] 
            
            if buy_price > 0:
                profit_pct = ((sell_price - buy_price) / buy_price) * 100
                trades.append({
                    '代號': stock_id,
                    '進場日期': df.at[i, '資料日期'],
                    f'大戶相關({corr_window}週)': round(float(corr_val), 3),
                    f'散戶相關({corr_window}週)': round(float(retail_corr_val), 3),
                    f'均張相關({corr_window}週)': round(float(avg_corr_val), 3),
                    '進場價': buy_price,
                    '出場價': sell_price,
                    '週報酬%': profit_pct
                })

    return trades

def get_stock_ids(url, force_update=False):
    list_file = os.path.join(DATA_DIR, "stock_list.txt")
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
    file_path = os.path.join(DATA_DIR, f"{stock_id}.csv")
    if not force_update and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # 防呆檢查：如果快取檔案沒有新版的 1000張 欄位，強制重新下載
            if '>1000張百分比' in df.columns:
                df['股票代號'] = df['股票代號'].astype(str).str.zfill(4)
                df['資料日期'] = df['資料日期'].astype(str)
                return df
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
                        '平均張數/人': numbers[3], '>400張百分比': numbers[5], '>600張百分比': numbers[7], 
                        '>800張百分比': numbers[9], '>1000張百分比': numbers[11], '收盤價': numbers[12]
                    }
                    if 0 < float(row['收盤價']) < 100000: data_rows.append(row)
                except: continue
        
        if not data_rows: return None
        df = pd.DataFrame(data_rows).drop_duplicates(subset=['資料日期'])
        for col in ['總張數','總股東人數', '平均張數/人', '>400張百分比', '>600張百分比', '>800張百分比', '>1000張百分比', '收盤價']: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        df.insert(0, '股票代號', str(stock_id).zfill(4))
        df = df.sort_values('資料日期', ascending=True).reset_index(drop=True)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return df
    except: return None