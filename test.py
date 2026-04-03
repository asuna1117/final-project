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

# --- 設定 ---
list_url = "https://norway.twsthr.info/StockHoldersDividendTop.aspx?CID=0&Show=1"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ==========================================
# 輔助函式 (標籤、星星、箭頭)
# ==========================================
def get_concentration_stars(pct):
    """籌碼集中度星等"""
    if pct >= 70: return '⭐⭐⭐⭐⭐'
    elif pct >= 60: return '⭐⭐⭐⭐'
    elif pct >= 50: return '⭐⭐⭐'
    elif pct >= 40: return '⭐⭐'
    elif pct >= 30: return '⭐'
    else: return '☆'

def get_change_arrow(val):
    if val > 0: return f'△{val:.2f}'
    elif val < 0: return f'▽{abs(val):.2f}'
    else: return f'－{val:.2f}'

def get_next_monday_open_price(stock_id, signal_date):
    """用 yfinance 抓取訊號日後的下週一開盤價。"""
    try:
        signal_dt = datetime.strptime(str(signal_date), "%Y%m%d")
        days_until_monday = (7 - signal_dt.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        next_monday = signal_dt + timedelta(days=days_until_monday)
        end_dt = next_monday + timedelta(days=3)

        # 台股先嘗試 .TW，若抓不到再嘗試 .TWO
        for suffix in [".TW", ".TWO"]:
            ticker = f"{stock_id}{suffix}"
            data = yf.download(
                ticker,
                start=next_monday.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if data is None or data.empty:
                continue

            open_series = data["Open"]
            if isinstance(open_series, pd.DataFrame):
                open_series = open_series.iloc[:, 0]
            open_series = open_series.dropna()

            if not open_series.empty:
                return float(open_series.iloc[0])

        return np.nan
    except Exception:
        return np.nan

# ==========================================
# 核心功能：硬規則回測 (大戶連4買 > 1%)
# ==========================================
def backtest_squeeze_strategy(df_group, corr_threshold=0.5):
    stock_id = df_group['股票代號'].iloc[0]
    # 確保資料依照日期從舊到新排列
    df = df_group.sort_values('資料日期', ascending=True).reset_index(drop=True)

    df['平均力道'] = df['總張數'] / df['總股東人數']
    trades = []
    
    if len(df) < 13: return []
    
    for i in range(11, len(df)-1):
        
        # 條件 A: 連續 4 週每週漲幅皆 > 0.1%
        weekly_growth_a = []
        for j in range(4):
            prev_val = df.at[i-j-1, '>400張百分比']
            curr_val = df.at[i-j, '>400張百分比']
            if prev_val <= 0:
                weekly_growth_a.append(-np.inf)
            else:
                weekly_growth_a.append(((curr_val - prev_val) / prev_val) * 100)
        is_continuous_buy = all(g > 0.1 for g in weekly_growth_a)
        
        # 條件 B: 平均張數/人連續 4 週每週漲幅皆 > 0.1%
        weekly_growth_b = []
        for j in range(4):
            prev_val = df.at[i-j-1, '平均張數/人']
            curr_val = df.at[i-j, '平均張數/人']
            if prev_val <= 0:
                weekly_growth_b.append(-np.inf)
            else:
                weekly_growth_b.append(((curr_val - prev_val) / prev_val) * 100)
        is_avg_per_person_continuous_up = all(g > 0.1 for g in weekly_growth_b)
        
        # 條件 C: 總股東人數 4 週總下跌 > 0.5%
        pop_decline_pct = ((df.at[i-4, '總股東人數'] - df.at[i-1, '總股東人數']) / df.at[i-4, '總股東人數']) * 100

        if is_continuous_buy and is_avg_per_person_continuous_up and pop_decline_pct > 0.5:
            
            # 計算 12 週特徵與下週收盤價的相關係數
            x_400 = df.loc[i-11:i, '>400張百分比'].reset_index(drop=True)
            x_avg_per_person = df.loc[i-11:i, '平均張數/人'].reset_index(drop=True)
            x_shareholders = df.loc[i-11:i, '總股東人數'].reset_index(drop=True)
            y_next_close = df.loc[i-10:i+1, '收盤價'].reset_index(drop=True)

            corr_val = x_400.corr(y_next_close)
            avg_per_person_corr_val = x_avg_per_person.corr(y_next_close)
            retail_corr_val = x_shareholders.corr(y_next_close)

            corr_val = 0.0 if pd.isna(corr_val) else corr_val
            avg_per_person_corr_val = 0.0 if pd.isna(avg_per_person_corr_val) else avg_per_person_corr_val
            retail_corr_val = 0.0 if pd.isna(retail_corr_val) else retail_corr_val

            # 條件 D: 相關係數門檻
            if not (corr_val > 0.6 or avg_per_person_corr_val > 0.6 or retail_corr_val < -0.6):
                continue

            # 觸發買入：改用 yfinance 取得下週一開盤價
            buy_price = get_next_monday_open_price(stock_id, df.at[i, '資料日期'])
            sell_price = df.at[i+1, '收盤價'] # 假設週五賣(以下週收盤價計)
            
            if buy_price > 0:
                profit_pct = ((sell_price - buy_price) / buy_price) * 100
                trades.append({
                    '代號': stock_id,
                    '進場日期': df.at[i, '資料日期'],
                    #'大戶增%': round(total_growth, 2),
                    #'人數減%': round(pop_decline * 100, 2),
                    #'平均張增%': round(avg_growth * 100, 2),
                    #'大戶相關係數(5週)': round(float(corr_val), 3),
                    #'散戶相關係數(5週)': round(float(retail_corr_val), 3),
                    #'平均張數/人相關係數(5週)': round(float(avg_per_person_corr_val), 3),
                    #'當週大戶增減': round(float(weekly_change), 3),
                    '進場價': buy_price,
                    '出場價': sell_price,
                    '週報酬%': profit_pct
                })

    return trades

# ==========================================
# 爬蟲與輔助函式
# ==========================================
def get_stock_ids(url):
    print(f"正在從 {url} 抓取股票清單...")
    try:
        r = requests.get(url, headers=headers)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'lxml')
        stock_ids = []
        for a in soup.find_all('a', href=True):
            match = re.search(r'STOCK=(\d{4,6})', a['href'])
            if match and len(match.group(1)) == 4:
                stock_ids.append(match.group(1))
        stock_ids = sorted(list(set(stock_ids)))
        print(f"   找到 {len(stock_ids)} 檔股票")
        return stock_ids
    except Exception as e:
        print(f"❌ 抓取清單錯誤: {e}")
        return []

def get_individual_stock_data(stock_id):
    url = f"https://norway.twsthr.info/StockHolders.aspx?stock={stock_id}"
    try:
        r = requests.get(url, headers=headers)
        r.encoding = 'utf-8'
        soup = BeautifulSoup(r.text, 'lxml')
        
        data_rows = []
        
        # 🌟 終極解析法：無視網頁標籤，直接抽取純文字裡的數字
        for tr in soup.find_all('tr'):
            # 1. 拔除所有 HTML 標籤，把整列轉成純文字 (用空白隔開)
            row_text = tr.get_text(separator=' ')
            
            # 2. 清除千分位逗號，避免數字被切開
            row_text_clean = row_text.replace(',', '')
            
            # 3. 從純文字中按順序抓出所有數字 (包含小數點)
            numbers = re.findall(r'\d+\.?\d*', row_text_clean)
            
            # 4. 籌碼表至少 13 個數據，且第一筆必須是 8 位數的日期
            if len(numbers) >= 13 and re.match(r'^202\d{5}$', numbers[0]):
                try:
                    row = {
                        '資料日期': numbers[0],
                        '總張數' : numbers[1],
                        '總股東人數': numbers[2],
                        '平均張數/人': numbers[3],
                        '>400張百分比': numbers[5],
                        '>1000張百分比': numbers[11],
                        '收盤價': numbers[12]
                    }
                    
                    test_val = float(row['收盤價'])
                    if 0 < test_val < 100000:
                        data_rows.append(row)
                except: 
                    continue
        
        if not data_rows: return None
        
        df = pd.DataFrame(data_rows).drop_duplicates(subset=['資料日期'])
        for col in ['總張數','總股東人數', '平均張數/人', '>400張百分比', '>1000張百分比', '收盤價']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        if df.empty: return None
        
        df.insert(0, '股票代號', stock_id)
        df = df.sort_values('資料日期', ascending=True).reset_index(drop=True)
        return df

    except Exception as e:
        return None

# ==========================================
# 總司令函式
# ==========================================
def run_all_analysis(target_list):
    all_dfs = []
    all_trades = []
    total = len(target_list)

    for i, sid in enumerate(target_list):
        print(f"[{i + 1}/{total}] {sid}...", end=" ", flush=True)
        df = get_individual_stock_data(sid)

        if df is None or df.empty:
            print("Skip")
            continue

        all_dfs.append(df)

        trades = backtest_squeeze_strategy(df, corr_threshold=0.5)
        all_trades.extend(trades)

        print(f"OK ({len(df)}筆, 訊號{len(trades)}筆)")

    if all_trades:
        trades_df = pd.DataFrame(all_trades).sort_values(['進場日期', '代號'], ascending=[False, True])
        return trades_df
    else:
        return pd.DataFrame()

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    stock_list = get_stock_ids(list_url)
    total_available = len(stock_list)

    if total_available == 0:
        print("❌ 沒抓到股票清單，程式結束。")
        raise SystemExit

    print(f"\n✅ 成功取得 {total_available} 檔股票清單。")
    print("--------------------------------")
    print("1. 前 10 個 (快速測試)")
    print("2. 前 100 個 (建議)")
    print(f"3. 全部 ({total_available} 個)")
    print("4. 自訂範圍")
    print("--------------------------------")

    choice = input("👉 請輸入選項 (1/2/3/4): ").strip()
    start_index, end_index = 0, 10

    if choice == '2':
        end_index = min(100, total_available)
    elif choice == '3':
        end_index = total_available
    elif choice == '4':
        try:
            start_index = int(input("👉 從第幾檔開始? (預設 0): ").strip() or 0)
            count = int(input("👉 要抓幾檔? (預設 10): ").strip() or 10)
            end_index = min(start_index + count, total_available)
        except:
            start_index, end_index = 0, min(10, total_available)

    target_list = stock_list[start_index:end_index]
    print(f"\n準備抓取 {len(target_list)} 檔股票的籌碼資料...\n")

    trades_df = run_all_analysis(target_list)

    if not trades_df.empty:
        print("\n" + "=" * 90)
        print("📈 硬規則 + 相關係數：篩選通過清單（回測）")
        print("=" * 90)
        
        display_df = trades_df.fillna({'出場價': '等待開獎', '週報酬%': '等待開獎'})
        print(tabulate(display_df, headers='keys', tablefmt='simple', showindex=False))

        completed_trades = trades_df.dropna(subset=['週報酬%'])
        if not completed_trades.empty:
            win_rate = (completed_trades['週報酬%'] > 0).mean() * 100
            avg_return = completed_trades['週報酬%'].mean()
            print(f"\n勝率: {win_rate:.2f}% | 平均週報酬: {avg_return:.2f}% | 訊號總數: {len(completed_trades)} (已結算)")
        else:
            print("\n⚠️ 目前只有最新訊號，尚無歷史結算資料可計算勝率。")

    else:
        print("\n⚠️ 沒有符合硬規則 + 相關係數條件的回測訊號。")