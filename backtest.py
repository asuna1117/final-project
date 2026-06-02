import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
from tabulate import tabulate
import crawler

def _get_large_holder_series(df):
    """依該列收盤價動態決定使用 >400 或 >1000 張百分比。"""
    if '>400張百分比' not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    if '>1000張百分比' not in df.columns:
        return pd.to_numeric(df['>400張百分比'], errors='coerce')

    close_price = pd.to_numeric(df['收盤價'], errors='coerce')
    large_400 = pd.to_numeric(df['>400張百分比'], errors='coerce')
    large_1000 = pd.to_numeric(df['>1000張百分比'], errors='coerce')

    return pd.Series(np.where(close_price > 100, large_400, large_1000), index=df.index)

# ==========================================
# ⚡ 極速核心：向量化預先計算訊號 (Vectorized Engine)
# ==========================================
def _get_signal_indices(df, large_holder_series, continuous_weeks, min_growth, last_week_threshold, pop_decline_threshold):
    """使用 Pandas 向量化運算，一次性找出所有符合條件 A、B、C 的資料列索引 (速度提升 1000 倍)"""
    if len(df) < continuous_weeks + 1:
        return []

    # 計算每週成長率 (pct_change)
    growth_a = large_holder_series.pct_change() * 100
    growth_a = growth_a.replace([np.inf, -np.inf], -999).fillna(-999)
    
    growth_b = df['平均張數/人'].pct_change() * 100
    growth_b = growth_b.replace([np.inf, -np.inf], -999).fillna(-999)

    # 條件 A: 連續 N 週漲幅 > 0，且最後一週 > last_week_threshold
    cond_a_continuous = growth_a.rolling(window=continuous_weeks).min() > 0
    cond_a_last = growth_a > last_week_threshold
    cond_a = cond_a_continuous & cond_a_last

    # 條件 B: 平均張數連續 N 週漲幅 > min_growth
    cond_b = growth_b.rolling(window=continuous_weeks).min() > min_growth

    # 條件 C: 總股東人數 N 週總下跌 > pop_decline_threshold
    # 使用 shift 取得 N 週前的數值來計算總跌幅
    pop_n_weeks_ago = df['總股東人數'].shift(continuous_weeks)
    pop_decline_pct = ((pop_n_weeks_ago - df['總股東人數']) / pop_n_weeks_ago) * 100
    cond_c = pop_decline_pct > pop_decline_threshold

    # 交集：同時符合 A, B, C 的列
    final_signal = cond_a & cond_b & cond_c
    
    # 回傳這些列的 Index 列表
    return df.index[final_signal].tolist()

# ==========================================
# 核心功能：回測邏輯 (極速版)
# ==========================================
def backtest_squeeze_strategy(df_group, continuous_weeks=3, min_growth=0.0479, last_week_threshold=0.179, pop_decline_threshold=0.198,
                              corr_window=156, large_corr_thresh=0.6, 
                              retail_corr_thresh=-0.6, avg_corr_thresh=0.6): 
    
    stock_id = df_group['股票代號'].iloc[0]
    df = df_group.sort_values('資料日期', ascending=True).reset_index(drop=True)
    trades = []
    large_holder_series = _get_large_holder_series(df)
    
    # ⚡ 瞬間找出所有符合 ABC 條件的列，略過無效的 for 迴圈
    valid_indices = _get_signal_indices(df, large_holder_series, continuous_weeks, min_growth, last_week_threshold, pop_decline_threshold)
    
    for i in valid_indices:
        if i < 1: continue

        # 計算相關係數 (條件 D)
        x_large = large_holder_series.iloc[0:i].reset_index(drop=True)
        x_avg_per_person = df.loc[0:i-1, '平均張數/人'].reset_index(drop=True)
        x_shareholders = df.loc[0:i-1, '總股東人數'].reset_index(drop=True)
        y_next_close = df.loc[1:i, '收盤價'].reset_index(drop=True)

        corr_val = x_large.corr(y_next_close)
        avg_corr_val = x_avg_per_person.corr(y_next_close)
        retail_corr_val = x_shareholders.corr(y_next_close)

        corr_val = 0.0 if pd.isna(corr_val) else corr_val
        avg_corr_val = 0.0 if pd.isna(avg_corr_val) else avg_corr_val
        retail_corr_val = 0.0 if pd.isna(retail_corr_val) else retail_corr_val

        if not (corr_val >= large_corr_thresh or avg_corr_val >= avg_corr_thresh or retail_corr_val <= retail_corr_thresh):
            continue

        # 抓取股價 (條件 E)
        buy_price = crawler.get_next_monday_open_price(stock_id, df.at[i, '資料日期'])
        sell_price = crawler.get_next_friday_close_price(stock_id, df.at[i, '資料日期'])

        if buy_price <= 0 or pd.isna(sell_price) or not crawler.check_condition_e_with_yfinance(stock_id, df.at[i, '資料日期'], buy_price):
            continue
        
        if buy_price > 0 and not pd.isna(sell_price):
            profit_pct = ((sell_price - buy_price) / buy_price) * 100
            trades.append({
                '代號': stock_id,
                '進場日期(籌碼公告)': df.at[i, '資料日期'],
                '大戶相關係數': round(float(corr_val), 3),
                '散戶相關係數': round(float(retail_corr_val), 3),
                '平均張數相關': round(float(avg_corr_val), 3),
                '週一開盤進場價': round(buy_price, 2),
                '下週收盤出場價': round(sell_price, 2),
                '週報酬%': profit_pct
            })

    return trades

def has_any_ad_signal(df_group, continuous_weeks=4, min_growth=0.1, last_week_threshold=2.0, pop_decline_threshold=0.5, **kwargs):
    """預篩器：只負責檢查最基礎的 A, B, C 條件，絕不算相關係數，確保極速！"""
    df = df_group.sort_values('資料日期', ascending=True).reset_index(drop=True)
    large_holder_series = _get_large_holder_series(df)

    # ⚡ 直接呼叫向量化引擎，瞬間算出符合 A, B, C 的列
    valid_indices = _get_signal_indices(
        df, 
        large_holder_series, 
        continuous_weeks, 
        min_growth, 
        last_week_threshold, 
        pop_decline_threshold
    )

    # 只要有任何一週符合條件 A, B, C，就回傳 True 讓它進入正式回測
    return len(valid_indices) > 0

# ==========================================
# 🧠 系統總司令函式 (自動掛載 AI 大腦版)
# ==========================================
def run_all_analysis(target_list):
    all_dfs = []
    all_trades = []
    total = len(target_list)

    # 🌟 讀取 GA 訓練出來的 JSON 參數檔
    best_params_path = os.path.join(os.path.dirname(__file__), 'best_params.json')
    try:
        with open(best_params_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)['params']
            print(f"✅ 成功載入 AI 黃金參數：{best_params}")
    except FileNotFoundError:
        print("⚠️ 找不到 best_params.json！系統將使用保守預設值。請記得去跑 run_ga.py！")
        best_params = {'continuous_weeks': 4, 'min_growth': 0.1, 'last_week_threshold': 2.0, 'pop_decline_threshold': 0.5}

    for i, sid in enumerate(target_list):
        print(f"[{i + 1}/{total}] {sid}...", end=" ", flush=True)
        
        df = crawler.get_individual_stock_data(sid)
        if df is None or df.empty:
            print("Skip (無籌碼資料)")
            continue

        price_data = crawler.download_stock_price_history(sid)
        if price_data is None or price_data.empty:
            print("Skip (無價格數據)")
            continue

        # 🌟 精準傳遞 AI 參數給預篩器
        if not has_any_ad_signal(df, **best_params):
            print("Skip (未觸發A~D)")
            continue

        all_dfs.append(df)
        
        # 🌟 精準傳遞 AI 參數給核心回測引擎
        trades = backtest_squeeze_strategy(df, **best_params)
        all_trades.extend(trades)

        print(f"OK ({len(df)}週籌碼, 訊號{len(trades)}筆)")

    if all_trades:
        trades_df = pd.DataFrame(all_trades).sort_values(['進場日期(籌碼公告)', '代號'], ascending=[False, True])
        return trades_df
    else:
        return pd.DataFrame()

# ==========================================
# 終端機執行主程式
# ==========================================
if __name__ == "__main__":
    stock_list = crawler.get_stock_ids(crawler.list_url)
    total_available = len(stock_list)

    if total_available == 0:
        print("❌ 沒抓到股票清單，程式結束。")
        raise SystemExit

    print(f"\n✅ 成功取得 {total_available} 檔股票清單。")
    print("--------------------------------")
    print("1. 前 10 個 (快速測試)")
    print("2. 前 50 個 (建議)")
    print(f"3. 全部 ({total_available} 個)")
    print("4. 自訂範圍")
    print("--------------------------------")

    choice = input("👉 請輸入選項 (1/2/3/4): ").strip()
    start_index, end_index = 0, 10

    if choice == '2':
        end_index = min(50, total_available)
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
    print(f"\n準備抓取 {len(target_list)} 檔股票的籌碼資料與 Yahoo 歷史開盤價...\n")

    trades_df = run_all_analysis(target_list)

    if not trades_df.empty:
        print("\n" + "=" * 90)
        print("📈 籌碼策略回測結果 (模組化升級版)")
        print("=" * 90)
        print(tabulate(trades_df, headers='keys', tablefmt='simple', showindex=False))
        
        completed_trades = trades_df.dropna(subset=['週報酬%'])
        if not completed_trades.empty:
            win_rate = (completed_trades['週報酬%'] > 0).mean() * 100
            avg_return = completed_trades['週報酬%'].mean()
            print(f"\n【總體績效統計】")
            print(f"勝率: {win_rate:.2f}% | 平均週報酬: {avg_return:.2f}% | 訊號總數: {len(completed_trades)} (已結算)")
    else:
        print("\n⚠️ 沒有符合條件的回測訊號。")
