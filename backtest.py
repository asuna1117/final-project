import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tabulate import tabulate
import crawler # 🌟 引入剛剛建好的爬蟲模組




# ==========================================
# 核心功能：回測邏輯 (相關係數升級為動態 3 年/156週)
# ==========================================
def backtest_squeeze_strategy(df_group, continuous_weeks=4, min_growth=0.1, pop_decline_threshold=0.5,
                              corr_window=156, large_corr_thresh=0.6, 
                              retail_corr_thresh=-0.6, avg_corr_thresh=0.6): 
    
    stock_id = df_group['股票代號'].iloc[0]
    df = df_group.sort_values('資料日期', ascending=True).reset_index(drop=True)
    trades = []
    
    large_holder_col = '>400張百分比'
    
    if len(df) < continuous_weeks + 1: return []
    
    for i in range(continuous_weeks, len(df)-1):

        # 條件 A: 連續 4 週每週漲幅皆 > 0.1%
        weekly_growth_a = [((df.at[i-j, large_holder_col] - df.at[i-j-1, large_holder_col]) / df.at[i-j-1, large_holder_col]) * 100 if df.at[i-j-1, large_holder_col] > 0 else -np.inf for j in range(continuous_weeks)]
        is_continuous_buy = all(g > min_growth for g in weekly_growth_a)
        
        # 條件 B: 平均張數/人連續 4 週每週漲幅皆 > 0.1%
        weekly_growth_b = [((df.at[i-j, '平均張數/人'] - df.at[i-j-1, '平均張數/人']) / df.at[i-j-1, '平均張數/人']) * 100 if df.at[i-j-1, '平均張數/人'] > 0 else -np.inf for j in range(continuous_weeks)]
        is_avg_per_person_continuous_up = all(g > min_growth for g in weekly_growth_b)
        
        # 條件 C: 總股東人數 4 週總下跌 > 0.5%
        pop_decline_pct = ((df.at[i-continuous_weeks, '總股東人數'] - df.at[i, '總股東人數']) / df.at[i-continuous_weeks, '總股東人數']) * 100

        if is_continuous_buy and is_avg_per_person_continuous_up and pop_decline_pct > pop_decline_threshold:
            
            # 計算 156 週特徵與下週收盤價的相關係數
            actual_window = min(corr_window, i + 1)
            
            x_large = df.loc[i-actual_window+1:i, large_holder_col].reset_index(drop=True)
            x_avg_per_person = df.loc[i-actual_window+1:i, '平均張數/人'].reset_index(drop=True)
            x_shareholders = df.loc[i-actual_window+1:i, '總股東人數'].reset_index(drop=True)
            y_next_close = df.loc[i-actual_window+2:i+1, '收盤價'].reset_index(drop=True)

            corr_val = x_large.corr(y_next_close)
            avg_corr_val = x_avg_per_person.corr(y_next_close)
            retail_corr_val = x_shareholders.corr(y_next_close)

            corr_val = 0.0 if pd.isna(corr_val) else corr_val
            avg_corr_val = 0.0 if pd.isna(avg_corr_val) else avg_corr_val
            retail_corr_val = 0.0 if pd.isna(retail_corr_val) else retail_corr_val

            # 條件 D: 相關係數門檻
            if not (corr_val >= large_corr_thresh or avg_corr_val >= avg_corr_thresh or retail_corr_val <= retail_corr_thresh):
                continue

            # 🌟 呼叫 crawler 裡的抓股價功能
            buy_price = crawler.get_next_monday_open_price(stock_id, df.at[i, '資料日期'])
            sell_price = df.at[i+1, '收盤價'] 

            # 條件 E: 檢查下週二到下週四收盤價連續走高
            if buy_price <= 0 or not crawler.check_condition_e_with_yfinance(stock_id, df.at[i, '資料日期'], buy_price):
                continue
            
            if buy_price > 0:
                profit_pct = ((sell_price - buy_price) / buy_price) * 100
                trades.append({
                    '代號': stock_id,
                    '進場日期(籌碼公告)': df.at[i, '資料日期'],
                    f'大戶相關係數({actual_window}週)': round(float(corr_val), 3),
                    f'散戶相關係數({actual_window}週)': round(float(retail_corr_val), 3),
                    f'平均張數相關({actual_window}週)': round(float(avg_corr_val), 3),
                    '週一開盤進場價': round(buy_price, 2),
                    '下週收盤出場價': round(sell_price, 2) if sell_price else None,
                    '週報酬%': profit_pct
                })

    return trades

# ==========================================
# 回測總司令函式
# ==========================================
def run_all_analysis(target_list):
    all_dfs = []
    all_trades = []
    total = len(target_list)

    for i, sid in enumerate(target_list):
        print(f"[{i + 1}/{total}] {sid}...", end=" ", flush=True)
        
        # 🌟 呼叫 crawler 抓資料
        df = crawler.get_individual_stock_data(sid)
        if df is None or df.empty:
            print("Skip (無籌碼資料)")
            continue

        all_dfs.append(df)
        trades = backtest_squeeze_strategy(df)
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
        display_df = trades_df.fillna({'下週收盤出場價': '等待開獎', '週報酬%': '等待開獎'})
        print("\n" + "=" * 90)
        print("📈 籌碼策略回測結果 (模組化升級版)")
        print("=" * 90)
        print(tabulate(display_df, headers='keys', tablefmt='simple', showindex=False))
        
        completed_trades = trades_df.dropna(subset=['週報酬%'])
        if not completed_trades.empty:
            win_rate = (completed_trades['週報酬%'] > 0).mean() * 100
            avg_return = completed_trades['週報酬%'].mean()
            print(f"\n【總體績效統計】")
            print(f"勝率: {win_rate:.2f}% | 平均週報酬: {avg_return:.2f}% | 訊號總數: {len(completed_trades)} (已結算)")
    else:
        print("\n⚠️ 沒有符合條件的回測訊號。")