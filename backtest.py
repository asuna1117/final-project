import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tabulate import tabulate
import crawler # 🌟 引入剛剛建好的爬蟲模組
import crawler_finmind

ml_dataset = []

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
# 核心功能：回測邏輯 (相關係數使用進場日前全部週次)
# ==========================================
def backtest_squeeze_strategy(df_group, continuous_weeks=3, min_growth=0.0479, last_week_threshold=0.179, pop_decline_threshold=0.198,
                              corr_window=156, large_corr_thresh=0.6, 
                              retail_corr_thresh=-0.6, avg_corr_thresh=0.6, skip_cond_e=False):
    
    stock_id = df_group['股票代號'].iloc[0]
    df = df_group.sort_values('資料日期', ascending=True).reset_index(drop=True)
    trades = []
    
    large_holder_series = _get_large_holder_series(df)
    
    if len(df) < continuous_weeks + 1: return []
    
    for i in range(continuous_weeks, len(df)-1):

        # 條件 A: 連續 4 週每週漲幅皆 > 0，且最後一週 > last_week_threshold
        weekly_growth_a = [((large_holder_series.iat[i-j] - large_holder_series.iat[i-j-1]) / large_holder_series.iat[i-j-1]) * 100 if large_holder_series.iat[i-j-1] > 0 else -np.inf for j in range(continuous_weeks)]
        is_continuous_buy = all(g > 0 for g in weekly_growth_a) and (weekly_growth_a[0] > last_week_threshold)
        
        # 條件 B: 平均張數/人連續 4 週每週漲幅皆 > 0.1%
        weekly_growth_b = [((df.at[i-j, '平均張數/人'] - df.at[i-j-1, '平均張數/人']) / df.at[i-j-1, '平均張數/人']) * 100 if df.at[i-j-1, '平均張數/人'] > 0 else -np.inf for j in range(continuous_weeks)]
        is_avg_per_person_continuous_up = all(g > min_growth for g in weekly_growth_b)
        
        # 條件 C: 總股東人數 4 週總下跌 > 0.5%
        pop_decline_pct = ((df.at[i-continuous_weeks, '總股東人數'] - df.at[i, '總股東人數']) / df.at[i-continuous_weeks, '總股東人數']) * 100

        if is_continuous_buy and is_avg_per_person_continuous_up and pop_decline_pct > pop_decline_threshold:

            # 計算進場日前全部週次特徵與下一週收盤價的相關係數
            # 使用配對 (X_t, Y_{t+1})，僅用到進場公告日前資料。
            if i < 1:
                continue

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

            # 條件 D: 相關係數門檻
            if not (corr_val >= large_corr_thresh or avg_corr_val >= avg_corr_thresh or retail_corr_val <= retail_corr_thresh):
                continue

            # 🌟 修復：確保傳給 crawler 的日期是標準的 'YYYY-MM-DD' 字串
            entry_date = df.at[i, '資料日期']
            if isinstance(entry_date, pd.Timestamp):
                date_str = entry_date.strftime('%Y-%m-%d')
            else:
                date_str = str(entry_date)

            # 🌟 呼叫 crawler 裡的抓股價功能 (使用 date_str)
            buy_price = crawler.get_next_monday_open_price(stock_id, date_str)
            sell_price = crawler.get_next_friday_close_price(stock_id, date_str)

            # 🌟 修正條件 E 區塊：如果 skip_cond_e 為 True，就不檢查連漲條件
            if buy_price <= 0 or pd.isna(sell_price):
                continue
                
            if not skip_cond_e and not crawler.check_condition_e_with_yfinance(stock_id, date_str, buy_price):
                continue
                
            if buy_price > 0 and not pd.isna(sell_price):
                profit_pct = ((sell_price - buy_price) / buy_price) * 100
                
                # ==========================================
                # 🌟 新增：提取 FinMind 外部籌碼特徵
                # ==========================================
                # 計算外資淨買賣超 (買進 - 賣出)，並加入防呆處理確認欄位存在
                if 'Foreign_Buy_Sum' in df.columns and 'Foreign_Sell_Sum' in df.columns:
                    foreign_net_buy = df.at[i, 'Foreign_Buy_Sum'] - df.at[i, 'Foreign_Sell_Sum']
                else:
                    foreign_net_buy = 0
                    
                margin_bal = df.at[i, 'MarginPurchaseBalance'] if 'MarginPurchaseBalance' in df.columns else 0
                short_bal = df.at[i, 'ShortSaleBalance'] if 'ShortSaleBalance' in df.columns else 0

                # 🌟 提取機器學習要用的特徵 (包含神秘金字塔與 FinMind 數據)
                trades.append({
                    '代號': stock_id,
                    '進場日期(籌碼公告)': df.at[i, '資料日期'],
                    
                    # --- 機器學習特徵 (X) ---
                    '大戶相關係數': round(float(corr_val), 3),
                    '散戶相關係數': round(float(retail_corr_val), 3),
                    '均張相關係數': round(float(avg_corr_val), 3),
                    '大戶四週成長率': round(float(weekly_growth_a[0]), 3), 
                    '散戶衰退率': round(float(pop_decline_pct), 3),
                    '外資買賣超': float(foreign_net_buy),  # 新增外部特徵
                    '融資餘額': float(margin_bal),        # 新增外部特徵
                    '融券餘額': float(short_bal),         # 新增外部特徵
                    
                    # --- 機器學習標籤 (y) ---
                    '週一開盤進場價': round(buy_price, 2),
                    '下週收盤出場價': round(sell_price, 2),
                    '週報酬%': profit_pct,
                    '是否獲利': 1 if profit_pct > 0 else 0  
                })

                # 🌟 提取純籌碼特徵，加入 ml_dataset
                ml_dataset.append({
                    '股票代號': stock_id,
                    '進場日期': df.at[i, '資料日期'],
                    
                    # --- 特徵 (X) ---
                    '大戶相關係數': round(float(corr_val), 3),
                    '散戶相關係數': round(float(retail_corr_val), 3),
                    '均張相關係數': round(float(avg_corr_val), 3),
                    '大戶四週成長率': round(float(weekly_growth_a[0]), 3), 
                    '散戶衰退率': round(float(pop_decline_pct), 3),
                    '外資買賣超': float(foreign_net_buy),  # 新增外部特徵
                    '融資餘額': float(margin_bal),        # 新增外部特徵
                    '融券餘額': float(short_bal),         # 新增外部特徵
                    
                    # --- 標籤 (y) ---
                    '是否獲利': 1 if profit_pct > 0 else 0  
                })

    return trades


def has_any_ad_signal(df_group, continuous_weeks=3, min_growth=0.0479, last_week_threshold=0.179, pop_decline_threshold=0.198,
                      corr_window=156, large_corr_thresh=0.6,
                      retail_corr_thresh=-0.6, avg_corr_thresh=0.6, skip_cond_e=False):
    """檢查是否曾出現符合 A~D 的任一訊號，作為是否進入 Yahoo 抓價流程的預篩。"""
    df = df_group.sort_values('資料日期', ascending=True).reset_index(drop=True)
    large_holder_series = _get_large_holder_series(df)

    if len(df) < continuous_weeks + 2:
        return False

    for i in range(continuous_weeks, len(df) - 1):
        weekly_growth_a = [((large_holder_series.iat[i-j] - large_holder_series.iat[i-j-1]) / large_holder_series.iat[i-j-1]) * 100 if large_holder_series.iat[i-j-1] > 0 else -np.inf for j in range(continuous_weeks)]
        is_continuous_buy = all(g > 0 for g in weekly_growth_a) and (weekly_growth_a[0] > last_week_threshold)

        weekly_growth_b = [((df.at[i-j, '平均張數/人'] - df.at[i-j-1, '平均張數/人']) / df.at[i-j-1, '平均張數/人']) * 100 if df.at[i-j-1, '平均張數/人'] > 0 else -np.inf for j in range(continuous_weeks)]
        is_avg_per_person_continuous_up = all(g > min_growth for g in weekly_growth_b)

        pop_decline_pct = ((df.at[i-continuous_weeks, '總股東人數'] - df.at[i, '總股東人數']) / df.at[i-continuous_weeks, '總股東人數']) * 100
        if not (is_continuous_buy and is_avg_per_person_continuous_up and pop_decline_pct > pop_decline_threshold):
            continue

        if i < 1:
            continue

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

        if corr_val >= large_corr_thresh or avg_corr_val >= avg_corr_thresh or retail_corr_val <= retail_corr_thresh:
            return True

    return False


# ==========================================
# 🌟 回測總司令函式 (支援動態參數與獨立訓練模式)
# ==========================================
def run_all_analysis(target_list, params=None, is_training=False):
    # 若未傳遞參數，給予空字典，讓內部函式使用預設值
    if params is None:
        params = {}
        
    all_dfs = []
    all_trades = []
    total = len(target_list)
    
    # 每次執行前先清空全域的機器學習特徵庫，避免重複疊加
    global ml_dataset
    if is_training:
        ml_dataset = [] 

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

        # 🌟 動態傳入參數 (預篩選)
        if not has_any_ad_signal(df, **params):
            print("Skip (未觸發A~D)")
            continue

        df['資料日期'] = pd.to_datetime(df['資料日期'])
        df = df.sort_values('資料日期') 
        start_date = df['資料日期'].min().strftime('%Y-%m-%d')
        end_date = df['資料日期'].max().strftime('%Y-%m-%d')
        
        df_finmind = crawler_finmind.fetch_weekly_chip_data(sid, start_date, end_date)
        
        if df_finmind is not None and not df_finmind.empty:
            df_finmind['date'] = pd.to_datetime(df_finmind['date'])
            df_finmind = df_finmind.sort_values('date')
            df = pd.merge_asof(
                df, df_finmind, left_on='資料日期', right_on='date', direction='backward', tolerance=pd.Timedelta(days=3)
            )
            
            if 'MarginPurchaseBalance' in df.columns:
                df['MarginPurchaseBalance'] = df['MarginPurchaseBalance'].ffill().fillna(0)
            if 'ShortSaleBalance' in df.columns:
                df['ShortSaleBalance'] = df['ShortSaleBalance'].ffill().fillna(0)
            if 'Foreign_Buy_Sum' in df.columns:
                df['Foreign_Buy_Sum'] = df['Foreign_Buy_Sum'].fillna(0)
            if 'Foreign_Sell_Sum' in df.columns:
                df['Foreign_Sell_Sum'] = df['Foreign_Sell_Sum'].fillna(0)
                
            print(f"[{sid}] 合併處理後資料筆數: {len(df)}")
        else:
            print(f"[{sid}] ⚠️ 查無 FinMind 資料，將略過合併")

        all_dfs.append(df)
        
        # 🌟 動態傳入參數 (主策略回測)
        trades = backtest_squeeze_strategy(df, **params)
        all_trades.extend(trades)

        print(f"OK ({len(df)}週籌碼, 訊號{len(trades)}筆)")

    # 針對訓練模式特製的防呆清空邏輯 (確保不影響非訓練模式)
    if not is_training:
        ml_dataset = []

    if all_trades:
        trades_df = pd.DataFrame(all_trades).sort_values(['進場日期(籌碼公告)', '代號'], ascending=[False, True])
        return trades_df
    else:
        return pd.DataFrame()
    
# ==========================================
# 匯出機器學習資料的專屬函式 (給 test.py 呼叫)
# ==========================================
def export_ml_data():
    global ml_dataset # 宣告使用全域變數
    if ml_dataset:
        print("\n" + "=" * 90)
        print("💾 正在匯出機器學習特徵...")
        ml_df = pd.DataFrame(ml_dataset)
        ml_df.to_csv("ml_training_data.csv", index=False, encoding="utf-8-sig")
        print(f"✅ 機器學習訓練資料已匯出至 ml_training_data.csv，共 {len(ml_df)} 筆樣本。")
    else:
        print("\n⚠️ 此次執行沒有產生任何可用於 ML 訓練的資料。")

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
