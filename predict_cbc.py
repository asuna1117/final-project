import pandas as pd
import numpy as np
import crawler  
from tabulate import tabulate
import re
import unicodedata

# ==========================================
# 核心邏輯：判斷某個時間點是否符合進場條件
# ==========================================
def check_conditions(df, i, continuous_weeks=4, min_growth=0.4, pop_decline_threshold=0.4,
                     corr_window=156, large_corr_thresh=0.6, retail_corr_thresh=-0.6, avg_corr_thresh=0.6):
    large_holder_col = '>400張百分比'
    
    if i < continuous_weeks: return False, 0, 0, 0, 0

    weekly_growth_a = [((df.at[i-j, large_holder_col] - df.at[i-j-1, large_holder_col]) / df.at[i-j-1, large_holder_col]) * 100 if df.at[i-j-1, large_holder_col] > 0 else -np.inf for j in range(continuous_weeks)]
    if not all(g > min_growth for g in weekly_growth_a): return False, 0, 0, 0, 0

    weekly_growth_b = [((df.at[i-j, '平均張數/人'] - df.at[i-j-1, '平均張數/人']) / df.at[i-j-1, '平均張數/人']) * 100 if df.at[i-j-1, '平均張數/人'] > 0 else -np.inf for j in range(continuous_weeks)]
    if not all(g > min_growth for g in weekly_growth_b): return False, 0, 0, 0, 0

    pop_decline_pct = ((df.at[i-continuous_weeks, '總股東人數'] - df.at[i, '總股東人數']) / df.at[i-continuous_weeks, '總股東人數']) * 100
    if pop_decline_pct <= pop_decline_threshold: return False, 0, 0, 0, 0

    actual_window = min(corr_window, i + 1)
    x_large = df.loc[i-actual_window+1:i, large_holder_col].reset_index(drop=True)
    x_avg_per_person = df.loc[i-actual_window+1:i, '平均張數/人'].reset_index(drop=True)
    x_shareholders = df.loc[i-actual_window+1:i, '總股東人數'].reset_index(drop=True)
    y_close = df.loc[i-actual_window+1:i, '收盤價'].reset_index(drop=True) 

    corr_val = x_large.corr(y_close)
    avg_corr_val = x_avg_per_person.corr(y_close)
    retail_corr_val = x_shareholders.corr(y_close)

    corr_val = 0.0 if pd.isna(corr_val) else corr_val
    avg_corr_val = 0.0 if pd.isna(avg_corr_val) else avg_corr_val
    retail_corr_val = 0.0 if pd.isna(retail_corr_val) else retail_corr_val

    if corr_val >= large_corr_thresh or avg_corr_val >= avg_corr_thresh or retail_corr_val <= retail_corr_thresh:
        return True, corr_val, retail_corr_val, avg_corr_val, actual_window

    return False, 0, 0, 0, 0

# ==========================================
# 預測與歷史釣魚模組
# ==========================================
def scan_latest_and_history(df): 
    stock_id = df['股票代號'].iloc[0]
    i_latest = len(df) - 1
    
    is_triggered, corr, retail_corr, avg_corr, actual_win = check_conditions(df, i_latest)
    if not is_triggered:
        return None, None

    past_trades = []
    
    for i_hist in range(4, len(df)-1):
        hist_trigger, _, _, _, _ = check_conditions(df, i_hist)
        
        if hist_trigger:
            buy_price = df.at[i_hist, '收盤價']
            prev_price = buy_price
            consecutive_drops = 0
            exit_k = 0
            weekly_records = [] 

            for k in range(1, len(df) - i_hist):
                curr_price = df.at[i_hist+k, '收盤價']
                week_ret = ((curr_price - prev_price) / prev_price) * 100
                weekly_records.append(f"W{k}: {week_ret:+.1f}%")

                if week_ret < 0:
                    consecutive_drops += 1
                else:
                    consecutive_drops = 0

                prev_price = curr_price
                exit_k = k

                if consecutive_drops >= 2:
                    break

            cum_ret = ((prev_price - buy_price) / buy_price) * 100
            past_trades.append({
                '進場日': df.at[i_hist, '資料日期'],
                '持股週數': exit_k,
                '累積報酬': cum_ret,
                '歷程': ", ".join(weekly_records),
                '開局秒出場': consecutive_drops >= 2 and exit_k == 2 
            })

    suggestion = '🎯 建議進場'
    hist_summary = "無歷史前例"
    hist_details_str = "無"
    
    if past_trades:
        avg_ret = np.mean([t['累積報酬'] for t in past_trades])
        
        bad_starts = sum(1 for t in past_trades if t['開局秒出場'] and t['累積報酬'] < 0)
        if avg_ret < 0 or (bad_starts / len(past_trades) >= 0.5):
            suggestion = '❌ 回測不佳'

        hist_summary = f"發生 {len(past_trades)} 次, 平均 {avg_ret:+.2f}%"
        
        details_list = []
        for pt in past_trades:
            status = "⚠️ 連跌兩週停損" if pt['開局秒出場'] else "✅波段結算"
            # 確保格式一致，方便下面主程式進行字串切割與排版
            details_list.append(f"[{pt['進場日']}] 總計 {pt['累積報酬']:>+5.1f}% | 軌跡: {pt['歷程']} ({status})")
        
        hist_details_str = "\n".join(details_list)

    result_dict = {
        '代號': stock_id,
        '發布日': df.at[i_latest, '資料日期'],
        f'大戶({actual_win}週)': round(float(corr), 3),
        f'散戶({actual_win}週)': round(float(retail_corr), 3),
        f'均張({actual_win}週)': round(float(avg_corr), 3),
        '收盤價': df.at[i_latest, '收盤價'],
        '相似型態勝率': hist_summary,
        '歷史走勢明細': hist_details_str, 
        '建議': suggestion
    }

    return result_dict, past_trades

# ==========================================
# 預測總司令
# ==========================================
def get_next_week_recommendations(target_list):
    recommendations = []
    total = len(target_list)

    for i, sid in enumerate(target_list):
        print(f"🔎 掃描預測 [{i + 1}/{total}] {sid}...", end="\r", flush=True) 
        
        df = crawler.get_individual_stock_data(sid)
        if df is None or df.empty:
            continue

        res, past_trades = scan_latest_and_history(df)
        
        if res:
            recommendations.append(res)
            print(f"🔎 掃描預測 [{i + 1}/{total}] {sid}... 🔔 發現預測訊號！{' ' * 20}")

    if recommendations:
        return pd.DataFrame(recommendations).sort_values('代號')
    else:
        return pd.DataFrame()

# ==========================================
# 輔助函式：計算中英文混合字串的視覺寬度
# ==========================================
def get_display_width(text):
    """精準計算終端機上的字元寬度 (全形佔2格，半形佔1格)"""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    print("🚀 啟動下週推薦股掃描引擎...")
    
    stock_list = crawler.get_stock_ids(crawler.list_url)
    
    if not stock_list:
        print("❌ 無法取得股票清單，程式結束。")
    else:
        target_list = stock_list 
        print(f"準備掃描全部共 {len(target_list)} 檔股票... \n")
        
        recommend_df = get_next_week_recommendations(target_list)
        
        print("\n" + "=" * 110)
        if not recommend_df.empty:
            print("🎯 掃描完畢！發現以下【下週實戰推薦清單】：")
            print("=" * 110)
            
            display_df = recommend_df.drop(columns=['歷史走勢明細'])
            print(tabulate(display_df, headers='keys', tablefmt='simple', showindex=False))
            
            print("\n" + "=" * 110)
            print("📜 【歷史相似走勢 - 深度明細解析】")
            print("=" * 110)
            
            # 🌟 魔法核心：自動偵測寬度並精準換行對齊
            for idx, row in recommend_df.iterrows():
                print(f"🔸 股票代號: 【 {row['代號']} 】 | 綜合建議: {row['建議']}")
                if row['歷史走勢明細'] == "無":
                    print("   └─ 歷史上尚無完全相同之訊號可供比對。")
                else:
                    trades = row['歷史走勢明細'].split('\n')
                    for trade_str in trades:
                        # 用正則表達式把字串切成：前綴、軌跡明細、結尾狀態
                        match = re.search(r'(.*軌跡: )(.*) (\(.*)', trade_str)
                        if match:
                            prefix = "   └─ " + match.group(1)
                            trajectory_str = match.group(2)
                            status_str = " " + match.group(3)
                            
                            weeks = trajectory_str.split(', ')
                            
                            # 測量前綴在終端機佔了多寬，產生對應數量的空白
                            indent_width = get_display_width(prefix)
                            indent_spaces = " " * indent_width
                            
                            # 每 8 週斷行一次 (你也可以改成 10 或其他數字)
                            chunk_size = 8
                            lines = []
                            for i in range(0, len(weeks), chunk_size):
                                lines.append(", ".join(weeks[i:i+chunk_size]))
                            
                            # 組合！第一行不加空白，第二行開始加上精準的空白縮排
                            formatted_trajectory = f",\n{indent_spaces}".join(lines)
                            
                            print(f"{prefix}{formatted_trajectory}{status_str}")
                        else:
                            # 萬一格式抓錯的防呆機制
                            print(f"   └─ {trade_str}")
                print("-" * 110)

            print("\n💡 判讀教學：")
            print("若「綜合建議」顯示為『❌ 回測不佳』，代表此股票過去發生相同訊號時，")
            print("多半會立刻遭遇連續兩週下跌的停損出場，或歷史平均報酬為負，請避開陷阱。")
        else:
            print("⚠️ 掃描完畢，目前的清單中【沒有】剛好在最新一週觸發進場訊號的股票。")
            print("=" * 110)