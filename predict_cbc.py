import pandas as pd
import numpy as np
import json
import os
import crawler
import backtest

# ==========================================
# 🔧 輔助函式：動態讀取 GA 最佳化參數
# ==========================================
def load_best_params():
    """自動讀取 run_ga.py 訓練出來的 best_params.json"""
    best_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')
    try:
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('params', {})
    except Exception as e:
        print(f"⚠️ 讀取最佳參數失敗，將使用預設值 ({e})")
    return {}

# ==========================================
# 🧠 核心邏輯：掃描最新一週並比對歷史軌跡
# ==========================================
def scan_latest_and_history(df, **kwargs):
    """
    掃描股票的「最新一週」是否觸發訊號，若觸發則回溯歷史勝率。
    回傳: (結果字典, 歷史明細字串)
    """
    if df is None or df.empty:
        return None, None
        
    # 1. 載入參數 (優先使用 kwargs，否則用 GA 的最佳參數，最後用預設值)
    ga_params = load_best_params()
    ga_params.update(kwargs) # app.py 傳來的參數會覆蓋 GA 參數
    
    c_weeks = int(ga_params.get('continuous_weeks', 4))
    min_g = ga_params.get('min_growth', 0.1)
    last_w_thresh = ga_params.get('last_week_threshold', 0.5)
    pop_d = ga_params.get('pop_decline_threshold', 0.5)
    use_tej_filters = ga_params.get('use_tej_filters', False)
    foreign_net_min = ga_params.get('foreign_net_min', 0)
    investment_net_min = ga_params.get('investment_net_min', 0)
    margin_change_max = ga_params.get('margin_change_max', 0)
    short_change_min = ga_params.get('short_change_min', 0)
    use_twse_extra_filters = ga_params.get('use_twse_extra_filters', False)
    require_foreign_3d = ga_params.get('require_foreign_3d', False)
    require_margin_low = ga_params.get('require_margin_low', False)
    margin_low_lookback_days = ga_params.get('margin_low_lookback_days', 60)
    margin_low_percentile = ga_params.get('margin_low_percentile', 20)
    
    df = df.sort_values('資料日期', ascending=True).reset_index(drop=True)
    large_holder_series = backtest._get_large_holder_series(df)
    
    if len(df) < c_weeks + 1:
        return None, None
        
    # 2. 鎖定「最新一週」進行條件判定
    latest_idx = len(df) - 1
    
    # 條件 A: 大戶連續買超，且最後一週達標
    weekly_growth_a = [((large_holder_series.iat[latest_idx-j] - large_holder_series.iat[latest_idx-j-1]) / large_holder_series.iat[latest_idx-j-1]) * 100 if large_holder_series.iat[latest_idx-j-1] > 0 else -np.inf for j in range(c_weeks)]
    is_continuous_buy = all(g > 0 for g in weekly_growth_a) and (weekly_growth_a[0] > last_w_thresh)
    
    # 條件 B: 平均張數/人連續上升
    weekly_growth_b = [((df.at[latest_idx-j, '平均張數/人'] - df.at[latest_idx-j-1, '平均張數/人']) / df.at[latest_idx-j-1, '平均張數/人']) * 100 if df.at[latest_idx-j-1, '平均張數/人'] > 0 else -np.inf for j in range(c_weeks)]
    is_avg_per_person_continuous_up = all(g > min_g for g in weekly_growth_b)
    
    # 條件 C: 總股東人數下降
    pop_decline_pct = ((df.at[latest_idx-c_weeks, '總股東人數'] - df.at[latest_idx, '總股東人數']) / df.at[latest_idx-c_weeks, '總股東人數']) * 100

    if use_tej_filters:
        tej_checks = []
        if '外資買賣超' in df.columns:
            v = df.at[latest_idx, '外資買賣超']
            tej_checks.append(pd.notna(v) and v >= foreign_net_min)
        if '投信買賣超' in df.columns:
            v = df.at[latest_idx, '投信買賣超']
            tej_checks.append(pd.notna(v) and v >= investment_net_min)
        if '融資增減' in df.columns:
            v = df.at[latest_idx, '融資增減']
            tej_checks.append(pd.notna(v) and v <= margin_change_max)
        if '融券增減' in df.columns:
            v = df.at[latest_idx, '融券增減']
            tej_checks.append(pd.notna(v) and v >= short_change_min)
        if tej_checks and not all(tej_checks):
            return None, None

    if use_twse_extra_filters:
        twse_checks = []
        if require_foreign_3d:
            twse_checks.append(crawler.check_twse_foreign_consecutive_buy(df['股票代號'].iloc[0], df.at[latest_idx, '資料日期'], consecutive_days=3))
        if require_margin_low:
            twse_checks.append(
                crawler.check_twse_margin_balance_low(
                    df['股票代號'].iloc[0],
                    df.at[latest_idx, '資料日期'],
                    lookback_days=margin_low_lookback_days,
                    percentile=margin_low_percentile,
                )
            )
        if twse_checks and not all(twse_checks):
            return None, None
    
    # 若最新一週沒觸發，直接略過
    if not (is_continuous_buy and is_avg_per_person_continuous_up and pop_decline_pct > pop_d):
        return None, None
        
    # ==========================================
    # 🎯 進入歷史回測區 (因為最新一週觸發了！)
    # ==========================================
    stock_id = df['股票代號'].iloc[0]
    history_details = []
    win_count = 0
    total_history = 0
    
    # 掃描過去所有週次，尋找相同的型態
    for j in range(c_weeks, len(df) - 1):
        h_growth_a = [((large_holder_series.iat[j-k] - large_holder_series.iat[j-k-1]) / large_holder_series.iat[j-k-1]) * 100 if large_holder_series.iat[j-k-1] > 0 else -np.inf for k in range(c_weeks)]
        h_is_buy = all(g > 0 for g in h_growth_a) and (h_growth_a[0] > last_w_thresh)
        
        h_growth_b = [((df.at[j-k, '平均張數/人'] - df.at[j-k-1, '平均張數/人']) / df.at[j-k-1, '平均張數/人']) * 100 if df.at[j-k-1, '平均張數/人'] > 0 else -np.inf for k in range(c_weeks)]
        h_is_avg_up = all(g > min_g for g in h_growth_b)
        
        h_pop_d = ((df.at[j-c_weeks, '總股東人數'] - df.at[j, '總股東人數']) / df.at[j-c_weeks, '總股東人數']) * 100

        if use_tej_filters:
            tej_checks = []
            if '外資買賣超' in df.columns:
                v = df.at[j, '外資買賣超']
                tej_checks.append(pd.notna(v) and v >= foreign_net_min)
            if '投信買賣超' in df.columns:
                v = df.at[j, '投信買賣超']
                tej_checks.append(pd.notna(v) and v >= investment_net_min)
            if '融資增減' in df.columns:
                v = df.at[j, '融資增減']
                tej_checks.append(pd.notna(v) and v <= margin_change_max)
            if '融券增減' in df.columns:
                v = df.at[j, '融券增減']
                tej_checks.append(pd.notna(v) and v >= short_change_min)
            if tej_checks and not all(tej_checks):
                continue

        if use_twse_extra_filters:
            twse_checks = []
            if require_foreign_3d:
                twse_checks.append(crawler.check_twse_foreign_consecutive_buy(stock_id, df.at[j, '資料日期'], consecutive_days=3))
            if require_margin_low:
                twse_checks.append(
                    crawler.check_twse_margin_balance_low(
                        stock_id,
                        df.at[j, '資料日期'],
                        lookback_days=margin_low_lookback_days,
                        percentile=margin_low_percentile,
                    )
                )
            if twse_checks and not all(twse_checks):
                continue
        
        # 歷史上也發生過一樣的訊號
        if h_is_buy and h_is_avg_up and h_pop_d > pop_d:
            total_history += 1
            buy_p = crawler.get_next_monday_open_price(stock_id, df.at[j, '資料日期'])
            sell_p = crawler.get_next_friday_close_price(stock_id, df.at[j, '資料日期'])
            
            if buy_p > 0 and not pd.isna(sell_p):
                ret = ((sell_p - buy_p) / buy_p) * 100
                if ret > 0: win_count += 1
                status = f"(報酬: {ret:.2f}%)"
                # 配合 test.py 的正則表達式，格式必須精準
                history_details.append(f"{df.at[j, '資料日期']} 軌跡: 買進 {buy_p:.2f}, 賣出 {sell_p:.2f} {status}")
            else:
                history_details.append(f"{df.at[j, '資料日期']} 軌跡: 缺價, 無法結算 (未平倉)")
                
    win_rate = (win_count / total_history * 100) if total_history > 0 else 0
    
    # 判定綜合建議
    if total_history == 0:
        recommendation = "⚠️ 尚無歷史可考 (盲測)"
        hist_str = "無"
    elif win_rate >= 50:
        recommendation = "🎯 建議進場"
        hist_str = "\n".join(history_details)
    else:
        recommendation = "❌ 回測不佳"
        hist_str = "\n".join(history_details)
        
    # 產出該股票的預測報告
    res = {
        '代號': stock_id,
        '最新觸發日': df.at[latest_idx, '資料日期'],
        '歷史觸發次數': total_history,
        '歷史勝率': f"{win_rate:.1f}%",
        '建議': recommendation,
        '歷史走勢明細': hist_str
    }
    return res, hist_str

# ==========================================
# 🚀 總司令：取得下週推薦清單 (供 test.py 呼叫)
# ==========================================
def get_next_week_recommendations(target_list, **kwargs):
    """輸入股票清單，自動回傳下週實戰買進 DataFrame"""
    import sys
    recommendations = []
    total = len(target_list)
    print(f"\n🔍 開始啟動預測雷達，掃描全市場最新訊號，共 {total} 檔...")
    
    # 🌟 新增：一開始就先載入「代號 -> 名稱」對應表
    stock_mapping = crawler.get_stock_name_mapping() if hasattr(crawler, 'get_stock_name_mapping') else {}
    
    for idx, sid in enumerate(target_list):
        print(f"  👉 [{idx+1:04d}/{total:04d}] 掃描 {sid}...", end="\r", flush=True)
        
        # 呼叫爬蟲取得籌碼資料
        df = crawler.get_individual_stock_data(sid)
        if df is not None and not df.empty:
            res, _ = scan_latest_and_history(df, **kwargs)
            if res:
                # 把名稱塞進這檔股票的預測結果中
                res['名稱'] = stock_mapping.get(sid, "未知")
                recommendations.append(res)
                
    print("\n✅ 掃描完成！準備產生實戰清單...")
    
    if recommendations:
        df_res = pd.DataFrame(recommendations)
        
        # 重新排序 DataFrame 欄位，讓「名稱」緊跟在「代號」後面
        cols = ['代號', '名稱', '最新觸發日', '歷史觸發次數', '歷史勝率', '建議', '歷史走勢明細']
        cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[cols]
        
        # 🌟 新增：客製化多重排序邏輯
        # 1. 定義「建議」欄位的自訂優先順序
        recommendation_order = ["🎯 建議進場", "❌ 回測不佳", "⚠️ 尚無歷史可考 (盲測)"]
        df_res['建議'] = pd.Categorical(df_res['建議'], categories=recommendation_order, ordered=True)
        
        # 2. 先照「建議」的自訂順序排，如果相同，再照「代號」由小到大排
        return df_res.sort_values(['建議', '代號'], ascending=[True, True])
    else:
        return pd.DataFrame()

