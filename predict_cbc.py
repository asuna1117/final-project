import pandas as pd
import numpy as np
import crawler  
import crawler_finmind # 🌟 新增：引入 FinMind 爬蟲
import backtest
from tabulate import tabulate
import re
import unicodedata
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

SEQUENCE_LENGTH = 5
FEATURE_COLS = [
    '1週報酬率', '4週報酬率', '8週報酬率',
    '5日均線偏離率', '20日均線偏離率', '成交量增幅',
    '外資淨買賣超/成交量', '融資融券比率','大戶持股比(TEJ)',
    '散戶持股比(TEJ)','大戶散戶差'
]

# ==========================================
# 模型載入區塊 
# ==========================================
model_filename = Path(__file__).resolve().parent / 'lstm_trading_model.keras'
scaler_filename = Path(__file__).resolve().parent / 'lstm_feature_scaler.npz'
try:
    with np.load(scaler_filename) as scaler_data:
        feature_mean = scaler_data['mean']
        feature_scale = scaler_data['scale']
        target_mean = float(scaler_data['target_mean'])
        target_scale = float(scaler_data['target_scale'])
    ml_model = keras.Sequential([
        keras.Input(shape=(SEQUENCE_LENGTH, len(FEATURE_COLS))),
        layers.LSTM(16, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    ml_model.load_weights(model_filename)
    print(f"✅ 成功載入機器學習大腦：{model_filename}")
except (FileNotFoundError, OSError, ValueError, KeyError):
    ml_model = None
    feature_mean = None
    feature_scale = None
    target_mean = 0.0
    target_scale = 1.0
    print(f"⚠️ 找不到模型或標準化參數，將跳過 ML 濾網功能。")


def _get_numeric_value(df, row_idx, candidates, default=0.0):
    """回傳 DataFrame 某一列的數值，支援中英欄位名與缺欄位安全回退。"""
    for col in candidates:
        if col in df.columns:
            value = df.at[row_idx, col]
            if pd.notna(value):
                return float(value)
    return float(default)


def _signed_log1p(value):
    if pd.isna(value):
        return 0.0
    return float(np.sign(value) * np.log1p(abs(value)))


def filter_with_ml(latest_features_dict):
    """傳入最新一週的特徵字典，回傳預測結果與報酬率。"""
    if ml_model is None:
        return True, 0.0

    if isinstance(latest_features_dict, dict):
        feature_rows = [latest_features_dict]
    else:
        feature_rows = latest_features_dict

    seq_values = pd.DataFrame(feature_rows)[FEATURE_COLS].to_numpy(dtype=float)
    if len(seq_values) < SEQUENCE_LENGTH:
        return False, 0.0
    seq_values = seq_values[-SEQUENCE_LENGTH:]
    seq_values = (seq_values - feature_mean) / feature_scale
    X_new = seq_values.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLS))

    pred_return_scaled = float(ml_model.predict(X_new, verbose=0)[0][0])
    pred_return = pred_return_scaled * target_scale + target_mean
    prediction = (pred_return >= 0.0)
    return prediction, pred_return

# ==========================================
# 核心邏輯：判斷某個時間點是否符合進場條件
# ==========================================
def check_conditions(df, i, continuous_weeks=3, min_growth=0.0479, pop_decline_threshold=0.198,
                     corr_window=156, large_corr_thresh=0.6, retail_corr_thresh=-0.6, avg_corr_thresh=0.6, 
                     last_week_threshold=0.179, skip_cond_e=False):
                     
    large_holder_col = '>400張百分比'
    if '>400張百分比' not in df.columns:
        return False, 0, 0, 0, 0
        
    if i < continuous_weeks: return False, 0, 0, 0, 0

    weekly_growth_a = [((df.at[i-j, large_holder_col] - df.at[i-j-1, large_holder_col]) / df.at[i-j-1, large_holder_col]) * 100 if df.at[i-j-1, large_holder_col] > 0 else -np.inf for j in range(continuous_weeks)]
    if not (all(g > 0 for g in weekly_growth_a) and (weekly_growth_a[0] > last_week_threshold)): 
        return False, 0, 0, 0, 0

    weekly_growth_b = [((df.at[i-j, '平均張數/人'] - df.at[i-j-1, '平均張數/人']) / df.at[i-j-1, '平均張數/人']) * 100 if df.at[i-j-1, '平均張數/人'] > 0 else -np.inf for j in range(continuous_weeks)]
    if not all(g > min_growth for g in weekly_growth_b): 
        return False, 0, 0, 0, 0

    pop_decline_pct = ((df.at[i-continuous_weeks, '總股東人數'] - df.at[i, '總股東人數']) / df.at[i-continuous_weeks, '總股東人數']) * 100
    if pop_decline_pct <= pop_decline_threshold: 
        return False, 0, 0, 0, 0

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
def _build_feature_row(stock_id, df, row_idx, price_df):
    row_date = pd.to_datetime(df.at[row_idx, '資料日期'])
    price_history = price_df[price_df.index <= row_date].sort_index()
    closes = pd.to_numeric(price_history['Close'], errors='coerce').dropna()
    volumes = pd.to_numeric(price_history['Volume'], errors='coerce').dropna()

    if closes.empty:
        return {feature: 0.0 for feature in FEATURE_COLS}

    close = float(closes.iloc[-1])

    def return_pct(lookback):
        if len(closes) <= lookback * 5:
            return 0.0
        previous_close = float(closes.iloc[-lookback * 5 - 1])
        return (close - previous_close) / previous_close * 100 if previous_close else 0.0

    def moving_average_deviation(window):
        if len(closes) < window:
            return 0.0
        average_close = closes.tail(window).mean()
        return (close - average_close) / average_close * 100 if average_close else 0.0

    latest_volume = float(volumes.iloc[-1]) if not volumes.empty else 0.0
    volume_change = 0.0
    if len(volumes) > 5:
        previous_volume = float(volumes.iloc[-6])
        raw_volume_change = ((latest_volume - previous_volume) / previous_volume * 100
                             if previous_volume else 0.0)
        volume_change = _signed_log1p(raw_volume_change)

    foreign_net = 0.0
    if 'Foreign_Buy_Sum' in df.columns and 'Foreign_Sell_Sum' in df.columns:
        foreign_net = (_get_numeric_value(df, row_idx, ['Foreign_Buy_Sum'], 0.0)
                       - _get_numeric_value(df, row_idx, ['Foreign_Sell_Sum'], 0.0))
    margin_balance = _get_numeric_value(df, row_idx, ['MarginPurchaseBalance'], 0.0)
    short_balance = _get_numeric_value(df, row_idx, ['ShortSaleBalance'], 0.0)

    foreign_to_volume = _signed_log1p(foreign_net / max(latest_volume, 1))
    margin_short_ratio = margin_balance / (abs(margin_balance) + abs(short_balance) + 1.0)
    tej_large = _get_numeric_value(df, row_idx, ['TEJ_大戶持股比'])
    tej_retail = _get_numeric_value(df, row_idx, ['TEJ_散戶持股比'])

    return {
        '1週報酬率': round(return_pct(1), 6),
        '4週報酬率': round(return_pct(4), 6),
        '8週報酬率': round(return_pct(8), 6),
        '5日均線偏離率': round(moving_average_deviation(5), 6),
        '20日均線偏離率': round(moving_average_deviation(20), 6),
        '成交量增幅': round(volume_change, 6),
        '外資淨買賣超/成交量': round(foreign_to_volume, 6),
        '融資融券比率': round(margin_short_ratio, 6),
        '大戶持股比(TEJ)': round(tej_large, 6),
        '散戶持股比(TEJ)': round(tej_retail, 6),
        '大戶散戶差': round(tej_large - tej_retail, 6)
    }


def _merge_tej_data(stock_id, df):
    """將 TEJ 分組欄位合併到預測資料，供條件與 ML 特徵共用。"""
    df_tej = backtest.load_local_tej_data(stock_id)
    if df_tej is None or df_tej.empty:
        return None

    df = df.copy()
    df['資料日期'] = pd.to_datetime(df['資料日期'])
    df = df.sort_values('資料日期')
    df = df.drop(
        columns=[
            '>400張百分比',
            '>1000張百分比',
            '總股東人數',
            '總張數',
            'TEJ_大戶持股比',
            'TEJ_散戶持股比'
        ],
        errors='ignore'
    )
    df = pd.merge_asof(
        df,
        df_tej,
        left_on='資料日期',
        right_on='年月日',
        direction='backward',
        tolerance=pd.Timedelta(days=3)
    )
    for column in [
        '>400張百分比',
        '>1000張百分比',
        '總股東人數',
        'TEJ_大戶持股比',
        'TEJ_散戶持股比'
    ]:
        df[column] = pd.to_numeric(df[column], errors='coerce').ffill().fillna(0)
    return df.reset_index(drop=True)


def scan_latest_and_history(df, params): 
    df['資料日期'] = pd.to_datetime(df['資料日期'])
    df = df.sort_values('資料日期').reset_index(drop=True)
    stock_id = df['股票代號'].iloc[0]
    df = _merge_tej_data(stock_id, df)
    if df is None or df.empty:
        return None, None
    i_latest = len(df) - 1
    
    # 1. 嚴格初篩：先用 GA 參數檢查最新一週
    is_triggered, corr, retail_corr, avg_corr, actual_win = check_conditions(df, i_latest, **params)
    if not is_triggered:
        return None, None

    # ==========================================
    # 🌟 2. 初篩通過！動態抓取 FinMind 最新特徵並合併
    # ==========================================
    start_str = df['資料日期'].min().strftime('%Y-%m-%d')
    end_str = df['資料日期'].max().strftime('%Y-%m-%d')
    
    df_finmind = crawler_finmind.fetch_weekly_chip_data(stock_id, start_str, end_str)
    
    if df_finmind is not None and not df_finmind.empty:
        df_finmind['date'] = pd.to_datetime(df_finmind['date'])
        df_finmind = df_finmind.sort_values('date')
        df = df.sort_values('資料日期')
        df = pd.merge_asof(df, df_finmind, left_on='資料日期', right_on='date', direction='backward', tolerance=pd.Timedelta(days=3))
        df = df.reset_index(drop=True)
        i_latest = len(df) - 1

    price_df = crawler.download_stock_price_history(stock_id)
    if price_df is None or price_df.empty:
        return None, None

    feature_history = [_build_feature_row(stock_id, df, row_idx, price_df)
                       for row_idx in range(len(df))]

    # 3. 呼叫大腦進行最終預測
    ml_pass, ml_prob = filter_with_ml(feature_history[-SEQUENCE_LENGTH:])
    # ==========================================

    past_trades = []
    for i_hist in range(4, len(df)-1):
        hist_trigger, _, _, _, _ = check_conditions(df, i_hist, **params)
        
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
                if consecutive_drops >= 2: break

            cum_ret = ((prev_price - buy_price) / buy_price) * 100
            past_trades.append({
                '進場日': df.at[i_hist, '資料日期'].strftime('%Y-%m-%d') if isinstance(df.at[i_hist, '資料日期'], pd.Timestamp) else df.at[i_hist, '資料日期'],
                '持股週數': exit_k,
                '累積報酬': cum_ret,
                '歷程': ", ".join(weekly_records),
                '開局秒出場': consecutive_drops >= 2 and exit_k == 2 
            })

    # 預設建議
    suggestion = '🎯 建議進場'
    hist_summary = "無歷史前例"
    hist_details_str = "無"

    # 第一關：ML 大腦審查
    if not ml_pass:
        suggestion = '❌ ML大腦退件'

    # 第二關：歷史前例審查
    if past_trades:
        avg_ret = np.mean([t['累積報酬'] for t in past_trades])
        bad_starts = sum(1 for t in past_trades if t['開局秒出場'] and t['累積報酬'] < 0)
        
        hist_summary = f"發生 {len(past_trades)} 次, 平均 {avg_ret:+.2f}%"
        
        # 如果歷史回測不佳，覆蓋原有建議
        if avg_ret < 0 or (bad_starts / len(past_trades) >= 0.5):
            suggestion = '❌ 歷史回測不佳'
            
        details_list = []
        for pt in past_trades:
            status = "⚠️ 連跌兩週停損" if pt['開局秒出場'] else "✅波段結算"
            details_list.append(f"[{pt['進場日']}] 總計 {pt['累積報酬']:>+5.1f}% | 軌跡: {pt['歷程']} ({status})")
        hist_details_str = "\n".join(details_list)

    result_dict = {
        '代號': stock_id,
        '發布日': df.at[i_latest, '資料日期'].strftime('%Y-%m-%d') if isinstance(df.at[i_latest, '資料日期'], pd.Timestamp) else df.at[i_latest, '資料日期'],
        f'大戶({actual_win}週)': round(float(corr), 3),
        f'散戶({actual_win}週)': round(float(retail_corr), 3),
        f'均張({actual_win}週)': round(float(avg_corr), 3),
        '收盤價': df.at[i_latest, '收盤價'],
        'ML預測': f"{'✅' if ml_pass else '❌'} ({ml_prob:+.1f}%)",
        '相似型態勝率': hist_summary,
        '歷史走勢明細': hist_details_str, 
        '建議': suggestion
    }

    return result_dict, past_trades

# ==========================================
# 預測總司令 (支援動態傳入參數)
# ==========================================
def get_next_week_recommendations(target_list, params=None):
    if params is None: params = {}
    recommendations = []
    total = len(target_list)

    for i, sid in enumerate(target_list):
        print(f"🔎 掃描預測 [{i + 1}/{total}] {sid}...", end="\r", flush=True) 
        
        df = crawler.get_individual_stock_data(sid)
        if df is None or df.empty:
            continue

        res, past_trades = scan_latest_and_history(df, params)
        
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
