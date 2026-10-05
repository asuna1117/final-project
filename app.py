import streamlit as st
import pandas as pd
import altair as alt

# 🌟 完美引入組員寫好的三大模組 (使用你最新的 _2 版本)
import crawler as crawler
import backtest as backtest
import predict_cbc as predict_cbc
import train_ml  # 🌟 新增：引入 LSTM 訓練模組

# 🌟 定義用來收集平衡樣本的寬鬆參數
TRAIN_PARAMS = {
    "continuous_weeks": 2,
    "min_growth": -0.05,
    "pop_decline_threshold": -0.05,
    "last_week_threshold": -0.05,
    "large_corr_thresh": 0.0,
    "retail_corr_thresh": 0.0,
    "avg_corr_thresh": 0.0,
    "skip_cond_e": True
}

# ==========================================
# 核心管線 1：抓取資料 (透過 crawler 模組)
# ==========================================
def fetch_data_pipeline(stock_list):
    raw_data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(stock_list)
    
    for i, sid in enumerate(stock_list):
        status_text.text(f"⏳ 正在下載籌碼資料: {sid} ({i+1}/{total})")
        df = crawler.get_individual_stock_data(sid)
        if df is not None and not df.empty:
            raw_data[sid] = df
        progress_bar.progress((i + 1) / total)
        
    status_text.text("✨ 資料下載完畢！請前往左側設定參數並點擊「確認篩選」。")
    progress_bar.empty()
    return raw_data

# ==========================================
# 核心管線 2：整合 TEJ 資料
# ==========================================
def enrich_with_tej_data(raw_data):
    enriched_data = {}
    for sid, df in raw_data.items():
        tej_data = backtest.load_local_tej_data(sid)
        if tej_data is not None and not tej_data.empty:
            # 確保日期欄位的型別一致
            df['資料日期'] = pd.to_datetime(df['資料日期'], errors='coerce')
            tej_data['年月日'] = pd.to_datetime(tej_data['年月日'], errors='coerce')

            # 🌟 修正：在合併前先刪除原始 df 中與 TEJ 重複的籌碼欄位，避免產生 _x, _y 後綴
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

            # 合併 TEJ 資料
            df = pd.merge_asof(
                df.sort_values('資料日期'),
                tej_data.sort_values('年月日'),
                left_on='資料日期',
                right_on='年月日',
                direction='backward',
                tolerance=pd.Timedelta(days=3)
            )

            # 將合併後可能產生的空值補 0
            for column in [
                '>400張百分比',
                '>1000張百分比',
                '總股東人數',
                'TEJ_大戶持股比',
                'TEJ_散戶持股比'
            ]:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce').ffill().fillna(0)

        enriched_data[sid] = df
    return enriched_data

# ==========================================
# 核心管線 3：歷史回測 (透過 backtest 模組)
# ==========================================
@st.cache_data(show_spinner=False)
def analyze_data_pipeline(_raw_data_dict, strategy_params):
    all_trades = []
    for sid, df in _raw_data_dict.items():
        # 呼叫組員最新的 backtest 模組
        trades = backtest.backtest_squeeze_strategy(df, **strategy_params)
        if trades:
            all_trades.extend(trades)
            
    if all_trades:
        return pd.DataFrame(all_trades).sort_values(['進場日期(籌碼公告)', '代號'], ascending=[False, True])
    else:
        return pd.DataFrame()

# ==========================================
# 核心管線 4：下週預測 (透過 predict_cbc 模組)
# ==========================================
@st.cache_data(show_spinner=False)
def predict_data_pipeline(_raw_data_dict, strategy_params):
    recommendations = []
    for sid, df in _raw_data_dict.items():
        # 🌟 修正：移除 ** 避免 unexpected keyword argument 錯誤
        res, _ = predict_cbc.scan_latest_and_history(df, strategy_params)
        if res is not None:
            recommendations.append(res)
            
    if recommendations:
        return pd.DataFrame(recommendations).sort_values('代號')
    else:
        return pd.DataFrame()

# ==========================================
# 🌟 核心管線 5：收集特徵並訓練 ML 模型
# ==========================================
def train_model_pipeline(raw_data_dict):
    st.info("⏳ 步驟 1/3：正在使用寬鬆參數掃描全市場，收集正負樣本 (背景執行中)...")
    target_list = list(raw_data_dict.keys())
    
    # 呼叫 backtest 進行資料收集 (is_training=True)
    backtest.run_all_analysis(target_list, params=TRAIN_PARAMS, is_training=True)
    
    st.info("💾 步驟 2/3：正在匯出機器學習特徵至 CSV...")
    backtest.export_ml_data()
    
    st.info("🧠 步驟 3/3：正在啟動 LSTM 類神經網路訓練 (這可能需要幾分鐘，請耐心等候)...")
    try:
        # 🌟 修改：接收 train_ml 傳出來的成績單字典
        metrics = train_ml.main()  
        st.success("🎉 模型訓練完畢！最新的 lstm_trading_model.keras 大腦已上線！您可以重新執行預測了。")
        
        # 🌟 新增：在網頁上動態渲染出漂亮的報表
        if metrics:
            st.markdown("### 📊 模型分類與機率驗證報告")
            
            # 使用四個直排版塊顯示核心數據
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("測試集實際獲利比例", f"{metrics['actual_win_rate'] * 100:.1f}%")
            col2.metric("無腦瞎猜準確率", f"{metrics['baseline_acc'] * 100:.1f}%")
            
            # 判斷模型有沒有打敗瞎猜，給予不同的顏色提示 (Streamlit 預設會顯示為綠色或紅色指標)
            acc_diff = (metrics['accuracy'] - metrics['baseline_acc']) * 100
            col3.metric("🤖 模型預測準確率", f"{metrics['accuracy'] * 100:.1f}%", f"{acc_diff:+.1f}% 相較瞎猜")
            col4.metric("🎯 R平方分數 (R2 Score)", f"{metrics['r2']:.4f}")

            st.markdown("#### 【詳細分類指標】")
            # 使用 code 區塊保持純文字報表的完美等寬排版
            st.code(metrics['report'], language="text")

    except Exception as e:
        st.error(f"❌ 訓練過程中發生錯誤: {e}")

# ==========================================
# 網頁前端介面區 (Web UI)
# ==========================================
st.set_page_config(page_title="大戶籌碼追蹤系統", layout="wide")
st.title("📈 大戶籌碼追蹤與實戰回測系統")

# --- 左側欄 ---
st.sidebar.header("📥 階段一：啟動爬蟲抓取資料")
st.sidebar.info("先將最新的籌碼資料抓取至系統記憶體中。")

use_tej_enrichment = st.sidebar.checkbox("啟用 TEJ 資料擴充", value=False)

fetch_count = st.sidebar.slider("抓取股票數量", min_value=10, max_value=2000, step=10, value=50, key="slider_val")
fetch_button = st.sidebar.button("1️⃣ 啟動爬蟲更新資料", type="secondary", use_container_width=True)

if fetch_button:
    stock_list = crawler.get_stock_ids(crawler.list_url)[:fetch_count]
    st.session_state['raw_data'] = fetch_data_pipeline(stock_list)

    if use_tej_enrichment:
        st.session_state['raw_data'] = enrich_with_tej_data(st.session_state['raw_data'])

    st.sidebar.success("✅ 爬蟲執行完畢，資料已就緒！")

# --- 第二階段參數設定 ---
st.sidebar.header("⚙️ 階段二：客製化參數設定")
c_weeks = st.sidebar.number_input("連續買超週數", min_value=2, max_value=12, value=4)
min_g = st.sidebar.number_input("大戶每週最低增長率 (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.1)
pop_d = st.sidebar.number_input("散戶減少最低門檻 (%)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
last_week_threshold = st.sidebar.number_input("最後一週增長門檻 (%)", min_value=-5.0, max_value=5.0, value=0.1, step=0.1)
large_corr_thresh = st.sidebar.slider("大戶相關係數門檻", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
retail_corr_thresh = st.sidebar.slider("散戶相關係數門檻", min_value=-1.0, max_value=0.0, value=-0.6, step=0.05)
avg_corr_thresh = st.sidebar.slider("平均張數相關係數門檻", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

strategy_params = {
    'continuous_weeks': c_weeks,
    'min_growth': min_g,
    'pop_decline_threshold': pop_d,
    'last_week_threshold': last_week_threshold,
    'large_corr_thresh': large_corr_thresh,
    'retail_corr_thresh': retail_corr_thresh,
    'avg_corr_thresh': avg_corr_thresh,
}

# --- 第三階段確認篩選 ---
st.sidebar.header("🔍 階段三：執行分析")
filter_button = st.sidebar.button("2️⃣ 確認篩選 (產出報表)", type="primary", use_container_width=True)

if filter_button:
    if 'raw_data' not in st.session_state:
        st.sidebar.warning("⚠️ 請先完成階段一：啟動爬蟲抓取資料！")
    else:
        # 執行回測與預測
        st.session_state['filtered_trades'] = analyze_data_pipeline(st.session_state['raw_data'], strategy_params)
        st.session_state['predictions'] = predict_data_pipeline(st.session_state['raw_data'], strategy_params)

# --- 第四階段模型訓練 ---
st.sidebar.markdown("---")
st.sidebar.header("🧠 階段四：AI 大腦訓練中心")
st.sidebar.warning("⚠️ 訓練模型非常耗時，請確保已於階段一抓取足夠的股票資料再執行。")
train_button = st.sidebar.button("3️⃣ 重新訓練 LSTM 預測模型", type="primary", use_container_width=True)

if train_button:
    if 'raw_data' not in st.session_state or not st.session_state['raw_data']:
        st.sidebar.error("⚠️ 請先完成階段一：啟動爬蟲抓取資料！")
    else:
        with st.spinner("AI 訓練所已啟動，請耐心等候，請勿關閉網頁..."):
            train_model_pipeline(st.session_state['raw_data'])

# --- 主畫面報表顯示區 ---
if 'filtered_trades' in st.session_state:
    trades_df = st.session_state['filtered_trades']
    pred_df = st.session_state.get('predictions', pd.DataFrame())
    
    st.success(f"✅ 篩選成功！ (基於已抓取的 {len(st.session_state['raw_data'])} 檔標的)")
    
    # 切換「回測」與「預測」頁籤
    main_tab1, main_tab2 = st.tabs(["🕰️ 歷史勝率回測總表", "🔮 下週飆股預測 (實戰清單)"])
    
    with main_tab1:
        if trades_df.empty:
            st.error("⚠️ 依據您目前設定的參數，查無符合條件的數據。請嘗試放寬條件。")
        else:
            completed_trades = trades_df.dropna(subset=['週報酬%']).copy()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric(label="✅ 歷史觸發總次數", value=f"{len(trades_df)} 次")
            with col2:
                win_rate = (completed_trades['週報酬%'] > 0).mean() * 100 if not completed_trades.empty else 0
                st.metric(label="🏆 策略結算勝率", value=f"{win_rate:.1f} %" if win_rate else "尚無結算")
            with col3:
                avg_return = completed_trades['週報酬%'].mean() if not completed_trades.empty else 0
                st.metric(label="💰 策略平均週報酬", value=f"{avg_return:.2f} %" if avg_return else "尚無結算")
            with col4:
                max_return = completed_trades['週報酬%'].max() if not completed_trades.empty else 0
                st.metric(label="🔝 最高單次報酬", value=f"{max_return:.2f} %" if max_return else "尚無結算")
            
            st.markdown("---")
            display_df = trades_df.copy()
            display_df['下週收盤出場價'] = display_df['下週收盤出場價'].fillna('等待開獎')
            display_df['週報酬%'] = display_df['週報酬%'].fillna('等待開獎')
            st.dataframe(display_df, width="stretch", hide_index=True)

    with main_tab2:
        if pred_df.empty:
            st.info("⚠️ 掃描完畢，目前的清單中【沒有】剛好在最新一週觸發進場訊號的股票。")
        else:
            clean_pred_df = pred_df.drop(columns=['歷史走勢明細'])
            st.dataframe(clean_pred_df, width="stretch", hide_index=True)
            
            st.markdown("""
            **💡 判讀教學：**
            * 若「綜合建議」顯示為『❌ 回測不佳』，代表此股票過去發生相同訊號時，多半會立刻遭遇連續兩週下跌的停損出場，或歷史平均報酬為負，請避開陷阱。
            * 若顯示為『🎯 建議進場』，代表該股歷史上出現此形態時具備正向期望值。
            """)
