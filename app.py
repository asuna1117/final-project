import streamlit as st
import pandas as pd
import altair as alt

# 🌟 完美引入組員寫好的三大模組
import crawler
import backtest
import predict_cbc

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
# 核心管線 2：歷史回測 (透過 backtest 模組)
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
# 核心管線 3：下週預測 (透過 predict_cbc 模組)
# ==========================================
@st.cache_data(show_spinner=False)
def predict_data_pipeline(_raw_data_dict):
    recommendations = []
    for sid, df in _raw_data_dict.items():
        # 呼叫組員最新的 predict_cbc 模組
        res, _ = predict_cbc.scan_latest_and_history(df)
        if res is not None:
            recommendations.append(res)
            
    if recommendations:
        return pd.DataFrame(recommendations).sort_values('代號')
    else:
        return pd.DataFrame()


# ==========================================
# 網頁前端介面區 (Web UI)
# ==========================================
st.set_page_config(page_title="大戶籌碼追蹤系統", layout="wide")
st.title("📈 大戶籌碼追蹤與實戰回測系統")

# --- 左側欄 ---
st.sidebar.header("📥 階段一：啟動爬蟲抓取資料")
st.sidebar.info("先將最新的籌碼資料抓取至系統記憶體中。")

def update_slider(): st.session_state.slider_val = st.session_state.num_val
def update_num(): st.session_state.num_val = st.session_state.slider_val
if 'slider_val' not in st.session_state: st.session_state.slider_val = 50
if 'num_val' not in st.session_state: st.session_state.num_val = 50

st.sidebar.number_input("手動輸入抓取數量：", min_value=10, max_value=2000, step=10, key='num_val', on_change=update_slider)
st.sidebar.slider("或用滑鼠微調：", min_value=10, max_value=2000, step=10, key='slider_val', on_change=update_num, label_visibility="collapsed")
fetch_count = st.session_state.num_val

fetch_button = st.sidebar.button("1️⃣ 啟動爬蟲更新資料", type="secondary", use_container_width=True)

if fetch_button:
    stock_list = crawler.get_stock_ids(crawler.list_url)[:fetch_count] 
    st.session_state['raw_data'] = fetch_data_pipeline(stock_list)
    st.sidebar.success("✅ 爬蟲執行完畢，資料已就緒！")

st.sidebar.markdown("---")

# --- 第二階段參數設定 ---
st.sidebar.header("⚙️ 階段二：客製化參數設定")

c_weeks = st.sidebar.number_input("連續買超週數", min_value=2, max_value=12, value=4)
min_g = st.sidebar.number_input("大戶每週最低增長率 (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.1)
pop_d = st.sidebar.number_input("散戶減少最低門檻 (%)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)

st.sidebar.markdown("##### 📐 相關係數門檻設定")
# 配合組員升級為 3年(156週)，調整拉桿上限與預設值
c_win = st.sidebar.slider("相關係數計算區間 (週)", min_value=4, max_value=200, value=156)
l_corr = st.sidebar.slider("大戶相關係數門檻", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
r_corr = st.sidebar.slider("散戶相關係數門檻", min_value=-1.0, max_value=0.0, value=-0.6, step=0.05)
a_corr = st.sidebar.slider("平均張數相關係數門檻", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

strategy_params = {
    'continuous_weeks': c_weeks,
    'min_growth': min_g,
    'pop_decline_threshold': pop_d,
    'corr_window': c_win,
    'large_corr_thresh': l_corr,
    'retail_corr_thresh': r_corr,
    'avg_corr_thresh': a_corr
}

st.sidebar.markdown("---")

# --- 第三階段確認篩選 ---
st.sidebar.header("🔍 階段三：執行分析")
filter_button = st.sidebar.button("2️⃣ 確認篩選 (產出報表)", type="primary", use_container_width=True)

# ==========================================
# 主畫面顯示邏輯
# ==========================================
if filter_button:
    if 'raw_data' not in st.session_state:
        st.warning("⚠️ 請先在左側執行「階段一：啟動爬蟲抓取資料」，系統才有資料可以篩選喔！")
    else:
        # 同時運算回測與預測
        st.session_state['filtered_trades'] = analyze_data_pipeline(st.session_state['raw_data'], strategy_params)
        st.session_state['predictions'] = predict_data_pipeline(st.session_state['raw_data'])

if 'filtered_trades' in st.session_state:
    trades_df = st.session_state['filtered_trades']
    pred_df = st.session_state.get('predictions', pd.DataFrame())
    
    st.success(f"✅ 篩選成功！ (基於已抓取的 {len(st.session_state['raw_data'])} 檔標的)")
    
    # 🌟 新增外層 Tabs，切換「回測」與「預測」
    main_tab1, main_tab2 = st.tabs(["🕰️ 歷史勝率回測總表", "🔮 下週飆股預測 (實戰清單)"])
    
    with main_tab1:
        if trades_df.empty:
            st.error("⚠️ 依據您目前設定的參數，查無符合條件的數據。請放寬條件。")
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

            if not completed_trades.empty:
                st.markdown("<br>", unsafe_allow_html=True) 
                st.subheader("📊 回測績效視覺化")
                chart_tab1, chart_tab2 = st.tabs(["📈 策略獲利時間軸", "📊 個股回測績效"])
                
                with chart_tab1:
                    scatter_chart = alt.Chart(completed_trades).mark_circle(
                        size=120, opacity=0.7, stroke="white", strokeWidth=1
                    ).encode(
                        x=alt.X('進場日期(籌碼公告):N', axis=alt.Axis(labelAngle=-45, title='進場日期')),
                        y=alt.Y('週報酬%:Q', axis=alt.Axis(title='週報酬 (%)')),
                        color=alt.condition(alt.datum['週報酬%'] > 0, alt.value("#00ff00"), alt.value("#ff3333")),
                        tooltip=['代號', '進場日期(籌碼公告)', '週一開盤進場價', '下週收盤出場價', '週報酬%']
                    ).properties(height=350).interactive() 
                    st.altair_chart(scatter_chart, width="stretch")
                    
                with chart_tab2:
                    bar_chart = alt.Chart(completed_trades).mark_bar().encode(
                        x=alt.X('代號:N', axis=alt.Axis(labelAngle=0, title='股票代號')),
                        y=alt.Y('週報酬%:Q', axis=alt.Axis(title='週報酬 (%)')),
                        color=alt.condition(alt.datum['週報酬%'] > 0, alt.value("#ffaa00"), alt.value("#0077ff")),
                        tooltip=['代號', '進場日期(籌碼公告)', '週一開盤進場價', '下週收盤出場價', '週報酬%']
                    ).properties(height=350).interactive() 
                    st.altair_chart(bar_chart, width="stretch")

    # 🌟 呼應組員成果的新頁籤
    with main_tab2:
        st.subheader("🎯 剛好於【最新一週】觸發訊號的推薦名單")
        if pred_df.empty:
            st.info("⚠️ 掃描完畢，目前的清單中【沒有】剛好在最新一週觸發進場訊號的股票。")
        else:
            # 移除文字明細，讓網頁只顯示乾淨的推薦大表
            clean_pred_df = pred_df.drop(columns=['歷史走勢明細'])
            st.dataframe(clean_pred_df, width="stretch", hide_index=True)
            
            st.markdown("""
            **💡 判讀教學：**
            * 若「綜合建議」顯示為『❌ 回測不佳』，代表此股票過去發生相同訊號時，多半會立刻遭遇連續兩週下跌的停損出場，或歷史平均報酬為負，請避開陷阱。
            * 若顯示為『🎯 建議進場』，代表該股歷史上出現此形態時具備正向期望值。
            """)
