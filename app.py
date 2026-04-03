import streamlit as st
import pandas as pd
import altair as alt
import test  # 呼叫組員的真實程式

# ==========================================
# 核心管線：讓網頁自己呼叫組員的工具來跑迴圈
# ==========================================
@st.cache_data(show_spinner=False)
def run_backend_pipeline(stock_list):
    all_dfs = []
    all_trades = []
    
    # 建立動態進度條與狀態文字
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(stock_list)
    for i, sid in enumerate(stock_list):
        status_text.text(f"⏳ 正在抓取並分析: {sid} ({i+1}/{total})")
        df = test.get_individual_stock_data(sid)
        
        if df is not None and not df.empty:
            all_dfs.append(df)
            trades = test.backtest_hard_rule(df)
            if trades:
                all_trades.extend(trades)
        
        progress_bar.progress((i + 1) / total)
        
    status_text.text("✨ 資料抓取完成，正在計算相關係數...")
    
    trades_df = pd.DataFrame(all_trades)
    final_df = pd.DataFrame()
    
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        final_df = test.analyze_correlation_logic(master_df)
        
    progress_bar.empty()
    status_text.empty()
    
    return trades_df, final_df

# ==========================================
# 網頁前端介面區 (Web UI)
# ==========================================
st.set_page_config(page_title="大戶籌碼追蹤系統", layout="wide")
st.title("📈 大戶籌碼追蹤與回測系統")
st.write("結合「大戶連四買」硬規則與「相關係數」演算法，精準捕捉主力連續吃貨的黃金訊號。")

# --- 左側欄 ---
st.sidebar.header("⚙️ 系統操作")
st.sidebar.info(
    "💡 **策略核心邏輯**\n\n"
    "1. **大戶硬規則**：尋找大戶連續四週買進的標的。\n"
    "2. **相關係數驗證**：精算大戶動向與股價的連動性，確保主力具備控盤能力。"
)
st.sidebar.markdown("---")

# ==========================================
# 替換原本的滑桿，改成「輸入框 + 滑桿」雙向連動模組
# ==========================================
st.sidebar.markdown("**📊 設定抓取股票數量**")

# 1. 建立連動用的回呼函式 (Callback)
def update_slider():
    st.session_state.slider_val = st.session_state.num_val
def update_num():
    st.session_state.num_val = st.session_state.slider_val

# 2. 初始化預設值為 200
if 'slider_val' not in st.session_state:
    st.session_state.slider_val = 200
if 'num_val' not in st.session_state:
    st.session_state.num_val = 200

# 3. 畫出輸入框與滑桿，並綁定連動函式
st.sidebar.number_input("手動輸入數字：", min_value=10, max_value=2000, step=10, key='num_val', on_change=update_slider)
st.sidebar.slider("或用滑鼠微調：", min_value=10, max_value=2000, step=10, key='slider_val', on_change=update_num, label_visibility="collapsed")

# 4. 把最終決定好的數字交給 fetch_count 變數
fetch_count = st.session_state.num_val
# ==========================================

# 下面接原本的執行按鈕
run_button = st.sidebar.button("開始分析", type="primary", use_container_width=True)

# --- 主畫面 ---
if run_button:
    stock_list = test.get_stock_ids(test.list_url)[:fetch_count] 
    trades_df, final_df = run_backend_pipeline(stock_list)
    
    if trades_df.empty and final_df.empty:
        st.warning("⚠️ 近期查無符合條件的數據，請嘗試擴大抓取數量。")
        st.stop()
        
    st.success("分析完成！以下為最新的籌碼數據與歷史回測結果。")
    
    # 1. KPI 指標卡 (4 個重要數據)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="✅ 歷史觸發次數", value=f"{len(trades_df)} 次")
    with col2:
        if not trades_df.empty:
            win_rate = (len(trades_df[trades_df['週報酬%'] > 0]) / len(trades_df)) * 100
            st.metric(label="🏆 策略勝率", value=f"{win_rate:.1f} %")
        else:
            st.metric(label="🏆 策略勝率", value="0 %")
    with col3:
        if not trades_df.empty:
            avg_return = trades_df["週報酬%"].mean()
            st.metric(label="💰 平均週報酬", value=f"{avg_return:.2f} %")
        else:
            st.metric(label="💰 平均週報酬", value="0 %")
    with col4:
        if not trades_df.empty:
            max_return = trades_df["週報酬%"].max()
            st.metric(label="🔝 最高單次報酬", value=f"{max_return:.2f} %")
        else:
            st.metric(label="🔝 最高單次報酬", value="0 %")
    
    st.markdown("---")
    
    # 2. 呈現組員的兩份表格 (改回最直覺的左右並排)
    st.subheader("📋 籌碼數據與回測明細")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("##### 🎯 最新籌碼連動推薦清單")
        if not final_df.empty:
            high_corr_df = final_df[final_df['相關係數'] > 0.5].sort_values('相關係數', ascending=False).copy()
            high_corr_df['週增減'] = high_corr_df['週增減'].apply(test.get_change_arrow)
            st.dataframe(high_corr_df, width="stretch", hide_index=True)
            st.info(f"💡 共有 {len(high_corr_df)} 檔股票相關係數高於 0.5，值得關注其籌碼動向。")
        else:
            st.info("目前無相關係數大於 0.5 的標的")
            
    with col_right:
        st.markdown("##### 🕰️ 歷史回測明細 (大戶連四買)")
        if not trades_df.empty:
            # 這裡不鎖定筆數，直接讓 Streamlit 原生的滾動條發揮作用
            st.dataframe(trades_df, width="stretch", hide_index=True)
        else:
            st.info("近期沒有符合條件的回測紀錄。")
            
    # 3. 視覺化圖表 (保留好用的滑鼠縮放功能 interactive)
    st.markdown("<br>", unsafe_allow_html=True) 
    st.subheader("📊 數據視覺化分析")
    
    tab1, tab2 = st.tabs(["🎯 相關係數分佈 (散佈圖)", "📈 回測獲利分佈 (長條圖)"])
    
    with tab1:
        if not final_df.empty:
            scatter_chart = alt.Chart(final_df).mark_circle(
                color="#00ff00", size=120, opacity=0.7, stroke="white", strokeWidth=1
            ).encode(
                x=alt.X('代號:N', axis=alt.Axis(labelAngle=0, title='股票代號')),
                y=alt.Y('相關係數:Q', axis=alt.Axis(title='相關係數', titleAngle=0, titleY=-15, titleX=-20)),
                tooltip=['代號', '相關係數', '狀態', '大戶%']
            ).properties(height=350).interactive() 
            st.altair_chart(scatter_chart, width="stretch")
            
    with tab2:
        if not trades_df.empty:
            bar_chart = alt.Chart(trades_df).mark_bar().encode(
                x=alt.X('股票代號:N', axis=alt.Axis(labelAngle=0, title='股票代號')),
                y=alt.Y('週報酬%:Q', axis=alt.Axis(title='週報酬 (%)', titleAngle=0, titleY=-15, titleX=-20)),
                color=alt.condition(alt.datum['週報酬%'] > 0, alt.value("#ffaa00"), alt.value("#0077ff")),
                tooltip=['股票代號', '進場日期', '週報酬%', '4週變化']
            ).properties(height=350).interactive() # 長條圖順便也加上縮放功能
            st.altair_chart(bar_chart, width="stretch")