import crawler
import backtest
import predict_cbc
from tabulate import tabulate
import re
import unicodedata

# ==========================================
# 輔助排版函式：精準計算終端機字元寬度
# ==========================================
def get_display_width(text):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)

# ==========================================
# 系統主程式 (控制中心)
# ==========================================
def main():
    print("========================================")
    print(" 🌟 歡迎使用【籌碼面量化分析系統】🌟")
    print("========================================")
    
    # 1. 取得股票清單 (呼叫 crawler 模組)
    stock_list = crawler.get_stock_ids(crawler.list_url)
    total_available = len(stock_list)

    if total_available == 0:
        print("❌ 沒抓到股票清單，程式結束。")
        return

    print(f"✅ 成功取得 {total_available} 檔股票清單。")
    print("--------------------------------")
    print("1. 前 10 個 (快速測試)")
    print("2. 前 50 個 (建議)")
    print(f"3. 全部 ({total_available} 個)")
    print("4. 自訂範圍")
    print("--------------------------------")

    choice = input("👉 請輸入掃描範圍 (1/2/3/4): ").strip()
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

    # 2. 選擇你要呼叫的模組
    print("\n--------------------------------")
    print("請選擇要執行的功能：")
    print("1. 📊 執行【歷史勝率回測】 (呼叫 backtest 模組)")
    print("2. 🔮 執行【下週飆股預測】 (呼叫 predict_cbc 模組)")
    print("--------------------------------")
    mode = input("👉 請選擇 (1/2): ").strip()

    print(f"\n🚀 準備處理 {len(target_list)} 檔股票...\n")

    # ==========================================
    # 模組一：歷史回測
    # ==========================================
    if mode == '1':
        trades_df = backtest.run_all_analysis(target_list)
        if not trades_df.empty:
            display_df = trades_df.fillna({'下週收盤出場價': '等待開獎', '週報酬%': '等待開獎'})
            print("\n" + "=" * 90)
            print("📈 籌碼策略回測結果 (3年大數據版)")
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

    # ==========================================
    # 模組二：未來預測
    # ==========================================
    elif mode == '2':
        recommend_df = predict_cbc.get_next_week_recommendations(target_list)
        if not recommend_df.empty:
            print("\n" + "=" * 110)
            print("🎯 掃描完畢！發現以下【下週實戰推薦清單】：")
            print("=" * 110)
            
            display_df = recommend_df.drop(columns=['歷史走勢明細'])
            print(tabulate(display_df, headers='keys', tablefmt='simple', showindex=False))
            
            print("\n" + "=" * 110)
            print("📜 【歷史相似走勢 - 深度明細解析】")
            print("=" * 110)
            
            for idx, row in recommend_df.iterrows():
                print(f"🔸 股票代號: 【 {row['代號']} 】 | 綜合建議: {row['建議']}")
                if row['歷史走勢明細'] == "無":
                    print("   └─ 歷史上尚無完全相同之訊號可供比對。")
                else:
                    trades = row['歷史走勢明細'].split('\n')
                    for trade_str in trades:
                        match = re.search(r'(.*軌跡: )(.*) (\(.*)', trade_str)
                        if match:
                            prefix = "   └─ " + match.group(1)
                            trajectory_str = match.group(2)
                            status_str = " " + match.group(3)
                            
                            weeks = trajectory_str.split(', ')
                            indent_width = get_display_width(prefix)
                            indent_spaces = " " * indent_width
                            
                            chunk_size = 8
                            lines = []
                            for i in range(0, len(weeks), chunk_size):
                                lines.append(", ".join(weeks[i:i+chunk_size]))
                            
                            formatted_trajectory = f",\n{indent_spaces}".join(lines)
                            print(f"{prefix}{formatted_trajectory}{status_str}")
                        else:
                            print(f"   └─ {trade_str}")
                print("-" * 110)
        else:
            print("\n⚠️ 掃描完畢，沒有剛好在最新一週觸發進場訊號的股票。")
    else:
        print("\n❌ 輸入錯誤，請重新執行程式並輸入 1 或 2。")

if __name__ == "__main__":
    main()