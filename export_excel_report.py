import os
import pandas as pd
from datetime import datetime
import crawler
import backtest
from test import STRICT_PARAMS

def main():
    print("========================================")
    print(" 📊 回測產出報告匯出工具 (Excel 專用)")
    print("========================================")
    
    stock_list = crawler.get_stock_ids(crawler.list_url)
    total_available = len(stock_list)

    if total_available == 0:
        print("❌ 沒抓到股票清單，程式結束。")
        return

    print(f"✅ 成功取得 {total_available} 檔股票清單。")
    print("--------------------------------")
    
    # 這裡的邏輯讓你能夠自由輸入特定的股票起始位置與數量
    try:
        start_index = int(input(f"👉 請輸入要從第幾個開始測試? (預設 0, 最大 {total_available-1}): ").strip() or 0)
        count = int(input("👉 要往後測試幾檔股票? (預設 10): ").strip() or 10)
    except ValueError:
        start_index, count = 0, 10
        
    end_index = min(start_index + count, total_available)
    target_list = stock_list[start_index:end_index]

    print(f"\n🚀 準備測試 {len(target_list)} 檔股票 (索引 {start_index} 到 {end_index-1})...\n")

    # 執行回測 (使用嚴格參數)
    trades_df = backtest.run_all_analysis(target_list, params=STRICT_PARAMS, is_training=False)

    if not trades_df.empty:
        # 填補空值以利閱讀
        display_df = trades_df.fillna({'下週收盤出場價': '等待開獎', '週報酬%': '等待開獎'})
        
        # 產生存檔檔名 (加上時間戳記避免重複或覆蓋)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"實戰回測報告_{timestamp}.xlsx"
        
        # 匯出成 Excel
        try:
            display_df.to_excel(filename, index=False, engine='openpyxl')
            print("\n" + "=" * 60)
            print(f"🎉 成功！完整的報表已經匯出為：【 {filename} 】")
            print("💡 請在 VS Code 左側檔案總管找到該檔案，按右鍵選擇『在檔案總管中顯示』(Reveal in File Explorer)，並用 Excel 開啟。")
            print("=" * 60)
        except Exception as e:
            print(f"\n❌ 匯出失敗，請確認是否有安裝 openpyxl 套件。錯誤訊息：{e}")
            print("👉 可以嘗試在終端機輸入：pip install openpyxl")
    else:
        print("\n⚠️ 嚴格參數下，沒有出現任何符合條件的回測訊號，無法匯出。")

if __name__ == "__main__":
    main()