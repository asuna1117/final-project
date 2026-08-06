import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def main():
    data_path = "ml_training_data.csv"
    
    # 1. 檢查資料是否存在
    if not os.path.exists(data_path):
        print(f"❌ 找不到訓練資料 {data_path}，請先執行回測程式。")
        return

    # 2. 讀取資料
    print("讀取訓練資料...")
    df = pd.read_csv(data_path)
    
    # 🌟 更新：將 FinMind 外部籌碼特徵加入訓練陣列中
    feature_cols = [
        '大戶相關係數', '散戶相關係數', '均張相關係數', 
        '大戶四週成長率', '散戶衰退率',
        '外資買賣超', '融資餘額', '融券餘額'
    ]
    
    # 移除包含空值的列，確保訓練品質
    df = df.dropna(subset=feature_cols + ['是否獲利'])
    
    if len(df) < 50:
        print("⚠️ 警告：有效樣本數少於 50 筆，模型可能無法有效學習。")

    X = df[feature_cols]
    y = df['是否獲利']

    # 3. 切分資料集 (80% 訓練, 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 建立與訓練模型
    print("⏳ 正在使用 sklearn 訓練隨機森林模型...")
    # 設定參數：n_estimators(決策樹數量), max_depth(最大深度，避免過擬合)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 5. 模型成效評估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n📊 模型驗證報告:")
    print(f"準確率 (Accuracy): {acc:.2%}")
    print("-" * 30)
    print(classification_report(y_test, y_pred))

    # 6. 輸出特徵重要性 (觀察哪個籌碼條件最關鍵)
    print("\n🔍 特徵重要性分析:")
    importances = model.feature_importances_
    for name, imp in zip(feature_cols, importances):
        print(f" - {name}: {imp:.2%}")

    # 7. 儲存模型
    model_filename = 'rf_trading_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\n✅ 模型已成功儲存為 {model_filename}")

if __name__ == "__main__":
    main()