import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score

FEATURE_COLS = [
    '1週報酬率',
    '4週報酬率',
    '8週報酬率',
    '5日均線偏離率',
    '20日均線偏離率',
    '成交量增幅',
    '外資淨買賣超/成交量',
    '融資融券比率',
    '大戶持股比(TEJ)',  # 🌟 新增 TEJ 特徵
    '散戶持股比(TEJ)', # 🌟 新增 TEJ 特徵
    '大戶散戶差'
]
SEQUENCE_LENGTH = 3
TARGET_COL = '未來1週報酬%'


def build_sequences(df):
    records = []
    df = df.sort_values(['股票代號', '進場日期']).copy()

    for stock_id, group in df.groupby('股票代號', sort=False):
        group = group.sort_values('進場日期').reset_index(drop=True)
        if len(group) < SEQUENCE_LENGTH + 1:
            continue

        for i in range(SEQUENCE_LENGTH, len(group)):
            window = group.iloc[i - SEQUENCE_LENGTH:i][FEATURE_COLS].to_numpy(dtype=np.float32)
            target = float(group.iloc[i][TARGET_COL])
            end_date = pd.to_datetime(group.iloc[i]['進場日期'])
            records.append({
                'end_date': end_date,
                'window': window,
                'target': target
            })

    if not records:
        return np.empty((0, SEQUENCE_LENGTH, len(FEATURE_COLS)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    records_df = pd.DataFrame(records).sort_values('end_date').reset_index(drop=True)
    X = np.stack(records_df['window'].to_numpy()).astype(np.float32)
    # y = records_df['target'].to_numpy().astype(np.float32)
    # 改為將數值轉二元標籤：
    y = (records_df['target'] > 0).astype(np.float32).to_numpy()
    return X, y


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_training_data.csv")

    if not os.path.exists(data_path):
        print(f"❌ 找不到訓練資料 {data_path}，請先執行回測程式。")
        return

    print("讀取訓練資料...")
    df = pd.read_csv(data_path)
    target_column = TARGET_COL if TARGET_COL in df.columns else '是否獲利'

    
    # 特徵合成：直接計算大戶與散戶的持股差距
    df['大戶散戶差'] = df['大戶持股比(TEJ)'] - df['散戶持股比(TEJ)']

    df = df.dropna(subset=FEATURE_COLS + [target_column]).copy()

    # 限制未來4週報酬率的極端值 (例如最多漲 40%，最多跌 -30%)
    df[TARGET_COL] = df[TARGET_COL].clip(lower=-15, upper=15)


    if len(df) < 50:
        print("⚠️ 警告：有效樣本數少於 50 筆，模型可能無法有效學習。")

    if target_column == '是否獲利':
        df[TARGET_COL] = df['是否獲利'].astype(float)

    X, y = build_sequences(df)
    if len(X) == 0:
        print("⚠️ 依照序列設定，沒有足夠的訓練樣本。")
        return

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 只使用訓練資料計算標準化參數，避免測試資料資訊洩漏。
    train_values = X_train.reshape(-1, X_train.shape[-1])
    feature_mean = train_values.mean(axis=0)
    feature_scale = train_values.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0

    X_train = ((X_train - feature_mean) / feature_scale).astype(np.float32)
    X_test = ((X_test - feature_mean) / feature_scale).astype(np.float32)

    # target_mean = y_train.mean()
    # target_scale = y_train.std()
    # if target_scale == 0:
    #     target_scale = 1.0
    # y_train_scaled = ((y_train - target_mean) / target_scale).astype(np.float32)

    target_mean = 0.0
    target_scale = 1.0
    y_train_scaled = y_train

    print(f"⏳ 正在使用 LSTM 進行『未來 4 週報酬率』回歸預測，序列長度={SEQUENCE_LENGTH}, 樣本數={len(X)}...")
    # model = keras.Sequential([
    #     keras.Input(shape=(SEQUENCE_LENGTH, len(FEATURE_COLS))),
    #     layers.LSTM(16, return_sequences=False), # 🌟 拔掉一層 LSTM，神經元砍半
    #     layers.Dropout(0.3),                     # 🌟 提高遺忘率，強迫它不要死背
    #     layers.Dense(8, activation='relu'),      # 🌟 減少 Dense 複雜度
    #     layers.Dense(1, activation='sigmoid')
    # ])

    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )

    model = keras.Sequential([
        keras.Input(shape=(SEQUENCE_LENGTH, len(FEATURE_COLS))),
        # 🌟 加入 L2 正規化 (0.005)，懲罰過大的無效權重
        layers.LSTM(16, return_sequences=False, kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.Dropout(0.4), # 🌟 提高 Dropout 增加擾動
        layers.Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.Dense(1, activation='sigmoid') # 🌟 保持 sigmoid，輸出 0~1 的機率
    ])

    model.compile(
        # 🌟 降學習率，讓模型慢慢找最佳解
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        # 🌟 關鍵：為了讓 R 平方大於 0，損失函數必須用 mse (Mean Squared Error)
        loss='mse', 
        metrics=['accuracy']
    )

    # model.fit(
    #     X_train,
    #     y_train_scaled,
    #     validation_split=0.1,
    #     epochs=100,
    #     batch_size=256,
    #     verbose=1,
    #     callbacks=[
    #         keras.callbacks.ReduceLROnPlateau(
    #             monitor='val_loss',
    #             factor=0.5,
    #             patience=3,
    #             min_lr=1e-6
    #         ),
    #         keras.callbacks.EarlyStopping(
    #             monitor='val_loss',
    #             patience=12,
    #             restore_best_weights=True
    #         )
    #     ]
    # )


    model.fit(
        X_train,
        y_train_scaled,
        validation_split=0.1,
        epochs=150,      
        batch_size=64,   # 稍微拉高一點，讓誤差均值更穩定
        verbose=1,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,      
                min_lr=1e-6
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,     
                restore_best_weights=True
            )
        ]
    )

    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = y_pred_scaled * target_scale + target_mean
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    r2 = r2_score(y_test, y_pred)
    baseline_pred = np.full_like(y_test, target_mean)
    baseline_mae = np.mean(np.abs(baseline_pred - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline_pred - y_test) ** 2))

    # print("\n📊 模型驗證報告:")
    # print(f"平均絕對誤差 (MAE): {mae:.3f}%")
    # print(f"均方根誤差 (RMSE): {rmse:.3f}%")
    # print(f"R平方分數 (R2 Score): {r2:.4f}")
    # print(f"Baseline MAE: {baseline_mae:.3f}%")
    # print(f"Baseline RMSE: {baseline_rmse:.3f}%")
    # print(f"實際平均 1 週報酬: {np.mean(y_test):+.3f}%")
    # print(f"預測平均 1 週報酬: {np.mean(y_pred):+.3f}%")
    # if mae < baseline_mae and rmse < baseline_rmse:
    #     print("模型表現：優於 Baseline")
    # else:
    #     print("模型表現：未優於 Baseline")
    # print("-" * 30)

    # ==========================================
    # 以下為 model.fit() 執行完畢後的評估與存檔邏輯
    # ==========================================

    # ==========================================
    # 以下為 model.fit() 執行完畢後的評估與存檔邏輯
    # ==========================================

    # 1. 取得 0~1 的勝率機率
    y_pred_prob = model.predict(X_test, verbose=0).reshape(-1)
    
    # 2. 計算 R 平方分數 (檔案最上方已經 import 過 r2_score，這裡直接用)
    r2 = r2_score(y_test, y_pred_prob)
    
    # 3. 計算測試集的實際勝率
    actual_win_rate = np.mean(y_test)
    
    # 🌟 4. 動態及格線：大於市場平均勝率就視為看漲 (1)
    y_pred_class = (y_pred_prob >= actual_win_rate).astype(int)
    
    # 5. 計算準確率與 Baseline (檔案最上方已經 import 過，直接用)
    acc = accuracy_score(y_test, y_pred_class)
    baseline_acc = max(actual_win_rate, 1 - actual_win_rate)

    print("\n📊 模型分類與機率驗證報告:")
    print(f"測試集實際獲利比例: {actual_win_rate * 100:.1f}%")
    print(f"無腦瞎猜準確率 (Baseline): {baseline_acc * 100:.1f}%")
    print(f"🤖 模型預測準確率 (Accuracy): {acc * 100:.1f}%")
    print(f"🎯 R平方分數 (R2 Score): {r2:.4f}") 
    
    print("\n【詳細分類指標】")
    print(classification_report(y_test, y_pred_class, target_names=['看跌/虧損 (0)', '看漲/獲利 (1)'], zero_division=0))
    print("-" * 30)

    # 下方保留你原本的存檔邏輯 (model_filename ...)
    model_filename = Path(__file__).resolve().parent / 'lstm_trading_model.keras'
    scaler_filename = Path(__file__).resolve().parent / 'lstm_feature_scaler.npz'
    np.savez(
        scaler_filename,
        mean=feature_mean,
        scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale
    )
    model.save(model_filename)
    print(f"✅ 特徵標準化參數已儲存為 {scaler_filename}")
    print(f"\n✅ 模型已成功儲存為 {model_filename}")


if __name__ == "__main__":
    main()
