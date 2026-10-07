import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor

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
    '大戶散戶差',
    '短中期動能差',
    '短長均線差'
]
SEQUENCE_LENGTH = 5
TARGET_COL = '未來1週報酬%'
XGB_MODEL_FILENAME = 'xgb_return_model.json'


def _transform_return(values):
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def _inverse_return(values):
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.expm1(np.abs(values))


def build_sequences(df):
    records = []
    df = df.sort_values(['股票代號', '進場日期']).copy()

    for stock_id, group in df.groupby('股票代號', sort=False):
        group = group.sort_values('進場日期').reset_index(drop=True)
        dates = pd.to_datetime(group['進場日期'])
        if len(group) < SEQUENCE_LENGTH + 1:
            continue

        for i in range(SEQUENCE_LENGTH, len(group)):
            window_dates = dates.iloc[i - SEQUENCE_LENGTH:i + 1]
            if window_dates.diff().dt.days.max() > 14:
                continue
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
    y = records_df['target'].to_numpy(dtype=np.float32)
    return X, y


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_training_data.csv")

    if not os.path.exists(data_path):
        print(f"❌ 找不到訓練資料 {data_path}，請先執行回測程式。")
        return

    print("讀取訓練資料...")
    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        print(f"❌ 訓練資料缺少回歸目標欄位：{TARGET_COL}")
        return
    target_column = TARGET_COL

    
    # 特徵合成：直接計算大戶與散戶的持股差距
    df['大戶散戶差'] = df['大戶持股比(TEJ)'] - df['散戶持股比(TEJ)']
    df['短中期動能差'] = df['1週報酬率'] - df['4週報酬率'] / 4.0
    df['短長均線差'] = df['5日均線偏離率'] - df['20日均線偏離率']

    df = df.dropna(subset=FEATURE_COLS + [target_column]).copy()

    # 保留原始報酬，後續排除疑似除權異常值並做 signed-log 轉換。


    if len(df) < 50:
        print("⚠️ 警告：有效樣本數少於 50 筆，模型可能無法有效學習。")

    X, y_raw = build_sequences(df)
    valid_targets = np.abs(y_raw) <= 50.0
    X = X[valid_targets]
    y_raw = y_raw[valid_targets]
    y = _transform_return(y_raw)
    if len(X) == 0:
        print("⚠️ 依照序列設定，沒有足夠的訓練樣本。")
        return

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    y_raw_train, y_raw_test = y_raw[:split_idx], y_raw[split_idx:]

    # 只使用訓練資料計算標準化參數，避免測試資料資訊洩漏。
    train_values = X_train.reshape(-1, X_train.shape[-1])
    feature_lower = np.quantile(train_values, 0.01, axis=0)
    feature_upper = np.quantile(train_values, 0.99, axis=0)
    X_train = np.clip(X_train, feature_lower, feature_upper)
    X_test = np.clip(X_test, feature_lower, feature_upper)
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

    target_mean = float(y_train.mean())
    target_scale = float(y_train.std())
    if target_scale == 0:
        target_scale = 1.0
    y_train_scaled = ((y_train - target_mean) / target_scale).astype(np.float32)

    print(f"⏳ 正在使用 LSTM 進行『未來 1 週報酬率』回歸預測，序列長度={SEQUENCE_LENGTH}, 樣本數={len(X)}...")
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
        layers.Dense(1, activation='linear')
    ])

    model.compile(
        # 🌟 降學習率，讓模型慢慢找最佳解
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.Huber(),
        metrics=['mae', 'mse']
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

    encoder = keras.Model(inputs=model.inputs[0], outputs=model.layers[-2].output)
    train_latent = encoder.predict(X_train, verbose=0)
    test_latent = encoder.predict(X_test, verbose=0)

    xgb_model = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=4
    )
    xgb_model.fit(train_latent, y_train, eval_set=[(test_latent, y_test)], verbose=False)
    xgb_pred_transformed = xgb_model.predict(test_latent)
    xgb_pred = _inverse_return(xgb_pred_transformed)

    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    lstm_pred_transformed = y_pred_scaled * target_scale + target_mean
    lstm_pred = _inverse_return(lstm_pred_transformed)
    lstm_mae = np.mean(np.abs(lstm_pred - y_raw_test))
    lstm_rmse = np.sqrt(np.mean((lstm_pred - y_raw_test) ** 2))
    lstm_r2 = r2_score(y_raw_test, lstm_pred)
    mae = np.mean(np.abs(xgb_pred - y_raw_test))
    rmse = np.sqrt(np.mean((xgb_pred - y_raw_test) ** 2))
    r2 = r2_score(y_raw_test, xgb_pred)
    baseline_pred = np.full_like(y_raw_test, y_raw_train.mean())
    baseline_mae = np.mean(np.abs(baseline_pred - y_raw_test))
    baseline_rmse = np.sqrt(np.mean((baseline_pred - y_raw_test) ** 2))

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

    print("\n📊 模型回歸驗證報告:")
    print(f"測試集平均實際報酬率: {np.mean(y_raw_test):+.3f}%")
    print(f"LSTM 平均預測報酬率: {np.mean(lstm_pred):+.3f}%")
    print(f"XGBoost 平均預測報酬率: {np.mean(xgb_pred):+.3f}%")
    print(f"LSTM MAE / RMSE / R²: {lstm_mae:.3f}% / {lstm_rmse:.3f}% / {lstm_r2:.4f}")
    print(f"平均絕對誤差 (MAE): {mae:.3f}%")
    print(f"均方根誤差 (RMSE): {rmse:.3f}%")
    print(f"R平方分數 (R2 Score): {r2:.4f}")
    print(f"Baseline MAE: {baseline_mae:.3f}%")
    print(f"Baseline RMSE: {baseline_rmse:.3f}%")
    print("-" * 30)

    # 下方保留你原本的存檔邏輯 (model_filename ...)
    model_filename = Path(__file__).resolve().parent / 'lstm_trading_model.keras'
    scaler_filename = Path(__file__).resolve().parent / 'lstm_feature_scaler.npz'
    np.savez(
        scaler_filename,
        mean=feature_mean,
        scale=feature_scale,
        lower=feature_lower,
        upper=feature_upper,
        target_mean=target_mean,
        target_scale=target_scale
    )
    model.save(model_filename)
    xgb_model.save_model(str(Path(__file__).resolve().parent / XGB_MODEL_FILENAME))
    print(f"✅ 特徵標準化參數已儲存為 {scaler_filename}")
    print(f"\n✅ 模型已成功儲存為 {model_filename}")


if __name__ == "__main__":
    main()
