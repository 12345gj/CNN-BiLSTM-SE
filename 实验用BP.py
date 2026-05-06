import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout

# 1. 数据加载与预处理
df = pd.read_excel("汇总结果改1215.xlsx").sort_values(by='起始深度').reset_index(drop=True)
features = ['AC', 'CNL', 'DEN', 'GR', 'SP', 'R04']
X_raw = StandardScaler().fit_transform(df[features].fillna(df[features].mean()).values)
y_raw = df['解释结论'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

# 2. 构造时序维度 (窗口=10)
seq_length = 10
X_seq, y_seq = [], []
for i in range(len(X_raw) - seq_length + 1):
    X_seq.append(X_raw[i:i+seq_length])
    y_seq.append(y_encoded[i+seq_length-1])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# 3. 全局高斯噪声扩充
augmented_X, augmented_y = [], []
for cls in np.unique(y_seq):
    if le.inverse_transform([cls])[0] != '非水淹层':
        cls_X = X_seq[np.where(y_seq == cls)[0]]
        for _ in range(3):
            augmented_X.append(cls_X + np.random.normal(0.0, 0.05, cls_X.shape))
            augmented_y.append(np.full(len(cls_X), cls))

if augmented_X:
    X_seq_all = np.concatenate([X_seq] + augmented_X, axis=0)
    y_seq_all = np.concatenate([y_seq] + augmented_y, axis=0)
else:
    X_seq_all, y_seq_all = X_seq, y_seq

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_seq_all, y_seq_all, test_size=0.2, random_state=42, stratify=y_seq_all
)
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

# 【BP 特有处理】：将 3D 时序数据展平为 2D 向量 (样本数, 10 * 6 = 60)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 5. 构建 BP 神经网络模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(seq_length * 6,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\n=== 正在训练 BP 神经网络 ===")
model.fit(X_train_flat, y_train, epochs=100, batch_size=32, class_weight=class_weights, validation_data=(X_test_flat, y_test), verbose=0)

y_pred = np.argmax(model.predict(X_test_flat), axis=1)
print(f"BP 准确率: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))