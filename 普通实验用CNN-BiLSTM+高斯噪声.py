import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# 引入深度学习相关的库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense

# 1. 加载数据并沿深度时序排序
file_path = "汇总结果改1215.xlsx"
df = pd.read_excel(file_path)
df = df.sort_values(by='起始深度').reset_index(drop=True)

# 2. 数据预处理
features = ['AC', 'CNL', 'DEN', 'GR', 'SP', 'R04']
X_raw = df[features].fillna(df[features].mean()).values
y_raw = df['解释结论'].values

# 数据标准化
scaler = StandardScaler()
X_raw = scaler.fit_transform(X_raw)

# 将文本标签转换为数字标签
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

# 3. 构造时序维度：(深度序列长度, 6)
seq_length = 10  # 滑动窗口大小
X_seq, y_seq = [], []
for i in range(len(X_raw) - seq_length + 1):
    X_seq.append(X_raw[i:i+seq_length])
    y_seq.append(y_encoded[i+seq_length-1])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# 4. 【第一步：先在全局范围内加入高斯噪声扩充少数类】=================
augmented_X, augmented_y = [], []
for cls in np.unique(y_seq):
    cls_name = le.inverse_transform([cls])[0]
    # 对水淹层（非主流类别）进行噪声扩充
    if cls_name != '非水淹层':
        cls_indices = np.where(y_seq == cls)[0]
        cls_X = X_seq[cls_indices]
        
        # 扩充 3 倍样本，注入标准差为 0.05 的高斯微扰噪声
        for _ in range(3):
            noise = np.random.normal(loc=0.0, scale=0.05, size=cls_X.shape)
            augmented_X.append(cls_X + noise)
            augmented_y.append(np.full(len(cls_X), cls))

# 将生成的增强样本与原全局数据进行拼接
if augmented_X:
    X_seq_all = np.concatenate([X_seq] + augmented_X, axis=0)
    y_seq_all = np.concatenate([y_seq] + augmented_y, axis=0)
else:
    X_seq_all = X_seq
    y_seq_all = y_seq
# =================================================================

# 5. 【第二步：再划分训练集和测试集】==============================
# 使用分层抽样确保各类别在训练集和测试集中比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X_seq_all, y_seq_all, test_size=0.2, random_state=42, stratify=y_seq_all
)
# =================================================================

# 计算类别权重以应对剩余的不平衡 (基于 y_train 计算)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# 6. 构建并训练 CNN-BiLSTM 模型
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(seq_length, 6)),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n开始训练 CNN-BiLSTM 模型...")
model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    class_weight=class_weight_dict, 
    validation_data=(X_test, y_test), 
    verbose=1
)

# 7. 预测与评估
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n--- 模型评估报告 ---")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=le.classes_))