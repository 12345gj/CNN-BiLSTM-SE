import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# 引入深度学习相关的库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense

# 1. 加载数据
file_path = "汇总结果改1215.xlsx"
df = pd.read_excel(file_path)

# 【核心修改 1】：利用“序号”重置为 1 的特性，自动识别并生成独立的“井号”
# (df['序号'] == 1) 会在每口井开头产生 True(1)，cumsum 累加后就会变成 1, 2, 3... 的独立井号
df['井号'] = (df['序号'] == 1).cumsum()

# 在每口井的内部，确保按照深度排序，避免地层错乱
df = df.sort_values(by=['井号', '起始深度']).reset_index(drop=True)

# 2. 数据预处理
features =  ['AC', 'AT30', 'CNL', 'GR', 'R25', 'SP']
# 填充缺失值
df[features] = df[features].fillna(df[features].mean())

# 数据全局标准化 (保持所有井的量纲一致)
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 将文本标签转换为数字标签
le = LabelEncoder()
df['标签编码'] = le.fit_transform(df['解释结论'])

# 3. 构造时序维度：(深度序列长度, 6)
seq_length = 10  # 滑动窗口大小
X_seq_list, y_seq_list = [], []

# 【核心修改 2】：按“井号”分组，在每口井内部独立进行滑动窗口切片，绝不跨井！
for well_id, well_data in df.groupby('井号'):
    well_X = well_data[features].values
    well_y = well_data['标签编码'].values
    
    # 如果某口井的数据长度还不足一个窗口，则跳过
    if len(well_X) < seq_length:
        continue
        
    for i in range(len(well_X) - seq_length + 1):
        X_seq_list.append(well_X[i:i+seq_length])
        y_seq_list.append(well_y[i+seq_length-1])

X_seq = np.array(X_seq_list)
y_seq = np.array(y_seq_list)

print(f"按井划分完成，共生成了 {len(X_seq)} 个不跨井边界的合法时序窗口。")

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