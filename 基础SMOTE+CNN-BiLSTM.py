import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 引入深度学习相关的库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense

# 1. 加载数据 (根据您上传的文件名)
file_path = "汇总结果改1215.xlsx"
df = pd.read_excel(file_path)

# 2. 数据预处理
# 去除不参与预测的列，保留测井曲线特征
X = df.drop(columns=['序号', '起始深度', '终止深度', '解释结论'])
y = df['解释结论']

# 缺失值填充
X = X.fillna(X.mean())

# 将文本标签转换为数字标签
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. 【第一步】先进行全局 SMOTE 过采样
smote = SMOTE(random_state=42)
# 如果由于某些类别样本极少导致报错，可以尝试添加 k_neighbors 参数，例如：
# smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

print(f"过采样前总样本数: {len(y)}")
print(f"过采样后总样本数: {len(y_resampled)}")

# 4. 【第二步】再划分训练集和测试集
# test_size=0.2 表示 20% 的数据用于测试，80% 用于训练
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# ======== 以下为替换的 CNN-BiLSTM 模型部分 ========

# CNN-BiLSTM 需要 3D 格式的输入：(样本数, 特征步长, 特征通道数)
# 原数据是 2D 的 (样本数, 特征数)，需扩充最后一维
X_train_3d = np.expand_dims(X_train, axis=-1)
X_test_3d = np.expand_dims(X_test, axis=-1)

# 5. 构建并训练 CNN-BiLSTM 模型
model = Sequential([
    # 一维卷积层
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train_3d.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    
    # 双向 LSTM 层
    Bidirectional(LSTM(128, return_sequences=False)),
    
    # 全连接层与输出层
    Dense(64, activation='relu'),
    Dense(4, activation='softmax') # 多分类输出
])

# 编译模型 (因为标签是整数编码，所以使用 sparse_categorical_crossentropy)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print("\n开始训练 CNN-BiLSTM 模型...")
model.fit(X_train_3d, y_train, epochs=100, batch_size=32, verbose=1)

# 6. 预测与评估
y_pred_prob = model.predict(X_test_3d)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n--- 模型评估报告 ---")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")

print("\n分类报告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

