import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. 加载数据
file_path = "汇总结果改1215.xlsx"
df = pd.read_excel(file_path)

# 确保时序轴为深度
df = df.sort_values(by='起始深度').reset_index(drop=True)

# 2. 特征与标签：数据维度调整为 (深度序列长度, 6)
features = ['AC', 'CNL', 'DEN', 'GR', 'SP', 'R04']
X = df[features].fillna(df[features].mean())
y = df['解释结论']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. 按序列划分数据集
split_index = int(len(X) * 0.8)
X_train = X.iloc[:split_index].copy()
y_train = y_encoded[:split_index]
X_test = X.iloc[split_index:].copy()
y_test = y_encoded[split_index:]

# 4. 类别加权 + 时序样本增强
augmented_X, augmented_y = [], []
for cls in np.unique(y_train):
    if le.inverse_transform([cls])[0] != '非水淹层':
        cls_X = X_train[y_train == cls]
        # 扩充少数类样本并注入微小波动
        for _ in range(3):
            noise = np.random.normal(0, 0.01, cls_X.shape) * cls_X.values
            augmented_X.append(cls_X + noise)
            augmented_y.append(np.full(len(cls_X), cls))

if augmented_X:
    X_train_aug = pd.concat([X_train] + augmented_X, ignore_index=True)
    y_train_aug = np.concatenate([y_train] + augmented_y)
else:
    X_train_aug, y_train_aug = X_train, y_train

# 5. 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_aug, y_train_aug)

# 6. 预测与评估
y_pred = rf.predict(X_test)

print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}\n")
print("分类报告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=le.classes_))