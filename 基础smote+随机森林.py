import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

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

# 5. 构建并训练随机森林模型
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 6. 预测与评估
y_pred = rf_classifier.predict(X_test)

print("\n--- 模型评估报告 ---")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")

print("\n分类报告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 7. 查看特征重要性
feature_importances = pd.DataFrame(
    rf_classifier.feature_importances_,
    index = X.columns,
    columns=['重要性']
).sort_values('重要性', ascending=False)

print("\n特征重要性排名:")
print(feature_importances)