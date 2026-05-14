import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 加载原始数据
# ==========================================
TRAIN_FILE = "0330用数据.xlsx"
VALID_FILE = "0330实例.xlsx"
TARGET_COL = "解释结论"
FEATURES = ['AC', 'CNL', 'DEN', 'AT90', 'R25', 'SP']

print("正在加载原始数据...")
train_df = pd.read_excel(TRAIN_FILE)
train_df.columns = train_df.columns.astype(str).str.strip()

# 加载验证集（用于了解数据结构）
valid_dict = pd.read_excel(VALID_FILE, sheet_name=None)

# ==========================================
# 统计原始数据分布特征
# ==========================================
print("\n正在统计分析原始数据特征...")

# 提取特征数据
X_train_raw = train_df[FEATURES].values
y_train_raw = train_df[TARGET_COL].values

# 统计每个特征的均值和标准差
feature_means = np.mean(X_train_raw, axis=0)
feature_stds = np.std(X_train_raw, axis=0)
feature_mins = np.min(X_train_raw, axis=0)
feature_maxs = np.max(X_train_raw, axis=0)

# 统计类别分布
unique_labels, label_counts = np.unique(y_train_raw, return_counts=True)
label_probs = label_counts / len(y_train_raw)

print(f"特征数量: {len(FEATURES)}")
print(f"特征名称: {FEATURES}")
print(f"特征均值: {feature_means}")
print(f"特征标准差: {feature_stds}")
print(f"类别分布: {dict(zip(unique_labels, label_counts))}")

# ==========================================
# 生成合成数据 0514synthetic_sample.xlsx
# ==========================================
print("\n正在生成合成训练数据 0514synthetic_sample.xlsx...")

np.random.seed(42)  # 设置随机种子，确保可重复性

n_synthetic_samples = 1500  # 生成1500个合成样本

synthetic_data = []
synthetic_labels = []

for i in range(n_synthetic_samples):
    # 随机选择一个类别（符合原始分布）
    label = np.random.choice(unique_labels, p=label_probs)
    synthetic_labels.append(label)
    
    # 根据该类别的特征分布生成合成特征
    # 获取该类别的原始数据
    class_mask = y_train_raw == label
    class_data = X_train_raw[class_mask]
    
    if len(class_data) > 1:
        # 计算该类别的均值和协方差
        class_mean = np.mean(class_data, axis=0)
        class_cov = np.cov(class_data.T)
        
        # 添加噪声使数据更真实
        noise = np.random.normal(0, 0.1, len(FEATURES))
        synthetic_sample = np.random.multivariate_normal(class_mean, class_cov + np.eye(len(FEATURES)) * 0.01)
        synthetic_sample = synthetic_sample + noise * feature_stds * 0.05
    else:
        # 如果类别样本太少，使用全局统计加噪声
        synthetic_sample = feature_means + np.random.normal(0, 0.5, len(FEATURES)) * feature_stds
    
    # 确保数值在合理范围内（基于原始数据边界）
    synthetic_sample = np.clip(synthetic_sample, feature_mins - 0.5 * feature_stds, feature_maxs + 0.5 * feature_stds)
    synthetic_data.append(synthetic_sample)

# 创建DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=FEATURES)
synthetic_df[TARGET_COL] = synthetic_labels

# 添加序号列
synthetic_df.insert(0, '序号', range(1, len(synthetic_df) + 1))

# 保存文件
synthetic_df.to_excel("0514synthetic_sample.xlsx", index=False)
print(f"✅ 已生成合成训练数据: 0514synthetic_sample.xlsx")
print(f"   样本数量: {len(synthetic_df)}")
print(f"   特征数量: {len(FEATURES)}")
print(f"   类别分布: \n{synthetic_df[TARGET_COL].value_counts()}")

# ==========================================
# 生成实例验证合成数据 0514synthetic_sample_example.xlsx
# ==========================================
print("\n正在生成合成实例验证数据 0514synthetic_sample_example.xlsx...")

# 获取第一个验证集的sheet名称用于参考
sheet_names = list(valid_dict.keys())
example_sheet_name = sheet_names[0] if sheet_names else "实例井"

# 生成验证集合成数据（较少的样本，模拟实际场景）
n_example_samples = 300

example_data = []
example_labels = []

for i in range(n_example_samples):
    # 随机选择类别
    label = np.random.choice(unique_labels, p=label_probs)
    example_labels.append(label)
    
    # 生成特征（使用全局统计加噪声，模拟实际测井数据）
    sample = feature_means + np.random.normal(0, 0.8, len(FEATURES)) * feature_stds
    sample = np.clip(sample, feature_mins - 0.3 * feature_stds, feature_maxs + 0.3 * feature_stds)
    example_data.append(sample)

# 创建DataFrame
example_df = pd.DataFrame(example_data, columns=FEATURES)
example_df[TARGET_COL] = example_labels

# 添加模拟的深度信息（可选）
example_df.insert(0, '起始深度', np.linspace(1000, 1500, n_example_samples))
example_df.insert(1, '终止深度', example_df['起始深度'] + 0.5)

# 保存文件
example_df.to_excel("0514synthetic_sample_example.xlsx", index=False)
print(f"✅ 已生成合成实例数据: 0514synthetic_sample_example.xlsx")
print(f"   样本数量: {len(example_df)}")
print(f"   类别分布: \n{example_df[TARGET_COL].value_counts()}")

# ==========================================
# 验证合成数据与原始数据的统计相似性
# ==========================================
print("\n" + "="*60)
print("合成数据与原始数据统计对比:")
print("="*60)

comparison_data = []
for i, feat in enumerate(FEATURES):
    comparison_data.append({
        '特征': feat,
        '原始均值': round(feature_means[i], 4),
        '合成均值': round(np.mean(synthetic_df[feat]), 4),
        '原始标准差': round(feature_stds[i], 4),
        '合成标准差': round(np.std(synthetic_df[feat]), 4)
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

print("\n" + "="*60)
print("生成完成！文件列表:")
print("1. 0514synthetic_sample.xlsx - 合成训练数据 (1500样本)")
print("2. 0514synthetic_sample_example.xlsx - 合成实例验证数据 (300样本)")
print("="*60)