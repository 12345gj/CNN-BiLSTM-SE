import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Layer, BatchNormalization, GlobalAveragePooling1D, Reshape, Multiply, Flatten
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical

# ==========================================
# 解决 Matplotlib 中文显示问题
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # Windows用黑体，Mac用Arial
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. 自定义 SE (Squeeze-and-Excitation) 模块
# ==========================================
def se_block(input_tensor, ratio=8):
    """
    1D 版本的 SE (Squeeze-and-Excitation) 通道注意力机制。
    能够自适应地学习各个特征通道（测井曲线）的重要性权重。
    """
    filters = input_tensor.shape[-1]
    
    # Squeeze: 全局平均池化，压缩时间步维度
    se = GlobalAveragePooling1D()(input_tensor)
    
    # Excitation: 两个全连接层提取通道间的非线性关系
    se = Dense(max(1, filters // ratio), activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    
    # 将权重扩展回 (batch_size, 1, channels) 以便与原张量相乘
    se = Reshape((1, filters))(se)
    
    # Scale: 对原始输入的通道进行加权
    x = Multiply()([input_tensor, se])
    return x

# ==========================================
# 2. 滑动窗口生成函数
# ==========================================
def create_sliding_windows(X, y, window_size=5):
    """
    为测井数据添加深度上下文。
    使用 edge 模式填充头尾，确保输出序列长度与原数据条数完全一致。
    """
    pad_size = window_size // 2
    X_padded = np.pad(X, ((pad_size, pad_size), (0, 0)), mode='edge')
    X_windows = []
    for i in range(len(X)):
        X_windows.append(X_padded[i:i + window_size])
    return np.array(X_windows), y

# ==========================================
# 3. 数据加载与预处理核心逻辑 (全部特征 + 先SMOTE)
# ==========================================
TRAIN_FILE = "0330用数据.xlsx"
VALID_FILE = "0330实例.xlsx"
TARGET_COL = "解释结论"
EXCLUDE_COLS = ['序号', '起始深度', '终止深度', TARGET_COL]
WINDOW_SIZE = 5 # 设置滑动窗口大小

def load_and_preprocess():
    print(f"正在加载训练集与实例验证集 (自动提取全部特征)...")
    
    train_df = pd.read_excel(TRAIN_FILE)
    train_df.columns = train_df.columns.astype(str).str.strip()
    
    # 自动获取所有数值类型的特征列，排除非特征列
    features = ['AC','CNL','DEN','AT90','R25','SP']
    print(f"✅ 共提取到 {len(features)} 个特征: {features}")
    
    X_train_raw = train_df[features].values
    y_train_raw = train_df[TARGET_COL].values
    
    all_labels = [y_train_raw]
    valid_datasets_raw = {}
    valid_dict = pd.read_excel(VALID_FILE, sheet_name=None)
    
    for sheet_name, df in valid_dict.items():
        df.columns = df.columns.astype(str).str.strip() 
        if df.empty or TARGET_COL not in df.columns:
            continue
        all_labels.append(df[TARGET_COL].values)
        valid_datasets_raw[sheet_name] = df
        
    le = LabelEncoder()
    le.fit(np.concatenate(all_labels))
    num_classes = len(le.classes_)
    
    # ----------------------------------------------------
    # 【训练集处理】：归一化 -> 切窗口 -> 展平 -> SMOTE过采样 -> 还原3D
    # ----------------------------------------------------
    y_train_encoded = le.transform(y_train_raw)
    scaler_train = RobustScaler()
    X_train_scaled = scaler_train.fit_transform(X_train_raw)
    
    # 切分滑动窗口
    X_train_seq, y_train_seq = create_sliding_windows(X_train_scaled, y_train_encoded, window_size=WINDOW_SIZE)
    X_train_seq_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    
    print("\n【核心流程】：正在对滑动窗口特征进行全局 SMOTE 过采样...")
    k_neighbors = min(3, min(np.bincount(y_train_seq)) - 1)
    smote = SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))
    X_train_res_flat, y_train_res = smote.fit_resample(X_train_seq_flat, y_train_seq)
    
    # 还原回 3D 张量形状 (样本数, 窗口大小, 特征数)
    X_train_res = X_train_res_flat.reshape(X_train_res_flat.shape[0], WINDOW_SIZE, len(features))
    y_train_cat = to_categorical(y_train_res, num_classes=num_classes)
    
    # ----------------------------------------------------
    # 【实例验证集处理】：独立归一化 -> 切窗口
    # ----------------------------------------------------
    valid_datasets = {}
    for sheet_name, df in valid_datasets_raw.items():
        X_val_raw = df[features].values
        y_val_raw = df[TARGET_COL].values
        
        y_val_encoded = le.transform(y_val_raw)
        
        scaler_val = RobustScaler()
        X_val_scaled = scaler_val.fit_transform(X_val_raw)
        
        # 验证集同样切分滑动窗口
        X_val_seq, _ = create_sliding_windows(X_val_scaled, y_val_encoded, window_size=WINDOW_SIZE)
        
        valid_datasets[sheet_name] = {
            "X": X_val_seq,  
            "y_encoded": y_val_encoded
        }
        print(f"✅ 实例集 [{sheet_name}] 预处理完成，序列形状: {X_val_seq.shape}")
    
    return X_train_res, y_train_cat, valid_datasets, le, num_classes

# ==========================================
# 4. 构建专属 CNN-BiLSTM-SE 模型 
# ==========================================
def build_cnn_bilstm_se(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    reg = l2(0.005) 
    
    x = Conv1D(filters=64, kernel_size=3, padding='same', kernel_regularizer=reg)(inputs)
    x = BatchNormalization()(x) 
    x = tf.keras.layers.Activation('relu')(x)
    x = se_block(x, ratio=8)
    x = MaxPooling1D(pool_size=2)(x) 
    x = Dropout(0.3)(x) 
    
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=reg))(x)
    x = BatchNormalization()(x) 
    x = Dropout(0.3)(x)
    
    # ----------------------------------------------------
    # 核心修改：接入 SE 通道注意力机制
    # ----------------------------------------------------
    
    
    # 由于 SE 模块输出的仍是带时间步的三维张量 (batch, steps, channels)
    # 我们需要将其展平，以便接入后续的 Dense 层
    x = Flatten()(x)
    
    x = Dense(32, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Smooth_CNN_BiLSTM_SE")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# ==========================================
# 5. 可视化训练过程
# ==========================================
def plot_training_history(history):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'r-', label='Test Acc')
    plt.title('准确率变化 (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Test Loss')
    plt.title('损失变化 (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("CNN_BiLSTM_SE_监控图.png", dpi=300)
    print("\n📈 训练监控图已保存至: CNN_BiLSTM_SE_监控图.png")
    plt.show()

# ==========================================
# 6. 主程序
# ==========================================
def main():
    X_train_res, y_train_cat, valid_datasets, le, num_classes = load_and_preprocess()
    input_shape = (X_train_res.shape[1], X_train_res.shape[2])
    
    print("\n【核心流程】：正在将过采样后的数据集划分为内部 Train / Validation (8:2) 供监控使用...")
    X_t, X_val, y_t, y_val = train_test_split(X_train_res, y_train_cat, test_size=0.2, random_state=42)
    y_val_labels = np.argmax(y_val, axis=1)
    
    print("\n" + "="*70)
    print("🚀 开始训练专属模型: 【CNN-BiLSTM-SE】 (全部特征 + SE通道注意力)")
    print("="*70)
    
    model = build_cnn_bilstm_se(input_shape, num_classes)
    model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=20, 
        restore_best_weights=True, 
        verbose=1
    )
    
    history = model.fit(
        X_t, y_t,
        epochs=150, 
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping], 
        verbose=1 
    )
    
    plot_training_history(history)
    
    # ==========================================
    # 输出内部验证集 (0330用数据) 的分类报告
    # ==========================================
    train_val_probs = model.predict(X_val, verbose=0)
    train_val_preds = np.argmax(train_val_probs, axis=1)
    train_val_acc = accuracy_score(y_val_labels, train_val_preds)
    train_val_f1 = f1_score(y_val_labels, train_val_preds, average='macro')
    
    print("\n" + "="*70)
    print(f"🎯 【0330用数据】内部验证期 (过采样域) 分类报告")
    print(f"Accuracy: {train_val_acc:.4f} | Macro F1: {train_val_f1:.4f}")
    print("-" * 50)
    print(classification_report(y_val_labels, train_val_preds, 
                                labels=np.arange(num_classes), 
                                target_names=le.classes_, 
                                zero_division=0))
    
    # ==========================================
    # 输出独立实例验证集 (0330实例) 的分类报告
    # ==========================================
    print("\n" + "="*70)
    print("📊 【0330实例】独立实例集 最终分类报告")
    print("="*70)
    
    for sheet_name, data_dict in valid_datasets.items():
        X_v = data_dict["X"]
        y_v_encoded = data_dict["y_encoded"]
        
        probs = model.predict(X_v, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        acc = accuracy_score(y_v_encoded, preds)
        macro_f1 = f1_score(y_v_encoded, preds, average='macro')
        
        print(f"\n👉 实例井 【{sheet_name}】 (Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}):")
        
        print(classification_report(y_v_encoded, preds, 
                                    labels=np.arange(num_classes), 
                                    target_names=le.classes_, 
                                    zero_division=0))

if __name__ == "__main__":
    main()