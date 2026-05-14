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
# Fix Matplotlib Chinese display issue
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # For Windows: SimHei, For Mac: Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. Custom SE (Squeeze-and-Excitation) Module
# ==========================================
def se_block(input_tensor, ratio=8):
    """
    1D version of SE (Squeeze-and-Excitation) channel attention mechanism.
    Adaptively learns the importance weights of each feature channel (well logging curve).
    """
    filters = input_tensor.shape[-1]
    
    # Squeeze: Global average pooling, compress time step dimension
    se = GlobalAveragePooling1D()(input_tensor)
    
    # Excitation: Two fully connected layers to extract non-linear relationships between channels
    se = Dense(max(1, filters // ratio), activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    
    # Expand weights back to (batch_size, 1, channels) for multiplication with original tensor
    se = Reshape((1, filters))(se)
    
    # Scale: Weight the channels of the original input
    x = Multiply()([input_tensor, se])
    return x

# ==========================================
# 2. Sliding Window Generation Function
# ==========================================
def create_sliding_windows(X, y, window_size=5):
    """
    Add depth context for well logging data.
    Pad head and tail using 'edge' mode to ensure output sequence length exactly matches original data count.
    """
    pad_size = window_size // 2
    X_padded = np.pad(X, ((pad_size, pad_size), (0, 0)), mode='edge')
    X_windows = []
    for i in range(len(X)):
        X_windows.append(X_padded[i:i + window_size])
    return np.array(X_windows), y

# ==========================================
# 3. Data Loading and Preprocessing Core Logic (Using Synthetic Data)
# ==========================================
# UPDATED: Using synthetic data files instead of original proprietary data
TRAIN_FILE = "0514synthetic_sample.xlsx"
VALID_FILE = "0514synthetic_sample_example.xlsx"
TARGET_COL = "解释结论"
WINDOW_SIZE = 5  # Set sliding window size

def load_and_preprocess():
    print(f"Loading synthetic training set and synthetic validation set...")
    print(f"Training file: {TRAIN_FILE}")
    print(f"Validation file: {VALID_FILE}")
    
    train_df = pd.read_excel(TRAIN_FILE)
    train_df.columns = train_df.columns.astype(str).str.strip()
    
    # Define features (matching the synthetic data generation)
    features = ['AC', 'CNL', 'DEN', 'AT90', 'R25', 'SP']
    print(f"✅ Using {len(features)} features: {features}")
    
    X_train_raw = train_df[features].values
    y_train_raw = train_df[TARGET_COL].values
    
    all_labels = [y_train_raw]
    valid_datasets_raw = {}
    
    # Load validation synthetic data (single sheet)
    valid_df = pd.read_excel(VALID_FILE)
    valid_df.columns = valid_df.columns.astype(str).str.strip()
    
    if TARGET_COL in valid_df.columns:
        all_labels.append(valid_df[TARGET_COL].values)
        valid_datasets_raw["synthetic_well"] = valid_df
        print(f"✅ Loaded synthetic validation set with {len(valid_df)} samples")
    else:
        print(f"⚠️ Warning: '{TARGET_COL}' not found in validation file")
        
    le = LabelEncoder()
    le.fit(np.concatenate(all_labels))
    num_classes = len(le.classes_)
    print(f"✅ Number of classes: {num_classes}")
    print(f"   Classes: {list(le.classes_)}")
    
    # ----------------------------------------------------
    # Training set processing: Normalization -> Window -> Flatten -> SMOTE -> Reshape to 3D
    # ----------------------------------------------------
    y_train_encoded = le.transform(y_train_raw)
    scaler_train = RobustScaler()
    X_train_scaled = scaler_train.fit_transform(X_train_raw)
    
    # Split sliding windows
    X_train_seq, y_train_seq = create_sliding_windows(X_train_scaled, y_train_encoded, window_size=WINDOW_SIZE)
    X_train_seq_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    
    print("\n[Core Process]: Performing global SMOTE oversampling on sliding window features...")
    k_neighbors = min(3, min(np.bincount(y_train_seq)) - 1)
    smote = SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))
    X_train_res_flat, y_train_res = smote.fit_resample(X_train_seq_flat, y_train_seq)
    
    # Reshape back to 3D tensor shape (samples, window_size, num_features)
    X_train_res = X_train_res_flat.reshape(X_train_res_flat.shape[0], WINDOW_SIZE, len(features))
    y_train_cat = to_categorical(y_train_res, num_classes=num_classes)
    
    print(f"✅ After SMOTE: {X_train_res.shape[0]} training samples")
    
    # ----------------------------------------------------
    # Validation set processing: Independent normalization -> Window
    # ----------------------------------------------------
    valid_datasets = {}
    for sheet_name, df in valid_datasets_raw.items():
        X_val_raw = df[features].values
        y_val_raw = df[TARGET_COL].values
        
        y_val_encoded = le.transform(y_val_raw)
        
        scaler_val = RobustScaler()
        X_val_scaled = scaler_val.fit_transform(X_val_raw)
        
        # Validation set also uses sliding windows
        X_val_seq, _ = create_sliding_windows(X_val_scaled, y_val_encoded, window_size=WINDOW_SIZE)
        
        valid_datasets[sheet_name] = {
            "X": X_val_seq,  
            "y_encoded": y_val_encoded
        }
        print(f"✅ Validation set [{sheet_name}] preprocessing completed, sequence shape: {X_val_seq.shape}")
    
    return X_train_res, y_train_cat, valid_datasets, le, num_classes

# ==========================================
# 4. Build the Dedicated CNN-BiLSTM-SE Model
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
    # Core modification: Integrate SE channel attention mechanism
    # ----------------------------------------------------
    
    # Since the SE module outputs a 3D tensor with time steps (batch, steps, channels)
    # We need to flatten it to connect to subsequent Dense layers
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
# 5. Visualize Training Process
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
    plt.title('Accuracy Variation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Test Loss')
    plt.title('Loss Variation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("CNN_BiLSTM_SE_Training_Curve.png", dpi=300)
    print("\n📈 Training monitoring chart saved to: CNN_BiLSTM_SE_Training_Curve.png")
    plt.show()

# ==========================================
# 6. Main Program
# ==========================================
def main():
    print("="*70)
    print("CNN-BiLSTM-SE Model Training with Synthetic Data")
    print("="*70)
    print("Note: This model is trained on synthetic data for code validation purposes.")
    print("Results may differ from those obtained with original proprietary data.\n")
    
    X_train_res, y_train_cat, valid_datasets, le, num_classes = load_and_preprocess()
    input_shape = (X_train_res.shape[1], X_train_res.shape[2])
    
    print("\n[Core Process]: Splitting the oversampled dataset into internal Train / Validation (8:2) for monitoring...")
    X_t, X_val, y_t, y_val = train_test_split(X_train_res, y_train_cat, test_size=0.2, random_state=42)
    y_val_labels = np.argmax(y_val, axis=1)
    
    print("\n" + "="*70)
    print("🚀 Start training dedicated model: [CNN-BiLSTM-SE] (Synthetic Data + SE Channel Attention)")
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
    # Output classification report for internal validation set
    # ==========================================
    train_val_probs = model.predict(X_val, verbose=0)
    train_val_preds = np.argmax(train_val_probs, axis=1)
    train_val_acc = accuracy_score(y_val_labels, train_val_preds)
    train_val_f1 = f1_score(y_val_labels, train_val_preds, average='macro')
    
    print("\n" + "="*70)
    print(f"🎯 [Synthetic Training Data] Internal Validation Set Classification Report")
    print(f"Accuracy: {train_val_acc:.4f} | Macro F1: {train_val_f1:.4f}")
    print("-" * 50)
    print(classification_report(y_val_labels, train_val_preds, 
                                labels=np.arange(num_classes), 
                                target_names=le.classes_, 
                                zero_division=0))
    
    # ==========================================
    # Output classification report for validation set
    # ==========================================
    print("\n" + "="*70)
    print("📊 [Synthetic Validation Data] Final Classification Report")
    print("="*70)
    
    for sheet_name, data_dict in valid_datasets.items():
        X_v = data_dict["X"]
        y_v_encoded = data_dict["y_encoded"]
        
        probs = model.predict(X_v, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        acc = accuracy_score(y_v_encoded, preds)
        macro_f1 = f1_score(y_v_encoded, preds, average='macro')
        
        print(f"\n👉 Validation Set [{sheet_name}] (Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}):")
        
        print(classification_report(y_v_encoded, preds, 
                                    labels=np.arange(num_classes), 
                                    target_names=le.classes_, 
                                    zero_division=0))

if __name__ == "__main__":
    main()