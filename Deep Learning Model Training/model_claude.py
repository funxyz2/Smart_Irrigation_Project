import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import os
import warnings
warnings.filterwarnings('ignore')

# Tạo thư mục cho kết quả
os.makedirs('results', exist_ok=True)

# Đọc dữ liệu
print("Đang đọc dữ liệu...")
df = pd.read_excel("smart_irrigation_dataset_1000.xlsx")

# Kiểm tra dữ liệu
print("Thông tin dữ liệu:")
print(df.info())
print("\nMô tả thống kê:")
print(df.describe())

# Xử lý dữ liệu thiếu
missing_data = df.isnull().sum()
print("\nDữ liệu thiếu:")
print(missing_data)
if missing_data.sum() > 0:
    # Thay thế bằng giá trị trung bình
    df = df.fillna(df.mean())
    print("Đã xử lý dữ liệu thiếu bằng giá trị trung bình")

# Kiểm tra và xử lý outliers
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\nKiểm tra dữ liệu ngoại lệ:")
outliers_count = 0
for col in df.select_dtypes(include=np.number).columns:
    outliers, _, _ = detect_outliers(df, col)
    if not outliers.empty:
        outliers_count += len(outliers)
        print(f"- {col}: {len(outliers)} outliers")

# Sử dụng RobustScaler nếu có nhiều outliers
use_robust_scaler = outliers_count > len(df) * 0.05
print(f"\nTổng số outliers: {outliers_count}")
print(f"Sử dụng RobustScaler: {use_robust_scaler}")

# Feature engineering
print("\nThực hiện feature engineering...")

# Cyclical encoding cho time_of_day (24h)
df['time_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
df['time_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24)

# Feature mới: Thời gian kể từ lần tưới cuối
df['hours_since_watered'] = (df['last_watered_hour']) % 24

# Chỉ số khô hạn (ví dụ đơn giản)
df['drought_index'] = df['temperature'] / (df['humidity_air'] + 1) * 10

# Tách input/output
print("\nChuẩn bị dữ liệu cho mô hình...")
X = df.drop(["ml_water", "time_of_day", "last_watered_hour"], axis=1)
y = df["ml_water"]

# In ra các features được sử dụng
print(f"Features sử dụng ({X.shape[1]}):")
for col in X.columns:
    print(f"- {col}")

# Chia train/validation/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\nKích thước dữ liệu:")
print(f"- Train: {X_train.shape[0]} mẫu")
print(f"- Validation: {X_val.shape[0]} mẫu")
print(f"- Test: {X_test.shape[0]} mẫu")

# Chuẩn hóa dữ liệu
if use_robust_scaler:
    scaler = RobustScaler()
    y_scaler = RobustScaler()
else:
    scaler = StandardScaler()
    y_scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# Chuyển sang Tensor + Dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nSử dụng thiết bị: {device}")

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Xây mô hình
class WaterNet(nn.Module):
    def __init__(self, input_dim):
        super(WaterNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),

            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

model = WaterNet(X_train_scaled.shape[1]).to(device)
print("\nCấu trúc mô hình:")
print(model)

# Loss, Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

# Hàm đánh giá mô hình
def evaluate_model(model, X_tensor, y_true, y_scaler):
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
    y_true = y_true.flatten()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

# Train mô hình
print("\nBắt đầu huấn luyện mô hình...")
epochs = 1000
patience = 20
best_val_loss = float("inf")
counter = 0
history = {
    'train_loss': [],
    'val_loss': [],
    'lr': []
}

for epoch in range(epochs):
    # Training
    model.train()
    train_epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_epoch_loss += loss.item()
    
    train_loss = train_epoch_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            val_loss += batch_loss.item()
    
    val_loss = val_loss / len(val_loader)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Lưu lịch sử
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['lr'].append(current_lr)
    
    # Hiển thị thông tin
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}")
    
    # Learning rate scheduler
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered sau {epoch+1} epochs.")
            break

# Tải mô hình tốt nhất
model.load_state_dict(best_model)

# Đánh giá mô hình trên tập test
print("\nĐánh giá mô hình trên tập test:")
test_metrics = evaluate_model(model, X_test_tensor, y_test.values, y_scaler)

print(f"- MSE: {test_metrics['mse']:.4f}")
print(f"- RMSE: {test_metrics['rmse']:.4f}")
print(f"- MAE: {test_metrics['mae']:.4f}")
print(f"- R²: {test_metrics['r2']:.4f}")

# Vẽ biểu đồ kết quả
# 1. Biểu đồ so sánh giá trị thực tế và dự đoán
plt.figure(figsize=(10, 8))
plt.scatter(y_test, test_metrics['predictions'], alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('So sánh giá trị thực tế và dự đoán')
plt.xlabel('Giá trị thực tế (ml)')
plt.ylabel('Giá trị dự đoán (ml)')
plt.grid(True)
plt.text(0.05, 0.95, f"MSE: {test_metrics['mse']:.2f}\nRMSE: {test_metrics['rmse']:.2f}\nMAE: {test_metrics['mae']:.2f}\nR²: {test_metrics['r2']:.4f}",
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('results/prediction_comparison.png')

# 2. Biểu đồ quá trình huấn luyện
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Quá trình huấn luyện')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(history['lr'])
plt.title('Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.savefig('results/training_history.png')

# 3. Biểu đồ phân phối lỗi
plt.figure(figsize=(10, 6))
errors = y_test - test_metrics['predictions']
plt.hist(errors, bins=30, alpha=0.7, color='blue')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Phân phối lỗi dự đoán')
plt.xlabel('Lỗi (ml)')
plt.ylabel('Tần suất')
plt.grid(True)
plt.savefig('results/error_distribution.png')

# 4. Biểu đồ Feature Importance
def calculate_feature_importance(model, X_scaled, feature_names):
    importance = []
    model.eval()
    baseline = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).detach().cpu().numpy().mean()
    
    for i in range(X_scaled.shape[1]):
        X_modified = X_scaled.copy()
        X_modified[:, i] = 0  # Nullify feature
        with torch.no_grad():
            output = model(torch.tensor(X_modified, dtype=torch.float32).to(device)).detach().cpu().numpy().mean()
        importance.append(abs(output - baseline))
    
    # Normalize importance
    importance = np.array(importance) / sum(importance)
    return dict(zip(feature_names, importance))

feature_importance = calculate_feature_importance(model, X_test_scaled, X.columns)
sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(12, 8))
plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
plt.title('Feature Importance')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('results/feature_importance.png')

# Lưu mô hình và scaler
print("\nLưu mô hình và scaler...")
# Thay vì chỉ lưu state_dict
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': model  # Lưu cả kiến trúc
}, 'results/deep_model.pth')
joblib.dump(scaler, 'results/scaler.pkl')
joblib.dump(y_scaler, 'results/y_scaler.pkl')
joblib.dump(history, 'results/training_history.pkl')

# Lưu thông tin mô hình
with open('results/model_info.txt', 'w') as f:
    f.write(f"Model Architecture:\n{model}\n\n")
    f.write(f"Input Features ({X.shape[1]}):\n")
    for col in X.columns:
        f.write(f"- {col}\n")
    f.write("\nTest Metrics:\n")
    f.write(f"- MSE: {test_metrics['mse']:.4f}\n")
    f.write(f"- RMSE: {test_metrics['rmse']:.4f}\n")
    f.write(f"- MAE: {test_metrics['mae']:.4f}\n")
    f.write(f"- R²: {test_metrics['r2']:.4f}\n")

# Dự đoán cho dữ liệu mới
print("\nDự đoán cho dữ liệu mới:")
input_data = {
    "temperature": 31.5,
    "soil_moisture": 40.2,
    "water_level": 75,
    "humidity_air": 39.6,
    "light_intensity": 2286,
    "time_of_day": 9,
    "rain_prediction": 0,
    "last_watered_hour": 9
}

# Feature engineering cho dữ liệu mới
input_data["time_sin"] = np.sin(2 * np.pi * input_data["time_of_day"] / 24)
input_data["time_cos"] = np.cos(2 * np.pi * input_data["time_of_day"] / 24)
input_data["hours_since_watered"] = (input_data["last_watered_hour"]) % 24
input_data["drought_index"] = input_data["temperature"] / (input_data["humidity_air"] + 1) * 10

# Xóa các features không sử dụng
keys_to_remove = ["time_of_day", "last_watered_hour"]
for key in keys_to_remove:
    if key in input_data:
        del input_data[key]

# Chuyển thành DataFrame
X_input = pd.DataFrame([input_data])
X_scaled = scaler.transform(X_input)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    pred_scaled = model(X_tensor).cpu().numpy()
    prediction = y_scaler.inverse_transform(pred_scaled)[0][0]

print(f"Lượng nước dự đoán: {prediction:.2f} ml")

print("\nQuá trình huấn luyện và đánh giá hoàn tất!")
print("Các kết quả đã được lưu trong thư mục 'results/'")