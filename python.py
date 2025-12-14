import pandas as pd
import numpy as np

"""
FILE TIỀN XỬ LÝ DỮ LIỆU (PHIÊN BẢN CHUẨN HỌC THUẬT)
--------------------------------------------------
Mục tiêu:
- Sử dụng German Credit Dataset
- TUYỆT ĐỐI KHÔNG data leakage
- Giả lập thêm Big Data (Telco, Social) một cách GIÁN TIẾP
- Phù hợp cho báo cáo cuối kì (AI + Blockchain)
"""

# ==================================================
# 1. LOAD DATASET GỐC
# ==================================================
try:
    df = pd.read_csv("german_credit_data.csv")
except Exception as e:
    raise RuntimeError("Không tìm thấy file german_credit_data.csv")

print(f"Loaded German Credit Dataset: {df.shape[0]} rows")

# ==================================================
# 2. XÁC ĐỊNH CỘT RISK / TARGET
# ==================================================
# Kaggle German Credit thường có cột 'Risk'
if 'Risk' not in df.columns:
    raise ValueError("Dataset không có cột 'Risk'. Hãy kiểm tra lại file CSV.")

# Chuẩn hóa nhãn
# good -> 1 (không vỡ nợ)
# bad  -> 0 (rủi ro cao)
df['Target'] = df['Risk'].map({'good': 1, 'bad': 0})

# ==================================================
# 3. CHỌN FEATURE CỐT LÕI (KHÔNG LEAKAGE)
# ==================================================
required_cols = ['Age', 'Credit amount', 'Duration']
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Thiếu cột bắt buộc: {c}")

# Đảm bảo kiểu số
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Credit amount'] = pd.to_numeric(df['Credit amount'], errors='coerce')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Drop NA
df = df.dropna(subset=required_cols + ['Target'])

# ==================================================
# 4. GIẢ LẬP TELCO_BILL (KHÔNG DÙNG RISK)
# ==================================================
# Telco liên quan GIÁN TIẾP tới hành vi chi tiêu và thời hạn vay
np.random.seed(42)

df['Telco_Bill'] = (
    df['Credit amount'] / df['Credit amount'].max() * 700_000
    + df['Duration'] / df['Duration'].max() * 400_000
    + np.random.normal(0, 120_000, size=len(df))
).clip(50_000, 1_500_000)

# ==================================================
# 5. GIẢ LẬP SOCIAL_SCORE (KHÔNG DÙNG RISK)
# ==================================================
# Social score bị ảnh hưởng bởi độ ổn định tài chính (proxy)

df['Social_Score'] = (
    100
    - df['Duration'] / df['Duration'].max() * 25
    - df['Credit amount'] / df['Credit amount'].max() * 25
    + np.random.normal(0, 10, size=len(df))
).clip(10, 95)

# ==================================================
# 6. CHỌN DATASET CUỐI CÙNG CHO AI
# ==================================================
final_cols = [
    'Age',
    'Credit amount',
    'Duration',
    'Telco_Bill',
    'Social_Score',
    'Target'
]

final_df = df[final_cols]

# ==================================================
# 7. LƯU FILE CHO STREAMLIT APP
# ==================================================
final_df.to_csv("final_thesis_data.csv", index=False)

print("\n✅ Tiền xử lý hoàn tất!")
print("File 'final_thesis_data.csv' đã sẵn sàng cho huấn luyện AI.")
print("Các cột sử dụng:")
print(final_df.columns.tolist())

print("\nPreview dữ liệu:")
print(final_df.head())