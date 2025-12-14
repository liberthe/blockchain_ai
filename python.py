import pandas as pd
import numpy as np
import random

# 1. Load file bạn vừa tải từ Kaggle (giả sử tên file là 'german_credit_data.csv')
# Lưu ý: Bạn cần xóa cột đầu tiên (index) nếu có
try:
    df = pd.read_csv("german_credit_data.csv", index_col=0)
except:
    # Trường hợp chưa tải file, tạo demo frame để code không lỗi
    print("Chưa thấy file, đang tạo demo...")
    df = pd.DataFrame({'Age': [25, 30, 45], 'Credit amount': [1000, 5000, 2000], 'Risk': ['good', 'bad', 'good']})

print(f"Dữ liệu gốc có {len(df)} dòng.")

# 2. GIẢ LẬP THÊM CÁC CỘT "BIG DATA" (Để khớp với đề tài)
# Chúng ta sẽ thêm cột "Tiền điện thoại" và "Điểm hoạt động mạng xã hội"

def generate_telco_data(risk_status):
    # Nếu Risk là 'bad', xu hướng tiền cước thấp hoặc nợ cước (giả định)
    if risk_status == 'bad':
        return random.randint(50, 200) * 1000  # 50k - 200k
    else:
        return random.randint(200, 1000) * 1000 # 200k - 1 triệu

# Ensure 'Risk' column exists (try common alternatives, otherwise create heuristically)
if 'Risk' not in df.columns:
    alt = None
    for c in df.columns:
        if c.lower() in ['risk', 'class', 'creditability', 'target', 'credit_risk', 'default']:
            alt = c
            break
    if alt:
        df['Risk'] = df[alt]
        print(f"Đã dùng cột '{alt}' làm 'Risk'.")
    else:
        print("Cột 'Risk' không tồn tại. Tạo nhãn giả (heuristic) dựa trên 'Credit amount' và 'Duration'.")
        if 'Credit amount' in df.columns and 'Duration' in df.columns:
            ca = pd.to_numeric(df['Credit amount'], errors='coerce').fillna(0)
            du = pd.to_numeric(df['Duration'], errors='coerce').fillna(0)
            # simple score: larger credit amount and longer duration -> higher risk
            score = (ca / (ca.median() if ca.median() != 0 else 1)) + (du / (du.median() if du.median() != 0 else 1))
            df['Risk'] = np.where(score > np.median(score), 'bad', 'good')
        else:
            df['Risk'] = np.random.choice(['good', 'bad'], size=len(df))
            print("Không tìm thấy 'Credit amount' hoặc 'Duration'; gán nhãn ngẫu nhiên.")

# Tạo cột Telco_Bill (Hóa đơn viễn thông)
df['Telco_Bill'] = df['Risk'].apply(generate_telco_data)

# Tạo cột Social_Score (Điểm uy tín mạng xã hội - giả lập từ 0 đến 100)
# Người 'good' thường có social score cao hơn (giả định cho AI học)
df['Social_Score'] = df['Risk'].apply(lambda x: random.randint(40, 80) if x == 'bad' else random.randint(60, 99))

# 3. Chuyển đổi nhãn Risk từ chữ sang số để AI học (Good -> 1, Bad -> 0)
df['Target'] = df['Risk'].map({'good': 1, 'bad': 0})

# 4. Lưu lại thành file mới để dùng cho Demo Streamlit
df.to_csv("final_thesis_data.csv", index=False)

print("Đã xử lý xong! File 'final_thesis_data.csv' đã có cả dữ liệu Ngân hàng lẫn Big Data.")
print(df[['Credit amount', 'Risk', 'Telco_Bill', 'Social_Score']].head())