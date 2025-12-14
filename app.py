# =============================================================
# APP STREAMLIT – GIỮ DEMO CŨ + BỔ SUNG AI & BANK LOGIC (NO LEAKAGE)
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import json
import time
import graphviz
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

# =============================================================
# 1. CẤU HÌNH & CSS (GIỮ NGUYÊN)
# =============================================================
st.set_page_config(layout="wide", page_title="Hệ thống Tín dụng Blockchain Pro")

st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .success-score { color: green; font-weight: bold; }
    .fail-score { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================
# 2. SESSION STATE (GIỮ + THÊM)
# =============================================================
if 'blockchain' not in st.session_state:
    st.session_state['blockchain'] = []
    st.session_state['access_rights'] = {}
    st.session_state['credit_scores'] = {}
    st.session_state['user_inputs'] = {}
    st.session_state['trained'] = False
    st.session_state['model'] = None
    st.session_state['metrics'] = {}
    st.session_state['feature_names'] = ['Age', 'Credit amount', 'Duration', 'Telco_Bill', 'Social_Score']

# =============================================================
# 3. BLOCKCHAIN GIẢ LẬP (GIỮ)
# =============================================================
class SimpleBlockchain:
    @staticmethod
    def create_block(data, previous_hash="0"*64):
        block = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            'data': data,
            'previous_hash': previous_hash,
            'nonce': np.random.randint(0, 1000000),
            'validator': f"Node_{np.random.randint(1,5)}"
        }
        block_string = json.dumps(block, sort_keys=True).encode()
        block['hash'] = hashlib.sha256(block_string).hexdigest()
        return block

    @staticmethod
    def add_to_chain(data):
        chain = st.session_state['blockchain']
        prev_hash = chain[-1]['hash'] if chain else "0"*64
        block = SimpleBlockchain.create_block(data, prev_hash)
        chain.append(block)
        return block

# =============================================================
# 4. LOAD DATA (GIỮ)
# =============================================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("final_thesis_data.csv")
    except:
        return pd.DataFrame()

# =============================================================
# 5. TRAIN AI (SỬA – KHÔNG OVERFITTING)
# =============================================================
def train_ai_model(df):
    X = df[st.session_state['feature_names']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=np.random.randint(0, 1000)
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=25,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return model, acc, recall

# =============================================================
# 6. BANK RISK LOGIC (THÊM)
# =============================================================
def assess_risk(score, user_data):
    reasons = []

    if user_data['Telco_Bill'] > 1_200_000:
        reasons.append("Chi tiêu viễn thông cao")
    if user_data['Duration'] > 48:
        reasons.append("Thời hạn vay dài")
    if user_data['Social_Score'] < 50:
        reasons.append("Điểm xã hội thấp")
    if user_data['Credit amount'] > 15_000:
        reasons.append("Khoản vay lớn")

    if score >= 700:
        level = "Thấp"
        decision = "Duyệt tự động"
    elif score >= 600:
        level = "Trung bình"
        decision = "Duyệt có điều kiện"
    else:
        level = "Cao"
        decision = "Từ chối / Yêu cầu thế chấp"

    return level, decision, reasons

# =============================================================
# 7. UI CHÍNH (GIỮ STRUCTURE CŨ)
# =============================================================
st.title("🛡️ Hệ thống Chấm điểm Tín dụng Blockchain & AI")
st.markdown("---")

df = load_data()

role = st.sidebar.radio(
    "CHỌN VAI TRÒ TRUY CẬP",
    ["1. ⚙️ Admin & AI", "2. 👤 User", "3. 🏦 Bank", "4. 🌐 Network"]
)

# =============================================================
# ADMIN & AI
# =============================================================
if "1." in role:
    st.header("⚙️ Huấn luyện AI & Giả lập")

    if not df.empty:
        st.write(f"Số bản ghi: {df.shape[0]}")

        if st.button("🚀 Huấn luyện AI"):
            with st.spinner("Đang huấn luyện mô hình..."):
                model, acc, recall = train_ai_model(df)
                st.session_state['model'] = model
                st.session_state['trained'] = True
                st.session_state['metrics'] = {
                    'accuracy': acc,
                    'recall': recall
                }

            st.success(f"Accuracy: {acc*100:.2f}% | Recall (rủi ro): {recall*100:.2f}%")

    st.markdown("---")
    st.subheader("Giả lập người vay mới")

    with st.form("sim_form"):
        age = st.slider("Tuổi", 18, 80, 30)
        credit = st.slider("Số tiền vay", 500, 20000, 8000)
        duration = st.slider("Thời hạn (tháng)", 6, 72, 24)
        telco = st.slider("Cước viễn thông", 50_000, 2_000_000, 500_000)
        social = st.slider("Điểm xã hội", 0, 100, 60)
        submit = st.form_submit_button("⚡ Chấm điểm & Đóng block")

    if submit and st.session_state['trained']:
        # Hiệu ứng mining (GIỮ DEMO)
        progress = st.progress(0)
        for i in range(5):
            progress.progress((i+1)*20)
            time.sleep(0.3)

        X_input = pd.DataFrame([[age, credit, duration, telco, social]],
            columns=st.session_state['feature_names'])

        proba = st.session_state['model'].predict_proba(X_input)[0][1]
        score = int(proba * 850)
        user_id = f"UID_{np.random.randint(10000,99999)}"

        st.session_state['credit_scores'][user_id] = score
        st.session_state['user_inputs'][user_id] = {
            'Age': age,
            'Credit amount': credit,
            'Duration': duration,
            'Telco_Bill': telco,
            'Social_Score': social
        }

        SimpleBlockchain.add_to_chain({
            'event': 'CREDIT_SCORING',
            'user': user_id,
            'score': score
        })

        st.success(f"Đã tạo {user_id} | Điểm: {score}")

# =============================================================
# USER
# =============================================================
elif "2." in role:
    st.header("👤 Cổng thông tin người dùng")

    if st.session_state['credit_scores']:
        uid = st.selectbox("Chọn UID", list(st.session_state['credit_scores'].keys()))
        score = st.session_state['credit_scores'][uid]

        st.metric("Điểm tín dụng", score)

        if st.button("Cấp quyền cho Bank_A"):
            st.session_state['access_rights'].setdefault(uid, []).append("Bank_A")
            SimpleBlockchain.add_to_chain({'event': 'GRANT_ACCESS', 'user': uid})
            st.success("Đã cấp quyền")

# =============================================================
# BANK
# =============================================================
elif "3." in role:
    st.header("🏦 Bảng điều khiển Ngân hàng")

    uid = st.text_input("Nhập UID")

    if st.button("Tra cứu"):
        if "Bank_A" not in st.session_state['access_rights'].get(uid, []):
            st.error("⛔ Không có quyền truy cập")
        else:
            score = st.session_state['credit_scores'].get(uid)
            user_data = st.session_state['user_inputs'].get(uid)

            level, decision, reasons = assess_risk(score, user_data)

            st.metric("Điểm tín dụng", score)
            st.write(f"Mức rủi ro: **{level}**")
            st.write(f"Khuyến nghị: **{decision}**")

            st.write("### Yếu tố rủi ro")
            if reasons:
                for r in reasons:
                    st.write(f"- {r}")
            else:
                st.write("Không phát hiện rủi ro đáng kể")

            SimpleBlockchain.add_to_chain({
                'event': 'BANK_DECISION',
                'user': uid,
                'decision': decision
            })

# =============================================================
# NETWORK
# =============================================================
elif "4." in role:
    st.header("🌐 Sơ đồ mạng lưới")

    g = graphviz.Digraph()
    g.attr(rankdir='LR')

    g.node('U', 'User')
    g.node('AI', 'AI Engine')
    g.node('BC', 'Blockchain')
    g.node('B', 'Bank')

    g.edge('U', 'AI')
    g.edge('AI', 'BC')
    g.edge('U', 'BC')
    g.edge('BC', 'B')

    st.graphviz_chart(g)

    st.subheader("📦 Blockchain Ledger")
    st.dataframe(pd.DataFrame(st.session_state['blockchain']))
