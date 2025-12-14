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

# =====================================================
# 1. CẤU HÌNH APP
# =====================================================
st.set_page_config(layout="wide", page_title="Hệ thống Tín dụng Blockchain & AI")

st.markdown("""
<style>
.big-font { font-size:20px !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. SESSION STATE
# =====================================================
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = []
    st.session_state.access_rights = {}
    st.session_state.credit_scores = {}
    st.session_state.user_inputs = {}
    st.session_state.trained = False
    st.session_state.model = None
    st.session_state.feature_names = ['Age', 'Credit amount', 'Duration', 'Telco_Bill', 'Social_Score']

# =====================================================
# 3. BLOCKCHAIN GIẢ LẬP
# =====================================================
class SimpleBlockchain:
    @staticmethod
    def create_block(data, previous_hash):
        block = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data': data,
            'previous_hash': previous_hash,
            'nonce': np.random.randint(0, 1_000_000)
        }
        block_string = json.dumps(block, sort_keys=True).encode()
        block['hash'] = hashlib.sha256(block_string).hexdigest()
        return block

    @staticmethod
    def add_block(data):
        chain = st.session_state.blockchain
        prev_hash = chain[-1]['hash'] if chain else '0'*64
        block = SimpleBlockchain.create_block(data, prev_hash)
        chain.append(block)
        return block

# =====================================================
# 4. LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("final_thesis_data.csv")
    except:
        return pd.DataFrame()

# =====================================================
# 5. TRAIN AI
# =====================================================
def train_ai_model(df):
    X = df[st.session_state.feature_names]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=np.random.randint(0, 1000)
    )

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return model, acc, recall

# =====================================================
# 6. ASSESS RISK (BANK LOGIC)
# =====================================================
def assess_risk(score, credit, duration, telco, social):
    reasons = []

    if telco > 1_200_000:
        reasons.append("Chi tiêu viễn thông cao")
    if duration > 48:
        reasons.append("Thời hạn vay dài")
    if social < 50:
        reasons.append("Điểm xã hội thấp")
    if credit > 15_000:
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

# =====================================================
# 7. UI CHÍNH
# =====================================================
st.title("🛡️ Hệ thống Chấm điểm Tín dụng Blockchain & AI")
st.markdown("---")

df = load_data()

role = st.sidebar.radio(
    "Chọn vai trò",
    ["Admin & AI", "User", "Bank", "Network"]
)

# =====================================================
# ADMIN
# =====================================================
if role == "Admin & AI":
    st.header("⚙️ Huấn luyện AI")

    if df.empty:
        st.error("Không tìm thấy dữ liệu CSV")
    else:
        st.write(f"Số bản ghi: {df.shape[0]}")

        if st.button("🚀 Huấn luyện mô hình"):
            with st.spinner("Đang huấn luyện AI..."):
                model, acc, recall = train_ai_model(df)
                st.session_state.model = model
                st.session_state.trained = True

            st.success(f"Accuracy: {acc*100:.2f}% | Recall (rủi ro): {recall*100:.2f}%")

        st.markdown("---")
        st.subheader("Giả lập người vay")

        with st.form("loan_form"):
            age = st.slider("Tuổi", 18, 80, 30)
            credit = st.slider("Số tiền vay", 500, 20000, 8000)
            duration = st.slider("Thời hạn (tháng)", 6, 72, 24)
            telco = st.slider("Cước viễn thông", 50000, 2000000, 500000)
            social = st.slider("Điểm xã hội", 0, 100, 60)
            submit = st.form_submit_button("Chấm điểm")

        if submit and st.session_state.trained:
            X_input = pd.DataFrame([[age, credit, duration, telco, social]],
                columns=st.session_state.feature_names)

            proba = st.session_state.model.predict_proba(X_input)[0][1]
            score = int(proba * 850)

            user_id = f"UID_{np.random.randint(10000,99999)}"

            st.session_state.credit_scores[user_id] = score
            st.session_state.user_inputs[user_id] = {
                "credit": credit,
                "duration": duration,
                "telco": telco,
                "social": social
            }

            SimpleBlockchain.add_block({
                "event": "CREDIT_SCORING",
                "user": user_id,
                "score": score
            })

            st.success(f"Đã tạo người dùng {user_id} | Điểm: {score}")

# =====================================================
# USER
# =====================================================
elif role == "User":
    st.header("👤 Cổng người dùng")

    if not st.session_state.credit_scores:
        st.info("Chưa có dữ liệu người dùng")
    else:
        uid = st.selectbox("Chọn UID", list(st.session_state.credit_scores.keys()))
        score = st.session_state.credit_scores[uid]

        st.metric("Điểm tín dụng", score)

        if st.button("Cấp quyền cho Bank"):
            st.session_state.access_rights.setdefault(uid, []).append("Bank")
            SimpleBlockchain.add_block({"event": "GRANT_ACCESS", "user": uid})
            st.success("Đã cấp quyền")

# =====================================================
# BANK
# =====================================================
elif role == "Bank":
    st.header("🏦 Bảng điều khiển Ngân hàng")

    uid = st.text_input("Nhập UID")

    if st.button("Tra cứu"):
        if uid not in st.session_state.access_rights:
            st.error("Không có quyền truy cập")
        else:
            score = st.session_state.credit_scores.get(uid)
            inputs = st.session_state.user_inputs.get(uid)

            level, decision, reasons = assess_risk(
                score,
                inputs['credit'],
                inputs['duration'],
                inputs['telco'],
                inputs['social']
            )

            st.metric("Điểm tín dụng", score)
            st.write(f"Mức rủi ro: **{level}**")
            st.write(f"Khuyến nghị: **{decision}**")

            st.write("### Yếu tố rủi ro")
            if reasons:
                for r in reasons:
                    st.write(f"- {r}")
            else:
                st.write("Không có rủi ro đáng kể")

            SimpleBlockchain.add_block({
                "event": "BANK_DECISION",
                "user": uid,
                "decision": decision
            })

# =====================================================
# NETWORK
# =====================================================
elif role == "Network":
    st.header("🌐 Sơ đồ hệ thống")

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
    st.json(st.session_state.blockchain)
