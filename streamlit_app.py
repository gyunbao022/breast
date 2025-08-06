%%writefile streamlit_app.py
# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


# ----------------------------------------
# 1. 모델 로드 (가장 최근 .keras 파일)
# ----------------------------------------
MODEL_DIR = "saved_models"

def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

latest_model_path = get_latest_model()
model = load_model(latest_model_path) if latest_model_path else None

# ----------------------------------------
# 2. 데이터 로딩 및 스케일링 학습
# ----------------------------------------
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names  # ['setosa', 'versicolor', 'virginica']

scaler = StandardScaler()
scaler.fit(X)

# ----------------------------------------
# 3. Streamlit UI
# ----------------------------------------
st.title("붓꽃 품종 분류기 (Iris Classifier)")
if model:
    st.markdown(f"불러온 모델: `{os.path.basename(latest_model_path)}`")
else:
    st.error("저장된 모델을 찾을 수 없습니다. 학습을 먼저 진행하세요.")

st.sidebar.header("입력값을 설정하세요")

user_input = []
for i, feature in enumerate(feature_names):
    val = st.sidebar.slider(
        label=feature,
        min_value=float(X[:, i].min()),
        max_value=float(X[:, i].max()),
        value=float(X[:, i].mean()),
        format="%.2f"
    )
    user_input.append(val)

user_array = np.array(user_input).reshape(1, -1)
scaled_input = scaler.transform(user_array)

# ----------------------------------------
# 4. 예측 수행
# ----------------------------------------
if st.button("예측 실행") and model:
    pred_probs = model.predict(scaled_input)[0]  # [0.1, 0.7, 0.2]
    pred_class = np.argmax(pred_probs)
    pred_label = class_names[pred_class]

    st.subheader("예측 결과")
    st.write("클래스 별 확률:")
    for i, prob in enumerate(pred_probs):
        st.write(f"- {class_names[i]}: **{prob * 100:.2f}%**")

    st.write(f"👉 최종 예측: **{pred_label}**")
    st.success(f"{pred_label} 품종으로 분류되었습니다.")
