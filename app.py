import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from torchdiffeq import odeint

# ======================================================
# LOAD PREPROCESSOR & MODELS
# ======================================================
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

lstm_model = load_model("lstm_model.keras")
transformer_model = load_model("transformer_model.keras")

# ======================================================
# NEURAL ODE MODEL
# ======================================================
class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )

    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.odefunc = ODEFunc(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        t = torch.tensor([0., 1.])
        out = odeint(self.odefunc, x, t)
        out = out[-1]
        return self.classifier(out)

INPUT_DIM = preprocessor.transform(
    pd.DataFrame(
        [np.zeros(len(preprocessor.feature_names_in_))],
        columns=preprocessor.feature_names_in_
    )
).shape[1]

neural_ode_model = NeuralODE(INPUT_DIM, 3)
neural_ode_model.load_state_dict(
    torch.load("neural_ode_model.pt", map_location="cpu")
)
neural_ode_model.eval()

# ======================================================
# UI HEADER
# ======================================================
st.title("🎓 Prediksi Performa Belajar Mahasiswa")

st.markdown("""
Aplikasi ini membandingkan **LSTM, Transformer, dan Neural ODE**
dalam memprediksi status akademik mahasiswa.
""")

# ======================================================
# MODEL PERFORMANCE (FROM NOTEBOOK)
# ======================================================
st.subheader("📊 Performa Model (Validation Set)")

st.markdown("""
- **LSTM** → Accuracy: **0.86**, F1-score: **0.85**  
- **Transformer** → Accuracy: **0.88**, F1-score: **0.87**  
- **Neural ODE** → Accuracy: **0.83**, F1-score: **0.82**
""")

# ======================================================
# MODEL SELECTION
# ======================================================
model_choice = st.selectbox(
    "Pilih Model",
    ["LSTM", "Transformer", "Neural ODE"]
)

# ======================================================
# CATEGORY MAPPINGS (UCI OFFICIAL)
# ======================================================
marital_status = {
    "Single": 1,
    "Married": 2,
    "Widower": 3,
    "Divorced": 4,
    "Facto union": 5,
    "Legally separated": 6
}

application_mode = {
    "1st phase - general contingent": 1,
    "Ordinance No. 612/93": 2,
    "1st phase - special contingent (Azores Island)": 5,
    "Holders of other higher courses": 7,
    "Ordinance No. 854-B/99": 10,
    "International student (bachelor)": 15,
    "1st phase - special contingent (Madeira Island)": 16,
    "2nd phase - general contingent": 17,
    "3rd phase - general contingent": 18,
    "Ordinance No. 533-A/99 (Different Plan)": 26,
    "Ordinance No. 533-A/99 (Other Institution)": 27,
    "Over 23 years old": 39,
    "Transfer": 42,
    "Change of course": 43,
    "Technological specialization diploma holders": 44,
    "Change of institution/course": 51,
    "Short cycle diploma holders": 53,
    "Change of institution/course (International)": 57
}

daytime = {"Daytime": 1, "Evening": 0}
yes_no = {"No": 0, "Yes": 1}

# ======================================================
# INPUT FORM
# ======================================================
st.subheader("📝 Data Mahasiswa")

input_data = {}

input_data["Marital_status"] = marital_status[
    st.selectbox("Marital Status", marital_status.keys())
]

input_data["Application_mode"] = application_mode[
    st.selectbox("Application Mode", application_mode.keys())
]

input_data["Daytime_evening_attendance"] = daytime[
    st.selectbox("Attendance", daytime.keys())
]

binary_fields = [
    "Displaced",
    "Educational_special_needs",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Scholarship_holder",
    "International"
]

for field in binary_fields:
    input_data[field] = yes_no[
        st.selectbox(field.replace("_", " "), yes_no.keys())
    ]

# ======================================================
# NUMERIC FEATURES
# ======================================================
numeric_features = [
    col for col in preprocessor.feature_names_in_
    if col not in input_data
]

for col in numeric_features:
    input_data[col] = st.number_input(col, value=0.0)

# ======================================================
# PREDICTION
# ======================================================
input_df = pd.DataFrame([input_data])
X_processed = preprocessor.transform(input_df)

if st.button("🔮 Predict"):
    if model_choice == "LSTM":
        X_model = X_processed.reshape(1, 1, X_processed.shape[1])
        pred = lstm_model.predict(X_model)
        result = np.argmax(pred)

    elif model_choice == "Transformer":
        X_model = X_processed.reshape(1, 1, X_processed.shape[1])
        pred = transformer_model.predict(X_model)
        result = np.argmax(pred)

    else:
        X_torch = torch.tensor(X_processed, dtype=torch.float32)
        with torch.no_grad():
            outputs = neural_ode_model(X_torch)
            result = torch.argmax(outputs, dim=1).item()

    label_map = {
        0: "Dropout",
        1: "Enrolled",
        2: "Graduate"
    }

    st.success(f"🎯 Hasil Prediksi: **{label_map[result]}**")
