import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from torchdiffeq import odeint

# ===============================
# LOAD MODELS & PREPROCESSOR
# ===============================
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

lstm_model = load_model("lstm_model.keras")
transformer_model = load_model("transformer_model.keras")

# ===============================
# NEURAL ODE MODEL
# ===============================
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

# ===============================
# UI HEADER
# ===============================
st.title("🎓 Prediksi Performa Belajar Mahasiswa")

st.markdown("""
Aplikasi ini membandingkan **LSTM, Transformer, dan Neural ODE**
dalam memprediksi status akademik mahasiswa.
""")


# ===============================
# MODEL CHOICE
# ===============================
model_choice = st.selectbox(
    "Pilih Model",
    ["LSTM", "Transformer", "Neural ODE"]
)

# ===============================
# CATEGORY MAPPING
# ===============================
marital_status = {
    "Single": 1,
    "Married": 2,
    "Widower": 3,
    "Divorced": 4,
    "Facto Union": 5,
    "Legally Separated": 6
}

daytime = {
    "Daytime": 1,
    "Evening": 0
}

yes_no = {
    "No": 0,
    "Yes": 1
}

# ===============================
# INPUT FORM
# ===============================
st.subheader("📝 Data Mahasiswa")

input_data = {}

# Dropdown categorical
input_data["Marital_status"] = marital_status[
    st.selectbox("Marital Status", marital_status.keys())
]

input_data["Daytime_evening_attendance"] = daytime[
    st.selectbox("Attendance", daytime.keys())
]

input_data["Displaced"] = yes_no[
    st.selectbox("Displaced", yes_no.keys())
]

input_data["Educational_special_needs"] = yes_no[
    st.selectbox("Educational Special Needs", yes_no.keys())
]

input_data["Debtor"] = yes_no[
    st.selectbox("Debtor", yes_no.keys())
]

input_data["Tuition_fees_up_to_date"] = yes_no[
    st.selectbox("Tuition Fees Up To Date", yes_no.keys())
]

input_data["Scholarship_holder"] = yes_no[
    st.selectbox("Scholarship Holder", yes_no.keys())
]

input_data["International"] = yes_no[
    st.selectbox("International Student", yes_no.keys())
]

# Numeric inputs
numeric_features = [
    col for col in preprocessor.feature_names_in_
    if col not in input_data
]

for col in numeric_features:
    input_data[col] = st.number_input(col, value=0.0)

# ===============================
# PREDICTION
# ===============================
input_df = pd.DataFrame([input_data])
X_processed = preprocessor.transform(input_df)

if st.button("🔮 Predict"):
    if model_choice == "LSTM":
        X_lstm = X_processed.reshape(1, 1, X_processed.shape[1])
        pred = lstm_model.predict(X_lstm)
        result = np.argmax(pred)

    elif model_choice == "Transformer":
        X_tr = X_processed.reshape(1, 1, X_processed.shape[1])
        pred = transformer_model.predict(X_tr)
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
