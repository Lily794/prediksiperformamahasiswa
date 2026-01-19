import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from torchdiffeq import odeint

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Prediksi Performa Mahasiswa",
    layout="centered"
)

# ===============================
# LOAD PREPROCESSOR
# ===============================
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# ===============================
# LOAD MODELS
# ===============================
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


INPUT_DIM = 36  # hasil preprocessing
neural_ode_model = NeuralODE(INPUT_DIM, 3)
neural_ode_model.load_state_dict(
    torch.load("neural_ode_model.pt", map_location="cpu")
)
neural_ode_model.eval()

# ===============================
# CATEGORY MAPPINGS (UCI DATASET)
# ===============================
categorical_mappings = {
    "Marital_status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto Union",
        6: "Legally Separated"
    },
    "Application_mode": {
        1: "General Contingent",
        2: "Ordinance 612/93",
        5: "Special Contingent (Azores)",
        7: "Other Higher Course Holder",
        10: "Ordinance 854-B/99",
        15: "International Student"
    },
    "Course": {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service",
        9003: "Nursing",
        9070: "Management"
    },
    "Gender": {
        0: "Female",
        1: "Male"
    },
    "Scholarship_holder": {
        0: "No",
        1: "Yes"
    },
    "Debtor": {
        0: "No",
        1: "Yes"
    },
    "Tuition_fees_up_to_date": {
        0: "No",
        1: "Yes"
    }
}

# ===============================
# STREAMLIT UI
# ===============================
st.title("🎓 Prediksi Performa Belajar Mahasiswa")
st.markdown(
    """
    Aplikasi ini membandingkan **LSTM, Transformer, dan Neural ODE**
    untuk memprediksi status akademik mahasiswa.
    """
)

model_choice = st.selectbox(
    "🤖 Pilih Model",
    ["LSTM", "Transformer", "Neural ODE"]
)

# ===============================
# INPUT FORM
# ===============================
with st.form("student_form"):
    st.subheader("🧾 Data Mahasiswa")

    input_data = {}

    for col in preprocessor.feature_names_in_:
        label = col.replace("_", " ")

        if col in categorical_mappings:
            mapping = categorical_mappings[col]
            selected_label = st.selectbox(
                label,
                list(mapping.values())
            )
            input_data[col] = [
                k for k, v in mapping.items()
                if v == selected_label
            ][0]
        else:
            input_data[col] = st.number_input(
                label,
                value=0.0
            )

    submitted = st.form_submit_button("🔮 Predict")

# ===============================
# PREDICTION
# ===============================
if submitted:
    input_df = pd.DataFrame([input_data])
    X_processed = preprocessor.transform(input_df)

    if model_choice == "LSTM":
        X_lstm = X_processed.reshape(1, 1, X_processed.shape[1])
        pred = lstm_model.predict(X_lstm)
        result = np.argmax(pred, axis=1)[0]

    elif model_choice == "Transformer":
        X_tr = X_processed.reshape(1, 1, X_processed.shape[1])
        pred = transformer_model.predict(X_tr)
        result = np.argmax(pred, axis=1)[0]

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

    st.success(f"📊 Hasil Prediksi: **{label_map[result]}**")
