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
# LOAD PREPROCESSOR & SCHEMA
# ===============================
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("schema.pkl", "rb") as f:
    schema = pickle.load(f)

categorical_columns = schema["categorical_columns"]
categorical_mappings = schema["categorical_mappings"]

# ===============================
# LOAD KERAS MODELS
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
# STREAMLIT UI
# ===============================
st.title("🎓 Prediksi Performa Belajar Mahasiswa")
st.markdown(
    """
    Aplikasi ini membandingkan **LSTM, Transformer, dan Neural ODE**
    dalam memprediksi status akademik mahasiswa.
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
    st.subheader("🧾 Data Mahasiswa
    Marital status = 1 – single 2 – married 3 – widower 4 – divorced 5 – facto union 6 – legally separated")

    input_data = {}

    for col in preprocessor.feature_names_in_:
        label = col.replace("_", " ")

        if col in categorical_columns:
            mapping = categorical_mappings[col]
            selected_label = st.selectbox(
                label,
                list(mapping.values())
            )
            # convert label -> original numeric code
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
