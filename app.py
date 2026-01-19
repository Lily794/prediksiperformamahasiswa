import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from torchdiffeq import odeint

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

lstm_model = load_model("lstm_model.keras")
transformer_model = load_model("transformer_model.keras")

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
    pd.DataFrame([preprocessor.feature_names_in_])
).shape[1]

neural_ode_model = NeuralODE(INPUT_DIM, 3)
neural_ode_model.load_state_dict(
    torch.load("neural_ode_model.pt", map_location="cpu")
)
neural_ode_model.eval()

st.title("Prediksi Performa Belajar Mahasiswa")

model_choice = st.selectbox(
    "Pilih Model",
    ["LSTM", "Transformer", "Neural ODE"]
)

input_data = {}

st.subheader("Masukkan Data Mahasiswa")

for col in preprocessor.feature_names_in_:
    input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

X_processed = preprocessor.transform(input_df)

if st.button("Predict"):
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

st.success(f"Hasil Prediksi: **{label_map[result]}**")
