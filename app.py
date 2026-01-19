import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from torchdiffeq import odeint

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

    Marital Status: 1 – single 2 – married 3 – widower 4 – divorced 5 – facto union 6 – legally separated

    Application Mode: 1 - 1st phase - general contingent 2 - Ordinance No. 612/93 5 - 1st phase - special contingent (Azores Island) 7 - Holders of other higher courses 10 - Ordinance No. 854-B/99 15 - International student (bachelor) 16 - 1st phase - special contingent (Madeira Island) 17 - 2nd phase - general contingent 18 - 3rd phase - general contingent 26 - Ordinance No. 533-A/99, item b2) (Different Plan) 27 - Ordinance No. 533-A/99, item b3 (Other Institution) 39 - Over 23 years old 42 - Transfer 43 - Change of course 44 - Technological specialization diploma holders 51 - Change of institution/course 53 - Short cycle diploma holders 57 - Change of institution/course (International)
    
    Daytime/evening attendance: 1 – daytime 0 - evening

    Previous qualification: 1 - Secondary education 2 - Higher education - bachelor's degree 3 - Higher education - degree 4 - Higher education - master's 5 - Higher education - doctorate 6 - Frequency of higher education 9 - 12th year of schooling - not completed 10 - 11th year of schooling - not completed 12 - Other - 11th year of schooling 14 - 10th year of schooling 15 - 10th year of schooling - not completed 19 - Basic education 3rd cycle (9th/10th/11th year) or equiv. 38 - Basic education 2nd cycle (6th/7th/8th year) or equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 42 - Professional higher technical course 43 - Higher education - master (2nd cycle)

    Nacionality: 1 - Portuguese; 2 - German; 6 - Spanish; 11 - Italian; 13 - Dutch; 14 - English; 17 - Lithuanian; 21 - Angolan; 22 - Cape Verdean; 24 - Guinean; 25 - Mozambican; 26 - Santomean; 32 - Turkish; 41 - Brazilian; 62 - Romanian; 100 - Moldova (Republic of); 101 - Mexican; 103 - Ukrainian; 105 - Russian; 108 - Cuban; 109 - Colombian

    Mother/Father's qualifications: 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 22 - Technical-professional course 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle)

    Mother/Father's occupations: 0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 122 - Health professionals 123 - teachers 125 - Specialists in information and communication technologies (ICT) 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 171 - Skilled construction workers and the like, except electricians 173 - Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like 175 - Workers in food processing, woodworking, clothing and other industries and crafts 191 - cleaning workers 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants

    Displaced, Educational special needs, Debtor, Tuition fees up to date, Scholarship holder, International: 1 – yes 0 – no

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
