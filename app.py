import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Judul Aplikasi
# =========================
st.title("📊 Prediksi Status Gizi Anak")
st.write("Aplikasi Machine Learning untuk memprediksi status gizi anak berdasarkan data kesehatan")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("malnutrition_children_ethiopia.csv")

df = load_data()

st.subheader("📂 Data Awal")
st.dataframe(df.head())

# Tampilkan kolom dataset
st.write("📋 Kolom tersedia di dataset:", list(df.columns))

# =========================
# Cari kolom target secara fleksibel
# =========================
target_col = None
for col in df.columns:
    if "status" in col.lower():  # cari kolom yang mengandung kata 'status'
        target_col = col
        break

if target_col is None:
    st.error("❌ Tidak ditemukan kolom target 'Status'. Periksa dataset Anda.")
else:
    # =========================
    # Exploratory Data Analysis
    # =========================
    st.subheader("📊 Distribusi Status Gizi")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=target_col, ax=ax)
    st.pyplot(fig)

    # =========================
    # Preprocessing
    # =========================
    X = df.drop(columns=[target_col])  # fitur
    y = df[target_col]                 # target

    # Ubah kolom kategorikal jadi numerik
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # Model Training
    # =========================
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # =========================
    # Evaluasi
    # =========================
    y_pred = model.predict(X_test)
    st.subheader("📈 Evaluasi Model")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    fig2, ax2 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt="d", cmap="Blues", ax=ax2)
    st.pyplot(fig2)

    # =========================
    # Prediksi Manual
    # =========================
    st.subheader("🔮 Prediksi Status Gizi Anak Baru")

    # Hanya gunakan kolom utama dari dataset
    umur = st.number_input("Umur Anak (bulan)", 0, 60, 24)
    berat = st.number_input("Berat Badan (kg)", 0.0, 30.0, 10.0)
    tinggi = st.number_input("Tinggi Badan (cm)", 40.0, 120.0, 80.0)

    # Buat dataframe input user
    if st.button("Prediksi"):
        input_data = pd.DataFrame(
            [[umur, berat, tinggi]],
            columns=["Age (months)", "Weight_kg", "Height_cm"]
        )

        # Samakan kolom input dengan kolom model
        for col in X_train.columns:
            if col not in input_data.columns:
                input_data[col] = 0  # default 0 untuk kolom yang tidak ada

        # Susun ulang kolom sesuai urutan training
        input_data = input_data[X_train.columns]

        # Prediksi
        hasil = model.predict(input_data)[0]
        st.success(f"✅ Status gizi anak diprediksi: **{hasil}**")
