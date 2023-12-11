import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.markdown(
    "<h1 style='text-align: center;'>Klasifikasi Penyakit Liver Menggunakan Model Random Forest</h1>", unsafe_allow_html=True
)


# load dataset -------------------------------------------------------------------
dataset = pd.read_csv('dataset_baru.csv')

# split dataset menjadi data training dan data testing ---------------------------
fitur = dataset.drop(columns=['Selector'], axis =1)
target = dataset['Selector']
fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

# normalisasi dataset ------------------------------------------------------------
# memanggil kembali model normalisasi zscore dari file pickle
with open('zscorescaler.pkl', 'rb') as file_normalisasi:
    zscore = pickle.load(file_normalisasi)

zscoretraining = zscore.transform(fitur_train)
zscoretesting = zscore.transform(fitur_test)

# implementasi data pda model
with open('model_rf.pkl', 'rb') as file_model:
    model_rf = pickle.load(file_model)

model_rf.fit(zscoretraining, target_train)
prediksi_target = model_rf.predict(zscoretesting)


Age = st.number_input ('Input Umur anda')

TB = st.number_input ('Input Total Bilirubin anda')

DB = st.number_input ('Input Kadar Direct Bilirubin dalam darah anda')

Sgpt = st.number_input ('Input Alanine Aminotransferase anda')

if st.button('Cek Status'):
    if all(x is not None for x in [Age, TB, DB, Sgpt]):
        st.text('Prediksi : ')
        prediksi = model_rf.predict([[Age, TB, DB, Sgpt]])
        if prediksi == 1.0:
            st.success("Anda diprediksi tidak memiliki penyakit liver !")
        elif prediksi == 2.0:
            st.warning("Anda diprediksi liver !")
    else:
        st.text('Data tidak boleh kosong. Harap isi semua kolom.')