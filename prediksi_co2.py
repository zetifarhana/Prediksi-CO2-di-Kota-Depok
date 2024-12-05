import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

model = pickle.load(open('prediksi_co2.sav', 'rb'))

url = 'https://github.com/adyanamul/dataset/raw/main/CO2_dataset.xlsx'
df = pd.read_excel(url)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index(['Year'], inplace=True)

st.title('Kualitas CO2 di Kota Depok')

# Fungsi untuk halaman Deskripsi
def show_deskripsi():
    st.write("Selamat datang di aplikasi prediksi kualitas CO2.")
    st.write("<div style='text-align: justify;'>Aplikasi ini menggunakan teknologi <i>Machine Learning</i> untuk memberikan prediksi yang akurat terkait kualitas udara di Depok. Dengan memasukkan data terkait emisi CO2 dan faktor lingkungan lainnya, pengguna dapat memprediksi kondisi kualitas udara di masa depan. Model <i>Machine Learning</i> yang digunakan dalam aplikasi ini telah dilatih dengan data historis yang luas dan akurat, memungkinkan sistem memberikan informasi yang berguna dan handal. Aplikasi ini bertujuan untuk membantu masyarakat dan pengambil kebijakan dalam mengidentifikasi potensi polusi udara, serta mengambil tindakan pencegahan yang lebih efektif. Dengan tampilan yang sederhana dan responsif, aplikasi ini dapat membantu dalam upaya peningkatan kualitas lingkungan hidup di kawasan urban.</div>", unsafe_allow_html=True)
    st.write("Sumber data: https://github.com/adyanamul/dataset/raw/main/CO2_dataset.xlsx")

# Fungsi untuk halaman Dataset
def show_dataset():
    st.header("Dataset")
    st.dataframe(df)
    st.markdown("""
( 1 ) **Tahun**
    Kualitas CO2 dimulai dari Tahun 1800.
  \n(
2 ) **CO2(Karbon Dioksida)**
    adalah gas tidak berwarna dan tidak berbau yang secara alami ada di atmosfer Bumi.
  \n
  """)

def show_grafik():
    st.header("Grafik")
    decompose_add = seasonal_decompose(df['CO2'], model='additive')
    fig = decompose_add.plot()
    st.pyplot(fig)
    st.markdown("""
( 1 ) **CO2 (Garis Utama)**
     - Ini adalah data asli yang menunjukkan pengukuran karbon dioksida (CO2) dalam rentang waktu tertentu.
    \n(
2 ) **Trend (Tren)**
    - Grafik ini menunjukkan tren jangka panjang dalam data CO2. Tren mengindikasikan arah umum perubahan nilai CO2 dari waktu ke waktu. Pada grafik ini, tampak ada peningkatan yang konsisten, yang menunjukkan bahwa tingkat CO2 cenderung naik seiring waktu.
\n(
3 ) **Seasonal (Musiman)**
    - Bagian ini menunjukkan pola musiman dalam data. Pola musiman adalah fluktuasi yang terjadi secara periodik pada waktu-waktu tertentu (misalnya, musiman tahunan atau bulanan). Pada grafik ini, tampaknya tidak ada fluktuasi musiman yang signifikan, yang mungkin menunjukkan bahwa data CO2 tidak memiliki pola musiman yang jelas.
\n(
4 ) **Residual (Sisa)**
    - Grafik ini menunjukkan komponen sisa yang didapat setelah tren dan musiman dihapus dari data asli. Komponen sisa ini menunjukkan variasi yang tidak dapat dijelaskan oleh tren dan pola musiman. Pada grafik ini, sisa tampak relatif stabil dan mendekati nol, menunjukkan bahwa tidak ada banyak fluktuasi yang signifikan di luar pola yang telah diidentifikasi.
""")

# Fungsi untuk halaman Prediksi
def show_prediksi():
    st.header("Halaman Prediksi")
    year = st.slider("Tentukan Beberapa Tahun Kedepan", 1, 30, step=1)
    pred = model.forecast(year)
    pred = pd.DataFrame(pred, columns=['CO2'])

    if st.button("Predict"):
        col1, col2 = st.columns([2, 3])
        with col1:
            st.dataframe(pred)
        with col2:
            fig, ax = plt.subplots()
            df['CO2'].plot(style='--', color='gray', legend=True, label='known')
            pred['CO2'].plot(color='b', legend=True, label='Prediction')
            st.pyplot(fig)

# Pilih menu di sidebar
add_selectbox = st.sidebar.selectbox(
    "PILIH MENU",
    ("Deskripsi", "Dataset", "Grafik", "Prediksi")
)

if add_selectbox == "Deskripsi":
    show_deskripsi()
elif add_selectbox == "Dataset":
    show_dataset()
elif add_selectbox == "Grafik":
    show_grafik()
elif add_selectbox == "Prediksi":
    show_prediksi()
