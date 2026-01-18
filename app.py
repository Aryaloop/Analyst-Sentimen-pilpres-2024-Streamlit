import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from deep_translator import GoogleTranslator

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(page_title="Analisis Sentimen Pilpres", layout="wide")

# Inisialisasi Session State
if 'data_raw' not in st.session_state: st.session_state['data_raw'] = None
if 'data_clean' not in st.session_state: st.session_state['data_clean'] = None
if 'model_trained' not in st.session_state: st.session_state['model_trained'] = None
if 'vectorizer_trained' not in st.session_state: st.session_state['vectorizer_trained'] = None

# ==============================================================================
# 2. FUNGSI LOGIKA
# ==============================================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    
    # Cari kolom teks
    text_col = None
    if 'tweet_normalized' in df.columns: text_col = 'tweet_normalized'
    elif 'text' in cols: text_col = df.columns[cols.index('text')]
    elif 'tweet' in cols: text_col = df.columns[cols.index('tweet')]
    else: text_col = df.columns[0]
    
    # Cari kolom label
    label_col = None
    if 'label' in cols: label_col = df.columns[cols.index('label')]
    elif 'sentiment' in cols: label_col = df.columns[cols.index('sentiment')]
    else: label_col = df.columns[-1]
    
    return text_col, label_col

def balance_data_auto(df, label_col):
    df[label_col] = df[label_col].astype(str).str.strip()
    g = df.groupby(label_col)
    min_samples = g.size().min()
    df_balanced = g.apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)
    return df_balanced

# ==============================================================================
# 3. UI UTAMA
# ==============================================================================
st.title("Pipeline Analisis Sentimen Pilpres (Export/Import Model)")
st.caption("Sistem analisis sentimen berbasis Logistic Regression dengan fitur training ulang.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Upload Data", 
    "2. Preprocessing", 
    "3. Training & Download", 
    "4. Evaluasi", 
    "5. Simulator (Upload .pkl)"
])

# --- TAB 1: UPLOAD DATA ---
with tab1:
    st.header("Upload Dataset CSV")
    uploaded_file = st.file_uploader("Upload CSV Training Data", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['data_raw'] = df
        st.success(f"Data berhasil dimuat: {len(df)} baris.")
        st.dataframe(df.head(3))

# --- TAB 2: PREPROCESSING ---
with tab2:
    st.header("Pembersihan Data Otomatis")
    if st.session_state['data_raw'] is not None:
        df = st.session_state['data_raw'].copy()
        text_col, label_col = detect_columns(df)
        st.info(f"Deteksi Kolom: Teks='{text_col}', Label='{label_col}'")
        
        if st.button("Jalankan Preprocessing"):
            with st.spinner("Menyeimbangkan data 50:50..."):
                df['final_text'] = df[text_col].apply(clean_text)
                df_balanced = balance_data_auto(df, label_col)
                st.session_state['data_clean'] = df_balanced
                
                st.success("Selesai! Data seimbang.")
                st.write(df_balanced[label_col].value_counts())
    else:
        st.warning("Upload data terlebih dahulu di Tab 1.")

# --- TAB 3: TRAINING & DOWNLOAD ---
with tab3:
    st.header("Training Model")
    st.markdown("Note: Mapping label otomatis (Positive=1, Negative=0).")
    
    if st.session_state['data_clean'] is not None:
        if st.button("Mulai Training"):
            with st.spinner("Melatih model..."):
                df_train = st.session_state['data_clean']
                _, label_col = detect_columns(df_train)
                
                # Fixed Mapping
                y_raw = df_train[label_col].str.lower()
                y = y_raw.map({'positive': 1, 'negative': 0})
                if y.isnull().any(): y = y_raw.astype('category').cat.codes
                
                # Vectorizer & Model
                vectorizer = TfidfVectorizer(max_features=5000)
                X = vectorizer.fit_transform(df_train['final_text'])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = LogisticRegression()
                model.fit(X_train, y_train)
                
                # Save to Session
                st.session_state['model_trained'] = model
                st.session_state['vectorizer_trained'] = vectorizer
                st.session_state['test_data'] = (X_test, y_test)
                
                st.success("Training Selesai!")
                
        # --- FITUR DOWNLOAD MODEL ---
        if st.session_state['model_trained']:
            st.divider()
            st.subheader("Unduh Model Anda")
            st.markdown("Download kedua file ini untuk digunakan di halaman Simulator (Tab 5).")
            
            c_down1, c_down2 = st.columns(2)
            
            # Persiapan File Model
            model_buffer = io.BytesIO()
            pickle.dump(st.session_state['model_trained'], model_buffer)
            model_bytes = model_buffer.getvalue()
            
            # Persiapan File Vectorizer
            vec_buffer = io.BytesIO()
            pickle.dump(st.session_state['vectorizer_trained'], vec_buffer)
            vec_bytes = vec_buffer.getvalue()
            
            with c_down1:
                st.download_button(
                    label="Download Model (.pkl)",
                    data=model_bytes,
                    file_name="model_sentiment.pkl",
                    mime="application/octet-stream"
                )
                
            with c_down2:
                st.download_button(
                    label="Download Vectorizer (.pkl)",
                    data=vec_bytes,
                    file_name="tfidf_vectorizer.pkl",
                    mime="application/octet-stream"
                )
            
    else:
        st.warning("Lakukan Preprocessing dahulu di Tab 2.")

# --- TAB 4: EVALUASI ---
with tab4:
    st.header("Evaluasi Model")
    if st.session_state['model_trained']:
        model = st.session_state['model_trained']
        X_test, y_test = st.session_state['test_data']
        y_pred = model.predict(X_test)
        
        st.metric("Akurasi", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
        
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Lakukan Training dahulu di Tab 3.")

# --- TAB 5: SIMULATOR (UPLOAD MANUAL) ---
with tab5:
    st.header("Uji Coba Sentimen")
    
    # --- BAGIAN INPUT MODEL MANUAL ---
    with st.expander("Opsi: Upload Model Manual (.pkl)", expanded=True):
        st.caption("Jika Anda memiliki file model (.pkl), silakan upload di sini.")
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_model_pkl = st.file_uploader("Upload Model File", type="pkl")
        with col_up2:
            uploaded_vec_pkl = st.file_uploader("Upload Vectorizer File", type="pkl")

    # --- LOGIKA PENENTUAN MODEL ---
    active_model = None
    active_vec = None
    source_info = ""

    # Prioritas 1: File Upload User
    if uploaded_model_pkl and uploaded_vec_pkl:
        try:
            active_model = pickle.load(uploaded_model_pkl)
            active_vec = pickle.load(uploaded_vec_pkl)
            source_info = "Status: Menggunakan Model dari File Upload"
        except Exception as e:
            st.error(f"Error memuat file: {e}")

    # Prioritas 2: Model Training Sesi Ini
    elif st.session_state['model_trained'] and st.session_state['vectorizer_trained']:
        active_model = st.session_state['model_trained']
        active_vec = st.session_state['vectorizer_trained']
        source_info = "Status: Menggunakan Model Hasil Training (Sesi Ini)"

    # Prioritas 3: Model Default (Bawaan)
    else:
        try:
            with open('model_sentiment_pilpres.pkl', 'rb') as f: active_model = pickle.load(f)
            with open('tfidf_vectorizer.pkl', 'rb') as f: active_vec = pickle.load(f)
            source_info = "Status: Menggunakan Model Default (Server)"
        except:
            source_info = "Status: Belum ada model tersedia."

    st.info(source_info)

    # --- INTERFACE PREDIKSI ---
    if active_model and active_vec:
        txt = st.text_area("Masukkan Kalimat:", placeholder="Contoh: Kinerja sangat memuaskan!")
        
        if st.button("Prediksi Sentimen"):
            if txt:
                # 1. Translate
                try:
                    trans = GoogleTranslator(source='auto', target='en').translate(txt)
                except:
                    trans = txt 
                
                # 2. Clean & Vectorize
                clean = clean_text(trans)
                vec_txt = active_vec.transform([clean])
                
                # 3. Predict
                pred = active_model.predict(vec_txt)[0]
                proba = active_model.predict_proba(vec_txt)[0]
                
                # 4. Result
                st.divider()
                if pred == 1:
                    st.success(f"SENTIMEN: POSITIF ({proba[1]*100:.1f}%)")
                else:
                    st.error(f"SENTIMEN: NEGATIF ({proba[0]*100:.1f}%)")
                
                st.caption(f"Terjemahan: {trans}")
                
                # Chart
                st.bar_chart(pd.DataFrame({'Label':['Neg', 'Pos'], 'Score':proba}).set_index('Label'))
            else:
                st.warning("Mohon isi teks terlebih dahulu.")
st.caption("Arya Abdul Mughni - 20221310006 - UAS Deep Learning")