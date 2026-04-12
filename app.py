import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit_authenticator as stauth
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

# --- 1. إعداد الصفحة والربط السحابي ---
st.set_page_config(page_title="Radar Signal Cloud Intelligence", layout="wide")

# إنشاء اتصال مع Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

def get_all_users():
    try:
        df = conn.read(ttl=0)
        creds = {"usernames": {}}
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            u_name = str(row['Username']).strip()
            if u_name and u_name != 'nan':
                creds["usernames"][u_name] = {
                    "name": str(row['Name']),
                    "password": str(row['Password']),
                    "email": str(row.get('Email', ''))
                }
        return creds
    except Exception:
        return {"usernames": {}}

credentials = get_all_users()

# نظام الحماية المحدث
authenticator = stauth.Authenticate(
    credentials,
    "radar_intelligence_cookie",
    "auth_signature_key_2026",
    cookie_expiry_days=30
)

# --- 2. واجهة الدخول والتسجيل ---
if not st.session_state.get('authentication_status'):
    tab1, tab2 = st.tabs(["Login", "Register New Engineer"])
    
    with tab2:
        try:
            if authenticator.register_user(location='main'):
                all_users_dict = st.session_state['config']['credentials']['usernames']
                new_username = list(all_users_dict.keys())[-1]
                user_info = all_users_dict[new_username]
                new_entry = pd.DataFrame([{
                    'Name': user_info.get('name', ''),
                    'Last name': 'Engineer',
                    'Email': user_info.get('email', 'N/A'),
                    'Username': new_username,
                    'Password': user_info.get('password', ''),
                    'Password confirmation': user_info.get('password', ''),
                    'Password hint': 'Radar Project',
                    'Captcha': 'Verified'
                }])
                existing_df = conn.read(ttl=0).dropna(how='all')
                updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
                conn.update(data=updated_df)
                st.success('✅ Cloud Sync Success! Now go to Login tab.')
        except Exception:
            st.info("Please fill the registration form to access the system.")

    with tab1:
        # حل مشكلة الـ TypeError
        authenticator.login(location='main')
        if st.session_state.get('authentication_status') is False:
            st.error('Username/password is incorrect')
        elif st.session_state.get('authentication_status') is None:
            st.warning('Please enter your credentials')

# --- 3. صفحة الرادار الرئيسية ---
if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'sidebar')
    
    with st.sidebar:
        st.success(f"Engineer: {st.session_state.get('name', 'User')}")
        st.markdown("---")
        st.subheader("🌐 Cloud Infrastructure")
        st.info("**Platform:** Streamlit Cloud (PaaS)")
        st.info("**Virtualization:** Docker Container")
        st.info("**VDI Ready:** FusionAccess Compatible")

    st.title("📡 Radar Signal Cloud Intelligence")
    st.markdown("---")

    def get_spec(signal):
        _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Signal Generation")
        signal_option = st.selectbox("Select Modulation Type:", ["AM Signal", "FM Signal"])
        gen_btn = st.button("Generate & Classify 🚀")

    if gen_btn:
        model = load_my_model()
        fs = 5000
        t = np.linspace(0, 1, fs, endpoint=False)
        
        if signal_option == "AM Signal":
            signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
        else:
            signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
        
        with col_res:
            # 1. Time Domain
            st.subheader("1. Time Domain (Waveform)")
            fig1, ax1 = plt.subplots(figsize=(10, 3))
            ax1.plot(t[:500], signal[:500], color='dodgerblue')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

            with st.spinner('Running AI Analysis...'):
                spec = get_spec(signal)
                prediction = model.predict(spec.reshape(1, 129, 38, 1))
                res_label = ['AM', 'FM'][np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                # 2. Spectrogram (تصحيح المسافات بالكامل)
                st.subheader("2. Spectrogram (Signal Fingerprint)")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                img = ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                fig2.colorbar(img, ax=ax2, label='Normalized Intensity')
                st.pyplot(fig2)

                # 3. Intelligence Results
                st.subheader("3. Intelligence Results")
                c1, c2 = st.columns(2)
                c1.metric("Detected Modulation", res_label)
                c2.metric("Confidence Score", f"{confidence:.2f}%")  