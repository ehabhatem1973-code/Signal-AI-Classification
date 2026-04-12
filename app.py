import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit_authenticator as stauth
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# 1. إعداد الصفحة (لازم يكون أول أمر برمي بعد الـ imports)
st.set_page_config(page_title="Radar Signal Cloud Intelligence", layout="wide")

# 2. كود إخفاء أيقونة GitHub فقط مع الحفاظ على القائمة (تعديل القناص)
st.markdown("""
    <style>
    /* إخفاء زر الجيت هب والقطة وأي أثر لها */
    .stAppDeployButton, #StyledgithubIcon, [data-testid="bundle_github_cursor_detector"] {
        display: none !important;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. إعداد الاتصال بالجوجل شيت (الطريقة المتوافقة مع نسختك)
url_sheet = "https://docs.google.com/spreadsheets/d/13kcl0WS0LE1rXWm4aanpby8wO5542JaR76038ofa1-E/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)


def get_all_users():
    try:
        # قراءة البيانات مع تمرير الرابط داخل الـ read لتجنب الـ TypeError
        df = conn.read(spreadsheet=url_sheet, ttl=0)
        creds = {"usernames": {}}
        if df is not None and not df.empty:
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

# تحميل البيانات الحالية
credentials = get_all_users()

# نظام الحماية (Authenticator)
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
            # عملية التسجيل
            result = authenticator.register_user(location='main')
            
            if result:
                # التأكد من تحديث القائمة المحلية
                usernames_list = list(credentials["usernames"].keys())
                if usernames_list:
                    new_username = usernames_list[-1]
                    user_info = credentials["usernames"][new_username]

                    # تجهيز البيانات للإرسال للسحابة
                    new_entry = pd.DataFrame([{
                        'Name': user_info.get('name', 'N/A'),
                        'Last name': 'Engineer',
                        'Email': user_info.get('email', 'N/A'),
                        'Username': new_username,
                        'Password': user_info.get('password', ''),
                        'Password confirmation': user_info.get('password', ''),
                        'Password hint': 'Radar Project',
                        'Captcha': 'Verified'
                    }])

                    # رفع البيانات للجوجل شيت مع تمرير الرابط
                    existing_df = conn.read(spreadsheet=url_sheet, ttl=0)
                    updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
                    conn.update(spreadsheet=url_sheet, data=updated_df)
                    
                    st.success('✅ Engineer Registered & Cloud Synced!')
                    st.balloons()
        except Exception as e:
            st.error(f"Error during registration: {e}")

    with tab1:
        authenticator.login(location='main')
        if st.session_state.get('authentication_status') is False:
            st.error('Username/password is incorrect')
        elif st.session_state.get('authentication_status') is None:
            st.info('Please enter your credentials to access the Radar System')

# --- 3. صفحة الرادار (تظهر فقط بعد نجاح الدخول) ---
if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'sidebar')
    
    with st.sidebar:
        st.success(f"Welcome, Eng. {st.session_state.get('name', 'User')}")
        st.markdown("---")
        st.subheader("🌐 System Infrastructure")
        st.info("**Environment:** Docker Container")
        st.info("**Database:** Google Cloud Real-time")

    st.title("📡 Radar Signal Intelligence System")
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
        signal_option = st.selectbox("Select Modulation:", ["AM Signal", "FM Signal"])
        gen_btn = st.button("Generate & Classify 🚀")

    if gen_btn:
        model = load_my_model()
        t = np.linspace(0, 1, 5000, endpoint=False)
        
        if signal_option == "AM Signal":
            signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
        else:
            signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / 5000))
        
        with col_res:
            st.subheader("1. Time Domain")
            fig1, ax1 = plt.subplots(figsize=(10, 3))
            ax1.plot(t[:500], signal[:500], color='dodgerblue')
            st.pyplot(fig1)

            with st.spinner('Analyzing...'):
                spec = get_spec(signal)
                prediction = model.predict(spec.reshape(1, 129, 38, 1))
                res_label = ['AM', 'FM'][np.argmax(prediction)]
                
                st.subheader("2. Spectrogram")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                st.pyplot(fig2)

                st.subheader("3. Intelligence Results")
                c1, c2 = st.columns(2)
                confidence = np.max(prediction) * 100
                c1.metric("Detected Modulation", res_label)
                c2.metric("Confidence Score", f"{confidence:.2f}%")

                