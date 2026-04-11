import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit_authenticator as stauth
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

# --- 1. إعداد الاتصال بـ Google Sheets ---
# تأكد من وضع [connections.gsheets] في الـ Secrets كما فعلنا
conn = st.connection("gsheets", type=GSheetsConnection)

def get_all_users():
    # قراءة البيانات من الشيت (الأعمدة: Name, User Name, Password)
    df = conn.read(ttl=0) 
    creds = {"usernames": {}}
    for _, row in df.iterrows():
        u_name = str(row['User Name'])
        creds["usernames"][u_name] = {
            "name": str(row['Name']),
            "password": str(row['Password'])
        }
    return creds

# تحميل اليوزرز من الشيت عند بداية التشغيل
try:
    credentials = get_all_users()
except Exception as e:
    st.error("Connection Error: Check your Google Sheet Secrets!")
    credentials = {"usernames": {}}

# تهيئة الـ Authenticator
authenticator = stauth.Authenticate(
    credentials,
    "radar_dashboard",
    "auth_key",
    cookie_expiry_days=30
)

# --- 2. التحكم في واجهة المستخدم (Login/Register) ---
login_placeholder = st.empty()

# إذا لم يكن المستخدم مسجلاً دخوله بعد
if not st.session_state.get('authentication_status'):
    with login_placeholder.container():
        # --- نموذج التسجيل المعدل لتجنب الأخطاء ---
        try:
            if authenticator.register_user(location='main'):
                # بدلاً من استخدام ['config']، هنستخدم البيانات المتاحة في الـ session_state مباشرة
                if 'credentials' in st.session_state:
                    all_users = st.session_state['credentials']['usernames']
                else:
                    # حل احتياطي: استخدام القائمة الحالية المضافة إليها اليوزر الجديد
                    all_users = credentials['usernames']
                
                # تحويل البيانات لـ DataFrame ورفعها للشيت
                new_df = pd.DataFrame.from_dict(all_users, orient='index').reset_index()
                new_df.columns = ['User Name', 'Name', 'Password']
                
                # رفع البيانات للجوجل شيت (تأكد أن الصلاحية Editor)
                conn.update(data=new_df)
                
                st.success('Registration successful! Please login now.')
                st.rerun() 
        except Exception as e:
            # معالجة خطأ الـ config المشهور في المكتبة
            if "config" in str(e):
                st.info("System is initializing context... Please wait or refresh.")
            else:
                st.error(f"Sign-up Error: {e}")
        
        # نموذج الدخول التقليدي
        result = authenticator.login(location='main')

# فك تشفير حالة الدخول
if isinstance(st.session_state.get('authentication_status'), bool):
    authentication_status = st.session_state['authentication_status']
    name = st.session_state.get('name')
    username = st.session_state.get('username')
else:
    authentication_status = None

# --- 3. صفحة الرادار (تظهر فقط بعد نجاح الدخول) ---
if authentication_status:
    login_placeholder.empty() # مسح شاشة الدخول تماماً
    
    # واجهة الخروج والترحيب
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f'Welcome Engineer *{name}*')

    # محتوى الصفحة الرئيسي
    st.title("📡 Radar Signal Intelligence")
    st.markdown("---")
    st.write("Current Status: **Secure Cloud Access Granted**")

    # دالة الـ Spectrogram
    def get_spec(signal):
        _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    # تحميل الموديل (Caching لتحسين السرعة)
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    model = load_my_model()

    signal_option = st.selectbox("Select Signal Type to Generate:", ["AM Signal", "FM Signal"])

    if st.button("Generate & Classify 🚀"):
        fs = 5000
        t = np.linspace(0, 1, fs, endpoint=False)
        
        if signal_option == "AM Signal":
            signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
        else:
            signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
        
        # 1. رسم الموجة الزمنية
        st.subheader("1. Time Domain Representation")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(t[:500], signal[:500], color='dodgerblue')
        ax1.set_title(f"Waveform: {signal_option}")
        st.pyplot(fig1)

        with st.spinner('AI analyzing signal fingerprint...'):
            try:
                spec = get_spec(signal)
                input_data = spec.reshape(1, 129, 38, 1)
                prediction = model.predict(input_data)
                classes = ['AM', 'FM']
                res = classes[np.argmax(prediction)]
                conf = np.max(prediction) * 100

                # 2. عرض نتائج التحليل
                st.subheader("2. Intelligence Analysis")
                c1, c2 = st.columns(2)
                c1.metric("Predicted Class", res)
                c2.metric("Confidence Level", f"{conf:.2f}%")

                # 3. رسم الـ Spectrogram
                st.subheader("3. Spectrogram (Signal Fingerprint)")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                ax2.set_ylabel("Frequency Bin")
                ax2.set_xlabel("Time Bin")
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Analysis Error: {e}")

elif authentication_status == False:
    st.error('Username/password is incorrect')