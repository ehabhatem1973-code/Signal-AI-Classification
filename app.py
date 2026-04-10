import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # ضفنا ده وشلنا requests

# تحميل الموديل مرة واحدة في أول تشغيل البرنامج
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('signal_cnn_model.h5')

model = load_my_model()

st.set_page_config(page_title="Signal Intelligence Radar", page_icon="📡")

st.title("📡 Signal Classification Radar")
st.write("Welcome, Engineer! Upload or Generate a signal to classify it using AI.")

# 1. قائمة اختيار نوع الإشارة
signal_type = st.selectbox("Select Signal Type to Generate:", ["AM Signal", "FM Signal"])

if st.button("Generate & Classify 🚀"):
    # توليد إشارة حقيقية (5000 عينة) عشان الموديل ينبهر
    fs = 5000
    t = np.linspace(0, 1, fs, endpoint=False)
    
    if signal_type == "AM Signal":
        signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
    else:
        signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
    
    # رسم الإشارة
    fig, ax = plt.subplots()
    ax.plot(t[:500], signal[:500]) # رسم أول 500 عينة بس للوضوح
    ax.set_title(f"Generated {signal_type} (First 500 samples)")
    st.pyplot(fig)

with st.spinner('Analyzing signal...'):
            try:
                # 1. تأكد من حجم الإشارة (مثلاً لو الموديل متدرب على 1024 عينة)
                # هناخد أول 1024 عينة بس من الإشارة المولدة
                target_size = 1024 # غير الرقم ده للرقم اللي الموديل اتدرب عليه (مهم جداً)
                
                trimmed_signal = signal[:target_size] 
                
                # 2. لو الإشارة أصغر من المطلوب، نزودها أصفار (Padding)
                if len(trimmed_signal) < target_size:
                    trimmed_signal = np.pad(trimmed_signal, (0, target_size - len(trimmed_signal)))

                # 3. التجهيز للموديل (Batch, Length, Channels)
                input_signal = trimmed_signal.reshape(1, target_size, 1)
                
                # 4. التوقع
                prediction_probs = model.predict(input_signal)
                
                classes = ['AM', 'FM'] 
                result = classes[np.argmax(prediction_probs)]
                confidence = np.max(prediction_probs) * 100

                st.success(f"### Prediction: {result}")
                st.info(f"### Confidence: {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"حدث خطأ أثناء التصنيف: {e}")
            #import requests هشيل ديه عشان محتاجش اعمل سرفر تاني 