import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram 
# ضفنا دي عشان نحول الإشارة لصورة

# 1. دالة الـ Spectrogram (لازم تكون نسخة طبق الأصل من كود التدريب)
def get_spec(signal):
    _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    # عمل Normalization (مهم جداً عشان الموديل يفهم الأرقام)
    return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

# تحميل الموديل
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('signal_cnn_model.h5')

model = load_my_model()

st.title("📡 Radar Signal Intelligence")

# ... (كود توليد الإشارة اللي عندك زي ما هو) ...

if st.button("Generate & Classify 🚀"):
    # (كود توليد الـ signal هنا)
    fs = 5000
    t = np.linspace(0, 1, fs, endpoint=False)
    if signal_type == "AM Signal":
        signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
    else:
        signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))

    with st.spinner('Converting signal to Spectrogram and Analyzing...'):
        try:
            # الخطوة السحرية: تحويل الإشارة لـ Spectrogram بنفس مقاس التدريب
            spec = get_spec(signal) 
            
            # التأكد من الأبعاد (1, 129, 38, 1)
            input_data = spec.reshape(1, 129, 38, 1)
            
            # التوقع
            prediction = model.predict(input_data)
            classes = ['AM', 'FM']
            res = classes[np.argmax(prediction)]
            conf = np.max(prediction) * 100

            st.success(f"### Prediction: {res}")
            st.info(f"### Confidence: {conf:.2f}%")
            
            # رسم الـ Spectrogram عشان الشغل يبقى احترافي
            fig, ax = plt.subplots()
            ax.imshow(spec, aspect='auto', origin='lower')
            ax.set_title("Spectrogram used for Prediction")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error logic: {e}")