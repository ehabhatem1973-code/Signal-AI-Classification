import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split

# --- 1. دوال التوليد (AM & FM) ---
def generate_am(fs=5000):
    t = np.linspace(0, 1, fs, endpoint=False)
    return np.sin(2 * np.pi * 100 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))

def generate_fm(fs=5000):
    t = np.linspace(0, 1, fs, endpoint=False)
    return np.sin(2 * np.pi * (100 * t + 20 * (np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs)))

def get_spec(signal):
    _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

# --- 2. تجهيز الـ Dataset (400 عينة) ---
X, y = [], []
print("Preparing Dataset... 🚧")
for _ in range(200):
    X.append(get_spec(generate_am() + np.random.normal(0, 0.1, 5000)))
    y.append(0)
    X.append(get_spec(generate_fm() + np.random.normal(0, 0.1, 5000)))
    y.append(1)

# تحويل القائمة لمصفوفة NumPy
X = np.array(X)

# طباعة الأبعاد قبل الـ Reshape عشان تتأكد
print(f"Current X shape: {X.shape}") 

# عمل الـ Reshape بطريقة مرنة
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y = np.array(y)

print(f"New X shape: {X.shape} 🔥✅")

# --- 3. بناء الـ CNN Model ---
# 1. تحويل y لمصفوفة NumPy (لو مكنتش عملتها)
y = np.array(y)

# 2. تقسيم البيانات (السطر اللي كان ناقص)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. بناء الموديل (تأكد إن الـ input_shape مطابق للواقع اللي هو 129x38)
model = models.Sequential([
    layers.Input(shape=(129, 38, 1)), # التعديل ده هيشيل الـ UserWarning اللي ظهرلك
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. التدريب (دلوقتي X_train بقت معرفة)
print("Training started... 🔥")
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. التدريب (The Big Moment)
#print("Training started... 🔥")
#model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# --- 5. حفظ الموديل النهائي ---
model.save("signal_cnn_model.h5")
# --- 6. تقرير الأداء النهائي (Confusion Matrix) ---
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Generating Confusion Matrix... 📊")

# 1. التوقع على كل بيانات الاختبار مرة واحدة
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 2. حساب المصفوفة (بتقارن y_test الحقيقية بـ y_pred_classes اللي الموديل توقعها)
cm = confusion_matrix(y_test, y_pred_classes)

# 3. رسمها بشكل احترافي
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['AM (0)', 'FM (1)'], 
            yticklabels=['AM (0)', 'FM (1)'])

plt.xlabel('Predicted ')
plt.ylabel('Actual ')
plt.title('Signal Classification Report 📋')
plt.show()
print("CNN Model Saved! 🏆✅")
