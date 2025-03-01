import os
import datetime
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import cv2
from sklearn.model_selection import train_test_split

# 📂 Définition des chemins
spectrogram_folder = "data/audio/spectrograms"
categories = ["cats", "dogs"]

# 📂 Dossier de logs pour TensorBoard
log_dir = "logs/audio_training/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 🔄 Chargement des spectrogrammes
spectrogram_data = []
labels = []

for category in categories:
    label = 0 if category == "cats" else 1
    folder_path = os.path.join(spectrogram_folder, category)

    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            file_path = os.path.join(folder_path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64)) / 255.0
            spectrogram_data.append(img)
            labels.append(label)

# 🏷 Conversion en tenseurs
X = np.array(spectrogram_data).reshape(-1, 64, 64, 1)
y = np.array(labels)

# 🔄 Séparation Train/Validation/Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 📦 Modèle CNN pour audio
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 🏋️ Entraînement du modèle avec TensorBoard
history = model.fit(
    X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=8,
    callbacks=[tensorboard_callback]
)

# 🔬 Évaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🎯 Test Accuracy: {test_acc}")

# 💾 Sauvegarde du modèle
model.save("models/audio_classifier.keras")
print("✅ Modèle audio entraîné et sauvegardé !")
