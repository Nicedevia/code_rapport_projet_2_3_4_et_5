import tensorflow as tf

model_path = "C:/Users/briac/Desktop/projet_3/chat_dog_classification/models/image_classifier.keras"
model = tf.keras.models.load_model(model_path)

print(model.summary())  # Vérifie la structure du modèle
