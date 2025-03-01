import tensorflow as tf

# Charger le modèle existant
model_path = "C:/Users/briac/Desktop/projet_3/chat_dog_classification/models/image_classifier.keras"
model = tf.keras.models.load_model(model_path, compile=False)

# Sauvegarde en format HDF5
new_model_path = "C:/Users/briac/Desktop/projet_3/chat_dog_classification/models/image_classifier.h5"
model.save(new_model_path)

print(f"✅ Nouveau modèle sauvegardé en HDF5 : {new_model_path}")
import tensorflow as tf

# Charger le modèle existant
model_path = "C:/Users/briac/Desktop/projet_3/chat_dog_classification/models/audio_classifier.keras"
model = tf.keras.models.load_model(model_path, compile=False)

# Sauvegarde en format HDF5
new_model_path = "C:/Users/briac/Desktop/projet_3/chat_dog_classification/models/audio_classifier.h5"
model.save(new_model_path)

print(f"✅ Nouveau modèle sauvegardé en HDF5 : {new_model_path}")
import tensorflow as tf

# Charger le modèle existant
model_path = "C:/Users/briac/Desktop/projet_3/chat_dog_classification/models/image_audio_fusion_model_v2.keras"
model = tf.keras.models.load_model(model_path, compile=False)

# Sauvegarde en format HDF5
new_model_path = "C:/Users/briac/Desktop/projet_3/chat_dog_classification/models/image_audio_fusion_model_v2.h5"
model.save(new_model_path)

print(f"✅ Nouveau modèle sauvegardé en HDF5 : {new_model_path}")
