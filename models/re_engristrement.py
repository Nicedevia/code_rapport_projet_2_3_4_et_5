import os
import tensorflow as tf

class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs:
            batch_shape = kwargs.pop("batch_shape")
            kwargs["batch_input_shape"] = tuple(batch_shape)
        super().__init__(*args, **kwargs)


IMAGE_MODEL_PATH = "models/image.keras"
AUDIO_MODEL_PATH = "models/audio.keras"


image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, custom_objects={"InputLayer": CustomInputLayer})
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects={"InputLayer": CustomInputLayer})


dummy_image = tf.zeros((1, 64, 64, 1))
dummy_audio = tf.zeros((1, 64, 64, 1))
_ = image_model(dummy_image)
_ = audio_model(dummy_audio)

image_model.save("image.h5")
audio_model.save("audio.h5")

print("Modèles re-sauvegardés en .h5 avec succès !")
