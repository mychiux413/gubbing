from gubbing.models.networks.encoders import ImageEncoder, AudioEncoder
import tensorflow as tf
from tensorflow.keras import backend

channels_last = backend.image_data_format() == "channels_last"

def test_image_encoder():
    enc = ImageEncoder("input")
    if channels_last:
        imgs = tf.random.normal((2, 256, 256, 3))
    else:
        imgs = tf.random.normal((2, 3, 256, 256))
    features = enc(imgs)
    if channels_last:
        assert features.shape == (2, 1, 1, 512)
    else:
        assert features.shape == (2, 512, 1, 1)

def test_audio_encoder():
    enc = AudioEncoder()
    if channels_last:
        audio = tf.random.normal((2, 24, 64))
    else:
        audio = tf.random.normal((2, 64, 24))
    features = enc(audio)

    if channels_last:
        assert features.shape == (2, 1, 512)
    else:
        assert features.shape == (2, 512, 1)