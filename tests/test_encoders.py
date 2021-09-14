from gubbing.models.networks.encoders import ImageEncoder, AudioEncoder, ReferenceEncoder
import tensorflow as tf
from gubbing.utils import is_channels_last


def test_image_encoder():
    enc = ImageEncoder()
    if is_channels_last:
        imgs = tf.random.normal((2, 256, 256, 3))
    else:
        imgs = tf.random.normal((2, 3, 256, 256))
    features = enc(imgs)
    if is_channels_last:
        assert features[-1].shape == (2, 1, 1, 512)
    else:
        assert features[-1].shape == (2, 512, 1, 1)


def test_reference_encoder():
    ref = ReferenceEncoder()
    if is_channels_last:
        imgs = tf.random.normal((2, 256, 256, 3))
    else:
        imgs = tf.random.normal((2, 3, 256, 256))
    features = ref(imgs)
    if is_channels_last:
        assert features.shape == (2, 1, 1, 512)
    else:
        assert features.shape == (2, 512, 1, 1)


def test_audio_encoder():
    enc = AudioEncoder()
    if is_channels_last:
        audio = tf.random.normal((2, 24, 64))
    else:
        audio = tf.random.normal((2, 64, 24))
    features = enc(audio)

    if is_channels_last:
        assert features.shape == (2, 1, 1, 512)
    else:
        assert features.shape == (2, 512, 1, 1)
