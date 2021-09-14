import tensorflow as tf
from gubbing.models.gubbing import Embedder
from gubbing.utils import is_channels_last


def test_gubbing():
    emb = Embedder()
    if is_channels_last:
        refs = tf.random.normal((2, 256, 256, 3))
        imgs = tf.random.normal((2, 256, 256, 3))
        audio = tf.random.normal((2, 24, 64))
    else:
        refs = tf.random.normal((2, 3, 256, 256))
        imgs = tf.random.normal((2, 3, 256, 256))
        audio = tf.random.normal((2, 64, 24))
    outputs = emb({
        "images": imgs,
        "references": refs,
        "audio": audio,
    })
    embeddings = outputs["embeddings"]
    if is_channels_last:
        assert embeddings.shape == (2, 1, 1, 1536)
    else:
        assert embeddings.shape == (2, 1536, 1, 1)
