import tensorflow as tf
from tensorflow.keras import layers, Model
from gubbing.models.networks.encoders import ImageEncoder, ReferenceEncoder, AudioEncoder
from gubbing.utils import is_channels_last

class Embedder(Model):
    def __init__(self):
        super().__init__()
        self.img_encoder = ImageEncoder()
        self.ref_encoder = ReferenceEncoder()
        self.audio_encoder = AudioEncoder()

    def call(self, inputs: dict):
        img_feature_blocks = self.img_encoder(inputs['images'])
        ref_embeddings = self.ref_encoder(inputs['references'])
        audio_embeddings = self.audio_encoder(inputs['audio'])
        
        img_embeddings = img_feature_blocks.pop()
        embeddings = tf.concat([
            audio_embeddings,
            ref_embeddings,
            img_embeddings,
        ], axis=3 if is_channels_last else 1)

        return {
            "embeddings": embeddings,
            "img_feature_blocks": img_feature_blocks,
        }
