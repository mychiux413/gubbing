import tensorflow as tf
from tensorflow.keras import layers, Model


class ResNetV2Block(Model):

    def __init__(self, filter, downsampled=True, **kwargs):
        super().__init__(**kwargs)
        self.conv0 = layers.Conv2D(
            filter, 3, activation=layers.ReLU(),
            strides=(2, 2) if downsampled else (1, 1),
            padding='same',
        )

        self.norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            filter, 3, activation=layers.ReLU(),
            padding='same',
        )
        self.norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            filter, 3, activation=layers.ReLU(),
            padding='same',
        )

    def call(self, inputs):
        identidy = self.conv0(inputs)
        x = self.norm1(identidy)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        return x + identidy

class ResNetConv1D(Model):
    def __init__(self, filter, downsampled=True, **kwargs):
        super().__init__(**kwargs)
        self.conv0 = layers.Conv1D(
            filter, 5, activation=layers.ReLU(),
            strides=2 if downsampled else 1,
            padding='same',
        )

        self.norm1 = layers.BatchNormalization()
        self.conv1 = layers.Conv1D(
            filter, 5, activation=layers.ReLU(),
            padding='same',
        )
        self.norm2 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(
            filter, 5, activation=layers.ReLU(),
            padding='same',
        )

    def call(self, inputs):
        identidy = self.conv0(inputs)
        x = self.norm1(identidy)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        return x + identidy

class AudioEncoder(Model):
    def __init__(self):
        super().__init__()
        self.res1 = ResNetConv1D(128)
        self.res2 = ResNetConv1D(512)
        self.global_pool = layers.GlobalAveragePooling1D(keepdims=True)

    def call(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.global_pool(x)
        x = tf.expand_dims(x, axis=1)
        return x


class ImageEncoder(Model):
    def __init__(self):
        super().__init__()
        self.res_blocks = [
            ResNetV2Block(16, downsampled=False),
            ResNetV2Block(32),
            ResNetV2Block(64),
            ResNetV2Block(128),
            ResNetV2Block(256),
            ResNetV2Block(512),
            ResNetV2Block(512),
            ResNetV2Block(512),
            ResNetV2Block(512),
        ]

    def call(self, x):
        outputs = [x]
        for block in self.res_blocks:
            x = block(x)
            outputs.append(x)
        return outputs


class ReferenceEncoder(Model):
    def __init__(self):
        super().__init__()
        self.res_blocks = [
            ResNetV2Block(16, downsampled=False),
            ResNetV2Block(32),
            ResNetV2Block(64),
            ResNetV2Block(128),
            ResNetV2Block(256),
            ResNetV2Block(512),
            ResNetV2Block(512),
            ResNetV2Block(512),
            ResNetV2Block(512),
        ]

    def call(self, x):
        for block in self.res_blocks:
            x = block(x)
        return x
