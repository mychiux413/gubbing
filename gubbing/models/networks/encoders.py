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
        return x


class ImageEncoder(Model):
    def __init__(self, name: str):
        super().__init__()
        self.res_blocks = [
            ResNetV2Block(16, downsampled=False, name=f"{name}_res1"),
            ResNetV2Block(32, name=f"{name}_res2"),
            ResNetV2Block(64, name=f"{name}_res3"),
            ResNetV2Block(128, name=f"{name}_res4"),
            ResNetV2Block(256, name=f"{name}_res5"),
            ResNetV2Block(512, name=f"{name}_res6"),
            ResNetV2Block(512, name=f"{name}_res7"),
            ResNetV2Block(512, name=f"{name}_res8"),
            ResNetV2Block(512, name=f"{name}_res9"),
        ]

    def call(self, x):
        for block in self.res_blocks:
            x = block(x)
        return x
