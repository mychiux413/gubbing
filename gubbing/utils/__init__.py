from tensorflow.keras import backend

is_channels_last = backend.image_data_format() == "channels_last"