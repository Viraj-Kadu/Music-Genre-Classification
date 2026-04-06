import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):
    """
    Builds a Deep CNN for Music Genre Classification.
    """
    model = models.Sequential()

    # 1st Conv Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())

    # 2nd Conv Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())

    # 3rd Conv Block
    model.add(layers.Conv2D(128, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3)) # Prevent overfitting

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    # Test building the model
    model = build_model((128, 130, 1), 10)
    model.summary()