import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model import build_model

# 1. Load Data
data = np.load("data/processed/data.npz")
X = data["X"]
y = data["y"]

# 2. Split into Train, Validation, and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# 3. Build Model
input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = build_model(input_shape, num_classes=10)

# 4. Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    batch_size=32, 
                    epochs=30)

# 6. Save the model
model.save("models/genre_classifier.h5")
print("Model saved to models/genre_classifier.h5")