import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data and Model
data = np.load("data/processed/data.npz")
X_test = data["X"]
y_test = data["y"]
mapping = data["mapping"]

model = tf.keras.models.load_model("models/genre_classifier.h5")

# 2. Get Predictions
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# 3. Build Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=mapping, yticklabels=mapping, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Music Genre Classification')
plt.savefig('models/confusion_matrix.png')
plt.show()