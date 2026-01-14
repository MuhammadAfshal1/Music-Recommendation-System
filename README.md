import numpy as np
import pandas as pd
# Reproducibility
np.random.seed(42)

NUM_USERS = 10_000
NUM_SONGS = 5_000
NUM_SAMPLES = 200_000

data = {
    "user_id": np.random.randint(0, NUM_USERS, NUM_SAMPLES),
    "song_id": np.random.randint(0, NUM_SONGS, NUM_SAMPLES),
    "play_count": np.random.poisson(lam=3, size=NUM_SAMPLES),
    "time_gap_days": np.random.randint(0, 30, NUM_SAMPLES),
    "song_popularity": np.random.rand(NUM_SAMPLES),
}

df = pd.DataFrame(data)

# Label: repeated play within a month
df["repeated_play"] = (
    (df["play_count"] > 2) &
    (df["time_gap_days"] < 7)
).astype(int)

df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
features = ["play_count", "time_gap_days", "song_popularity"]
X = df[features].values
y = df["repeated_play"].values
# Normalize numerical features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  # Binary classification
])
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=1024,
    validation_split=0.1,
    verbose=1
)
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test AUC: {test_auc:.3f}")
sample = np.array([[5, 2, 0.85]])  # play_count, time_gap_days, popularity
sample = scaler.transform(sample)

prob = model.predict(sample)[0][0]
print(f"Probability of repeated listening: {prob:.2f}")


import matplotlib.pyplot as plt #to visualize

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss Curves")
plt.legend()
plt.show()

plt.figure() #to  visualize the accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()
