#pip install tensorflow matplotlib


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Custom callback for cleaner logs
class ClearLogsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Accuracy = {logs['accuracy']:.4f}, Loss = {logs['loss']:.4f}")

# Load the Reuters dataset
vocab_size = 10000  # Top 10,000 words
maxlen = 200  # Maximum sequence length

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform input length
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# Convert labels to one-hot encoding
num_classes = max(train_labels) + 1
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# Build the model
model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=128),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with custom callback
history = model.fit(
    train_data,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_data=(test_data, test_labels),
    verbose=0,  # Suppress dynamic progress bar
    callbacks=[ClearLogsCallback()]  # Use custom callback for logs
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
