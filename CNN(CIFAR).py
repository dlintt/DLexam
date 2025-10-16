import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

NUM_EPOCHS = 3
BATCH_SIZE = 32
TRAIN_SAMPLES = 5000  
TEST_SAMPLES = 1000   

print("Loading a subset of the CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, y_train = x_train[:TRAIN_SAMPLES], y_train[:TRAIN_SAMPLES]
x_test, y_test = x_test[:TEST_SAMPLES], y_test[:TEST_SAMPLES]

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("Building the lightweight CNN model...")
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

print("Compiling the model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(f"Starting model training for {NUM_EPOCHS} epochs...")
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=(x_test, y_test))
print("Model training complete.")

print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

print("\nDisplaying plots...")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

