import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
(xt, yt), (xts, yts) = mnist.load_data()
xt = xt.reshape(60000, 28, 28, 1).astype('float32')/255
xts = xts.reshape(10000, 28, 28, 1).astype('float32')/255
nc = 10
yt = to_categorical(yt, num_classes = nc)
yts = to_categorical(yts, num_classes = nc)
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(nc, activation = 'softmax'))
    return model
optimizers = ['adam', 'adagrad', 'adadelta', 'adamax', 'nadam']
history = {}
for optim in optimizers:
    model = create_model()
    model.compile(optimizer = optim, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history[optim] = model.fit(xt, yt, epochs = 5, batch_size = 64, validation_data = (xts, yts))
for optim in optimizers:
    plt.figure(figsize = (5, 3))
    plt.plot(history[optim].history['accuracy'], label = 'Training Accuracy')
    plt.plot(history[optim].history['val_accuracy'], label = 'Validation Accuracy')
    plt.title(optim)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
plt.show()