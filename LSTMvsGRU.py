import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

max_features = 5000
maxlen = 200
batch_size = 32
epochs = 3
history_dict = {}

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

lstm_model = Sequential()
lstm_model.add(Embedding(max_features, 16))
lstm_model.add(LSTM(16))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("LSTM Training")
hist_lstm = lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)
history_dict['LSTM'] = hist_lstm.history

gru_model = Sequential()
gru_model.add(Embedding(max_features, 16))
gru_model.add(GRU(16))
gru_model.add(Dense(1, activation='sigmoid'))
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("GRU Training")
hist_gru = gru_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)
history_dict['GRU'] = hist_gru.history

loss_lstm, acc_lstm = lstm_model.evaluate(x_test, y_test, verbose=0)
loss_gru, acc_gru = gru_model.evaluate(x_test, y_test, verbose=0)

print(f"LSTM Test Loss: {loss_lstm:.4f}")
print(f"LSTM Test Accuracy: {acc_lstm:.4f}")
print(f"GRU Test Loss: {loss_gru:.4f}")
print(f"GRU Test Accuracy: {acc_gru:.4f}")

plt.figure(figsize=(6,3))
plt.plot(history_dict['LSTM']['loss'], label='LSTM Train Loss')
plt.plot(history_dict['LSTM']['val_loss'], label='LSTM Val Loss')
plt.plot(history_dict['GRU']['loss'], label='GRU Train Loss')
plt.plot(history_dict['GRU']['val_loss'], label='GRU Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
