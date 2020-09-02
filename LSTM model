
import os, numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

data = pd.read_csv('dataset.csv', encoding = 'utf8')
print('The number of data: ', len(data))

categories = ['1', '2', '3']
nb_classes = len(categories)

data['type'] = data['type'].replace(['protein', 'RNA', 'sequence'], [1, 2, 3])
data[:5]

data.info()
data['type'].value_counts().plot(kind='bar')
print(data.groupby('type').size().reset_index(name='count'))

# x와 y 분리
X_data = data['text']
y_data = data['type']
print('The number of text: {}'.format(len(X_data)))
print('The number of type_label: {}'.format(len(y_data)))

y_data = np_utils.to_categorical(y_data, nb_classes + 1)  # one-hot vector 변형
print(y_data)

max_word = 2000
tok = Tokenizer(num_words = max_word)
tok.fit_on_texts(X_data)
sequences = tok.texts_to_sequences(X_data)
print(len(sequences[0]))
print(sequences[0])

X_data = sequences
print('max_len : %d' % max(len(l) for l in X_data))
max_len = 2936
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
print(sequences_matrix)
print(sequences_matrix[0])
print(len(sequences_matrix[0]))

print(len(tok.word_index))

X_train, X_test, y_train, y_test = train_test_split(sequences_matrix, y_data, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

model = Sequential()

model.add(Embedding(max_word, 64, input_length=max_len))
model.add(LSTM(120))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4, activation='softmax'))

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
checkpoint = ModelCheckpoint('best_model.h5', monitor="val_loss", mode='max', verbose=1, save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, batch_size=256, epochs=20, callbacks=[checkpoint, early_stopping], validation_split=0.2)

print("accuracy : %.4f" % (model.evaluate(X_test, y_test)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()
