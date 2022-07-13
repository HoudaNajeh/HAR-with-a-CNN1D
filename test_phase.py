# author: houda.najeh@imt-atlantique.fr
""
" Import librairies "
""
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
""
" Model definition "
""
def model(X, y):
	n_timestamps = X.shape[1]
	n_features = X.shape[2]
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_timestamps, n_features)))#input_shape=(3, 4) for the basic example
	model.add(MaxPooling1D(pool_size=1))
	model.add(Flatten())
	model.add(Dense(50, activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer='adam',  metrics=['accuracy'])
	# fit model
	model.compile (
		loss='mean_squared_error',
		optimizer=tf.keras.optimizers.Adam(0.1))
	model.fit(x = X, y = y, epochs=3)
	accuracy = model.evaluate(X, y, batch_size=32, verbose=1)
	print(accuracy)
	return accuracy
""
" pad test sequence "
""
sequences_test = [
       [1, 0, 0, 0, 1, 1, 0],
                [1, 1, 0],
    ]
padded_test = pad_sequences(sequences_test)
print("padded_test =", padded_test)
X_test = np.expand_dims(padded_test, axis = 0)
print('Shape of X_test is ', X_test.shape) # (1, 3, 4)

#y = np.array([1, 0, 1])
y_test = np.array([0, 0])
y_test = y_test.reshape(1,-1)
print('Shape of y_test is', y_test.shape) # (1, 3)
accuracy = model(X_test, y_test)
print(accuracy)