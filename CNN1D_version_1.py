# author: houda.najeh@imt-atlantique.fr
""
" Import librairies "
""
import math
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
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import std

""
" Model definition "
""
def model(x_test, y_test, x_train, y_train):
    n_timestamps = x_train.shape[1]
    n_features = x_train.shape[2]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_timestamps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print('--- Train phase ---')
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1), metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10)
    # history = model.fit(x=x_train, y=y_train, epochs=10)
    # print('history=', history.history.keys())
    # # summarize history for accuracy
    # plt.subplot(2, 1, 1)
    # plt.plot(history.history['accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # # summarize history for loss
    # plt.subplot(2, 1, 2)
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    print('--- Test phase ---')
    accuracy = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
    print('accuracy=', accuracy)
    return accuracy

""
"summarize scores"
""
def summarize_results(scores):
    print('scores=', scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

""
"run an experiment"
""
def run_experiment(trainX, trainy, testX, testy, repeats=10):
    # load data
    # trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = model(trainX, trainy, testX, testy)
        score = np.array(score) * 100.0
        print('score=', score)
        #print('>#%d: %.3f' % (r + 1, score))
        print('>#%d'% (r + 1))
        print('%.3f' % (score[0]))
        scores.append(score)
    # summarize results
    summarize_results(scores)


""
" define train sequences "
""
train_sequences = [[1, 1, 1, 1, 1, 1, 1, 1], [2, 2],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2], [8, 8, 8, 8, 8, 8, 8, 8, 8], [4], [8], [5], [4], [8, 3], [8], [3, 3, 3, 3, 3, 3, 3], [8], [4], [8], [5], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [8], [7], [8, 8, 8, 8, 8, 8, 8], [2], [8], [5], [7, 7], [6, 6, 6, 6], [8, 8], [5], [4], [3, 3], [8, 8, 8], [3], [4], [8, 8], [5, 5], [8], [2, 2], [8], [6, 6], [8, 3], [4], [8], [3, 3, 3, 3, 3],[8], [3, 3], [8], [4], [8], [4], [8], [4, 4], [8], [4], [3], [8, 8], [5, 5], [8, 8, 8, 8, 8], [5, 5], [7], [5, 5], [11], [10, 10], [5, 5, 5], [8, 8, 8, 8], [6, 6, 6], [8], [5], [8], [4], [8], [4], [8], [7, 7, 7, 7, 7, 7, 7], [6, 6, 6, 6], [7, 7], [8, 8], [4, 4], [8, 8, 8, 8], [6, 6, 6], [5, 5, 5], [7],[8], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [8], [4], [8], [4], [5], [8, 8], [5, 5, 5], [4], [8, 8],[4], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [8, 8], [7, 7, 7, 7], [6], [8],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [8], [5],[8, 4], [8, 8, 8, 8, 8, 8, 8], [5, 5, 5, 5], [6, 6, 6], [8], [1], [8, 8, 8, 8, 8], [4], [3], [4],[8], [3], [8], [3, 3], [8], [3, 3, 3], [4], [8], [5, 5, 5], [4], [8, 8, 8], [4], [5], [8], [7],[8, 8], [5, 5], [8], [2, 2, 2, 2, 2, 2], [8], [5, 5], [8, 8, 8], [1], [8, 8, 8], [6, 6], [5],[7, 7, 7], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [8, 8, 8, 8], [6, 6, 6, 6], [5, 5, 5], [8, 8, 8, 8, 8, 8, 8, 8], [11], [10, 10], [11, 11], [10, 10], [11], [8], [11], [10], [11], [10], [8, 8], [4], [8], [3, 3], [4], [7], [6, 6, 6, 6, 6, 6], [8], [2, 2], [8], [5], [8], [5], [6], [5, 5], [8, 8], [2, 2], [8, 8, 8], [6, 6, 6, 6, 6, 6, 6], [7], [6, 6], [5], [8, 8, 8], [4], [3], [4], [5],[8], [5], [8], [4, 4], [8, 8], [4], [7, 7], [6, 6], [5], [7, 7], [6, 6], [5], [8, 8, 8], [5, 5], [8, 8], [3], [4],[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [4], [3], [8], [5], [7], [6, 6], [7, 7, 7],[6, 6, 6, 6, 6, 6, 6, 6], [7, 7, 7], [6, 6, 6, 6, 6, 6, 6, 6]]
# print('train_sequences=', train_sequences)
# print('len_train_sequences=', np.shape(train_sequences))
""
" pad sequence "
""
train_padded = pad_sequences(train_sequences)
#("train_padded =", train_padded)
train_x = np.expand_dims(train_padded, axis=0)
#print('Shape of X is ', train_x.shape)  # (1, 3, 4)

train_y = np.array([1, 2, 1, 2, 11, 11, 11, 11, 11, 4, 11, 11, 11, 11, 9, 10, 11, 11, 6, 11, 11, 11, 9, 9, 10, 6, 11, 11, 11, 11, 3, 11, 2, 11, 9, 10, 11, 11, 11, 4, 11, 11, 11, 11, 11, 11, 11, 11, 3, 3, 11, 3, 11, 8, 3, 9, 11, 11, 11, 11, 11, 11, 11, 9, 9, 10, 9, 3, 11, 11, 3, 11, 11, 11, 11, 11, 3, 11, 11, 3, 9, 11, 11, 1, 11, 11, 6, 3, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 3, 11, 6, 11, 11, 11, 11, 11, 11, 11, 9, 11, 9, 3, 9, 3, 11, 7, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 9, 10, 11, 2, 11, 11, 11, 11, 11, 2, 9, 10, 11, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 6, 3, 11, 11, 4, 11, 11, 11, 11, 11, 9, 9, 10, 9, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 2, 11, 9, 10, 11, 3, 11, 11, 3, 11])
train_y = train_y.reshape(1, -1)

#('Shape of train_y is', train_y.shape)  # (1, 3)

""
" define test sequences "
""
test_sequences = [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,3, 3, 3, 3, 3], [5], [8], [4], [8], [4, 4, 4], [8, 8], [4], [8], [4], [3, 3, 3], [8], [3], [4], [2, 2], [8],[6, 6, 6, 6, 6], [7, 7], [6], [5, 5, 5, 5], [8], [7], [5, 5, 5], [8, 8], [4], [8], [4], [8, 8], [6, 6, 6], [7],[8, 8, 8, 8, 8, 8], [5], [7, 7, 7], [6, 6, 6, 6, 6, 6], [7, 7, 7, 7], [11], [10, 10], [5, 5, 5],[6, 6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [5], [8, 4], [8, 8, 8, 8, 8], [5, 5], [6, 6], [5],[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [6, 6, 6, 6], [7], [6, 6], [7, 7],[6, 6, 6, 6, 6, 6], [7], [6], [5], [7], [5], [8], [4], [3, 3, 3, 3, 3], [8], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [8],[4], [5], [7], [6, 6], [7], [5, 5], [8, 8], [7], [6], [7, 7], [6], [7], [6, 6], [7], [6], [7],[6, 6, 6, 6, 6, 6, 6, 6], [7], [6, 6, 6, 6], [7, 7, 7], [5], [7, 7, 7, 7], [5, 5], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],[11], [10, 10], [6, 6, 6, 6, 6], [5], [8], [4, 4], [8, 4], [8], [4], [8, 8], [5], [6, 6, 6], [7],[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7, 7],[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7, 7, 7], [6, 6], [7, 7, 7], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7, 7], [6, 6], [7, 7, 7, 7], [6, 6, 6], [5], [8], [4], [3, 3], [8], [3], [4], [5], [6, 6, 6, 6], [5], [8, 8], [4], [8], [3], [8], [3], [8], [4], [5, 5, 5, 5, 5, 5, 5], [6, 6], [7], [6, 6, 6, 6], [7, 7], [6, 6], [7, 7], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [5, 5], [7], [6, 6, 6, 6], [5], [7], [6, 6, 6, 6, 6, 6],[7, 7], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [7], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7, 7, 7], [6, 6, 6, 6, 6], [7], [6, 6], [7], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [7, 7, 7], [6], [5], [8], [4], [8, 8, 8, 8], [4], [6, 6, 6, 6, 6], [9, 9, 9], [5], [6, 6], [7, 7, 7, 7], [6, 6, 6, 6, 6, 6, 6], [7, 7], [6, 6, 6, 6, 6], [8, 8], [5, 5], [8], [6, 6], [7], [5], [8, 8], [1, 1, 1, 1, 1], [8, 8], [5], [8], [5], [8, 8], [5], [7], [8], [2, 2], [8, 8], [1, 1, 1, 1, 1], [8], [6, 6, 6, 6, 6, 6], [7, 7, 7], [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [5, 5, 5, 5, 5, 5, 5, 5], [7, 7], [5, 5], [6, 6], [7], [8, 8, 8, 8, 8, 8], [2, 2, 2, 2, 2], [1], [8], [5], [8, 8]]
test_padded = pad_sequences(test_sequences)
test_padded = test_padded[0:208]
test_x = np.expand_dims(test_padded, axis=0)
# test_x = np.expand_dims(test_x, axis=-1)
test_y = np.array([11, 11, 11, 6, 9, 11, 11, 9, 9, 10, 9, 11, 7, 8, 3, 9, 10, 6, 11, 9, 11, 5, 9, 11, 9, 9, 9, 10, 11, 11, 11, 11, 11,11, 11, 4, 11, 4, 11, 11, 11, 11, 9, 11, 3, 11, 11, 9, 11, 11, 11, 11, 11, 9, 10, 11, 9, 9, 11, 9, 9, 10, 11, 8, 9,10, 11, 11, 11, 11, 11, 9, 11, 9, 10, 9, 9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 9, 9, 9, 9, 10, 9, 9, 9, 11, 11, 11, 4, 11, 11, 11, 11, 9, 11, 11, 11, 11, 11, 11, 11, 11, 3, 9, 11, 9, 10, 9, 9, 9, 9, 10, 3, 11, 9, 11, 11, 9, 10,9, 9, 10, 5, 11, 9, 10, 9, 9, 10, 11, 9, 11, 9, 10, 9, 11, 11, 11, 11, 11, 9, 5, 11, 9, 9, 9, 10, 9, 10, 3, 11, 9, 10, 11, 11, 6, 11, 11, 11, 11, 11, 11, 2, 6, 1, 11, 9, 10, 9, 9, 10, 3, 9, 9, 10, 11, 2, 11, 11, 11])
test_y = test_y.reshape(1, -1)

res1 = model(train_x, train_y, test_x, test_y)

# run the experiment
#run_experiment(train_x, train_y, test_x, test_y, repeats=10)
