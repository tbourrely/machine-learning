from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks.tensorboard_v1 import TensorBoard # v2 fails to show histograms
from keras.callbacks.callbacks import ModelCheckpoint

import datetime

# Callbacks

logs_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_filepath = "models/best-model.hdf5"

modelCheckpointCallback = ModelCheckpoint(
    model_save_filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    period=1
)

tensorboardCallback = TensorBoard(
    log_dir=logs_dir, 
    histogram_freq=1, 
    batch_size=32, 
    write_graph=True, 
    write_grads=True, 
    write_images=False, 
    embeddings_freq=0, 
    embeddings_layer_names=None, 
    embeddings_metadata=None, 
    embeddings_data=None, 
    update_freq='epoch')


(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_dimension = 784
num_classes = 10
epochs = 12

x_train = x_train.reshape(60000, input_dimension)
x_test = x_test.reshape(10000, input_dimension)

# Normalize data to be between 0-1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(200, activation='relu', input_dim=input_dimension))
model.add(Dropout(0.3))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

model.fit(
    x_train, 
    y_train, 
    epochs=epochs, 
    verbose=1, 
    validation_data=(x_test, y_test),
    callbacks=[tensorboardCallback, modelCheckpointCallback])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))
