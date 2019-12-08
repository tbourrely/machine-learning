import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM

from tb_ml_lib.AudioDataPreprocessor import AudioDataPreprocessor

LOG_DIR = "./log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DATA_SOURCE = 'audio_data.txt'
MODEL_SAVE_PATH = './models/TP2_RNN.hdf5'

tensorboardCallback = TensorBoard(
    log_dir=LOG_DIR, 
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

modelCheckpointCallback = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    period=1
)

audioDataPreprocessor = AudioDataPreprocessor()
audioDataPreprocessor.load(DATA_SOURCE)

NUM_CLASSES = len(audioDataPreprocessor.completeDataset.voyelle.unique())
INPUT_DIMENSION = 7
EPOCHS = 2

ACTIVATION='relu'

MAX_EMBEDDING_VALUE = int(audioDataPreprocessor.training.get('x').max().max()) + 1

rnnModel = Sequential()
rnnModel.add(Embedding(MAX_EMBEDDING_VALUE, NUM_CLASSES))
rnnModel.add(Dropout(0.3))
rnnModel.add(LSTM(512, activation=ACTIVATION, return_sequences=True))
rnnModel.add(Dropout(0.3))
rnnModel.add(LSTM(256, activation=ACTIVATION, return_sequences=True))
rnnModel.add(Dropout(0.3))
rnnModel.add(LSTM(128, activation=ACTIVATION))
rnnModel.add(Dropout(0.3))
rnnModel.add(Dense(NUM_CLASSES, activation='softmax'))

rnnModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
rnnModel.fit(
    audioDataPreprocessor.training.get('x'), 
    audioDataPreprocessor.training.get('y'), 
    epochs=EPOCHS, 
    verbose=1, 
    validation_data=(audioDataPreprocessor.validation.get('x'), audioDataPreprocessor.validation.get('y')),
    callbacks=[tensorboardCallback, modelCheckpointCallback])

loss, acc = rnnModel.evaluate(audioDataPreprocessor.test.get('x'), audioDataPreprocessor.test.get('y'), verbose=1)
print('Test loss:', loss)
print('Test accuracy:', acc)