import datetime
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.backend import clear_session

from tb_ml_lib.ModelConfigReader import ModelConfigReader
from tb_ml_lib.AudioDataPreprocessor import AudioDataPreprocessor


LOG_DIR = "./log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DATA_SOURCE = 'audio_data.txt'

BEST_CONFIG = None
BEST_CONFIG_LOSS = 0
BEST_CONFIG_ACC = 0

modelConfigReader = ModelConfigReader()
modelConfigReader.load("model_config.json")

for modelConfig in modelConfigReader.configs:
    clear_session()

    audioDataPreprocessor = AudioDataPreprocessor()
    audioDataPreprocessor.load(DATA_SOURCE)

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

    NUM_CLASSES = len(audioDataPreprocessor.completeDataset.voyelle.unique())
    INPUT_DIMENSION = 7
    EPOCHS = 1

    print("-----------------------------------------")
    print(modelConfig)

    ACTIVATION = modelConfig.activation_function
    INPUT_ACTIVATION = modelConfig.input_activation_function
    OUTPUT_ACTIVATION = modelConfig.output_activation_function

    model = Sequential()
    model.add(Dense(NUM_CLASSES, activation=INPUT_ACTIVATION, input_dim=INPUT_DIMENSION))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation=ACTIVATION))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation=ACTIVATION))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=ACTIVATION))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION))

    model.compile(loss=modelConfig.loss, optimizer=modelConfig.optimizer, metrics=["accuracy"])

    model.fit(
        audioDataPreprocessor.training.get('x'), 
        audioDataPreprocessor.training.get('y'), 
        epochs=EPOCHS, 
        verbose=1, 
        validation_data=(audioDataPreprocessor.validation.get('x'), audioDataPreprocessor.validation.get('y')),
        callbacks=[tensorboardCallback])

    loss, acc = model.evaluate(audioDataPreprocessor.test.get('x'), audioDataPreprocessor.test.get('y'), verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    print("-----------------------------------------")

    if (acc > BEST_CONFIG_ACC):
        model.save('./models/TP2_FEEDFORWARD.hdf5')
        BEST_CONFIG = modelConfig
        BEST_CONFIG_LOSS = loss
        BEST_CONFIG_ACC = acc


print("--------------------RESULTS---------------------")
print(BEST_CONFIG)
print('Test loss:', BEST_CONFIG_LOSS)
print('Test accuracy:', BEST_CONFIG_ACC)
print("------------------------------------------------")
