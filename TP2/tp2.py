import pandas
from os import path
import sys, datetime
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer

from tb_ml_lib import ModelConfigReader


LOG_DIR = "./log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def quitIfFileNotExist(file):
    if not path.exists(file):
        print(file + ' does not exists')
        sys.exit(1)

def dataframeToXandY(dataframe):
    encoder = LabelBinarizer()
    y = dataframe['voyelle']
    transformedY = encoder.fit_transform(y)
    x = dataframe.drop(columns=['voyelle'])

    return (x, transformedY)

dataSource = 'audio_data.txt'

quitIfFileNotExist(dataSource)

completeDataset = pandas.read_csv(
    dataSource,
    sep='\t',
    usecols=['voyelle', 'F1', 'F2', 'F3', 'F4', 'Z1', 'Z2', 'f0']
)

completeDataset.dropna() # remove NaN rows
completeDataset = completeDataset[completeDataset['F4'] != '--undefined--'] # remove rows with F4 equal to --undefined--

# Types casting
completeDataset[['F1', 'F2', 'F3', 'F4', 'f0']] = completeDataset[['F1', 'F2', 'F3', 'F4', 'f0']].astype(float)

splittingSourceDataset = completeDataset.copy()

# Use 80% as training
training_sample = splittingSourceDataset.sample(frac=0.8, random_state=0)
x_train, y_train = dataframeToXandY(training_sample)

# Update splitting source dataset
splittingSourceDataset = splittingSourceDataset.drop(x_train.index)

# Split what's left in 2 
# 100% - 80% = 20% / 2 = 10% of total dataset
validation_sample = splittingSourceDataset.sample(frac=0.5, random_state=0)
x_validation, y_validation = dataframeToXandY(validation_sample)

test_sample = splittingSourceDataset.drop(validation_sample.index)
x_test, y_test = dataframeToXandY(test_sample)

print(x_train)
print(y_train)
print(len(y_train))
print(x_validation)
print(y_validation)
print(len(y_validation))
print(x_test)
print(y_test)
print(len(y_test))

print(completeDataset.voyelle.unique())
print(len(completeDataset.voyelle.unique()))

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


NUM_CLASSES = len(completeDataset.voyelle.unique())
INPUT_DIMENSION = 7
EPOCHS = 2

modelConfigReader = ModelConfigReader.ModelConfigReader()
modelConfigReader.load("model_config.json")

BEST_CONFIG = None
BEST_CONFIG_LOSS = 0
BEST_CONFIG_ACC = 0

for modelConfig in modelConfigReader.configs:
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
        x_train, 
        y_train, 
        epochs=EPOCHS, 
        verbose=0, 
        validation_data=(x_validation, y_validation),
        callbacks=[tensorboardCallback])

    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    print("-----------------------------------------")

    if (acc > BEST_CONFIG_ACC):
        BEST_CONFIG = modelConfig
        BEST_CONFIG_LOSS = loss
        BEST_CONFIG_ACC = acc


print("--------------------RESULTS---------------------")
print(BEST_CONFIG)
print('Test loss:', BEST_CONFIG_LOSS)
print('Test accuracy:', BEST_CONFIG_ACC)
print("------------------------------------------------")
