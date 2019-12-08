from os import path
import pandas
import sys
from sklearn.preprocessing import LabelBinarizer

class AudioDataPreprocessor:
    def __init__(self):
        self.completeDataset = None
        self.training = None
        self.validation = None
        self.test = None

    def quitIfFileNotExist(self, file):
        if not path.exists(file):
            print(file + ' does not exists')
            sys.exit(1)

    # Return one-hot encoded labels (y) and raw data (x)
    def dataframeToXandY(self, dataframe):
        encoder = LabelBinarizer()
        y = dataframe['voyelle']
        transformedY = encoder.fit_transform(y)
        x = dataframe.drop(columns=['voyelle'])

        return (x, transformedY)

    def load(self, filename):
        self.quitIfFileNotExist(filename)

        completeDataset = pandas.read_csv(
            filename,
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
        x_train, y_train = self.dataframeToXandY(training_sample)

        # Update splitting source dataset
        splittingSourceDataset = splittingSourceDataset.drop(x_train.index)

        # Split what's left in 2 
        # 100% - 80% = 20% / 2 = 10% of total dataset
        validation_sample = splittingSourceDataset.sample(frac=0.5, random_state=0)
        x_validation, y_validation = self.dataframeToXandY(validation_sample)

        test_sample = splittingSourceDataset.drop(validation_sample.index)
        x_test, y_test = self.dataframeToXandY(test_sample)

        self.completeDataset = completeDataset
        self.training = {'x': x_train, 'y': y_train}
        self.validation = {'x': x_validation, 'y': y_validation}
        self.test = {'x': x_test, 'y': y_test}



    
