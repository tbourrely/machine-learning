import pandas
from os import path

def quitIfFileNotExist(file):
    if not path.exists(file):
        print(file + ' does not exists')
        sys.exit(1)

#============================================================

dataSource = 'audio_data.txt'

quitIfFileNotExist(dataSource)

completeDataset = pandas.read_csv(
    dataSource,
    sep='\t',
    usecols=['voyelle', 'F1', 'F2', 'F3', 'F4', 'Z1', 'Z2', 'f0']
)

splittingSourceDataset = completeDataset.copy()

# Use 80% as training
learningData = splittingSourceDataset.sample(frac=0.8, random_state=0)

# Update splitting source dataset
splittingSourceDataset = splittingSourceDataset.drop(learningData.index)

# Split what's left in 2 
# 100% - 80% = 20% / 2 = 10% of total dataset
validationData = splittingSourceDataset.sample(frac=0.5, random_state=0)
testData = splittingSourceDataset.drop(validationData.index)

print(learningData)
print(validationData)
print(testData)
