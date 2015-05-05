import numpy as np
import os
import h5py as h5df

# Generate a HDF5 file from the 14_AUC Data file

print 'Write Train and Test HDF5 File'
dirname = '/home/hmendoza/workspace/prakcaffe/caffedataset/'
dataTrain = np.loadtxt(dirname + '14_auc_train.data')
#dataTrain = dataTrain[:, np.newaxis, np.newaxis,:]
labelsTrainEnc = np.loadtxt(dirname +'14_auc_train.solution')
labelsTrain = np.array(np.argmax(labelsTrainEnc, axis=1), dtype=np.float32)
dataTest = np.loadtxt(dirname +'14_auc_test.data')
#dataTest = dataTest[:, np.newaxis, np.newaxis,:]
labelsTestEnc = np.loadtxt(dirname +'14_auc_test.solution')
labelsTest = np.array(np.argmax(labelsTestEnc, axis=1), dtype=np.float32)

fTrainName = 'trainData.h5'
fTestName = 'testData.h5'
fTrain = h5df.File(dirname + fTrainName, 'w')
fTest = h5df.File(dirname + fTestName, 'w')

#f.create_dataset('data', data=data)
#f.create_dataset('label', data=labels)

fTrain['data'] = dataTrain
fTrain['label'] = labelsTrain
fTest['data'] = dataTest
fTest['label'] = labelsTest

fTrain.close()
fTest.close()

with open(os.path.join(dirname, 'train.txt'), 'w') as f:
    f.write(dirname + fTrainName + '\n') 

with open(os.path.join(dirname, 'test.txt'), 'w') as f:
    f.write(dirname + fTestName + '\n') 
