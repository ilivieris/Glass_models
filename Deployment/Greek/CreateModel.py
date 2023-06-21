# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Libraries
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Basic libraries
#
import platform
import sys
import numpy as np
import time


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Sklearn library
#
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import *
from sklearn.datasets        import load_iris
from sklearn.model_selection import train_test_split

print("[INFO] Platform: ", platform.platform())
print("[INFO] Python version: ", sys.version)

print('[INFO] Libraries imported')




class Iris_Dataset():
    def __init__( self, train_size = 0.8, verbose = True, random_state = 42):

        self.train_size   = train_size
        self.verbose      = verbose
        self.random_state = random_state


        # Load dataset
        #
        self._dataset = load_iris()

        # Select inputs
        self._X = np.array( self._dataset.data )
        # Select targets
        self._Y = np.array( self._dataset.target )
        # 
        self.target_names = self._dataset.target_names


    def getData( self ):

        trainX, testX, trainY, testY = train_test_split(self._X, self._Y, train_size = self.train_size, random_state = self.random_state)

        # Training instances
        self.trainX = trainX
        self.trainY = trainY

        # Testing instances
        self.testX = testX
        self.testY = testY

        return trainX, trainY, testX, testY

# Create Dataset
dataset = Iris_Dataset( train_size = 0.9 )

# Training/Testing data created
trainX, trainY, testX, testY = dataset.getData()
print('[INFO] Training/Testing data created')

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Prediction model
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

print(50*'-')
print('[INFO] Create prediction model')
# Setup model
#
model = RandomForestClassifier(n_jobs = -1, random_state = 42)

print('[INFO] Forecasting model is created')




# Start time
#
start = time.time()

# Train model
#
model.fit(trainX, trainY)

print('[INFO] Model trained')
print('[INFO] Time: %.2fs' % (time.time() - start))


# # Save trained model
# #
# filename = 'model/forecasting_model.sav'
# pickle.dump(model, open(filename, 'wb'))


# Save trained model
#
import joblib
import gzip
filename = 'model/forecasting_model.dat.gz'
joblib.dump(model, gzip.open(filename, "wb"))
print('[INFO] Path: ', filename)
print()




# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Model evaluation
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Get predictions
#
pred = model.predict( testX )

# Evaluation performance
#
print(50*'-')
print('[INFO] Model evaluation\n')
#
# Get predictions
pred = model.predict( testX )
# Classification report
print( classification_report(y_pred = pred, y_true = testY, target_names = dataset.target_names) )