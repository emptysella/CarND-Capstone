import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import time
import pickle
import features_color


"""
Extrator Module
********************************************************************************
"""

# Reading images from folder

ImagesClassA  = glob.glob('/Volumes/Samsung_T5/train/red/*.jpg')
ImagesClassB  = glob.glob('/Volumes/Samsung_T5/train/green/*.jpg')

classATrainSet_ = 260
classBTrainSet_ = 130
# extract features
classes = features_color.featuresByClasses(
                                    ImagesClassA,
                                    ImagesClassB,
                                    cspace         = 'HSV',
                                    spatial_size   = (32,64),
                                    hist_bins      = 128,
                                    hist_range     = (0, 256),
                                    orient         = 9,
                                    pix_per_cell   = 8,
                                    cell_per_block = 2,
                                    classATrainSet = classATrainSet_,
                                    classBTrainSet = classBTrainSet_
                                    )

print('Features Extracted...')



"""
Classifier Module
********************************************************************************
"""

### Definition of the training
#-----------------------------
# Define a labels vector based on features lists
labels = np.hstack((np.ones(classATrainSet_), np.zeros(classBTrainSet_)))

# Define train set
rand_state = np.random.randint(10, 100)
X_train, X_test, y_train, y_test = train_test_split( classes,
                                                     labels,
                                                     test_size = 0.2,
                                                     random_state = rand_state
                                                     )



# Training SVM
#-----------------------------
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)

with open('./models/model_classifier.pkl', 'wb') as f:
    pickle.dump(svc, f)


# Statistics
#-----------------------------
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC

print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
CM = confusion_matrix(y_test, svc.predict(X_test))
print(CM)
