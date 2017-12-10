import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import pickle




"""
********************************************************************************
Extract HOG features
********************************************************************************
"""
def get_hog_features(img, orient, pix_per_cell, cell_per_block):

    hog_features = hog(
                        img,
                        orientations    = orient,
                        pixels_per_cell = (pix_per_cell, pix_per_cell),
                        cells_per_block = (cell_per_block, cell_per_block),
                        transform_sqrt  = False,
                        visualise       = False,
                        feature_vector  = False
                       )

    return hog_features


"""
********************************************************************************
Color histogram features
********************************************************************************
"""
def color_hist(img, nbins=32, bins_range=(0, 256)):

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(( channel1_hist[0], channel2_hist[0], channel3_hist[0] ))
    hist_features = hist_features/8192 
    # Return the individual histograms, bin_centers and feature vector

    return hist_features



"""
********************************************************************************
Binned color features
********************************************************************************
"""
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


"""
********************************************************************************
Extract features by in a list of images
********************************************************************************
"""
def extract_features(
                    imgs,
                    cspace          = 'RGB',
                    spatial_size    = (32, 32),
                    hist_bins       = 32,
                    hist_range      = (0, 256),
                    orient          = 9,
                    pix_per_cell    = 8,
                    cell_per_block  = 2,
                    tam = 0
                    ):
    # Create a list to append feature vectors to
    features = []

    #
    """
    Iterate through the list of images for file in imgs:
    ____________________________________________________
    """
    rand_state = np.random.randint(10, 100)
    img_shuffle = shuffle(imgs, random_state=rand_state)
    endRange = len(img_shuffle)
    i = tam
    for idx in range(0, tam):

        imagePath = img_shuffle[idx]
        # Read in each one by one
        imageA_ = cv2.imread(imagePath, cv2.IMREAD_COLOR)  # uint8 image
        #imageA_ = mpimg.imread(imagePath)
        imageB = imageA_
        imageA = cv2.normalize(imageA_, imageB, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = cv2.resize(imageA, (32,64))
        
        path_ = "image_" + str(i) + ".jpg"
        #cv2.imwrite(path_  ,image)
        i += 1
        # apply color conversion if other than 'RGB'
        #____________________________________________________
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(image)

        # Apply bin_spatial() to get spatial color features
        #____________________________________________________
        spatial_features = bin_spatial( feature_image, size = spatial_size )


        # Apply color_hist() also with a color space option now
        #____________________________________________________
        hist_features = color_hist( feature_image,
                                    nbins = hist_bins,
                                    bins_range = hist_range
                                   )


        # Apply ExtractHOG() to get HOG features
        #____________________________________________________
        HOG_feature_H = get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block).ravel()
        HOG_feature_S = get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block).ravel()
        HOG_feature_V = get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block).ravel()

        # Append the new feature vector to the features list
        #____________________________________________________
        features.append( np.concatenate(( spatial_features,
                                          hist_features,
                                          HOG_feature_H,
                                          HOG_feature_S,
                                          HOG_feature_V
                                         )))

    # Return list of feature vectors
    return features




"""
********************************************************************************
Extract features by class
********************************************************************************
"""
def featuresByClasses(  classA,
                        classB,
                        cspace         = 'RGB',
                        spatial_size   = (32, 32),
                        hist_bins      = 32,
                        hist_range     = (0, 256),
                        orient         = 9,
                        pix_per_cell   = 8,
                        cell_per_block = 2,
                        classATrainSet = 0,
                        classBTrainSet = 0
                      ):

    print('Extracting Class A...')
    classA_features = extract_features( classA,
                                        cspace,
                                        spatial_size,
                                        hist_bins,
                                        hist_range,
                                        orient,
                                        pix_per_cell,
                                        cell_per_block,
                                        classATrainSet
                                       )


    print('Extracting Class B...')
    classB_features = extract_features( classB,
                                        cspace,
                                        spatial_size,
                                        hist_bins,
                                        hist_range,
                                        orient,
                                        pix_per_cell,
                                        cell_per_block,
                                        classBTrainSet
                                       )

    X = np.vstack((classA_features, classB_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    with open('./models/scaler_detector.pkl', 'wb') as f:
        pickle.dump(X_scaler, f)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    return scaled_X
