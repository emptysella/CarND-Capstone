import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import math
from scipy.ndimage.measurements import label
import glob
import cv2
import numpy as np
import features

class TLClassifier_SRG(object):

    def __init__(self):
        self.blocksize = 96

        """
        light code:
            1 == green light
            2 == yellow light
            3 == red light
            0 == unknow
        """
        self.light = 0


    """
    ****************************************************************************
    convert_color
    ****************************************************************************
    """

    def convert_color(self, img, conv='RGB2HSV'):

        if conv == 'RGB2HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)



    """
    ****************************************************************************
    Semaphore detection function
    ****************************************************************************
    """
    def find_semaphore( self,
                        img,
                        ystart,
                        ystop,
                        scale,
                        svc,
                        X_scaler,
                        orient,
                        pix_per_cell,
                        cell_per_block,
                        spatial_size,
                        hist_bins,
                        draw_img
                      ):


        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YCrCb')

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize( ctrans_tosearch,
                                          (np.int(imshape[1]/scale),
                                          np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1
        nfeat_per_block = orient*cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block)
        hog2 = features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block)
        hog3 = features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block)

        boxes = []
        bbList = []

        for xb in range(nxsteps):
            for yb in range(nysteps):

                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                # Extract HOG for this patch
                #---------------------------------------------------------------
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                #---------------------------------------------------------------
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                #---------------------------------------------------------------
                spatial_features = features.bin_spatial(subimg, size=spatial_size)
                hist_features    = features.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                #---------------------------------------------------------------
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                #---------------------------------------------------------------
                test_prediction = svc.predict(test_features)



                if test_prediction == 1:

                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)

                    cv2.rectangle(  draw_img,
                                    (xbox_left, ytop_draw+ystart),
                                    ( (xbox_left + win_draw), (ytop_draw + win_draw + ystart) ),
                                    (0, 255, 0),
                                    3
                                  )

                    box = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw+ystart))
                    bb = [xbox_left,ytop_draw+ystart,xbox_left+win_draw, ytop_draw+win_draw+ystart ]
                    boxes.append(box)
                    bbList.append(bb)

        return draw_img, boxes, bbList


    """
    ****************************************************************************
    Run Semaphore detector and Classifier
    ****************************************************************************
    """
    def classifyTL(self, image_data):

        # HoG Features Parameters
        orient         = 9
        pix_per_cell   = 8
        cell_per_block = 2 #

        # Histogram Features Parameters
        spatial_size, hist_bins = (64, 64), 128

        # Predictor Parameters
        svc = pickle.load( open("./train_smaphore_detector/model.pkl", "rb" ) )
        X_scaler = pickle.load( open("./train_smaphore_detector/scaler.pkl", "rb" ) )

        frame = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        heat = np.zeros_like(frame[:,:,0]).astype(np.float)
        draw_img = np.copy(frame)

        #### PARAMETER TO TUNE #################################################
        parameters = [ (400, 500, 1.0),
                       (400, 550, 1.5),
                       (400, 700, 2.2)]

        DETECTIONS = []
        for ystart, ystop, scale in parameters:
            out_img, box_list, bbList = self.find_semaphore(   C_frames,
                                                                ystart,
                                                                ystop,
                                                                scale,
                                                                svc,
                                                                X_scaler,
                                                                orient,
                                                                pix_per_cell,
                                                                cell_per_block,
                                                                spatial_size,
                                                                hist_bins,
                                                                draw_img
                                                             )
            # out_img
            DETECTIONS.append(box_list)


        return DETECTIONS
