import sys
sys.path.insert(0,'./training')
import features
import pickle
import cv2
import numpy as np




class TLClassifier_SRG():

    def __init__(self):
        self.blocksize = 64

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
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)





    """
    ****************************************************************************
    isRed
    ****************************************************************************
    """ 
    def isRed(self, im):
           
        hsv     = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)   
        
        red     =  cv2.inRange(hsv, np.array([0, 100, 120]), np.array([6, 190, 250]))
        yellow  = cv2.inRange(hsv, np.array([20, 55, 100]), np.array([35, 255, 255]))     
        green   = cv2.inRange(hsv, np.array([80, 20, 200]), np.array([100, 255, 255]))
        
        num_red = np.sum(red) + np.sum(yellow)
        num_green = np.sum(green)    

        
        if num_red > num_green:
            return True
        else:
            return False
    
    

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
                        svc_classifier,
                        X_scaler_classifier,
                        orient,
                        pix_per_cell,
                        cell_per_block,
                        spatial_size,
                        hist_bins,
                        draw_img
                      ):

        img_tosearch = img[ystart:ystop,:,:]
        img_raw = img[ystart:ystop,:,:]

        
        ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YCrCb')
        ctrans_tosearch_hsv = self.convert_color(img_tosearch, conv='RGB2HSV')
        

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
        window_width = spatial_size[0]
        window_high  = spatial_size[1]
        
        
        nblocks_per_window = (window_width // pix_per_cell)-1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        nysteps = 8
        
        # Compute individual channel HOG features for the entire image
        hog1 = features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block)
        hog2 = features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block)
        hog3 = features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block)

        boxes = []
        bbList = []

        stop = False
        accumulate_stop = []
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
                xbox_left = np.int(xleft)
                ytop_draw = np.int(ytop)                

                # Extract the image patch
                #---------------------------------------------------------------
                                
                subimg      = ctrans_tosearch[ytop:ytop + window_high, xleft:xleft + window_width]
                subimg_HSV  = ctrans_tosearch_hsv[ytop:ytop + window_high, xleft:xleft + window_width]
          
                # HOG real time TODO: change to make it fast
                hog1_ = features.get_hog_features(subimg[:,:,0], orient, pix_per_cell, cell_per_block).ravel()
                hog2_ = features.get_hog_features(subimg[:,:,1], orient, pix_per_cell, cell_per_block).ravel()
                hog3_ = features.get_hog_features(subimg[:,:,2], orient, pix_per_cell, cell_per_block).ravel()
                hog_features_ = np.hstack((hog1_, hog2_, hog3_))
                            
                # Get color features
                #---------------------------------------------------------------
                spatial_features = features.bin_spatial(subimg, size=spatial_size)
                hist_features    = features.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                #---------------------------------------------------------------
                detector_features = X_scaler.transform(np.hstack((spatial_features,hist_features, hog_features_)).reshape(1, -1))      
                detector_prediction = svc.predict(detector_features)
                                
                if detector_prediction == 1:
                    
                    #---------------------------------------------------------------
                    # featutes extractor for the classifier
                    hist_features_hsv = features.color_hist(subimg_HSV, nbins=hist_bins)
                    test_features_HSV = X_scaler_classifier.transform(np.hstack((  
                                                                                hist_features_hsv,
                                                                                subimg_HSV[:,:,0].ravel(),
                                                                                subimg_HSV[:,:,1].ravel()
                                                                                )).reshape(1, -1))  
 
                     
                    color_prediction = svc_classifier.predict(test_features_HSV)
                    
                    subimg_color =  img_raw[ytop:ytop+window_high, xleft:xleft+window_width]
                    isred = self.isRed((subimg_color*255).astype(np.uint8))
                                        
                    if (color_prediction == 1) and isred:                      
                        cv2.rectangle(  draw_img,
                                        (xbox_left, ytop_draw + ystart),
                                        ( (xbox_left + window_width), (ytop_draw + window_high + ystart) ),
                                        (0, 0, 255),
                                        1
                                      )
    
                        box = ((xbox_left, ytop_draw + ystart), (xbox_left + window_width, ytop_draw + window_high + ystart))
                        bb  = [xbox_left, ytop_draw + ystart, xbox_left + window_width, ytop_draw + window_high + ystart ] 
                        boxes.append(box)
                        bbList.append(bb)
                        
                        
                        accumulate_stop.append(True)
                        #print("RED")
                        
                        
                    if ( (color_prediction == 0) and (isred==False ) ):
                        
                        
                        accumulate_stop.append(False)
                        #print("GRENN")
                        
                        cv2.rectangle(  draw_img,
                                        (xbox_left, ytop_draw + ystart),
                                        ( (xbox_left + window_width), (ytop_draw + window_high + ystart) ),
                                        (0, 255, 0),
                                        1
                                      )
            
        #---------------------------------------------------------------
        # Logic to manage false possitives
        if len(accumulate_stop) > 0:
            num_stop = sum(accumulate_stop)
            len_stop = len(accumulate_stop) 
            if num_stop > (len_stop/2):
                stop = True
            else:
                stop = False
        else:
            stop = False
            
                  
        return draw_img, boxes, bbList, stop
        
        
    

    """
    ****************************************************************************
    Run Semaphore detector and Classifier
    ****************************************************************************
    """
    def classifyTL(self, image_data):
        
        imageB = image_data
        image_data = cv2.normalize(image_data, 
                                   imageB,
                                   alpha=0,
                                   beta=1,
                                   norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
        
        # HoG Features Parameters
        orient         = 9
        pix_per_cell   = 8
        cell_per_block = 2 #

        # Histogram Features Parameters
        spatial_size, hist_bins = (48, 96), 128

        # Predictor Parameters
        svc = pickle.load( open("./training/models/model_detector.pkl", "rb" ) )
        X_scaler = pickle.load( open("./training/models/scaler_detector.pkl", "rb" ) )
        
        svc_classifier = pickle.load( open("./training/models/model_classifier.pkl", "rb" ) )
        X_scaler_classifier = pickle.load( open("./training/models/scaler_classifier.pkl", "rb" ) ) 

        frame = image_data

        draw_img = np.copy(frame)

        #### PARAMETER TO TUNE #################################################
        parameters = [ (100, 350, 1.0)]
        
        
        # Run the detector
        for ystart, ystop, scale in parameters:
            out_img, box_list, bbList, stop = self.find_semaphore(  frame,
                                                                    ystart,
                                                                    ystop,
                                                                    scale,
                                                                    svc,
                                                                    X_scaler,
                                                                    svc_classifier,
                                                                    X_scaler_classifier,
                                                                    orient,
                                                                    pix_per_cell,
                                                                    cell_per_block,
                                                                    spatial_size,
                                                                    hist_bins,
                                                                    draw_img
                                                                 )
            
            flag_image_out = True
            if flag_image_out:
                cv2.imwrite('output.jpg'  ,draw_img*255)
  
       
        return stop
