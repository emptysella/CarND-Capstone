import cv2
import numpy as np

class TLClassifier_SWRP(object):

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

    def classifyTL(self, image):

        # Colour converstion from BGR to HSV
        image_data = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        cv2.imwrite('image_data.jpg', image_data)
        # get the image center geometry
        stride          = self.blocksize/2
        nredcircles     = 0
        nyellowcircles  = 0
        ngreencircles   = 0

        # Threshold  colors for HSV block to exact red high , red low  and yellow block bit masks
        """ red color """
        redImageLowTh   = cv2.inRange(image_data, np.array([0, 100, 100]), np.array([10, 255, 255]))
        redImageHighTh  = cv2.inRange(image_data, np.array([150, 100, 100]), np.array([179, 255, 255]))
        
        """ yellow color """

        yellowImage       = cv2.inRange(image_data, np.array([30, 70, 50]), np.array([60, 255, 255]))

        cv2.imwrite('redImageLowTh.jpg'  ,redImageLowTh)
        cv2.imwrite('redImageHighTh.jpg'  ,redImageHighTh)
        cv2.imwrite('yellowImage.jpg'  ,yellowImage)
        nprocessblockrows = (image_data.shape[0]) - (image_data.shape[0]/4)

        """
        Loop through each 64x64 block in selected ROI
        """
        for row in range(0, int(nprocessblockrows), int(stride)):
            for col in range(int(image_data.shape[1]/4), int((image_data.shape[1]) - stride), int(stride)):

                # Extract block data for detection of colors
                YellowRoi   = yellowImage[row:row + self.blocksize, col:col + self.blocksize]
                RedHRoi     = redImageHighTh[row:row + self.blocksize, col:col + self.blocksize]
                RedLRoi     = redImageLowTh[row:row + self.blocksize, col:col + self.blocksize]

                # Combine high and low range red circle masks to single bit mask
                RedROImask = cv2.addWeighted(RedLRoi, 1.0, RedHRoi, 1.0, 0.0)

                # Apply gaussian blur filter on each color bit masks to remove
                # unwanted noise and Apply Hough circle detection on each
                # induvisual masks to get count of number of circles
                redcircles    = None
                yellowcircles = None

                # optimization to reduce processing time check any red pixel before applying hough circles
                Ysum = np.sum(YellowRoi)
                Rsum = np.sum(RedROImask)

                if Rsum > 0 :
                    RedROImask = cv2.GaussianBlur(RedROImask,(9, 9), 2, 2)
                    redcircles = cv2.HoughCircles(
                                                    RedROImask,
                                                    cv2.HOUGH_GRADIENT,
                                                    1,
                                                    20,
                                                    param1 = 100,
                                                    param2 = 25,
                                                    minRadius = 0,
                                                    maxRadius = self.blocksize
                                                   )

                if Ysum > 0 :
                    YellowRoi = cv2.GaussianBlur( YellowRoi, (9, 9), 2, 2)
                    yellowcircles = cv2.HoughCircles( YellowRoi,
                                                      cv2.HOUGH_GRADIENT,
                                                      1,
                                                      20,
                                                      param1 = 100,
                                                      param2 = 25,
                                                      minRadius = 0,
                                                      maxRadius = self.blocksize
                                                     )

                #Count number of circles detected in each bit masks
                if redcircles is not None:
                    nredcircles = nredcircles + len(redcircles)

                if yellowcircles is not None:
                    nyellowcircles = nyellowcircles + len(yellowcircles)


        # classify color of detected circles based on number of circles detected
        #print(nredcircles, nyellowcircles, ngreencircles)

        if (nredcircles > 0) and (nredcircles > nyellowcircles):
            self.light = 3
            return self.light # TrafficLight.RED

        elif (nyellowcircles > 0) and (nyellowcircles > nredcircles):
            self.light = 1
            return self.light # TrafficLight.YELLOW
        else:
            self.light = 0
            return self.light #TrafficLight.UNKNOWN
