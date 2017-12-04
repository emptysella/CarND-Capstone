import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

"""
imagePath  = '/Users/esevalero/Desktop/UDACITY_TERM3/team-robo4-master/data/simulator_images/sim_images_1/image101.jpg'
#imagePath  = '/Users/esevalero/Desktop/Udacity-CarND-Capstone/ros/src/tl_detector/training/Images/red_1.png'
#imagePath  = '/Users/esevalero/Desktop/Udacity-CarND-Capstone/ros/src/tl_detector/training/Images/red_2.png'
#imagePath  = '/Users/esevalero/Desktop/Udacity-CarND-Capstone/ros/src/tl_detector/training/Images/yellow_3.png'
#imagePath  = '/Users/esevalero/Desktop/Udacity-CarND-Capstone/ros/src/tl_detector/training/Images/yellow_4.png'
#imagePath  = '/Users/esevalero/Desktop/Udacity-CarND-Capstone/ros/src/tl_detector/training/Images/yellow_5.png'
#imagePath  = '/Users/esevalero/Desktop/Udacity-CarND-Capstone/ros/src/tl_detector/training/Images/red_0.png'
#imagePath  = '/Users/esevalero/Desktop/Udacity-CarND-Capstone/ros/src/tl_detector/training/Images/green_6.png'
#imagePath  = '/Users/esevalero/Desktop/Udacity-CarND-Capstone/ros/src/tl_detector/training/Images/green_7.png'
#imagePath  = '/Users/esevalero/Desktop/Udacity-CarND-Capstone/ros/src/tl_detector/training/Images/green_8.png'

#imageA = mpimg.imread(imagePath)
imageA = cv2.imread(imagePath)

high, width, _ = imageA.shape



itr = 683

crop_w = 70
crop_h = 170



for w in range(0,800,crop_w):
    for h in range(0,600,crop_h):
        c = imageA[h:(h+170), w:(w+70), : ]
        itr +=1
        pathOut = 'im_' + str(itr) + '.jpg'
        cv2.imwrite(pathOut,c)
        
print(itr)
"""

ImagesClassA  = glob.glob('/Users/esevalero/Desktop/tl_classifier_exceptsmall/real/clasA/*.png')

high_=[]
width_=[]
itr = 1000
for im in ImagesClassA:
    imageA = cv2.imread(im)
    high, width, _ = imageA.shape
    itr +=1
    pathOut = 'im_' + str(itr) + '.png'
    cv2.imwrite(pathOut,imageA)
    high_.append(high)
    width_.append(width)
    
    
    
    
    