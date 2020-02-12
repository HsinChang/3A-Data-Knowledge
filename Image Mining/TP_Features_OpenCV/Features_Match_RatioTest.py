import numpy as np
import cv2

from matplotlib import pyplot as plt

detector=1

#Reading the image pair
img1 = cv2.imread('Image_Pairs/torb_small1.png')
print("Dimension of image 1:",img1.shape[0],"rows x",img1.shape[1],"columns")
print("Type of image 1:",img1.dtype)
img2 = cv2.imread('Image_Pairs/torb_small2.png')
print("Dimension of image 2:",img2.shape[0],"lignes x",img2.shape[1],"columns")
print("Type of image 2:",img2.dtype)

#Beginning the calculus
t1 = cv2.getTickCount()
#Creation of objects "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 500,#By default : 500
                       scaleFactor = 1.2,#By default : 1.2
                       nlevels = 8)#By default : 8
  kp2 = cv2.ORB_create(nfeatures=500,
                        scaleFactor = 1.2,
                        nlevels = 8)
  print("Detector: ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#By default : false
    		        threshold = 0.001,#By default : 0.001
  		        nOctaves = 4,#By default : 4
		        nOctaveLayers = 4,#By default : 4
		        diffusivity = 2)#By default : 2
  kp2 = cv2.KAZE_create(upright = False,
	  	        threshold = 0.001,
		        nOctaves = 4,
		        nOctaveLayers = 4,
		        diffusivity = 2)
  print("Detector: KAZE")
#Conversion to gray scale
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Detection and description of keypoints
pts1, desc1 = kp1.detectAndCompute(gray1,None)
pts2, desc2 = kp2.detectAndCompute(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Detection of points and computation of descriptors:",time,"s")
# Beginning of matching
t1 = cv2.getTickCount()
if detector == 1:
  #Hamming distance for descriptor BRIEF (ORB)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
else:
  #L2 distance for descriptor M-SURF (KAZE)
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# 2-nearest-neighbours list extraction
matches = bf.knnMatch(desc1,desc2, k=2)
# Application of the ratio test
good = []
for m,n in matches:
  if m.distance < 0.7*n.distance:
    good.append([m])
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Matching Computation:",time,"s")

# Displaying the matches that respect the ratio test
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 0)
img3 = cv2.drawMatchesKnn(gray1,pts1,gray2,pts2,good,None,**draw_params)

Nb_ok = len(good)
plt.imshow(img3),plt.title('Ratio Test, %i matches OK'%Nb_ok)
plt.show()


