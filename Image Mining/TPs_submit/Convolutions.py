import numpy as np
import cv2

from matplotlib import pyplot as plt

#Read grayscale image and conversion to float64
img=np.float64(cv2.imread('Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Image dimension:",h,"rows x",w,"columns")

#Direct method
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    valx = -img[y-1,x-1] + img[y-1,x+1] - 2*img[y,x-1] + 2*img[y,x+1] - img[y+1,x-1] + img[y+1,x-1]
    valy = img[y+1,x-1]-img[y-1,x-1]+2*img[y+1,x]-2*img[y-1,x]+img[y+1,x+1]-img[y-1,x+1]
    img2[y,x] = np.sqrt(valx**2+valy**2)
img2 = 255*(img2-np.amin(img2))/(np.amax(img2)-np.amin(img2))
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Direct method:",time,"s")

plt.subplot(121)
plt.imshow(img2, cmap = 'gray')
plt.title('Direct method (normalized)')

#Method filter2D
t1 = cv2.getTickCount()
kernelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
imgx = cv2.filter2D(img,-1,kernelx)
kernely = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
imgy = cv2.filter2D(img,-1,kernely)
img3 = np.sqrt(np.square(imgx)+np.square(imgy))
img3 = 255*(img3-np.amin(img3))/(np.amax(img3)-np.amin(img3))
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method filter2D :",time,"s")

plt.subplot(122)
plt.imshow(img3,cmap = 'gray', vmin = 0.0,vmax = 255.0)
plt.title('filter2D (normalized)')

plt.show()
