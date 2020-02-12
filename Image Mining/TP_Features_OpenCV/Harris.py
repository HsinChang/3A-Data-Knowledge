from scipy.ndimage import filters
import numpy as np
import cv2

from matplotlib import pyplot as plt

#Reading grayscale image and conversion to float64
img=np.float64(cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension of image:",h,"rows x",w,"columns")
print("Type of image:",img.dtype)
def Harris(alh, dml):
#Beginning of calculus
    t1 = cv2.getTickCount()
    Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
    # Put here Harris interest function calculation
    alpha = alh
    scale1 = 1
    scale2 = 2*scale1
    #Blur the image for better detection
    #imgBlur = cv2.GaussianBlur(img,(3,3),0,0,cv2.BORDER_REPLICATE)
    #Compute the first Gaussian derivatives
    dir_gauss = cv2.getGaussianKernel(3, scale1)
    gauss = np.multiply(dir_gauss.T, dir_gauss)
    gauss_dev_x = np.multiply(-np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])/scale1**2,gauss)
    gauss_dev_y = np.multiply(-np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])/scale1**2,gauss)
    img_x = cv2.filter2D(img, -1, gauss_dev_x)
    img_y = cv2.filter2D(img, -1, gauss_dev_y)

    # img_x = cv2.Sobel(img, axis=0)
    # img_y = cv2.Sobel(img, axis=1)
    # #Compute the autocorrelation matirx and the result
    # I_xx = np.square(img_x)
    # I_xy = np.square(img_y)
    # I_yy = np.multiply(img_x, img_y)

    # gradient_imx, gradient_imy = np.zeros(img.shape), np.zeros(img.shape)
    # filters.gaussian_filter(img, (scale1, scale1), (0, 1), gradient_imx)
    # filters.gaussian_filter(img, (scale1, scale1), (1, 0), gradient_imy)
    #
    # I_xx = filters.gaussian_filter(gradient_imx*gradient_imx, scale2)
    # I_xy = filters.gaussian_filter(gradient_imx*gradient_imy, scale2)
    # I_yy = filters.gaussian_filter(gradient_imy*gradient_imy, scale2)
    I_xx = cv2.GaussianBlur(np.square(img_x),(3,3),sigmaX=scale2,sigmaY=scale2)
    I_yy = cv2.GaussianBlur(np.square(img_y),(3,3),sigmaX=scale2,sigmaY=scale2)
    I_xy = cv2.GaussianBlur(np.multiply(img_x, img_y),(3,3),sigmaX=scale2,sigmaY=scale2)
    det_H = I_xx*I_yy - I_xy**2
    trace_H = I_xx + I_yy
    Theta = det_H - alpha*(trace_H**2)
    Theta_norm = cv2.normalize(Theta, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Computing local maxima and thresholding
    Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
    d_maxloc = dml
    seuil_relatif = 0.001
    se = np.ones((d_maxloc,d_maxloc),np.uint8)
    Theta_dil = cv2.dilate(Theta,se)
    #Suppression of non-local-maxima
    Theta_maxloc[Theta < Theta_dil] = 0.0
    #Values to small are also removed
    Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
    t2 = cv2.getTickCount()
    time = (t2 - t1)/ cv2.getTickFrequency()
    print("My computation of Harris points:",time,"s")
    #print("Number of cycles per pixel:",(t2 - t1)/(h*w),"cpp")

    # plt.subplot(131)
    # plt.imshow(img,cmap = 'gray')
    # plt.title('Original image')
    #
    # plt.subplot(132)
    # plt.imshow(Theta_norm,cmap = 'gray')
    # plt.title('Harris function')

    se_croix = np.uint8([[1, 0, 0, 0, 1],
                         [0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
    Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
    #Re-read image for colour display
    Img_pts=cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
    (h,w,c) = Img_pts.shape
    print("Dimension of image:",h,"rows x",w,"columns x",c,"channels")
    print("Type of image:",Img_pts.dtype)
    Img_pts[Theta_ml_dil > 0] = [255,0,0]
    plt.subplot(131)
    plt.imshow(Theta_norm, cmap='gray')
    plt.title('Theta before dilation')

    plt.subplot(132)
    plt.imshow(Theta_dil, cmap='gray')
    plt.title('First dilation')

    plt.subplot(133)
    plt.imshow(Theta_ml_dil, cmap='gray')
    plt.title('Second dilation')
    return Img_pts
#Points are displayed as red crosses
# Img_pts[Theta_ml_dil > 0] = [255,0,0]
# plt.subplot(133)
# plt.imshow(Img_pts)
# plt.title('Harris points')



# plt.subplot(221)
# plt.imshow(Harris(0, 7))
# plt.title(r"$\alpha = 0, maxloc = 7$ ")
#
# plt.subplot(222)
# plt.imshow(Harris(0.01, 7))
# plt.title(r"$\alpha = 0.01, maxloc = 7$ ")
#
# plt.subplot(223)
# plt.imshow(Harris(0.15, 7))
# plt.title(r"$\alpha = 0.15, maxloc = 7$ ")
#
# plt.subplot(224)
# plt.imshow(Harris(0.3, 7))
# plt.title(r"$\alpha = 0.3, maxloc = 7$ ")
Harris(0.06, 3)
plt.show()
