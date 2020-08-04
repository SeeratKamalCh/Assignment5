import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import sift as sf


def getFeaturePoints(image1, image2):
    # Initiate SIFT detector
    #sift = cv2.xfeatures2d.SIFT_create()
    #kp1 = sift.detect(image1,None)
    #kp2 = sift.detect(image2,None)
    sift1 = sf.SIFT(image1)
    sift2 = sf.SIFT(image2)
    kp1 = sift1.detect()
    kp2 = sift2.detect()
    return (kp1, kp2)



def computeMatchesAndHomography(image1, image2, features_1, features_2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.compute(image1,features_1)
    kp2, des2 = sift.compute(image2,features_2)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])

    good = sorted(good, key = lambda x:x[0].distance)
    good = good[0:20]

    image1_points = []
    image2_points = []

    for match in good :
        key_point_1 = kp1[match[0].queryIdx].pt
        key_point_2 = kp2[match[0].trainIdx].pt

        image1_points.append(key_point_1)
        image2_points.append(key_point_2)
    image1_points =np.array(image1_points)
    image2_points =np.array(image2_points)
    M, mask = cv2.findHomography(image2_points, image1_points, cv2.RANSAC,5.0)
    return (good, M)


def stitch2Images(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
#Get feature points
    (kp1,kp2) = getFeaturePoints(gray1, gray2)
#Get matched points and homography
    (matches, M) = computeMatchesAndHomography(gray1, gray2, kp1, kp2)
    
#Now lets stitch the images together

#Figure out the size of the resultant image
    srcHeight, srcWidth = gray2.shape
    srcPoints = np.array([[0, 0, 1],
                          [0, srcHeight, 1],
                          [srcWidth, 0, 1],
                          [srcWidth, srcHeight, 1]])
    srcPoints=srcPoints.T

    dstPoints = np.matmul(M, srcPoints)
    dstPoints = dstPoints.T

    for point in dstPoints:
        point[0] = point[0]/point[2]
        point[1] = point[1]/point[2]
        point[2] = 1

    dstPoints = dstPoints.T
    
    minX = min(min(dstPoints[0]),min(srcPoints[0]))
    maxX = max(max(dstPoints[0]),max(srcPoints[0]))
    
    minX = math.ceil(minX)
#     minY = min(dstPoints[1])
#     maxY = max(dstPoints[1])

    img1Origin = min(srcPoints[0])
    
    dstWidth = maxX-minX
    dstWidth = math.ceil(dstWidth)

    transformX = np.asarray([[1,0,-minX],
                  [0,1,0],
                  [0,0,1]],dtype=np.float32)
    
    M = np.matmul(transformX, M)
    
    image1 = cv2.warpPerspective(image1, transformX, (dstWidth, srcHeight))
    
#Now warpperspective
    warped2 = cv2.warpPerspective(image2, M, (dstWidth, srcHeight))

#     for row in range(len(warped2)):
#         for col in range(len(warped2[0])):
#             if np.sum(warped2[row,col]) == 0:
#                 warped2[row,col] = image1[row,col]
#     warped2[0:srcHeight, -minX:-minX+srcWidth] = image1[0:srcHeight, -minX:-minX+srcWidth]

    rows,cols,channels = warped2.shape    
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(warped2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(image1,image1,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(warped2,warped2,mask = mask)
    
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    warped2[0:rows, 0:cols ] = dst

    return warped2

def createPanorama(dirName):
    files = os.listdir(dirName)

    prev_image = None
    
    for file in files:
        if '.jpg' in file:
            print(file)
            
            image = cv2.imread(dirName + file)

            if prev_image is not None:
                prev_image = stitch2Images(prev_image, image)
                
                plt.figure(num=None, figsize=(8, 6), dpi=50, facecolor='w', edgecolor='k')
                plt.imshow(prev_image)
                plt.show()
            else:
                prev_image = image
                
    return prev_image
            
image1 = cv2.imread('images/hill1.jpg')
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

image2 = cv2.imread('images/hill2.jpg')
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

(kp1,kp2) = getFeaturePoints(gray1, gray2)
print(len(kp1))

img1=cv2.drawKeypoints(gray1,kp1,np.array([]))
img2=cv2.drawKeypoints(gray2,kp2,np.array([]))

#plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
plt.imshow(img1)
plt.show()
#plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
plt.imshow(img2)
plt.show()

(matches, M) = computeMatchesAndHomography(gray1, gray2, kp1, kp2)

print(matches)

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,matches,flags=2,outImg=np.array([]))


plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
plt.imshow(img3)
plt.savefig('image')
plt.show()

image1 = cv2.imread('images/hill1.jpg')
image2 = cv2.imread('images/hill2.jpg')

warped = stitch2Images(image2, image1)
cv2.imwrite("panorama.jpg", warped)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

plt.figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
plt.imshow(warped)
plt.show()


# task 4

panorama = createPanorama("./")

plt.figure(num=None, figsize=(8, 6), dpi=250, facecolor='w', edgecolor='k')
plt.imshow(panorama)
plt.show()