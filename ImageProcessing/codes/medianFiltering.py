import cv2
import numpy as np 
from numpy import zeros
import matplotlib.pyplot as plt


class GetMedianFilter:
    def __init__(self,path,kernelSize):
        self.path=path
        self.img=cv2.imread(self.path)
        self.grayscaledImg =cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.height,self.width = self.grayscaledImg.shape
        self.kernelSize = kernelSize
    def getPaddingOfTheImage(self):
        borderType = cv2.BORDER_CONSTANT
        borderLen = self.kernelSize//2
        dst = cv2.copyMakeBorder(self.grayscaledImg, borderLen ,borderLen, borderLen, borderLen, borderType)
        return dst
    def median_filter(self):
        ImgWithPadding=self.getPaddingOfTheImage()
        height,weight = ImgWithPadding.shape
        filteredImg = np.zeros(ImgWithPadding.shape)
        borderLen=self.kernelSize//2
        for i in range(borderLen,height):
            for j in range(borderLen,weight):
                leftXindex = i-borderLen
                rightXindex = i+borderLen+1
                upYindex =  j-borderLen
                downYindex = j+borderLen+1
                kernelMat=ImgWithPadding[leftXindex:rightXindex,upYindex:downYindex]
                filteredImg[i,j] = np.median(kernelMat)#Get the median Val
        return filteredImg[borderLen:-borderLen,borderLen:-borderLen]        


fingerprint = GetMedianFilter('images/soru2-fingerprint.tif',7)

dst = fingerprint.median_filter()
cv2.imshow("dst",dst)
cv2.waitKey()