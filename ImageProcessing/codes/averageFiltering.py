import cv2
import numpy as np 
from numpy import zeros
import matplotlib.pyplot as plt


class GetAverageFilterThenThreshold:
    def __init__(self,path,kernelSize,thresholdVal):
        self.path=path
        self.img=cv2.imread(self.path)
        self.grayscaledImg =cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.height,self.width = self.grayscaledImg.shape
        self.kernelSize = kernelSize
        self.thresholdVal=thresholdVal
    def getPaddingOfTheImage(self):
        borderType = cv2.BORDER_CONSTANT
        borderLen = self.kernelSize//2
        dst = cv2.copyMakeBorder(self.grayscaledImg, borderLen ,borderLen, borderLen, borderLen, borderType)
        return dst
    def average_filter(self):
        ImgWithPadding=self.getPaddingOfTheImage()
        filteredImg = np.zeros(ImgWithPadding.shape)
        borderLen=self.kernelSize//2
        for i in range(borderLen,self.height+borderLen+1):
            for j in range(borderLen,self.width+borderLen+1):
                kernelMat=ImgWithPadding[i-borderLen:i+borderLen+1,j-borderLen:j+borderLen+1]
                filteredImg[i,j] = np.average(kernelMat)#Get the average Val
        return filteredImg[borderLen:-borderLen,borderLen:-borderLen]        
    def threshold(self,mat):
        height,width = mat.shape
        for i in range(height):
            for j in range(width):
                if(mat[i][j]>self.thresholdVal):
                    mat[i][j] = 255
                else:
                    mat[i][j] = 0
        return mat
    def exec(self):
        return self.threshold(self.average_filter())               
    

characters = GetAverageFilterThenThreshold('images/soru3-ocr.tif',3,60)
dst=characters.exec()
cv2.imshow("dst",dst)
cv2.waitKey()